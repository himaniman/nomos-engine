"""Exam 3: growth within one life. A brain is given new cells in mid-life; does it pay?

    python -m tissue.grow GENOME.pt ... [--small 512] [--big 2048] [--first 12] [--second 60]

One unbroken stream life, as in exam 2, in a world whose loops lie at places along the ring:

  first    the eyes are open; the first loops are walked round, lap after lap;
  growth   the brain is given new cells, or not (the arms below);
  second   the second loops are walked round, lap after lap;
  test     the world jumps to a loop of either half, the eyes see `open` items and close
           for `closed` ticks while the world goes on. Twice per loop.

Arms, living the same world tick for tick:

  grown      born with `small` interneurons, given `big - small` more between the halves;
  born big   `big` interneurons from the first tick: what growth is read against;
  small      `small` interneurons to the end: what the growth was for.

A place is counted in cells, so it does not move when the ring grows. The first loops lie
on the first `small` cells, clear of the seam where the ring will open; the second loops
lie anywhere on the ring the brain has by then. New cells are born after the last one,
with innate wires of their own and nothing learnt (`Grown.more`).

Score: bit accuracy with the eyes closed, at the first hop and the last, for the first
loops and the second apart.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from tissue.assoc import Task
from tissue.stream import Life, world
from tissue.train import load

ARMS = ("grown", "born big", "small")


@dataclass(frozen=True)
class Halves:
    """The shape of a life in two halves."""
    first: int          # loops learnt before the growth
    second: int         # loops learnt after it
    length: int = 8
    laps: int = 2
    open: int = 3
    closed: int = 8

    @property
    def loops(self) -> int:
        return self.first + self.second


def script(h: Halves, batch: int, gen: torch.Generator) -> tuple[list, list]:
    """The world's side: (ticks before the growth, ticks after), each tick
    (kind, seen, loop (B,), place (B,)) with kind in "study" | "echo" | "closed"."""
    n = h.length
    loop = torch.zeros(batch, dtype=torch.long)
    place = torch.zeros(batch, dtype=torch.long)

    def study(lo: int, hi: int) -> list:
        nonlocal loop, place
        ticks = []
        for _ in range(h.laps):
            order = lo + torch.rand(batch, hi - lo, generator=gen).argsort(dim=1)
            for i in range(hi - lo):
                loop, place = order[:, i], torch.randint(n, (batch,), generator=gen)
                ticks.append(("study", True, loop, place))
                for _ in range(n):
                    place = (place + 1) % n
                    ticks.append(("study", True, loop, place))
        return ticks

    before, after = study(0, h.first), study(h.first, h.loops)
    for _ in range(2 * h.loops):
        loop = torch.randint(h.loops, (batch,), generator=gen)
        place = torch.randint(n, (batch,), generator=gen)
        after.append(("echo", True, loop, place))
        for k in range(h.open + h.closed - 1):
            place = (place + 1) % n
            after.append(("echo" if k < h.open - 1 else "closed", k < h.open - 1, loop, place))
    return before, after


def run_life(tissue, task: Task, h: Halves, small: int, big: int, arm: str, lives: int,
             seed: int) -> dict:
    """One arm's life. {"first" | "second": accuracy by hop (list)} over the closings on
    loops of that half. The world is a pure function of `seed`, whoever lives it."""
    wgen = torch.Generator().manual_seed(seed)
    bgen = torch.Generator().manual_seed(seed + 1)
    items = world(task, Life(loops=h.loops, length=h.length), lives, wgen)
    before, after = script(h, lives, wgen)
    clear = 2 * tissue.plan.field
    at_first = clear + torch.rand(lives, h.first, generator=wgen) * (small - 2 * clear)
    at_second = torch.rand(lives, h.second, generator=wgen)        # a share of the ring it lands on
    ring = small if arm == "small" else big
    at = torch.cat([at_first, at_second * ring], dim=1)             # every loop's place, in cells
    brain = tissue.grow(big if arm == "born big" else small)
    rows, d = torch.arange(lives), items.shape[-1]
    silent = torch.zeros(lives, d)
    right = {"first": torch.zeros(h.closed), "second": torch.zeros(h.closed)}
    count = {"first": torch.zeros(h.closed), "second": torch.zeros(h.closed)}
    with torch.no_grad():
        state = brain.born(lives, bgen)
        was, was_seen, was_kind, was_loop, hop = silent, False, "study", None, 0
        for part in (before, after + [("end", False, None, None)]):
            if part is not before and arm == "grown":
                brain, state = brain.more(state, big - small, bgen)
            n = brain.n["I"]
            for kind, seen, loop, place in part:
                item = silent if loop is None else items[rows, loop, place]
                felt = None if loop is None else at[rows, loop] / n
                state, out = brain.tick(state, item if seen else silent, was if was_seen else silent,
                                        float(was_seen), float(seen), where=felt)
                if was_kind == "closed":            # what is said now is about the tick before
                    ok = ((out > 0) == (was > 0)).float().mean(dim=1)
                    for half, mask in (("first", was_loop < h.first), ("second", was_loop >= h.first)):
                        right[half][hop] += float(ok[mask].sum())
                        count[half][hop] += float(mask.sum())
                    hop += 1
                else:
                    hop = 0
                was, was_seen, was_kind, was_loop = item, seen, kind, loop
    return {half: (right[half] / count[half].clamp(min=1)).tolist() for half in right}


def examine(path: str, h: Halves, small: int, big: int, lives: int = 16, seed: int = 50_000,
            **switches) -> dict:
    tissue, task, _ = load(path, **switches)
    if not tissue.plan.field:
        raise ValueError(f"{path} has no sense of place")
    return {arm: run_life(tissue, task, h, small, big, arm, lives, seed) for arm in ARMS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--small", type=int, default=512)
    p.add_argument("--big", type=int, default=2048)
    p.add_argument("--first", type=int, default=12, help="loops learnt before the growth")
    p.add_argument("--second", type=int, default=60, help="loops learnt after it")
    p.add_argument("--closed", type=int, default=8)
    p.add_argument("--lives", type=int, default=16)
    p.add_argument("--fan", type=int, default=None)
    p.add_argument("--field", type=int, default=None)
    p.add_argument("--lean", type=float, default=None)
    p.add_argument("--worlds", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    switches = {k: v for k, v in (("fan", a.fan), ("field", a.field), ("lean", a.lean)) if v is not None}
    h = Halves(a.first, a.second, closed=a.closed)
    tag = f".grow{a.small}to{a.big}.{a.first}+{a.second}" + "".join(
        f".{k}{v:g}" for k, v in switches.items()) + (f".worlds{a.worlds}" if a.worlds else "")
    for path in a.genomes:
        res = examine(path, h, a.small, a.big, a.lives, 50_000 + 1_000_000 * a.worlds, **switches)
        print(f"\n{path}  {a.small} -> {a.big} cells; {a.first * h.length} items before the growth,"
              f" {a.second * h.length} after; accuracy at hop 1 and at hop {a.closed}")
        print(f"{'':>10} | the first loops | the second loops")
        for arm in ARMS:
            f, s = res[arm]["first"], res[arm]["second"]
            print(f"{arm:>10} |  {f[0]:.3f}  {f[-1]:.3f} |  {s[0]:.3f}  {s[-1]:.3f}")
        with open(path.rsplit(".", 1)[0] + tag + ".json", "w") as out:
            json.dump(res, out)


if __name__ == "__main__":
    main()
