"""Exam 3: growth within one life. A brain is given new cells in mid-life; does it pay?

    python -m tissue.grow GENOME.pt ... [--small 512] [--big 2048] [--first 12] [--second 60]
    python -m tissue.grow GENOME.pt ... --alone [--small 256] [--big 2048] [--loops 96]

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

`--alone` is the other half of the question: a brain that is told nothing, neither where a
thing lies nor when to grow. Loops come in batches. By hand (`Places`): the brain puts a
run of new things at the emptiest stretch of its ring, finds a known thing by its index,
and when every stretch is in use it gives itself more cells. Arms: GROWS (from `small`, by
itself, up to `big`), BORN BIG, SMALL. Score: accuracy at the first hop and the last over
all the loops at the end, the size the brain ended with, and what it cost: cells x ticks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from tissue.assoc import Task
from tissue.hand import Index
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


class Places(Index):
    """Where things lie, and where the next new thing shall lie, by hand, for a ring that
    may grow. A place is kept in cells (as a share of `most`, the largest ring there may
    ever be), so it stays where it is when the ring grows. A run of new things is put at
    the stretch of the ring least in use (counting the stretches on either side of it, then
    by itself), the furthest from the last place among equals. `crowded`: every stretch of the ring is in
    use at least `full` times, which is when a brain that decides for itself grows."""

    def __init__(self, batch: int, cells: int, d: int, ring: int, most: int, full: float = 2.0,
                 grain: int = 16):
        super().__init__(batch, cells, d, ring=most, grain=grain, runs=True)
        self.ring, self.most, self.grain, self.full = ring, most, grain, full
        self.use = torch.zeros(batch, most // grain)

    def worn(self) -> torch.Tensor:
        """How much each stretch of the ring as it is now is in use, its neighbours counted."""
        n = self.ring // self.grain
        use = self.use[:, :n]
        return sum(torch.roll(use, k, dims=1) for k in (-2, -1, 0, 1, 2))

    def crowded(self) -> torch.Tensor:
        return self.worn().min(dim=1).values >= self.full

    def tape(self, item: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("bom,bm->bo", self.w, self.code(item))
        sure = (out.abs() > 0.5).all(dim=1)
        move = ~sure & self.knew
        if bool(move.any()):
            n = self.ring // self.grain
            at = (self.head * self.most / self.grain).long().clamp(max=n - 1)
            away = (torch.arange(n).unsqueeze(0) - at.unsqueeze(1)).abs()
            away = torch.minimum(away, n - away)
            pick = (self.worn() * 100 * n + self.use[:, :n] * 10 * n - away).argmin(dim=1)
            # the least in use with its neighbours, then by itself, then the furthest
            new = (pick.float() + 0.5) * self.grain / self.most
            self.head = torch.where(move, new, self.head)
            self.use[move, pick[move]] += 1
        here = torch.where(sure, self.say(item), self.head)
        self.learn(item, here)
        self.knew = sure
        return here


def alone(tissue, task: Task, loops: int, batches: int, small: int, big: int, arm: str,
          lives: int = 8, step: int = 256, full: float = 2.0, index: int = 4, closed: int = 8,
          seed: int = 60_000) -> dict:
    """One arm of a life in which nothing is told. `loops` loops come in `batches` equal
    batches, each walked round twice; after a batch the brain that GROWS gives itself
    `step` more cells if its ring is crowded. Then the eyes close on every loop, twice.
    {"hop": accuracy by hop, "cells": the size it ended with, "cost": cells x ticks}."""
    wgen = torch.Generator().manual_seed(seed)
    bgen = torch.Generator().manual_seed(seed + 1)
    h = Halves(first=loops, second=0, closed=closed)
    items = world(task, Life(loops=loops, length=h.length), lives, wgen)
    n, d = h.length, items.shape[-1]
    rows, silent = torch.arange(lives), torch.zeros(lives, d)
    start = big if arm == "born big" else small
    brain = tissue.grow(start)
    places = Places(lives, index * big, d, ring=start, most=big, full=full)
    right, count, cost = torch.zeros(closed), torch.zeros(closed), 0
    with torch.no_grad():
        state = brain.born(lives, bgen)
        was, was_seen, was_closed, hop, felt = silent, False, False, 0, torch.zeros(lives)

        def tick(item, seen, closing):
            nonlocal state, was, was_seen, was_closed, hop, felt, cost
            if seen:
                felt = places.tape(item) * big / brain.n["I"]
            state, out = brain.tick(state, item if seen else silent, was if was_seen else silent,
                                    float(was_seen), float(seen), where=felt)
            cost += brain.n["I"]
            if was_closed:                          # what is said now is about the tick before
                right[hop] += float(((out > 0) == (was > 0)).float().mean())
                count[hop] += 1
                hop += 1
            else:
                hop = 0
            was, was_seen, was_closed = item, seen, closing

        per = loops // batches
        for b in range(batches):
            for _ in range(h.laps):
                order = b * per + torch.rand(lives, per, generator=wgen).argsort(dim=1)
                for i in range(per):
                    place = torch.randint(n, (lives,), generator=wgen)
                    for k in range(n + 1):
                        tick(items[rows, order[:, i], (place + k) % n], True, False)
            if arm == "grows" and brain.n["I"] < big and bool(places.crowded().all()):
                brain, state = brain.more(state, min(step, big - brain.n["I"]), bgen)
                places.ring = brain.n["I"]
        for _ in range(2 * loops):
            loop = torch.randint(loops, (lives,), generator=wgen)
            place = torch.randint(n, (lives,), generator=wgen)
            for k in range(h.open + closed):
                tick(items[rows, loop, (place + k) % n], k < h.open, k >= h.open)
        tick(silent, False, False)
    return {"hop": (right / count.clamp(min=1)).tolist(), "cells": brain.n["I"], "cost": cost}


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
    p.add_argument("--alone", action="store_true",
                   help="nothing is told: the brain chooses its places and when to grow")
    p.add_argument("--loops", type=int, nargs="*", default=[24, 96],
                   help="--alone: loops in the world, a poor one and a rich one")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--step", type=int, default=256, help="--alone: cells added at a time")
    p.add_argument("--full", type=float, default=2.0,
                   help="--alone: the ring is crowded when every stretch is in use this often")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    switches = {k: v for k, v in (("fan", a.fan), ("field", a.field), ("lean", a.lean)) if v is not None}
    if a.alone:
        for path in a.genomes:
            tissue, task, _ = load(path, **switches)
            res = {}
            print(f"\n{path}  told nothing; from {a.small} cells, at most {a.big}; accuracy at"
                  f" hop 1 and hop {a.closed}, cells at the end, cost (million cell-ticks)")
            for loops in a.loops:
                for arm in ("grows", "born big", "small"):
                    r = alone(tissue, task, loops, a.batches, a.small, a.big, arm, a.lives,
                              a.step, a.full, closed=a.closed, seed=60_000 + 1_000_000 * a.worlds)
                    res[f"{loops} {arm}"] = r
                    print(f"  {loops * 8:>5} items  {arm:>8} | {r['hop'][0]:.3f}  {r['hop'][-1]:.3f} |"
                          f" {r['cells']:>5} | {r['cost'] / 1e6:6.1f}", flush=True)
            tag = f".alone{a.small}to{a.big}" + "".join(f".{k}{v:g}" for k, v in switches.items()) + (
                f".worlds{a.worlds}" if a.worlds else "")
            with open(path.rsplit(".", 1)[0] + tag + ".json", "w") as out:
                json.dump(res, out)
        return
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
