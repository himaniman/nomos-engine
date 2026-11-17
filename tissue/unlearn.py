"""A look, no exam: what a brain says of a way that changed, by the walks it took to say so.

    python -m tissue.unlearn GENOME.pt ... [--side 8] [--shift 0.5] [--sizes 256] [--lives 8]

One life in a land that changes at its half (exam 6 with `--shift`, `tissue/world.py`), the
exam's lands and the exam's thought with a road kept. Every `--every` ticks the examiner
carries a copy into every room and asks what lies one hop by each way (`Mind.atlas`: the brain
is not touched). The ways are sorted by what the change did to them:

    closed   a way that became a wall        opened   a wall that became a way
    way      a way before and after          wall     a wall before and after

and by how often the body has gone them: before the change since its birth (how many walks
WRITE a way), after it since the change (how many walks UNDO one). Per kind and count: the
share of ways said `right` (every bit of the room that lies there now), the share said as
`before` (the room that lay there before the change), and how `loud` (the mean size of a bit).
After the change a way not gone since it (count 0) shows what is still held of the old land.
`--also explore=0.4` and the like pass a switch of the thought on.
"""

from __future__ import annotations

import argparse
import json

import torch

from tissue import world
from tissue.train import load

BINS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 9))
KINDS = ("closed", "opened", "way", "wall")


def label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else (f"{lo}+" if hi > 10 ** 6 else f"{lo}-{hi}")


def kinds(nxt: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """(rooms, 4): the index into KINDS of every way, by what the change did to it."""
    here = torch.arange(len(nxt)).unsqueeze(1)
    was, now = nxt != here, end != here         # a way (not a wall) before, after
    return torch.where(was & ~now, 0, torch.where(~was & now, 1, torch.where(was & now, 2, 3)))


def look(path: str, shape: world.Land, m: int, lives: int = 8, every: int = 40, seed: int = 90_000,
         **think) -> dict:
    """{"before" | "after": {kind: {count: {"n", "right", "before", "loud"}}}, "ever": {kind: the
    share of the ways of that kind gone at least once after the change}, "meals": a life}."""
    model, task, _ = load(path, still=True)
    half = shape.life // 2
    cells: dict = {}
    ever = {k: [0, 0] for k in KINDS}
    meals = 0
    for i in range(lives):
        gen = torch.Generator().manual_seed(seed + 1000 * shape.side + i)       # the exam's lands and dice
        looks, nxt, trees, start = world.land(task.d_out, shape, gen)
        end = world.moved(nxt, shape, torch.Generator().manual_seed(seed + 1000 * shape.side + i + 7919))
        kind = kinds(nxt, end)
        walked = {"before": torch.zeros(len(looks), 4), "after": torch.zeros(len(looks), 4)}
        with torch.no_grad():
            mind = world.Mind(model.grow(m), gen, **think)

            def watch(t, room, fruit, way, planned):
                phase = "before" if t < half else "after"
                if t and t % every == 0:
                    says = mind.atlas(looks)
                    right = ((says > 0) == (looks[end if t >= half else nxt] > 0)).all(-1)
                    old = ((says > 0) == (looks[nxt] > 0)).all(-1)
                    loud = says.abs().mean(-1)
                    for k, name in enumerate(KINDS):
                        for lo, hi in BINS:
                            mask = (kind == k) & (walked[phase] >= lo) & (walked[phase] <= hi)
                            n = int(mask.sum())
                            if n:
                                c = cells.setdefault((phase, name, label(lo, hi)), [0, 0.0, 0.0, 0.0])
                                c[0] += n
                                c[1] += float(right[mask].sum())
                                c[2] += float(old[mask].sum())
                                c[3] += float(loud[mask].sum())
                walked[phase][room, way] += 1

            life = world.live(mind, looks, nxt, trees, start, shape.life, gen, fruits=seed + i, watch=watch,
                              later=(half, end))
        meals += len(life["meals"])
        for k, name in enumerate(KINDS):
            ever[name][0] += int((kind == k).sum())
            ever[name][1] += int(((kind == k) & (walked["after"] > 0)).sum())
    out: dict = {"before": {}, "after": {}, "meals": meals / lives,
                 "ever": {name: (gone / n if n else 0.0) for name, (n, gone) in ever.items()}}
    for (phase, name, count), (n, right, old, loud) in cells.items():
        out[phase].setdefault(name, {})[count] = {"n": n, "right": right / n, "before": old / n, "loud": loud / n}
    return out


def show(name: str, res: dict) -> str:
    lines = [f"{name}: {res['meals']:.1f} meals a life; ways of each kind gone after the change: "
             + ", ".join(f"{k} {v:.2f}" for k, v in res["ever"].items())]
    for phase in ("before", "after"):
        for kind in KINDS:
            row = res[phase].get(kind)
            if not row or (phase == "before" and kind in ("closed", "opened")):
                continue
            lines.append(f"  {phase:>6} {kind:>6} | " + " | ".join(
                f"{count}: {c['right']:.2f} {c['before']:.2f} {c['loud']:.1f} ({c['n']})"
                for count, c in ((label(lo, hi), row.get(label(lo, hi))) for lo, hi in BINS) if c))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--side", type=int, default=8)
    p.add_argument("--shift", type=float, default=0.5)
    p.add_argument("--sizes", type=int, nargs="*", default=[256])
    p.add_argument("--lives", type=int, default=8)
    p.add_argument("--every", type=int, default=40, help="the ticks between two askings")
    p.add_argument("--budget", type=int, default=4096)
    p.add_argument("--also", nargs="*", default=[], help="switches of the thought, as explore=0.4 curious=0.7")
    p.add_argument("--seed", type=int, default=90_000)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    shape = world.Land(a.side, shift=a.shift)
    think = {"budget": a.budget, "keep": True, **{k: float(v) for k, v in (s.split("=") for s in a.also)}}
    tag = (f".unlearn{a.side}.shift{a.shift:g}" + "".join(f".{s.replace('=', '')}" for s in a.also)
           + ("" if a.sizes == [256] else ".m" + "-".join(map(str, a.sizes)))
           + ("" if a.seed == 90_000 else f".seed{a.seed}"))
    print(f"{a.side}x{a.side} rooms, a share {a.shift:g} of the walls moves at tick {shape.life // 2} of {shape.life},"
          f" {a.lives} lives; per count of walks: right, as before, loud (ways asked)")
    for path in a.genomes:
        res = {}
        for m in a.sizes:
            res[m] = look(path, shape, m, a.lives, a.every, a.seed, **think)
            print(show(f"{path.split('/')[-1][:-3]} {m}", res[m]), flush=True)
        with open(path.rsplit(".", 1)[0] + tag + ".json", "w") as f:
            json.dump({str(m): row for m, row in res.items()}, f)


if __name__ == "__main__":
    main()
