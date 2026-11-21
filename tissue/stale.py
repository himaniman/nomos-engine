"""A look at doubt in a world of forks whose ways change (no exam).

    python -m tissue.stale GENOME.pt ... [--ways 4] [--sizes 64 256] [--wander 1.5]
                                         [--hazard 0.3] [--burst 0] [--fade 0.3] [--load 0.5]
                                         [--lives 32]

Exam 4's life with a dendrite per way, and then the body walks on by ways drawn at random
with its eyes open while, unknown to the brain, ways come to lead somewhere new
(`fork.Walk.wander`, `hazard`). At the test the brain is asked, one hop into the dark, about
ways that are as the body last found them (FRESH) and about ways that have changed since
(STALE): of those it cannot be right. A brain that doubts well speaks of an old way more
quietly than of a new one, and so of a stale way more quietly than of a fresh one.

Per size: the bits right and the loudness (the mean size of what is said) one hop into the
dark, for fresh ways by the ticks since the body last went them (in times the land's links)
and for stale ways; and the error a search would see there (softplus), which punishes a wrong
word said loudly. `--fade`: the life is lived with wires that fade (`Plan.fade`), by hand.
`--fade` and `--hazard` are counted a lap, as many ticks as the land has links: the share of
itself a wire loses in that time, and the chance that a way changes in it. `--load`: links
per interneuron.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from tissue import fork
from tissue.train import load

AGES = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9))    # ticks since last gone, in times the links


def look(path: str, sizes, ways: int = 4, wander: float = 1.5, hazard: float = 0.3, fade: float = 0.0,
         links: float = 0.5, lives: int = 32, seed: int = 72_000, closed: int = 4, burst: float = 0.0) -> dict:
    """{m: {"fresh": {age band: {"n", "right", "loud"}}, "stale": {"n", "right", "loud"},
    "all": {...}, "error", "error_fresh", "error_stale"}}, one hop into the dark."""
    out = {}
    for m in sizes:
        rooms = max(2, round(m * links / ways))
        model, task, _ = load(path, **({"fade": fade / (rooms * ways)} if fade else {}))
        walk = fork.Walk(rooms, ways, laps=2, blinks=max(8, rooms), open=3, closed=closed,
                         wander=wander, hazard=hazard / (rooms * ways), burst=burst)
        gen = torch.Generator().manual_seed(seed + m)
        items, nxt = fork.world(task.d_out, walk, lives, gen)
        notes: dict = {}
        with torch.no_grad():
            lived = fork.run_life(model.grow(m), items, nxt, walk, gen, body="dendrites", notes=notes)
        said, true = (x.view(lives, walk.blinks, walk.closed, -1)[:, :, 0] for x in lived["closed"])
        stale = torch.stack(notes["stale"], 1).view(lives, walk.blinks, walk.closed)[:, :, 0]
        age = torch.stack(notes["age"], 1).view(lives, walk.blinks, walk.closed)[:, :, 0]
        right = ((said > 0) == (true > 0)).float().mean(-1)
        loud = said.abs().mean(-1)
        error = F.softplus(-true * said).mean(-1)

        def band(mask):
            n = int(mask.sum())
            return {"n": n, "right": float(right[mask].mean()) if n else 0.0,
                    "loud": float(loud[mask].mean()) if n else 0.0}

        out[m] = {"fresh": {f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+": band(~stale & (age >= lo) & (age < hi))
                            for lo, hi in AGES},
                  "stale": band(stale), "all": band(torch.ones_like(stale)),
                  "error": float(error.mean()),
                  "error_fresh": float(error[~stale].mean()) if bool((~stale).any()) else 0.0,
                  "error_stale": float(error[stale].mean()) if bool(stale.any()) else 0.0}
    return out


def show(m: int, row: dict) -> str:
    fresh = " | ".join(f"{k}: {v['right']:.2f} {v['loud']:.1f} ({v['n']})" for k, v in row["fresh"].items())
    s = row["stale"]
    return (f"{m:>5} | fresh, by age: {fresh} | stale: {s['right']:.2f} {s['loud']:.1f} ({s['n']})"
            f" | error {row['error']:.3f} (fresh {row['error_fresh']:.3f}, stale {row['error_stale']:.3f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--ways", type=int, default=4)
    p.add_argument("--sizes", type=int, nargs="*", default=[64, 256])
    p.add_argument("--wander", type=float, default=1.5, help="the walk after the study, in times the land's links")
    p.add_argument("--hazard", type=float, default=0.3, help="the chance a lap that a way leads somewhere new")
    p.add_argument("--burst", type=float, default=0.0,
                   help="in every other world, at one tick of the walk, this share of the ways change at once")
    p.add_argument("--fade", type=float, default=0.0, help="by hand: the share of itself a wire loses a lap")
    p.add_argument("--load", type=float, default=0.5, help="links per interneuron")
    p.add_argument("--lives", type=int, default=32)
    p.add_argument("--seed", type=int, default=72_000)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    tag = (f".stale{a.ways}.w{a.wander:g}.h{a.hazard:g}" + (f".b{a.burst:g}" if a.burst else "")
           + (f".fade{a.fade:g}" if a.fade else "")
           + (f".load{a.load:g}" if a.load != 0.5 else ""))
    print("one hop into the dark: bits right, loudness (askings)")
    for path in a.genomes:
        res = look(path, a.sizes, a.ways, a.wander, a.hazard, a.fade, a.load, a.lives, a.seed, burst=a.burst)
        print(f"\n{path}  {tag.strip('.')}")
        for m, row in res.items():
            print(show(m, row), flush=True)
        with open(path.rsplit(".", 1)[0] + tag + ".json", "w") as f:
            json.dump({str(m): row for m, row in res.items()}, f)


if __name__ == "__main__":
    main()
