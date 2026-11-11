"""What a found brain makes of its sense of place: who speaks, and who learns, how far away.

    python -m tissue.where runs/tissue/place/place-s0.pt ... [--m 512] [--loops 8] [--fan 32]
                                                         [--field 24] [--lean 0]

A stream life in a world whose loops lie at places along the ring, then the eyes close for
64 ticks. Interneurons are sorted by their distance from the place the world is at, and per
band of distance the table gives:

  * open     their mean |activity| while the eyes are open (study);
  * closed   the same through the closing;
  * learns   the mean change of a wire into them per tick of study, relative to the band
             nearest the place.

A brain that keeps each loop in its own patch is loud and learning near the place and
silent and still far from it. One that ignores the sense is the same at every distance.
"""

from __future__ import annotations

import argparse

import torch

from tissue import stream
from tissue.train import load

BANDS = (0, 6, 12, 18, 24, 36, 64, 10 ** 9)     # cells away from the place


def where(path: str, m: int = 512, loops: int = 8, closing: int = 64, lives: int = 16,
          seed: int = 11, **switches) -> dict:
    tissue, task, _ = load(path, **switches)
    if not tissue.plan.field:
        raise ValueError(f"{path} has no sense of place")
    brain = tissue.grow(m)
    life = stream.Life(loops=loops, length=8, laps=2, idle=closing, blinks=1, open=3, closed=1)
    gen = torch.Generator().manual_seed(seed)
    items = stream.world(task, life, lives, gen)
    b, d = items.shape[0], items.shape[-1]
    rows = torch.arange(b)
    ticks = stream.script(life, b, gen)
    place_of = torch.rand(b, loops, generator=gen)
    inner = next(i for i, blk in enumerate(tissue.plan.blocks) if blk.pre == blk.post)
    sums = {k: torch.zeros(len(BANDS) - 1) for k in ("open", "closed", "learns")}
    counts = {k: torch.zeros(len(BANDS) - 1) for k in sums}

    def add(kind, value, band):                 # value, band: (B, M)
        for j in range(len(BANDS) - 1):
            mask = band == j
            sums[kind][j] += float(value[mask].sum())
            counts[kind][j] += float(mask.sum())

    with torch.no_grad():
        state = brain.born(b, gen)
        silent = torch.zeros(b, d)
        before, before_seen = silent, False
        for kind, seen, loop, place in ticks:
            item = items[rows, loop, place]
            felt = place_of[rows, loop]
            away = (torch.arange(m).unsqueeze(0) - felt.unsqueeze(1) * m).abs()
            away = torch.minimum(away, m - away)
            band = torch.bucketize(away, torch.tensor(BANDS[1:-1], dtype=away.dtype), right=True)
            old = state[1][inner]
            state, _ = brain.tick(state, item if seen else silent, before if before_seen else silent,
                                  float(before_seen), float(seen), where=felt)
            if kind == "study":
                add("open", state[0]["I"].abs(), band)
                add("learns", (state[1][inner] - old).abs().mean(-1), band)
            elif kind == "idle":
                add("closed", state[0]["I"].abs(), band)
            before, before_seen = item, seen
    out = {k: (sums[k] / counts[k].clamp(min=1)).tolist() for k in sums}
    near = max(out["learns"][0], 1e-12)
    out["learns"] = [x / near for x in out["learns"]]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--loops", type=int, default=8)
    p.add_argument("--fan", type=int, default=None,
                   help="wires into each interneuron from other interneurons (default: as found)")
    p.add_argument("--field", type=int, default=None, help="the width of the sense of place")
    p.add_argument("--lean", type=float, default=None,
                   help="how much of the hand gate is left (1: all of it, 0: none)")
    a = p.parse_args()
    torch.set_num_threads(4)
    switches = {k: v for k, v in (("fan", a.fan), ("field", a.field), ("lean", a.lean)) if v is not None}
    names = [f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+" for lo, hi in zip(BANDS, BANDS[1:])]
    for path in a.genomes:
        r = where(path, a.m, a.loops, **switches)
        print(f"{path}  {a.m} cells, {a.loops * 8} items; cells away from the place:")
        print("            " + " ".join(f"{n:>7}" for n in names))
        for key in ("open", "closed", "learns"):
            print(f"   {key:<8} " + " ".join(f"{x:7.2f}" for x in r[key]))


if __name__ == "__main__":
    main()
