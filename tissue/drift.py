"""Through a long closing: is it the loudness of a thought that goes, or its pattern?

    python -m tissue.drift runs/tissue/stream/line-s0.pt ... [--m 128] [--loops 1 4] [--fan 32]

A stream life on loops of 8, then the eyes close for 64 ticks while the world goes on round
its loop. Per lap (8 ticks, so every lap holds the same items):

  * loud     the interneurons' mean |activity| (with the eyes open it is about 1);
  * pattern  the cosine between the interneurons' pattern and the code this same brain
             gives the item the world is really at, when it sees it;
  * speak    the share of interneurons that speak for an item: (sum |a|)^2 / (M sum a^2).
             1 if all are equally loud, 0.64 for a bell-shaped spread, f if a share f of
             them speak and the rest are silent.

A walk that holds keeps both. One that whispers keeps neither for long: on the straight
part of a cell's curve nothing saturates, so nothing pulls a wrong pattern back.
"""

from __future__ import annotations

import argparse
import math

import torch

from tissue import stream
from tissue.train import load


def drift(path: str, m: int = 128, loops: int = 1, closing: int = 64, lives: int = 64,
          seed: int = 11, fan: int | None = None) -> dict:
    tissue, task, _ = load(path) if fan is None else load(path, fan=fan)
    brain = tissue.grow(m)
    life = stream.Life(loops=loops, length=8, laps=2, idle=closing, blinks=1, open=3, closed=1)
    gen = torch.Generator().manual_seed(seed)
    items = stream.world(task, life, lives, gen)
    b, d = items.shape[0], items.shape[-1]
    rows = torch.arange(b)
    ticks = stream.script(life, b, gen)
    gain = torch.exp(tissue.gain["0"])
    with torch.no_grad():
        state = brain.born(b, gen)
        wires_in = state[1][0]                  # sensory -> inter, this individual's

        def code(item):                         # what the interneurons say when they SEE it
            ff = torch.einsum("bmd,bd->bm", wires_in, item) * gain / math.sqrt(d)
            return tissue.cells["I"](torch.stack([ff, torch.zeros_like(ff), torch.ones_like(ff)], -1))[..., 0]

        silent = torch.zeros(b, d)
        before, before_seen = silent, False
        loud, pattern, speak = [], [], []
        for kind, seen, loop, place in ticks:
            item = items[rows, loop, place]
            state, _ = brain.tick(state, item if seen else silent,
                                  before if before_seen else silent, float(before_seen), float(seen))
            if kind == "idle":
                act = state[0]["I"]
                loud.append(float(act.abs().mean()))
                pattern.append(float(torch.cosine_similarity(act, code(item), dim=1).mean()))
                speak.append(float((act.abs().sum(1) ** 2 / (m * (act ** 2).sum(1) + 1e-12)).mean()))
            before, before_seen = item, seen

    def per_lap(xs):
        return [sum(xs[i:i + 8]) / 8 for i in range(0, closing, 8)]
    return {"loud": per_lap(loud), "pattern": per_lap(pattern), "speak": per_lap(speak)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--loops", type=int, nargs="*", default=[1, 4])
    p.add_argument("--fan", type=int, default=None,
                   help="wires into each interneuron from other interneurons (default: as found)")
    a = p.parse_args()
    torch.set_num_threads(4)
    for path in a.genomes:
        for loops in a.loops:
            r = drift(path, a.m, loops, fan=a.fan)
            print(f"{path}  {a.m} cells, {loops * 8} items, per lap of a 64-tick closing")
            print("   loud     " + " ".join(f"{x:.2f}" for x in r["loud"]))
            print("   pattern  " + " ".join(f"{x:+.2f}" for x in r["pattern"]))
            print("   speak    " + " ".join(f"{x:.2f}" for x in r["speak"]))


if __name__ == "__main__":
    main()
