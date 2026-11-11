"""Examine a found genome at every size, including the sizes its search never saw.

    python -m tissue.exam GENOME.pt ... [--no-learn] [--taught-only] [--off 1] [--length 4]

Rung 0 genomes ("tissue", "gru<h>"): pairs. Per size: K90 (pairs held at 90% of the bits),
K90 per interneuron, accuracy at a quarter, a half, one and two pairs per interneuron.
Rung 1 genomes ("chain"): links, in chains of `--length`. Per size: K90 at each hop, and
the accuracy by hop at M/4 links, the echo of the first item (hop 0) first.

Switches run the same genome with a mechanism off: `--no-learn` freezes every plastic
wire, `--off N` freezes block N, `--taught-only` lets motor wires change only when the
teacher is on. The result is saved beside the genome as JSON.
"""

from __future__ import annotations

import argparse
import json

import torch

from tissue import chain
from tissue.assoc import curve, k90, k_grid, run_life
from tissue.hand import SIZES
from tissue.train import load


def examine(path: str, sizes=SIZES, length: int = 1, **switches) -> dict:
    """Rung 0: {m: {"ks", "acc", "k90"}} (m = 0 for a net with no size).
    Rung 1: {m: {"ks", "acc": [[by hop] per K], "k90": [by hop]}}."""
    model, task, saved = load(path, **switches)
    out = {}
    if saved["arm"] == "chain":
        for m in sizes:
            brain = model.grow(m)
            ks, accs = chain.curve(lambda items, g: chain.run_life(brain, task, items, g),
                                   task, chain.k_grid(m, task, length), length)
            out[m] = {"ks": ks, "acc": accs,
                      "k90": [k90(ks, [a[h] for a in accs]) for h in range(length + 1)]}
        return out
    for m in (sizes if saved["arm"] == "tissue" else (0,)):
        brain = model.grow(m)
        ks = k_grid(m or 64, task)
        accs = curve(lambda c, a, g: run_life(brain, task, c, a, g), task, ks)
        out[m] = {"ks": ks, "acc": accs, "k90": k90(ks, accs)}
    return out


def at(ks: list, accs: list, k: float):
    """What was measured at the examined load nearest to k."""
    return accs[min(range(len(ks)), key=lambda j: abs(ks[j] - k))]


def show(res: dict, length: int, frozen: bool) -> None:
    chained = isinstance(next(iter(res.values()))["k90"], list)
    if chained:
        hops = [h for h in sorted({1, 2, length // 2, length}) if 1 <= h <= length]
        print(f"{'M':>5} | K90 at hop " + " ".join(f"{h:>7}" for h in hops)
              + " | per interneuron, hop 1 | acc at M/4 links by hop, echo (hop 0) first")
        for m, row in res.items():
            print(f"{m:>5} |            " + " ".join(f"{row['k90'][h]:7.1f}" for h in hops)
                  + f" | {row['k90'][1] / m:6.2f}                 | "
                  + " ".join(f"{a:.3f}" for a in at(row["ks"], row["acc"], m / 4)))
    elif frozen:
        print(f"{'M':>5} | accuracy at 4 pairs and more: lowest .. highest")
        for m, row in res.items():
            accs = [x for k, x in zip(row["ks"], row["acc"]) if k >= 4]
            print(f"{m:>5} | {min(accs):.3f} .. {max(accs):.3f}")
    else:
        print(f"{'M':>5} {'K90':>8} {'K90/M':>6} |  acc at M/4   M/2     M    2M")
        for m, row in res.items():
            ref = m or 16
            print(f"{m:>5} {row['k90']:8.1f} {row['k90'] / ref:6.2f} |       "
                  + "  ".join(f"{at(row['ks'], row['acc'], ref * x):.3f}" for x in (0.25, 0.5, 1, 2)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--no-learn", action="store_true")
    p.add_argument("--taught-only", action="store_true")
    p.add_argument("--off", type=int, nargs="*", default=[])
    p.add_argument("--length", type=int, default=1)
    p.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    switches, tag = {}, ""
    if a.no_learn:
        switches["learn"], tag = False, tag + ".frozen"
    if a.taught_only:
        switches["taught_only"], tag = True, tag + ".taught-only"
    if a.off:
        switches["off"], tag = tuple(a.off), tag + ".off" + "-".join(map(str, a.off))
    if a.length > 1:
        tag += f".len{a.length}"
    for path in a.genomes:
        res = examine(path, a.sizes, a.length, **switches)
        print(f"\n{path}  {tag.strip('.') or 'as found'}")
        show(res, a.length, a.no_learn)
        with open(path.rsplit(".", 1)[0] + (tag or ".exam") + ".json", "w") as f:
            json.dump({str(m): row for m, row in res.items()}, f)


if __name__ == "__main__":
    main()
