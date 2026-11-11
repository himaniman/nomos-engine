"""Examine a found genome at every size, including the sizes its search never saw.

    python -m tissue.exam out/tissue/v0/tissue-s0.pt [--no-learn] [--json out.json]

Per size: K90 (pairs held at 90% of the bits), K90 per interneuron, and the accuracy at a
quarter, a half, one and two pairs per interneuron. `--no-learn` freezes the plastic wires:
the control that says the answers came from what the life stored.
"""

from __future__ import annotations

import argparse
import json

import torch

from tissue.assoc import curve, k90, k_grid, run_life
from tissue.hand import SIZES
from tissue.train import load


def examine(path: str, learn: bool = True, sizes=SIZES) -> dict:
    """{m: {"ks", "acc", "k90"}} for a tissue; one row (m = 0) for a net with no size."""
    model, task, saved = load(path, learn)
    out = {}
    for m in (sizes if saved["arm"] == "tissue" else (0,)):
        brain = model.grow(m)
        ks = k_grid(m or 64, task)
        accs = curve(lambda c, a, g: run_life(brain, task, c, a, g), task, ks)
        out[m] = {"ks": ks, "acc": accs, "k90": k90(ks, accs)}
    return out


def at(row: dict, k: float) -> float:
    """Accuracy at the examined load nearest to k pairs."""
    i = min(range(len(row["ks"])), key=lambda j: abs(row["ks"][j] - k))
    return row["acc"][i]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--no-learn", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    for path in a.genomes:
        res = examine(path, learn=not a.no_learn)
        print(f"\n{path}{'  (plastic wires frozen)' if a.no_learn else ''}")
        if a.no_learn:
            print(f"{'M':>5} | accuracy at 4 pairs and more: lowest .. highest")
            for m, row in res.items():
                accs = [x for k, x in zip(row["ks"], row["acc"]) if k >= 4]
                print(f"{m:>5} | {min(accs):.3f} .. {max(accs):.3f}")
        else:
            print(f"{'M':>5} {'K90':>8} {'K90/M':>6} |  acc at M/4   M/2     M    2M")
            for m, row in res.items():
                ref = m or 16
                print(f"{m:>5} {row['k90']:8.1f} {row['k90'] / ref:6.2f} |       "
                      + "  ".join(f"{at(row, ref * x):.3f}" for x in (0.25, 0.5, 1, 2)))
        tag = ".frozen.json" if a.no_learn else ".exam.json"
        with open(path.rsplit(".", 1)[0] + tag, "w") as f:
            json.dump({str(m): row for m, row in res.items()}, f)


if __name__ == "__main__":
    main()
