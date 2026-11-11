"""Examine a found genome at every size, including the sizes its search never saw.

    python -m tissue.exam GENOME.pt ... [--length 4] [--stream] [--closed 4] [--idle 64]
                                        [--no-learn] [--taught-only] [--off 1]

Rung 0 genomes ("tissue", "gru<h>"): pairs. Per size: K90 (pairs held at 90% of the bits),
K90 per interneuron, accuracy at a quarter, a half, one and two pairs per interneuron.
Rung 1 genomes ("chain", "stream"): either rested episodes of links in chains of
`--length`, or (`--stream`, and always for a "stream" genome) the unbroken stream with the
eyes closing for `--closed` ticks and an optional `--idle` closing before the test. Per
size: K90 at each hop, and the accuracy by hop at M/4 links, the echo (hop 0) first.

Switches run the same genome with a mechanism off: `--no-learn` freezes every plastic
wire, `--off N` freezes block N, `--taught-only` lets motor wires change only when the
teacher is on. The result is saved beside the genome as JSON.
"""

from __future__ import annotations

import argparse
import json

import torch

from tissue import chain, stream
from tissue.assoc import curve, k90, k_grid, run_life
from tissue.hand import SIZES
from tissue.train import load


def examine(path: str, sizes=SIZES, length: int = 1, shape: stream.Life | None = None,
            **switches) -> dict:
    """Rung 0: {m: {"ks", "acc", "k90"}} (m = 0 for a net with no size).
    Rung 1: {m: {"ks", "acc": [[echo, hop 1, ...] per K], "k90": [by hop]}}; on the stream
    (`shape` given: a `stream.Life`, its loop count ignored) also "idle": [curve per K]."""
    model, task, saved = load(path, **switches)
    out = {}
    if shape is not None:
        for m in sizes:
            brain = model.grow(m)
            ks = stream.k_grid(m, task, shape.length)
            rows = stream.curve(lambda items, life, g: stream.run_life(brain, task, items, life, g),
                                task, ks, shape)
            accs = [[r["echo"]] + r["hop"] for r in rows]
            out[m] = {"ks": ks, "acc": accs, "idle": [r["idle"] for r in rows],
                      "k90": [k90(ks, [a[h] for a in accs]) for h in range(shape.closed + 1)]}
        return out
    if saved["arm"] in ("chain", "stream"):
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


def show(res: dict, frozen: bool) -> None:
    first = next(iter(res.values()))
    if isinstance(first["k90"], list):
        depth = len(first["k90"]) - 1
        hops = [h for h in sorted({1, 2, depth // 2, depth}) if 1 <= h <= depth]
        print(f"{'M':>5} | K90 at hop " + " ".join(f"{h:>7}" for h in hops)
              + " | per interneuron, hop 1 | acc at M/4 by hop, echo (hop 0) first")
        for m, row in res.items():
            print(f"{m:>5} |            " + " ".join(f"{row['k90'][h]:7.1f}" for h in hops)
                  + f" | {row['k90'][1] / m:6.2f}                 | "
                  + " ".join(f"{a:.3f}" for a in at(row["ks"], row["acc"], m / 4)))
        if first.get("idle") and first["idle"][0]:
            print("      through the idle closing at M/4, bit accuracy per stretch of 8 ticks:")
            for m, row in res.items():
                print(f"{m:>5} | " + " ".join(f"{a:.2f}" for a in at(row["ks"], row["idle"], m / 4)))
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
    p.add_argument("--length", type=int, default=None,
                   help="links in a chain (default 1), or items in a loop of the stream (default 8)")
    p.add_argument("--stream", action="store_true", help="examine a rung 1 genome on the stream")
    p.add_argument("--chains", action="store_true",
                   help="examine a stream genome on rested episodes of chains instead")
    p.add_argument("--closed", type=int, default=4, help="stream: ticks the eyes stay closed")
    p.add_argument("--idle", type=int, default=0, help="stream: a long closing before the test")
    p.add_argument("--laps", type=int, default=2, help="stream: times each loop is walked round")
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
    for path in a.genomes:
        streamed = not a.chains and (a.stream or torch.load(path, weights_only=True)["arm"] == "stream")
        shape, length, mark = None, a.length or 1, tag
        if streamed:
            shape = stream.Life(1, a.length or 8, a.laps, a.idle, 8, 3, a.closed)
            mark += f".stream{shape.length}x{a.closed}" + (f".idle{a.idle}" if a.idle else "") \
                    + (f".laps{a.laps}" if a.laps != 2 else "")
        elif length > 1:
            mark += f".len{length}"
        res = examine(path, a.sizes, length, shape, **switches)
        print(f"\n{path}  {mark.strip('.') or 'as found'}")
        show(res, a.no_learn)
        with open(path.rsplit(".", 1)[0] + (mark or ".exam") + ".json", "w") as f:
            json.dump({str(m): row for m, row in res.items()}, f)


if __name__ == "__main__":
    main()
