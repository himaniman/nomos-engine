"""Teach a cell its own gate: its perceptron is fitted to what the hand gate makes of it.

    python -m tissue.imitate GATED.pt ... [--field 24] [--fan 32] [--steps 3000]

A genome that lives with the gate by hand (`Plan.gate`) is lived through stream lives with
places. Every interneuron on every tick is one lesson: what the cell was given (its
dendrites, whether the eyes are open, how near the place is) and what came of it after the
gate (what it said and its learning signals; nothing, if it was asleep). The cell's
perceptron, the same shape and starting from itself, is fitted to those lessons.

The genome is saved beside the old one as `NAME.own.pt`, still carrying the hand gate, so
that a search can take the gate away step by step (`--anneal`) or an exam at once
(`--lean 0`). Nothing but the interneuron's perceptron changes.

Printed: how much of what an awake cell says the fitted perceptron gets wrong, and how loud
it leaves a cell that should be asleep (as a share of an awake cell's speech).
"""

from __future__ import annotations

import argparse
import copy

import torch

from tissue import stream
from tissue.train import load

LIVES = ((256, 6), (512, 12), (512, 24))        # cells, loops: the lives the lessons come from


def lessons(tissue, task, seed: int = 5, most: int = 400_000):
    """(given, said under the hand gate, awake) over interneuron-ticks of a few lives."""
    given, said = [], []

    def keep(module, inputs, output):
        given.append(inputs[0].detach())
        said.append(output.detach())

    hook = tissue.cells["I"].register_forward_hook(keep)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m, loops in LIVES:
            life = stream.Life(loops=loops, length=8, laps=2, idle=32, blinks=6, open=3, closed=8)
            stream.run_life(tissue.grow(m), task, stream.world(task, life, 8, gen), life, gen, places=True)
    hook.remove()
    x = torch.cat([a.reshape(-1, a.shape[-1]) for a in given])
    y = torch.cat([a.reshape(-1, a.shape[-1]) for a in said])
    awake = x[:, -1] >= tissue.plan.gate            # the nearness is a cell's last input
    y = y * awake.unsqueeze(1)
    keep_these = torch.rand(len(x), generator=gen) < min(1.0, most / len(x))
    return x[keep_these], y[keep_these], awake[keep_these]


def imitate(tissue, task, steps: int = 3000, seed: int = 5) -> dict:
    """Fit the interneuron's perceptron, in place, to its own speech under the hand gate.
    Returns what is left wrong, per output (the activity first)."""
    if not (tissue.plan.field and tissue.plan.gate):
        raise ValueError("only a genome that lives with a gate by hand can be taught it")
    x, y, awake = lessons(tissue, task, seed)
    cell = tissue.cells["I"]
    torch.manual_seed(seed)
    opt = torch.optim.Adam(cell.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    for _ in range(steps):
        idx = torch.randint(len(x), (8192,))
        loss = ((cell(x[idx]) - y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        err, loud = (cell(x) - y) ** 2, (y[awake] ** 2).mean(0)
        return {"awake, got wrong": (err[awake].mean(0) / loud).tolist(),
                "asleep, left loud": (err[~awake].mean(0) / loud).sqrt().tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--field", type=int, default=None, help="the width of the sense of place")
    p.add_argument("--fan", type=int, default=None)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    switches = {k: v for k, v in (("fan", a.fan), ("field", a.field)) if v is not None}
    for path in a.genomes:
        tissue, task, saved = load(path, **switches)
        left = imitate(tissue, task, a.steps)
        out = path.rsplit(".", 1)[0] + ".own.pt"
        saved = copy.copy(saved)
        saved.update({"genome": tissue.state_dict(), "lean": 1.0, "init": path})
        torch.save(saved, out)
        print(f"{out}: " + "; ".join(f"{k} " + " ".join(f"{v:.3f}" for v in vs) for k, vs in left.items()))


if __name__ == "__main__":
    main()
