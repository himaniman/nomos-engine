"""Find a genome by gradient: the error at the end of a life corrects the rules.

    python -m tissue.train --arm tissue --seed 0 --out out/tissue/v0/tissue-s0.pt
    python -m tissue.train --arm gru64  --seed 0 --out out/tissue/v0/gru64-s0.pt

A step is a batch of whole lives. The tissue is grown to a size drawn from TRAIN_SIZES and
given 1/8 to 2 pairs per interneuron, so no single size or load can be tuned for; the sizes
above TRAIN_SIZES are never seen here and belong to the exam.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dataclasses import replace

from tissue import chain
from tissue.assoc import Task, accuracy, lives, run_life
from tissue.kernel import RUNG0, RUNG1, Gru, Tissue

TRAIN_SIZES = (8, 16, 32, 64)
GRU_PAIRS = 64          # a GRU has no size: it is trained on 1..this many pairs
CHAIN_TASK = Task(d_in=12, d_out=12)    # an item is 12 bits, seen and said alike


def make(arm: str, **switches):
    """(model, task) for an arm name: "tissue" (rung 0), "chain" (rung 1) or "gru<hidden>".
    `switches` override the plan's off-switches (learn, off, taught_only)."""
    if arm == "tissue":
        return Tissue(replace(RUNG0, **switches)), Task()
    if arm == "chain":
        return Tissue(replace(RUNG1, **switches), CHAIN_TASK), CHAIN_TASK
    task = Task(dwell=1)
    return Gru(task, int(arm[3:])), task


def load(path: str, **switches):
    saved = torch.load(path, weights_only=True)
    model, task = make(saved["arm"], **switches)
    model.load_state_dict(saved["genome"])
    return model, task, saved


def train(arm: str, seed: int, steps: int, out: Path, batch: int = 32, lr: float = 3e-3,
          hops: int = 1, echo: bool = False) -> None:
    """`hops`, `echo` (chain arm): the longest chain trained on, and whether the echo of
    the first item (hop 0) is part of the error along with the hops."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model, task = make(arm)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 10)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".jsonl").open("w")
    n_genome = sum(p.numel() for p in model.parameters())
    print(f"{arm} seed {seed}: genome of {n_genome} numbers", flush=True)
    ema_loss = ema_acc = None
    t0 = time.time()
    for step in range(1, steps + 1):
        u = torch.rand(3, generator=gen)
        m = TRAIN_SIZES[int(u[0] * len(TRAIN_SIZES))]
        if arm == "chain":              # 1/8 .. 1 link per interneuron, chains of 1..hops links
            length = 1 + int(u[2] * hops)
            k = max(1, round(m * 2 ** (float(u[1]) * 3 - 3)))
            items = chain.worlds(task, max(1, round(k / length)), length, batch, gen)
            logits = chain.run_life(model.grow(m), task, items, gen)[:, :, 0 if echo else 1:]
            answers = items[:, :, 0 if echo else 1:]
        else:
            if arm == "tissue":         # 1/8 .. 2 pairs per interneuron
                k = max(1, round(m * 2 ** (float(u[1]) * 4 - 3)))
            else:
                k = max(1, round(GRU_PAIRS ** float(u[1])))
            cues, answers = lives(task, k, batch, gen)
            logits = run_life(model.grow(m), task, cues, answers, gen)
        loss = F.softplus(-answers * logits).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        acc, cost = accuracy(logits, answers), loss.item()
        if not math.isfinite(cost):
            raise RuntimeError(f"loss is not finite at step {step} (m={m}, k={k})")
        ema_loss = cost if ema_loss is None else 0.98 * ema_loss + 0.02 * cost
        ema_acc = acc if ema_acc is None else 0.98 * ema_acc + 0.02 * acc
        if step % 100 == 0 or step == steps:
            row = {"step": step, "loss": round(ema_loss, 4), "acc": round(ema_acc, 4),
                   "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
            torch.save({"arm": arm, "seed": seed, "step": step, "hops": hops, "echo": echo,
                        "genome": model.state_dict()}, out)
    log.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", default="tissue", help='"tissue", "chain" or "gru<hidden>"')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--hops", type=int, default=1, help="chain arm: longest chain trained on")
    p.add_argument("--echo", action="store_true", help="chain arm: the echo is in the error too")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    train(a.arm, a.seed, a.steps, a.out, hops=a.hops, echo=a.echo)


if __name__ == "__main__":
    main()
