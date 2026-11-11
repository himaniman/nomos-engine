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

from tissue import chain, stream
from tissue.assoc import Task, accuracy, lives, run_life
from tissue.kernel import RUNG0, RUNG1, Gru, Tissue

TRAIN_SIZES = (8, 16, 32, 64)
GRU_PAIRS = 64          # a GRU has no size: it is trained on 1..this many pairs
CHAIN_TASK = Task(d_in=12, d_out=12)    # an item is 12 bits, seen and said alike
ROOM = 1.0e8            # a sparse stream life: lives x ticks x cells x wires kept under this, by
#                         living fewer lives at once (under 2 GB of memory for the search)


def make(arm: str, **switches):
    """(model, task) for an arm name: "tissue" (rung 0), "chain" (rung 1) or "gru<hidden>".
    `switches` override the plan's off-switches (learn, off, taught_only)."""
    if arm == "tissue":
        return Tissue(replace(RUNG0, **switches)), Task()
    if arm in ("chain", "stream"):      # the same brain; only the life it is searched on differs
        return Tissue(replace(RUNG1, **switches), CHAIN_TASK), CHAIN_TASK
    task = Task(dwell=1)
    return Gru(task, int(arm[3:])), task


def adopt(model, genome: dict) -> None:
    """Take a genome found for a plan whose cells had fewer inputs: what it has is copied
    and a new input starts at zero, so the brain begins exactly as the old one lived."""
    mine = model.state_dict()
    for name, old in genome.items():
        new = mine[name]
        if new.shape != old.shape:
            if new.dim() != 2 or new.shape[0] != old.shape[0] or new.shape[1] < old.shape[1]:
                raise ValueError(f"{name}: {tuple(old.shape)} does not fit {tuple(new.shape)}")
            old = torch.cat([old, torch.zeros(new.shape[0], new.shape[1] - old.shape[1])], dim=1)
        mine[name] = old
    model.load_state_dict(mine)


def load(path: str, **switches):
    """A found genome as (model, task, what was saved). It is grown with the fan, reach and
    field it was searched with unless a switch says otherwise; other switches are the plan's
    off-switches."""
    saved = torch.load(path, weights_only=True)
    for dial in ("fan", "reach", "field", "spoken", "gate", "lean"):
        if dial in saved and dial not in switches:
            switches[dial] = saved[dial]
    model, task = make(saved["arm"], **switches)
    adopt(model, saved["genome"])           # a switch may give the cells a sense they never had
    return model, task, saved


def train(arm: str, seed: int, steps: int, out: Path, batch: int = 32, lr: float = 3e-3,
          hops: int = 1, echo: bool = False, idle: int = 0, init: str | None = None,
          vary: bool = False, fan: int = 0, sizes: tuple = TRAIN_SIZES,
          load: tuple = (0.25, 2.0), reach: int = 0, field: int = 0,
          spoken: bool = False, gate: float = 0.0, anneal: int = 0) -> None:
    """`hops`, `echo` (chain arm): the longest chain trained on, and whether the echo of
    the first item (hop 0) is part of the error along with the hops. Stream arm: `hops` is
    how long the eyes stay closed, `idle` a long closing between study and test; the echo
    is always part of the error there; `vary` draws each life's closing from `hops`, `hops`
    / 2, / 4, / 8. `init`: start from a genome found earlier (same brain plan) instead of
    from random numbers. `fan`: wires into each interneuron from other interneurons (0: all);
    a stream life then draws its fan from `fan` / 2, `fan`, 2 `fan` and its load between
    `load[0]` and `load[1]` items per wire (log-uniform), since wires are what a sparse brain's
    capacity follows. `sizes`: the brain sizes a life is drawn from. `reach`: those wires
    come from neighbours on a ring. `field`: every loop lies at a place and interneurons
    feel how near it is; the load is then counted per wire AND per patch of the ring (2 x
    `reach` cells), since a patch is what such a brain can keep a loop in. `spoken`: plastic
    dendrites are scaled by what was said to them, not by their wires. `gate`: by hand, an
    interneuron less near the place than this is silent and does not learn. `anneal`: that
    gate is taken away evenly over this many steps and the rest are lived without it (the
    genome is then saved as one that needs no gate)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model, task = (make(arm, fan=fan, reach=reach, field=field, spoken=spoken, gate=gate)
                   if fan else make(arm))
    if init:
        adopt(model, torch.load(init, weights_only=True)["genome"])
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
        m = sizes[int(u[0] * len(sizes))]
        if arm == "stream":             # 1/16 .. 1/2 item per interneuron, loops of 4 or 8
            length = (4, 8)[int(u[2] * 2)]
            if fan:                     # a sparse brain: its load is counted per wire
                wires = fan * 2 ** (int(torch.rand(1, generator=gen) * 3) - 1)
                lean = max(0.0, 1 - (step - 1) / anneal) if anneal else model.plan.lean
                model.plan = replace(model.plan, fan=int(wires), lean=lean)
                patches = max(1.0, m / (2 * reach)) if field and reach else 1.0
                loops = max(1, round(patches * wires * load[0] * (load[1] / load[0]) ** float(u[1])
                                     / length))
            else:
                loops = max(1, round(m * 2 ** (float(u[1]) * 3 - 4) / length))
            laps = 2 + int(torch.rand(1, generator=gen) * 2)        # 2 or 3 times round
            closed = hops >> int(torch.rand(1, generator=gen) * 4) if vary else hops
            life = stream.Life(loops, length, laps, idle=idle, blinks=6 if closed <= 8 else 3,
                               open=3, closed=closed)
            lives_now = batch if not fan else max(4, min(batch, int(ROOM / (life.ticks * m * wires))))
            lived = stream.run_life(model.grow(m), task, stream.world(task, life, lives_now, gen),
                                    life, gen, places=field > 0)
            logits = torch.cat([lived["echo"][0], lived["closed"][0]], dim=1)
            answers = torch.cat([lived["echo"][1], lived["closed"][1]], dim=1)
        elif arm == "chain":            # 1/8 .. 1 link per interneuron, chains of 1..hops links
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
                        "idle": idle, "vary": vary, "init": init or "", "fan": fan,
                        "reach": reach, "field": field, "spoken": spoken, "gate": gate,
                        "anneal": anneal, "lean": model.plan.lean if arm == "stream" else 1.0,
                        "sizes": list(sizes), "load": list(load), "genome": model.state_dict()}, out)
    log.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", default="tissue", help='"tissue", "chain", "stream" or "gru<hidden>"')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--hops", type=int, default=1,
                   help="chain: longest chain trained on; stream: ticks the eyes stay closed")
    p.add_argument("--echo", action="store_true", help="chain arm: the echo is in the error too")
    p.add_argument("--idle", type=int, default=0, help="stream arm: a long closing before the test")
    p.add_argument("--init", help="start from this genome (same brain plan) instead of random")
    p.add_argument("--vary", action="store_true",
                   help="stream arm: closings of hops, hops/2, hops/4 or hops/8 ticks")
    p.add_argument("--fan", type=int, default=0,
                   help="wires into each interneuron from other interneurons (0: from all)")
    p.add_argument("--sizes", type=int, nargs="*", default=list(TRAIN_SIZES),
                   help="brain sizes a life is drawn from")
    p.add_argument("--load", type=float, nargs=2, default=[0.25, 2.0],
                   help="with --fan: items per wire, lowest and highest (log-uniform)")
    p.add_argument("--reach", type=int, default=0,
                   help="with --fan: those wires come from cells within this many places on a ring")
    p.add_argument("--field", type=int, default=0,
                   help="with --fan: a sense of place this many cells wide; loops lie at places")
    p.add_argument("--spoken", action="store_true",
                   help="with --fan: plastic dendrites are scaled by what was said, not by their wires")
    p.add_argument("--gate", type=float, default=0.0,
                   help="with --field: by hand, an interneuron less near the place than this is silent")
    p.add_argument("--anneal", type=int, default=0,
                   help="with --gate: take the gate away evenly over this many steps")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    train(a.arm, a.seed, a.steps, a.out, hops=a.hops, echo=a.echo, idle=a.idle, init=a.init,
          vary=a.vary, fan=a.fan, sizes=tuple(a.sizes), load=tuple(a.load), reach=a.reach,
          field=a.field, spoken=a.spoken, gate=a.gate, anneal=a.anneal)


if __name__ == "__main__":
    main()
