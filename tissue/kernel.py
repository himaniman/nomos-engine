"""The tissue: cells of a few types, wires between them, one synchronous tick.

GENOME (the module's parameters, the same at every brain size):
  * one small perceptron per cell type: every cell of that type carries the same one;
  * one gain per block of innate wires.
INDIVIDUAL (made at birth, lives in the state):
  * the innate wires, random in each individual: a genome sets their statistics and can
    not address a single wire;
  * the plastic wires, which start at zero and hold what the life has taught.

A tick, for every cell at once, from a snapshot of the previous tick:
  1. each dendrite averages what arrives over one block of wires;
  2. the cell's perceptron reads its dendrites (a motor cell also the teacher) and says its
     new activity and, per plastic dendrite, a learning signal;
  3. a plastic wire changes by (what its source cell said) x (its target's learning signal).
So learning is local: a wire sees only the two cells it joins.

By hand, not found: the cell types, which blocks exist, the average on a dendrite (so a
cell's input does not depend on how many wires reach it), the product form of the wire rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from tissue.assoc import Task


@dataclass(frozen=True)
class Block:
    """All the wires from one cell type to another."""
    pre: str
    post: str
    plastic: bool       # False: innate, random in each individual, fixed for life


@dataclass(frozen=True)
class Plan:
    """Which mechanisms exist. Their numbers are the genome."""
    blocks: tuple[Block, ...] = (Block("S", "I", False), Block("I", "O", True))
    hidden: int = 8     # the perceptron inside a cell: fixed, the same at every brain size
    mode: bool = False  # interneurons hear one global signal: are the senses on
    # off-switches, for controls and ablations
    learn: bool = True              # False: no plastic wire ever changes
    off: tuple[int, ...] = ()       # blocks (by index) whose wires stay as they were born
    taught_only: bool = False       # motor wires change only on a tick the teacher is on


RUNG0 = Plan()
RUNG1 = Plan(blocks=(Block("S", "I", False), Block("I", "I", True), Block("I", "O", True)), mode=True)


class Cell(nn.Module):
    """The perceptron every cell of one type carries."""

    def __init__(self, n_in: int, n_out: int, hidden: int):
        super().__init__()
        self.a = nn.Linear(n_in, hidden)
        self.b = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.tanh(self.a(x)))


class Tissue(nn.Module):
    """A genome. `grow(m)` makes brains of any size from it.

    "S" cells are sensory organs (they carry the cue), "O" cells are motor (they hear the
    teacher and are read as the answer); every other type is an interneuron type.
    """

    def __init__(self, plan: Plan = Plan(), task: Task = Task()):
        super().__init__()
        self.plan, self.task = plan, task
        seen = [t for b in plan.blocks for t in (b.pre, b.post)]
        self.types = [t for i, t in enumerate(seen) if t != "S" and t not in seen[:i]]
        self.incoming = {t: [i for i, b in enumerate(plan.blocks) if b.post == t]
                         for t in self.types}
        self.cells = nn.ModuleDict({
            t: Cell(len(self.incoming[t]) + (2 if t == "O" else int(plan.mode)),
                    1 + sum(plan.blocks[i].plastic for i in self.incoming[t]), plan.hidden)
            for t in self.types})
        self.gain = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(()))
                                      for i, b in enumerate(plan.blocks) if not b.plastic})

    def grow(self, m) -> "Grown":
        """A brain with `m` interneurons (or {type: count} once there are several types)."""
        return Grown(self, m if isinstance(m, dict) else {"I": m})


class Grown:
    """One genome at one size: the brain the exams drive."""

    def __init__(self, tissue: Tissue, inter: dict):
        self.t = tissue
        self.n = {"S": tissue.task.d_in, "O": tissue.task.d_out, **inter}

    def born(self, batch: int, gen: torch.Generator):
        act = {t: torch.zeros(batch, self.n[t]) for t in self.t.types}
        wires = {}
        for i, b in enumerate(self.t.plan.blocks):
            shape = (batch, self.n[b.post], self.n[b.pre])
            wires[i] = torch.zeros(shape) if b.plastic else torch.randn(shape, generator=gen)
        return act, wires

    def rest(self, state):
        """An episode ends: every cell falls silent, every wire stays as it is."""
        act, wires = state
        return {t: torch.zeros_like(a) for t, a in act.items()}, wires

    def tick(self, state, cue: torch.Tensor, teach: torch.Tensor, flag: float, on: float = 1.0):
        """One tick. `teach`/`flag`: what the teacher shows the motor cells and whether it
        is there; `on`: whether the senses are on (heard by interneurons under `plan.mode`)."""
        act, wires = state
        plan = self.t.plan
        seen = {**act, "S": cue}                    # sensory cells carry the cue as it is
        new_act, new_wires = {}, dict(wires)
        for t in self.t.types:
            feats = []
            for i in self.t.incoming[t]:
                b = plan.blocks[i]
                reach = torch.einsum("bij,bj->bi", wires[i], seen[b.pre])
                if b.plastic:
                    feats.append(reach / self.n[b.pre])
                else:
                    feats.append(reach * torch.exp(self.t.gain[str(i)]) / math.sqrt(self.n[b.pre]))
            if t == "O":
                feats += [teach, torch.full_like(teach, flag)]
            elif plan.mode:
                feats.append(torch.full_like(feats[0], on))
            said = self.t.cells[t](torch.stack(feats, dim=-1))
            new_act[t] = said[..., 0]
            j = 1
            for i in self.t.incoming[t]:
                b = plan.blocks[i]
                if b.plastic:
                    if plan.learn and i not in plan.off:
                        signal = said[..., j] * flag if plan.taught_only and t == "O" else said[..., j]
                        new_wires[i] = wires[i] + signal.unsqueeze(-1) * seen[b.pre].unsqueeze(1)
                    j += 1
        return (new_act, new_wires), new_act["O"]


class Gru(nn.Module):
    """The ordinary net the tissue is read against: one recurrent layer, no size dial."""

    def __init__(self, task: Task, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(task.d_in + task.d_out + 1, hidden)
        self.out = nn.Linear(hidden, task.d_out)

    def grow(self, m=None) -> "Gru":
        return self

    def born(self, batch: int, gen: torch.Generator):
        return torch.zeros(batch, self.hidden)

    def tick(self, h, cue, teach, flag, on: float = 1.0):
        h = self.cell(torch.cat([cue, teach, torch.full_like(teach[:, :1], flag)], dim=1), h)
        return h, self.out(h)
