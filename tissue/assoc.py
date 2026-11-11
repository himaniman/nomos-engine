"""Exam 0: associative memory. Random pairs, new in every life, shown once, then asked.

A life: K pairs (cue -> answer) of random bits. Study: each cue stays on the sensory cells
for `dwell` ticks, and on the last tick the teacher shows the answer to the motor cells.
Test: each cue again, in a fresh order, no teacher; the answer is read from the motor cells
on the last tick. Nothing about a pair can be innate, so whatever is recalled was stored
during the life.

The score is bit accuracy (chance 0.5); the headline is K90, how many pairs a brain holds
at 90% of the bits right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Task:
    d_in: int = 12      # sensory cells: a cue is this many bits
    d_out: int = 4      # motor cells: an answer is this many bits
    dwell: int = 2      # ticks a cue stays on; teacher and read-out come on the last one


def lives(task: Task, k: int, batch: int, gen: torch.Generator):
    """`batch` lives of `k` pairs: cues (B, K, d_in) distinct within a life, answers (B, K, d_out)."""
    n = 2 ** task.d_in
    if k > n:
        raise ValueError(f"{k} distinct cues do not fit in {task.d_in} bits")
    idx = torch.rand(batch, n, generator=gen).argsort(dim=1)[:, :k]
    cues = ((idx.unsqueeze(-1) >> torch.arange(task.d_in)) & 1).float() * 2 - 1
    answers = torch.randint(0, 2, (batch, k, task.d_out), generator=gen).float() * 2 - 1
    return cues, answers


def run_life(brain, task: Task, cues: torch.Tensor, answers: torch.Tensor,
             gen: torch.Generator) -> torch.Tensor:
    """Live one life per row of the batch; returns what was answered, (B, K, d_out) logits.

    `brain` has `born(batch, gen) -> state` and `tick(state, cue, teach, flag) -> (state, out)`.
    """
    b, k, _ = cues.shape
    state = brain.born(b, gen)
    silent = torch.zeros(b, task.d_out)
    for i in range(k):
        for t in range(task.dwell):
            last = t == task.dwell - 1
            state, _ = brain.tick(state, cues[:, i], answers[:, i] if last else silent, float(last))
    said: list = [None] * k
    for i in torch.randperm(k, generator=gen).tolist():
        for _ in range(task.dwell):
            state, out = brain.tick(state, cues[:, i], silent, 0.0)
        said[i] = out
    return torch.stack(said, dim=1)


def accuracy(logits: torch.Tensor, answers: torch.Tensor) -> float:
    return float(((logits > 0) == (answers > 0)).float().mean())


def k_grid(m: int, task: Task) -> list[int]:
    """Loads to examine a brain of M interneurons at: 1 pair up to 8 per interneuron."""
    top = min(2 ** task.d_in, 8 * m)
    return sorted({min(top, round(2 ** (j / 2))) for j in range(2 * int(math.log2(top)) + 1)})


def curve(answer: Callable, task: Task, ks: list[int], seed: int = 10_000) -> list[float]:
    """Bit accuracy at each K. `answer(cues, answers, gen) -> logits` is the whole brain."""
    out = []
    for k in ks:
        gen = torch.Generator().manual_seed(seed + k)
        cues, answers = lives(task, k, max(4, min(32, 2048 // k)), gen)
        with torch.no_grad():
            out.append(accuracy(answer(cues, answers, gen), answers))
    return out


def k90(ks: list[int], accs: list[float], bar: float = 0.9) -> float:
    """Pairs held at `bar` accuracy: the first crossing, interpolated on a log scale."""
    if accs[0] < bar:
        return 0.0
    for i in range(1, len(ks)):
        if accs[i] < bar:
            t = (accs[i - 1] - bar) / (accs[i - 1] - accs[i])
            return math.exp(math.log(ks[i - 1]) + t * (math.log(ks[i]) - math.log(ks[i - 1])))
    return float(ks[-1])
