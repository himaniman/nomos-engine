"""Exam 1: links and chains. What is recalled has to pass through the brain's own wires.

A life holds C chains of L links over distinct random items (12 bits each). STUDY: every
link (a -> b) is its own short episode, in a random order, so a chain is never seen whole:

    tick 1   a on the senses
    tick 2   b on the senses;  the teacher shows the motor cells `a`  (say what you just saw)
    tick 3   senses silent;    the teacher shows `b`
    rest     every cell falls silent, wires stay

The teacher only ever teaches the motor cells to ECHO. That b follows a is in no teacher's
signal: it can be kept only in wires between interneurons. TEST: the first item of a chain
for one tick, then silence; the motor cells are read on each later tick, and hop h (the
item h links down the chain) is due at tick h + 2. One tick, one hop. Hop 0 is the echo
of the first item itself.

Score: bit accuracy per hop (chance 0.5). K = C * L links in the life; K90 at a hop is the
number of links held with 90% of the bits right at that hop.
"""

from __future__ import annotations

from typing import Callable

import torch

from tissue.assoc import Task


def worlds(task: Task, chains: int, length: int, batch: int, gen: torch.Generator) -> torch.Tensor:
    """`batch` lives: items (B, C, length + 1, d), all distinct within a life."""
    n, need = 2 ** task.d_in, chains * (length + 1)
    if need > n:
        raise ValueError(f"{need} distinct items do not fit in {task.d_in} bits")
    idx = torch.rand(batch, n, generator=gen).argsort(dim=1)[:, :need]
    bits = ((idx.unsqueeze(-1) >> torch.arange(task.d_in)) & 1).float() * 2 - 1
    return bits.view(batch, chains, length + 1, task.d_in)


def run_life(brain, task: Task, items: torch.Tensor, gen: torch.Generator,
             hops: int | None = None) -> torch.Tensor:
    """Study every link, then walk each chain from its first item with the senses silent.
    Returns what the motor cells said at hops 0..`hops`: (B, C, hops + 1, d) logits. Hop 0
    is the echo of the item just shown: the one thing the teacher did teach."""
    b, c, n, d = items.shape
    hops = hops or n - 1
    state = brain.born(b, gen)
    silent = torch.zeros(b, d)
    links = [(ci, li) for ci in range(c) for li in range(n - 1)]
    for j in torch.randperm(len(links), generator=gen).tolist():
        ci, li = links[j]
        first, second = items[:, ci, li], items[:, ci, li + 1]
        state = brain.rest(state)
        state, _ = brain.tick(state, first, silent, 0.0, 1.0)
        state, _ = brain.tick(state, second, first, 1.0, 1.0)
        state, _ = brain.tick(state, silent, second, 1.0, 0.0)
    said = []
    for ci in range(c):
        state = brain.rest(state)
        state, _ = brain.tick(state, items[:, ci, 0], silent, 0.0, 1.0)
        walk = []
        for _ in range(hops + 1):
            state, out = brain.tick(state, silent, silent, 0.0, 0.0)
            walk.append(out)
        said.append(torch.stack(walk, dim=1))
    return torch.stack(said, dim=1)


def by_hop(logits: torch.Tensor, items: torch.Tensor) -> list[float]:
    """Bit accuracy at each hop the walk was read at, hop 0 (the echo) first."""
    right = (logits > 0) == (items[:, :, :logits.shape[2]] > 0)
    return right.float().mean(dim=(0, 1, 3)).tolist()


def k_grid(m: int, task: Task, length: int = 1) -> list[int]:
    """Links to examine a brain of M interneurons at: one chain up to 4 links per interneuron."""
    top = min(4 * m, (2 ** task.d_in // (length + 1)) * length)
    ks, k = [], float(length)
    while round(k / length) * length <= top:
        ks.append(round(k / length) * length)
        k *= 2 ** 0.5
    return sorted(set(ks))


def curve(answer: Callable, task: Task, ks: list[int], length: int = 1, seed: int = 20_000,
          stop_below: float | None = None):
    """Accuracy by hop at each K: (ks examined, [[acc at hop 0..length] per K]).
    `answer(items, gen) -> logits`. With `stop_below`, stops once hop 1 falls under it."""
    done, out = [], []
    for k in ks:
        gen = torch.Generator().manual_seed(seed + k)
        items = worlds(task, k // length, length, max(4, min(32, 2048 // k)), gen)
        with torch.no_grad():
            accs = by_hop(answer(items, gen), items)
        done.append(k)
        out.append(accs)
        if stop_below is not None and accs[1] < stop_below:
            break
    return done, out
