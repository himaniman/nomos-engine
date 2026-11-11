"""Left alone: what a brain does when nothing is asked of it.

    python -m tissue.alone runs/tissue/rung1a-echo/echo-s0.pt ... [--m 64] [--ticks 60]

A life of 4 chains of 4 links is studied, then ONE item is shown and the senses stay silent
for many ticks. Nothing here was searched for: the search only ever read the first hops.
After the chain it was started on has ended (tick 6 on), per tick:

  * how loud the interneurons still are (does it fall silent, hold, or blow up);
  * whether the motor cells say a stored item (at least 11 of its 12 bits), against what
    noise would manage;
  * when two ticks in a row both say stored items, whether the second is the LEARNED
    SUCCESSOR of the first, against the chance that a random pair of items is a link.
"""

from __future__ import annotations

import argparse

import torch

from tissue import chain
from tissue.train import load

CHAINS, LENGTH = 4, 4


def alone(path: str, m: int = 64, ticks: int = 60, lives: int = 128, seed: int = 7) -> dict:
    tissue, task, _ = load(path)
    brain = tissue.grow(m)
    gen = torch.Generator().manual_seed(seed)
    items = chain.worlds(task, CHAINS, LENGTH, lives, gen)
    b, c, n, d = items.shape
    silent = torch.zeros(b, d)
    flat = items.view(b, c * n, d)
    with torch.no_grad():
        state = brain.born(b, gen)
        links = [(ci, li) for ci in range(c) for li in range(n - 1)]
        for j in torch.randperm(len(links), generator=gen).tolist():
            ci, li = links[j]
            state = brain.rest(state)
            state, _ = brain.tick(state, items[:, ci, li], silent, 0.0, 1.0)
            state, _ = brain.tick(state, items[:, ci, li + 1], items[:, ci, li], 1.0, 1.0)
            state, _ = brain.tick(state, silent, items[:, ci, li + 1], 1.0, 0.0)
        state = brain.rest(state)
        state, _ = brain.tick(state, items[:, 0, 0], silent, 0.0, 1.0)
        best, sure, loud = [], [], []
        for _ in range(ticks):
            state, out = brain.tick(state, silent, silent, 0.0, 0.0)
            agree = ((out.unsqueeze(1) > 0) == (flat > 0)).float().mean(2)
            top, idx = agree.max(1)
            best.append(idx)
            sure.append(top >= (d - 1) / d - 1e-6)
            loud.append(state[0]["I"].abs().mean(1))
    best, sure, loud = torch.stack(best, 1), torch.stack(sure, 1), torch.stack(loud, 1)
    a, nxt = best[:, 6:-1], best[:, 7:]
    both = sure[:, 6:-1] & sure[:, 7:]
    link = (nxt == a + 1) & (a % n != n - 1)
    return {
        "loud": [float(loud[:, t].mean()) for t in (0, 9, 29, ticks - 1)],
        "stored": float(sure[:, 6:].float().mean()),
        "stored_noise": 1 - (1 - (d + 1) / 2 ** d) ** (c * n),
        "successor": float(link[both].float().mean()),
        "same": float((nxt == a)[both].float().mean()),
        "successor_chance": c * (n - 1) / (c * n) ** 2,
        "pairs": int(both.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--ticks", type=int, default=60)
    a = p.parse_args()
    torch.set_num_threads(4)
    for path in a.genomes:
        r = alone(path, a.m, a.ticks)
        print(path)
        print("  interneurons' mean |activity| at tick 1 / 10 / 30 / last: "
              + " / ".join(f"{x:.2f}" for x in r["loud"]))
        print(f"  ticks that say a stored item: {r['stored']:.2f}   (noise: {r['stored_noise']:.2f})")
        print(f"  two such ticks in a row: learned successor {r['successor']:.2f}, same item "
              f"{r['same']:.2f}   (chance for a successor: {r['successor_chance']:.2f}; "
              f"{r['pairs']} pairs)")


if __name__ == "__main__":
    main()
