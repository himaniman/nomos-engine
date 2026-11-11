"""Hand-made memories for exam 0: what each size can hold when the rule is written by hand.

    python -m tissue.hand

Nothing here is searched for. They are the yardstick a found genome is read against:

  direct   no interneurons: sensory -> motor wires, Hebb. Does not depend on size.
  hebb     M interneurons with random innate input wires and a threshold, Hebb on their
           wires to the motor cells (Marr's expansion). One multiply per wire per pair.
  delta    the same code, but a wire changes by the motor cell's ERROR, one pass.
  lstsq    the same code, the best possible linear read-out of it (not a local rule):
           the ceiling for this kind of interneuron.

Each uses the sparseness (share of interneurons a cue lights) that serves it best.
"""

from __future__ import annotations

import math

import torch

from tissue.assoc import Task, curve, k90, k_grid

SPARSE = (0.5, 0.2, 0.1, 0.05, 0.02)
SIZES = (8, 16, 32, 64, 128, 256, 512)


def codes(cues: torch.Tensor, m: int, f: float, gen: torch.Generator) -> torch.Tensor:
    """What M threshold interneurons say about each cue, centred: (B, K, M)."""
    b, _, d = cues.shape
    wires = torch.randn(b, m, d, generator=gen)
    drive = torch.einsum("bmd,bkd->bkm", wires, cues) / math.sqrt(d)
    theta = math.sqrt(2) * torch.erfinv(torch.tensor(1 - 2 * f))
    return (drive > theta).float() - f


def hebb(c: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bko,bkm,bjm->bjo", answers, c, c)


def delta(c: torch.Tensor, answers: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    w = torch.zeros(c.shape[0], answers.shape[2], c.shape[2])
    for k in range(c.shape[1]):
        ck = c[:, k]
        err = answers[:, k] - torch.einsum("bom,bm->bo", w, ck)
        w = w + rate * err.unsqueeze(-1) * (ck / (ck * ck).sum(-1, keepdim=True)).unsqueeze(1)
    return torch.einsum("bom,bkm->bko", w, c)


def lstsq(c: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    return c @ (torch.linalg.pinv(c) @ answers)


RULES = {
    "hebb": {"hebb": hebb},
    "delta": {f"rate {r}": (lambda c, a, r=r: delta(c, a, r)) for r in (1.0, 0.5, 0.25)},
    "lstsq": {"lstsq": lstsq},
}


def direct(cues, answers, gen):
    return hebb(cues, answers)


def expanded(rule, m: int, f: float):
    return lambda cues, answers, gen: rule(codes(cues, m, f, gen), answers)


def yardstick(task: Task = Task(), sizes=SIZES) -> dict:
    """K90 of every hand memory at every size, each at its best setting:
    {name: {m: (k90, sparseness, variant)}}."""
    ks = k_grid(16, task)
    out: dict = {"direct": {0: (k90(ks, curve(direct, task, ks)), None, None)}}
    for name, variants in RULES.items():
        out[name] = {}
        for m in sizes:
            ks = k_grid(m, task)
            out[name][m] = max((k90(ks, curve(expanded(rule, m, f), task, ks)), f, v)
                               for f in SPARSE for v, rule in variants.items())
    return out


def main() -> None:
    torch.set_num_threads(8)
    y = yardstick()
    print(f"direct (no interneurons): K90 = {y['direct'][0][0]:.1f}")
    print(f"{'M':>5} | " + " | ".join(f"{r:>24}" for r in RULES))
    for m in SIZES:
        print(f"{m:>5} | " + " | ".join(
            f"{y[r][m][0]:7.1f} f={y[r][m][1]:<4} {y[r][m][2]:<9}" for r in RULES))


if __name__ == "__main__":
    main()
