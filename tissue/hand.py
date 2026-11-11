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


def delta_wires(c: torch.Tensor, answers: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    """One pass of the delta rule over the pairs in order: wires (B, answer bits, M)."""
    w = torch.zeros(c.shape[0], answers.shape[2], c.shape[2])
    for k in range(c.shape[1]):
        ck = c[:, k]
        err = answers[:, k] - torch.einsum("bom,bm->bo", w, ck)
        w = w + rate * err.unsqueeze(-1) * (ck / (ck * ck).sum(-1, keepdim=True)).unsqueeze(1)
    return w


def delta(c: torch.Tensor, answers: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    return torch.einsum("bom,bkm->bko", delta_wires(c, answers, rate), c)


def hop(items: torch.Tensor, m: int, f: float, gen: torch.Generator, rate: float = 1.0,
        exact: bool = False) -> torch.Tensor:
    """Exam 1 by hand, one hop: items (B, K, 2, d) are links first -> second.

    Wires between interneurons store code(first) -> code(second); the recalled code is
    cleaned by the interneuron's own threshold; a second set of wires, taught on every item
    seen, decodes a code back into the item. `exact`: both sets by least squares (not local).
    """
    b, _, _, d = items.shape
    wires = torch.randn(b, m, d, generator=gen)
    theta = math.sqrt(2) * torch.erfinv(torch.tensor(1 - 2 * f))

    def code(x):
        return (torch.einsum("bmd,bkd->bkm", wires, x) / math.sqrt(d) > theta).float() - f

    first, second = code(items[:, :, 0]), code(items[:, :, 1])
    seen, said = torch.cat([first, second], 1), torch.cat([items[:, :, 0], items[:, :, 1]], 1)
    if exact:
        recalled = first @ (torch.linalg.pinv(first) @ second)
        decode = (torch.linalg.pinv(seen) @ said).transpose(1, 2)
    else:
        recalled = delta(first, second, rate)
        decode = delta_wires(seen, said, rate)
    clean = (recalled > 0.5 - f).float() - f
    return torch.stack([torch.einsum("bom,bkm->bko", decode, first),      # hop 0, the echo
                        torch.einsum("bom,bkm->bko", decode, clean)], dim=2)


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


def yardstick_hop(task: Task, sizes=SIZES) -> dict:
    """K90 of one hand-made hop at every size: {"delta" | "lstsq": {m: (k90, f, rate)}}."""
    from tissue import chain
    out: dict = {"delta": {}, "lstsq": {}}
    for m in sizes:
        ks = chain.k_grid(m, task)
        for name, rates in (("delta", (1.0, 0.5, 0.25)), ("lstsq", (None,))):
            best = (0.0, None, None)
            for f in SPARSE:
                for rate in rates:
                    done, accs = chain.curve(
                        lambda items, g, f=f, rate=rate: hop(items, m, f, g, rate or 1.0, rate is None),
                        task, ks, stop_below=0.9)
                    best = max(best, (k90(done, [a[1] for a in accs]), f, rate))
            out[name][m] = best
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hop", action="store_true", help="the yardstick for exam 1 (one hop)")
    p.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    a = p.parse_args()
    torch.set_num_threads(8)
    if a.hop:
        y = yardstick_hop(Task(d_in=12, d_out=12), a.sizes)
        print(f"{'M':>5} | {'delta, one pass':>26} | {'least squares':>20}")
        for m in a.sizes:
            d, e = y["delta"][m], y["lstsq"][m]
            print(f"{m:>5} | {d[0]:7.1f} f={d[1]} rate {d[2]} | {e[0]:7.1f} f={e[1]}", flush=True)
        return
    y = yardstick(sizes=a.sizes)
    print(f"direct (no interneurons): K90 = {y['direct'][0][0]:.1f}")
    print(f"{'M':>5} | " + " | ".join(f"{r:>24}" for r in RULES))
    for m in a.sizes:
        print(f"{m:>5} | " + " | ".join(
            f"{y[r][m][0]:7.1f} f={y[r][m][1]:<4} {y[r][m][2]:<9}" for r in RULES))


if __name__ == "__main__":
    main()
