"""Look inside a found genome: what its cells say, as plain tables.

    python -m tissue.inside out/tissue/v0/tissue-s0.pt

Three things a rung-0 genome consists of:
  * the interneuron's CODE: its activity as a function of its dendrite, and how alike the
    codes of two cues are when the cues differ in h of their bits (0 = unrelated codes);
  * the motor cell's ANSWER as a function of its dendrite (no teacher);
  * the motor cell's LEARNING SIGNAL as a function of its dendrite and the teacher.
Then the found code is put under the hand-written rules, to tell the code's merit from the
rule's.
"""

from __future__ import annotations

import argparse
import math

import torch

from tissue import hand
from tissue.assoc import curve, k90, k_grid
from tissue.train import load


def code_likeness(code, d: int, m: int = 4096, pairs: int = 200, seed: int = 1) -> list[float]:
    """Mean correlation between the codes of two cues that differ in h = 1..d/2 bits."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for h in range(1, d // 2 + 1):
        a = torch.randint(0, 2, (pairs, d), generator=gen).float() * 2 - 1
        b = a.clone()
        flip = torch.rand(pairs, d, generator=gen).argsort(dim=1)[:, :h]
        b.scatter_(1, flip, -a.gather(1, flip))
        wires = torch.randn(pairs, m, d, generator=gen)
        ca = code(torch.einsum("pmd,pd->pm", wires, a) / math.sqrt(d))
        cb = code(torch.einsum("pmd,pd->pm", wires, b) / math.sqrt(d))
        ca, cb = ca - ca.mean(1, keepdim=True), cb - cb.mean(1, keepdim=True)
        out.append(float(((ca * cb).sum(1) / (ca.norm(dim=1) * cb.norm(dim=1))).mean()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genome")
    p.add_argument("--sizes", type=int, nargs="*", default=[64, 512])
    a = p.parse_args()
    torch.set_num_threads(6)
    tissue, task, _ = load(a.genome)
    gain = float(torch.exp(tissue.gain["0"].detach()))
    cell_i, cell_o = tissue.cells["I"], tissue.cells["O"]

    def code(drive):                       # drive: unit-variance input before the gain
        return cell_i((drive * gain).unsqueeze(-1))[..., 0]

    with torch.no_grad():
        print(f"input gain {gain:.2f}")
        xs = torch.linspace(-3, 3, 25)
        print("\ninterneuron: activity against its dendrite (in standard deviations)")
        print("  " + " ".join(f"{x:+.2f}" for x in xs.tolist()))
        print("  " + " ".join(f"{y:+.2f}" for y in code(xs).tolist()))

        print("\ncode likeness of two cues h bits apart (of %d)" % task.d_in)
        print("  h          " + " ".join(f"{h:6d}" for h in range(1, task.d_in // 2 + 1)))
        print("  found      " + " ".join(f"{v:6.3f}" for v in code_likeness(code, task.d_in)))
        for f in (0.5, 0.05, 0.02):
            theta = math.sqrt(2) * torch.erfinv(torch.tensor(1 - 2 * f))
            like = code_likeness(lambda dr: (dr > theta).float(), task.d_in)
            print(f"  hand f={f:<4}" + " ".join(f"{v:6.3f}" for v in like))

        ms = torch.linspace(-2, 2, 9)

        def motor(m, y, flag):
            return cell_o(torch.stack([m, torch.full_like(m, y), torch.full_like(m, flag)], -1))

        print("\nmotor cell against its dendrite m:   " + " ".join(f"{m:+.2f}" for m in ms.tolist()))
        print("  answer, no teacher                 "
              + " ".join(f"{v:+.2f}" for v in motor(ms, 0.0, 0.0)[..., 0].tolist()))
        print("  learning signal, no teacher        "
              + " ".join(f"{v:+.2f}" for v in motor(ms, 0.0, 0.0)[..., 1].tolist()))
        print("  learning signal, teacher says +1   "
              + " ".join(f"{v:+.2f}" for v in motor(ms, 1.0, 1.0)[..., 1].tolist()))
        print("  learning signal, teacher says -1   "
              + " ".join(f"{v:+.2f}" for v in motor(ms, -1.0, 1.0)[..., 1].tolist()))

        # whose merit: the found CODE under the hand-written RULES (hand codes: `tissue.hand`)
        mean = float(code(torch.randn(200_000)).mean())

        def found_codes(cues, m, gen):
            wires = torch.randn(cues.shape[0], m, task.d_in, generator=gen)
            return code(torch.einsum("bmd,bkd->bkm", wires, cues) / math.sqrt(task.d_in)) - mean

        print("\nK90 of the found code under hand rules (each at its best setting)")
        for m in a.sizes:
            ks = k_grid(m, task)
            print(f"  M={m:<4} " + " | ".join(
                f"{name} {max(k90(ks, curve(lambda c, t, g, r=r: r(found_codes(c, m, g), t), task, ks)) for r in variants.values()):6.1f}"
                for name, variants in hand.RULES.items()))


if __name__ == "__main__":
    main()
