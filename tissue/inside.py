"""Look inside a found genome: what its cells say, as plain tables.

    python -m tissue.inside out/tissue/v0/tissue-s0.pt

What a genome consists of:
  * the interneuron's CODE: its activity against its sensory dendrite, and how alike the
    codes of two cues are when the cues differ in h of their bits (0 = unrelated codes);
  * rung 1 only: what an interneuron does with the dendrite from the OTHER interneurons,
    senses on and senses silent, and what it teaches those wires;
  * the motor cell's ANSWER against its dendrite, and its LEARNING SIGNAL with and without
    the teacher.
For rung 0 the found code is then put under the hand-written rules, to tell the code's
merit from the rule's.
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


def row(label: str, values) -> None:
    print(f"  {label:<38}" + " ".join(f"{v:+.2f}" for v in values))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genome")
    p.add_argument("--sizes", type=int, nargs="*", default=[64, 512])
    a = p.parse_args()
    torch.set_num_threads(6)
    tissue, task, saved = load(a.genome)
    chained = saved["arm"] in ("chain", "stream")
    gain = float(torch.exp(tissue.gain["0"].detach()))
    cell_i, cell_o = tissue.cells["I"], tissue.cells["O"]

    def inter(drive, rec=0.0, on=1.0):
        """An interneuron's outputs. `drive`: sensory dendrite in standard deviations."""
        feats = [drive * gain]
        if chained:
            feats += [torch.full_like(drive, rec) if isinstance(rec, float) else rec,
                      torch.full_like(drive, on)]
        return cell_i(torch.stack(feats, -1))

    def code(drive):
        return inter(drive)[..., 0]

    with torch.no_grad():
        print(f"{a.genome}: {saved['arm']}, input gain {gain:.2f}")
        xs = torch.linspace(-3, 3, 25)
        print("\ninterneuron: activity against its sensory dendrite (standard deviations)")
        print("  " + " ".join(f"{x:+.2f}" for x in xs.tolist()))
        print("  " + " ".join(f"{y:+.2f}" for y in code(xs).tolist()))

        print("\ncode likeness of two cues h bits apart (of %d)" % task.d_in)
        print("  h          " + " ".join(f"{h:6d}" for h in range(1, task.d_in // 2 + 1)))
        print("  found      " + " ".join(f"{v:6.3f}" for v in code_likeness(code, task.d_in)))
        for f in (0.5, 0.05):
            theta = math.sqrt(2) * torch.erfinv(torch.tensor(1 - 2 * f))
            like = code_likeness(lambda dr: (dr > theta).float(), task.d_in)
            print(f"  hand f={f:<4}" + " ".join(f"{v:6.3f}" for v in like))

        if chained:
            rs = torch.linspace(-2, 2, 9)
            zero = torch.zeros_like(rs)
            print("\ninterneuron against the dendrite from OTHER interneurons:")
            row("that dendrite", rs.tolist())
            row("activity, senses silent", inter(zero, rs, 0.0)[..., 0].tolist())
            row("teaches its wires, senses silent", inter(zero, rs, 0.0)[..., 1].tolist())
            for drive in (-1.0, 0.0, 1.0):
                d = torch.full_like(rs, drive)
                said = inter(d, rs, 1.0)
                print(f"  senses on, sensory dendrite {drive:+.1f} (alone it says {float(code(d[:1])):+.2f}):")
                row("    activity", said[..., 0].tolist())
                row("    teaches its wires", said[..., 1].tolist())

        ms = torch.linspace(-2, 2, 9)

        def motor(m, y, flag):
            return cell_o(torch.stack([m, torch.full_like(m, y), torch.full_like(m, flag)], -1))

        print("\nmotor cell:")
        row("its dendrite", ms.tolist())
        row("answer, no teacher", motor(ms, 0.0, 0.0)[..., 0].tolist())
        row("learning signal, no teacher", motor(ms, 0.0, 0.0)[..., 1].tolist())
        row("learning signal, teacher says +1", motor(ms, 1.0, 1.0)[..., 1].tolist())
        row("learning signal, teacher says -1", motor(ms, -1.0, 1.0)[..., 1].tolist())

        if not chained:
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
