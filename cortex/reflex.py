"""Innate reflexes: arcs the genome is born knowing, and can unlearn only with effort.

Why these exist, precisely — because the reason matters more than the mechanism and the
obvious reason is the wrong one. It is NOT "to raise the score". It is this, measured
(docs/lab-journal.md, 2026-07-24): with the eligibility trace finally long enough to
reach the behaviour that earned a meal, within-life learning still contributes nothing,
because **what the agent does and whether it eats are uncorrelated.** It blunders into
food, so what gets reinforced is whatever it happened to be doing at the time, and a
gradient over noise is noise.

An innate approach fixes that at the root: it makes eating a CONSEQUENCE of a behaviour,
which is the precondition for reinforcing that behaviour at all. Reflexes do not replace
learning here. They manufacture the correlation learning needs.

**An arc, not a shortcut.** The temptation is to write `command = reflex(sensors) +
sheet_output` and be done in ten lines. That would make the tissue a decoration on a
hand-written policy, which is the opposite of the whole playground. So a reflex here is
built out of what the brain already is:

  1. a SENSORY PROJECTION — the odorant's forward/lateral components drive a reserved
     state channel, so the organs that smell are holding the signal;
  2. a PIONEER TRACT — axons from those organs to the effector cells, grown at the first
     development tick and thereafter treated like any other wire, except that they decay
     far more slowly. Biology's pioneer axons are guided to a specific target by specific
     cues, which is exactly this, and it is not the same thing as `_sprout`'s growth
     cones wandering up a gradient;
  3. a COHERENT READ-OUT — that channel wired to the command it means.

All three are ordinary weights and ordinary axons. Mutation jitters them, plasticity
moves them, and the cell rules write to the same channels, so the tissue can suppress or
amplify or repurpose any of it. Hardwired, and retunable with effort — which is what was
asked for.

Ablating them is one call (`probes.snapshot(..., no_reflex=True)`), and it is the whole
reason to be careful here: without it, "the tissue improved on its reflexes" cannot be
told apart from "the tissue rode its reflexes", and we would have built a reflex agent
with a decorative brain and no way to notice.
"""

from __future__ import annotations

import numpy as np

from cortex.senses import ODOUR_CH
from cortex.world import N_ODOUR

# The three commands, repeated here rather than imported: `brain` imports this module to
# grow the tracts, so importing back would be a cycle. A test asserts they still agree.
THRUST, TURN, INTERACT = 0, 1, 2

# name -> (which odour sub-channel drives it, which command it means, sign)
#
# Deliberately only SMELL. Smell is the one sense that says which way to GO — it is
# geodesic here, so it flows around walls and its direction is a walkable path, not a
# bearing through rock. Vision would need the agent to already know which hue is food,
# and that is the poison experiment's whole question; an innate arc must not answer it.
ARCS: dict[str, tuple[int, int, float]] = {
    # the walkable path to the strongest smell leads forward -> go
    "smell_forward": (1, THRUST, +1.0),
    # ...or off to one side -> turn that way
    "smell_lateral": (2, TURN, +1.0),
    # a smell this strong means it is under your nose -> try to eat it
    "smell_here": (0, INTERACT, +1.0),
}
NAMES = tuple(ARCS)
ODOUR_MOD = 7                # the modality id of the odour block (cortex/senses.layout)


def channels(genome) -> dict[str, int]:
    """Which state channel carries each arc.

    The first electric channels, so they are ordinary signalling channels that the cell
    rules also write to. That is on purpose: a reflex the tissue cannot argue with is
    not a reflex, it is a wire, and this playground is about tissue that computes."""
    return {name: i for i, name in enumerate(NAMES) if i < genome.C}


def install(genome, layout, gain: float = 2.5, out_gain: float = 2.5) -> None:
    """Write the innate arcs into a genome's projections, in place.

    `gain` drives the reserved channel from the smell; `out_gain` reads that channel
    out as its command. Both are ordinary weights afterwards — this only decides what
    the brain is BORN believing."""
    if gain <= 0.0 and out_gain <= 0.0:
        return
    # BORN WITH A NOSE. `organ_map` says which chemical region becomes which sense, and
    # it is drawn at random from nine modalities — so most fresh genomes had no odour
    # organ anywhere and could not smell at all, which is why the first tracts refused
    # to grow (0, 3 and 3 of a requested 16 across three brains). A reflex has to bring
    # the organ it needs; a species is born with a nose, not with a lottery over senses.
    # Only ONE region is claimed, so the other chemicals still decide their own fates.
    if ODOUR_MOD not in list(genome.organ_map):
        genome.organ_map[0] = ODOUR_MOD
    lo, hi = layout["slices"]["odour"]
    ch = channels(genome)
    for name, (sub, cmd, sign) in ARCS.items():
        c = ch.get(name)
        if c is None:
            continue
        # every odorant feeds the same arc: the agent is not told WHICH smell is food,
        # only that a smell is worth walking towards. Which one to trust is learnable
        # and is exactly what the poison probe asks about later.
        idx = [lo + j * ODOUR_CH + sub for j in range(N_ODOUR) if lo + j * ODOUR_CH + sub < hi]
        genome.P_in[idx, c] = sign * gain
        genome.P_out[c, :] = 0.0
        genome.P_out[c, cmd] = sign * out_gain


def sense_rows(phenotype) -> np.ndarray:
    """Rows of the living body that can HEAR a smell — the organs a tract starts from.
    Modality 9 is interoception, 8 touch, 7 the odour block, 1..6 single rays; only 7
    hears smell, and an arc that started anywhere else would carry nothing."""
    mod = phenotype.organ_mod[phenotype._ys, phenotype._xs]
    alive_in = phenotype.in_mask[phenotype._ys, phenotype._xs]
    return np.nonzero(alive_in & (mod == 7))[0]


def pioneer(phenotype, per_organ: int = 1, weight: float = 0.9,
            max_tracts: int = 24) -> int:
    """Grow the innate tracts: nose organs -> effectors, nearest first.

    A PIONEER, not a growth cone. `Phenotype._sprout` walks up a chemical and lands
    wherever the gradient leads, which is how a topology gets discovered; a pioneer has
    a target and is guided to it, which is how a species is born already able to do
    something. Both end up as ordinary axons in the same list.

    Deterministic: sources in row order, each paired with its nearest free effector,
    ties broken by row index — the same discipline as everywhere else here."""
    srcs = sense_rows(phenotype)
    dsts = phenotype._out_rows
    if not len(srcs) or not len(dsts):
        return 0                       # nothing to wire yet; try again next dev tick
    ys, xs = phenotype._ys, phenotype._xs
    existing = {(a[0], a[1], a[2], a[3]) for a in phenotype.axon_cells}
    made = 0
    for i in srcs:
        if made >= max_tracts:
            break
        sy, sx = int(ys[i]), int(xs[i])
        d = (ys[dsts] - sy) ** 2 + (xs[dsts] - sx) ** 2
        for k in np.argsort(d, kind="stable")[:per_organ]:
            j = int(dsts[int(k)])
            dy, dx = int(ys[j]), int(xs[j])
            if (sy, sx) == (dy, dx) or (sy, sx, dy, dx) in existing:
                continue
            phenotype.axon_cells.append([sy, sx, dy, dx, float(weight), 1])
            existing.add((sy, sx, dy, dx))
            made += 1
    return made


def cut(phenotype) -> int:
    """Sever every innate tract, leaving grown ones alone. The ablation arm."""
    keep = [a for a in phenotype.axon_cells if not (len(a) > 5 and a[5])]
    n = len(phenotype.axon_cells) - len(keep)
    if n:
        phenotype.axon_cells = keep
        phenotype._rewire()
    return n


def silence(genome) -> None:
    """Zero the innate projections. Together with `cut` this removes the whole arc —
    the sensory drive, the tract and the read-out — rather than only its wiring, which
    is the difference between asking "did the wires matter" and "did the reflex"."""
    for c in channels(genome).values():
        genome.P_in[:, c] = 0.0
        genome.P_out[c, :] = 0.0


def count(phenotype) -> int:
    return sum(1 for a in phenotype.axon_cells if len(a) > 5 and a[5])
