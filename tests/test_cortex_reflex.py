"""Innate reflexes: the arcs a genome is born knowing.

The claims worth pinning are the ones that keep the idea HONEST. A reflex is easy to
build as a bypass — `command = policy(sensors) + sheet` — and that would make the tissue
a decoration on a hand-written policy while every number improved. So: the arc must run
through the brain's own machinery, it must be ordinary weights that mutation and
plasticity can argue with, and it must be ABLATABLE, because without that "the tissue
improved on its reflexes" cannot be told apart from "the tissue rode its reflexes"."""

import numpy as np
import pytest

from cortex import brain as B
from cortex import probes
from cortex import reflex as R
from cortex.agent import Mind, fresh_genome
from cortex.senses import layout, rays_from_size, sensor_size
from cortex.world import Config as WorldConfig
from cortex.world import World

N_IN = sensor_size(64)
LAYOUT = layout(64)


def _grown(seed=0, ticks=500, **program):
    """A brain that has actually lived a little, so its body has organs and effectors
    and its tracts have had development ticks to form in."""
    g = fresh_genome(seed=seed)
    if program:
        g.program.update(program)
    w = World(seed=0, cfg=WorldConfig())
    mind = Mind(g)
    mind.attach(w)
    for _ in range(ticks):
        mind.step(w)
    return mind


# --- the arc is made of the brain's own parts ----------------------------------------

def test_the_command_ids_still_agree_with_the_brain():
    """`reflex` re-declares them rather than importing, to break a cycle."""
    assert (R.THRUST, R.TURN, R.INTERACT) == (B.THRUST, B.TURN, B.INTERACT)


def test_a_fresh_genome_is_born_knowing_the_arcs():
    g = fresh_genome(seed=0)
    ch = R.channels(g)
    assert set(ch) == set(R.NAMES)
    for name, (_, cmd, _) in R.ARCS.items():
        c = ch[name]
        assert np.abs(g.P_in[:, c]).max() > 0, f"{name} hears nothing"
        assert np.argmax(np.abs(g.P_out[c])) == cmd, f"{name} is read out as the wrong command"


def test_a_fresh_genome_is_born_with_a_nose():
    """The first tracts refused to grow because `organ_map` draws modalities at random
    from nine and most bodies simply had no odour organ — so there was nothing for an
    arc to start from. A reflex has to bring the organ it needs."""
    for s in range(6):
        assert R.ODOUR_MOD in list(fresh_genome(seed=s).organ_map), f"seed {s} has no nose"


def test_only_one_chemical_region_is_claimed_for_smell():
    """The other chemicals must still be free to decide their own fates, or development
    stops choosing anything."""
    g = fresh_genome(seed=0)
    assert list(g.organ_map).count(R.ODOUR_MOD) <= 2


def test_the_arcs_are_ordinary_weights_that_mutation_can_move():
    """'Retunable with effort' has to be true, not aspirational."""
    g = fresh_genome(seed=0)
    c = R.channels(g)["smell_forward"]
    rng = np.random.default_rng(0)
    before = g.P_in[:, c].copy()
    child = g
    for _ in range(40):
        child = B.mutate(child, rng)
    assert not np.array_equal(child.P_in[:, c], before)


def test_reflex_gain_zero_is_a_brain_born_knowing_nothing():
    """The control arm has to be reachable from the program alone — and it is the
    ORDINARY random genome, not a genome with holes punched in it: `install` simply
    declines to write anything, so no channel is reserved and nothing is zeroed."""
    cfg = B.Config(reflex_gain=0.0, reflex_out=0.0)
    g = B.random_genome(N_IN, np.random.default_rng(0), cfg)
    plain = B.random_genome(N_IN, np.random.default_rng(0),
                            B.Config(reflex_gain=0.0, reflex_out=0.0, axons=False))
    assert np.array_equal(g.P_in, plain.P_in) and np.array_equal(g.P_out, plain.P_out)
    for name, (_, cmd, _) in R.ARCS.items():
        c = R.channels(g)[name]
        assert np.count_nonzero(g.P_out[c]) > 1, "an arc was written anyway"
    assert R.count(_grown(seed=2, reflex_gain=0.0).phenotype) == 0


# --- the tracts ------------------------------------------------------------------------

def test_tracts_grow_from_nose_organs_to_effectors():
    mind = _grown(seed=2)
    p = mind.phenotype
    assert R.count(p) > 0, "no innate tract ever formed"
    noses = {(int(p._ys[i]), int(p._xs[i])) for i in R.sense_rows(p)}
    effs = {(int(p._ys[i]), int(p._xs[i])) for i in p._out_rows}
    for a in p.axon_cells:
        if len(a) > 5 and a[5]:
            assert (a[0], a[1]) in noses, "a tract starts somewhere that cannot smell"
            assert (a[2], a[3]) in effs, "a tract ends somewhere with no voice"


def test_tracts_are_capped_by_the_program():
    mind = _grown(seed=2, reflex_tracts=4)
    assert R.count(mind.phenotype) <= 4


def test_a_tract_is_not_pruned_when_a_grown_axon_would_be():
    """The asymmetry IS 'hardwired but retunable': plasticity may move a tract's weight
    anywhere, but neglect alone must not dissolve it."""
    mind = _grown(seed=2, reflex_decay=0.0)
    p = mind.phenotype
    for a in p.axon_cells:
        a[4] = p.cfg.axon_min_w * 0.01           # every wire now far below the floor
    before_innate = R.count(p)
    assert before_innate > 0
    p._prune_axons()
    assert R.count(p) == before_innate, "an innate tract was pruned"
    assert all(len(a) > 5 and a[5] for a in p.axon_cells), "a grown axon survived the floor"


def test_a_tract_decays_far_more_slowly_than_a_grown_axon():
    p = _grown(seed=2).phenotype
    innate = [a for a in p.axon_cells if len(a) > 5 and a[5]]
    assert innate
    assert p.cfg.reflex_decay < p.cfg.axon_decay


def test_tracts_survive_a_body_that_keeps_changing():
    """Development re-points every wire; a tract must not silently become a grown one,
    nor point at whatever cell inherited its row."""
    mind = _grown(seed=2, ticks=900)
    p = mind.phenotype
    n = R.count(p)
    p._reindex()
    assert R.count(p) == n


# --- the ablation, which is the whole reason to be careful ---------------------------

def test_cutting_the_tracts_leaves_the_grown_ones():
    p = _grown(seed=2).phenotype
    grown = [a for a in p.axon_cells if not (len(a) > 5 and a[5])]
    n_innate = R.count(p)
    assert n_innate and R.cut(p) == n_innate
    assert R.count(p) == 0
    assert len(p.axon_cells) == len(grown)


def test_ablation_keeps_the_body_and_removes_the_whole_arc():
    mind = _grown(seed=2)
    src = mind.phenotype
    p = probes.snapshot(mind, no_reflex=True)
    assert np.array_equal(p.mask, src.mask), "the ablation changed the body"
    assert np.array_equal(p.state, src.state)
    assert R.count(p) == 0                                  # the tracts...
    for c in R.channels(mind.genome).values():              # ...and the projections
        assert np.abs(p.genome.P_in[:, c]).max() == 0.0
        assert np.abs(p.genome.P_out[c]).max() == 0.0


def test_ablation_never_damages_the_real_genome():
    """A probe that quietly lobotomised its subject would poison every later run."""
    mind = _grown(seed=2)
    before = mind.genome.P_in.copy()
    probes.snapshot(mind, no_reflex=True)
    probes.run_suite(mind, ticks=60, no_reflex=True)
    assert np.array_equal(mind.genome.P_in, before)
    assert R.count(mind.phenotype) > 0, "the live brain lost its tracts to an exam"


def test_the_arc_actually_changes_what_the_brain_does():
    """If ablating it changed nothing, the reflex would be decoration and every number
    it improved would be a coincidence."""
    mind = _grown(seed=2)
    with_arc = probes.run_suite(mind, ticks=120)["profile"]
    without = probes.run_suite(mind, ticks=120, no_reflex=True)["profile"]
    assert with_arc != without


def test_a_smell_dead_ahead_drives_thrust_through_the_tissue():
    """End to end, and through the brain's own machinery rather than around it: a
    sensor vector in, a command out, with only the arc to carry it."""
    mind = _grown(seed=2)
    p = probes.snapshot(mind)
    lo, _ = mind.layout["slices"]["odour"]
    quiet = np.zeros(mind.genome.n_in)
    loud = quiet.copy()
    loud[lo + 0] = 1.0                      # intensity...
    loud[lo + 1] = 1.0                      # ...and the walkable path leads FORWARD
    for _ in range(3):                      # let the sheet settle under each input
        base = p.act(quiet)[0]
    for _ in range(3):
        smelled = p.act(loud)[0]
    assert smelled > base, "a smell straight ahead did not make it go"


def test_a_lineage_cannot_lose_its_last_nose_in_one_mutation():
    """A cliff, not a gradient: with no odour region nothing smells, the pioneer tracts
    never grow, and every descendant is blind to the one sense the reflexes are built
    on — silently, because there is nothing left for selection to act on."""
    g = fresh_genome(seed=0)
    rng = np.random.default_rng(0)
    for i in range(300):
        g = B.mutate(g, rng)
        if g.program["reflex_gain"] <= 0.0:
            break            # arcs switched off: the organ they needed may go too
        assert R.ODOUR_MOD in list(g.organ_map), f"went noseless at mutation {i}"


def test_but_a_species_with_no_reflexes_may_lose_it():
    """The guard is for arcs that need the organ, not a permanent fixture of biology.

    Re-mapping is driven hard here rather than waited for: the claim is about what the
    guard PERMITS, and tying it to the shipped rate would make it a slow test of
    `organ_mod_prob` instead."""
    g = fresh_genome(seed=0)
    g.program.update({"reflex_gain": 0.0})
    cfg = B.Config(organ_mod_prob=1.0)
    rng = np.random.default_rng(1)
    lost = False
    for _ in range(200):
        g = B.mutate(g, rng, cfg)
        g.program.update({"reflex_gain": 0.0})
        if R.ODOUR_MOD not in list(g.organ_map):
            lost = True
            break
    assert lost
