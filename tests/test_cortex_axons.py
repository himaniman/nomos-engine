"""Axons: the one thing a 3x3 neighbourhood cannot express.

Why they exist, measured before they were built: the SAME grown body scored 0.09 ->
0.16 -> 0.26 -> 0.34 on the `open` probe as its think-time went 10 -> 20 -> 40 -> 160
sub-ticks. A sheet without shortcuts is a RELAY — one cell per sub-tick — so a sense on
one rim and an effector on the other are further apart in TIME than a thought is long.
An axon is a shortcut with a weight, and a set of shortcuts is a topology."""

import math

import numpy as np

from cortex import brain as B
from cortex.senses import layout, sensor_size

N_IN = sensor_size(64)
LAYOUT = layout(64)


def _pheno(cfg=None, seed=0):
    cfg = cfg or B.Config()
    g = B.random_genome(N_IN, np.random.default_rng(seed), cfg)
    return g, B.Phenotype(g, cfg, LAYOUT)


def _grow(p, rounds=60, energy=1.0):
    v = np.zeros(p.genome.n_in)
    for _ in range(rounds):
        p.act(v, hormone=[0.6, 0.0, 0.0])
        p.develop(energy_frac=energy, embryo=True)


# --- they grow -----------------------------------------------------------------------

def test_axons_grow_and_reach_further_than_the_neighbourhood():
    _, p = _pheno()
    _grow(p)
    st = p.axon_stats()
    assert st["n"] > 0, "no axon was ever grown"
    assert st["mean_len"] >= p.cfg.axon_min_len, st
    for a in p.axon_cells:
        assert math.hypot(a[2] - a[0], a[3] - a[1]) >= p.cfg.axon_min_len
        assert p.mask[a[0], a[1]] and p.mask[a[2], a[3]], "an axon ends nowhere"


def test_growing_a_wire_costs_resource():
    _, p = _pheno()
    _grow(p, 30, energy=1.0)
    fed = p.axon_stats()["n"]
    _, q = _pheno()
    _grow(q, 30, energy=0.0)
    assert fed > q.axon_stats()["n"], (fed, q.axon_stats()["n"])


def test_axons_are_pruned_when_nothing_reinforces_them():
    """Use it or lose it: without reward the weights decay and the wire is removed, so
    a topology is LOST as well as grown. Otherwise the sheet only accumulates."""
    _, p = _pheno()
    _grow(p, 40)
    assert p.axon_stats()["n"] > 0
    before = p.axon_stats()["mean_w"]
    for _ in range(200):                      # no reward, no rehearsal
        p.develop(energy_frac=0.0)
    assert p.axon_stats()["mean_w"] < before
    assert p.axon_stats()["n"] == 0, "a wire nothing uses should eventually go"


def test_reward_keeps_the_wires_that_fired_together():
    _, p = _pheno()
    _grow(p, 40)
    n0 = p.axon_stats()["n"]
    assert n0 > 0
    w0 = p._ax_w.copy()
    v = np.zeros(p.genome.n_in)
    p.act(v, hormone=[0.6, 0.0, 0.0])
    p.axon_learn(1.0)
    assert (p._ax_w > w0).any(), "a reward left no mark on any wire"
    assert np.abs(p._ax_w).max() <= p.cfg.axon_clip + 1e-9


# --- they carry signal ----------------------------------------------------------------

def test_an_axon_actually_moves_signal():
    cfg = B.Config()
    _, p = _pheno(cfg)
    _grow(p, 40)
    assert p.axon_stats()["n"] > 0
    v = np.random.default_rng(3).normal(0, 0.4, p.genome.n_in)
    p.act(v, hormone=[0.5, 0.0, 0.0])
    with_wires = p.state.copy()
    p.state[:] = 0.0
    keep, p.axon_cells = p.axon_cells, []      # same body, wires cut
    p._rewire()
    p.act(v, hormone=[0.5, 0.0, 0.0])
    assert not np.allclose(with_wires, p.state), "cutting every axon changed nothing"
    p.axon_cells = keep


def test_axons_can_be_switched_off_for_the_ab():
    off = B.Config(axons=False)
    _, p = _pheno(off)
    _grow(p)
    assert p.axon_stats()["n"] == 0
    assert p.cells() > off.seed_cells, "the control arm must still develop"


def test_a_wire_dies_with_its_cell():
    _, p = _pheno()
    _grow(p, 40)
    assert p.axon_cells
    ax = p.axon_cells[0]
    p.mask[ax[2], ax[3]] = False               # kill the far end
    p._reindex()
    assert all(not (a[2] == ax[2] and a[3] == ax[3]) for a in p.axon_cells), \
        "an axon outlived the cell it pointed at"


def test_axon_growth_is_reproducible():
    _, a = _pheno(seed=5)
    _, b = _pheno(seed=5)
    _grow(a, 40)
    _grow(b, 40)
    assert a.axon_cells == b.axon_cells
