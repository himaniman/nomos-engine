"""The observer: fixed exams, on a copy, frozen, and comparable across players."""

import numpy as np
import pytest

from cortex import brain as B
from cortex import tiles as T
from cortex.agent import Mind, fresh_genome, live
from cortex.baseline import random_policy, reflex_policy
from cortex.probes import (ARENA, PROBES, BrainDriver, PolicyDriver, probe_door,
                           probe_open, probe_poison, run_suite, run_suite_policy,
                           snapshot)
from cortex.senses import layout

L = layout(64)
TICKS = 500


def _mind(ticks=800):
    g = fresh_genome(seed=0)
    w, m = live(g, seed=2, max_ticks=ticks, think_every=3)
    return m


# --- the three rules a probe obeys ---------------------------------------------------

def test_a_probe_examines_a_COPY_and_cannot_touch_the_brain():
    m = _mind()
    before = (m.phenotype.mask.copy(), m.phenotype.state.copy(),
              m.phenotype.dW.copy(), m.phenotype.cells())
    run_suite(m, ticks=200)
    assert np.array_equal(m.phenotype.mask, before[0])
    assert np.array_equal(m.phenotype.state, before[1]), "the exam moved the brain"
    assert np.array_equal(m.phenotype.dW, before[2])
    assert m.phenotype.cells() == before[3]


def test_the_copy_carries_the_grown_body_and_what_was_learned():
    """The exam is of the developed, trained mind — not of the genome."""
    m = _mind(ticks=1500)
    copy = snapshot(m)
    assert copy.cells() == m.phenotype.cells() > m.genome.cells()
    assert np.array_equal(copy.mask, m.phenotype.mask)
    assert np.array_equal(copy.dW, m.phenotype.dW)
    assert np.array_equal(copy.state, m.phenotype.state)


def test_plasticity_is_frozen_unless_the_probe_is_about_learning():
    m = _mind()
    assert snapshot(m, learning=False).cfg.plastic is False
    assert snapshot(m, learning=False).cfg.rule_plastic is False
    assert snapshot(m, learning=True).cfg.plastic is True
    frozen = snapshot(m, learning=False)
    was = frozen.dW.copy()
    probe_open(BrainDriver(frozen), ticks=200)
    assert np.array_equal(frozen.dW, was), "the exam taught, so it is not an exam"


def test_a_body_does_not_grow_during_an_exam():
    m = _mind()
    copy = snapshot(m)
    n = copy.cells()
    probe_open(BrainDriver(copy), ticks=300)
    assert copy.cells() == n


def test_each_probe_gets_a_fresh_copy_so_order_cannot_matter():
    m = _mind()
    a = run_suite(m, ticks=200)["profile"]
    b = run_suite(m, ticks=200)["profile"]
    assert a == b


# --- the arena -----------------------------------------------------------------------

def test_the_arena_is_built_from_the_world_s_own_vocabulary():
    """A probe must not be a different game: same materials, same physics, same senses."""
    r = probe_open(PolicyDriver(random_policy, L), ticks=120)
    mats = set(np.unique(r.world.tiles.region(0, 0, ARENA, ARENA)).tolist())
    assert mats <= set(range(T.N_MATERIALS))
    assert T.GROUND in mats and T.STONE in mats
    assert r.world.cfg.n_rays == 64, "the arena changed the sensorium"


def test_the_live_world_is_never_touched():
    g = fresh_genome(seed=0)
    w, m = live(g, seed=2, max_ticks=600, think_every=3)
    tick, pos, tiles_v = w.tick, (w.agent.y, w.agent.x), w.tiles.version
    run_suite(m, ticks=200)
    assert (w.tick, (w.agent.y, w.agent.x), w.tiles.version) == (tick, pos, tiles_v)


# --- the profile ---------------------------------------------------------------------

def test_a_profile_is_a_vector_not_a_number():
    """The reason the suite exists: one life scored by one number is noise."""
    out = run_suite_policy(reflex_policy, L, ticks=TICKS)
    prof = out["profile"]
    assert len(prof) == len(PROBES) + 1
    for name, v in prof.items():
        assert 0.0 <= v <= 1.0, (name, v)
    assert set(out["detail"]) == set(p.__name__.replace("probe_", "") for p in PROBES)


def test_the_suite_separates_a_reflex_from_a_coin_flip():
    """Calibration: if it cannot tell twenty lines of steering from a random walk it
    cannot tell us anything about a brain either."""
    reflex = run_suite_policy(reflex_policy, L, ticks=TICKS)["profile"]
    coin = run_suite_policy(random_policy, L, ticks=TICKS)["profile"]
    assert reflex["open"] > coin["open"] + 0.3, (reflex["open"], coin["open"])


def test_the_poison_probe_catches_a_nose_that_cannot_see():
    """The reflex follows its nose and the nightcap is the nearer smell, so it eats
    what it should not. That is the specific gap a brain is meant to close."""
    r = probe_poison(PolicyDriver(reflex_policy, L), ticks=TICKS)
    assert r.detail["poison"] == 1, "the reflex avoided poison it cannot perceive"
    assert r.score < 0.5


def test_the_door_probe_is_solvable():
    """A probe nobody can pass measures nothing. A scripted walker that presses as it
    goes must get through."""
    def push_east(vec, w):
        return 1.0, 0.0, 1.0
    r = probe_door(push_east, ticks=TICKS)
    assert r.detail["opened"] == 1, "the door probe cannot be passed at all"
    assert r.score > 0.4


def test_probes_are_reproducible():
    a = run_suite_policy(reflex_policy, L, ticks=300)["profile"]
    b = run_suite_policy(reflex_policy, L, ticks=300)["profile"]
    assert a == b


def test_the_strip_renders():
    pytest.importorskip("pygame")
    import os
    out = run_suite_policy(random_policy, L, ticks=200)
    path = os.path.join("out", "cortex", "_test_probes.png")
    assert os.path.exists(__import__("cortex.probes", fromlist=["render"])
                          .render(out, path))
    os.remove(path)
