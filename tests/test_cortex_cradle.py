"""The cradle: the bench that answers whether learning does anything.

Nothing here checks a SCORE — the cradle is deliberately an easy world and no number
from it counts for anything. What is worth pinning down is that the experiment can
answer its question at all: that the arms differ only in the one thing they claim to,
that reward actually arrives (otherwise the bench is testing starvation again), and that
running it cannot damage the brain being measured."""

import numpy as np
import pytest

from cortex import cradle
from cortex import probes
from cortex import reflex as R
from cortex.agent import Mind, fresh_genome
from cortex.world import Config as WorldConfig
from cortex.world import World

TICKS, PROBE = 700, 80


def test_the_cradle_actually_feeds():
    """The whole point: in the real world a brain gets ZERO positive reward events in
    4000 ticks, and you cannot tell a broken learner from a starved one."""
    _, _, life = cradle.live_in_cradle(fresh_genome(seed=0), seed=0, ticks=1500)
    assert life["meals"] > 0
    assert life["r_pos"] > 0, "no reward ever arrived — this is not a cradle"


def test_the_cradle_is_not_the_real_world():
    c, real = cradle.cradle_config(), WorldConfig()
    assert c.food_per_chunk > real.food_per_chunk
    assert c.poison_weight == 0.0            # a nightcap answers a different question


def test_the_ablation_arms_differ_only_in_what_they_claim_to():
    """Same body, same states, same wiring — the only difference is the deltas."""
    mind = Mind(fresh_genome(seed=0))
    w = World(seed=0, cfg=cradle.cradle_config())
    mind.attach(w)
    for _ in range(TICKS):
        mind.step(w)
    kept = probes.snapshot(mind)
    wiped = probes.snapshot(mind, unlearn=True)
    assert np.array_equal(kept.mask, wiped.mask)
    assert np.array_equal(kept.state, wiped.state)
    assert len(kept.axon_cells) == len(wiped.axon_cells)
    assert not np.allclose(kept.dW, wiped.dW) or np.allclose(kept.dW, 0.0)


def test_the_matrix_reports_a_learning_delta_with_and_without_the_arcs():
    """Four arms, two knobs. The claim the reflexes were built on is that learning
    should contribute MORE when eating is a consequence of a behaviour — which is only
    checkable if both deltas are reported."""
    out = cradle.matrix(seeds=(0,), ticks=TICKS, probe_ticks=PROBE, world="cradle",
                        quiet=True)
    assert set(out["rows"][0]["arms"]) == {(True, True), (True, False),
                                           (False, True), (False, False)}
    for key in ("with_arcs", "without_arcs"):
        assert "mean" in out[key]


def test_measuring_a_brain_does_not_change_it():
    """Every arm runs on a fresh deep copy; a bench that lobotomised its subject would
    poison every number after the first."""
    mind = Mind(fresh_genome(seed=0))
    w = World(seed=0, cfg=cradle.cradle_config())
    mind.attach(w)
    for _ in range(TICKS):
        mind.step(w)
    before = (mind.phenotype.dW.copy(), mind.phenotype.mask.copy(),
              R.count(mind.phenotype), mind.genome.P_in.copy())
    cradle.learning_delta(mind, probe_ticks=PROBE)
    probes.run_suite(mind, ticks=PROBE, no_reflex=True)
    assert np.array_equal(mind.phenotype.dW, before[0])
    assert np.array_equal(mind.phenotype.mask, before[1])
    assert R.count(mind.phenotype) == before[2]
    assert np.array_equal(mind.genome.P_in, before[3])


def test_the_sweep_scores_by_ablation_not_by_the_raw_ladder():
    """Growth must not be able to take the credit for learning."""
    rows = cradle.sweep(seeds=(0,), ticks=TICKS, probe_ticks=PROBE,
                        grid=((0.5, 0.002), (0.95, 0.0002)), quiet=True)
    assert len(rows) == 2
    for r in rows:
        assert set(r) == {"elig", "forget", "meals", "delta"}
