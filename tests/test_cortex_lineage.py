"""The lineage: what crosses a life boundary, and how a dead end is named."""

import json
import os

import numpy as np
import pytest

from cortex import brain as B
from cortex import lineage
from cortex.agent import fresh_genome
from cortex.lineage import diagnose, evaluate, run


def test_a_generation_returns_a_profile_not_a_number():
    """At lambda = 1 there is one sample per turn, so it had better say more than
    'better' or 'worse' — this is the thing an LLM would be handed later."""
    g = fresh_genome(seed=0)
    prof = evaluate(g, seeds=[2], max_ticks=200, think_every=4)
    for key in ("ticks", "meals", "gained", "distance", "entropy", "dW",
                "meal_rate", "poisoned", "immortal", "fitness"):
        assert key in prof, f"the profile says nothing about {key}"
    assert prof["ticks"] == 200


def test_evaluation_is_reproducible():
    g = fresh_genome(seed=0)
    a = evaluate(g, seeds=[2, 3], max_ticks=150, think_every=4)
    b = evaluate(g, seeds=[2, 3], max_ticks=150, think_every=4)
    assert a["fitness"] == b["fitness"] and a["ate"] == b["ate"]


def test_dead_ends_are_named_not_just_detected():
    """The point is which INTERVENTION is called for, so the flags must separate."""
    ok = {"entropy": 2.0, "meals": 9, "dW": 0.5}
    assert diagnose(ok, [1.0, 2.0, 3.0, 4.0]) == []
    assert "frozen-behaviour" in diagnose({**ok, "entropy": 0.0}, [1.0])
    assert "not-eating" in diagnose({**ok, "meals": 0}, [1.0])
    assert "not-learning" in diagnose({**ok, "dW": 0.0}, [1.0])
    # a lineage whose best has not moved in `patience` generations has stalled
    assert "stalled" in diagnose(ok, [5.0, 5.0, 5.0, 5.0, 5.0])
    assert "stalled" not in diagnose(ok, [1.0, 2.0, 3.0, 9.0])


def test_only_the_genome_crosses_the_boundary():
    """A life's learned weights must not ride along: the next life regrows from the
    program, so an improvement is attributable to the program."""
    g = fresh_genome(seed=0)
    out = run(generations=1, seeds=[2], max_ticks=120, think_every=4,
              save="", log=False, quiet=True)
    child = out["genome"]
    from cortex.agent import Mind
    fresh = Mind(child)
    assert fresh.phenotype.plastic_norm() == 0.0, "learning leaked across the boundary"
    assert fresh.thoughts == 0


def test_the_best_genome_is_saved_and_reloadable(tmp_path):
    path = str(tmp_path / "best.json")
    out = run(generations=1, seeds=[2], max_ticks=120, think_every=4,
              save=path, log=False, quiet=True)
    assert os.path.exists(path)
    back = B.Genome.from_dict(json.load(open(path, encoding="utf-8")))
    assert back.cells() == out["genome"].cells()
    assert back.n_in == out["genome"].n_in
    v = np.random.default_rng(0).normal(0, 0.4, back.n_in)
    from cortex.senses import layout
    L = layout(64)
    assert B.Phenotype(back, B.Config(), L).act(v) == \
           B.Phenotype(out["genome"], B.Config(), L).act(v)


def test_a_lineage_never_goes_backwards():
    """(1 + lambda): the parent is only replaced by something at least as good."""
    out = run(generations=4, seeds=[2], max_ticks=150, think_every=4,
              save="", log=False, quiet=True)
    h = out["history"]
    assert all(h[i + 1] >= h[i] - 1e-9 for i in range(len(h) - 1)), h


# --- the step size, and why it adapts --------------------------------------------------

def test_a_scaled_config_changes_only_how_far_a_mutation_reaches():
    """Not the ceilings, not which mechanisms exist — only the rates."""
    base = B.Config()
    half = lineage.scaled(base, 0.5)
    for k in lineage._STEP_KEYS:
        assert getattr(half, k) < getattr(base, k) + 1e-12
    for k in ("max_cells", "max_types", "max_layers", "grid", "channels", "axons"):
        assert getattr(half, k) == getattr(base, k)


def test_probabilities_never_scale_past_one():
    big = lineage.scaled(B.Config(), lineage.STEP_MAX * 10)
    for k in lineage._STEP_KEYS:
        if k.endswith("prob"):
            assert getattr(big, k) <= 1.0


def test_the_step_shrinks_on_rejection_and_grows_on_acceptance():
    """Rechenberg's 1/5 rule. The first lineage run on a foraging brain accepted ZERO
    of fourteen children, which says the operator was leaping, not stepping."""
    step = 1.0
    for _ in range(6):
        step = lineage.adapt(step, accepted=False)
    assert step < 0.5
    up = step
    for _ in range(6):
        up = lineage.adapt(up, accepted=True)
    assert up > step


def test_the_step_is_bounded_both_ways():
    down, up = 1.0, 1.0
    for _ in range(200):
        down = lineage.adapt(down, accepted=False)
        up = lineage.adapt(up, accepted=True)
    assert down == pytest.approx(lineage.STEP_MIN)
    assert up == pytest.approx(lineage.STEP_MAX)


def test_mutation_is_gentle_enough_to_be_a_step():
    """Measured: the old rates changed all the weights, a quarter of the growth program
    and several structural moves at once, on a body grown from a 24-cell seed."""
    cfg = B.Config()
    assert cfg.weight_mut <= 0.03 and cfg.weight_mut_prob <= 0.05
    assert cfg.program_mut_prob <= 0.10
    assert cfg.grow_prob <= 0.10 and cfg.add_in_prob <= 0.05


def test_a_generation_is_judged_on_a_rotating_window_of_worlds():
    """A lineage scored on one map learns that map — nomos/lamarck.py measured exactly
    that. And it is what makes re-measuring the incumbent mean anything: `evaluate` is
    a pure function of (genome, seeds, ticks), so re-scoring on the SAME seeds returns
    the same number to the last digit."""
    pool = [0, 1, 2, 3, 4, 5]
    seen = {tuple(lineage.worlds_for(pool, g, 2)) for g in range(6)}
    assert len(seen) >= 3, "every generation saw the same worlds"
    for g in range(20):
        w = lineage.worlds_for(pool, g, 2)
        assert len(w) == 2 and len(set(w)) == 2
        assert all(s in pool for s in w)


def test_a_pool_smaller_than_the_window_still_works():
    assert lineage.worlds_for([7], 3, 2) == [7]
    assert sorted(lineage.worlds_for([7, 9], 3, 5)) == [7, 9]


def test_the_window_is_a_pure_function_of_the_generation():
    """Reproducibility: a run is a pure function of (genome, seeds, steps)."""
    assert lineage.worlds_for(range(6), 4, 2) == lineage.worlds_for(range(6), 4, 2)


def test_the_exam_does_not_select():
    """docs/cortex-design.md: the ladder is an OBSERVER, not a curriculum. It used to
    sit in fitness at 40x the profile mean, called a tie-break — against a `gained` of
    100 to 280 that is too small to steer and big enough to flip a coin toss, which is
    the worst of both. A measurement that also steers the thing it measures is not a
    measurement."""
    g = fresh_genome(seed=0)
    prof = evaluate(g, seeds=[2], max_ticks=200, think_every=4, probe_ticks=60)
    assert prof["fitness"] == pytest.approx(prof["gained"] + 0.002 * prof["ticks"])
    assert prof["skills"], "the profile must still be measured and reported"


def test_a_brain_good_at_tests_and_bad_at_living_does_not_win():
    """The other half of the same claim, and the reason the term was there at all."""
    lived = {"gained": 200.0, "ticks": 8000, "skills": {"mean": 0.10}}
    exam = {"gained": 40.0, "ticks": 5000, "skills": {"mean": 0.95}}
    fit = lambda p: p["gained"] + 0.002 * p["ticks"]            # noqa: E731
    assert fit(lived) > fit(exam)


def test_parent_and_child_are_judged_on_the_same_worlds():
    """The comparison has to be PAIRED. Averaging the parent's fresh score with one it
    carried from different maps adds two incomparable numbers: the bar swung between 84
    and 152 over eighteen generations — a spread as large as the differences between
    children — and acceptance became a lottery on which maps a child happened to draw."""
    seen = []
    real = lineage.evaluate

    def spy(genome, seeds, *a, **kw):
        seen.append(tuple(seeds))
        return real(genome, seeds, *a, **kw)

    lineage.evaluate = spy
    try:
        run(generations=2, seeds=(0, 1, 2, 3), max_ticks=150, think_every=4,
            per_gen=2, save="", log=False, quiet=True, strips="")
    finally:
        lineage.evaluate = real
    # generation 0 is scored once; each generation after scores a child and the parent,
    # and those two calls must name the same worlds
    for child_worlds, parent_worlds in zip(seen[1::2], seen[2::2]):
        assert child_worlds == parent_worlds
