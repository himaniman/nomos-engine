"""The lineage: what crosses a life boundary, and how a dead end is named."""

import json
import os

import numpy as np

from cortex import brain as B
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
