"""Baselines and session records — the numbers that give a brain's score a meaning."""

import json

import numpy as np

from cortex import tiles as T
from cortex.baseline import play, random_policy, reflex_policy
from cortex.senses import ScentField, layout, sense
from cortex.session import load, record, summarise
from cortex.world import BERRY, Config, FoodSpot, World


def _bare(seed=0, side=64):
    w = World(seed=seed, cfg=Config(size=side))
    for y in range(side):
        for x in range(side):
            w.tiles.set(y, x, T.GROUND)
    for cy in range(side // w.cfg.chunk + 1):
        for cx in range(side // w.cfg.chunk + 1):
            w.food[(cy, cx)] = []
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    return w


# --- sessions ----------------------------------------------------------------------

def test_a_session_records_what_matters(tmp_path):
    w = _bare()
    w.food[(0, 0)] = [FoodSpot(y=20.1, x=20.1, kind=BERRY)]
    w.step(interact=1.0)
    for _ in range(30):
        w.step(thrust=1.0)
    path = str(tmp_path / "sessions.jsonl")
    rec = record(w, "tester", note="unit", path=path)
    assert rec["meals"] == 1 and rec["ate"] == {"berry": 1}
    assert rec["gained"] == 12.0
    assert rec["deeds"]["eat"] == 1
    assert rec["distance"] > 5.0
    assert rec["seed"] == w.seed and rec["ticks"] == w.tick
    assert json.loads(open(path, encoding="utf-8").read().strip()) == rec


def test_sessions_accumulate_and_summarise(tmp_path):
    path = str(tmp_path / "s.jsonl")
    assert "no sessions" in summarise(path)
    for who in ("human", "reflex", "reflex"):
        w = _bare()
        for _ in range(20):
            w.step(thrust=1.0)
        record(w, who, path=path)
    assert len(load(path)) == 3
    table = summarise(path)
    assert "human" in table and "reflex" in table
    assert table.count("\n") == 2, "one header and one row per player"


def test_distance_is_what_was_walked_not_what_was_asked_for():
    """A wall takes its cut: commanding movement into stone must not count as travel."""
    w = _bare()
    for y in range(64):
        w.tiles.set(y, 22, T.STONE)
    for _ in range(60):
        w.step(thrust=1.0)
    assert w.distance < 3.0, f"walked {w.distance:.1f} cells into a wall"


def test_deeds_count_what_the_verb_achieved():
    w = _bare()
    w.agent.heading = 0.0
    w.tiles.set(20, 21, T.DOOR)
    w.step(interact=1.0)                                  # one press: one deed
    assert w.deeds["door"] == 1 and w.deeds["idle"] == 0
    w.tiles.set(20, 21, T.GROUND)
    for _ in range(w.cfg.interact_every):                 # wait out the cooldown, press
        w.step()
    w.step(interact=1.0)
    assert w.deeds["idle"] == 1, "a press that achieves nothing is still a press"
    assert w.deeds["door"] == 1


# --- policies ----------------------------------------------------------------------

def test_policies_emit_a_valid_action_tuple():
    w = World(seed=3)
    vec = sense(w, ScentField(w.cfg.smell_radius))
    rng = np.random.default_rng(0)
    for policy in (random_policy, reflex_policy):
        state = {"layout": layout(w.cfg.n_rays)}
        for _ in range(50):
            thrust, turn, act = policy(vec, w, rng, state)
            assert 0.0 <= thrust <= 1.0
            assert -1.0 <= turn <= 1.0
            assert act in (0.0, 1.0)


def test_the_reflex_walks_to_a_smell_and_eats_it():
    """The whole loop in one assertion: geodesic smell -> steering -> the verb."""
    w = _bare()
    w.food[(0, 0)] = [FoodSpot(y=20.0, x=30.0, kind=BERRY)]   # 10 cells dead ahead
    field = ScentField(w.cfg.smell_radius)
    rng = np.random.default_rng(0)
    state = {"layout": layout(w.cfg.n_rays)}
    for _ in range(400):
        w.step(*reflex_policy(sense(w, field), w, rng, state))
        if w.agent.ate[BERRY]:
            break
    assert w.agent.ate[BERRY] == 1, "the reflex never found a berry ten cells away"


def test_the_reflex_only_reads_the_sensor_vector():
    """It must be beatable on the same terms a brain gets — no peeking at the world."""
    w = World(seed=3)
    field = ScentField(w.cfg.smell_radius)
    state = {"layout": layout(w.cfg.n_rays)}
    vec = sense(w, field)
    a = reflex_policy(vec, w, np.random.default_rng(0), dict(state))
    w.agent.y += 11.0                                        # move the world, not the
    w.agent.x -= 7.0                                         # senses it was handed
    b = reflex_policy(vec, w, np.random.default_rng(0), dict(state))
    assert a == b, "the policy read the world behind the sensor vector"


def test_a_baseline_life_is_reproducible():
    a = play("reflex", seed=3, max_ticks=800)
    b = play("reflex", seed=3, max_ticks=800)
    assert (a.tick, a.agent.y, a.agent.x, a.agent.energy) == \
           (b.tick, b.agent.y, b.agent.x, b.agent.energy)
    assert a.agent.ate == b.agent.ate
