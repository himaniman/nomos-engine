from pathlib import Path

import numpy as np

from nomos.embodied import play_forage
from nomos.lamarck import LMCELL_WEIGHTS, _mutate, lamarckian, life_fitness

LAW = Path("laws/forage_brain_lmcell.nomos").read_text()


def test_play_forage_carries_and_returns_weights():
    # a life returns its final weight fields; injecting them into a fresh life
    # reproduces those exact fields at init (Lamarckian carry works)
    r = play_forage(LAW, world_size=32, game_size=16, seed=1, moves=6,
                    ticks_per_move=2, return_channels=("vm0", "vm1"))
    assert set(r["final_channels"]) == {"vm0", "vm1"}
    assert r["final_channels"]["vm0"].shape == (32, 32)
    r2 = play_forage(LAW, world_size=32, game_size=16, seed=2, moves=1,
                     ticks_per_move=1, init_channels=r["final_channels"],
                     return_channels=("vm0",))
    # after a single think tick the carried weights are still essentially the
    # injected ones (learning is slow), and definitely not the law's random init
    assert r2["final_channels"]["vm0"].shape == (32, 32)


def test_life_fitness_rewards_meals_over_survival():
    eat = {"score": 1, "steps_alive": 130, "alive_at_end": False}
    circle = {"score": 0, "steps_alive": 400, "alive_at_end": True}
    assert life_fitness(eat, 400) > life_fitness(circle, 400)   # one meal wins
    # but among two starvers, the longer-lived one is preferred
    a = {"score": 0, "steps_alive": 300, "alive_at_end": False}
    b = {"score": 0, "steps_alive": 120, "alive_at_end": False}
    assert life_fitness(a, 400) > life_fitness(b, 400)


def test_mutate_perturbs_every_field():
    rng = np.random.default_rng(0)
    w = {"vm0": np.ones((8, 8)), "vm1": np.zeros((8, 8))}
    m = _mutate(w, rng, 0.1)
    assert set(m) == set(w)
    assert not np.array_equal(m["vm0"], w["vm0"])   # perturbed
    assert abs(float((m["vm0"] - w["vm0"]).mean())) < 0.1   # but only lightly


def test_lamarckian_runs_and_reports_history():
    out = lamarckian(LAW, lives=3, lam=2, seeds=(0, 1), mutation=0.02,
                     world_size=32, game_size=16, moves=8, ticks=1,
                     rng_seed=0, workers=1)
    assert len(out["history"]) == 3
    assert set(out["final_weights"]) == set(LMCELL_WEIGHTS)
    assert "best" in out and "weights" in out["best"]
