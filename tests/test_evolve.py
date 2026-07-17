import random

from nomos import evolve
from nomos.dsl import parse_program


def test_law_from_genome_is_valid_nomos():
    law = evolve.law_from_genome(dict(evolve.DEFAULT_GENOME))
    prog = parse_program(law)                       # must parse
    names = {c.name for c in prog.channels}
    assert {"food", "danger", "motor"} <= names


def test_mutate_stays_within_bounds():
    rng = random.Random(0)
    g = dict(evolve.DEFAULT_GENOME)
    for _ in range(200):
        g = evolve.mutate(g, rng, strength=0.5)
        for k, (lo, hi) in evolve.BOUNDS.items():
            assert lo <= g[k] <= hi


def test_forage_law_from_genome_is_valid_and_evolves():
    law = evolve.law_from_genome(dict(evolve.FORAGE_DEFAULT), game="forage")
    prog = parse_program(law)
    names = {c.name for c in prog.channels}
    assert {"food", "wall", "sight", "motor", "mem"} <= names
    # a short forage evolution runs, stays deterministic, and never regresses
    res = evolve.evolve(generations=2, lam=2, seeds=(0,), world_size=24,
                        game_size=12, moves=12, ticks=1, rng_seed=1, game="forage")
    assert len(res["history"]) == 2
    fits = [h["fitness"] for h in res["history"]]
    assert all(b >= a for a, b in zip(fits, fits[1:]))


def test_forage_mutate_stays_within_bounds():
    rng = random.Random(0)
    g = dict(evolve.FORAGE_DEFAULT)
    for _ in range(100):
        g = evolve.mutate(g, rng, strength=0.5, game="forage")
        for k, (lo, hi) in evolve.FORAGE_BOUNDS.items():
            assert lo <= g[k] <= hi


def test_fitness_is_deterministic_and_scored():
    a = evolve.fitness(evolve.DEFAULT_GENOME, seeds=(0, 1), world_size=24,
                       game_size=10, moves=30, ticks=2)
    b = evolve.fitness(evolve.DEFAULT_GENOME, seeds=(0, 1), world_size=24,
                       game_size=10, moves=30, ticks=2)
    assert a == b
    # meals dominate, plus survival ticks and an immortal bonus
    expected = a["score"] * 100.0 + a["survival"] * 0.1 + 40.0 * (a["immortal"] / 2)
    assert a["fitness"] == expected


def test_evolve_reports_history_and_best_never_regresses():
    calls = []
    res = evolve.evolve(generations=3, lam=3, seeds=(0, 1), world_size=24,
                        game_size=10, moves=30, ticks=2, rng_seed=1,
                        on_generation=lambda g, genome, fit: calls.append(g))
    assert calls == [0, 1, 2]
    assert len(res["history"]) == 3
    # elitist parent fitness is monotone non-decreasing across generations
    fits = [h["fitness"] for h in res["history"]]
    assert all(b >= a for a, b in zip(fits, fits[1:]))
    assert res["best_fitness"]["fitness"] >= fits[0]
