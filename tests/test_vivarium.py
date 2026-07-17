import numpy as np

from vivarium.brain import Brain, N_ACTIONS, input_size
from vivarium.config import Config
from vivarium.run import run_headless
from vivarium.world import World


def test_world_inits_bounded_with_population_and_food():
    w = World(Config(size=40, seed=1, init_population=20))
    assert w.walls[0, :].all() and w.walls[:, 0].all()      # border walls
    assert w.walls[-1, :].all() and w.walls[:, -1].all()
    assert w.population() == 20
    assert w.food.sum() > 0                                  # food seeded
    # agents sit on distinct open cells
    cells = {(a.y, a.x) for a in w.agents.values()}
    assert len(cells) == 20
    assert not any(w.walls[y, x] for y, x in cells)


def test_brain_genome_roundtrip_and_deterministic_action():
    n_in = input_size(5)
    rng = np.random.default_rng(0)
    b = Brain.random(n_in, 8, rng)
    assert b.genome.size == Brain.genome_size(n_in, 8)
    x = rng.normal(size=n_in)
    assert b.act(x) == b.act(x)                              # pure function
    assert 0 <= b.act(x) < N_ACTIONS


def test_run_is_deterministic():
    cfg = Config(size=48, seed=7, init_population=25)
    a = run_headless(cfg, ticks=300, every=50)
    b = run_headless(cfg, ticks=300, every=50)
    assert a == b                                           # same seed -> same history


def test_different_seed_diverges():
    a = run_headless(Config(size=48, seed=1, init_population=25), 200, 50)
    b = run_headless(Config(size=48, seed=2, init_population=25), 200, 50)
    assert a != b


def test_food_stays_near_target_and_sim_survives_long_run():
    cfg = Config(size=64, seed=3, init_population=40)
    w = World(cfg)
    open_cells = int((~w.walls).sum())
    target = cfg.food_target * open_cells
    for _ in range(500):
        w.step()
        assert w.food.sum() <= target + cfg.food_regrow + 1   # regrow is capped
    # the world never gets stuck empty: reseed keeps life going
    assert w.population() > 0


def test_corpse_blocks_then_decays():
    from vivarium.brain import TURN_LEFT
    w = World(Config(size=20, seed=0, init_population=1, corpse_decay=5))
    a = next(iter(w.agents.values()))
    y, x = a.y, a.x
    a.brain.w2[:] = 0.0                  # force it to turn in place (won't move/eat)
    a.brain.b2[:] = 0.0
    a.brain.b2[TURN_LEFT] = 10.0
    a.energy = 0.1                       # will die next metabolism tick
    w.step()
    assert w.corpse[y, x] > 0            # a corpse now blocks that cell
    assert not w._free(y, x)
    for _ in range(6):
        w.step()
    assert w.corpse[y, x] == 0          # it has decayed away


def test_reproduction_can_occur():
    # a brain hardwired to always REPRODUCE, dropped into a fed world, breeds
    from vivarium.brain import REPRODUCE
    w = World(Config(size=30, seed=0, init_population=6, repro_min_energy=10.0,
                     start_energy=40.0))
    for a in w.agents.values():
        # bias the reproduce output high so argmax picks it
        a.brain.b2[:] = 0.0
        a.brain.b2[REPRODUCE] = 10.0
        a.brain.w2[:] = 0.0
    before = w.population()
    w.step()
    assert w.stats["births"] > 0
    assert w.population() > before
