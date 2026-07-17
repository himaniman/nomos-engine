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


def test_inner_walls_generated_and_reproducible():
    a = World(Config(size=64, seed=4))
    b = World(Config(size=64, seed=4))
    assert (a.walls == b.walls).all()                       # map is part of the seed
    interior = a.walls[1:-1, 1:-1]
    assert interior.sum() > 0                               # inner walls exist
    # roughly the requested density (loose bounds — strokes overlap/hit doorways)
    frac = interior.sum() / interior.size
    assert 0.04 < frac < 0.30


def test_vision_is_blocked_by_walls():
    # a wall between the agent and food hides the food (line of sight, no smell)
    w = World(Config(size=30, seed=0, init_population=1, n_rays=1, fov=0.0,
                     ray_length=10))
    a = next(iter(w.agents.values()))
    w.walls[:] = False
    w.walls[0, :] = w.walls[-1, :] = w.walls[:, 0] = w.walls[:, -1] = True
    w.food[:] = False
    a.y, a.x, a.heading = 10, 10, 1          # facing east (+x)
    w.food[10, 14] = True                    # food 4 cells ahead, clear line
    seen = w._sense(a)
    assert seen[0] > 0                       # food channel of the single ray fires
    w.walls[10, 12] = True                   # drop a wall between agent and food
    blocked = w._sense(a)
    assert blocked[0] == 0                   # food no longer visible...
    assert blocked[1] > 0                    # ...an obstacle is seen instead


def test_app_renders_headless():
    # the drawing must work without a display, so screenshots/tests are possible
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    from vivarium.app import draw, genome_color
    pygame.init()
    font = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 13)
    w = World(Config(size=32, seed=1, init_population=15))
    for _ in range(50):
        w.step()
    sel = next(iter(w.agents), None)
    surf = draw(w, cell_px=6, panel_w=200, font=font, small=small, selected=sel)
    assert surf.get_width() == w.w * 6 + 200
    assert surf.get_height() == w.h * 6
    # a genome maps to a stable colour
    g = next(iter(w.agents.values())).brain.genome
    assert genome_color(g) == genome_color(g.copy())
    pygame.quit()


def test_genome_save_load_roundtrip(tmp_path):
    from vivarium.genome_io import load_genomes, save_genomes
    genomes = [np.arange(10.0), np.ones(10), np.linspace(-1, 1, 10)]
    p = str(tmp_path / "pool.npz")
    save_genomes(p, genomes)
    back = load_genomes(p)
    assert len(back) == 3
    assert all(np.allclose(a, b) for a, b in zip(genomes, back))


def test_genome_pool_seeds_the_population():
    # with a pool of one known genome, every initial agent is that genome + a
    # little mutation (not fresh random)
    rng = np.random.default_rng(0)
    n_in = input_size(Config().n_rays)
    base = Brain.random(n_in, Config().hidden, rng).genome
    w = World(Config(size=40, seed=2, init_population=12, mutation=0.01),
              genome_pool=[base])
    for a in w.agents.values():
        assert np.abs(a.brain.genome - base).mean() < 0.1     # near the pool genome


def test_living_walls_drift_but_hold_density():
    w = World(Config(size=64, seed=1, init_population=5, wall_drift_every=10,
                     wall_drift_frac=0.05))
    before = w.walls.copy()
    d0 = int(before.sum())
    for _ in range(40):
        w.step()
    assert not np.array_equal(before, w.walls)               # the map has shifted
    assert abs(int(w.walls.sum()) - d0) <= d0 * 0.2 + 10      # density roughly held
    assert w.walls[0, :].all() and w.walls[:, 0].all()        # border intact


def test_reproduction_is_automatic_when_well_fed():
    # reproduction is an innate drive: a well-fed agent breeds without choosing to
    w = World(Config(size=30, seed=0, init_population=6, repro_min_energy=20.0,
                     start_energy=50.0))
    before = w.population()
    w.step()
    assert w.stats["births"] > 0                 # well-fed agents bred automatically
    assert w.population() > before
