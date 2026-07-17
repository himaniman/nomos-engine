import numpy as np

from vivarium.config import Config
from vivarium.neat import N_ACTIONS, Genome, Phenotype, crossover, random_genome
from vivarium.run import run_headless
from vivarium.world import World, input_size


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


def test_neat_phenotype_acts_deterministically():
    n_in = input_size(5)
    rng = np.random.default_rng(0)
    g = random_genome(n_in, N_ACTIONS, rng)
    x = rng.normal(size=n_in)
    p1, p2 = Phenotype(g), Phenotype(g)
    assert p1.act(x) == p2.act(x)                            # same genome -> same act
    assert 0 <= p1.act(x) < N_ACTIONS


def test_neat_crossover_and_mutation_grow_topology():
    from vivarium.config import Config as Cfg
    n_in = input_size(3)
    rng = np.random.default_rng(1)
    g = random_genome(n_in, N_ACTIONS, rng)
    base_hidden = g.complexity()[0]
    nid = n_in + N_ACTIONS
    from vivarium.neat import mutate
    # force many add-node mutations -> hidden nodes appear
    cfg = Cfg(add_node_prob=1.0, add_conn_prob=1.0, toggle_prob=0.0)
    for _ in range(5):
        g, nid = mutate(g, rng, cfg, nid)
    assert g.complexity()[0] > base_hidden                  # topology grew
    # a child of two genomes is a valid runnable net
    g2 = random_genome(n_in, N_ACTIONS, rng)
    child = crossover(g, g2, rng)
    assert Phenotype(child).act(np.zeros(n_in)) in range(N_ACTIONS)


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
    w = World(Config(size=20, seed=0, init_population=1, corpse_decay=5))
    w.food[:] = False                    # nothing to eat -> it must die of metabolism
    a = next(iter(w.agents.values()))
    a.energy = 0.1                       # dies next tick whatever it does
    w.step()
    assert (w.corpse > 0).sum() == 1     # exactly one corpse appeared
    cy, cx = map(int, np.argwhere(w.corpse > 0)[0])
    assert not w._free(cy, cx)           # a corpse blocks its cell
    for _ in range(6):
        w.step()
    assert w.corpse[cy, cx] == 0         # that corpse has decayed away


def test_inner_walls_generated_and_reproducible():
    a = World(Config(size=64, seed=4))
    b = World(Config(size=64, seed=4))
    assert (a.walls == b.walls).all()                       # map is part of the seed
    interior = a.walls[1:-1, 1:-1]
    assert interior.sum() > 0                               # inner walls exist
    # roughly the requested density (loose bounds — strokes overlap/hit doorways)
    frac = interior.sum() / interior.size
    assert 0.02 < frac < 0.30


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
    rng = np.random.default_rng(0)
    n_in = input_size(5)
    genomes = [random_genome(n_in, N_ACTIONS, rng) for _ in range(3)]
    p = str(tmp_path / "pool.json")
    save_genomes(p, genomes)
    back = load_genomes(p)
    assert len(back) == 3
    for a, b in zip(genomes, back):
        assert a.nodes == b.nodes
        assert a.key() == b.key()                             # topology + weights match


def test_genome_pool_seeds_the_population():
    # with a pool of one known genome and mutation off, every initial agent IS
    # that genome (seeded from the pool, not fresh random)
    rng = np.random.default_rng(0)
    n_in = input_size(Config().n_rays)
    base = random_genome(n_in, N_ACTIONS, rng)
    cfg = Config(size=40, seed=2, init_population=12, weight_mut_prob=0.0,
                 add_conn_prob=0.0, add_node_prob=0.0, toggle_prob=0.0)
    w = World(cfg, genome_pool=[base])
    for a in w.agents.values():
        assert a.brain.genome.key() == base.key()             # exactly the pool genome


def test_walking_walls_shift_but_keep_shape():
    w = World(Config(size=64, seed=1, init_population=5, wall_drift_every=5,
                     wall_step_prob=1.0))
    before = w.walls.copy()
    d0 = int(before.sum())
    for _ in range(30):
        w.step()
    assert not np.array_equal(before, w.walls)               # positions shifted
    # strokes keep their shape, so total wall count barely changes (only clipping
    # at the border / centre patch moves it a little)
    assert abs(int(w.walls.sum()) - d0) <= d0 * 0.25 + 20
    assert w.walls[0, :].all() and w.walls[:, 0].all()        # border intact
    # each stroke is still a straight line of its original length
    for s in w.strokes:
        assert s["axis"] in ("h", "v") and s["length"] >= 3


def test_asexual_fallback_when_no_mate():
    # a lone very-full agent breeds by clone+mutate (no neighbour needed)
    w = World(Config(size=30, seed=0, init_population=3, repro_min_energy=25.0,
                     asexual_energy=40.0, start_energy=60.0))
    w.food[:] = False
    before = w.population()
    w.step()
    assert w.stats["births"] > 0
    assert w.population() > before


def test_sexual_reproduction_by_proximity():
    # two adjacent fed agents produce a crossover child (asexual disabled)
    w = World(Config(size=20, seed=0, init_population=2, repro_min_energy=20.0,
                     asexual_energy=9999.0))
    w.food[:] = False
    w.occ[:] = -1
    a0, a1 = list(w.agents.values())
    a0.y, a0.x = 10, 10
    a1.y, a1.x = 10, 11
    w.occ[10, 10] = a0.id
    w.occ[10, 11] = a1.id
    a0.energy = a1.energy = 50.0
    # call reproduction directly (a normal step would let them move apart first)
    w._try_reproduce(a0, set())
    assert w.stats["births"] > 0                 # only sexual could have fired
    assert w.population() == 3                    # a child was added
