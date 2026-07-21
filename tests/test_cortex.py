"""Stage-1 ground truth for the cortex world: generation is deterministic and
seam-free, the body never overlaps a solid tile, and nothing tunnels at any speed."""

import math
import os

import numpy as np
import pytest

from cortex import tiles as T
from cortex.tiles import TileMap
from cortex.world import Config, World


# --- generation ------------------------------------------------------------------

def test_same_seed_same_map():
    a = TileMap(seed=7).region(0, 0, 96, 96)
    b = TileMap(seed=7).region(0, 0, 96, 96)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a = TileMap(seed=1).region(0, 0, 96, 96)
    b = TileMap(seed=2).region(0, 0, 96, 96)
    assert not np.array_equal(a, b)


def test_visit_order_does_not_matter():
    """A chunk rasterises its 3x3 neighbourhood, so a feature straddling a chunk edge
    must look identical whether the neighbour was generated before or after it."""
    forward = TileMap(seed=3)
    whole = forward.region(0, 0, 128, 128)
    backward = TileMap(seed=3)
    for cy in range(3, -1, -1):                       # touch the chunks in reverse
        for cx in range(3, -1, -1):
            backward.region(cy * 32, cx * 32, 32, 32)
    assert np.array_equal(whole, backward.region(0, 0, 128, 128))


def test_region_matches_per_cell_reads():
    m = TileMap(seed=11)
    reg = m.region(40, 40, 20, 20)
    for i in range(0, 20, 3):
        for j in range(0, 20, 3):
            assert reg[i, j] == m.material(40 + i, 40 + j)


def test_world_is_walled_and_edits_stick():
    m = TileMap(seed=5, size=64)
    assert m.material(-1, 10) == T.STONE and m.material(64, 10) == T.STONE
    m.set(10, 10, T.RUBBLE)
    assert m.material(10, 10) == T.RUBBLE
    m.set(-5, -5, T.GROUND)                            # out of bounds is a no-op
    assert m.material(-5, -5) == T.STONE


def test_map_has_a_mix_of_materials():
    reg = TileMap(seed=4).region(0, 0, 128, 128)
    seen = set(np.unique(reg).tolist())
    for mat in (T.GROUND, T.STONE, T.TRUNK, T.FOLIAGE):
        assert mat in seen, f"{T.NAMES[mat]} never generated"
    open_frac = float((~T.SOLID[reg]).mean())
    assert 0.35 < open_frac < 0.95, f"world is not walkable enough: {open_frac:.2f}"


# --- collision -------------------------------------------------------------------

def test_blocked_circle_is_a_circle_not_a_point():
    m = TileMap(seed=0, size=32)
    for y in range(32):                                # clear a patch, then wall one cell
        for x in range(32):
            m.set(y, x, T.GROUND)
    m.set(10, 10, T.STONE)
    assert m.blocked_circle(10.0, 10.0, 0.45)
    assert m.blocked_circle(10.0, 10.9, 0.45)          # body overlaps, centre does not
    assert not m.blocked_circle(10.0, 11.0, 0.45)      # exactly one cell away: clear


def test_body_never_overlaps_a_wall_over_a_run():
    w = World(seed=2)
    rng = np.random.default_rng(0)
    for _ in range(600):
        w.step(thrust=float(rng.random()), turn=float(rng.uniform(-1, 1)))
        assert not w.tiles.blocked_circle(w.agent.y, w.agent.x, w.cfg.radius)


def test_no_tunnelling_at_any_speed():
    """A huge single move must stop AT the wall, never behind it."""
    w = World(seed=0, cfg=Config(size=64))
    for y in range(64):
        for x in range(64):
            w.tiles.set(y, x, T.GROUND)
    for y in range(64):
        w.tiles.set(y, 40, T.STONE)                    # a full-height wall at x = 40
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    w._advance(20.0, 60.0)                             # 40 cells in one call
    assert w.agent.x < 40.0
    assert not w.tiles.blocked_circle(w.agent.y, w.agent.x, w.cfg.radius)


def test_wall_hit_slides_instead_of_sticking():
    w = World(seed=0, cfg=Config(size=64))
    for y in range(64):
        for x in range(64):
            w.tiles.set(y, x, T.GROUND)
    for y in range(64):
        w.tiles.set(y, 30, T.STONE)
    w.agent.y, w.agent.x = 20.0, 28.0
    y_before = w.agent.y
    w._advance(w.agent.y + 3.0, w.agent.x + 3.0)       # diagonal into the wall
    assert w.agent.y > y_before + 1.0, "should have slid along the wall"
    assert w.agent.x < 30.0


# --- the tick --------------------------------------------------------------------

def test_moving_costs_more_than_standing():
    a = World(seed=1)
    b = World(seed=1)
    for _ in range(50):
        a.step(thrust=1.0)
        b.step(thrust=0.0)
    assert a.agent.energy < b.agent.energy
    assert b.agent.energy < b.cfg.start_energy, "standing still is not free"


def test_energy_is_life_and_age_only_caps():
    w = World(seed=1, cfg=Config(size=128, start_energy=0.05))
    while w.agent.alive and w.tick < 100:
        w.step(thrust=1.0)
    assert not w.agent.alive and w.agent.energy <= 0.0
    assert w.agent.age < w.cfg.max_age, "died of starvation, not of old age"


def test_run_is_a_pure_function_of_seed_and_actions():
    def run():
        w = World(seed=9)
        rng = np.random.default_rng(4)
        for _ in range(300):
            w.step(thrust=float(rng.random()), turn=float(rng.uniform(-1, 1)))
        return w.agent.y, w.agent.x, w.agent.heading, w.agent.energy
    assert run() == run()


def test_spawn_is_free_and_stable():
    a, b = World(seed=6), World(seed=6)
    assert (a.agent.y, a.agent.x) == (b.agent.y, b.agent.x)
    assert not a.tiles.blocked_circle(a.agent.y, a.agent.x, a.cfg.radius)


def test_dead_agent_stops_moving():
    w = World(seed=1, cfg=Config(start_energy=0.005))
    w.step(thrust=1.0)
    assert not w.agent.alive
    y, x = w.agent.y, w.agent.x
    w.step(thrust=1.0)
    assert (w.agent.y, w.agent.x) == (y, x)


# --- render ----------------------------------------------------------------------

def test_headless_render():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    pygame.init()
    from cortex.app import App
    app = App(seed=1, view=21, cell=8)
    surf = app.draw()
    assert surf.get_size() == (app.w, app.h)
    arr = pygame.surfarray.array3d(surf)
    assert arr.std() > 0, "the render is a flat colour"
    pygame.quit()


def test_keyboard_maps_to_the_action_tuple():
    """The human and the brain must go through the same call."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    pygame.init()
    from cortex.app import App
    app = App(seed=1, view=21, cell=8)
    keys = {k: False for k in (pygame.K_w, pygame.K_UP, pygame.K_a, pygame.K_LEFT,
                               pygame.K_d, pygame.K_RIGHT, pygame.K_e)}
    keys[pygame.K_w] = True
    keys[pygame.K_d] = True
    assert app.action(keys) == (1.0, 1.0, 0.0)
    pygame.quit()


def test_facing_cell_follows_heading():
    w = World(seed=1)
    w.agent.y, w.agent.x, w.agent.heading = 50.0, 50.0, 0.0
    assert w.facing_cell() == (50, 51)
    w.agent.heading = math.pi / 2
    assert w.facing_cell() == (51, 50)
