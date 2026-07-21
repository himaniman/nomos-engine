"""Stage-1 ground truth for the cortex world: generation is deterministic and
seam-free, the body never overlaps a solid tile, and nothing tunnels at any speed."""

import math
import os

import numpy as np
import pytest

from cortex import tiles as T
from cortex.tiles import TileMap
from cortex.world import BERRY, FRUIT, MUSHROOM, Config, World


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


# --- food and the one verb -------------------------------------------------------

def _clear(w, side=64):
    for y in range(side):
        for x in range(side):
            w.tiles.set(y, x, T.GROUND)


def _press(w):
    """Wait out the interact cooldown, then press E once."""
    while w.tick < w._next_interact:
        w.step()
    w.step(interact=1.0)


def test_food_is_planted_deterministically_and_off_grid():
    a, b = World(seed=8), World(seed=8)
    fa = a.nearby_food(a.agent.y, a.agent.x, 60)
    fb = b.nearby_food(b.agent.y, b.agent.x, 60)
    assert fa and len(fa) == len(fb)
    assert [(s.y, s.x, s.kind) for s in fa] == [(s.y, s.x, s.kind) for s in fb]
    assert any(abs(s.y - round(s.y)) > 1e-6 for s in fa), "food should be off-grid"


def test_food_never_sits_inside_a_wall():
    w = World(seed=12)
    for s in w.nearby_food(w.agent.y, w.agent.x, 90):
        assert not w.tiles.blocked_circle(s.y, s.x, w.cfg.radius)


def test_walking_over_food_does_not_eat_it():
    """Eating is a DECISION: only INTERACT consumes food."""
    w = World(seed=3)
    s = w.nearby_food(w.agent.y, w.agent.x, 80)[0]
    w.agent.y, w.agent.x = s.y, s.x
    e0 = w.agent.energy
    for _ in range(10):
        w.step(thrust=1.0)
    assert s.ready, "food was eaten by walking over it"
    assert w.agent.energy < e0, "moving should only have cost energy"


def test_interact_eats_food_in_reach():
    w = World(seed=3)
    s = w.nearby_food(w.agent.y, w.agent.x, 80)[0]
    w.agent.y, w.agent.x = s.y, s.x
    e0 = w.agent.energy
    w.step(interact=1.0)
    assert not s.ready
    assert w.agent.energy > e0
    assert w.agent.ate[s.kind] == 1
    assert "ate" in w.last_event


def test_renewable_food_regrows_and_one_shot_does_not():
    w = World(seed=3)
    spot = w.nearby_food(w.agent.y, w.agent.x, 80)[0]
    spot.renewable = True
    w.agent.y, w.agent.x = spot.y, spot.x
    w.step(interact=1.0)
    assert not spot.ready
    for _ in range(w.cfg.regrow_ticks + 2):
        w.step()
    assert spot.ready, "a renewable spot must come back — this is what makes memory pay"

    w2 = World(seed=3)
    s2 = w2.nearby_food(w2.agent.y, w2.agent.x, 80)[0]
    s2.renewable = False
    w2.agent.y, w2.agent.x = s2.y, s2.x
    w2.step(interact=1.0)
    for _ in range(w2.cfg.regrow_ticks + 2):
        w2.step()
    assert not s2.ready


def test_interact_is_rate_limited():
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    w.tiles.set(20, 21, T.RUBBLE)
    for _ in range(3):                                # three ticks, one hit allowed
        w.step(interact=1.0)
    assert w.damage.get((20, 21), 0) == 1


def test_breaking_takes_several_hits_then_clears_the_tile():
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    w.tiles.set(20, 21, T.RUBBLE)
    need = int(T.HP[T.RUBBLE])
    for hit in range(1, need + 1):
        _press(w)
        if hit < need:
            assert w.tiles.material(20, 21) == T.RUBBLE, "cleared too early"
            assert f"{hit}/{need}" in w.last_event
    assert w.tiles.material(20, 21) == T.GROUND
    assert "broke" in w.last_event


def test_stone_is_not_breakable():
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    w.tiles.set(20, 21, T.STONE)
    for _ in range(200):
        w.step(interact=1.0)
    assert w.tiles.material(20, 21) == T.STONE


def test_door_opens_and_closes_and_never_traps_the_body():
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    w.tiles.set(20, 21, T.DOOR)
    assert w.tiles.blocked_circle(20.0, 21.0, w.cfg.radius)
    _press(w)
    assert w.tiles.material(20, 21) == T.DOOR_OPEN
    _press(w)
    assert w.tiles.material(20, 21) == T.DOOR                     # closes again
    _press(w)
    assert w.tiles.material(20, 21) == T.DOOR_OPEN                # and reopens
    # ...but it must never close onto our own body
    # close enough that the body overlaps cell 21, near enough that it still FACES it
    w.agent.y, w.agent.x = 20.0, 20.3
    _press(w)
    assert w.tiles.material(20, 21) == T.DOOR_OPEN
    assert "doorway" in w.last_event


def test_pushable_block_moves_one_cell_and_is_stopped_by_a_wall():
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    w.tiles.set(20, 21, T.PUSHABLE)
    w.step(interact=1.0)
    assert w.tiles.material(20, 21) == T.GROUND
    assert w.tiles.material(20, 22) == T.PUSHABLE
    w.tiles.set(20, 23, T.STONE)
    w.agent.x = 21.0
    for _ in range(w.cfg.interact_every + 1):
        w.step()
    w.step(interact=1.0)
    assert w.tiles.material(20, 22) == T.PUSHABLE, "should not push into stone"
    assert "budge" in w.last_event


def test_afforded_matches_what_interact_does():
    """The HUD prompt and the actual effect must never disagree."""
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    assert w.afforded() == ""
    w.tiles.set(20, 21, T.DOOR)
    assert w.afforded() == "open door"
    w.tiles.set(20, 21, T.PUSHABLE)
    assert w.afforded() == "push block"
    w.tiles.set(20, 21, T.RUBBLE)
    assert w.afforded().startswith("break")


def test_food_beats_the_tile_in_front():
    """Priority order: eat what is in reach BEFORE acting on the tile."""
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    w.tiles.set(20, 21, T.DOOR)
    w.food[(0, 0)] = [__import__("cortex.world", fromlist=["FoodSpot"])
                      .FoodSpot(y=20.2, x=20.2, kind=0)]
    assert w.afforded() == "eat berry"
    w.step(interact=1.0)
    assert w.tiles.material(20, 21) == T.DOOR                     # door untouched
    assert "ate" in w.last_event


def test_interaction_costs_energy_even_when_it_achieves_nothing():
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    w.agent.y, w.agent.x = 20.0, 20.0
    e0 = w.agent.energy
    w.step(interact=1.0)
    assert w.agent.energy < e0 - w.cfg.stand_cost
    assert w.last_event == "nothing to do here"


# --- habitats, damage and the sealed-pocket guard ----------------------------------

def test_spawn_can_actually_reach_the_world():
    """Two coffins in a row: first seed 1 spawned inside a hollow rock with no door,
    then inside a room whose door opened onto solid stone. Only walking THROUGH the
    door (and through what can be broken) proves an exit exists."""
    for seed in range(16):
        w = World(seed=seed)
        _, _, bounded = w.enclosure(int(w.agent.y), int(w.agent.x),
                                    limit=w.cfg.spawn_room, openable=True)
        assert not bounded, f"seed {seed} spawned in a coffin"


def test_a_door_into_solid_rock_is_not_an_exit():
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    for k in range(10, 21):                                   # a sealed 9x9 box...
        w.tiles.set(10, k, T.STONE)
        w.tiles.set(20, k, T.STONE)
        w.tiles.set(k, 10, T.STONE)
        w.tiles.set(k, 20, T.STONE)
    for y in range(64):                                       # ...buried in rock
        for x in range(21, 40):
            w.tiles.set(y, x, T.STONE)
    cells, has_door, bounded = w.enclosure(15, 15)
    assert bounded and not has_door and len(cells) == 81
    w.tiles.set(15, 20, T.DOOR)                               # a door onto solid stone
    _, has_door, bounded = w.enclosure(15, 15)
    assert has_door and bounded, "the room test should still call this a room"
    _, _, bounded = w.enclosure(15, 15, openable=True)
    assert bounded, "a door with rock behind it is not a way out"
    for x in range(21, 40):                                   # now bore a way out
        w.tiles.set(15, x, T.GROUND)
    _, _, bounded = w.enclosure(15, 15, openable=True)
    assert not bounded, "now it really is a way out"


def test_breakables_count_as_a_way_out():
    """The agent owns a pickaxe, so rubble is a door that takes longer."""
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    for k in range(10, 21):
        for cell in ((10, k), (20, k), (k, 10), (k, 20)):
            w.tiles.set(*cell, T.STONE)
    _, _, bounded = w.enclosure(15, 15, openable=True)
    assert bounded
    w.tiles.set(15, 20, T.RUBBLE)
    _, _, bounded = w.enclosure(15, 15, openable=True)
    assert not bounded


def test_a_one_cell_doorway_is_passable_from_anywhere_in_its_row():
    """A 0.45 body fits a 1.0 gap with +-0.05 to spare — passable with sub-cell
    precision, which is not a game. Aiming at the right ROW must be enough."""
    def run(off):
        w = World(seed=0, cfg=Config(size=64))
        _clear(w)
        for y in range(64):
            w.tiles.set(y, 30, T.STONE)
        w.tiles.set(20, 30, T.GROUND)                         # a one-cell gap
        w.agent.y, w.agent.x, w.agent.heading = 20.0 + off, 25.0, 0.0
        for _ in range(300):
            w.step(thrust=1.0)
            assert not w.tiles.blocked_circle(w.agent.y, w.agent.x, w.cfg.radius)
        return w.agent.x
    for off in (0.0, 0.15, 0.3, 0.45, -0.45):
        assert run(off) > 31.0, f"could not thread the doorway from offset {off:+.2f}"


def test_centring_never_opens_a_solid_wall():
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    for y in range(64):
        w.tiles.set(y, 30, T.STONE)                           # no gap at all
    w.agent.y, w.agent.x, w.agent.heading = 20.3, 25.0, 0.0
    for _ in range(300):
        w.step(thrust=1.0)
        assert not w.tiles.blocked_circle(w.agent.y, w.agent.x, w.cfg.radius)
    assert w.agent.x < 30.0, "walked through a solid wall"


def test_fruit_only_grows_by_a_trunk():
    w = World(seed=2)
    fruit = [s for s in w.nearby_food(256, 256, 150) if s.kind == FRUIT]
    assert fruit, "fruit should exist"
    for s in fruit:
        near = w.tiles.region(int(round(s.y)) - 2, int(round(s.x)) - 2, 5, 5)
        assert (near == T.TRUNK).any(), "fruit away from any tree"


def test_every_kind_has_a_habitat_and_food_is_not_lavish():
    w = World(seed=2)
    spots = w.nearby_food(256, 256, 150)
    kinds = {s.kind for s in spots}
    assert {BERRY, FRUIT, MUSHROOM} <= kinds
    per = math.pi * 150 * 150 / len(spots)
    assert 250 < per < 700, f"one spot per {per:.0f} cells^2 — retune the economy"


def test_felling_a_tree_takes_its_crown():
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    w.tiles.set(20, 21, T.TRUNK)
    for dy, dx in ((-1, 1), (0, 2), (1, 1)):                  # leaves around the trunk
        w.tiles.set(20 + dy, 21 + dx, T.FOLIAGE)
    for _ in range(int(T.HP[T.TRUNK])):
        _press(w)
    assert w.tiles.material(20, 21) == T.GROUND
    for dy, dx in ((-1, 1), (0, 2), (1, 1)):
        assert w.tiles.material(20 + dy, 21 + dx) == T.GROUND, "canopy left floating"
    assert "felled" in w.last_event


def test_damage_is_visible_as_a_fraction():
    w = World(seed=0, cfg=Config(size=64))
    _clear(w)
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    w.tiles.set(20, 21, T.RUBBLE)
    assert w.damage_frac(20, 21) == 0.0
    _press(w)
    assert 0.0 < w.damage_frac(20, 21) < 1.0
    _press(w)
    assert w.damage_frac(20, 21) > 1 / 3


def test_rooms_behind_doors_are_stocked():
    """A door that leads to an empty room teaches the agent doors are not worth
    opening — larders are what make the room worth entering."""
    w = World(seed=2)
    stocked = 0
    checked = 0
    for cy in range(6, 10):
        for cx in range(6, 10):
            for s in w._ensure_food(cy, cx):
                cells, has_door, bounded = w.enclosure(int(round(s.y)), int(round(s.x)),
                                                       limit=260)
                if bounded and has_door:
                    stocked += 1
                checked += 1
    assert checked and stocked, "no food ended up inside any room"


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


def _app(**kw):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pytest.importorskip("pygame")
    import pygame
    pygame.init()
    from cortex.app import App
    return pygame, App(seed=2, view=31, cell=10, **kw)


def test_blind_mode_hides_the_world():
    pygame, app = _app()
    lit = pygame.surfarray.array3d(app.draw())[:app.view * app.cell]
    app.blind = True
    dark = pygame.surfarray.array3d(app.draw())[:app.view * app.cell]
    assert dark.std() < lit.std() / 3, "blind mode still shows the map"
    pygame.quit()


def test_the_panel_cannot_cheat_it_draws_only_the_sensor_vector():
    """The whole point of the blind mode: the panel is a view of what the AGENT gets.
    Move the world without re-sensing and the panel must not notice; change the sensor
    vector without touching the world and it must."""
    pygame, app = _app()
    app.blind = True
    px = app.view * app.cell
    before = pygame.surfarray.array3d(app.draw())[px:]

    app.world.agent.y += 7.0                       # the world moved...
    app.world.agent.x += 5.0
    moved = pygame.surfarray.array3d(app.draw())[px:]
    assert np.array_equal(before, moved), "the panel read the world behind the senses"

    app.vec = app.vec * 0.0                        # ...now blind the senses instead
    blanked = pygame.surfarray.array3d(app.draw())[px:]
    assert not np.array_equal(before, blanked), "the panel ignored the sensor vector"
    pygame.quit()


def test_the_panel_stays_in_its_column():
    pygame, app = _app()
    app.blind = True
    px = app.view * app.cell
    a = pygame.surfarray.array3d(app.draw())[:px].copy()
    app.vec = np.abs(app.vec) * 0 + 0.9            # a very loud sensor vector
    b = pygame.surfarray.array3d(app.draw())[:px]
    assert np.array_equal(a, b), "the sense panel painted outside its column"
    pygame.quit()


def test_facing_cell_follows_heading():
    w = World(seed=1)
    w.agent.y, w.agent.x, w.agent.heading = 50.0, 50.0, 0.0
    assert w.facing_cell() == (50, 51)
    w.agent.heading = math.pi / 2
    assert w.facing_cell() == (51, 50)
