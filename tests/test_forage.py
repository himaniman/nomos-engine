import numpy as np
import pytest

from nomos.dsl import NomosError
from nomos.embodied import forage_food_sensor, play_forage, wall_sensor
from nomos.forage import Forage

BRAIN = ("channel food\nchannel motor: init 0\n"
         "rule think:\n  when 1\n  set motor = 0.6 * avg8(motor) + food\n")

BRAIN_WALLS = ("channel food\nchannel wall\nchannel motor: init 0\n"
               "channel avoid: init 0\n"
               "rule go:\n  when 1\n  set motor = 0.4*avg8(motor) + food\n"
               "rule fear:\n  when 1\n  set avoid = 0.4*avg8(avoid) + 0.5*wall\n")


def test_map_is_bounded_and_centre_is_clear():
    g = Forage(24, seed=0)
    assert g.walls[0, :].all() and g.walls[-1, :].all()
    assert g.walls[:, 0].all() and g.walls[:, -1].all()
    py, px = g.pos
    assert not g.walls[py - 1:py + 2, px - 1:px + 2].any()   # 3x3 spawn clearing


def test_map_and_food_are_deterministic():
    a, b = Forage(24, seed=5), Forage(24, seed=5)
    assert (a.walls == b.walls).all()
    assert a.foods == b.foods
    c = Forage(24, seed=6)
    assert (a.walls != c.walls).any()          # a different seed is a different map


def test_all_food_is_reachable():
    for seed in range(6):
        g = Forage(24, seed=seed, food_count=8)
        assert g.foods <= g.reachable          # food never spawns in sealed pockets
        assert len(g.foods) == 8


def test_walls_block_movement():
    g = Forage(12, seed=0)
    g.walls[:] = False
    g.walls[0, :] = g.walls[-1, :] = g.walls[:, 0] = g.walls[:, -1] = True
    g.pos = (1, 1)
    g.foods = set()
    g.step("up")                               # border wall above
    assert g.pos == (1, 1)                     # blocked: the player stays
    g.step("right")
    assert g.pos == (1, 2)                     # open cell: the player moves


def test_eating_scores_resets_hunger_and_respawns():
    g = Forage(16, seed=1, food_count=3)
    g.hunger = 10
    ny, nx = g.pos[0], g.pos[1] + 1
    g.walls[ny, nx] = False
    g.foods = {(ny, nx)}
    g.step("right")
    assert g.score == 1 and g.hunger == 0
    assert len(g.foods) == 1 and (ny, nx) not in g.foods    # respawned elsewhere


def test_starvation_kills():
    g = Forage(16, seed=2, starve_after=5)
    g.foods = set()                            # nothing to eat anywhere
    for _ in range(5):
        g.step("up")                           # moving costs a full hunger point
    assert not g.alive and g.starved
    assert g.field()[g.pos] == 1.0             # death is flashed for the render


def test_standing_still_is_cheaper_but_not_free():
    g = Forage(16, seed=2, starve_after=5)
    g.foods = set()
    for _ in range(5):
        g.step(None)                           # idle costs 0.5 -> hunger 2.5
    assert g.alive
    for _ in range(5):
        g.step(None)                           # hunger reaches 5.0
    assert not g.alive and g.starved           # thinking is cheap, never free


def test_game_is_deterministic_under_a_move_sequence():
    moves = ["up", "left", None, "down", "right", "right", "up"] * 4
    a, b = Forage(20, seed=3), Forage(20, seed=3)
    for m in moves:
        a.step(m)
        b.step(m)
    assert a.pos == b.pos and a.foods == b.foods and a.score == b.score


def _open_arena(size: int, seed: int = 0) -> Forage:
    """A Forage world with all interior walls removed — sensor geometry tests
    need controlled sight lines, not a seeded maze."""
    g = Forage(size, seed=seed)
    g.walls[1:-1, 1:-1] = False
    g._scent_cache.clear()
    return g


def test_food_sensor_attenuates_with_distance():
    g = _open_arena(24)
    py, px = g.pos
    g.foods = {(py, px + 2)}                   # near food
    near = forage_food_sensor((48, 48), g).max()
    g.foods = {(py, px + 10)}                  # far food
    far = forage_food_sensor((48, 48), g).max()
    assert near > far > 0                      # scent fades with walking distance


def test_food_sensor_points_toward_food():
    g = _open_arena(24)
    py, px = g.pos
    g.foods = {(py - 6, px)}                   # food straight up
    f = forage_food_sensor((48, 48), g)
    assert np.argmax(f.sum(axis=1)) < 24       # bump in the upper half


def test_food_scent_goes_around_walls_and_not_through():
    g = _open_arena(24)
    py, px = g.pos
    g.walls[py - 3:py + 4, px + 2] = True      # a wall slab between player and food
    g._scent_cache.clear()
    g.foods = {(py, px + 5)}                   # food straight right, behind the slab
    f = forage_food_sensor((48, 48), g)
    assert f.max() > 0                         # smell still arrives (around the slab)
    # the scent is attenuated by the WALKING distance (13 around the slab), so it
    # is fainter than the same food with no wall in the way
    g2 = _open_arena(24)
    g2.foods = set(g.foods)
    assert f.max() < forage_food_sensor((48, 48), g2).max()
    # a fully sealed pocket carries no scent at all
    g.walls[py - 3:py + 4, px + 2] = True
    g.walls[py - 3, px + 2:px + 8] = True
    g.walls[py + 3, px + 2:px + 8] = True
    g.walls[py - 3:py + 4, px + 7] = True
    g._scent_cache.clear()
    assert forage_food_sensor((48, 48), g).max() == 0.0


def test_wall_sensor_sees_near_walls_with_leverage():
    g = Forage(24, seed=0)
    g.walls[:] = False
    py, px = g.pos
    g.walls[py, px + 1] = True                 # wall right next to the player
    w = wall_sensor((48, 48), g)
    assert w.max() > 0.5                       # adjacent wall is loud
    cx = (w.sum(axis=0) * np.arange(48)).sum() / w.sum()
    assert cx > 27                             # magnified well right of centre
    g.walls[py, px + 1] = False
    g.walls[py, px + 11] = True                # beyond the vision radius
    assert wall_sensor((48, 48), g).max() == 0.0


def test_play_forage_is_deterministic_and_reports():
    a = play_forage(BRAIN_WALLS, world_size=32, game_size=16, seed=1, moves=30,
                    ticks_per_move=3)
    b = play_forage(BRAIN_WALLS, world_size=32, game_size=16, seed=1, moves=30,
                    ticks_per_move=3)
    assert a["move_seq"] == b["move_seq"]
    assert a["score"] == b["score"] and a["steps_alive"] == b["steps_alive"]


def test_play_forage_collects_paired_frames():
    r = play_forage(BRAIN_WALLS, world_size=24, game_size=12, seed=1, moves=8,
                    ticks_per_move=2, collect_frames=True, view_radius=8)
    assert len(r["life_frames"]) == len(r["game_frames"]) >= 1
    assert len(r["avoid_frames"]) == len(r["life_frames"])
    assert len(r["temp_frames"]) == len(r["life_frames"])   # hunger per frame
    assert r["life_frames"][0].shape == (24, 24)
    assert r["game_frames"][0].shape == (17, 17)   # egocentric camera crop


def test_view_keeps_player_centred_and_pads_with_wall():
    g = Forage(24, seed=0)
    v = g.view(radius=10)
    assert v.shape == (21, 21)
    assert v[10, 10] in (0.85, 1.0)                # the player sits at the centre
    g.pos = (1, 1)                                 # camera pressed into a corner
    v = g.view(radius=10)
    assert v[10, 10] in (0.85, 1.0)
    assert v[0, 0] == 0.3                          # beyond the map reads as wall


def test_view_fade_dims_the_periphery_not_the_player():
    g = Forage(24, seed=0)
    plain = g.view(radius=10)
    faded = g.view(radius=10, fade=0.7)
    assert faded[10, 10] == plain[10, 10]          # the player never fades
    corner = plain[1, 1]
    assert faded[1, 1] < corner or corner == 0.0   # periphery dims under vignette


def test_dense_brain_frames_are_aggregated():
    r = play_forage(BRAIN_WALLS, world_size=64, game_size=12, seed=1, moves=2,
                    ticks_per_move=1, collect_frames=True, render_cap=32)
    assert r["life_frames"][0].shape == (32, 32)   # block-averaged, not raw 64^2


def test_play_forage_requires_food_and_motor():
    with pytest.raises(NomosError):
        play_forage("channel a\nrule r: when 1 set a = 1\n", moves=1)


def test_idle_brain_starves():
    idle = "channel food\nchannel motor: init 0\nrule t: when 1 set motor = 0\n"
    r = play_forage(idle, world_size=24, game_size=12, seed=0, moves=200,
                    ticks_per_move=2, starve_after=15)
    assert not r["alive_at_end"]
    assert r["steps_alive"] < 200
