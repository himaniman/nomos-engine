import numpy as np
import pytest

from nomos.dsl import NomosError
from nomos.embodied import food_sensor, play, read_move
from nomos.engine import World
from nomos.dsl import parse_program
from nomos.game import Snake

BRAIN = ("channel food\nchannel motor: init 0\n"
         "rule think:\n  when 1\n  set motor = 0.6 * avg8(motor) + food\n")


def test_food_sensor_points_toward_food():
    s = Snake(16, seed=1)
    hy, hx = s.head()
    s.food = ((hy - 4) % 16, hx)           # food straight up
    f = food_sensor((40, 40), s)
    cy = np.argmax(f.sum(axis=1))          # brightest row
    assert cy < 20                         # bump sits in the upper half

    s.food = (hy, (hx + 5) % 16)           # food to the right
    f = food_sensor((40, 40), s)
    cx = np.argmax(f.sum(axis=0))
    assert cx > 20                         # bump sits in the right half


def test_read_move_follows_motor_centre_of_mass():
    prog = parse_program(BRAIN)
    world = World(prog, 20, seed=0)
    world.state["motor"][:] = 0.0
    world.state["motor"][2, 10] = 1.0      # mass near the top -> up
    assert read_move(world) == "up"
    world.state["motor"][:] = 0.0
    world.state["motor"][10, 18] = 1.0     # mass on the right -> right
    assert read_move(world) == "right"
    world.state["motor"][:] = 0.0
    assert read_move(world) is None        # silent motor -> keep heading


def test_play_requires_food_and_motor_channels():
    with pytest.raises(NomosError):
        play("channel a\nrule r: when 1 set a = 1\n", moves=1)


def test_play_is_deterministic_and_reports_score():
    a = play(BRAIN, world_size=32, game_size=12, seed=3, moves=40, ticks_per_move=3)
    b = play(BRAIN, world_size=32, game_size=12, seed=3, moves=40, ticks_per_move=3)
    assert a["move_seq"] == b["move_seq"]
    assert a["score"] == b["score"]
    assert a["steps_alive"] == b["steps_alive"]


def test_play_collects_paired_frames():
    r = play(BRAIN, world_size=24, game_size=10, seed=1, moves=8, ticks_per_move=2,
             collect_frames=True)
    assert len(r["life_frames"]) == len(r["game_frames"]) >= 1
    assert r["life_frames"][0].shape == (24, 24)
    assert r["game_frames"][0].shape == (10, 10)
