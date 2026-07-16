import numpy as np
import pytest

from nomos.dsl import NomosError
from nomos.embodied import body_sensor, food_sensor, play, read_move
from nomos.engine import World
from nomos.dsl import parse_program
from nomos.game import Snake

BRAIN = ("channel food\nchannel motor: init 0\n"
         "rule think:\n  when 1\n  set motor = 0.6 * avg8(motor) + food\n")

BRAIN_DANGER = ("channel food\nchannel danger\nchannel motor: init 0\n"
                "rule think:\n  when 1\n"
                "  set motor = 0.6*avg8(motor) + food - danger\n")


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


def test_food_sensor_amplitude_encodes_proximity():
    s = Snake(16, seed=1)
    hy, hx = s.head()
    s.food = (hy, (hx + 1) % 16)           # adjacent
    near = food_sensor((40, 40), s).max()
    s.food = (hy, (hx + 7) % 16)           # far
    far = food_sensor((40, 40), s).max()
    assert near > far                      # closer food -> brighter sense


def test_body_sensor_marks_body_direction_with_leverage():
    s = Snake(16, seed=1)                   # body extends left from the head
    d = body_sensor((40, 40), s)
    assert d.max() > 0                      # the body is felt
    # its mass sits left of centre (body is to the left of the head)...
    cx = (d.sum(axis=0) * np.arange(40)).sum() / d.sum()
    assert cx < 20
    # ...and with leverage the mark is pushed well OUT toward the edge, not stuck
    # near the centre where it would have no steering authority
    assert cx < 14                          # centre is 20; leverage moves it far left


def test_body_sensor_ignores_far_segments():
    # a long snake stretched away from the head: only near segments should mark
    s = Snake(20, seed=1)
    hy, hx = s.head()
    s.body = [(hy, hx)] + [(hy, (hx - k) % 20) for k in range(1, 12)]
    near = body_sensor((48, 48), s, radius_frac=0.2)   # ~4 cells of reach
    wide = body_sensor((48, 48), s, radius_frac=0.9)   # the whole body
    # a tighter radius silences the far tail, so fewer cells light up
    assert (near > 0.05).sum() < (wide > 0.05).sum()
    assert near.max() > 0.5                            # the neck is loud


def test_play_writes_danger_only_when_declared():
    # a danger-aware brain runs and stays deterministic
    a = play(BRAIN_DANGER, world_size=32, game_size=12, seed=2, moves=30, ticks_per_move=3)
    b = play(BRAIN_DANGER, world_size=32, game_size=12, seed=2, moves=30, ticks_per_move=3)
    assert a["move_seq"] == b["move_seq"]


def test_temperature_sensor_brain_runs_and_is_deterministic():
    # a brain that reads the global hunger field steers harder when hungry
    brain = ("channel food\nchannel temperature\nchannel motor: init 0\n"
             "rule think:\n  when 1\n"
             "  set motor = 0.5*avg8(motor) + (1 + temperature) * food\n")
    a = play(brain, world_size=32, game_size=12, seed=1, moves=40, ticks_per_move=4)
    b = play(brain, world_size=32, game_size=12, seed=1, moves=40, ticks_per_move=4)
    assert a["move_seq"] == b["move_seq"]
    assert a["score"] == b["score"]


def test_starvation_kills_a_do_nothing_brain():
    # a brain whose motor never moves the snake toward food will starve, not live
    # forever — hunger closes the immortal loophole
    idle = "channel food\nchannel motor: init 0\nrule t: when 1 set motor = 0\n"
    r = play(idle, world_size=32, game_size=12, seed=0, moves=1000, ticks_per_move=4,
             starve_every=20)
    assert not r["alive_at_end"]        # it dies (starves) rather than looping forever
    assert r["steps_alive"] < 1000


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


def test_avoid_layer_overrides_motor_at_the_decision():
    brain = ("channel food\nchannel motor: init 0\nchannel avoid: init 0\n"
             "rule t: when 1 set motor = 0\n")
    prog = parse_program(brain)
    world = World(prog, 20, seed=0)
    # motor pulls up, but a stronger avoid pull up should veto 'up'
    world.state["motor"][:] = 0.0
    world.state["motor"][2, 10] = 1.0
    world.state["avoid"][:] = 0.0
    world.state["avoid"][0, 10] = 5.0      # strong 'stay away from up'
    assert read_move(world, avoid="avoid") != "up"
    # with no avoid mass, the motor pull wins as usual
    world.state["avoid"][:] = 0.0
    assert read_move(world, avoid="avoid") == "up"


def test_two_layer_brain_plays_deterministically():
    from pathlib import Path
    law = Path("laws/snake_brain_layers.nomos").read_text(encoding="utf-8")
    a = play(law, world_size=32, game_size=12, seed=1, moves=40, ticks_per_move=4)
    b = play(law, world_size=32, game_size=12, seed=1, moves=40, ticks_per_move=4)
    assert a["move_seq"] == b["move_seq"]
    r = play(law, world_size=32, game_size=12, seed=1, moves=8, ticks_per_move=4,
             collect_frames=True)
    assert len(r["avoid_frames"]) == len(r["life_frames"]) >= 1   # avoid layer captured


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
