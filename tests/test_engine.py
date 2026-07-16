import numpy as np

from nomos.dsl import parse_program
from nomos.engine import World

GOL = open("laws/game_of_life.nomos", encoding="utf-8").read()


def make_gol(size=16):
    world = World(parse_program(GOL), size=size, seed=0)
    world.state["alive"][:] = 0.0
    return world


def test_blinker_oscillates():
    world = make_gol()
    world.state["alive"][5, 4:7] = 1.0          # horizontal blinker
    world.step()
    vertical = np.zeros((16, 16))
    vertical[4:7, 5] = 1.0
    assert np.array_equal(world.state["alive"], vertical)
    world.step()
    horizontal = np.zeros((16, 16))
    horizontal[5, 4:7] = 1.0
    assert np.array_equal(world.state["alive"], horizontal)


def test_glider_glides():
    world = make_gol()
    glider = [(3, 4), (4, 5), (5, 3), (5, 4), (5, 5)]
    for y, x in glider:
        world.state["alive"][y, x] = 1.0
    before = world.state["alive"].copy()
    for _ in range(4):
        world.step()
    # after 4 generations a glider is the same shape shifted by (+1, +1)
    assert np.array_equal(world.state["alive"], np.roll(before, (1, 1), axis=(0, 1)))


def test_toroidal_wrap():
    world = make_gol(8)
    world.state["alive"][0, 3:6] = 1.0          # blinker on the top edge
    world.step()
    assert world.state["alive"][7, 4] == 1.0    # wraps to the bottom row


def test_reproducibility_same_seed():
    prog = parse_program(open("laws/soup.nomos", encoding="utf-8").read())
    a, b = World(prog, size=32, seed=42), World(prog, size=32, seed=42)
    a.run(50)
    b.run(50)
    for ch in a.state:
        assert np.array_equal(a.state[ch], b.state[ch])


def test_different_seed_differs():
    prog = parse_program(GOL)
    a, b = World(prog, size=32, seed=1), World(prog, size=32, seed=2)
    assert not np.array_equal(a.state["alive"], b.state["alive"])


def test_add_rules_compose():
    prog = parse_program(
        "channel x\n"
        "rule a: when 1 add x = 1\n"
        "rule b: when 1 add x = 2\n"
    )
    world = World(prog, size=4, seed=0)
    world.step()
    assert np.all(world.state["x"] == 3.0)


def test_set_reads_snapshot_last_rule_wins():
    prog = parse_program(
        "channel x: init 5\n"
        "rule a: when 1 set x = x + 1\n"
        "rule b: when x == 5 set x = 100\n"   # sees snapshot x==5, overwrites rule a
    )
    world = World(prog, size=4, seed=0)
    world.step()
    assert np.all(world.state["x"] == 100.0)


def test_nan_and_inf_are_regularized():
    prog = parse_program(
        "channel x: init 1\n"
        "rule blowup: when 1 set x = x * 1e300\n"
    )
    world = World(prog, size=4, seed=0)
    world.run(3)
    assert np.all(np.isfinite(world.state["x"]))
    assert np.all(np.abs(world.state["x"]) <= 1e6)


def test_activity_measures_change():
    world = make_gol()
    assert world.step() == 0.0                  # empty world: nothing changes
    world.state["alive"][5, 4:7] = 1.0
    assert world.step() > 0.0                   # blinker flips cells
