"""The door experiment: the smell seeps through a door from beside it as from afar,
the nose points AT the door when the walk goes through it, the wave can lead into a
solid, and the leak of a hue is taught by what pressing on it did."""

import math

import numpy as np

from cortex import atlas as A
from cortex import probes
from cortex.senses import HSV, ODOUR_CH, ScentField, layout, sense
from cortex.tiles import DOOR, STONE


def _amps(w, x):
    w.agent.x = float(x)
    field = ScentField(w.cfg.smell_radius)
    lay = layout(w.cfg.n_rays)
    lo, hi = lay["slices"]["odour"]
    return sense(w, field)[lo:hi:ODOUR_CH]


def test_smell_seeps_through_a_door_from_beside_it():
    w, _ = probes.build("door")
    far, near = _amps(w, 6).max(), _amps(w, 15).max()
    assert far > 0.03
    assert near > far                       # it used to drop to exactly zero here


def test_waypoint_is_the_first_cell_out_at_least_ahead():
    w, _ = probes.build("door")
    w.agent.y, w.agent.x = 16.0, 15.0        # beside the door at (16, 16)
    field = ScentField(w.cfg.smell_radius)
    field.update(w)
    wy, wx = field.waypoint((16, 22), 6.0)
    assert (wy, wx) == (16, 16)              # the door itself, not the cell before it


def test_the_wave_can_lead_into_a_solid():
    """A source behind a wall, nothing else: the solid neighbour is the target and the
    command presses (a second leak discount in the read-out used to make this branch
    unreachable)."""
    # leak 0.3: at 0.12 the horizon ring round the map (a walk of ~2R+2 cells,
    # 0.94^26 = 0.2) beats leaking through any wall on a map this size
    at = A.Atlas(A.Config(radius=12, leak=0.3))
    R = at.R
    at.state[A.SOLID, :, R + 1] = 1.0        # a wall one column to the right
    at.state[A.FREE, :, :R + 1] = 1.0
    src = np.zeros((at.side, at.side))
    src[R, R + 3] = 1.0
    at._wave(src, 40)
    thrust, turn, act = at._read(0.0, np.zeros(at.layout["size"]))
    assert at.last_target == (0, 1) and act == 1.0 and at.last_press is not None


def test_leak_is_innate_then_taught_per_hue():
    at = A.Atlas(A.Config(leak=0.12, leak_lr=1.0))
    assert abs(at.leak_map().mean() - 0.12) < 1e-6
    door = at._feat(math.cos(HSV[DOOR][0] * 2 * math.pi), math.sin(HSV[DOOR][0] * 2 * math.pi))
    stone = at._feat(math.cos(HSV[STONE][0] * 2 * math.pi), math.sin(HSV[STONE][0] * 2 * math.pi))
    for _ in range(3):
        at.learn_leak(door, +1.0)
        at.learn_leak(stone, -1.0)
    sig = lambda f: 1.0 / (1.0 + math.exp(-(f @ at.leak_w)))
    assert sig(door) > 0.8 and sig(stone) < 0.05
    assert at.presses == 6
    # an unseen cell (no hue) still gets the innate guess: the bias is never taught
    assert abs(sig(at._feat(0.0, 0.0)) - 0.12) < 0.01


def test_door_probe_is_solved_with_a_curious_leak():
    r = probes.probe_door(A.Driver(A.Config(linger=0.99, smell_reach=7, leak=0.3)), 900)
    assert r.detail["opened"] == 1 and r.detail["ate"] == 1
