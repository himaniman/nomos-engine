"""The atlas's rule as a nomos law: it must BE the wave, and the harness must keep a
law's own channels between thoughts."""

import numpy as np
import pytest

from cortex import atlas as A
from cortex import atlas_law as LAW
from nomos.dsl import NomosError


def _random_state(rng, S):
    solid = (rng.random((S, S)) < 0.25) * rng.uniform(0.5, 1.0, (S, S))
    src = (rng.random((S, S)) < 0.05) * rng.uniform(0.2, 1.5, (S, S))
    v0 = rng.random((S, S)) * (rng.random((S, S)) < 0.5)
    return solid, src, v0


def test_reference_law_is_the_wave():
    rule = LAW.load()
    assert rule.extra == []
    at = A.Atlas(A.Config(linger=0.0))
    rng = np.random.default_rng(0)
    S = at.side
    for _ in range(10):
        solid, src, v0 = _random_state(rng, S)
        at.state[A.SOLID] = solid
        at.state[A.VALUE] = v0.copy()
        at._wave(src, 6)
        st = np.zeros((A.N_CH, S, S))
        st[A.SOLID] = solid
        st[A.VALUE] = v0.copy()
        rule.step(st, src, at._passable(), 6)
        assert np.abs(st[A.VALUE] - at.state[A.VALUE]).max() < 1e-5


def test_no_wrap_across_the_map():
    """A source at the left edge must not light the right edge through the torus."""
    rule = LAW.load()
    S = 7
    st = np.zeros((A.N_CH, S, S))
    src = np.zeros((S, S))
    src[3, 0] = 1.0
    # five passes: the wave lights the source and four cells to its right; round
    # the torus (source -> horizon -> edge -> far edge -> far horizon -> last
    # column) it would reach the last column on the fifth
    rule.step(st, src, np.ones((S, S)), 5)
    v = st[A.VALUE]
    assert v[3, 0] == 1.0 and v[3, 4] > 0.7
    assert v[3, S - 1] == 0.0


def test_hidden_channels_persist_between_thoughts():
    law = """
channel source
channel value
channel count: init 0

rule tick:
  when 1
  set count = count + 1
  set value = max(source, 0.9 * max8(value))
"""
    rule = LAW.LawRule(law)
    assert rule.extra == ["count"] and rule.hidden == 1
    S = 7
    st = np.zeros((A.N_CH + 1, S, S))
    src = np.zeros((S, S))
    rule.step(st, src, np.ones((S, S)), 4)
    assert np.all(st[A.N_CH] == 4.0)
    rule.step(st, src, np.ones((S, S)), 4)
    assert np.all(st[A.N_CH] == 8.0)


def test_law_must_write_value():
    with pytest.raises(NomosError):
        LAW.LawRule("channel source\nchannel x\nrule r:\n  when 1\n  set x = source\n")


def test_check_catches_runtime_trouble():
    LAW.check(open(LAW.REFERENCE, encoding="utf-8").read())
    with pytest.raises(NomosError):
        LAW.check("channel value\nrule r:\n  when 1\n  set value = nosuch(value)\n")


def test_jitter_keeps_a_law_loadable_and_moves_only_numbers():
    text = open(LAW.REFERENCE, encoding="utf-8").read()
    rng = np.random.default_rng(1)
    for _ in range(5):
        child = LAW.jitter(text, rng, prob=1.0)
        LAW.check(child)
    assert child != text
    # the structure — channels and rule names — is untouched
    lines_a = [l for l in text.splitlines() if l.startswith(("channel", "rule"))]
    lines_b = [l for l in child.splitlines() if l.startswith(("channel", "rule"))]
    assert lines_a == lines_b


def test_a_law_drives_a_life_and_pickles():
    import pickle
    rule = pickle.loads(pickle.dumps(LAW.load()))
    w, mind = A.live(0, 300, rule=rule)
    assert w.tick == 300
    assert mind.atlas.state.shape[0] == A.N_CH


def test_inputs_need_no_declaration():
    rule = LAW.LawRule("channel value\nrule r:\n  when 1\n  set value = source * (1 - solid)\n")
    assert rule.extra == []
    S = 5
    st = np.zeros((A.N_CH, S, S))
    src = np.zeros((S, S))
    src[2, 2] = 1.0
    rule.step(st, src, np.ones((S, S)), 1)
    assert st[A.VALUE][2, 2] == 1.0


def test_here_marks_the_centre():
    rule = LAW.LawRule("channel value\nrule r:\n  when 1\n  set value = here\n")
    S = 7
    st = np.zeros((A.N_CH, S, S))
    rule.step(st, np.zeros((S, S)), np.ones((S, S)), 1)
    assert st[A.VALUE].sum() == 1.0 and st[A.VALUE][3, 3] == 1.0


def test_hand_linger_is_the_ghost_law():
    """What the law found went back into the engine: Config.linger IS
    laws/atlas_ghost.nomos, pass for pass."""
    rule = LAW.load("laws/atlas_ghost.nomos")
    assert rule.extra == ["ghost"]
    at = A.Atlas(A.Config(linger=0.99, leak=0.12, leak_lr=0.0))
    rng = np.random.default_rng(3)
    S = at.side
    for _ in range(6):
        solid, src, _ = _random_state(rng, S)
        at.state[A.SOLID] = solid
        st = np.zeros((A.N_CH + 1, S, S))
        st[A.SOLID] = solid
        st[A.VALUE] = at.state[A.VALUE]
        st[A.N_CH] = at.state[A.GHOST]
        at._wave(src, 6)
        rule.step(st, src, at._passable(), 6)
        assert np.abs(st[A.VALUE] - at.state[A.VALUE]).max() < 1e-5
        assert np.abs(st[A.N_CH] - at.state[A.GHOST]).max() < 1e-5
    # and a live ghost is a live ghost: after the sources go, it fades, not stops
    assert at.state[A.GHOST].max() > 0


def test_law_sees_the_learned_leak():
    """Under a per-hue leak the reference law is still the hand wave: the harness
    hands it the leak map the body taught."""
    import math
    from cortex.senses import HSV
    from cortex.tiles import DOOR, STONE
    rule = LAW.load()
    at = A.Atlas(A.Config(linger=0.0, leak=0.5, leak_lr=1.0))
    door = at._feat(math.cos(HSV[DOOR][0] * 2 * math.pi), math.sin(HSV[DOOR][0] * 2 * math.pi))
    stone = at._feat(math.cos(HSV[STONE][0] * 2 * math.pi), math.sin(HSV[STONE][0] * 2 * math.pi))
    at.learn_leak(door, 1.0)
    at.learn_leak(stone, -1.0)
    rng = np.random.default_rng(5)
    S = at.side
    solid, src, _ = _random_state(rng, S)
    at.state[A.SOLID] = solid
    hues = rng.choice([HSV[DOOR][0], HSV[STONE][0]], size=(S, S)) * 2 * math.pi
    at.state[A.HUE_C], at.state[A.HUE_S] = np.cos(hues), np.sin(hues)
    lk = at.leak_map()
    assert lk.min() < 0.3 < 0.6 < lk.max()
    st = at.state.copy()
    at._wave(src, 6)
    rule.step(st, src, at._passable(), 6)
    assert np.abs(st[A.VALUE] - at.state[A.VALUE]).max() < 1e-5
