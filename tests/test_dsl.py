import numpy as np
import pytest

from nomos.dsl import NomosError, parse_expression, parse_program
from nomos.engine import EvalCtx, World, evaluate

GOL = """
channel alive: init bernoulli(0.2)

rule birth:
    when alive == 0 and sum8(alive) == 3
    set alive = 1

rule death:
    when alive == 1 and (sum8(alive) < 2 or sum8(alive) > 3)
    set alive = 0
"""


def _eval(text, state=None, shape=(1, 1), seed=0):
    ctx = EvalCtx(shape, np.random.default_rng(seed))
    return np.asarray(evaluate(parse_expression(text), state or {}, ctx))


def test_parse_gol_structure():
    prog = parse_program(GOL)
    assert prog.channel_names() == ["alive"]
    assert [r.name for r in prog.rules] == ["birth", "death"]
    assert prog.rules[0].actions == [("set", "alive", ("num", 1.0))]


def test_arithmetic_precedence():
    assert _eval("2 + 3 * 4") == 14
    assert _eval("(2 + 3) * 4") == 20
    assert _eval("2 + 3 * 4 == 14") == 1
    assert _eval("-2 * 3") == -6
    assert _eval("10 % 4 + 1") == 3


def test_boolean_logic():
    assert _eval("1 and 0") == 0
    assert _eval("1 or 0") == 1
    assert _eval("not 0") == 1
    assert _eval("not 1 == 2") == 1          # not (1 == 2)
    assert _eval("0 and 0 or 1") == 1        # (0 and 0) or 1
    assert _eval("1 == 1 and 2 > 1") == 1


def test_functions():
    assert _eval("clamp(5, 0, 2)") == 2
    assert _eval("max(1, 7) + min(1, 7)") == 8
    assert _eval("where(0, 10, 20)") == 20
    assert _eval("abs(-3)") == 3


def test_division_by_zero_is_zero():
    assert _eval("1 / 0") == 0


def test_neighborhood_sum():
    state = {"x": np.zeros((5, 5))}
    state["x"][2, 2] = 1.0
    ctx = EvalCtx((5, 5), np.random.default_rng(0))
    s = evaluate(parse_expression("sum8(x)"), state, ctx)
    assert s[2, 2] == 0            # center excluded
    assert s[1, 1] == 1 and s[3, 3] == 1
    assert s.sum() == 8


def test_argmax8_picks_value_at_max_key():
    key = np.zeros((5, 5)); val = np.zeros((5, 5))
    key[1, 2] = 9.0; val[1, 2] = 7.0        # the north neighbour of (2,2)
    key[3, 2] = 4.0; val[3, 2] = 3.0        # a weaker south neighbour
    ctx = EvalCtx((5, 5), np.random.default_rng(0))
    from nomos.dsl import parse_expression as pe
    from nomos.engine import evaluate as ev
    out = ev(pe("argmax8(key, val)"), {"key": key, "val": val}, ctx)
    assert out[2, 2] == 7.0                 # inherits value of the strongest neighbour


def test_argmax8_zero_when_no_positive_key():
    ctx = EvalCtx((3, 3), np.random.default_rng(0))
    from nomos.dsl import parse_expression as pe
    from nomos.engine import evaluate as ev
    state = {"k": np.zeros((3, 3)), "v": np.ones((3, 3))}
    out = ev(pe("argmax8(k, v)"), state, ctx)
    assert np.all(out == 0.0)               # no eligible parent -> 0


def test_time_and_trig_drive_changing_field():
    prog = parse_program(
        "channel f\n"
        "rule wave: when 1 set f = sin(time())\n"
    )
    world = World(prog, size=4, seed=0)
    world.step()                               # ctx.time == 0 -> sin(0) == 0
    assert np.allclose(world.state["f"], 0.0)
    world.step()                               # ctx.time == 1 -> sin(1)
    assert np.allclose(world.state["f"], np.sin(1.0))


def test_pi_builtin():
    assert abs(float(_eval("pi()")) - np.pi) < 1e-9


def test_pos_ramps():
    ctx = EvalCtx((4, 8), np.random.default_rng(0))
    from nomos.dsl import parse_expression as pe
    from nomos.engine import evaluate as ev
    x = ev(pe("posx()"), {}, ctx)
    y = ev(pe("posy()"), {}, ctx)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert np.allclose(x[0], np.arange(8) / 8)      # varies along columns
    assert np.allclose(y[:, 0], np.arange(4) / 4)   # varies along rows


def test_unknown_channel_rejected_at_load():
    with pytest.raises(NomosError, match="unknown channel"):
        World(parse_program("channel a\nrule r: when ghost > 0 set a = 1"), size=4)


def test_unknown_function_rejected_at_load():
    with pytest.raises(NomosError, match="unknown function"):
        World(parse_program("channel a\nrule r: when 1 set a = warp(a)"), size=4)


def test_rule_without_actions_rejected():
    with pytest.raises(NomosError, match="no set/add"):
        parse_program("channel a\nrule r: when 1")
