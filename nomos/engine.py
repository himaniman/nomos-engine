"""The world: a toroidal grid of float channels evolving under DSL rules.

Semantics (the physics contract):
- All expressions in a step read the SNAPSHOT of the world at the start of
  the step (synchronous update, like classic cellular automata).
- Rules apply in declaration order. `set` overwrites where the condition
  holds (last matching rule wins per cell); `add` accumulates, so several
  `add` rules on one channel compose within a step.
- After each step every channel is regularized: NaN -> 0, +/-inf clipped
  to +/-1e6, values clipped to [-1e6, 1e6]. A universe may die; it may not
  crash.
- Given the same law file, size and seed, a run is bit-for-bit reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nomos.dsl import NomosError, Program, walk_calls, walk_names

_CLIP = 1e6


@dataclass
class EvalCtx:
    shape: tuple[int, int]
    rng: np.random.Generator


def _grid(x, ctx: EvalCtx) -> np.ndarray:
    """Broadcast a scalar or array to the world shape."""
    return np.broadcast_to(np.asarray(x, dtype=np.float64), ctx.shape)


def _truthy(x) -> np.ndarray:
    return np.asarray(x) != 0


# ---- neighborhood offsets (dy, dx) ----

_MOORE = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
_VON_NEUMANN = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors(a: np.ndarray, offsets) -> list[np.ndarray]:
    return [np.roll(a, (dy, dx), axis=(0, 1)) for dy, dx in offsets]


def _make_neigh(reducer, offsets):
    def fn(ctx: EvalCtx, x):
        stack = _neighbors(_grid(x, ctx), offsets)
        return reducer(stack)
    return fn


def _make_argmax(offsets):
    """Inheritance primitive: value of the neighbor whose key is largest.

    Eligibility requires key > 0, so `argmax8(rand()*alive, gene)` copies the
    gene of a uniformly-random *living* neighbor, and returns 0 when there is
    none. `argmax8(fitness*alive, gene)` copies the fittest neighbor instead.
    """
    def fn(ctx: EvalCtx, key, value):
        keys = np.stack(_neighbors(_grid(key, ctx), offsets), axis=0)
        vals = np.stack(_neighbors(_grid(value, ctx), offsets), axis=0)
        idx = np.argmax(keys, axis=0)[None]
        picked = np.take_along_axis(vals, idx, axis=0)[0]
        best = np.take_along_axis(keys, idx, axis=0)[0]
        return np.where(best > 0, picked, 0.0)
    return fn


def _pos_ramp(axis):
    def fn(ctx: EvalCtx):
        h, w = ctx.shape
        n = w if axis == 1 else h
        ramp = (np.arange(n, dtype=np.float64) / n)
        ramp = ramp[None, :] if axis == 1 else ramp[:, None]
        return np.broadcast_to(ramp, ctx.shape).astype(np.float64)
    return fn


def _safe_div(a, b):
    b = np.asarray(b, dtype=np.float64)
    return np.divide(a, b, out=np.zeros(np.broadcast_shapes(np.shape(a), b.shape)),
                     where=b != 0)


# Builtin functions: name -> (arity, impl(ctx, *args)).
BUILTINS = {
    # neighborhood aggregates (center cell excluded, toroidal wrap)
    "sum8": (1, _make_neigh(lambda s: np.sum(s, axis=0), _MOORE)),
    "avg8": (1, _make_neigh(lambda s: np.mean(s, axis=0), _MOORE)),
    "max8": (1, _make_neigh(lambda s: np.max(s, axis=0), _MOORE)),
    "min8": (1, _make_neigh(lambda s: np.min(s, axis=0), _MOORE)),
    "sum4": (1, _make_neigh(lambda s: np.sum(s, axis=0), _VON_NEUMANN)),
    "avg4": (1, _make_neigh(lambda s: np.mean(s, axis=0), _VON_NEUMANN)),
    "max4": (1, _make_neigh(lambda s: np.max(s, axis=0), _VON_NEUMANN)),
    "min4": (1, _make_neigh(lambda s: np.min(s, axis=0), _VON_NEUMANN)),
    # shift(f, dx, dy): cell (x, y) sees f at (x-dx, y-dy) — the field
    # translated by (dx, dy). dx, dy must be constants.
    "shift": (3, lambda ctx, x, dx, dy: np.roll(_grid(x, ctx),
                                                (int(np.asarray(dy).item()),
                                                 int(np.asarray(dx).item())),
                                                axis=(0, 1))),
    # inheritance: value of the neighbor with the largest key (see _make_argmax)
    "argmax8": (2, _make_argmax(_MOORE)),
    "argmax4": (2, _make_argmax(_VON_NEUMANN)),
    # normalized cell coordinates in [0, 1) — for spatially varying environments
    "posx": (0, _pos_ramp(1)),
    "posy": (0, _pos_ramp(0)),
    # elementwise math
    "abs": (1, lambda ctx, x: np.abs(x)),
    "sign": (1, lambda ctx, x: np.sign(x)),
    "floor": (1, lambda ctx, x: np.floor(x)),
    "round": (1, lambda ctx, x: np.round(x)),
    "sqrt": (1, lambda ctx, x: np.sqrt(np.maximum(x, 0.0))),
    "exp": (1, lambda ctx, x: np.exp(np.minimum(x, 60.0))),
    "log": (1, lambda ctx, x: np.log(np.maximum(x, 1e-12))),
    "min": (2, lambda ctx, a, b: np.minimum(a, b)),
    "max": (2, lambda ctx, a, b: np.maximum(a, b)),
    "clamp": (3, lambda ctx, x, lo, hi: np.clip(x, lo, hi)),
    "where": (3, lambda ctx, c, a, b: np.where(_truthy(c), a, b)),
    # randomness (uses the world RNG -> reproducible under a fixed seed)
    "rand": (0, lambda ctx: ctx.rng.random(ctx.shape)),
    "bernoulli": (1, lambda ctx, p: (ctx.rng.random(ctx.shape) < p).astype(np.float64)),
    "uniform": (2, lambda ctx, a, b: a + ctx.rng.random(ctx.shape) * (np.asarray(b) - np.asarray(a))),
    "normal": (2, lambda ctx, mu, sigma: mu + ctx.rng.standard_normal(ctx.shape) * sigma),
}

_COMPARE = {
    "==": np.equal, "!=": np.not_equal,
    "<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
}


def evaluate(expr: tuple, state: dict[str, np.ndarray], ctx: EvalCtx):
    """Evaluate an AST node to a float64 array (or python float for literals)."""
    kind = expr[0]
    if kind == "num":
        return expr[1]
    if kind == "name":
        try:
            return state[expr[1]]
        except KeyError:
            raise NomosError(f"unknown channel {expr[1]!r}") from None
    if kind == "un":
        val = evaluate(expr[2], state, ctx)
        if expr[1] == "neg":
            return -np.asarray(val, dtype=np.float64)
        return (~_truthy(val)).astype(np.float64)
    if kind == "bin":
        op = expr[1]
        l = evaluate(expr[2], state, ctx)
        r = evaluate(expr[3], state, ctx)
        if op == "+":
            return np.add(l, r, dtype=np.float64)
        if op == "-":
            return np.subtract(l, r, dtype=np.float64)
        if op == "*":
            return np.multiply(l, r, dtype=np.float64)
        if op == "/":
            return _safe_div(l, r)
        if op == "%":
            return np.mod(l, np.where(np.asarray(r) == 0, 1.0, r))
        if op == "and":
            return (_truthy(l) & _truthy(r)).astype(np.float64)
        if op == "or":
            return (_truthy(l) | _truthy(r)).astype(np.float64)
        return _COMPARE[op](l, r).astype(np.float64)
    if kind == "call":
        fn = expr[1]
        if fn not in BUILTINS:
            raise NomosError(f"unknown function {fn!r}")
        arity, impl = BUILTINS[fn]
        args = [evaluate(a, state, ctx) for a in expr[2]]
        if len(args) != arity:
            raise NomosError(f"{fn}() takes {arity} argument(s), got {len(args)}")
        return impl(ctx, *args)
    raise NomosError(f"bad AST node {expr!r}")


def _validate(program: Program) -> None:
    """Static checks so a bad law fails loudly at load, not mid-run."""
    known: list[str] = []
    for ch in program.channels:
        if ch.name in known:
            raise NomosError(f"channel {ch.name!r} declared twice")
        if ch.init is not None:
            for name in walk_names(ch.init):
                if name not in known:
                    raise NomosError(
                        f"init of channel {ch.name!r} references {name!r}, "
                        "which is not declared yet"
                    )
            for fn in walk_calls(ch.init):
                if fn not in BUILTINS:
                    raise NomosError(f"unknown function {fn!r} in init of {ch.name!r}")
        known.append(ch.name)
    if not known:
        raise NomosError("a law file must declare at least one channel")
    for rule in program.rules:
        exprs = [rule.when] + [a[2] for a in rule.actions]
        for expr in exprs:
            for name in walk_names(expr):
                if name not in known:
                    raise NomosError(f"rule {rule.name!r} references unknown channel {name!r}")
            for fn in walk_calls(expr):
                if fn not in BUILTINS:
                    raise NomosError(f"rule {rule.name!r} uses unknown function {fn!r}")
        for _, ch, _ in rule.actions:
            if ch not in known:
                raise NomosError(f"rule {rule.name!r} writes to unknown channel {ch!r}")


class World:
    """A universe: named float64 grids + the rules they live by."""

    def __init__(self, program: Program, size: int | tuple[int, int], seed: int = 0):
        _validate(program)
        self.program = program
        self.shape = (size, size) if isinstance(size, int) else tuple(size)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.time = 0
        ctx = EvalCtx(self.shape, self.rng)
        self.state: dict[str, np.ndarray] = {}
        for ch in program.channels:
            if ch.init is None:
                val = np.zeros(self.shape)
            else:
                val = np.asarray(evaluate(ch.init, self.state, ctx), dtype=np.float64)
                val = np.broadcast_to(val, self.shape).copy()
            self.state[ch.name] = val

    def step(self) -> float:
        """Advance one tick. Returns activity: fraction of cells that changed."""
        snapshot = self.state
        ctx = EvalCtx(self.shape, self.rng)
        new = {k: v.copy() for k, v in snapshot.items()}
        for rule in self.program.rules:
            mask = _grid(_truthy(evaluate(rule.when, snapshot, ctx)), ctx)
            if not mask.any():
                continue
            for op, ch, expr in rule.actions:
                val = evaluate(expr, snapshot, ctx)
                if op == "set":
                    new[ch] = np.where(mask, val, new[ch])
                else:
                    new[ch] = new[ch] + np.where(mask, val, 0.0)
        changed = np.zeros(self.shape, dtype=bool)
        for k in new:
            arr = np.nan_to_num(new[k], nan=0.0, posinf=_CLIP, neginf=-_CLIP)
            np.clip(arr, -_CLIP, _CLIP, out=arr)
            new[k] = arr
            changed |= arr != snapshot[k]
        self.state = new
        self.time += 1
        return float(changed.mean())

    def run(self, steps: int) -> list[float]:
        """Advance many ticks; returns the per-step activity history."""
        return [self.step() for _ in range(steps)]
