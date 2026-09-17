"""The atlas genome, and a lineage that can hear a difference.

    .\\.venv\\Scripts\\python.exe -m cortex.atlas_lineage --generations 30 --jobs 16
    .\\.venv\\Scripts\\python.exe -m cortex.atlas_lineage --world sparse|rich ...

What is heritable here is NOT the rule (cortex/wave.py asks whether that can be found)
but the map's GROWTH PROGRAM: how big it is, how fast its wave decays, how long it
remembers, how much it explores, how much a wall leaks — a dozen named scalars with
ranges, the same shape as `cortex/program.py`, small enough to read and to hand to a
model. The radius is the one that matters most: it is the horizon of both memory and
planning, and every cell of it is paid for every tick.

The lineage is (1 + lambda) with lambda in the teens, evaluated IN PARALLEL: the
children are SCREENED on a rotating handful of worlds and the finalist is CONFIRMED
against the parent on all of them, both re-lived — because the best of fifteen on four
maps is usually the luckiest of fifteen, and without the confirmation the lineage
accepted a child almost every generation while going nowhere.
That is what the old cortex lineage could not do: one life per generation, serial, and
a fitness whose spread across worlds was wider than the spread across children
(journal, 2026-09-14). With the atlas a life costs ten seconds and sixteen fit on the
cores, so a generation sees sixteen children over four worlds each and the comparison
has a chance of meaning something.

And the experiment this exists for, `--world sparse` against `--world rich`: one
program, two economies. Does the map grow to a different size where food is far apart
than where it is everywhere? That is the "genome x world" question the design note
deferred, asked of the one gene where the answer is legible.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from cortex import atlas as A
from cortex.world import Config as WorldConfig

# name -> (default, lo, hi, integer?, what it does)
FIELDS: dict[str, tuple] = {
    "radius":      (12, 4, 24, True,
                    "map half-width: the horizon of memory and of planning, paid for"
                    " every tick"),
    "decay":       (0.06, 0.01, 0.30, False,
                    "value lost per cell of travel; high = only what is near matters"),
    "forget":      (0.0015, 0.0, 0.01, False,
                    "evidence lost per tick; high = a map that only shows the present"),
    "frontier":    (0.12, 0.0, 0.6, False,
                    "pull of an unseen cell beside a seen one: exploration"),
    "smell_floor": (0.35, 0.0, 1.0, False,
                    "the least a faint smell is painted at"),
    "thing_gain":  (1.0, 0.0, 2.0, False, "pull of a seen bright thing"),
    "leak":        (0.5, 0.0, 0.9, False,
                    "how much wave a solid cell lets through, innately (the hue learns)"),
    "linger":      (0.99, 0.0, 0.998, False,
                    "commitment: what a painted source keeps of itself per wave pass"
                    " after the paint moves on (0 = none)"),
    "subticks":    (6, 1, 16, True, "wave passes per thought"),
    "think_every": (3, 1, 8, True, "world ticks per thought"),
    "smell_here":  (0.85, 0.5, 1.0, False, "a smell this strong is under the nose"),
    "turn_gain":   (3.0, 0.5, 8.0, False, "how hard a heading error is steered against"),
    "slow_angle":  (1.0, 0.2, 3.0, False, "heading error above which the body slows"),
}
DEFAULTS = {k: (int(v[0]) if v[3] else float(v[0])) for k, v in FIELDS.items()}


def clamp(name: str, value):
    default, lo, hi, is_int, _ = FIELDS[name]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return DEFAULTS[name]
    if not np.isfinite(v):
        return DEFAULTS[name]
    v = min(float(hi), max(float(lo), v))
    return int(round(v)) if is_int else float(v)


class Program:
    """The heritable half of an atlas. `config()` is what a phenotype runs on."""

    _FLOOR_FRAC = 0.03

    def __init__(self, values: dict | None = None):
        self.values = dict(DEFAULTS)
        if values:
            for k, v in values.items():
                if k in FIELDS:
                    self.values[k] = clamp(k, v)

    def config(self, **fixed) -> A.Config:
        return A.Config(**self.values, **fixed)

    def mutate(self, rng, prob: float = 0.3, scale: float = 0.2) -> "Program":
        """A step sized by the value, floored by the range (program.py's lesson: a
        knob at zero must still be able to move, a knob near its floor must not be
        thrown across its range)."""
        out = Program(self.values)
        for k, (default, lo, hi, is_int, _) in FIELDS.items():
            if rng.random() >= prob:
                continue
            span = float(hi) - float(lo)
            step = scale * max(abs(float(out.values[k])), self._FLOOR_FRAC * span)
            out.values[k] = clamp(k, out.values[k] + rng.normal(0.0, step))
        return out

    def diff(self, other: "Program") -> dict:
        return {k: (other.values[k], self.values[k]) for k in FIELDS
                if other.values[k] != self.values[k]}

    def to_dict(self) -> dict:
        return dict(self.values)

    def __str__(self) -> str:
        return "  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                         for k, v in self.values.items())


# --- worlds: two economies -----------------------------------------------------------

WORLDS = {
    "default": {},
    # food far apart: memory and horizon should pay, if they ever do
    "sparse": dict(food_per_chunk=0, larder_max=1, tree_spots=1, cave_spots=2),
    # food everywhere: a big map is a tax on a problem that does not exist
    "rich": dict(food_per_chunk=3, larder_max=3, tree_spots=3, cave_spots=6),
}


# --- evaluation ------------------------------------------------------------------------

def _life(args) -> dict:
    prog_d, seed, ticks, world = args
    from cortex.world import FOOD_ENERGY
    w, mind = A.live(int(seed), int(ticks), Program(prog_d).config(),
                     wcfg=WorldConfig(**WORLDS[world]))
    gained = float(sum(n * FOOD_ENERGY[k] for k, n in enumerate(w.agent.ate)))
    return {"ticks": w.tick, "meals": int(sum(w.agent.ate)), "gained": gained,
            "fitness": gained + 0.002 * w.tick}


def evaluate(pool, programs, seeds, ticks: int, world: str) -> list[dict]:
    """Every program on every seed, in parallel; mean per program."""
    args = [(p.to_dict(), s, ticks, world) for p in programs for s in seeds]
    res = pool.map(_life, args) if pool else [_life(a) for a in args]
    n = len(seeds)
    out = []
    for i in range(len(programs)):
        rows = res[i * n:(i + 1) * n]
        out.append({k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
    return out


def worlds_for(pool, gen: int, per_gen: int) -> list:
    pool = list(pool)
    n = max(1, min(int(per_gen), len(pool)))
    start = (gen * n) % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(n)]


def run(generations: int = 30, lam: int = 15, seeds=tuple(range(8)), per_gen: int = 4,
        ticks: int = 8000, world: str = "default", jobs: int = 16, brain_seed: int = 0,
        start: Program | None = None, log=print, save: str = "") -> dict:
    from multiprocessing import Pool
    rng = np.random.default_rng(brain_seed)
    parent = start or Program()
    history = []
    pool = Pool(jobs) if jobs > 1 else None
    try:
        for gen in range(1, generations + 1):
            t0 = time.time()
            turn = worlds_for(seeds, gen, per_gen)
            rest = [sd for sd in seeds if sd not in turn]
            kids = [parent.mutate(rng) for _ in range(lam)]
            # SCREEN the children on this turn's worlds...
            scores = evaluate(pool, kids, turn, ticks, world)
            i = int(np.argmax([s["fitness"] for s in scores]))
            # ...then CONFIRM the finalist against the parent on ALL the worlds. The
            # best of fifteen on four maps is, more often than not, the luckiest of
            # fifteen: the first version accepted a child in eleven generations of
            # twelve and the parent's score never trended anywhere (the same winner's
            # curse the old cortex lineage showed under a population). A finalist has
            # to win on maps it was not picked on.
            fin, bar = evaluate(pool, [kids[i], parent], list(turn) + rest, ticks, world)
            kept = fin["fitness"] > bar["fitness"]
            if kept:
                changed = kids[i].diff(parent)
                parent = kids[i]
            history.append({"gen": gen, "worlds": turn, "parent": bar,
                            "best_child": fin, "screen": scores[i], "kept": kept,
                            "program": parent.to_dict()})
            log(f"gen {gen:>3}  screen {turn}  parent {bar['fitness']:>6.1f}"
                f" ({bar['meals']:.1f} meals, {bar['ticks']:.0f} t)  finalist"
                f" {fin['fitness']:>6.1f} ({fin['meals']:.1f}; screened {scores[i]['fitness']:.0f})"
                f"  kids mean {np.mean([s['fitness'] for s in scores]):>6.1f}"
                f"  {'KEPT ' + _fmt_diff(changed) if kept else 'rejected'}"
                f"  {time.time() - t0:.0f}s")
            if save:
                with open(save, "w", encoding="utf-8") as fh:
                    json.dump({"program": parent.to_dict(), "world": world,
                               "history": history}, fh, indent=1)
    finally:
        if pool:
            pool.close()
            pool.join()
    return {"program": parent, "history": history}


def _fmt_diff(d: dict) -> str:
    return " ".join(f"{k} {a:.3g}->{b:.3g}" for k, (a, b) in d.items())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evolve the atlas's growth program.")
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--lam", type=int, default=15)
    p.add_argument("--seeds", default="0-7", help="the pool of worlds")
    p.add_argument("--worlds", type=int, default=4, dest="per_gen")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--world", choices=sorted(WORLDS), default="default")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--brain-seed", type=int, default=0)
    p.add_argument("--out", default="out/cortex/atlas_lineage")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.world}-s{args.brain_seed}"
    logf = open(os.path.join(args.out, f"{tag}.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"start  world {args.world}  {Program()}")
    res = run(args.generations, args.lam, A._seeds(args.seeds), args.per_gen, args.ticks,
              args.world, args.jobs, args.brain_seed, log=log,
              save=os.path.join(args.out, f"{tag}.json"))
    log(f"final  {res['program']}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
