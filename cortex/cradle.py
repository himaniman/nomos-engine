"""The cradle: a world where a meal actually happens, and the experiment that needs it.

    .\\.venv\\Scripts\\python.exe -m cortex.cradle

This exists because of one measurement (docs/lab-journal.md, 2026-07-24). Over a life of
4000 ticks in the real world a brain receives **zero positive reward events and four
thousand negative ones**. Living costs energy every tick whatever you do, and the agent
never eats, so the three-factor learning rule — `dW += lr * reward * eligibility` — was
never failing. It was working perfectly, and faithfully teaching the tissue that
everything it does is bad, with no positive example anywhere to point at.

You cannot tell a broken learner from a starved one by looking at what it learned. So
the cradle removes the starvation: a small world thick with food and no poison, where a
meal happens by accident within a few hundred ticks. It is NOT the world we are building
and no score from it counts for anything. It is a test bench with one question:

> **Given reward that actually arrives, does within-life learning contribute anything?**

The answer branches the whole project. If yes, the machinery is sound and the problem is
that the real world's reward is too sparse to bootstrap from — which is an argument for
innate reflexes, something to make the first meal happen. If no, plasticity itself is
broken, and innate reflexes would only HIDE that: the agent would score well on the
reflex and we would never find out the tissue learns nothing.

**How the question is asked, which is the part that took thought.** A brain both grows
and learns during a life, so "it got better" cannot be attributed to either. The
ablation splits them: take ONE snapshot at the end of a life and sit two copies of it at
the same exams — one as it is, one with the within-life weight deltas zeroed and
everything else identical (same body, same cell states, same wiring). The difference is
learning, with growth held fixed.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from cortex import brain as B
from cortex import probes
from cortex.agent import Mind, fresh_genome
from cortex.world import Config as WorldConfig
from cortex.world import World


def cradle_config(**kw) -> WorldConfig:
    """A small world thick with food and free of poison.

    Every number here is chosen to make a MEAL LIKELY, not to make a fair test — the
    cradle is a bench, not a benchmark. Poison is off because a nightcap teaches a
    negative that has nothing to do with the question being asked; the world is small so
    a wanderer cannot get lost in it; food renews so a life cannot exhaust the supply
    and quietly become the sparse world again halfway through."""
    cfg = dict(size=96, food_per_chunk=14, tree_spots=6, cave_spots=8,
               larder_max=4, renewable_frac=1.0, regrow_ticks=300,
               poison_weight=0.0)
    cfg.update(kw)
    return WorldConfig(**cfg)


def live_in_cradle(genome, seed: int = 0, ticks: int = 4000,
                   wcfg: WorldConfig | None = None) -> tuple[World, Mind, dict]:
    """One life, counting the FEEDBACK rather than only the outcome. The reward tallies
    are the point: a run that ate nothing has not tested learning, it has tested
    starvation again, and the numbers have to say which happened."""
    w = World(seed=seed, cfg=wcfg or cradle_config())
    mind = Mind(genome)
    mind.attach(w)
    pos = neg = 0
    gain = 0.0
    while w.agent.alive and w.tick < ticks:
        mind.step(w)
        r = mind.last_reward
        if r > 1e-9:
            pos += 1
            gain += r
        elif r < -1e-9:
            neg += 1
    return w, mind, {"ticks": w.tick, "meals": int(sum(w.agent.ate)),
                     "r_pos": pos, "r_neg": neg, "r_gain": round(gain, 3),
                     "cells": mind.phenotype.cells()}


def learning_delta(mind, probe_ticks: int = 500) -> dict:
    """The ablation. Two copies of one brain at the same exams; the only difference is
    whether the within-life weight deltas are kept."""
    kept = probes.run_suite(mind, ticks=probe_ticks)["profile"]
    wiped = probes.run_suite(mind, ticks=probe_ticks, unlearn=True)["profile"]
    return {"kept": kept, "wiped": wiped,
            "delta": {k: round(kept[k] - wiped[k], 3) for k in kept}}


def _row(tag: str, prof: dict) -> str:
    keys = [k for k in prof if k != "mean"]
    return (f"  {tag:<22}" + "".join(f"{prof[k]:>+8.2f}" if tag.startswith("delta")
                                     else f"{prof[k]:>8.2f}" for k in keys)
            + f"{prof['mean']:>9.2f}")


def run(seeds=(0, 1, 2), ticks: int = 4000, probe_ticks: int = 500,
        baselines=(0.0, 0.01), quiet: bool = False) -> dict:
    """The experiment, both arms.

    `reward_baseline = 0` is the OLD behaviour and the control: reward is the raw change
    in energy, so the constant metabolic drain is a permanent negative signal. Nonzero
    subtracts a running mean, which makes the predictable part of the stream worth
    nothing to learn from and leaves a meal — a spike the mean cannot track — as the
    only thing that moves anything."""
    out = {}
    keys = ["near", "open", "side", "behind", "corner", "maze", "patience", "mean"]
    if not quiet:
        print("  " + " " * 22 + "".join(f"{k[:7]:>8}" for k in keys[:-1])
              + f"{'mean':>9}")
    for mix in baselines:
        lives, deltas = [], []
        for s in seeds:
            g = fresh_genome(seed=s)
            g.program.update({"reward_baseline": mix})
            w, mind, life = live_in_cradle(g, seed=0, ticks=ticks)
            lives.append(life)
            deltas.append(learning_delta(mind, probe_ticks))
            if not quiet:
                print(f"\n  reward_baseline {mix}  brain {s}:  "
                      f"{life['meals']} meals, {life['r_pos']} rewarding ticks "
                      f"(+{life['r_gain']}), {life['r_neg']} costly, "
                      f"{life['cells']} cells")
                print(_row("  learning kept", deltas[-1]["kept"]))
                print(_row("  learning wiped", deltas[-1]["wiped"]))
                print(_row("  delta (= learning)", deltas[-1]["delta"]))
                sys.stdout.flush()
        mean = {k: float(np.mean([d["delta"][k] for d in deltas])) for k in keys}
        out[mix] = {"lives": lives, "deltas": deltas, "mean_delta": mean}
        if not quiet:
            print()
            print(_row(f"delta MEAN over {len(seeds)} brains", mean))
    return out


def reflex_arms(seeds=(0, 1, 2), ticks: int = 2500, probe_ticks: int = 300,
                world: str = "cradle", quiet: bool = False) -> dict:
    """Do the innate reflexes help, and does the TISSUE add anything on top of them?

    Three arms, and the third is the one that stops us fooling ourselves:

      * `brain` — as it lived, arcs and all;
      * `no reflex` — the same brain with the whole innate arc removed (the sensory
        drive, the pioneer tracts and the read-out) and everything it grew and learned
        left alone;
      * `reflex only` — the twenty-line baseline policy, no brain at all.

    `brain` beating `no reflex` says the arcs are load-bearing. `brain` beating `reflex
    only` says the tissue is doing something a hand-written policy does not. Without the
    second number a reflex makes everything look good; without the third we would never
    learn that a decorative brain was riding it."""
    from cortex.baseline import POLICIES
    rows = []
    wcfg = cradle_config() if world == "cradle" else WorldConfig()
    for s in seeds:
        g = fresh_genome(seed=s)
        w = World(seed=0, cfg=wcfg)
        mind = Mind(g)
        mind.attach(w)
        while w.agent.alive and w.tick < ticks:
            mind.step(w)
        tracts = _innate_count(mind)
        with_arc = probes.run_suite(mind, ticks=probe_ticks)["profile"]
        without = probes.run_suite(mind, ticks=probe_ticks, no_reflex=True)["profile"]
        rows.append({"seed": s, "tracts": tracts, "meals": int(sum(w.agent.ate)),
                     "brain": with_arc, "no_reflex": without})
        if not quiet:
            print(f"\n  brain {s}: {rows[-1]['meals']} meals, {tracts} innate tracts, "
                  f"{mind.phenotype.cells()} cells")
            print(_row("  brain (arcs on)", with_arc))
            print(_row("  no reflex", without))
            print(_row("  delta (= the arcs)",
                       {k: with_arc[k] - without[k] for k in with_arc}))
            sys.stdout.flush()
    pol = probes.run_suite_policy(POLICIES["reflex"], mind.layout,
                                  ticks=probe_ticks)["profile"]
    keys = list(rows[0]["brain"])
    mean_arc = {k: float(np.mean([r["brain"][k] - r["no_reflex"][k] for r in rows]))
                for k in keys}
    if not quiet:
        print()
        print(_row(f"delta MEAN over {len(rows)} brains", mean_arc))
        print(_row("reflex POLICY (no brain)", pol))
    return {"rows": rows, "policy": pol, "mean_delta": mean_arc}


def _innate_count(mind) -> int:
    from cortex import reflex
    return reflex.count(mind.phenotype)


def sweep(seeds=(0, 1, 2), ticks: int = 2500, probe_ticks: int = 300,
          grid=((0.5, 0.002), (0.95, 0.0002), (0.98, 0.00005)),
          quiet: bool = False) -> list[dict]:
    """The two TIME CONSTANTS of learning, against the timescale the world rewards on.

    Both defaults were set for a dense-reward regime and this world is a sparse one, by
    one to two orders of magnitude:

      * `plastic_elig` decays the eligibility trace once per THOUGHT. At 0.5 its
        half-life is one thought, so credit reaches about five ticks back — but a meal
        is the END of an approach and the turns that aimed the agent happened tens of
        ticks earlier. The actions that caused the reward are outside the trace
        entirely; the only thing ever credited is pressing E while already on the food.
      * `plastic_forget` decays the learned delta every TICK. At 0.002, with a meal
        arriving every 300-800 ticks, a delta keeps 19-53% of itself between meals. So
        each lesson is mostly gone before the next one arrives and nothing accumulates.

    This sweeps both together, because they are one question — does credit survive long
    enough to reach the behaviour that earned it — and scores them by the ABLATION, not
    by the raw ladder, so growth cannot take the credit."""
    rows = []
    for elig, forget in grid:
        deltas, meals = [], []
        for s in seeds:
            g = fresh_genome(seed=s)
            g.program.update({"plastic_elig": elig, "plastic_forget": forget})
            _, mind, life = live_in_cradle(g, seed=0, ticks=ticks)
            meals.append(life["meals"])
            deltas.append(learning_delta(mind, probe_ticks)["delta"])
        mean = {k: float(np.mean([d[k] for d in deltas])) for k in deltas[0]}
        rows.append({"elig": elig, "forget": forget, "meals": float(np.mean(meals)),
                     "delta": mean})
        if not quiet:
            print(f"  elig {elig:<6} forget {forget:<8} meals {np.mean(meals):>5.1f}"
                  f"   " + _row("", mean).strip())
            sys.stdout.flush()
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cortex.cradle",
        description="Does within-life learning do anything, given reward that arrives?")
    p.add_argument("--seeds", default="0,1,2", help="brain seeds")
    p.add_argument("--ticks", type=int, default=4000)
    p.add_argument("--probe-ticks", type=int, default=500)
    p.add_argument("--baselines", default="0,0.01",
                   help="reward_baseline values to compare (0 = the old raw reward)")
    p.add_argument("--sweep", action="store_true",
                   help="sweep the two learning TIME CONSTANTS instead")
    p.add_argument("--reflex", action="store_true",
                   help="do the INNATE ARCS help, and does the tissue add anything?")
    p.add_argument("--world", choices=("cradle", "real"), default="cradle")
    args = p.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.reflex:
        print("  " + " " * 22 + "".join(f"{k[:7]:>8}" for k in
                                        ("near", "open", "side", "behind", "corner",
                                         "maze", "patience")) + f"{'mean':>9}")
        reflex_arms(seeds=seeds, ticks=args.ticks, probe_ticks=args.probe_ticks,
                    world=args.world)
        return 0
    if args.sweep:
        print("  the ablation delta, by how long credit survives")
        print("  " + " " * 42 + "".join(f"{k[:7]:>8}" for k in
                                        ("near", "open", "side", "behind", "corner",
                                         "maze", "patience")) + f"{'mean':>9}")
        sweep(seeds=seeds, ticks=args.ticks, probe_ticks=args.probe_ticks)
        return 0
    out = run(seeds=seeds, ticks=args.ticks, probe_ticks=args.probe_ticks,
              baselines=[float(b) for b in args.baselines.split(",")])
    print("\nRead the DELTA row. It is what within-life learning contributed, with the")
    print("body held fixed. If it is ~0 with meals on the board, plasticity is inert")
    print("and no amount of innate reflex would tell us — it would only hide it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
