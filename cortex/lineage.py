"""The lineage: one agent, one long life at a time, the genome carried forward.

    .\\.venv\\Scripts\\python.exe -m cortex.lineage --generations 8 --seeds 0-1

This is (1 + lambda) with lambda = 1 by default, which — as docs/cortex-design.md
argues — is exactly what "one long life plus a dead-end trigger" already is: something
has to change after the dead end, that something is the genome, and regrowing from it
is a new life.

What crosses the boundary is the GENOME and nothing else. The learned weights of a
life are thrown away (`--lamarck` keeps them, as the A/B arm), and the grown phenotype
is never inherited — inheriting it would stop the growth program from being expressed,
therefore from being selected, and it would rot unnoticed.

Lives rotate over several WORLD SEEDS, because a lineage scored on one map learns that
map. `nomos/lamarck.py` measured exactly that: skill accumulated beautifully on the
mazes a lineage had seen and collapsed on a held-out one.

DEAD ENDS are watched for, and the point is not merely to restart but to say what KIND
of intervention is needed — a nudge, a structural leap, or (later) a call to the LLM.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from cortex import brain as B
from cortex import probes
from cortex.agent import Mind, fresh_genome
from cortex.session import record, summarise
from cortex.world import Config as WorldConfig
from cortex.world import World

BEST = os.path.join("out", "cortex", "best_genome.json")


def evaluate(genome: B.Genome, seeds, max_ticks: int = 20000, think_every: int = 2,
             bcfg: B.Config | None = None, wcfg: WorldConfig | None = None,
             probe_ticks: int = 500) -> dict:
    """Live one life per world seed and average. Returns the life's PROFILE, not a
    single number: at lambda = 1 there is one sample per turn, so it had better say
    more than 'better' or 'worse' (and it is what an LLM would later be handed)."""
    lives, minds = [], []
    for seed in seeds:
        w = World(seed=int(seed), cfg=wcfg or WorldConfig())
        mind = Mind(genome, bcfg, think_every=think_every)
        mind.attach(w)
        acts = []
        while w.agent.alive and w.tick < max_ticks:
            mind.step(w)
            if w.tick % think_every == 0:
                acts.append(mind.command)
        lives.append(_profile(w, mind, acts, max_ticks))
        minds.append(mind)
    out = {k: float(np.mean([lf[k] for lf in lives])) for k in lives[0]
           if isinstance(lives[0][k], (int, float))}
    out["ate"] = [int(sum(lf["ate"][k] for lf in lives)) for k in range(4)]
    out["lives"] = lives
    # THE EXAM. A life's score is one noisy number; the profile is what an LLM (or we)
    # can actually reason about, and what stops a mean from flattering a brain that has
    # only learned to hold the interact button down.
    if probe_ticks:
        suites = [probes.run_suite(m, ticks=probe_ticks) for m in minds]
        keys = suites[0]["profile"].keys()
        out["skills"] = {k: round(float(np.mean([s["profile"][k] for s in suites])), 3)
                         for k in keys}
        out["suite"] = suites[0]              # kept so a generation can be LOOKED at
    else:
        out["skills"] = {}
    # fitness leans on the LIFE (what the world actually paid it) and takes the exam as
    # a tie-break, so a brain cannot win by being good at tests and bad at living
    out["fitness"] = (out["gained"] + 0.002 * out["ticks"]
                      + 40.0 * out["skills"].get("mean", 0.0))
    return out


def _profile(w: World, mind: Mind, acts, cap: int) -> dict:
    """One life, described by more than its score — including the three DEAD-END
    signals that can be read from inside a single life."""
    a = w.agent
    from cortex.world import FOOD_ENERGY
    gained = float(sum(n * FOOD_ENERGY[k] for k, n in enumerate(a.ate)))
    # 1. ENERGETIC: is it feeding itself at all?
    meal_rate = 1000.0 * sum(a.ate) / max(1, w.tick)
    # 2. BEHAVIOURAL: has the action distribution collapsed? A brain doing one thing
    #    forever is the reactive limit cycle the forage world showed us on a GIF.
    ent = _entropy(acts)
    # 3. STRUCTURAL: has learning stopped moving?
    dw = mind.phenotype.plastic_norm()
    return {"ticks": w.tick, "meals": int(sum(a.ate)), "gained": gained,
            "distance": w.distance, "immortal": float(w.tick >= cap),
            "poisoned": int(a.ate[3]), "entropy": ent, "meal_rate": meal_rate,
            "dW": dw, "idle": int(w.deeds["idle"]), "ate": list(a.ate)}


def _entropy(acts) -> float:
    """Entropy of the discretised command stream, 0 = does exactly one thing."""
    if not acts:
        return 0.0
    buckets: dict[tuple, int] = {}
    for thrust, turn, act in acts:
        key = (int(thrust > 0.5), int(round(max(-1.0, min(1.0, turn)))), int(act > 0.5))
        buckets[key] = buckets.get(key, 0) + 1
    n = sum(buckets.values())
    return float(-sum((c / n) * math.log(c / n + 1e-12) for c in buckets.values()))


# how far a mutation may be scaled up or down by the 1/5 rule, and how fast
STEP_MIN, STEP_MAX, STEP_UP, STEP_DOWN = 0.15, 3.0, 1.30, 0.82
# the rates a STEP scales. Not the ceilings and not the mechanisms — only how often and
# how far a mutation reaches.
_STEP_KEYS = ("weight_mut", "weight_mut_prob", "program_mut", "program_mut_prob",
              "grow_prob", "shrink_prob", "add_in_prob", "del_in_prob", "add_out_prob",
              "del_out_prob", "organ_mod_prob", "retype_prob", "split_prob",
              "add_layer_prob", "del_layer_prob", "widen_prob", "narrow_prob",
              "subtick_mut_prob")


def scaled(cfg: B.Config, step: float) -> B.Config:
    """A config whose mutation is `step` times as strong. One number for the whole
    operator, because the question the 1/5 rule asks — is this a step or a leap — is
    about the operator, not about any one of its rates."""
    out = B.Config(**vars(cfg))
    for k in _STEP_KEYS:
        setattr(out, k, min(1.0, getattr(cfg, k) * step) if k.endswith("prob")
                else getattr(cfg, k) * step)
    return out


def worlds_for(pool, gen: int, per_gen: int) -> list:
    """Which worlds this generation is judged on — a rotating window over the pool.

    Two reasons, and the second was found the hard way. **A lineage scored on one map
    learns that map**: `nomos/lamarck.py` measured exactly that, skill accumulating on
    the mazes a lineage had seen and collapsing on a held-out one, and this file's own
    docstring claimed lives rotated when the code ran the same pair every generation.

    And re-measuring the incumbent — added to catch a parent flattered by a lucky life —
    was a NO-OP. `evaluate` is a pure function of (genome, seeds, ticks), so re-scoring
    the same genome on the same seeds returns the same number to the last digit. It cost
    a full evaluation per generation and bought nothing; the printed bar sat at 283.6
    for eleven generations without moving. On rotating worlds it finally means what it
    was meant to: a parent that is good only where it grew up gets found out."""
    pool = list(pool)
    n = max(1, min(int(per_gen), len(pool)))
    start = (gen * n) % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(n)]


def adapt(step: float, accepted: bool) -> float:
    """Rechenberg's 1/5 success rule: grow the step when children are accepted, shrink
    it when they are not. A (1 + 1) strategy wants roughly one child in five to be an
    improvement — the first lineage run on a foraging brain accepted ZERO of fourteen,
    which says the operator was leaping, not stepping, and no amount of patience would
    have fixed it."""
    return float(min(STEP_MAX, max(STEP_MIN,
                                   step * (STEP_UP if accepted else STEP_DOWN))))


def diagnose(profile: dict, history: list[float], patience: int = 3) -> list[str]:
    """Which dead-end signals are firing. Not just "restart" — WHICH kind of change is
    called for, which is the whole reason to name them separately."""
    flags = []
    if profile["entropy"] < 0.35:
        flags.append("frozen-behaviour")            # does one thing forever
    if profile["meals"] < 1.0:
        flags.append("not-eating")                  # never solved the basic problem
    if profile["dW"] < 1e-3:
        flags.append("not-learning")                # plasticity is inert
    if len(history) > patience and max(history[-patience:]) <= max(history[:-patience]):
        flags.append("stalled")                     # the lineage stopped improving
    return flags


def run(generations: int = 8, lam: int = 1, seeds=(0, 1, 2, 3), max_ticks: int = 20000,
        think_every: int = 2, brain_seed: int = 0, lamarck: bool = False,
        save: str = BEST, log: bool = True, quiet: bool = False,
        strips: str = "out/cortex/probes", per_gen: int = 2) -> dict:
    rng = np.random.default_rng(brain_seed)
    bcfg = B.Config()
    parent = fresh_genome(seed=brain_seed)
    best = evaluate(parent, worlds_for(seeds, 0, per_gen), max_ticks,
                    think_every, bcfg)
    history = [best["fitness"]]
    if not quiet:
        _report(0, best, [], parent)
        if strips and best.get("suite"):
            os.makedirs(strips, exist_ok=True)
            probes.render(best["suite"], os.path.join(strips, "gen000.png"))
    step, accepts = 1.0, []
    for gen in range(1, generations + 1):
        turn = worlds_for(seeds, gen, per_gen)
        flags = diagnose(best, history)
        # THE STEP SIZE adapts (Rechenberg's 1/5 rule, `adapt`). What used to be here
        # did the opposite: a stalled lineage got its mutation rates multiplied by
        # 2.5-3x, on the assumption that stalling means the steps are too small. The
        # first run on a brain that could forage falsified that — fourteen children,
        # zero accepted, a parent eating 8.5 meals against children eating 0.5. The
        # steps were far too LARGE and the handler made them larger every generation
        # from the fourth on. Basin escape is still available, but only when the
        # acceptance rate says we are genuinely inching along a local optimum rather
        # than falling off one.
        cfg = scaled(bcfg, step)
        recent = accepts[-5:]
        if "stalled" in flags and recent and sum(recent) / len(recent) >= 0.4:
            cfg.split_prob = min(1.0, cfg.split_prob * 3.0)
            cfg.add_layer_prob = min(1.0, cfg.add_layer_prob * 3.0)
        children = []
        for _ in range(max(1, lam)):
            child = B.mutate(parent, rng, cfg)
            children.append(
                (evaluate(child, turn, max_ticks, think_every, bcfg), child))
        cand, cgen = max(children, key=lambda kv: kv[0]["fitness"])
        # RE-MEASURE THE INCUMBENT rather than trusting the score it happened to get
        # when it was young. A life is noisy, and in a (1 + 1) loop a lucky parent
        # measurement is permanent: it becomes a bar nothing can clear, and the run
        # reads as "no child was ever better" when it means "the parent was measured on
        # a good day". Averaging the estimate lets a flattered parent come back down.
        again = evaluate(parent, turn, max_ticks, think_every, bcfg)
        bar = 0.5 * (best["fitness"] + again["fitness"])
        kept = cand["fitness"] >= bar
        if kept:
            best, parent = cand, cgen
        else:
            best = dict(again, fitness=bar)     # the incumbent, re-scored
        accepts.append(kept)
        step = adapt(step, kept)
        history.append(best["fitness"])
        if log:
            # log the CHILD that was actually lived, accepted or not — logging the
            # surviving parent again would fill the table with lives nobody led
            record(_fake_world(cand), f"brain-g{gen}",
                   note=("kept" if kept else "rejected") + f" {' '.join(flags)}")
        if not quiet:
            _report(gen, cand, flags, cgen, kept=kept, step=step, bar=bar)
            if strips and cand.get("suite"):
                os.makedirs(strips, exist_ok=True)
                probes.render(cand["suite"], os.path.join(strips, f"gen{gen:03d}.png"))
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        with open(save, "w", encoding="utf-8") as fh:
            json.dump(parent.to_dict(), fh)
    return {"genome": parent, "profile": best, "history": history}


class _fake_world:
    """Adapter so a lineage generation lands in the same sessions table as a human's
    life — one schema for every player is what makes the numbers comparable at all."""

    def __init__(self, profile):
        p = profile
        self.seed = -1
        self.tick = int(p["ticks"])
        self.distance = p["distance"]
        self.deeds = {"eat": int(p["meals"]), "door": 0, "break": 0, "push": 0,
                      "idle": int(p["idle"])}
        self.cfg = WorldConfig()
        self.agent = type("A", (), {"alive": bool(p["immortal"]),
                                    "energy": 0.0, "ate": p["ate"]})()


def _report(gen: int, prof: dict, flags, genome: B.Genome, kept: bool = True,
            step: float | None = None, bar: float | None = None) -> None:
    sk = prof.get("skills") or {}
    skills = "  ".join(f"{k[:4]} {v:.2f}" for k, v in sk.items() if k != "mean")
    # the BAR and the STEP are printed because a rejection is uninformative without
    # them: it could mean the child was bad, or that the parent was measured on a good
    # day, or that the operator is still leaping
    extra = ("" if bar is None else f"  vs {bar:>7.1f}") + \
            ("" if step is None else f"  step {step:.2f}")
    print(f"gen {gen:>3}  fit {prof['fitness']:>8.1f}{extra}"
          f"  ticks {prof['ticks']:>7.0f}"
          f"  meals {prof['meals']:>5.1f}  H {prof['entropy']:.2f}"
          f"  types {genome.n_types}"
          f"  {'' if kept else '(rejected)'} {' '.join(flags)}")
    if skills:
        print(f"        skills  {skills}   | mean {sk.get('mean', 0):.2f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run a cortex lineage.")
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--lam", type=int, default=1, help="children per generation")
    p.add_argument("--seeds", default="0-5",
                   help="the POOL of worlds a lineage may be judged on")
    p.add_argument("--worlds", type=int, default=2, dest="per_gen",
                   help="how many of the pool each generation uses, rotating — a "
                        "lineage scored on one map learns that map")
    p.add_argument("--ticks", type=int, default=20000, help="cap on one life")
    p.add_argument("--think-every", type=int, default=2)
    p.add_argument("--brain-seed", type=int, default=0)
    p.add_argument("--save", default=BEST)
    p.add_argument("--strips", default="out/cortex/probes",
                   help="write one probe strip per generation here ('' to skip)")
    args = p.parse_args(argv)
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    out = run(generations=args.generations, lam=args.lam, seeds=seeds,
              max_ticks=args.ticks, think_every=args.think_every,
              brain_seed=args.brain_seed, save=args.save, strips=args.strips,
              per_gen=args.per_gen)
    print()
    print(summarise())
    print(f"\nbest genome -> {args.save}  "
          f"(play it: python -m cortex.app --load-brain {args.save})")


if __name__ == "__main__":
    main()
