"""Baseline players — the floor and the bar the brain has to clear.

    .\\.venv\\Scripts\\python.exe -m cortex.baseline --agent reflex --seeds 0-7

Two of them, both driving `World.step` through the same action tuple a human and a
brain use, and both reading ONLY the sensor vector (never the world), so the
comparison is honest:

  random  a correlated random walk. The floor. Anything that cannot beat this has
          learned nothing at all.

  reflex  twenty lines: turn toward the strongest smell, walk, back off walls, eat
          what is under your nose. No memory, no learning, no idea what poison is.
          This is the bar — the project's own history says a reactive baseline is
          embarrassingly hard to beat (the forage lab's naive food->motor brain came
          first for two rounds), and knowing its number stops us celebrating a brain
          that has merely reinvented it.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from cortex.senses import ODOUR_CH, RAY_CH, TOUCH_CH, ScentField, layout, sense
from cortex.session import record, summarise
from cortex.world import Config, World


def random_policy(vec, w, rng, state) -> tuple[float, float, float]:
    """A correlated random walk: keep a heading for a while, then pick a new one."""
    if state.get("left", 0) <= 0:
        state["left"] = int(rng.integers(20, 80))
        state["turn"] = float(rng.uniform(-1, 1))
    state["left"] -= 1
    return 1.0, state["turn"], 1.0 if rng.random() < 0.15 else 0.0


def reflex_policy(vec, w, rng, state) -> tuple[float, float, float]:
    """Steer by a potential field — attracted by the strongest smell, repelled by
    whatever the eye says is close — and eat what is under the nose.

    The one piece that is NOT obvious, and that the first version got wrong: escaping
    an obstacle needs COMMITMENT. A purely reactive "turn toward the emptier side"
    re-decides every tick, and in a corner the two sides trade places, so the agent
    oscillates against the wall forever. That is the reactive limit cycle the forage
    world already taught us (docs/lab-journal.md); the cure there and here is to pick
    a direction and hold it for a fixed number of ticks."""
    L = state["layout"]
    n, half = w.cfg.n_rays, w.cfg.fov / 2
    olo, _ = L["slices"]["odour"]
    rays = vec[:n * RAY_CH].reshape(n, RAY_CH)
    # nearness weighted by SOLIDITY. Using raw nearness made the policy flee its own
    # dinner: a morsel two cells ahead is a near return like any other, so the escape
    # branch fired every time it closed in and it orbited the food forever. This is
    # exactly what the `solid` channel is in the sensorium for.
    near = rays[:, 4] * rays[:, 5]
    if "rel" not in state:                                # ray bearings, computed once
        state["rel"] = -half + (np.arange(n) + 0.5) * (2 * half / n)
    rel = state["rel"]

    # attraction: the strongest odorant, and which way its WALKABLE path leads
    amp = fwd = lat = 0.0
    for j in range(4):
        a = float(vec[olo + j * ODOUR_CH])
        if a > amp:
            amp = a
            fwd = float(vec[olo + j * ODOUR_CH + 1])
            lat = float(vec[olo + j * ODOUR_CH + 2])
    attract = lat * 2.2 + (0.7 if (amp > 0.005 and fwd < 0) else 0.0)

    # repulsion: near returns push the heading to the other side
    repel = -float(np.sum(near ** 4 * np.sign(rel))) / n * 6.0
    band = near[max(0, n // 2 - 5): n // 2 + 5]
    ahead = float(band.max()) if len(band) else 0.0

    if state.get("escape", 0) > 0:                        # ...hold the escape turn
        state["escape"] -= 1
        turn, thrust = state["esc"], 0.4
    elif ahead > 0.90 or float(vec[L["slices"]["touch"][0]]) > 0.0:
        left, right = float(near[:n // 2].mean()), float(near[n // 2:].mean())
        state["esc"] = -1.0 if left < right else 1.0      # commit to the emptier side
        state["escape"] = 14
        turn, thrust = state["esc"], 0.4
    elif amp < 0.005:                                     # nothing to smell: wander,
        if state.get("left", 0) <= 0:                     # also with commitment
            state["left"] = int(rng.integers(30, 90))
            state["turn"] = float(rng.uniform(-1, 1))
        state["left"] -= 1
        turn, thrust = max(-1.0, min(1.0, state["turn"] + repel)), 1.0
    else:
        turn, thrust = max(-1.0, min(1.0, attract + repel)), 1.0

    # APPROACH SLOWLY. At full speed the body covers 2.1 cells between the 6-tick
    # interact attempts, while the reach is 1.1 — so a confident sprinter flies past
    # its dinner and never gets a press off inside the window. Easing off near the
    # source was worth more than any steering change.
    if amp > 0.85:
        return 0.25, turn * 0.5, 1.0
    return thrust, turn, 0.0


POLICIES = {"random": random_policy, "reflex": reflex_policy}


def play(agent: str, seed: int, max_ticks: int = 20000, cfg: Config | None = None,
         rng_seed: int = 0) -> dict:
    """One life, headless. Deterministic given (agent, seed, rng_seed)."""
    w = World(seed=seed, cfg=cfg or Config())
    field = ScentField(w.cfg.smell_radius)
    rng = np.random.default_rng(rng_seed)
    state = {"layout": layout(w.cfg.n_rays)}
    policy = POLICIES[agent]
    while w.agent.alive and w.tick < max_ticks:
        vec = sense(w, field)
        w.step(*policy(vec, w, rng, state))
    return w


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run a baseline player and log sessions.")
    p.add_argument("--agent", choices=sorted(POLICIES), default="reflex")
    p.add_argument("--seeds", default="0-7", help="e.g. 0-7 or 1,4,9")
    p.add_argument("--ticks", type=int, default=20000)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    for seed in seeds:
        w = play(args.agent, seed, max_ticks=args.ticks)
        rec = record(w, args.agent, note=f"ticks<={args.ticks}")
        if not args.quiet:
            print(f"seed {seed:>3}  ticks {rec['ticks']:>6}  meals {rec['meals']:>3}"
                  f"  gained {rec['gained']:>7.1f}  walked {rec['distance']:>7.0f}"
                  f"  {rec['ate']}")
    print()
    print(summarise())


if __name__ == "__main__":
    main()
