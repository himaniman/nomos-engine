"""Lamarckian evolution of a learning brain: within-life Hebbian learning PLUS
between-life inheritance of the *learned weights themselves*, carried forward
with small mutations.

Unlike nomos/evolve.py (which evolves a handful of CONSTANTS in a law), here the
genome is the brain's whole per-cell weight FIELD. Each life the brain also
learns online (the DSL learning rule reshapes the weights during the run); we
keep those learned changes and pass them to the next life — Lamarckian: acquired
characteristics are inherited. A (1+lambda) strategy adds variation (Gaussian
mutation of the carried weights) and selection (keep the best life's weights).
Mazes are rotated across lives so the lineage learns maze-GENERAL skill, not one
board.

The offspring of a generation are independent, so they are evaluated in
parallel (ProcessPoolExecutor). Determinism: given the same start, seeds,
mutation strength and rng seed, the lineage reproduces (mutation draws from a
seeded numpy Generator; each life is a pure function of its inputs).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from nomos.embodied import play_forage

# The LM-cell brain's weight channels: the fixed random hidden projection AND the
# learned readout. BOTH must travel together — a readout is only meaningful
# against the projection it learned on, so carrying one without the other is
# noise. (laws/forage_brain_lmcell.nomos.)
LMCELL_WEIGHTS = ("wh0_food", "wh0_sight", "wh0_wall",
                  "wh1_food", "wh1_sight", "wh1_wall",
                  "vm0", "vm1", "va0", "va1")


def _run_life(job: dict) -> dict:
    """One life (a forage run) with injected start weights; returns score and the
    final weight fields. Top-level so ProcessPoolExecutor can pickle it."""
    r = play_forage(job["law"], world_size=job["world_size"],
                    game_size=job["game_size"], seed=job["seed"],
                    moves=job["moves"], ticks_per_move=job["ticks"],
                    init_channels=job["init_channels"],
                    return_channels=tuple(job["weights"]))
    return {"score": r["score"], "steps_alive": r["steps_alive"],
            "alive_at_end": r["alive_at_end"], "weights": r["final_channels"]}


def _mutate(weights: dict, rng: np.random.Generator, strength: float) -> dict:
    """Add small Gaussian noise to every carried weight field — light random
    changes between passes, so the lineage explores around what it has learned."""
    return {k: v + rng.normal(0.0, strength, size=v.shape) for k, v in weights.items()}


def life_fitness(r: dict, moves: int) -> float:
    """Selection value of a life: meals dominate (x100), but a LONG life is
    well rewarded — survived ticks (x0.1) plus a flat bonus for going the whole
    distance (immortal). Eating still beats mere survival (one meal = 100 > the
    ~90 a full immortal life earns), so this rewards longevity without
    reopening the flee-food-and-circle-forever loophole."""
    surv = r["steps_alive"] * 0.1 + (40.0 if r.get("alive_at_end") else 0.0)
    return r["score"] * 100.0 + surv


def lamarckian(law_text: str, lives: int = 12, lam: int = 4,
               weight_channels=LMCELL_WEIGHTS, seeds=(0, 1, 2, 3),
               mutation: float = 0.02, world_size: int = 128, game_size: int = 96,
               moves: int = 400, ticks: int = 32, rng_seed: int = 0,
               workers: int | None = None, on_generation=None) -> dict:
    """Run a Lamarckian lineage for `lives` generations. Each generation: mutate
    the parent's carried weights `lam` ways, run each a life on the generation's
    maze (rotated through `seeds`), keep the best life's FINAL (learned) weights
    as the next parent. Returns the best weights and the per-generation history.
    `on_generation(gen, best)` is called each generation, e.g. to log or render."""
    rng = np.random.default_rng(rng_seed)

    # Life 0: no inherited weights — the law's own random init seeds the lineage.
    base = _run_life({"law": law_text, "world_size": world_size, "game_size": game_size,
                      "seed": seeds[0], "moves": moves, "ticks": ticks,
                      "init_channels": None, "weights": weight_channels})
    parent_weights = base["weights"]
    parent_fit = life_fitness(base, moves)
    best = {"score": base["score"], "steps_alive": base["steps_alive"],
            "fitness": parent_fit, "weights": parent_weights}
    history = [{"gen": 0, "score": base["score"], "steps_alive": base["steps_alive"],
                "best_score": best["score"]}]
    if on_generation is not None:
        on_generation(0, history[-1])

    for gen in range(1, lives):
        seed = seeds[gen % len(seeds)]
        children = [_mutate(parent_weights, rng, mutation) for _ in range(lam)]
        jobs = [{"law": law_text, "world_size": world_size, "game_size": game_size,
                 "seed": seed, "moves": moves, "ticks": ticks,
                 "init_channels": c, "weights": weight_channels} for c in children]
        if workers and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_run_life, jobs))
        else:
            results = [_run_life(j) for j in jobs]

        # elitist: the best child (meals, then long life) passes its LEARNED
        # weights to the next generation
        top = max(results, key=lambda r: life_fitness(r, moves))
        top_fit = life_fitness(top, moves)
        if top_fit >= parent_fit * 0.5:             # accept unless a big collapse
            parent_weights = top["weights"]
            parent_fit = top_fit
        if top_fit > best["fitness"]:
            best = {"score": top["score"], "steps_alive": top["steps_alive"],
                    "fitness": top_fit, "weights": top["weights"]}
        history.append({"gen": gen, "seed": seed, "score": top["score"],
                        "steps_alive": top["steps_alive"], "best_score": best["score"]})
        if on_generation is not None:
            on_generation(gen, history[-1])

    return {"best": best, "history": history, "final_weights": parent_weights}
