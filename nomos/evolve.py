"""Phase B: evolve a Snake brain across runs (one brain per run, selection
between runs). Fitness is the game score; the genome is a handful of weights
substituted into a brain-law template, and a (1+lambda) evolution strategy
mutates them. This is the smallest thing that makes the grid *learn* to play:
no gradients, just heredity + variation + selection, the same recipe as the
closed-world Experiments 3-4 but with the reward coming from an external game.

Deterministic: given (start genome, seeds, sizes, cadence, rng seed) the whole
run reproduces, so a generation can be re-rendered or re-checked exactly.
"""

from __future__ import annotations

import random

from nomos.embodied import play

# The brain the genome parameterizes. It reads the food sensor (attraction) and
# the body/danger sensor (repulsion), integrating them into a diffusing motor
# field whose centre of mass the harness turns into a move. The weights are the
# genes: how much memory, how strongly food pulls, how strongly the body repels.
TEMPLATE = (
    "channel food\n"
    "channel danger\n"
    "channel motor: init 0\n\n"
    "rule think:\n"
    "  when 1\n"
    "  set motor = {w_mem}*avg8(motor) + {w_food}*food - {w_danger}*danger\n"
)

# Gene bounds. w_mem < 1 keeps the motor field from diverging.
BOUNDS = {"w_mem": (0.0, 0.95), "w_food": (0.0, 3.0), "w_danger": (0.0, 3.0)}
DEFAULT_GENOME = {"w_mem": 0.5, "w_food": 1.0, "w_danger": 1.0}


def law_from_genome(genome: dict) -> str:
    return TEMPLATE.format(**{k: round(float(v), 4) for k, v in genome.items()})


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def mutate(genome: dict, rng: random.Random, strength: float = 0.15) -> dict:
    """Gaussian perturbation of every gene, kept within bounds."""
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        out[k] = _clip(genome[k] + rng.gauss(0.0, strength * (hi - lo)), lo, hi)
    return out


def fitness(genome: dict, seeds=(0, 1, 2), world_size: int = 48, game_size: int = 16,
            moves: int = 250, ticks: int = 4) -> dict:
    """Average game performance of a brain over several boards. Fitness rewards
    eating first (score, heavily weighted) and surviving second, so there is a
    gradient to climb even before a brain scores its first food."""
    law = law_from_genome(genome)
    scores, survivals = [], []
    for s in seeds:
        r = play(law, world_size=world_size, game_size=game_size, seed=s,
                 moves=moves, ticks_per_move=ticks)
        scores.append(r["score"])
        survivals.append(r["steps_alive"])
    score = sum(scores) / len(scores)
    survival = sum(survivals) / len(survivals)
    return {"fitness": score * 100.0 + survival, "score": score, "survival": survival,
            "scores": scores}


def evolve(generations: int, lam: int = 6, seeds=(0, 1, 2), world_size: int = 48,
           game_size: int = 16, moves: int = 250, ticks: int = 4, rng_seed: int = 0,
           start: dict | None = None, on_generation=None) -> dict:
    """(1+lambda) evolution strategy over brain genomes. Each generation makes
    `lam` mutants of the current parent, evaluates them, and keeps the best
    (elitist — the parent survives if none beat it). `on_generation(gen, genome,
    fit)` is called with the generation's best, e.g. to render a GIF."""
    rng = random.Random(rng_seed)
    parent = dict(start) if start else dict(DEFAULT_GENOME)
    parent_fit = fitness(parent, seeds, world_size, game_size, moves, ticks)
    best = (parent, parent_fit)
    history = []

    for gen in range(generations):
        pool = [(parent, parent_fit)]
        for _ in range(lam):
            child = mutate(parent, rng)
            pool.append((child, fitness(child, seeds, world_size, game_size, moves, ticks)))
        parent, parent_fit = max(pool, key=lambda t: t[1]["fitness"])
        if parent_fit["fitness"] > best[1]["fitness"]:
            best = (parent, parent_fit)
        history.append({"gen": gen, "genome": dict(parent),
                        "fitness": parent_fit["fitness"], "score": parent_fit["score"],
                        "survival": parent_fit["survival"]})
        if on_generation is not None:
            on_generation(gen, parent, parent_fit)

    return {"best_genome": best[0], "best_fitness": best[1], "history": history}
