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

from nomos.embodied import play, play_forage

# The brain the genome parameterizes. Its ARCHITECTURE is the one the LLM brain
# lab discovered (two non-interfering output layers + hunger-gated risk); here
# evolution tunes its constants. This is the hybrid: the model designs the
# structure, selection optimizes the parameters. `motor` (go-toward) soaks up the
# food smell, amplified by hunger; `avoid` (stay-away) soaks up body-danger,
# eased when starving so the snake dares more the hungrier it gets. The harness
# combines the two layers only at the move decision (motor-pull minus avoid-pull).
TEMPLATE = (
    "channel food\n"
    "channel danger\n"
    "channel temperature\n"
    "channel motor: init 0\n"
    "channel avoid: init 0\n\n"
    "rule go:\n"
    "  when 1\n"
    "  set motor = {w_mem}*avg8(motor) + food * (1 + {hunger_boost}*temperature)\n\n"
    "rule fear:\n"
    "  when 1\n"
    "  set avoid = {avoid_str}*danger * (1 - {avoid_relax}*temperature)\n"
)

# Gene bounds. w_mem < 1 keeps the motor field from diverging. hunger_boost = how
# hard hunger amplifies the food drive; avoid_str = base body-fear strength;
# avoid_relax in [0,1] = how much that fear is dropped at full hunger (1 = fully
# reckless when starving). Defaults start from the LLM's best brain (score 18.67).
BOUNDS = {"w_mem": (0.0, 0.9), "hunger_boost": (0.0, 4.0),
          "avoid_str": (0.0, 3.0), "avoid_relax": (0.0, 1.0)}
DEFAULT_GENOME = {"w_mem": 0.3, "hunger_boost": 2.0,
                  "avoid_str": 1.5, "avoid_relax": 0.3}

# The FORAGE genome: the constants of the LLM-discovered sight champion
# (laws/forage_brain_llm.nomos). Evolution tunes what the model set by hand:
# `mem_rate` = how fast the smell integrator tracks the food gradient;
# `mem_gain` = its weight in the motor; `noise_gain` = hunger-scaled
# exploration; `sight_gain` = how hard SEEN food dominates (the champion used
# 3); `wall_gain` = touch-fear strength. Same hybrid recipe as the snake
# TEMPLATE: LLM designs the structure, selection optimizes the numbers.
FORAGE_TEMPLATE = (
    "channel food\n"
    "channel wall\n"
    "channel sight\n"
    "channel temperature\n"
    "channel motor\n"
    "channel avoid\n"
    "channel mem: init 0\n"
    "channel noise: init 0\n\n"
    "rule integrate_mem:\n"
    "  when 1\n"
    "  add mem = {mem_rate} * (food - mem)\n\n"
    "rule add_noise:\n"
    "  when 1\n"
    "  add noise = {noise_gain} * (rand() - 0.5) * temperature\n\n"
    "rule set_motor:\n"
    "  when 1\n"
    "  set motor = max(0, food + {mem_gain}*mem + noise + {sight_gain}*sight)\n\n"
    "rule set_avoid:\n"
    "  when 1\n"
    "  set avoid = {wall_gain} * wall\n"
)
FORAGE_BOUNDS = {"mem_rate": (0.02, 0.6), "mem_gain": (0.0, 3.0),
                 "noise_gain": (0.0, 2.0), "sight_gain": (0.0, 6.0),
                 "wall_gain": (0.0, 3.0)}
FORAGE_DEFAULT = {"mem_rate": 0.2, "mem_gain": 1.0, "noise_gain": 0.5,
                  "sight_gain": 3.0, "wall_gain": 1.0}

# Per-game config: template + gene space + play fn + default sim sizes. `evolve`
# and `fitness` select on this, mirroring brain_lab.GAMES.
GAMES = {
    "snake": {"template": TEMPLATE, "bounds": BOUNDS, "default": DEFAULT_GENOME,
              "play": play, "world_size": 48, "game_size": 16, "moves": 600,
              "ticks": 64},
    "forage": {"template": FORAGE_TEMPLATE, "bounds": FORAGE_BOUNDS,
               "default": FORAGE_DEFAULT, "play": play_forage, "world_size": 128,
               "game_size": 96, "moves": 400, "ticks": 32},
}


def law_from_genome(genome: dict, game: str = "snake") -> str:
    tmpl = GAMES[game]["template"]
    return tmpl.format(**{k: round(float(v), 4) for k, v in genome.items()})


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def mutate(genome: dict, rng: random.Random, strength: float = 0.15,
           game: str = "snake") -> dict:
    """Gaussian perturbation of every gene, kept within bounds."""
    out = {}
    for k, (lo, hi) in GAMES[game]["bounds"].items():
        out[k] = _clip(genome[k] + rng.gauss(0.0, strength * (hi - lo)), lo, hi)
    return out


def fitness(genome: dict, seeds=(0, 1, 2), world_size: int = 48, game_size: int = 16,
            moves: int = 600, ticks: int = 64, game: str = "snake") -> dict:
    """Average game performance of a brain over several boards, each played until
    death (or the `moves` cap). Fitness rewards eating first (score, heavily
    weighted) and surviving second (a small term — enough to give a gradient
    before the first food and to break ties, but not so much that merely
    surviving without eating outscores actually playing)."""
    play_fn = GAMES[game]["play"]
    law = law_from_genome(genome, game)
    scores, survivals, immortal = [], [], 0
    for s in seeds:
        r = play_fn(law, world_size=world_size, game_size=game_size, seed=s,
                    moves=moves, ticks_per_move=ticks)
        scores.append(r["score"])
        survivals.append(r["steps_alive"])
        immortal += int(r["alive_at_end"])
    score = sum(scores) / len(scores)
    survival = sum(survivals) / len(survivals)
    return {"fitness": score * 100.0 + survival * 0.1, "score": score,
            "survival": survival, "scores": scores, "immortal": immortal}


def evolve(generations: int, lam: int = 6, seeds=(0, 1, 2), world_size: int = 48,
           game_size: int = 16, moves: int = 600, ticks: int = 64, rng_seed: int = 0,
           start: dict | None = None, on_generation=None, game: str = "snake") -> dict:
    """(1+lambda) evolution strategy over brain genomes. Each generation makes
    `lam` mutants of the current parent, evaluates them, and keeps the best
    (elitist — the parent survives if none beat it). `on_generation(gen, genome,
    fit)` is called with the generation's best, e.g. to render a GIF."""
    rng = random.Random(rng_seed)
    parent = dict(start) if start else dict(GAMES[game]["default"])
    parent_fit = fitness(parent, seeds, world_size, game_size, moves, ticks, game)
    best = (parent, parent_fit)
    history = []

    for gen in range(generations):
        pool = [(parent, parent_fit)]
        for _ in range(lam):
            child = mutate(parent, rng, game=game)
            pool.append((child, fitness(child, seeds, world_size, game_size,
                                        moves, ticks, game)))
        parent, parent_fit = max(pool, key=lambda t: t[1]["fitness"])
        if parent_fit["fitness"] > best[1]["fitness"]:
            best = (parent, parent_fit)
        history.append({"gen": gen, "genome": dict(parent),
                        "fitness": parent_fit["fitness"], "score": parent_fit["score"],
                        "survival": parent_fit["survival"]})
        if on_generation is not None:
            on_generation(gen, parent, parent_fit)

    return {"best_genome": best[0], "best_fitness": best[1], "history": history}
