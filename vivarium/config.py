"""Tunables for a vivarium run. A run is a pure function of (seed, Config), so
this dataclass + the seed fully determine a world's history."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # world
    size: int = 96                     # grid side (bounded, walls on the border)
    seed: int = 0
    wall_density: float = 0.05         # fraction of the interior that is INNER wall
                                       # (strokes with doorways) — mostly open space
                                       # with occasional walls; vision rays stop at
                                       # walls so agents can't see food behind them
    # "living" landscape: each wall STROKE keeps its shape but slowly WALKS along
    # its own line (like a snake that can lead with either end), sliding back and
    # forth. The maze's topology stays roughly the same; only positions drift.
    # drift_every=0 freezes the walls.
    wall_drift_every: int = 120        # a walk opportunity every N ticks (slow)
    wall_step_prob: float = 0.5        # chance each stroke steps on a walk tick

    # food (the carrying-capacity knob). Richer than a bare world because niches
    # (diet specialisation + spatial zones) make foraging harder — without this
    # the population can't bootstrap from random brains once niches are on.
    food_target: float = 0.07          # fraction of open cells that should hold food
    food_regrow: int = 8               # at most this many food items appear per tick
    food_energy: float = 30.0          # energy gained per food eaten

    # metabolism (energy drained per tick; standing still is cheapest)
    base_cost: float = 0.35            # paid every tick just to exist
    move_cost: float = 0.5             # extra to step forward
    turn_cost: float = 0.15            # extra to turn
    start_energy: float = 30.0
    max_energy: float = 60.0

    # reproduction: sexual crossover when two fed agents are adjacent, else a
    # high-energy asexual fallback so lone pioneers can still colonize.
    repro_min_energy: float = 45.0     # must be at least this well-fed to breed
    repro_cost: float = 25.0           # energy spent (split across parents/child)
    asexual_energy: float = 58.0       # lone-agent fallback breeds only when this full

    # NEAT mutation rates (applied to every child genome)
    weight_mut: float = 0.4            # gaussian std of weight jitter
    weight_mut_prob: float = 0.85      # chance each connection is jittered
    add_conn_prob: float = 0.06        # chance to add a new connection
    add_node_prob: float = 0.03        # chance to split a connection into a node
    toggle_prob: float = 0.02          # chance to flip a connection on/off
    hue_drift_prob: float = 0.08       # chance the heritable lineage colour drifts
    hue_drift: float = 0.02            # size of that colour drift (wraps at 1.0)
    trait_drift_prob: float = 0.10     # chance the diet / carnivore niche traits drift
    trait_drift: float = 0.05          # size of that drift (clamped to [0, 1])

    # niches: food types + diet
    n_food_types: int = 2              # distinct food types (a dietary niche axis)
    diet_min_eff: float = 0.35         # digestion efficiency on your WORST food type
                                       # (a real trade-off, but soft enough to bootstrap)
    # niches: predation (trophic)
    predation: bool = True             # carnivores can eat other agents
    carnivore_threshold: float = 0.5   # min carnivore gene to hunt at all
    meat_energy: float = 34.0          # energy a kill yields (scaled by carnivore)
    mate_within_species: bool = False  # if True, mating is restricted to same-colour
                                       # partners (reproductive isolation). OFF by
                                       # default: in a homogeneous world it just
                                       # collapses to one species (no niches yet).
    species_hue: float = 0.08          # colour distance that separates species — the
                                       # HUD count gap, and (when mate_within_species)
                                       # the mating tolerance

    # corpse: an OBSTACLE (not food) that decays away
    corpse_decay: int = 40             # ticks a corpse blocks its cell before clearing

    # population
    init_population: int = 40
    reseed_count: int = 6              # individuals dropped when the world goes extinct
    # immigration: rare fresh random-ish newcomers dropped into a LIVING world.
    # Keeps diversity from collapsing, tests whether fresh brains beat evolved
    # ones, and — crucially — introduces variants (e.g. carnivores) once the world
    # is populated enough for them to establish, which they can't during bootstrap.
    immigrate_every: int = 300         # ticks between immigration waves (0 = off)
    immigrate_count: int = 3           # newcomers per wave
    hall_offspring_weight: float = 25.0  # a child is worth this many ticks of age in
                                       # the hall-of-fame's success score (fecundity
                                       # + longevity, not longevity alone)

    # brain / senses
    n_rays: int = 5                    # vision rays fanned around the heading
    fov: float = 1.4                   # total angular spread of the rays (radians)
    ray_length: int = 12               # how far a ray sees (cells)
    hidden: int = 8                    # hidden units in the fixed-topology v1 net
    init_forward_bias: float = 1.5     # fresh brains lean toward moving forward, so
                                       # the population forages from tick 0 and
                                       # evolution has material to select on
