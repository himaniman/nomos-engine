"""Tunables for a vivarium run. A run is a pure function of (seed, Config), so
this dataclass + the seed fully determine a world's history."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # world
    size: int = 96                     # grid side (bounded, walls on the border)
    seed: int = 0

    # food (the carrying-capacity knob)
    food_target: float = 0.06          # fraction of open cells that should hold food
    food_regrow: int = 6               # at most this many food items appear per tick
    food_energy: float = 22.0          # energy gained per food eaten

    # metabolism (energy drained per tick; standing still is cheapest)
    base_cost: float = 0.35            # paid every tick just to exist
    move_cost: float = 0.5             # extra to step forward
    turn_cost: float = 0.15            # extra to turn
    start_energy: float = 30.0
    max_energy: float = 60.0

    # reproduction (asexual clone+mutate in v1; crossover comes with NEAT)
    repro_min_energy: float = 45.0     # must be at least this well-fed to breed
    repro_cost: float = 25.0           # energy spent, split into the child
    mutation: float = 0.1              # gaussian weight-mutation strength

    # corpse: an OBSTACLE (not food) that decays away
    corpse_decay: int = 40             # ticks a corpse blocks its cell before clearing

    # population
    init_population: int = 40
    reseed_count: int = 6              # individuals dropped when the world goes extinct

    # brain / senses
    n_rays: int = 5                    # vision rays fanned around the heading
    fov: float = 1.4                   # total angular spread of the rays (radians)
    ray_length: int = 12               # how far a ray sees (cells)
    hidden: int = 8                    # hidden units in the fixed-topology v1 net
