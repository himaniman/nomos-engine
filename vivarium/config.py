"""Tunables for a vivarium run. A run is a pure function of (seed, Config), so
this dataclass + the seed fully determine a world's history."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # world
    size: int = 48                     # grid side (bounded, walls on the border). Sets
                                       # the spatial EXTENT (how spread out, walls, halo
                                       # scale); the population is regulated by the food
                                       # controller below, not by size.
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

    # food (the carrying-capacity knob)
    food_target: float = 0.10          # STANDING food density (fraction of open cells).
                                       # RICH on purpose: sparse food never lets random
                                       # brains bootstrap foraging (the takeoff wall).
                                       # This is the BUFFER that makes food findable.
    food_regrow: int = 4               # fixed food/tick when the controller is OFF
                                       # (food_pop_target = 0)
    food_energy: float = 38.0          # energy gained per food eaten

    # ADAPTIVE food controller (homeostasis): sow lots of food while the population
    # is small (agents still learning to forage), then throttle the flow once the
    # population is established so it settles around a target — food throughput, not
    # world size, sets carrying capacity. Set food_pop_target = 0 to use fixed
    # food_regrow instead.
    food_pop_target: int = 20          # population the food flow steers toward
    food_regrow_boot: int = 12         # generous flow while pop <= target (learning)
    food_regrow_gain: float = 3.0      # how hard the flow is throttled per agent over
                                       # target (proportional negative feedback)

    # food diversity + DRIFTING HALOS (our take on niches — different from a static
    # zoning): food carries a TYPE, and each type spawns from a HALO centre that
    # slowly DRIFTS around the map, so food is a moving resource PATCH to track, not
    # a uniform sprinkle. Concentrated patches also help random foragers bootstrap.
    n_food_types: int = 2              # distinct food types (a dietary niche axis)
    food_halo_sigma: float = 0.30      # halo radius as a fraction of the grid side
                                       # (wide enough that food is FINDABLE — tight
                                       # halos concentrate food where random brains
                                       # can't find it and takeoff stalls)
    food_halo_drift: float = 2.0       # cells a halo centre may step when it drifts
    food_halo_drift_every: int = 40    # ticks between halo drifts (0 = static halos)

    # diet niche: a heritable trait sets which food TYPE an agent digests best (1.0 on
    # its own type, down to diet_min_eff on the far one); vision dims food it can't eat
    diet_min_eff: float = 0.15         # digestion efficiency on the worst-matched type.
                                       # LOW on purpose: if off-diet food is nearly
                                       # inedible, the second food type is a real
                                       # PROTECTED niche a diet specialist can own, so
                                       # dietary diversity persists instead of everyone
                                       # converging to one food (competitive exclusion)
    trait_drift_prob: float = 0.10     # chance the diet / carnivore traits drift
    trait_drift: float = 0.05          # size of that drift (clamped to [0, 1])

    # predation (trophic niche): a heritable carnivore trait; a committed carnivore
    # that steps into another agent eats it (no corpse). Plant energy scales with
    # (1 - carnivore), so it's a real herbivore/carnivore allocation.
    predation: bool = True
    carnivore_threshold: float = 0.5   # min carnivore gene to hunt at all
    meat_energy: float = 34.0          # energy a kill yields (scaled by carnivore)

    # metabolism (energy drained per tick; standing still is cheapest). Tuned cheap
    # enough that random foragers can bootstrap (a too-harsh drain starves them before
    # evolution finds foraging — the takeoff lesson).
    base_cost: float = 0.25            # paid every tick just to exist
    move_cost: float = 0.35            # extra to step forward
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
    species_hue: float = 0.08          # mating hue tolerance = reproductive isolation
                                       # by colour; a lineage that drifts past it from
                                       # its kin becomes a new species (also the gap
                                       # that separates species in the HUD count)

    # corpse: an OBSTACLE (not food) that decays away
    corpse_decay: int = 40             # ticks a corpse blocks its cell before clearing

    # population
    init_population: int = 20           # start near the food controller's target so the
                                        # world doesn't open with a crash-and-recover
    reseed_count: int = 8              # individuals dropped when the world goes extinct
    reseed_elites: int = 3             # of those, this many are the best genomes AS-IS;
                                       # the rest are STRONGLY-mutated variants of them,
                                       # so a collapse injects fresh variation instead of
                                       # replaying the population that just died
    reseed_mutate_passes: int = 5      # compounded mutation passes for a reseed mutant
    lifespan_window: int = 400         # ticks over which the mean/median age-at-death
                                       # stat is computed (a rolling survival readout)
    # diversity: (1) a very rare MACRO-mutation jolts a child's brain hard AND jumps its
    # lineage colour, so new species/lineages branch off visibly; (2) IMMIGRATION drops
    # a few COMPETENT newcomers (drawn from the hall, "smart like the brothers") but
    # re-coloured as a fresh lineage, seeding diversity a homogeneous world won't grow.
    macro_mut_prob: float = 0.004      # chance a mutation is a big-jump speciation event
    macro_mut_scale: float = 3.0       # weight-jitter multiplier for a macro-mutation
    immigrate_every: int = 400         # ticks between immigration waves (0 = off)
    immigrate_count: int = 2           # competent newcomers per wave (new lineage colour)
    hall_offspring_weight: float = 25.0  # a child is worth this many ticks of age in
                                       # the hall-of-fame's success score (fecundity
                                       # + longevity, not longevity alone)

    # brain / senses
    brain: str = "neat"                # which brain module: "neat" (graph of scalar
                                       # neurons) or "nca" (a small cellular automaton
                                       # per agent — grid + evolvable alive-mask shape)
    n_rays: int = 5                    # vision rays fanned around the heading
    fov: float = 1.4                   # total angular spread of the rays (radians)
    ray_length: int = 12               # how far a ray sees (cells)
    hidden: int = 8                    # hidden units in the fixed-topology v1 net
    init_forward_bias: float = 1.5     # fresh brains lean toward moving forward, so
                                       # the population forages from tick 0 and
                                       # evolution has material to select on

    # NCA brain (brain="nca"): each agent's brain is a G x G multi-channel CA with
    # a heritable ALIVE-MASK (grown shape). One shared update rule (a small MLP over
    # the 3x3 neighbourhood) relaxes `nca_subticks` steps per game tick; sensors are
    # broadcast in, the action is read from the pooled cell state. Shape + rule
    # weights both evolve.
    nca_grid: int = 12                 # max brain lattice side (the mask lives here)
    nca_channels: int = 8              # state channels per cell
    nca_hidden: int = 16               # hidden units in each cell-type's update rule
    nca_subticks: int = 32             # rule relaxation steps per game tick (think time)
    nca_start: int = 2                 # side of the fresh alive block (2 -> 2x2)
    nca_max_cells: int = 48            # cap on alive cells (bounds per-agent compute)
    nca_grow_prob: float = 0.06        # chance a mutation activates a new adjacent cell
    nca_shrink_prob: float = 0.02      # chance a mutation deactivates a cell
    # v1 — cell TYPES: the genome carries several update rules; each cell has a TYPE
    # pointing at one, so cells DIFFERENTIATE (a set of rules, not one shared rule).
    nca_max_types: int = 4             # cap on distinct cell-type rules
    nca_retype_prob: float = 0.05      # chance a mutation re-types one cell
    nca_split_prob: float = 0.02       # chance a mutation SPLITS a type into a new one
                                       # (clone+jitter a rule, reassign ~half its cells)
    nca_cell_cost: float = 0.02        # energy per brain cell per tick (NCA only): makes
                                       # a bigger brain COST more, so shape/size is under
                                       # selection, not a free random drift
