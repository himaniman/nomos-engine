"""Tunables for a vivarium run. A run is a pure function of (seed, Config), so
this dataclass + the seed fully determine a world's history.

Defaults tuned for a STABLE ECOLOGY (2026-07-18): fewer, longer-lived agents that
breed true. The levers, and why:
  - LOW mutation (weight_mut*/structural rates) so a child resembles its parent —
    heritability is what lets good foragers pass on their skill, stable lineages
    form, and species persist. High mutation re-rolls the population each
    generation (no heredity -> no species -> short lives).
  - RICH but RARE food (big food_energy, big energy reserves, gentle food flow) so
    a meal lasts, an agent banks energy and lives long, and the population settles
    SMALL (~10-20) on food throughput rather than crashing and reseeding.
  - immigration OFF and macro-mutation rare, so the "species" on the HUD are real
    coexisting lineages, not transient random newcomers that die in a few ticks.
"""

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
    food_target: float = 0.04          # STANDING food density (fraction of open cells) —
                                       # the CAP on how much food may sit at once. A
                                       # findable BUFFER, kept modest so the world isn't
                                       # carpeted — food is worth seeking.
    food_regrow: int = 5               # FIXED food sown per tick (the controller is OFF
                                       # by default — see food_pop_target). A constant
                                       # rate: no population-driven "waves", food just
                                       # rains at a steady pace up to the standing cap.
    food_energy: float = 70.0          # energy gained per food eaten — RICH: a single
                                       # meal funds a long stretch of life, so agents
                                       # survive long between finds and the population
                                       # stays small without starving out

    # ADAPTIVE food controller (homeostasis) — OFF by default (food_pop_target = 0), so
    # food flows at the FIXED food_regrow rate above. When ON (food_pop_target > 0) it
    # sows food while the population is at/under target, then throttles the flow as it
    # climbs past — but that couples the food rate to the population, which reads as
    # "waves". Leave it off for a steady rate; turn it on for population homeostasis.
    food_pop_target: int = 0           # 0 = OFF (fixed rate). >0 = steer pop toward this.
    food_regrow_boot: int = 10         # steady flow while pop <= target (controller ON)
    food_regrow_gain: float = 1.0      # throttle per agent over target (controller ON)

    # food diversity + DRIFTING HALOS (our take on niches — different from a static
    # zoning): food carries a TYPE, and each type spawns from a HALO centre that
    # slowly DRIFTS around the map, so food is a moving resource PATCH to track, not
    # a uniform sprinkle. Concentrated patches also help random foragers bootstrap.
    n_food_types: int = 2              # distinct food types (a dietary niche axis)
    food_halo_sigma: float = 0.11      # halo radius as a fraction of the grid side.
                                       # TIGHT: food forms real PATCHES (a wide sigma
                                       # sprinkles food uniformly and the halo means
                                       # nothing — then there's no niche to specialize
                                       # into); tight patches make the two food types
                                       # real places to own
    food_halo_drift: float = 2.0       # cells a halo centre may step when it drifts
    food_halo_drift_every: int = 1280  # ticks between halo drifts (0 = static halos) —
                                       # very slow, so food patches feel like stable
                                       # regions that only wander over long timescales

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
    # evolution finds foraging — the takeoff lesson). The STAY action pays only
    # base_cost, so an idle agent conserves energy (parking to digest is a strategy).
    base_cost: float = 0.25            # paid every tick just to exist
    move_cost: float = 0.35            # extra to step forward
    turn_cost: float = 0.15            # extra to turn
    start_energy: float = 55.0
    max_energy: float = 130.0          # big reserve: a well-fed agent banks a long life
                                       # (raised so agents live long enough to LEARN)

    # reproduction: sexual crossover when two fed agents are adjacent, else a
    # high-energy asexual fallback so lone pioneers can still colonize. Thresholds
    # scaled to the larger energy reserve, so breeding takes real investment (slower
    # breeding -> a smaller, older population).
    repro_min_energy: float = 80.0     # must be at least this well-fed to breed — high so
                                       # an agent LIVES (and learns) a good while before
                                       # spending energy on a child
    repro_cost: float = 42.0           # energy spent (split across parents/child)
    asexual_energy: float = 110.0      # lone-agent fallback breeds only when this full

    # NEAT mutation rates (applied to every child genome). LOW, so a child resembles
    # its parent — heritability is what makes lineages/species persist and lifespan
    # climb (see module docstring).
    weight_mut: float = 0.12           # gaussian std of weight jitter
    weight_mut_prob: float = 0.30      # chance each connection is jittered
    add_conn_prob: float = 0.03        # chance to add a new connection
    add_node_prob: float = 0.015       # chance to split a connection into a node
    toggle_prob: float = 0.01          # chance to flip a connection on/off
    hue_drift_prob: float = 0.08       # chance the heritable lineage colour drifts
    hue_drift: float = 0.02            # size of that colour drift (wraps at 1.0)
    species_hue: float = 0.08          # mating hue tolerance = reproductive isolation
                                       # by colour; a lineage that drifts past it from
                                       # its kin becomes a new species (also the gap
                                       # that separates species in the HUD count)

    # corpse: an OBSTACLE (not food) that decays away
    corpse_decay: int = 40             # ticks a corpse blocks its cell before clearing

    # population
    init_population: int = 18           # start near the food controller's target so the
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
    # lineage colour, so new species/lineages branch off visibly; (2) IMMIGRATION is OFF
    # by default — it dropped competent-but-randomly-recoloured newcomers that read as
    # fresh "species" on the HUD but died fast, inflating the species count while
    # dragging lifespan down. Real diversity should come from NICHES (food patches +
    # diet), not injection. Set immigrate_every > 0 to re-enable invasion experiments.
    macro_mut_prob: float = 0.001      # chance a mutation is a big-jump speciation event
    macro_mut_scale: float = 3.0       # weight-jitter multiplier for a macro-mutation
    immigrate_every: int = 0           # ticks between immigration waves (0 = OFF)
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
    nca_grow_prob: float = 0.03        # chance a mutation activates a new adjacent cell
    nca_shrink_prob: float = 0.01      # chance a mutation deactivates a cell
    # v1 — cell TYPES: the genome carries several update rules; each cell has a TYPE
    # pointing at one, so cells DIFFERENTIATE (a set of rules, not one shared rule).
    nca_max_types: int = 4             # cap on distinct cell-type rules
    nca_retype_prob: float = 0.025     # chance a mutation re-types one cell
    nca_split_prob: float = 0.01       # chance a mutation SPLITS a type into a new one
                                       # (clone+jitter a rule, reassign ~half its cells)
    # v2 — evolvable DEPTH: a type's rule is a stack of layers; mutation grows/shrinks
    # its depth (add a near-identity layer / remove a middle one), capped.
    nca_max_layers: int = 4            # max layers in a cell-type rule (depth cap)
    nca_add_layer_prob: float = 0.015  # chance a mutation deepens a rule by one layer
    nca_del_layer_prob: float = 0.01   # chance a mutation removes a middle layer
    nca_cell_cost: float = 0.02        # energy per brain cell per tick (NCA only): makes
                                       # a bigger brain COST more, so shape/size is under
                                       # selection, not a free random drift

    # LIFETIME LEARNING (meta-plasticity, NCA only) — ON by default: it's the point of
    # the NCA brain. Normally the weights are FIXED at birth and only the cell state
    # changes; with this on, the motor readout also learns WITHIN a life by reward-
    # modulated (three-factor) Hebbian plasticity: an eligibility TRACE remembers the
    # last few (pooled-state, action) decisions, and when the agent EATS (reward), those
    # decisions' readout weights are reinforced in a plastic delta added to P_out. So a
    # FIXED genome gets better at foraging as it ages, which should help TAKEOFF (a
    # random brain can improve inside one life instead of only across generations). The
    # learned delta is NOT inherited (Darwinian, not Lamarckian — evolution tunes the
    # ABILITY to learn, not the learned weights). The flag stays only for the A/B
    # experiment (plastic vs frozen); leave it True for normal runs.
    nca_plastic: bool = True           # within-life learning of the motor readout
    nca_plastic_lr: float = 0.03       # learning rate of the reward-gated update
    nca_plastic_forget: float = 0.002  # per-tick decay of the plastic delta (slow)
    nca_plastic_elig: float = 0.5      # per-act decay of the eligibility trace (credit
                                       # reaches back ~1/(1-elig) decisions before a meal)
    nca_plastic_clip: float = 1.0      # bound on |plastic delta| (keeps it from running off)

    # EXPLORATION (NCA only): a recurrent brain fed a near-constant input settles into a
    # fixed point and FREEZES — same action forever, so it stops moving, stops eating,
    # and plasticity has no reward to amplify (the real takeoff wall). A little
    # stochasticity in the action choice breaks the fixed point: the agent keeps moving,
    # its input changes, it finds food. Actions are sampled from softmax(logits / temp)
    # via the world's SEEDED rng, so runs stay reproducible. 0 = greedy argmax (off).
    nca_explore_temp: float = 0.2      # softmax temperature for the action choice.
                                       # A/B (populated preset): temp 0 gave the SHORTEST
                                       # life and the MOST extinctions (neighbours alone
                                       # don't unstick a frozen brain); a little here
                                       # lengthens life and cuts extinctions.
