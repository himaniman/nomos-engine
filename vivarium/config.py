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
    wall_thickness: float = 1.0        # wall bar thickness in cells (continuous aqua
                                       # world): strokes become float RECTANGLES this
                                       # thick, collided/seen analytically. 1.0 keeps
                                       # the old one-cell look
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
    food_radius: float = 0.25          # food morsel radius in cells (continuous aqua):
                                       # food is a float CIRCLE eaten when an agent body
                                       # overlaps it; also its drawn size

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
    food_halo_sigma: float = 0.12      # halo radius as a fraction of the grid side.
                                       # TIGHT: food forms real PATCHES (a wide sigma
                                       # sprinkles food uniformly and the halo means
                                       # nothing — then there's no niche to specialize
                                       # into); tight patches make the two food types
                                       # real places to own
    food_scatter_prob: float = 0.1     # chance a sown food ignores the halo and lands
                                       # on a RANDOM free cell anywhere — a little
                                       # background dispersal so the map isn't bare
                                       # between patches (0 = strictly halo-local)
    food_halo_drift: float = 2.0       # cells a halo centre may step when it drifts
    food_halo_drift_every: int = 320   # ticks between halo drifts (0 = static halos) —
                                       # food patches wander at a visible pace (patches
                                       # are a moving resource to track, not fixed spots)

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
    max_energy: float = 800.0          # HARD ceiling only. The real regulator is now the
                                       # GLUTTONY BRAKE below — a soft behavioural cap, not
                                       # a numeric clamp (was 130)

    # GLUTTONY BRAKE (soft energy cap): a well-fed agent moves SLOWER — its commanded speed
    # is scaled by a factor that is 1 below glut_start, eases in gently (flat onset), and
    # reaches 0 (full stop) at glut_full, on an exponential curve. So hoarding energy is
    # self-limiting: a glutton can neither forage nor reach a mate and bleeds back down.
    # This replaces the old hard energy clamp and makes lean-mobile-and-breed the winning
    # strategy (hoard-and-clone becomes a rare dead-end).
    glut_start: float = 150.0          # speed unaffected up to here
    glut_full: float = 800.0           # fully stopped at/above here
    glut_k: float = 4.0                # curve sharpness (bigger = stays fast longer, then plunges)
    energy_sense_scale: float = 150.0  # normaliser for the energy proprioception input, so the
                                       # sense stays meaningful in the working range instead of
                                       # being squashed against the high max_energy ceiling

    # reproduction: sexual CROSSOVER is the main path (two fed same-species agents
    # adjacent, fitter leads); asexual budding is now a RARE fallback for the very
    # overfed only (see clone_* below), so crossover dominates and hoard-and-clone is a
    # dead-end. Thresholds scaled to the energy reserve, so breeding takes real investment.
    repro_min_energy: float = 80.0     # must be at least this well-fed to breed — high so
                                       # an agent LIVES (and learns) a good while before
                                       # spending energy on a child
    repro_cost: float = 42.0           # energy spent (split across parents/child)
    asexual_energy: float = 110.0      # lone-agent asexual fallback threshold (grid world +
                                       # the aqua BRIDGE until the REPRODUCE command lands)
    # PLANNED (wired in with the REPRODUCE brain command): once agents can actively seek and
    # commit to mating, the asexual path above becomes a RARE bud for the very-overfed only,
    # so crossover dominates and hoard-and-clone is a dead-end. Unused until then.
    clone_min_energy: float = 400.0    # only bud when this overfed (deep in the gluttony zone)
    clone_prob: float = 0.02           # and only this fraction of eligible ticks

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
    corpse_decay: int = 40             # ticks a corpse blocks (its cell in the grid world;
                                       # its float circle in the continuous aqua world)
    corpse_radius: float = 0.4         # corpse morsel radius in cells (continuous aqua):
                                       # a dead body is a float obstacle circle this big

    # SIGNAL FIELD (communication): agents deposit a PHEROMONE into a shared scalar field
    # via the EMIT output; it diffuses and decays, and every agent senses its local gradient
    # (the SIGNAL sense). Meaning is NOT hardwired — evolution decides when to emit and how
    # to react, so alarm calls / trails / aggregation can EMERGE. Emitting costs a little
    # energy so signalling stays honest (a signal must earn its keep).
    signal_decay: float = 0.88         # fraction of field value kept each tick (fades away)
    signal_diffuse: float = 0.25       # blend toward the 4-neighbour average (spreads it)
    emit_scale: float = 1.0            # deposit amount at full EMIT intensity (sigmoid gated)
    emit_cost: float = 0.05            # energy per unit emitted (keeps signalling honest)

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
    n_rays: int = 5                    # vision rays fanned around the heading (grid world;
                                       # the SEED default for aqua, where it's heritable)
    fov: float = 1.4                   # total angular spread of the rays (radians)
    ray_length: int = 12               # how far a ray sees (cells) — grid world / the aqua
                                       # SEED default (heritable there)
    # HERITABLE VISION (continuous aqua world, vivarium.vision): each creature evolves how
    # many rays it fans and how far they reach, within these bounds. The brain input is
    # sized for n_rays_MAX ray slots; a genome using fewer leaves the rest zero. Ray signal
    # falls off with distance, so a long ray only makes far things register faintly — a
    # far-sighted / sharp-but-short trade-off evolution can settle, bounded so it stays sane
    # (more/longer rays cost compute per agent, nothing else).
    n_rays_min: int = 3                # fewest rays a genome may fan
    n_rays_max: int = 12               # most rays (also the fixed brain-input width)
    ray_length_min: int = 6            # shortest heritable vision reach (cells)
    ray_length_max: int = 24           # longest heritable vision reach (cells)
    vision_mut_prob: float = 0.12      # chance a child's ray count / reach drifts
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
    nca_start: int = 3                 # side of the fresh alive block (3 -> 3x3, sensory
                                       # organs on the PERIMETER, an effector at the centre).
                                       # Small on purpose: cells share simple per-TYPE rules,
                                       # so intelligence must CO-EVOLVE from spatial growth,
                                       # not start with a big pre-made sheet.
    # Structural mutation rates are GROWTH-BIASED on purpose (grow/deepen/widen/split
    # run ~3-5x more often than their shrink counterparts): mutation proposes bigger
    # brains, and the per-cell energy cost (nca_cell_cost) is the counterweight that
    # prunes growth that doesn't pay — so brain volume trends UP under mutation and
    # is trimmed by selection, not by the mutation operator itself.
    nca_max_cells: int = 48            # cap on alive cells (bounds per-agent compute)
    nca_grow_prob: float = 0.10        # chance a mutation activates a new adjacent cell
    nca_shrink_prob: float = 0.02      # chance a mutation deactivates a cell
    # v1 — cell TYPES: the genome carries several update rules; each cell has a TYPE
    # pointing at one, so cells DIFFERENTIATE (a set of rules, not one shared rule).
    nca_max_types: int = 4             # cap on distinct cell-type rules
    nca_retype_prob: float = 0.04      # chance a mutation re-types one cell
    nca_split_prob: float = 0.03       # chance a mutation SPLITS a type into a new one
                                       # (clone+jitter a rule, reassign ~half its cells)
    # LOCAL I/O — sensory ORGANS + effector readout: the sensor drive enters only cells
    # flagged as input organs, the action is read only from effector cells, so signal must
    # travel organ -> tissue -> effector and the brain SHAPE does real computational work.
    # These flip a random alive cell's in/out role, so organ placement evolves. A brain is
    # born with every cell BOTH (== old broadcast/mean-pool) and differentiates from there.
    # organ APPEARANCE is biased over disappearance (like cell growth vs shrink): a mutation
    # adds a sensory/effector organ ~3x more often than it removes one, so brains trend
    # toward RICHER sensing/actuation and selection (via the per-cell cost) prunes what
    # doesn't pay — not the mutation operator.
    nca_add_in_prob: float = 0.06      # chance a mutation turns an alive cell INTO a sensory organ
    nca_del_in_prob: float = 0.02      # chance a mutation removes a sensory organ (rarer)
    nca_add_out_prob: float = 0.06     # chance a mutation turns an alive cell INTO an effector
    nca_del_out_prob: float = 0.02     # chance a mutation removes an effector (rarer)
    # RETINOTOPIC organ specialisation: a sensory organ's MODALITY gene says WHICH sense
    # group it hears (0 = ALL, 1..nca_organ_mods = a world-defined modality e.g. food /
    # obstacle / kin / non-kin / intero / smell). For a directional (ray) modality the
    # organ's POSITION on the lattice picks the ray DIRECTION, so the shape lays out a
    # sensory MAP (a food organ on the left hears left rays) and neighbouring directions are
    # neighbouring cells — the CA geometry represents visual space. Born ALL (== broadcast).
    nca_organ_mods: int = 8            # count of specialised modalities (ids 1..8; 0 = ALL)
    nca_organ_mod_prob: float = 0.05   # chance a mutation re-rolls one organ's modality
    # v2 — evolvable DEPTH: a type's rule is a stack of layers; mutation grows/shrinks
    # its depth (add a near-identity layer / remove a middle one), capped.
    nca_max_layers: int = 4            # max layers in a cell-type rule (depth cap)
    nca_add_layer_prob: float = 0.05   # chance a mutation deepens a rule by one layer
    nca_del_layer_prob: float = 0.01   # chance a mutation removes a middle layer
    # v3 — evolvable WIDTH: a type's hidden layers all share a width that mutation grows
    # (append a hidden node, wired in silently) / shrinks. So the perceptron's node
    # count per hidden layer evolves too, not just depth.
    nca_widen_prob: float = 0.06       # chance a mutation adds a hidden node to a rule
    nca_narrow_prob: float = 0.015     # chance a mutation removes a hidden node
    nca_max_hidden: int = 32           # cap on a rule's hidden width
    nca_min_hidden: int = 3            # floor on a rule's hidden width
    nca_cell_cost: float = 0.01        # energy per brain cell per tick (NCA only): makes
                                       # a bigger brain COST more, so shape/size is under
                                       # selection, not a free random drift. Kept GENTLE
                                       # (was 0.02) so a bigger brain still has to earn its
                                       # keep but isn't starved out before its added
                                       # capacity can pay off — the counterweight to the
                                       # growth-biased mutation + the structural leap below
    # STRUCTURAL LEAP ("hopeful monster"): a macro-mutation adds a BURST of capacity in
    # one shot — several cells, a deeper rule, wider hidden layers — not just a weight
    # jolt. New cells/layers/units are wired in silently (near-identity / zero-outgoing)
    # so the leap adds CAPACITY without wrecking the parent's behaviour; the cell cost +
    # selection then judge whether it pays. Fires on ordinary children at macro_mut_prob,
    # and is FORCED on some injected immigrants / reseed mutants (nca_leap_prob).
    nca_macro_cells: int = 6           # cells a leap grows (capped at nca_max_cells)
    nca_macro_layers: int = 1          # layers a leap deepens a rule by (capped)
    nca_macro_widen: int = 4           # hidden units a leap adds (capped at nca_max_hidden)
    nca_leap_prob: float = 0.5         # chance an INJECTED newcomer (immigrant / reseed
                                       # mutant) is a forced structural leap — rare events
                                       # already, so a coin-flip keeps some big-brain shots
    # NEUROMODULATION (NCA only) — makes a cell's rule NON-CONSTANT within a life: the last
    # state channel acts as a HORMONE that GATES the rule output, delta *= 1 + tanh(mod).
    # Neutral (mod=0) leaves gate=1 (identity, so a fresh brain is unchanged), mod<0 DAMPS
    # and mod>0 BOOSTS the update — so the same weights respond differently by context, and
    # the modulator is itself part of the state the rules drive. Toggle for the A/B (does a
    # dynamic rule beat a constant one). Costs one channel + a tanh; the batched matmul stays.
    nca_neuromod: bool = True          # gate each cell's rule by its hormone channel

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

    # SOFT LAMARCK: fraction of a parent's LIFETIME-LEARNED readout delta (dW) folded
    # into the child's genome at birth. 0 = pure Darwinism (learning dies with the agent;
    # evolution only selects the ABILITY to learn — the current/Baldwin regime). 1 = full
    # write-back (the child is born knowing what its parent learned). A fraction in
    # between compounds learning across generations — faster adaptation — while limiting
    # how much of a parent's over-fitted lifetime quirks get baked into the gene line.
    nca_lamarck: float = 0.5           # how much learned skill is inherited (0..1)

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
