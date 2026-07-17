# vivarium — a living world of evolving brains

A design note for a NEW project (sibling to nomos, its own engine). Written
2026-07-17 when the user, looking at the single LM-cell brain, made a sharp
call: that brain doesn't really *learn* — it *follows its senses*, and the
Hebbian layer on top is marginal and maze-specific. The proposed pivot is
bigger and, notably, truer to this repo's founding thesis (CLAUDE.md: "the real
goal is a scientific loop ... not optimizing a score"): an OPEN-ENDED ecosystem
where many little brains live, eat, collide, and reproduce, and intelligence is
whatever survives — no fitness function at all.

`vivarium` = an enclosed ecosystem you watch life happen in. New folder, new
engine, because this is AGENT-based (discrete mobile individuals with their own
nets, collisions, reproduction), not the cellular-automaton substrate of nomos.
The nomos lessons that DO carry over: egocentric sensors, evolution with
crossover, the "LLM designs the brain" loop, watchable rendering, and hard
reproducibility (a run = a pure function of seed + config).

## Decisions locked (with the user, 2026-07-17)

- **Substrate: a bounded GRID, one agent per cell.** Opacity ("brains can't
  pass through each other") and collision are then free — a cell is occupied or
  not. Movement is step-wise (N/S/E/W, maybe turn). Simplest thing that gives
  the physics the vision needs.
- **Rendering: a real-time pygame desktop window.** We stay in Python (reuse
  numpy + all our sim code), it's the fastest path to a *playable* window
  (pause, click an agent to inspect its brain/genes, HUD), and mp4 export is
  trivial (imageio/ffmpeg from frames). GIF is retired for this project.
- **Genome: evolvable TOPOLOGY (NEAT-style).** The gene literally IS the network
  structure — nodes and connections that grow by mutation — so "how the neurons
  are arranged" is what's inherited and crossed over. This is the heart of the
  user's vision. (Pragmatic caveat below: start NEAT-*lite*.)

## The world

- A bounded grid (walls on the border; interior can have some obstacles later).
  Bigger than forage — think 128-256 a side — with a GLOBAL camera showing the
  whole thing (agents are points/small sprites; zoom/pan for inspection).
- **Renewable food**, spawned to hold a target density: food regrows at a rate
  tuned so the world supports a bounded population — not so much that everyone
  thrives (no selection pressure), not so little that it's instant death. The
  food rate is the main knob controlling carrying capacity.
- **No global fitness.** Selection is entirely implicit: an agent that keeps its
  energy above zero lives; one that reproduces leaves genes behind. That's it.
  This is the open-ended part, and the return to the "not optimizing a score"
  thesis.

## The agent

- State: position, heading, **energy** (the life currency), age, a genome, its
  phenotype net.
- **Metabolism**: every tick costs energy (a base rate + a cost per action, so
  standing still is cheaper than moving — the forage lesson). Energy hits 0 ->
  death. A corpse may drop food (closes the nutrient loop; optional).
- **Sensors (net inputs)**: a few forward vision rays, each returning what it
  first hits and how far — food / wall / another agent (three typed channels) —
  plus interoception: own energy level, age, maybe a clock. This is a small,
  fixed input vector (the net's INPUT nodes); NEAT grows the hidden structure.
- **Motors (net outputs)**: move-forward / turn-left / turn-right (or 4-way
  move), EAT (if on food), and REPRODUCE-intent. Outputs are read as the
  agent's action each tick (argmax or thresholded).
- **Reproduction**: sexual, with crossover. When two adjacent agents both raise
  reproduce-intent and both have enough energy, they spend a chunk of it to
  spawn a CHILD in a free neighbouring cell whose genome is a crossover of the
  parents' (+ mutation). Reproduction is optional (the brain chooses it) but the
  ecology should make it worthwhile — genes that never reproduce vanish. Asexual
  (clone+mutate) is the fallback when no mate is available, and the simpler v0.
- **Opacity**: two agents can't share a cell; a move into an occupied cell is
  blocked (or is a bump). This makes crowding real — a pressure of its own.

## Population dynamics

- Carrying capacity emerges from the food rate + metabolism: too many mouths ->
  food runs out -> die-back; too few -> food accumulates -> a boom. The
  population should oscillate/settle, not explode or vanish trivially.
- **Extinction handling** (user's rule): if everyone dies, RESEED — take the
  longest-lived genome(s) recorded so far and drop ~6 fresh individuals of them
  at random open cells. So a lineage's best idea isn't lost to a bad break, and
  the world never stays empty. (We keep a small hall-of-fame of long-lived
  genomes for this.)

## The brain: NEAT, pragmatically

Full canonical NEAT (Stanley 2002) has three heavy parts: (1) a genome of node +
connection genes, (2) structural mutations (add-node, add-connection, perturb/
toggle weight), (3) crossover aligned by global "innovation numbers", plus
speciation to protect new topologies. Parts 1-2 are the fun, visible core and
are cheap. Part 3's innovation-number bookkeeping + speciation is where NEAT
gets heavy.

Plan: **NEAT-lite for v1** — real evolvable topology (nodes/connections grow by
mutation, per-agent recurrent nets allowed), crossover by matching connection
(src,dst) pairs directly instead of a global innovation registry, and NO
speciation at first. This gives the user's "genes = network structure, children
mix their parents' structure" with a fraction of the machinery. Add innovation
tracking + speciation later if diversity collapses. The phenotype is evaluated
by a topological (or fixed-iteration, for recurrence) forward pass each tick.

Genome sketch (serializable — important, the LLM will read/write these later):

```
Node   = {id, kind: input|hidden|output, activation}
Conn   = {src_id, dst_id, weight, enabled}
Genome = {nodes: [Node], conns: [Conn]}     # + a small config header
```

## Rendering & interaction (pygame)

- Global view of the whole grid: food as small dots, agents as cells coloured by
  LINEAGE/genome (e.g. hash the genome to a hue, or colour by a "species" if we
  add speciation) so you SEE families spread and compete. Energy -> brightness.
- HUD: epoch/tick, population count, food count, births/deaths, mean age.
- Click an agent -> a side panel draws its NET (nodes + weighted edges) and its
  gene stats (node/conn counts, lineage) — "показать гены" made literal.
- Controls: pause/step, speed up (skip rendering to run epochs fast), save mp4,
  save/load a world snapshot.
- Performance: hundreds-thousands of small nets per tick. Vectorize the sensor
  pass and simple actions in numpy; nets are tiny so even a Python loop over
  agents is fine at first. If it bogs down, batch nets by topology or move the
  hot loop to numpy/numba. Realtime for a few hundred agents is very doable.

## Reproducibility

Keep nomos's discipline: the whole sim is a pure function of (seed, config).
Same seed -> same world, same food, same mutations, same history. This is what
lets us re-watch a run, A/B a config change, and (later) parallelize epochs.

## The LLM, later (not v1)

Once the playground breathes, bring back the project's real thesis: the external
LLM as a META-DESIGNER. It watches how agents behave (summaries + maybe a
rendered clip / genome stats), and proposes better STARTING genomes or
architecture priors — seeding the world with brains it designed, or biasing
mutation. The genome being clean serializable data is what makes this possible.
This mirrors brain_lab, but the objective is open-ended survival, not a score.

## Staged plan

0. **Skeleton**: new `vivarium/` package, config dataclass, deterministic RNG,
   an empty bounded grid + renewable food + a headless step loop + a smoke test.
1. **Agents, asexual**: energy/metabolism/eat/death, a FIXED tiny net (not NEAT
   yet) wired sensors->motors, movement + opacity, clone+mutate reproduction.
   Get a population that lives, eats, and dies with a stable-ish count.
2. **pygame view**: global camera, agents coloured by lineage, HUD, pause/click-
   inspect, mp4 export. Watch the ecology breathe.
3. **NEAT-lite genome**: replace the fixed net with evolvable topology; add
   sexual reproduction + crossover; the hall-of-fame + extinction reseed.
4. **Tune the ecology**: food rate, metabolism, repro cost, vision range — find
   the regime where population oscillates and behaviour visibly evolves (does
   foraging/fleeing/clustering emerge?).
5. **LLM meta-designer**: feed genome+behaviour summaries to DeepSeek, let it
   propose starting genomes; compare LLM-seeded vs random-seeded worlds.

## Open questions to settle before/while building

- **Movement model**: pure 4-way step, or heading + forward/turn (gives vision a
  natural "front")? Heading makes ray-vision meaningful; 4-way is simpler.
  Leaning heading+turn even on the grid.
- **Corpse -> food?** Closing the nutrient loop makes predation/scavenging
  possible later and stabilizes food; or keep food purely regrown for now.
- **Crossover on differing topologies**: how to align two different-structured
  genomes for a child. NEAT-lite: match by (src,dst); disjoint genes inherited
  from the fitter/longer-lived parent. Needs care so children are valid nets.
- **Speciation**: skip for v1, but without it a single lineage may take over and
  kill diversity. Watch for it; add if needed.
- **How much to vectorize up front** vs a clean per-agent loop. Start clean and
  readable; optimize when a real run is too slow, not before.
- **Determinism with a dynamic population**: agent update ORDER must be fixed
  (e.g. by a stable id) so the run stays reproducible as agents are born/die.
- **Predation?** Not in v1, but the sensors already distinguish agents — eating
  other agents is a natural later axis (herbivore/carnivore split).

---

## Progress (2026-07-17): stages 0-2 built; the bootstrap problem

Built and tested (13 tests): the world (bounded grid, inner walls with doorways,
renewable food, heading+turn movement, ray vision that stops at walls, energy
metabolism, asexual clone+mutate reproduction, corpse-as-decaying-obstacle,
extinction reseed), the v1 fixed-MLP brain (weights = genome), the pygame view
(global camera, agents coloured by genome, click to inspect a brain), genome
save/load + a seed pool, and "living walls" that slowly drift.

**The central finding so far — the bootstrap problem.** A thriving ecology does
emerge, but NOT reliably: with random initial brains, evolutionary takeoff (pop
exploding from ~2 to ~180, food actively eaten, births≈deaths) happened on only
1 of 3 seeds in an open world, and NEVER once inner walls were added. Two causes:
- Random MLPs are mostly DEGENERATE — argmax over 4 actions with random weights
  usually yields one constant action, so the population fills with spinners that
  never forage and evolution has nothing to select.
- Even sparse inner walls stop a random forager from sweeping food, killing the
  one bootstrap path the open world afforded.

Two fixes, both now in:
1. **Forward-bias on fresh brains** (`init_forward_bias`): lift the FORWARD
   output bias so a new brain tends to MOVE, giving evolution a foraging
   substrate from tick 0 instead of a field of spinners. (A mild motor prior;
   sensing still overrides.)
2. **Scaffold via the seed pool**: evolve competence in an EASY open world, save
   the longest-lived genomes, seed a HARD walled world from them (+mutation).
   Random brains can't bootstrap inside a maze, but evolved foragers dropped in
   live at once. This is also what the LLM will read/write later (stage 5).

The honest status: the *engine and the tooling* are solid; making the ecology
robustly LIVELY (and interesting to watch — foraging, fleeing, clustering) is
the live tuning problem (stage 4), and forward-bias + scaffolding are the levers.
Still ahead: NEAT-lite topology + sexual crossover (stage 3), then the LLM
meta-designer (stage 5).

---

## Stage 3 done (2026-07-17): NEAT topology + sexual crossover, validated

The fixed MLP is gone; brains are NEAT-lite genomes (graphs that grow by
mutation) and reproduction is sexual-by-proximity (two adjacent fed agents ->
crossover + mutation child; a high-energy asexual clone is the lone-pioneer
fallback). Plus, from user feedback: walking walls (strokes keep their shape and
slide along their own axis), sparser default walls, 6-letter genome names +
colours with a top-3-lineages leaderboard and a live NEAT-graph inspector, an
FPS/speed HUD with an uncapped fast-forward, VS Code F5/tasks, and genome
save/load as JSON (evolve open -> seed a walled world; also what the LLM reads).

Behavioural validation (open world, 8000 ticks): the ecology takes off (pop
~235, food eaten 530->~75) AND the topology GROWS - brains reach 3-5 hidden
nodes from a start of zero. So structure evolves, not just weights - the core
NEAT claim, working in a live ecosystem. Reproduction is also confirmed
deterministic.

Still ahead: tune the ecology for interesting emergent behaviour (clustering,
fleeing, niches), maybe speciation if one lineage dominates, and stage 5 - the
LLM meta-designer reading these JSON genomes to propose better ones.

---

## Post-stage-3 batch (2026-07-17): colour, speciation, islands, perf

From a design discussion, a batch of user-approved features:

- **Heritable lineage colour**: colour was an avalanche hash (every mutated child
  a new colour — families invisible, an honest bug). Now the genome carries an
  inherited hue that drifts slowly, so families share a colour and diverge
  gradually. Name still = brain fingerprint; colour = lineage.
- **Interactive leaders**: click a top-3 leader to pin and inspect its genome/net
  (they're dead lineages with no on-map agent).
- **Speciation by colour**: mating needs partners within `species_hue` (circular)
  — reproductive isolation by colour; a lineage that drifts past it becomes a new
  species; HUD shows the live species count. FINDING: the mechanism works (tests
  confirm isolation + counting) but does NOT sustain multiple species in a
  HOMOGENEOUS world — competitive exclusion collapses it to species=1 across a
  12k-tick run. Sustained coexistence needs ecological niches (food types /
  regions favouring different brains) or fitness-sharing (protect rare species).
  That's the honest next step for real diversity.
- **Fecundity-weighted selection**: agents count offspring; the hall ranks genomes
  by age + weight*offspring, so saved/migrated pools are the genomes that leave
  descendants, not just old survivors.
- **Island model** (`vivarium/island.py`): many independent worlds evolve in
  parallel (ProcessPoolExecutor — GIL-irrelevant), exchange top-K migrants between
  rounds via partial migration (random subsets). `run_island` maps 1:1 to a k8s
  Job; the controller is the between-round merge. The horizontal-scale answer to
  the GIL, as intended.
- **Perf**: precomputed vision rays removed the round()/linspace hot spot (~2x
  faster step, no numba/no new deps). numba stays a later lever if needed.

Backlog (user ideas, after the above): occasional IMMIGRATION of fresh
random-ish brains (test whether evolved always beats naive; maintains diversity)
— pairs naturally with the niche/fitness-sharing work for real speciation. Then
stage 5: the LLM meta-designer reading the JSON genomes.

---

## The NCA-brain pivot (branch `vivarium-nca`, 2026-07-17)

A new direction the user proposed: make each agent's brain not a NEAT graph but a
tiny **Neural Cellular Automaton** — a small grid of cells, each cell a little
net, that evolves and mutates. Motivation: "perceptrons playing a game" has been
done; a *population of CA brains* in an open ecology (with crossover, speciation,
LLM meta-design) has not, and it closes the arc — nomos gave the CA substrate,
vivarium the ecology, now the CA becomes the brain inside the ecology. This is
Neural Cellular Automata (Mordvintsev 2020; HyperNCA 2022), but as an open-ended
*evolving population*, which is the genuinely new part.

**Design settled with the user (2026-07-17):**
- **Grid + alive-MASK** (my synthesis, chosen over a free graph of cells): the
  brain lives on a bounded G x G lattice, but a heritable boolean mask says which
  cells are ALIVE, so the shape is GROWN and arbitrary (an L, a blob), not a fixed
  square. This resolves grid-vs-graph cleanly: keeps grid neighbourhoods + the
  killer "watch the texture think" visual + a single batchable tensor, while
  giving the user's "a SET of cells, grown one at a time, non-square". Evolved
  *connectivity* lives inside the rule's input weights over the neighbourhood, not
  as an explicit edge set — cheap, and it's what the user meant by "связи геномом".
- **Start 2x2**, grow/shrink by one cell per structural mutation. (Size 1 also
  works — it degrades to a plain recurrent net — but 2x2 gives live neighbourhood
  from tick 0, echoing the NEAT bootstrap lesson.)
- **32 sub-ticks of brain per game tick** (the forage think-time lesson): the rule
  relaxes across the grid so a signal injected at the senses reaches the motor
  read-out before the agent acts. Bigger brain -> more compute -> slower sim, which
  the user accepted (and which is thematically nice: intelligence costs time).
- **Sensor/motor wiring is size-invariant** (survives growth): the sensor vector
  is projected to a per-cell DRIVE broadcast into every alive cell each sub-tick;
  the action is read from the mean-POOLED cell state through a motor projection.
- **Language stays Python**; parallelism = batching + islands, not per-brain
  threads. (Bench on the i7-12700: 50 agents / 12x12 / K16 ~ 42 world-ticks/s, near
  realtime; 200 / 16x16 / K32 ~ 1.5/s, a night run. GPU/torch is the later lever.)

**Complexity staged** (don't build it all at once — the user's "set of rules /
NEAT-inside-the-cell + evolved brain size" is the full vision, reached in steps):
- **v0 (DONE, this commit): single shared rule, one cell type**; evolve the rule
  weights + the sensor/motor projections + the alive-mask SHAPE. `vivarium/nca.py`
  mirrors `neat.py`'s public surface (Genome / Phenotype / random_genome / mutate /
  crossover + action constants), so the World uses either brain via `brain="neat"|
  "nca"` (a `_BRAINS` registry; `--brain nca` on run/app/island). genome_io and
  islands dispatch on a `"brain"` tag in the JSON, so pools of either type load.
  The app inspector draws the CA brain as a live texture (first 3 channels -> RGB)
  instead of the NEAT graph. 27 tests green (9 new NCA ones: deterministic act,
  shape grow/shrink, valid crossover, deterministic world, save/load, headless
  render). The rule: perception (3x3, bounded, dead cells zeroed) -> MLP -> residual
  tanh state update; motor = pooled state -> logits.
- **v1 (next): cell TYPES = a SET of rules** — the genome carries K rule-nets, each
  cell has a type pointing at one; mutation adds/retypes cells and can split a new
  type. This is the real morphogenesis ("each cell its own").
- **v2 (later): NEAT INSIDE the rule** (the cell's own net grows topology), maybe
  sensory ORGANS on the shape's edge instead of uniform broadcast.

**v0 behavioural status (first live runs, open world, size 48, 25 agents):** the
ecology LIVES and is deterministic — population oscillates (7->1->10->3->19 over
1500 ticks) with births accelerating, no collapse. But like the NEAT bootstrap
problem (food stays pinned at target -> foraging is still weak), takeoff isn't
automatic; the same levers apply (forward-bias prior — already in via `b_out`
FORWARD bias; seed-pool scaffolding; open->walled curriculum). Engine + tooling
are solid; making the NCA ecology robustly LIVELY is the tuning frontier, exactly
as it was for NEAT. Open: does the CA brain, given the think-time, forage BETTER
than the reactive NEAT graph once tuned — the A/B the shared World now makes
trivial (`--brain neat` vs `--brain nca`, same world, same seed).

**v1 DONE — cell TYPES (a set of rules) + size under selection + the brain-net
viz.** The genome now carries a SET of update rules (rule arrays gained a leading
type axis: W1 (K,H,9C) etc.) and a per-cell `type_map`; each cell runs ITS type's
rule, so cells DIFFERENTIATE. v0 is the K=1 special case. Mutations: RE-TYPE a cell
(`nca_retype_prob`) and SPLIT a type (`nca_split_prob` — clone+jitter a rule into a
new type, hand it ~half the source type's cells); grown cells inherit a neighbour's
type so types form spatial regions. Crossover mixes rules by shared type index
(fitter supplies shape/type-map/extra types). A prompted fix to the "cell count
never moved" observation: in v0 brain size was a FREE random drift (no cost, and
founders/reseeds never mutate, so the inspected long-lived agents stayed 2x2). Now
`nca_cell_cost` charges energy per brain cell per tick (NCA only), so a bigger brain
must EARN its keep — size is under selection, not drift. **Viz** (user's request:
"draw the neural net by layers, like main; show what cells the brain is made of"):
the inspector now (a) borders each texture cell by its TYPE colour and (b) draws
each cell-type's rule as a compact layered net (input channels -> hidden -> channel
delta, strongest edges only). So you see the shape, the type regions, AND the little
nets the cells are made of. 155 tests pass (11 NCA: + type split, cross-type-count
crossover). Still ahead (v2): NEAT topology growth INSIDE a rule; sensory organs on
the edge; and the real tuning question — do differentiated multi-type brains, paid
for by the per-cell cost, out-forage the single-rule ones.

---

## Reviving the ecology first (stage 4, before v2) — food niches + drifting halos + predation (2026-07-17)

Plan agreed with the user: do the ecology-liveliness pass (б) BEFORE the v2 brain
(а). Our own take on niches, deliberately different from main's static Voronoi zones:
- **Food TYPES + DRIFTING HALOS**: food carries a type (`n_food_types`, default 2);
  each type spawns from a HALO centre that random-walks the map (`food_halo_*`), so
  food is a MOVING resource patch to track, not a uniform sprinkle (and not a fixed
  zone). Concentrated patches are also meant to help a lucky forager bootstrap.
- **DIET gene** (heritable, on both genomes): which food type you digest best (1.0
  own type, down to `diet_min_eff`); vision dims indigestible food (diet-aware).
- **PREDATION**: a heritable `carnivore` trait; a committed carnivore that steps into
  another agent eats it (no corpse), plant energy scales with (1-carnivore).
Both diet + carnivore are heritable analog traits added to neat.Genome AND
nca.Genome (inherit from a random parent, drift slowly), serialized, shown in the
inspector; HUD shows carnivore %. The world's `food` became int8 (type per cell);
`_regrow_food` samples near a random halo. Per the user, the default world also shrank
(size 96 -> 64; CLI + Config) so there are fewer mouths and the sim runs faster.
160 tests pass. The app draws food coloured by type, faint halo discs, diet/carn.

HONEST STATUS: the mechanics all work and the world is visibly richer, but — exactly
like NEAT and like main's niche experiment — takeoff from random brains is NOT
automatic: food stays pinned at target, foraging stays weak, population oscillates
low (1-10) instead of filling the map. Concentrated halos may even HURT random-brain
bootstrap (most of the map is empty). Tuning the regime (food richness / metabolism /
halo width) and/or the seed-pool scaffold (evolve an easy uniform-food world -> seed
the halo/predation world) is the live work. v2 waits until the ecology is
demonstrably lively.
