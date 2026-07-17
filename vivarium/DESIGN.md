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
