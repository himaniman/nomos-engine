# Design note — `cortex`: one player, a brain that grows itself

Status: DESIGN, nothing built. Written 2026-07-21 from a session with the user;
the lineage / inheritance / probe / LLM-slot decisions were settled in the same
session and are collected under "Decided" near the end. The user had just talked
the biology through with another model by voice, which framed it (no central
architect in a nervous system, glia + chemical gradients steer wiring, neuro-
modulators switch whole regions into a different mode, global order emerges from
local rules). This note fixes what we agreed so we can argue with it later
instead of re-deriving it.

The short version. A new playground `cortex/` next to `aqua/`: **one agent, one
big brain, a long life, an open world**. The genome stops being the brain and
becomes a **program for growing** a brain; the tissue develops from a seed,
differentiates by morphogen gradients, wires itself with axons, and keeps
adjusting its structure during life under an energy budget. The world stays open
and rich, but its richness is **economic** (every capability must pay), and the
task ladder is an **observer**, not a curriculum. The LLM's job moves up a level:
it writes the developmental and learning RULES, not weights.

## What we already have (and are forking, not rewriting)

Most of the machinery exists; it is spread across two playgrounds.

`vivarium/nca.py` — the brain. Already: heritable alive-MASK (grown shape),
per-cell TYPES with their own layer-stack rules (`_split_type` = differentiation
from within), evolvable rule DEPTH and WIDTH, local I/O (`in_mask`/`out_mask`)
with retinotopic sensory organs (`organ_mod` + `_build_listen`: a cell's POSITION
picks which ray it hears), NEUROMODULATION (the last channel gates each cell's
rule), within-life PLASTICITY at two levels (motor readout `dW`, and the rule
output layer `_dW_rule`, hard-clipped), heritable think-time / vision / smell,
and a numba kernel for the whole relaxation.

`aqua/` — the world. Continuous float geometry (wall rects, food morsels, corpse
circles), analytic ray-vs-AABB vision, inverse-square smell, deterministic
spatial bucket, marched sub-step movement, split render + brain inspector.

`nomos/` — the loop. `brain_lab.py` (LLM as mutation operator, leaderboard in the
prompt), `scientist.py` (hypothesis -> run -> observer -> conclusion), `elites.py`
(MAP-Elites behavior map), `lamarck.py` (learned weights inherited across lives),
`metrics.py` (the observer).

What is genuinely MISSING for this note: axons (long-range directed links),
chemistry/hormones as separate slow layers INSIDE the brain, and — the big one —
growth that lives in the tissue's own dynamics instead of in an external
`mutate()` called between generations.

## The compute argument for going single-player

At pop ~127 an aqua tick is ~64 ms, dominated by the brains' own `act`. That is
~127 agents x ~118 cells ~= 15k cell-updates per tick. **One player inherits that
whole budget**: a 64x64 = 4096-cell brain at 100 subticks is still cheaper than
the current population. Going single-player is not a narrowing — it is roughly
two orders of magnitude more brain for the same machine, and it lets a dense
lattice kernel (or a GPU one) replace the alive-only sparse path that is tuned
for many small brains.

## The cost of going single-player, stated honestly

Evolution in aqua works because there are thousands of deaths giving a dense
selection signal. Remove the population and we remove the optimizer. So the
design has to say explicitly what drives improvement. Answer: **nothing single —
a stack of timescales**, each of which already has a home in the codebase.

| timescale | what changes | mechanism | status |
|---|---|---|---|
| sub-tick — thinking | cell state | CA relaxation | exists |
| tick — learning | output-layer weights | 3-factor Hebb + reward | exists |
| ~50 ticks — **development** | shape, types, organs, axons | **rules inside the tissue** | NEW — this note |
| life — lineage | the developmental genome | (1+lambda) + Lamarck | exists (`lamarck.py`) |
| experiment — design | which rules exist at all | LLM | exists (`brain_lab`/`scientist`) |

Note where the optimizer really lives: **morphogenesis within a life is not
optimization, it is exploration of structure.** Selection across lives (or the
LLM) still does the optimizing. Self-modification buys us a cheap way to generate
good architectures from a compact genome; it does not replace learning. Selling
it as more than that is how this idea fails.

## Lineage: what (1+lambda) means here, and why "one long life" IS it

The unit that crosses a life boundary is the **genome** — the growth program —
never the tissue. A round is:

```
life N:   genome G -> embryogenesis -> brain -> lives, eats, learns -> dies
              |
              +-> G mutated / rewritten into lambda variants
              |
              +-> lambda new lives, each GROWING ITS BRAIN FROM ZERO
              |
              +-> the best becomes G for the next round
```

So "the genome determines growth from zero" and "a lineage of lives" are not in
tension: a lineage is the ONLY way to test a growth program, because the program
is expressed only through a grown brain and a grown brain is judged only by
living.

And that means **"one long life + dead-end detection + restart" is already
(1+lambda) with lambda=1.** Something has to change after the dead end, that
something is the genome, and regrowing from it is a new life. The real design
choice is three independent axes:

| axis | options | decision |
|---|---|---|
| **lambda** — candidates per round | 1 = one watchable agent / 4-8 = faster search, x lambda cores | **1** (fits the user's machine, and one agent on screen is the point). Later: fan out on the k8s cluster over a genome x world-SEED matrix — runs are pure CLI functions of (genome, seed), which is exactly what makes that trivial |
| **who proposes the next genome** | rng (Darwin) / LLM (design) / both | a switch. Default **LLM at the life boundary** (see below); rng mutation stays as the cheap fallback and the A/B control |
| **what is inherited** | see the next section | genome only |

### The immortality trap

If the single player cannot die, "one long life" destroys the measurement. This
is not hypothetical — the journal already records it twice: two LLM brains became
IMMORTAL by learning to RUN AWAY from food and circle forever (score ~0, survival
= the full cap), and the forage baseline oscillates between two equidistant foods
until it starves. **Death by starvation stays real.** The timeout is only a
backstop, and the metric is accumulated life + probe results, never survival.

### Dead-end detection, concretely

Four cheap signals, all with existing machinery behind them:

- **energetic** — food-per-tick over a sliding window stops improving;
- **structural** — no division / apoptosis for N development ticks AND `|dW|`
  converged: the brain has frozen;
- **behavioural** — the entropy of the action distribution has collapsed (this is
  precisely a detector for the reactive limit cycle the user spotted in the forage
  GIFs);
- **cartographic** — the brain's behaviour-map cell has not changed for N rounds.
  `scientist.py` ALREADY implements this as basin escape ("two identical landings
  in a row -> the next prompt demands a STRUCTURAL change, not a coefficient
  tweak"); it ports over unchanged with brains as the object instead of laws.

Worth noting these do not just say "restart" — they say **what kind of
intervention is needed**: a small mutation, a structural leap, or a call to the
LLM.

## Inheritance: what crosses the life boundary

Morphogenesis creates an option that did not exist before (d), so all four are
worth writing down:

| | carried | for | against |
|---|---|---|---|
| a | nothing (fresh random genome) | — | not evolution, just random search |
| b | **genome only** (pure Darwin) | any gain is unambiguously attributable to the growth program | slower |
| c | genome + learned weights (Lamarck) | faster | `lamarck.py` measured it: acquired skill IS inherited but SPECIALISES to the mazes seen, and can hurt on unseen ones |
| d | genome + the grown SHAPE | skips embryogenesis | see below |

**Decision: (b) by default, (c) behind a flag for the A/B, (d) rejected.**

(d) is rejected for a non-obvious reason worth recording: if the phenotype is
inherited, the growth program **stops being expressed, therefore stops being
selected, and rots.** The growth genes become vestigial, drift freely, and after
a hundred rounds the lineage is riding on copied tissue with a garbage program
underneath — a silent, hard-to-diagnose degradation. A second argument against
(d) appears below (genome x world).

(c) is kept as a flag only because we already have a prior on what it does; it
would be a waste not to compare.

## Three levels of self-modification (we are taking 1 and 2)

| | what changes | biological analogue | risk | decision |
|---|---|---|---|---|
| 1. Development | shape grows from a seed by a program | embryogenesis | low | **build first** |
| 2. Structural plasticity | cells/axons/organs appear and die DURING life | pruning, sprouting | medium | build, slow + budgeted |
| 3. Rule self-modification | a cell rewrites its own update rule | none really | high (recurrent blow-up) | already exists, bounded (`nca_rule_clip` 0.3) — leave as is |

User's call (2026-07-21): genes decide HOW the brain grows, with light in-life
adjustment on top. So: (1) is the backbone, (2) is on but slow, (3) stays where it
is.

## Morphogenesis, concretely

### The morphogenetic outputs are just a wider last layer

A cell-type rule today is a stack `9C -> H -> ... -> C` emitting a state delta.
Morphogenesis adds M extra output channels to the SAME rule:

```
delta, morph = rule(perc)      # C state channels + M morphogenetic channels
# morph = [divide(4 directions), apoptose, retype, become_in_organ,
#          become_out_organ, sprout_axon(direction, length)]
```

The cell already sees its 3x3 neighbourhood and its own state — that is enough to
decide "should I divide". No new primitive; `param_count` grows ~10%. The LLM /
evolution can read this as "the growth rule" because it is the same object as the
thinking rule.

### Growth fires on a rare DEVELOPMENT TICK, not every sub-tick

Morph channels accumulate during relaxation; structural changes are applied once
every K game ticks (K ~ 50). Two reasons, one biological (growth is slower than
signalling) and one that is load-bearing for us:

`Phenotype.__init__` precomputes `_centre`, `_nbr`, `_type_rows`, the numba typed
lists, `_M` (the listen matrix) and `_out_rows`. Changing the shape means
REBUILDING those, O(n_alive). At K=50 that is noise; at K=1 it would destroy the
fast path. **The rare development tick is what makes the whole idea compatible
with the existing kernel**, not a compromise.

### Symmetry breaking needs morphogen gradients — this is where "chemistry" earns its keep

Identical cells running an identical rule in an identical state never diverge. The
biological answer is positional information: a cell knows WHERE it is from a
concentration. So the brain lattice gets a small set of **chemical channels**
updated by blur + decay (a 3x3 convolution every N sub-ticks, much slower than the
electric layer). The seed cell emits a morphogen; it diffuses and decays; the
resulting gradient IS the coordinate system a rule can condition on ("high
concentration -> become a sensory organ, low -> interneuron"). Without this layer
morphogenesis degenerates into a uniform blob.

Payoff from existing code: `_build_listen` already maps a cell's POSITION to the
sensor index it hears (retinotopy). Under morphogenesis a cell that grows into the
"eye" region of the gradient automatically hears the right ray. The mechanism is
written; development is what finally makes it mean something.

### The budget is what stops runaway growth

Free division fills the lattice in three development ticks. Two limiters, both
already in the project's spirit:

- **global**: `nca_cell_cost` (0.01 energy per cell per tick) — a big brain has to
  be fed, so growth trades off against foraging. Already proven in aqua.
- **local**: a cell may only divide if its own RESOURCE channel is high. Resource
  enters at a "vascular" source and diffuses, so regions far from supply grow
  slowly. This is what produces a SHAPE rather than a disc.

### Stability: new tissue must be born silent

A growing recurrent system wants to blow up. The existing tricks carry over:
`tanh` every step, residual updates, and the `_resize_rule_width` idea — a new
hidden unit is born with ZERO outgoing weights so adding capacity does not change
behaviour. Same for a new CELL: born with zero state and its neighbour's type
(`_neighbour_type` already does this). The brain grows quietly, never in a jolt.

### Migration costs nothing new: it is divide + apoptose

Biology says cells migrate along gradients, and we want that — but slowly, since
restructuring is a rare event anyway. It needs NO new lattice bookkeeping: a
migration step is `divide` in one direction **carrying the parent's state and
type**, followed by `apoptose` on the parent. Same development tick, same index
rebuild, same two operations. A cell crawls up a gradient at one cell per
development tick, which is exactly the "slow" we want.

### Determinism

Unchanged project rule: a run is a pure function of (genome, world seed, steps).
Growth decisions must be a pure function of state, with ties broken in a fixed
(row-major) order, exactly as `_grow_candidates` already sorts its candidates.

### Genome vs phenotype — where state lives

Today the genome CARRIES the shape (`mask`, `type_map`, `in_mask`, `out_mask`,
`organ_mod` are 32x32 arrays) — the brain IS the genome. Under morphogenesis:

- **Genome** = seed placement + the type rules (which now include growth outputs)
  + chemistry constants + the sensor/motor projections. Small, and — the real
  prize — **small enough for an LLM to read and write.** A 18.6k-parameter weight
  matrix is not something a model can reason about; a developmental program is.
- **Phenotype** = the grown lattice, alive only within a life. Inherited only
  through the Lamarck channel, if at all.

## Genome x world: the same program grows different brains

This falls out for free and is probably the point of the whole playground.

The growth rule reads the cell's STATE; the state is driven by SENSORY input.
Therefore the genome does not determine the brain — **genome x world does.** The
same program dropped into an open plain and into a maze grows two different
brains, with no extra machinery, purely because the tissue develops under
different input statistics. This is the user's original "the brain grows on the
basis of the world", and it is automatic the moment growth is sensory-driven.

Which gives the headline experiment — named now, run later:

> Take ONE genome. Grow it in two worlds (open vs maze). Compare the brains. If
> the open-world body systematically grows more visual organs and the maze body
> more memory / chemical tissue, that is the result worth building all of this
> for: one genome, two bodies, the difference explained by the environment.

It also kills inheritance option (d) a third time: if a genome grows
world-appropriate brains, carrying the parent's PHENOTYPE forward hands the child
a brain shaped for the PARENT's world and destroys exactly the adaptivity we
built. The k8s fan-out (genome x world-seed) is the natural home for this
experiment once the sequential loop works.

## The other two signal layers (with axons)

Three carriers, three speeds — the "hormones / electricity / chemistry" split from
the user's biology conversation:

1. **Electric** — what exists: the 3x3 neighbourhood, every sub-tick, fast.
2. **Chemical** — 2-3 diffusing channels (blur + decay), updated every N sub-ticks.
   Carries morphogens, resource, and guidance for growing axons.
3. **Hormonal** — a global scalar (or a few) derived from interoception (hunger,
   damage, reward) that gates every cell's rule. This is `nca_neuromod` promoted
   from per-cell to regional/global — a whole-brain mode switch.

**Axons** are option B from `lm-cells-design.md`: a sparse edge list
`(src, dst, weight)` gathered once per sub-tick, O(E), rendered as sampled
overlay edges. The addition this note makes: an axon is not addressed abstractly —
it **grows along the chemical gradient** (a growth cone), so wiring stays spatial,
watchable, and growable by the same rules that grow cells. Pruning is weight
decay to zero.

## Two-phase life

Borrowed straight from biology and it answers "will the brain have time to grow?":

- **Embryonic phase** — a short window at the start of a life where development
  runs fast and is not charged against the game (the brain grows; the world is not
  yet lethal).
- **Adult phase** — structure changes slowly and costs energy; learning
  (plasticity) carries most of the adaptation.

Numbers: aqua lifespans are mean ~291, median ~51, leader ~3065 ticks. At K=50 a
leader gets ~60 development events (plenty to go from a 3x3 seed to a few hundred
cells), an average agent gets ~6 (not enough). Long single-player lives are
exactly what makes morphogenesis viable — and the embryonic phase removes the
dependence on luck entirely.

## The world: open, but with an economy

Agreed framing: an OPEN, rich world (Minecraft-ish in spirit, not in content), not
a level sequence. But one lesson from aqua has to be written down:

> **Richness must be economic, not decorative.** aqua has walls, corpses,
> pheromones, predators — and evolution produced competent foragers and nothing
> else, because nothing else PAYS. If the new world has a door, what is behind it
> must be worth more than the cost of learning to open it.

So the world-design work is not "add more stuff", it is **building an economy in
which each new capability pays for itself**.

**The world now has its own note: `cortex-world.md`** — geometry (tiles store it,
floats simulate it), the property-based sensorium (the world never says "food"),
materials, one contextual INTERACT, the economy table, poison, caves, the HUD, and
a world-first staged plan where a HUMAN plays it before any brain exists.

## The ladder is an OBSERVER, not a curriculum

This was the confusion worth fixing. The task ladder does not gate anything and
does not change the world. It is a set of fixed, reproducible **probes** — drop
the brain into a prepared situation, run N ticks, record the result, throw the
copy away. The main world is untouched.

That is the same philosophy as `metrics.py` + MAP-Elites: the world is free, the
observer says what grew in it. And it dissolves the open-world-vs-tasks tension
completely: the world stays open, the probes are the thermometer.

### How a probe runs

Agreed mechanics: **the exam happens at the END of a life, on a full deep COPY of
the brain — including after death.** The snapshot carries everything the life
produced: grown lattice, cell states, plastic deltas, axons. So a probe tests the
*developed, trained* brain, not the genome. The dead body is irrelevant; the mind
is what we examine.

- The copy is re-embodied in a small prepared arena drawn from the SAME world
  vocabulary (same walls, same food, same predators) so the skills transfer —
  situations resembling the world, each demanding one skill.
- Each probe is a fresh copy from the same snapshot, so probes cannot contaminate
  each other and their order does not matter.
- **Plasticity is FROZEN during a probe** by default, otherwise the exam becomes a
  training episode and scores stop being comparable across lives. A separate
  "learning allowed" variant is itself an interesting probe (how FAST does it
  adapt?) — flag, not default.
- **Probes must be watchable.** The user wants to see them, so a probe round
  renders as a small multi-panel strip (one panel per probe, world + brain), the
  same split-render machinery aqua already has. A probe that can only be read as a
  number is a probe we will not trust.

### Why probes are what makes the lambda=1 LLM loop viable

At lambda=1 the LLM gets ONE sample per life, and a life is a noisy thing to score
with a single number. The probe suite turns that sample into a **skill profile** —
a vector of "can navigate / can remember a location / can avoid a predator / can
wait" — which is a far richer thing to reason about and to write the next genome
against. The probes are not decoration on the loop; they are what gives the loop
enough signal to run at all.

The behaviour map (`elites.py`) gets re-aimed from LAWS to BRAINS — candidate
axes: modularity, fraction of long-range (axonal) connections, relaxation
settling time, dependence on memory vs reflex.

## The LLM's slot in the loop

Default: **the LLM rewrites the genome at the life boundary.** After a life ends
(death or timeout or a dead-end trigger) it receives

- the **skill profile** from the probe round,
- **world statistics** from the life (what the agent actually encountered and did
  — food rate, time in walls, encounters, where it spent its time),
- the **brain's own summary** (cells, types, organs, axons, where the growth went),
- its behaviour-map cell + NEW/visited, and the dead-end signal that fired,

and writes the next genome — the growth rule, the learning rule, the chemistry
constants. This is `scientist.py`'s loop with brains as the object of study, and
it is the same hybrid that has already won twice in this project (LLM proposes
structure, evolution tunes constants, the result beats either alone). Random
mutation stays available as the cheap fallback and as the A/B control: **is the
LLM actually better than rng at writing growth programs?** — a question the
harness should be able to answer, not assume.

## Cost / feasibility

- Compute: one 64x64 brain at ~100 subticks is comparable to today's whole aqua
  population. Dense-lattice kernel replaces the alive-only path.
- Index rebuild on a development tick: O(n_alive), amortised to nothing at K~50.
- Chemistry: one 3x3 convolution per chemical channel per N sub-ticks — cheap.
- Axons: O(E) gather per sub-tick; a few thousand edges is nothing next to the
  tissue.
- No GPU needed to start. GPU only if the lattice goes past a few hundred squared.
- Reproducibility: preserved, by the same rules we already live by.

## Staged plan (proposed)

0. **The WORLD, playable, before any brain** — see `cortex-world.md` for its own
   staged plan. Tile-stored / float-physics open world, top-down HUD, keyboard
   control writing to the same action interface the brain will use, senses that
   report properties instead of labels, and an economy verified by a human
   playing it. Only then fork `vivarium/nca.py` in as the brain and get a running
   single-agent baseline to A/B against.
1. ~~**Three signal layers**~~ BUILT 2026-07-22, NOT YET JUSTIFIED. Chemistry
   (diffusing channels) and a global hormone are in and separately switchable. The
   first A/B went to the CONTROL arm (electric only), on a run far too small to
   mean anything — see the journal. The layers stand on their design argument and
   on being what development reads position from; they have not yet paid for
   themselves on score, and that is recorded rather than re-run until it agrees.
2. ~~**Development**~~ DONE 2026-07-22. Morph outputs on the same per-cell rule, a
   rare development tick, an energy/resource budget, chemistry-driven organ fate,
   the embryo/adult split. The genome is a 24-cell SEED and a program; a body grows
   to ~400-600 cells and different programs grow visibly different bodies (5% to
   60% sensory across four fresh genomes). Structural plasticity is ON from the
   start rather than deferred — the same machinery runs in the adult phase, only
   rarely and charged for.
3. **Axons** — sparse edge list, growth cones following the chemical gradient,
   decay-pruning.
4. **Structural plasticity ON** — the same growth machinery allowed to run (slowly,
   charged) through adult life. This is the user's original "the brain rebuilds
   itself using its own senses".
5. **The observer** — the probe suite (watchable, on a post-life brain copy) +
   dead-end detection + the brain behaviour map.
6. **The LLM** — rewriting the genome at the life boundary from the skill profile
   and world statistics (the `scientist.py` loop, brains as the object of study),
   with rng mutation as the control arm.

Stages 1-2 are the ones that decide whether the idea is real. 4 is the research
prize. Everything stays SEQUENTIAL (lambda=1) on one machine; the k8s fan-out over
genome x world-seed comes after the loop works, and stage 3 of the deferred list
below is its first real use.

## Decided, so we stop re-litigating

- One agent, lambda=1, sequential. Death by starvation is real; timeout backstops.
- The genome is a growth PROGRAM; the brain regrows from zero every life.
- **Genome only** crosses the life boundary (Lamarck weights behind a flag for the
  A/B; inheriting the grown shape is rejected — the program would rot).
- Structure changes are budgeted, rare (development tick K~50), and quiet (new
  tissue born with zero state / silent outgoing weights).
- Cells migrate, slowly — implemented as divide-with-state + apoptose.
- The task ladder is an OBSERVER (probes on a copy), never a curriculum.
- The world is open; its richness must be ECONOMIC.
- The LLM writes the genome between lives; rng mutation is the control.

## Deferred (agreed, just not now)

1. Parallel lambda > 1, on the k8s cluster.
2. Structural plasticity through adult life (stage 4) — build the machinery for it
   from the start, ship it off first.
3. The genome x world experiment (one program, two worlds, compare the brains).
4. GPU / dense-lattice kernel, only if the lattice outgrows the CPU.

## Open questions (to settle in later sessions)

**The world** — settled in `cortex-world.md`; what remains open there is odorant
count, colour encoding, FOV, cave generation, and the human baseline score. Still
open here:
- Seasons / a changing environment (Exp 4 showed evolvability itself evolves when
  the environment moves) — probably needed for the hormone layer to matter.
- Which probes make up the first suite, and what world vocabulary do they share?

**The LLM** — the discussion after that:
- Exactly what does it write: growth rule, learning rule, chemistry constants,
  reward shaping? Does it ever get to touch the world's economy?
- Does it see the brain as a rendered IMAGE (the inspector panel) or only as
  numbers? The structure is spatial; a picture may carry more than a summary.
- Does it get the whole lineage history (the `memory()` block) or only the last
  life?
- Which model — the journal notes DeepSeek is a fine cheap mutation operator, but
  a smarter model should be the one setting direction.

**The brain**:
- Can a cell change its own TYPE during adult life, or is retyping
  development-only?
- What reward drives in-life plasticity in an OPEN world — food only, or shaped
  (the forage world uses geodesic-distance shaping, which an open world may not
  afford)?
- How big is the seed, and does the genome place it, or is it always one cell?
- What exactly is a "brain summary" the LLM can act on?

---

# Where this stands, and what comes next

Written 2026-07-23 at the end of the build session, for whoever picks this up next
(quite possibly a fresh session with none of the above in its head).

## Built

The world (`docs/cortex-world.md`) is finished and playable, and the sensorium passes
its own completeness test — the user played it from the SENSE PANEL ALONE. Baselines
are on the record, and they are the thing to beat:

```
who       runs   ticks  meals  gained   walked
reflex       8   15038   18.5   286.5    2410
human        1   17276    9.0   106.0     930
random       8    7035    3.4    71.8    1580
```

The brain is stage 0-2 of this note: a 24-cell SEED plus a growth program, three
signal layers (electric / chemical / hormonal), morphogenesis on a rare development
tick under a resource budget, within-life plasticity, a (1+lambda) lineage at
lambda=1 with named dead-end detectors, and an observer of six probes.

## Not built

Axons (stage 3). The LLM writing the growth program (stage 6). And, honestly, a brain
that forages.

## THE NEXT MILESTONE, stated as narrowly as it can be

**An LLM writes a growth program; the body develops from it, moves, and eats.**

Nothing else. Not doors, not poison, not memory — those are all downstream of an
agent that can get to food, and an agent that cannot get to food tells you nothing
about them. The user set this scope and it is the right one.

The measurement is the BASIC probe ladder, which was rebuilt for exactly this:

```
             near  open  side  behind  corner  maze  patience | mean
reflex       1.00  1.00  1.00   1.00    0.23   0.23    0.17   | 0.66
random       0.35  0.15  0.32   0.35    0.17   0.13    0.00   | 0.21
fresh brain  0.30  0.16  0.03   0.00    0.11   0.21    0.00   | 0.11
```

Read it as a ladder. The first four rungs are only "reach the food", varying where it
is: dead ahead, far, off to the side, behind you (out of sight — smell alone). The
reflex aces all four and stumbles on terrain, which makes it a perfect bar: to beat
0.66 a brain has to do what a nose-follower does AND cope with a wall. A fresh brain
is at 0.11, below a coin flip.

The earlier suite led with doors, breakables and poison, and every player scored ~0.1
on five of six probes — a whole suite carrying one probe's worth of signal. A
measurement has to have resolution WHERE ITS SUBJECT IS. The hard probes are kept in
`ADVANCED` and are worth running the day something can forage, not before.

## The next experiment, and it is cheap

Before reaching for a new primitive, settle WHY `open` fails. The hypothesis worth
testing first: **signal cannot get from an organ to an effector in time.** A grown
body is ~25 cells across, a thought is ~20 sub-ticks, and each sub-tick moves a signal
one cell — so a sense on one rim can barely reach an effector at the far side before
the command is read, and the readout pools over ALL effector cells, which averages
whatever does arrive with a lot of cells that heard nothing.

Sweep it: `subticks` in {10, 20, 40, 80} and effector count, scored on the `open`
probe alone. That is minutes of compute and it decides the next month of work:

- if `open` improves sharply with sub-ticks -> the tissue is a RELAY BOTTLENECK, and
  AXONS are the right answer (stage 3): sparse directed edges growing along the
  chemical gradient, which now exists and which development already reads;
- if it does not -> the problem is the read-out or the drive, and the fix is
  cheaper than a new primitive (pool over the strongest effectors rather than all
  of them; drop the +0.5 thrust bias that makes a fresh brain sprint by default).

## The LLM writes the growth program — the milestone above

This is now viable in a way it was not two days ago, for two specific reasons. The
genome is a seed plus a program rather than 30k weights, so it is small enough for a
model to read and write. And the observer returns a six-number PROFILE with per-probe
detail, so there is something to reason about instead of one noisy score.

What is missing to make it real is one adapter: a named, serialisable view of the
growth program — `grow_bias`, `sense_bias`, `act_bias`, the three thresholds,
`organ_map`, the chemistry constants, `subticks`, `think_every` — a couple of dozen
NAMED scalars, plus structural moves the model can request ("split a type", "add a
layer", "re-map chemical 2 to touch"). `nomos/brain_lab.py` is the pattern to follow;
what changes is that the object being authored is a developmental program rather than
a `.nomos` law.

Do NOT hand the model raw weight matrices. That was never the plan and it is the one
way to waste a good model.

## Housekeeping for a fresh session

- Branch `cortex-design`, not merged to `main`. `python -m pytest tests -q` -> 342.
- `cortex/tiles.py` map, `world.py` physics + economy, `senses.py` the sensorium,
  `brain.py` the sheet + development, `agent.py` the wiring, `lineage.py` the loop,
  `probes.py` the observer, `app.py` + `brainpanel.py` the HUD, `baseline.py` the bar.
- F5: PLAY / PLAY BLIND / WATCH a brain / WALK a probe arena / PROBES / LINEAGE.
- Read `docs/lab-journal.md` from 2026-07-21 onward: every wrong turn is written up,
  including the four that morphogenesis took and the three the probes forced.
