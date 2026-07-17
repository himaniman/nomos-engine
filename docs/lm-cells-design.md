# Design note — cells as tiny learning networks ("LM-cells")

Status: DESIGN / discussion. Nothing here is built yet. Written 2026-07-17 after
the user proposed making each brain cell a small multi-layer perceptron that
learns its own weights, with the external LLM (DeepSeek) designing the
structure and the *learning rule*, cells joined by axons, and some cells wired
to the external world.

The headline finding of this note: **most of the idea already fits the current
engine with no new machinery.** A per-cell perceptron whose weights are extra
channels, updated by a local Hebbian rule that is itself a `.nomos` law, is
buildable today. Exactly one part is genuinely new and hard — **axons**
(long-range directed connections) — because the DSL can only read the 8
neighbours. So the plan isolates that one hard thing instead of rebuilding
everything.

## The vision (user's, restated)

- A brain cell is not a bag of numbers under one shared formula, but a **small
  network**: a few layers, few neurons, reading its own state plus what reaches
  it, emitting a new state.
- **Weights adapt during the run** ("learning"), not just between runs.
- The external LLM sets the **structure** (how many layers/neurons, what
  connects to what) and may write the **learning rule** ("обучалка").
- Cells connect by **axons** so they can *join into structures*, not just talk
  to immediate neighbours.
- Some cells are wired to the **external world** (sensors/motors), as forage
  already does.

The reference point in the literature is the **Neural Cellular Automaton**
(Mordvintsev et al., "Growing NCA", 2020): one shared small network updates
every cell from its neighbourhood, trained by gradient descent. The user's
version differs in two important ways — **local learning instead of global
backprop**, and **axons** (grown topology) instead of a fixed 3x3 kernel. As
far as we know, the combination "external LLM designs the architecture AND the
local learning rule, topology grows via axons, reward comes from a game" has
not been assembled exactly this way.

## Decisions already made (this session)

- **Weight learning = a LOCAL RULE written in the DSL** (Hebbian-style), NOT
  gradient descent and NOT (only) between-run evolution. Rationale: keeps the
  engine's DSL-physics and `(law, seed, steps)` reproducibility intact, needs
  no PyTorch/GPU, and — the real prize — lets the LLM design *how the brain
  learns*, not just how it's wired.
- **Design first**, then decide what to build (this document).

## How it maps onto the current engine

The engine's state is a set of per-cell float **channels** updated by **laws**
(snapshot semantics, declaration order, `avg8`/`argmax8` neighbour reads). Three
observations make most of the vision fall out for free:

**1. A perceptron is just channels + laws.** With a few neurons and few inputs,
a forward pass is writable as explicit DSL. Inputs are the cell's own channels
and neighbour aggregates; weights are channels; the sum-and-squash is an
expression. Example of one hidden neuron:

```
# inputs: food, mem, avg8(motor);  weights: w_h0_food, w_h0_mem, w_h0_nb
rule hidden0:
  when 1
  set h0 = tanh_approx( w_h0_food*food + w_h0_mem*mem + w_h0_nb*avg8(motor) )
```

(`tanh` isn't a builtin yet; `clamp(x,-1,1)` or a rational approximation works,
or we add `tanh` — a one-line engine change.) The LLM writing these rules IS
the LLM "designing the structure": how many hidden channels, which inputs each
reads, how they feed the outputs.

**2. Weights as channels give the "shared → diverging" story for free.** If
every weight channel is `init`-ed to the same value, all cells start identical
(a shared network). A **local Hebbian law** then changes each cell's weights
using only that cell's own activity, so cells DIVERGE through experience — a
homogeneous sheet differentiating into regions, which is exactly the
"self-organizing brain" framing in CLAUDE.md and biologically the right
picture. No per-cell parameter store, no 16384 separate networks to train — the
divergence is emergent, not stored up front.

```
rule learn_w_h0_food:
  when 1
  set w_h0_food = clamp( w_h0_food + lr * pre_food * post_h0, -1, 1 )
```

where `pre_food` is the presynaptic activity (the input) and `post_h0` the
postsynaptic (the neuron's output). This is the "obучалка" — and it is data the
LLM writes and varies: pure Hebb, Oja's rule (Hebb + decay, stays bounded),
reward-modulated Hebb (multiply by a global `reward`/`temperature` channel so
the brain only strengthens what paid off). Reward-modulated Hebb is the sweet
spot: it turns the game score into a learning signal WITHOUT gradients.

**3. The reward signal already exists.** `temperature` (hunger) is a global
scalar field the harness injects every tick. A `reward` field (e.g. +1 the tick
food is eaten, decaying otherwise) is the same mechanism, one small harness
addition, and it's what a three-factor learning rule needs.

So a **v1 LM-cell brain — shared small perceptron, per-cell weight channels,
reward-modulated Hebbian learning, wired to forage sensors/motors — needs no
new engine primitive.** It is a `.nomos` file (large, but plain) plus maybe
`tanh` and a `reward` field. That is the cheap, high-information first
experiment: does a locally-learning network brain beat the hand/evolved
reactive brains on forage, and does watching it show regions differentiating?

## The one hard part: axons

Everything above is still a **local** CA — a cell only ever reads its 8
neighbours. "Cells joined by axons so they combine into structures" breaks that:
an axon is a **long-range, directed, weighted** connection from cell A to a
distant cell B. The DSL has no way to express "read that far cell through this
connection." This is the real research step, and it's genuinely new for the
engine.

Options, cheapest to most powerful:

- **A. Diffusible axons (no new primitive).** Approximate reach with a
  fast-diffusing "signal" channel: a cell emits into it, it spreads via `avg8`
  over many physics ticks, distant cells read the local concentration. This is
  chemical signalling, not wiring — undirected, unaddressed, already how forage
  scent works. Free, but it is NOT what the user means by an axon (no specific
  A→B link).
- **B. A sparse connection list beside the grid (new primitive).** Keep the
  grid, add a separate structure: a list of edges `(src_cell, dst_cell,
  weight)`. Each tick, a new engine step gathers `dst += weight * f(src)` along
  edges. Axons can be *grown* by a rule (a cell with high activity sprouts an
  edge toward the direction of a gradient) and *pruned* (weight decays to 0).
  This is the honest version of the idea — directed wiring, learnable, growable
  — and it turns the model from a CA into a **graph + grid hybrid**. Medium
  cost: a new state object, a new step phase, new DSL syntax to emit/read axon
  signals. Reproducibility is preserved if edge growth is a deterministic
  function of state + seed.
- **C. Full graph substrate (rebuild).** Drop the grid, make everything a graph
  of neurons with arbitrary edges. Maximum generality, maximum cost, loses the
  watchable spatial render and most of the current engine. Not recommended —
  the grid IS the visualization, and the user wants the aggregated point render
  kept.

Recommendation: **A for a quick taste of long-range coupling, B as the real
target.** B is where "клетки объединяются в структуры" actually lives, and it's
a well-contained addition (a second, sparse connectivity alongside the dense
grid) rather than a rewrite.

## Cost / feasibility numbers

- Engine speed today: 128^2 grid ~5.5 ms/step, 512^2 ~88 ms/step (measured this
  session). An LM-cell brain multiplies the *number of rules* (each weight
  update is a rule over the whole field), not the grid size. A perceptron with
  H hidden channels and K inputs each is ~H*K weight channels and ~H*K learn
  rules; at H=4, K=4 that's ~16 extra channel-ops per step. Expect a 128^2
  LM-cell brain in the tens-of-ms/step range — still fine for background runs,
  probably too slow for a live 512^2 showcase (keep those at 128^2 or smaller).
- Reproducibility: unchanged for v1 (still a pure function of law+seed+steps).
  For axons (option B), preserved as long as growth/pruning are seeded
  deterministic functions — a rule we already live by.
- No GPU needed for v1 or option B. GPU only enters if we ever go to gradient
  learning (option "torch" from the weights question, explicitly deferred).

## Staged plan (proposed, not started)

1. **Engine warm-ups** (tiny): add `tanh` builtin; add a `reward` sensor field
   to the forage harness (+1 on eat, exponential decay), mirroring
   `temperature`. Both are small, independently useful, and unblock three-factor
   learning.
2. **v1 LM-cell brain, hand-written**: a shared 1-hidden-layer perceptron with
   per-cell weight channels and reward-modulated Hebbian learning, wired to
   forage food/wall/sight/motor/avoid. Prove it plays and that weights diverge
   (render a weight channel — does it develop spatial structure?).
3. **Let the LLM design it**: extend `brain_lab` with an LM-cell spec — the
   model writes the perceptron rules AND the learning rule, scored on forage.
   This is the actual thesis for this idea: the LLM designing how a network
   brain learns.
4. **Axons, option B**: sparse directed connections with deterministic
   growth/pruning; a new DSL primitive to emit into / read from an axon bus.
   Start with a handful of hand-placed axons (sensor cell → motor region) before
   letting them grow.
5. **Evolve / tune** the learning-rule constants (lr, decay, reward window) with
   the `evolve.py` machinery, same hybrid design→tune loop we just closed on
   forage.

## Open questions to settle before building

- **Does local Hebbian learning actually produce goal-directed control, or just
  drift?** Hebb without a reward factor tends to blow up or wash out; Oja/reward
  modulation is likely mandatory. Worth a tiny isolated test (one neuron, one
  reward) before the full brain.
- **Shared-init weights or a little per-cell noise at init?** Perfectly
  identical cells may never break symmetry without a seed of noise (like the
  forage explore specks). Probably want `init` = shared value + small `rand()`.
- **What is "post" for a motor neuron?** For a hidden neuron, pre/post are
  clear. For the readout, the "post" that Hebb should use might be the actual
  move taken (a feedback of the harness decision) — which couples learning to
  behaviour and is closer to reinforcement than pure Hebb.
- **Axon addressing**: how does a cell name its target — by direction+distance
  (grid-relative, grows like a physical process) or by an abstract id (arbitrary
  wiring)? Direction+distance keeps the spatial/growable story; abstract ids are
  more general but less physical and harder to render.
- **Rendering axons**: the point-aggregate render shows fields well; drawing a
  few thousand directed edges is a different visualization problem. Maybe render
  axon *activity* as a field and only overlay a sampled few edges.

---

## v1 built (2026-07-17): plumbing works, learning is behaviourally inert

Built the cheapest version from the plan: engine got a `tanh` builtin and the
forage harness got a `reward` field (1.0 the move a meal is eaten, decays 0.6/
move, injected uniformly like `temperature`). `laws/forage_brain_lmcell.nomos`
is a per-cell 2-layer net: a FIXED random hidden projection (per-cell
`uniform(-1,1)` weights, never learned — the reservoir trick) and a LEARNED
readout updated by three-factor Hebb `dV = 0.05*reward*pre*post - decay`.

What works, verified: it's deterministic despite random-init weights (init uses
the seeded world RNG); the readout weights MOVE under reward; and the per-cell
weights DIVERGE from their shared start (std 0.43) — the differentiation-through-
experience story is real and visible. Plays forage at mean score 2.0 over seeds
0-2 (4/2/0), above the hand-built reactive/commitment brains (0.67/1.33), below
the LLM sight champion.

The honest finding: **learning does not yet change behaviour.** Ablation —
same brain with the learning rate set to 0 — scores IDENTICALLY (2.00/180 mean
over 6 seeds, per-seed identical). So this is the "drift, not control" horn of
the open question, and the reason is clear: reward is sparse (a pulse only at
the 2-4 meals per life) while the motor is dominated by the hardwired
`food + 2*sight` term, so the slow readout never accumulates enough to matter.
The mechanism is in place and reproducible; making it BITE needs one or more of:
a denser reward (shaped by approach, not just eating), a stronger readout /
weaker hardwired drive so learned weights carry the decision, longer lives, or
tuning the learning constants (lr, decay, reward window) — the last is exactly
an `evolve.py` / `brain_lab` job now that the substrate exists. That is the next
experiment, and it's cleanly set up.
