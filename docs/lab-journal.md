# Lab journal

Chronological record of experiments. Each entry: **hypothesis → law change →
observation → conclusion**. This is the artifact the project is really about —
a growing theory of our artificial physics, not a pile of runs.

---

## 2026-07-16 — Experiment 1: does the engine even work?

**Hypothesis.** If the DSL faithfully expresses local cellular-automaton rules,
Conway's Game of Life written in it must reproduce Life exactly.

**Setup.** `laws/game_of_life.nomos`. Unit tests place a blinker and a glider
on a small grid.

**Observation.** Blinker oscillates period-2; the glider is identical after 4
generations shifted by (+1,+1); shapes wrap across the torus edge. All pass.

**Conclusion.** Snapshot (synchronous) semantics and the neighbourhood builtins
are correct. The engine is trustworthy ground for real experiments.

---

## 2026-07-16 — Experiment 2: hand-built "primordial soup"

**Hypothesis.** Energy diffusion + metabolism + growth is enough for
persistent life.

**Observation, run 1.** Everything died; energy relaxed to a uniform bath
(`std ~ 2e-5`), population 0. The world "boiled" (every cell changing) but was
biologically dead.

**Law change.** Raised seeding density and sunlight, lowered metabolic cost.
Still uniform death.

**Law change 2.** Added `rule crowding` — cells with >5 living neighbours die.
This negative feedback broke the runaway. Population stabilized (~0.04), energy
developed real spatial structure (gzip ratio 0.68 → 0.33, i.e. more
compressible = more structured).

**Conclusion.** Positive feedback alone runs away to a dead uniform state; a
**density-limiting negative feedback** is what lets structure persist. First
lesson of the loop: stability needs a brake, not just a motor.

---

## 2026-07-16 — Experiment 3: the Darwinian threshold (user's hypothesis)

**Hypothesis (user).** Real evolution began not with birth/death but with
*heritable variation*: first cells copied their genome faithfully; only when
copying became imperfect (mutation) did open-ended adaptation appear.

**Engine change.** Added the inheritance primitive `argmax8(key, value)` (value
of the neighbour with the largest key) and coordinate ramps `posx()/posy()`.
Now a cell can carry a heritable gene channel and, on reproduction, copy a
*single* parent's gene (`argmax8(rand()*alive, g)`) rather than blending
neighbours.

**Setup.** Environment `food` = fixed left→right gradient (`posx()`). Gene `g` =
the food level a cell is adapted to. Selection kills cells with mismatch
`|food-g|`. Reproduction copies a random living neighbour's gene + `normal(0,mut)`.
Two worlds, identical except `mut`: `evolution.nomos` (mut=0.03) vs
`evolution_faithful.nomos` (mut=0).

**Observation, run A (gene init = `uniform(0,1)`).** *Both* worlds adapted
perfectly (corr(g,food)=+0.998, mismatch 0.01). Surprise, and instructive: with
rich **standing variation**, selection just sorts pre-existing variants into
their matching columns — mutation is unnecessary.

**Observation, run B (gene init = 0.5, a monoclone — zero standing variation).**
Now the regimes separate cleanly:

| world | population | corr(g,food) | mean \|g−food\| |
|-------|-----------|--------------|-----------------|
| mut=0.03 | 0.99 | +0.998 | **0.011** |
| mut=0    | 0.78 | — (g constant) | **0.230** |

With mutation the gene field *grew* the environment's gradient from a uniform
clone — visible as a left→right ramp in the rendered `g` (out/evo_mut_g.png).
Without mutation `g` stayed exactly 0.5 everywhere (renders flat black); life
survived only near the middle of the gradient, edges a dying churn.

**Conclusion.** The user's hypothesis holds in the engine, with a sharpened
statement: *mutation is what generates the variation selection needs — but only
matters when standing variation is absent.* Faithful replication + a rich gene
pool already adapts by sorting; a monoclone can adapt only by mutating. This is
the first result produced by the intended hypothesis→experiment→conclusion loop.

**Open threads.** (1) Moving/oscillating environment — does mutation rate itself
evolve toward higher evolvability when the optimum keeps shifting? (2) Let `mut`
be a heritable gene, not a constant, and see where selection pushes it.
(3) Observer can't yet *detect* "a gene tracked the environment" on its own — it
needs correlation/mutual-information metrics to notice adaptation unprompted.
