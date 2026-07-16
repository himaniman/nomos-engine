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

*Threads (1) and (2) → Experiment 4; thread (3) → Observer v2, both below.*

---

## 2026-07-16 — Observer v2: the observer that notices

**Motivation.** After Experiment 3 the observer could not, on its own, tell that
`g` had tracked `food`; we had to compute the correlation by hand. An observer
that can't detect adaptation is useless for the automated loop.

**Added.** Metrics that describe a run without being told what channels mean:
- `correlations` — top channel-pair Pearson correlations (finds `g↔food` = 0.997
  unprompted);
- per-channel `gradient` — alignment with a coordinate axis (finds `g` became
  organized along x, corr 0.997);
- per-channel `components` — 8-connected blob count/size via toroidal label
  propagation (no scipy), i.e. structure detection;
- `activity.periodicity` — dominant non-DC frequency of the activity series (FFT);
- `dynamics` (needs frame history) — per channel `persistence` (lag-1 self-corr),
  `recurrence` (the lag with peak self-corr; a period-2 blinker peaks at lag 2),
  and gzip/entropy at start vs end (the Kolmogorov-style "did complexity grow"
  signal: for Experiment 3's `g`, entropy 0.0 → 4.0 bits).
- `nomos/archive.py` — SQLite run archive (`--archive runs.db`); one row per run
  with the full law text + report JSON. Substrate for the automated loop.

**Conclusion.** Re-running Experiment 3 through v2, the observer reports the
adaptation by itself in three independent ways (pair correlation, axis gradient,
entropy rise). Ready to be the "eyes" of an automated scientist.

---

## 2026-07-16 — Experiment 4: does evolvability itself evolve?

**Hypothesis.** If the mutation *rate* is a heritable gene, selection should
raise it in an environment that keeps changing (mutators supply the variation
needed to keep re-adapting) and not raise it — or lower it — in a static one
(where mutation is pure load once adapted).

**Engine change.** Added `time()`, `sin`, `cos`, `pi()` so a law can drive a
*changing* environment. Environment `food` is a travelling sine wave
`0.5 + 0.4*sin(2π(2x − t·speed))`.

**Setup.** Genes `pref` (niche) and `mut` (mutation rate), both heritable, both
init constant (0.5, 0.02) — a monoclone. Selection kills on `|food−pref|`.
Key design point — **linked inheritance**: `pref` and `mut` must come from the
same parent or a mutator allele can't hitch-hike with the good `pref` it made.
Achieved with a shared random parent-key channel `pk` (reshuffled each step);
`argmax8(pk*alive, ·)` selects the same living neighbour for both genes.
`moving` vs `static` law files differ only by the presence of the `environment`
rule.

**Observation** (128², 1200 steps, seed 1; mean over living cells):

| environment | mean `mut` start → end | \|food−pref\| | population |
|-------------|------------------------|---------------|------------|
| moving | 0.020 → **0.096** (≈5×) | 0.09 | 0.88 |
| static | 0.020 → **0.025** (flat, then declining) | 0.01 | 0.99 |

In the moving world the mutation-rate gene rose ~5× and stayed high; the
population tracked the wave at a lag (mismatch ~0.09). In the static world `mut`
first rose (0.020→0.029 during initial adaptation) then declined once the
population sat on the optimum. The final `mut` field is a mosaic of competing
mutator lineages (out/evolvability_mut_final.png).

**Conclusion.** Evolvability evolves. A second-order gene (how fast to mutate)
is shaped by selection through its linkage to the first-order genes it
generates — upward under sustained environmental change, downward at a fixed
optimum. The static-world transient (rise-while-adapting, fall-once-adapted) is
a sharper prediction than the original hypothesis and is the kind of
causal detail the intended lab-journal loop is meant to surface.

**Open threads.** (1) Vary wave speed — is there an optimal `mut` per rate of
change (faster change → higher evolved `mut`)? (2) Cost of mutation is currently
only via maladapted offspring; add an explicit fidelity cost and see if it caps
`mut`. (3) Sexual recombination (blend two parents) vs the current asexual copy.
(4) These four experiments were all run by Claude-in-session — next is handing
the loop to an API model, with the SQLite archive as its memory.
