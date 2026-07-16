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

---

## 2026-07-16 — The automated scientist loop (DeepSeek via WaveSpeed)

**Built.** `nomos/llm.py` (provider-agnostic OpenAI-compatible adapter + a
MockClient for offline tests; default WaveSpeed unified API, model
`deepseek/deepseek-v4-flash`), `nomos/sim.py` (`simulate` + `digest` — a run to
a compact observation), `nomos/scientist.py` (the loop: system prompt = the DSL
spec, the model returns hypothesis + a ```nomos law + prediction, we run it,
feed the observation back, and log everything to a markdown journal + the SQLite
archive). Invalid laws are fed back as engine errors so the model self-corrects.

**First live run** (deepseek-v4-flash, 64², 120 steps, 2 iterations, ~$0.00001):
- The model's very first law omitted `when`; the engine error was returned and
  the model fixed it next turn — self-correction works.
- It then wrote `a = avg8(a) + 0.1*(rand()-0.5)`, predicted "alive", got the old
  verdict "boiling". It correctly diagnosed "noise amplitude too large", cut it
  to 0.01, and predicted "static". The observer again said "boiling".

**Finding (the loop earned its keep on run one).** DeepSeek's reasoning was
right and the *observer was wrong*. Old `activity` = fraction of cells that
changed by any ε; for any continuous law with noise that is always 1.0, so every
such world was mislabelled "boiling" regardless of amplitude. Fixed: `activity`
is now magnitude-based — mean |Δ| per channel divided by the channel's spatial
range, averaged. Verdicts recalibrated to the metric's real dynamic range
(uniform full-randomization tops out at |Δ| ≈ ⅓ range): static < 1e-3, settled
< 0.03, alive, boiling > 0.22. Now: pure chaos → boiling, pure diffusion →
settled, Game of Life → alive, the noisy-diffusion worlds → alive (they *do*
keep rearranging at small amplitude — "static" was the wrong intuition; the honest
description is low-amplitude but persistent change).

**Conclusion.** The automated loop reproduces the manual one (hypothesis →
law → observation → revised hypothesis) and immediately surfaced a measurement
error a human had missed — exactly the intended value of putting a scientist in
the loop. DeepSeek is a capable workhorse physicist; the observer, not the
model, was the weak link, and the loop found it.

**Open threads.** (1) The magnitude metric normalizes per channel by spatial
range, so a tiny-range field that churns still reads active — revisit if it
misleads. (2) Give the scientist the *archive* as memory (past laws + verdicts)
so it stops re-deriving basics each run. (3) Novelty archive / MAP-Elites: keep
a grid of (activity × structure × ...) and reward filling new cells, not a
single score. (4) Let a smarter model periodically read the journal and set
direction while DeepSeek does the many cheap mutations.

---

## 2026-07-16 — Scientist gets memory + a behavior map

**Motivation.** Open threads (2) and (3) above: each automated run started from
a blank slate and re-derived basics, and "novelty" was still a single verdict.
Both are now built.

**Memory.** `archive.memory(db)` returns the latest run of each *distinct* law
(deduped by law hash), newest first, with its stored hypothesis and outcome.
`scientist.memory_block()` renders these into the physicist's very first user
turn: "results of past experiments … build on them, do not re-test what is
already known." The HYPOTHESIS: line is now parsed out (`extract_hypothesis`)
and persisted in a new `hypothesis` column — old archives are auto-migrated with
an `ALTER TABLE` guarded by a try/except, so pre-existing `runs.db` files keep
working. Controlled by `--memory N` (default 10, `0` disables).

**Behavior map (MAP-Elites).** `nomos/elites.py` places every run in a coarse
grid of *regimes* rather than scoring it:
- **activity** (final |Δ|): frozen / static / settled / alive / boiling — the
  same thresholds the observer's verdict already uses, so the axis is honest to
  the metric.
- **structure** (mean gzip ratio across channels): uniform / ordered /
  structured / textured / noisy — 0 = flat field, 1 = incompressible noise.
- **order** (max temporal recurrence self-correlation): chaotic / drifting /
  recurrent / locked; a world that has stopped changing counts as trivially
  locked.
5×5×4 = 100 cells. The loop tells the model, after each run, its cell and
whether it is NEW, and the memory block lists visited regimes with a coverage
count. Cells are computed from stored reports on the fly, so the binning can be
re-tuned later without migrating the DB.

**Design choice.** Coverage is framed to the model as *a compass, not a score* —
consistent with the project's "seek understanding, not a number" stance. The map
just makes "somewhere no law has gone" a thing the physicist can aim at.

**Status.** Wired end-to-end and unit-tested (`test_elites.py`, extended
`test_archive.py` / `test_scientist.py`; 48 passing). Dry-run smoke test
confirmed run #2 sees run #1 in memory and is told its cell is "already visited."
Not yet exercised against a live model — this clone has no `.env`, so the
WaveSpeed key needs restoring before the next real session.

**Open threads.** (1) The three axes may be too coarse or too correlated
(structure vs order) — inspect the real occupancy once a live batch fills it, and
split/merge bins from evidence. (2) Actively *steer toward* empty cells: today
the model is merely informed; a wrapper could pick the seed law from the nearest
occupied neighbour of an empty target cell (true MAP-Elites illumination).
(3) Environment-coupled worlds: feed an external map or game (Snake — food/walls
as read-only channels, world activity read back as moves) so life must adapt to
a foreign environment rather than a self-contained one.

---

## 2026-07-16 — A gentle prompt nudge moves the physicist from blinkers to an excitable medium

**Setup.** One soft sentence added to the scientist's system prompt: stable
spatial structure that holds still, and oscillators with a *long* period, are
especially prized and rare — "a whole field flipping every step (period 2) is
easy; something that cycles over many steps … is the real find." Nothing else
changed. Then a live 6-experiment run (deepseek-v4-flash, 80², 200 steps).

**What DeepSeek did.** Before the nudge, its laws were single-channel
majority/threshold rules that produced field-wide **period-2 blinkers** (the
whole grid inverts each tick — negative lag-1 self-correlation, the flicker we
saw in the GIFs). After the nudge, on turn one it reached for a **two-channel
excitable medium**: an activator `a` (diffusion + decay + threshold excitation)
gated by a refractory variable `b` (spikes on excitation, then decays). That is
a FitzHugh–Nagumo / reaction–diffusion recipe — the textbook substrate for
spiral waves — and it was pursued with a coherent 6-step causal arc: it read the
observer, correctly diagnosed "`b` is globally synchronized (std≈0.04), so it
can't seed local wave fronts," and spent the remaining turns trying to localize
the refractory field (lower diffusion, sparser nucleation, `set b=1` with fast
decay).

**Two measurable effects.**
1. *Longer periods, as asked.* Recurrence moved from lag-2 (period-2 flip) to
   lag-7…lag-10 across runs — the world now self-recurs over many steps, not
   every step. The activator/refractory anti-correlation showed up cleanly
   (a~b = −0.41…−0.49), which is the correct physics of an excitable medium.
2. *Coherence.* Blob count for `a` collapsed from 47 to 3 — large connected
   regions instead of spatter.

**But the behavior map stalled at 9/100 cells** the whole run. This is a finding
about *our instrument*, not the physics: the structure axis (mean gzip ratio
over channels) cannot tell a coherent spiral-wave medium from noise — both
compress poorly — so genuinely different worlds kept binning as
`alive/noisy/recurrent` or `alive/textured/drifting`. The nudge changed the
physics in a way the map is blind to. (Confirms open thread (1) from the
previous entry: the structure axis needs a better descriptor — e.g. spatial
autocorrelation length or blob-size concentration — to separate order from
noise.)

**Conclusion.** A one-sentence, non-coercive steer meaningfully redirected the
automated physicist toward richer physics — from trivial blinkers to a
reaction–diffusion medium chasing spirals — validating the "gentle compass, not
a score" approach. The bottleneck is now the observer's structure axis, which is
too coarse to reward the coherence the model is actually producing.

---

## 2026-07-16 — Fixing the structure axis: coherence, not compressibility

**The fix.** The observer now measures *spatial coherence* per channel —
`metrics.spatial_autocorr`, the mean one-cell neighbour correlation on the torus
(Moran's-I-like): ~1 for smooth waves/blobs/gradients, ~0 for spatial noise, <0
for a checkerboard that flips sign each cell. The behavior map's structure axis
was rebuilt on this instead of mean gzip ratio (`nomos/elites.py`): a flat field
is `uniform`, otherwise coherence bins high→low into
`ordered / structured / textured / noisy`. Legacy archived runs, recorded before
coherence existed, fall back to a gzip proxy so they still bin.

**Effect.** Immediately, worlds the old axis flattened into `noisy` spread out:
a live batch produced `alive/structured/recurrent` and, notably,
`frozen/ordered/locked` — a coherent pattern that self-organized and then *held
still*, which is exactly the "spatial structure that persists" the nudge asked
for and the old gzip axis was blind to. Map coverage that had been stuck at 9/100
across a whole run moved again (→12) because the axis now varies with the physics
the model changes. gzip couldn't distinguish a spiral-wave medium from spatter
(both incompressible); neighbour correlation can.

**Also.** The auto-GIFs now carry a bottom progress strip (fill ∝ frame index)
and default to 6 fps (≈3× slower), so the fast period-2/anti-phase dynamics are
watchable rather than a blur; `--gif-fps` tunes it. Rendering never blocks the
science loop (failures are caught).

**Open thread.** Minor naming seam: the map's activity axis calls final-activity
< 1e-4 `frozen`, while the verdict reserves "frozen" for a *flat* field and says
"static" otherwise — so a coherent-but-still world reads verdict "static" /
map "frozen". Same number, slightly different flat-field handling; harmless but
worth unifying if it confuses later. Next real step: actively steer toward empty
cells (illumination), not just report the cell after the fact.

---

## 2026-07-16 — Illumination: steering the physicist toward empty regimes

**What changed.** A second scientist mode, `--mode illuminate`, turns the map
from a passive compass into an active target generator (MAP-Elites illumination
with the LLM as the mutation operator). Each turn: pick an empty cell on the
*frontier* (adjacent to an occupied one), hand the model the neighbour's law as
a **seed** plus exactly which single axis to move (activity / structure / order)
and a lever hint for that axis — e.g. "move structure up: raise spatial
coherence with diffusion or local coupling." Then run, and tell it whether it
HIT the target cell or landed in a different neighbour. Target choice is
deterministic under a seeded RNG and prefers well-supported, least-attempted
cells, so it doesn't fixate. `elites.py` gained the map geometry
(`neighbours`, `frontier`, `choose_target`, `axis_move`, `elite_laws`).

**First live run** (deepseek, 128², 400 steps, 6 turns, illuminate). Coverage
grew 12→14, with one direct target HIT (`boiling/textured/recurrent`) and one
other new cell reached off-target. The dominant observation: the model kept
landing back in `alive/structured/recurrent` even when aimed elsewhere — that
region is a strong attractor for these threshold/reaction-diffusion laws, and a
single-axis nudge often isn't enough to escape it. So illumination is working as
intended (it discovers neighbours and occasionally hits), but crossing *out* of
a dense basin will likely need either larger edits or seeding from cells on the
far side. A miss is still informative: it maps the basin's shape.

**Note on scale.** Ran at 128²/400 steps (up from 80²/200) to give structures
room and time to form; the engine handles it comfortably (pure-numpy torus).
Cells are scale-free (activity normalized by range, coherence and order are
correlations), so laws and their map positions stay comparable across sizes —
seeding a 128² run from an 80² archived law is legitimate.

**Next.** Either bias seed choice toward the frontier cell *nearest an empty
target on the far side of a basin*, or let illumination occasionally request a
larger structural change when repeated single-axis nudges stall.

---

## 2026-07-16 — Basin escape, and wrapping the closed-world scientific cycle

**Stall escape.** Implemented the second option above: when two consecutive laws
land in the *same* cell (a basin collapse), the next illumination prompt stops
asking for a minimal coefficient tweak and instead asks for a **structural**
change — add/remove a channel or rule, introduce a new mechanism. In a 12-turn
run it fired once (turn 7, after turns 5–6 both fell into
`alive/structured/drifting`) and the structural edit did break out — overshooting
into `boiling`, and the following turn reached a brand-new
`alive/ordered/recurrent`. So escalation works as an escape hatch, if a blunt
one.

**Coverage trajectory.** Across the whole memory+map+coherence+illumination arc,
distinct behavior-map cells grew 9 → 12 (coherence axis fix) → 14 (first
illuminate) → **19** (this 12-turn illuminate run). The map now spans the full
activity axis (frozen / settled / alive / boiling) and most of the structure
axis (uniform / structured / textured / ordered / noisy) — no longer a single
`alive/*` clump. New regimes this run included `frozen/uniform/locked`,
`settled/structured/locked`, `alive/textured/locked`, `alive/ordered/recurrent`,
`boiling/noisy/locked`.

**State of the closed-world scientific cycle — considered complete.** The loop
now has all the pieces it was missing: memory (the archive in the first turn),
a behavior map that measures coherence honestly, active illumination that aims
at empty regimes and escapes basins, and per-experiment GIFs (slowed, progress
strip) to watch it. A cheap model (deepseek-v4-flash) driving it explores a
genuinely broadening space of physics without chasing a score. Remaining
polish (not blockers): illumination misses far more than it hits (single-axis
reachability is optimistic), and the map's `frozen`/verdict-`static` naming seam.

**Next phase (deferred, by decision): the grid as a brain driving a game.** Not
"life plays via boxes" (cells can't manipulate objects) but the cell field as a
self-organizing controller that senses an external game through read-only
channels and acts on it through emitted signals (cell clustering in a region ->
a movement signal, e.g. Snake direction); rare game ticks relative to physics
ticks; the coupled system stays a pure function of (law, seed, rules) so
reproducibility survives. Rendering plan: a split GIF — life on the left, the
game (snake) on the right styled to match the life field, progress strip below;
aggregate cells for rendering once populations grow. Expect this phase to need
many more iterations than the closed-world cycle.

---

## 2026-07-16 — Game Phase A: the grid as a brain that drives Snake

**Built.** `nomos/game.py` — a deterministic toroidal Snake (wraps like the
world; dies only on self-collision; food placed by a seeded RNG). `nomos/
embodied.py` — the coupling. The engine was NOT touched: `World.state` is a
public dict, so sensing/acting happens from outside, preserving the physics
contract. An embodied law must declare two channels:
- `food` (SENSOR, read-only): each physics step the harness overwrites it with a
  Gaussian bump on the world grid pointing toward the food, egocentric to the
  snake's head (food to the right -> bump on the right).
- `motor` (MOTOR): the harness reads the non-negative centre of mass of this
  field relative to the grid centre and turns the snake that way (or keeps
  heading if the motor is silent).
The game ticks once per `--ticks` physics steps, giving the brain time to
organize between moves. `python -m nomos play <law> --gif` renders the split
watch-along view: life field left (fire colormap), Snake board right (same
palette on its fixed levels), progress strip below — exactly the layout we
sketched.

**It works.** Baseline brain `laws/snake_brain.nomos` is deliberately dumb —
`motor = 0.6*avg8(motor) + food`, i.e. let the food signal soak into a diffusing
motor field so its centre of mass sits toward the food. At world 48 / board 16 /
seed 0 it scored **22** (ate 22 food) and survived 188 ticks before boxing
itself in as it grew. So the sense->organize->act loop demonstrably produces
goal-directed motion; the snake homes on food without any explicit pathfinding,
purely from where activity concentrates on the grid.

**Reproducibility held.** `play()` is a pure function of (law, world size, game
size, seed, cadence, moves): world RNG seeded, game RNG seeded, no external
nondeterminism. Same inputs -> identical move sequence and score (unit-tested).
So causal experiments ("change one law, hold the game fixed") and parallel runs
survive the coupling, as intended.

**What this is and isn't.** This is *coupling*, not *learning*. The baseline
brain is hand-wired; it doesn't improve. The snake dies by self-trapping because
the brain has no sense of its own body — a `danger`/body sensor is an obvious
next sensor. But the milestone is met: a cell-physics grid can act as a
controller for an external game through pure signalling.

**Next (Phase B — learning).** Put selection on top: fitness = game score, and
let brains reproduce/mutate via the `argmax8` inheritance primitives (the Exp
3-4 machinery) so better players spread. Open design question: one brain per run
with evolution across runs (cheap, fits the archive/illumination tooling), or a
population of brains embedded in one world. Also: add a body/danger sensor so
"don't eat yourself" is even learnable. Expect many more iterations than Phase A.

---

## 2026-07-16 — Game Phase B: the brain learns to play, by selection

**Design chosen (user's call).** One brain per run, evolution *between* runs —
not a population inside one world. So the unit of selection is the whole brain,
and the loop is an external GA, not the in-world `argmax8` inheritance.

**Richer senses first.** Two additions to `nomos/embodied.py`: the `food` sensor
now scales its amplitude by PROXIMITY (bright when food is near, dim when far),
so the brain feels distance as well as direction; and a new optional `danger`
sensor paints the snake's own body onto the world grid, egocentric to the head,
in the same frame as food — a sense of one's own body. Sensors are written only
if the law declares them, so old brains still run.

**Learning loop.** `nomos/evolve.py` is a (1+lambda) evolution strategy over a
brain GENOME — the weights `w_mem`, `w_food`, `w_danger` substituted into a
template law `motor = w_mem*avg8(motor) + w_food*food - w_danger*danger`.
Fitness = mean(score)*100 + mean(survival) over a few board seeds; the *100 makes
eating dominate while survival gives a gradient before the first food.
`python -m nomos evolve --gif-dir out/evolve` renders a best-of-generation split
GIF each generation — the "output between runs" we wanted, so evolution is
watchable. Fully deterministic given (start genome, seeds, sizes, cadence, rng).

**Result.** A live run (48/16, 250 moves, 3 eval seeds, 8 generations, lambda 6)
improved mean score **15 -> 17** — selection genuinely tunes the brain to play
better. So the grid *learns*, by heredity + variation + selection with an
external reward, exactly the Exp 3-4 recipe pointed at a game.

**Finding: the body sensor, as built, isn't earning its keep.** Evolution drove
`w_danger` to ~0.04 (from 0.31) — it chose to almost ignore the danger channel.
Why: (1) the game already blocks 180° reversals, so the immediate "don't back
into your neck" case needs no sensor; (2) the body sensor sums one bump per
segment near the grid centre, so its magnitude grows with snake length and forms
a diffuse central cloud that competes with the food pull more than it guides.
The honest read is that a *diffuse whole-body* sense is the wrong encoding. The
fix (next): a FORWARD-COLLISION sensor — mark only the cells the head could
actually move into next tick (the three non-reverse neighbours), so "that way is
death" is a sharp, local, length-independent signal. That's a targeted next
iteration, not a tuning knob.

**Also worth trying next.** The genome is only 3 weights on a fixed template, so
the (1+lambda) climber plateaus fast (it did after gen 1). Two ways to open the
landscape: let an LLM be the brain-mutation operator (structural edits — new
channels/rules — like the scientist's illumination), and/or evolve a larger
parameter set / richer template. Current Phase B proves the principle; making
the brain *good* at Snake is the open frontier.

---

## 2026-07-16 — The LLM writes the brain: LLM × cellular automaton, on an objective

This is the project's thesis in one loop. `nomos/brain_lab.py` makes the LLM the
mutation operator: it writes whole brain *rules* (not just weights), we score
each on Snake over several boards, and carry a leaderboard of the best back into
the prompt so it iterates against a real external objective. Mirrors
`scientist.py`, but the reward is a game, not open-ended observation.

**First live run (deepseek, 48/16, 350 moves, 8 ticks/move, 3 seeds).** The
model reasoned like a designer, each brain a diagnosis of the last:
1. `motor = food` — baseline forage. **Score 19.67** (best of the run). It
   predicted, correctly, that with no body-awareness it would eventually coil
   into itself (died ~173 ticks).
2. `motor = food - danger` — add body repulsion. It expected survival to double.
   Instead **score 8**; it diagnosed "direct subtraction glued the motor down —
   indecision."
3. `motor = food * (1 - danger)` — gate out body directions. Survived the full
   350 ticks but **score 7.3**; diagnosed "when the body surrounds the head every
   direction is blocked — it starves."
4. A new **decaying trail-memory channel** (`memory *= 0.95; memory += 0.2*danger;
   motor = food - memory`) — stigmergy, a scent trail of where it has been.
   **Score 7.7**, survived 350.

**Findings.**
- *The idea works*: an LLM productively searches the space of CA laws against a
  concrete objective, inventing mechanisms (a trail memory) unprompted. The best
  brain it wrote (19.67) beat the parametrically-evolved one (17).
- *Independent corroboration of the danger problem*: two unrelated methods —
  weight evolution (drove w_danger→0) and LLM design (every danger-aware brain
  scored worse) — agree that the diffuse whole-body sensor hurts foraging. That
  is a real result about the *sensor encoding*, not the model: subtracting a
  length-scaled body cloud from the motor causes indecision. The fix stands:
  a sharp forward-collision sensor (only the cells the head could enter next).
- *A survival/foraging tension* is visible: naive eating dies fast but scores
  high; body-aware brains live to 350 but starve. A brain that both survives and
  eats is the actual open problem — likely needs the better danger sensor plus
  memory, which the model was already reaching toward.

**Fixed a perception bug (thanks to the user noticing).** The snake "seemed to
already know how to play" and death was never visible in the GIFs. Cause: the
step that kills sets `alive=False` and returns *without* advancing the head, so
the final frame was identical to the previous one — the collision was never
drawn. Now the crash cell is flashed and the death frame is held ~6 frames, so
the self-collision is watchable. Also raised the game/thinking ratio (8 physics
steps per move, up from 4) and the game length (300+ moves) per the user's read
that the brain needed more time to organize between moves.

**Next.** Forward-collision `danger` sensor; then re-run both evolve and the
brain lab to see if body-awareness finally pays; optionally a bordered-board
variant (walls as a sensor) for a harder game; a stronger model over deepseek's
cheap iterations.
