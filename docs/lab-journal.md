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

---

## 2026-07-16 — Concentrated senses, play-to-death, and an emergent immortal

Three changes the user asked for, and one striking result.

**Concentrated sensors.** Sensors are now sharp points, not blurred areas. The
food 'smell' is a tight bump (sigma ~ min(H,W)/24), still proximity-scaled in
amplitude. `danger` was rebuilt: instead of one bump per body segment summed into
a length-scaled cloud, it marks only segments WITHIN a small radius of the head,
each a sharp point that is louder the closer it is — the imminent collision risk,
not the whole tail. This is the fix both earlier findings pointed at.

**Play to death, longer thinking.** Cadence raised to 16 physics steps per game
move (the grid gets real time to settle). Each game now runs until the snake
dies; `moves` is only a safety cap, and a brain still alive at the cap is flagged
IMMORTAL. Fitness weights survival lightly (score*100 + survival*0.1) so eating
stays the objective.

**Did concentrated danger rescue self-avoidance? No — same verdict, more sharply.**
A quick sweep: food-only scored 19.0 avg (4 boards); food-danger 16.5; a
memory+danger variant 14.8. Danger helps on some boards and hurts on others
(seed 3: 18->32 with danger; seed 2: 28->4) — high variance, net wash. So even a
clean, concentrated, proximity-based body signal does not let a *reactive* linear
brain beat naive food-chasing. Robust self-avoidance is a planning/memory problem,
not a subtraction. This is now triply confirmed (weight evolution, LLM design v1,
LLM design v2).

**The user's prediction came true: an emergent IMMORTAL.** In the brain-lab run
(deepseek, 48/16, play-to-death cap 800, 4 boards), two brains discovered how to
live forever by refusing to play:
- Brain 2, `motor = 1 + avg8(food) - 2*avg8(danger)`: the constant `1` background
  plus body-repulsion makes the snake flee into open space endlessly. Survived
  800/800 ticks on ALL four boards; score 0.25.
- Brain 5, `motor = where(max8(food) > 0.01, food, 1 - 3*danger)`: "chase food if
  it's right here, otherwise run from your body into the void." Also IMMORTAL,
  score 1.5.
Exactly the outcome the user guessed: with survival on the table, the easiest
way to not die is to stop pursuing food. The GIF of Brain 2 is 800 frames of a
snake serenely circling, never eating — the progress bar fills to the end and it
never dies. Meanwhile the LLM kept *trying* to beat the naive baseline (19) with
body-aware brains (14.75, 12, 13) and couldn't, and correctly diagnosed its own
immortals ("a constant background and excessive danger scaling caused the snake
to flee endlessly without approaching food"). Naive food->motor (19) remained the
top scorer.

**Reading.** We now have two attractors in brain space: a fast, greedy eater that
dies young, and a deathless wanderer that never eats — and nothing in between that
does both well. That gap is the real problem the "LLM x cellular automaton" search
is here to attack. Getting there almost certainly needs a brain with internal
STATE (a trail/plan), so the next levers are: reward shaping that forbids the
trivial immortal (e.g. score-per-life, or a hunger cost that punishes not eating),
and encouraging the LLM toward memoryful brains. Also worth its own experiment:
deliberately reward survival and *study* the immortal — what does a law that has
learned to live forever look like, and how stable is it?

**Also fixed:** a brain that fails static validation at play time no longer takes
down the whole lab run (validate via a trial World before scoring).

---

## 2026-07-16 — Hunger closes the immortal loophole; the LLM starts building memory

Following the emergent immortal, we added the mechanic the user proposed: HUNGER.

**The mechanic.** Every move without eating makes the snake hungrier; after
`starve_every` (default 50) hungry moves it SHRINKS by a tail segment, and
shrinking a length-1 snake starves it to death. Eating resets hunger. So you
cannot live forever by avoiding food — survival now *requires* foraging, which
is exactly the balance we wanted: the reward for staying alive and the reward for
eating point the same way. Verified: the old immortal brain
(`motor = 1 + avg8(food) - 2*avg8(danger)`), which used to circle for the full
800-tick cap, now dies at ~165 ticks on every board (immortal 0/4). Loophole
closed.

**Hunger as a sense — "temperature."** Per the user's framing, hunger is exposed
as an optional `temperature` sensor: a single uniform value flooded across the
whole grid, 0 just after eating up toward 1 as starvation nears. A global "how
hungry am I" signal the brain can gate behaviour on (hunt harder when hot).

**Pushing internal state.** The brain-lab spec now argues explicitly that purely
reactive brains plateau and that the way past is memory/planning channels, with
examples (a decaying trail, an integrator, a temperature-gated switch). It
worked as a prompt: in the live run the LLM stopped writing only one-line
reflexes and built **stateful brains** — a decaying `scent` trail that remembers
where the body has been, a `heading` integrator that builds a stable direction,
and several temperature-gated designs. This is the qualitative shift we were
after (Phase-B point 2).

**But the reactive baseline still wins — an honest wall.** Across 6 brains none
beat the naive `motor = food` baseline (score 19) on the four test boards; the
memory/danger brains landed 13–16. The variance is telling: seed 1 gives *every*
brain exactly 8, and seed 2 gives the greedy eater 28 but body-aware brains only
4 — a single board where danger-deflection steers the snake away from a good run
dominates the average. So the baseline's edge is partly an artifact of few, high-
variance boards, and partly that on a wrapping 16x16 board a greedy eater that
dies at ~188 ticks has already eaten plenty before it traps itself.

**Reading / next.** The scaffolding is now right: balance mechanic in place,
memory brains emerging, immortals gone. The open problem — a brain that both eats
well AND avoids trapping itself — is unbeaten but now *attackable*. Concrete next
levers: (1) more eval seeds (e.g. 8–12) so one unlucky board stops dominating;
(2) let evolve tune the LLM's memory brains (hand the scent-trail law to the
parametric loop, or have the LLM propose the template and evolution set the
weights); (3) a stronger model for the design step over deepseek's cheap
iterations; (4) study the balance directly — sweep `starve_every` and watch where
the greedy vs cautious optimum flips.

---

## 2026-07-16 — Body-fear finally works: leverage, two layers, and 64:1

The user, watching the GIFs, called it: body-fear looked visually weak, and the
16:1 think ratio felt too small. Both were right, and fixing them (plus one
design idea from the user) turned body-awareness from a liability into a win.

**Why fear was toothless — geometry.** The `danger` sensor marked each nearby
body cell at its literal egocentric position. But the imminent threat — the neck,
one cell from the head — maps almost exactly onto the grid centre, where it can
barely shift the motor's centre of mass. So the loudest danger had the least
steering authority; distant food (mapped near the edge) always outvoted it.
Fix: place each danger mark in the body's DIRECTION at a large fixed radius
(leverage ~0.7 of the half-grid). Now "body is that way" has real pull.

**Two non-interfering layers (the user's idea).** Even leveraged, `motor =
food - danger` cancels: where attraction and repulsion overlap they subtract to
mush, and the centre of mass of a half-cancelled field is unreliable. So we split
the output into two layers that never touch in the field: `motor` (go-toward) and
optional `avoid` (stay-away). The move is chosen by scoring each direction as
motor-pull minus avoid-pull — they combine only at the decision. For rendering,
the two layers are blended into colour (green = go, red = fear) so you can watch
them side by side without them fighting.

**Think ratio to 64:1.** More physics steps per game move lets the motor field
settle far more before it's read, making both the food gradient and the fear
layer decisive.

**Result — body-fear pays now.** Over 6 boards (48/16, play-to-death, 64 ticks):
- naive `motor = food`: score 14.3, dies ~139 ticks (self-trap).
- `motor = food - danger` (single layer, but leveraged + 64 ticks): **20.3**,
  survival 212 — it now clearly BEATS naive eating (it was a wash before).
- two-layer `laws/snake_brain_layers.nomos` (motor=food, avoid=danger): score
  15.7 and **survives the full 800-move cap on every board** — a steady forager
  that keeps eating enough to never starve while avoiding its own body. Not the
  top scorer, but the first genuinely *balanced* creature: neither the greedy
  eater that dies young nor the immortal that never eats.

So the two attractors finally have something between them. The knobs that made
the difference were exactly the ones the user pointed at from watching the
animation — a good reminder that the GIFs are an instrument, not decoration.

**Next.** Let the LLM/evolve exploit two layers now that they work (score vs
survival is a real frontier to push); more eval seeds to cut the one-board
variance; sweep `starve_every` to map where greedy flips to cautious.

---

## 2026-07-16 — With two layers, the LLM finds the balance (and designs like an engineer)

Re-ran the brain lab now that body-fear works (two layers, leverage, 64 ticks),
6 boards, 8 brains. The gap between the two attractors is now full of good brains,
and deepseek navigated the score/survival frontier with real control-design
reasoning. Naive `motor = food` (14.33, dies at 140) came out DEAD LAST.

Selected designs (all play-to-death; IMMORTAL = reached the 800 cap alive, and
now that hunger is on, an immortal is one that eats enough to never starve):
- Brain 2 — the plain two-layer split (`motor = food`, `avoid = danger`): 17.17,
  survived 798, immortal 5/6. It reasoned explicitly that separate layers "combine
  without cancelling internal gradients" — it understood the point of the design.
- Brain 3 (top score, 18.67, survival 497) — HUNGER-GATED RISK:
  `motor = food*(1 + 2*temperature)`, `avoid = danger*(1 - 0.5*temperature)`.
  When hungry it hunts harder and fears less; when fed it plays safe. An elegant,
  correct risk/reward controller, discovered from game feedback alone. Saved as
  `laws/snake_brain_llm.nomos`.
- Brain 5 — a decaying `food_memory` layer to keep a heading when food is out of
  range (14.67, immortal 6/6). Brain 6 — constant strong avoid + hunger-boosted
  motor (16.50, immortal 6/6). Brain 7 — `avg8`-smoothed food for a wider sensor
  (18.17, survival 649). Brain 8 — a hunger THRESHOLD (`max(0, temperature-0.3)`)
  so it only takes risks near starvation (18.33, survival 604).

Across the run the LLM independently invented: the two-layer separation, hunger-
gated risk modulation, food memory, exploration noise, food smoothing for a wider
sensor, and a hunger threshold — each proposed as a diagnosis of the previous
brain's failure, and most of them working. This is the thesis realised: an LLM
doing genuine cellular-automaton control design against an external objective,
with the observer (the game score) closing the loop. It also self-corrected an
invalid law mid-run (`neighbourhood_sum8` -> engine error -> fixed next turn).

**Where we are.** The two attractors now have a populated frontier between them:
cautious immortals (13–16.5, survive the whole run) through aggressive scorers
(18–18.67, survive 500–650). No single brain yet both tops the score AND stays
immortal — that Pareto corner is the remaining target. Levers to push it: let
evolve tune the LLM's best template's constants; more/rotated eval boards; a
stronger design model. But the machine works end to end now: senses, hunger,
two-layer motor, LLM designer, game reward, watchable blended GIFs.

---

## 2026-07-16 — Hybrid: the LLM designs the brain, evolution tunes it, and beats both

The natural next move on the score/survival frontier: stop choosing between the
LLM (good at inventing structure) and evolution (good at tuning constants) and
USE BOTH. Rebuilt `evolve.py` around the architecture the brain lab discovered —
two non-interfering layers with hunger-gated risk — and let the (1+lambda)
strategy tune its four constants (`w_mem`, `hunger_boost`, `avoid_str`,
`avoid_relax`), starting from the LLM's best genome.

**Evolution run** (48/16, play-to-death, 64 ticks, 4 boards, 6 generations,
lambda 6). Score climbed 17.25 (LLM start) -> 18.25 -> 19.25 -> 20.0 -> **22.5**
by gen 4. Best genome: `w_mem 0.55, hunger_boost 2.75, avoid_str 2.28,
avoid_relax 0.0` (saved as `laws/snake_brain_evolved2.nomos`).

**The interesting bit: evolution overruled the LLM on an open question.** The
LLM's top brain (#3) *relaxed* body-fear when hungry (`avoid_relax = 0.5`) to
chase food harder. Evolution drove `avoid_relax` to **0.0** and pushed
`avoid_str` UP to 2.28 — its verdict: do NOT drop your guard when starving; keep
body-fear strong and constant, and get your aggression from the hunger-boosted
motor instead. Notably this matches what the LLM's *own* brain #6 had argued
("keep avoidance constant and strong") but couldn't tune well by hand. So the two
searches, run together, resolved a hypothesis neither settled alone.

**Holdout (8 fresh boards, not the 4 evolved on):**
- naive `motor = food`: score 14.0, survival 135, immortal 0/8.
- LLM best (brain #3): score 19.1, survival 565, immortal 3/8.
- evolved hybrid: **score 20.1, survival 693, immortal 5/8** — it DOMINATES the
  LLM brain on all three axes at once (more food, longer life, more boards
  survived to the cap). A real Pareto step toward the corner we'd named as the
  target. (On its own 4 training boards it reached 22.5; 20.1 on the holdout is
  the honest, slightly-regressed-by-overfit number.)

**Reading.** The pipeline is now: LLM proposes an architecture and competing
hypotheses from game feedback -> evolution tunes the winner's constants and
empirically settles the hypotheses the LLM left open -> the result beats either
method alone. That is the project's whole thesis — "combine the power of an LLM
with a cellular automaton" — closed into a working loop against a real objective.
The remaining frontier is the top-right corner (max score AND always immortal);
levers from here: more/rotated boards to fight overfit, letting the LLM read the
evolved constants and design a *next* structure around them (design<->tune
alternation), and a stronger design model.

---

## 2026-07-17 — The forage world: walls, scent that flows, and movement as a command

The snake balanced score against survival, but its world was a featureless
torus — no obstacles, no navigation, nothing to plan around. New external
world, an order harder: **Forage** (`nomos/forage.py`). A WIDE seeded map
(default 96x96, ~16% interior walls laid as strokes-with-doorways at any size)
full of rooms and nooks; the player is a single dot starting at the centre;
the only death is starvation (`starve_after` moves without food); several food
items live on the map at once and respawn INTO NOOKS (placement weighted by
how many walls hug a cell), so eating means threading corridors, not surfing
one gradient. The camera follows the player (`Forage.view()` — the egocentric
crop is what the split GIF shows), the brain got denser (128^2 default, 512^2
runs ~19 min for showcases, block-mean `render_cap` keeps GIFs small), and the
think ratio is 32 physics ticks per move.

Getting a playable interface took five harness lessons, each earned from a
watched failure (the user caught the key one in the GIF before I did):

1. **Relative decisions were not enough.** With walls omnipresent, requiring a
   positive net pull froze the player forever; picking the best net direction
   unfroze it but only moved the problem.
2. **Adjacent walls must scream.** Linear-fading wall marks let big far
   structures drown the one cell the player was about to walk into — it
   pressed into walls forever. 1/d^2 fixed the pressing.
3. **Blocked moves must SLIDE, tangentially.** Naive next-best sliding is a
   two-cell limit cycle (runner-up is usually the reverse). Contact dynamics:
   slide to the next commanded direction that isn't the exact reverse.
4. **Smell must diffuse THROUGH THE MAZE, not through walls.** The decisive
   fix. Euclidean smell pinned the player against the wall its food sat
   behind. Now each food's scent field is its geodesic BFS distance map over
   open cells (cached; walls are static), the sensor bump points along the
   walking path's first step, amplitude decays with path length, and sealed
   pockets carry no scent. The world computes diffusion; the brain does the
   thinking. First meal ever eaten the same run this landed.
5. **Point-repulsion makes corridors terrifying.** Summed 1/d^2 marks from a
   corridor's paired side walls repelled ALONG the corridor axis — the player
   stalled at the mouth of the passage leading to its food ("corridor fear").
   The wall sense is now a ray sense: for each of the four movement axes, only
   the FIRST wall on that ray (it is the one that blocks), 1/d^2 by distance.
   Global maze structure reaches the brain through the scent, which has
   already flowed around it.

On top of that, the user's rule change: **movement is a command**. The game
never moves the player; a step happens only when some direction's net pull
(motor minus avoid) is strictly positive, otherwise the player stands while
hunger grows. Far from every food the scent is silent, so a purely reactive
brain starves where it stands — exploration has to be GENERATED (the baseline
law now carries sparse `bernoulli()` motor specks: noise as physics).

**The observed phenomenon of the day: the reactive limit cycle.** The
baseline brain (`laws/forage_brain.nomos`, food->motor + light wall fear)
walks to a point between two foods and oscillates — one step toward A makes
B's summed pull win, and back, forever, with instrumented net pulls flipping
winner every single move (up 8865 on one cell, down 9119 on the next, nothing
blocked). The user predicted-by-eye exactly this from the GIF. The cure is
COMMITMENT, and it lives in the brain, not the harness:
`laws/forage_brain_mem.nomos` adds a saturating positive-feedback channel
(`mem = clamp(0.99*mem + 0.02*motor, 0, 2)` fed back into motor) — hysteresis
that amplifies and keeps whichever target currently leads. Result on seed 0:
baseline 1 meal / 185 moves; mem brain **3 meals / 351 moves**. First brain to
reliably eat more than once per life. (An earlier unclamped version diverged
into the 1e6 clip and scored 0 — the clamp is load-bearing.)

The LLM lab now plays this game: `python -m nomos.brain_lab --game forage`
(FORAGE_SPEC documents the embodiment, the command semantics, and names the
oscillation failure mode explicitly, pushing internal state). Next: a live
forage lab run — the open questions are commitment vs. flexibility (when
SHOULD a brain abandon its target?), hunger-gated exploration, and whether an
LLM invents a better tie-breaker than saturating positive feedback.

---

## 2026-07-17 (later) — Metabolism, desperation, and watching the brain go blue

Four refinements from watching the GIFs together (all user-driven):

1. **Thinking now has a price advantage.** Standing still costs 0.5 hunger vs
   1.0 for a move (even a blocked one — pressing is effort). Before this, a
   pause to "think" cost exactly as much as a wrong step, so there was no
   selection pressure for deliberation — the user noticed the play LOOKED like
   non-stop walking, and the reason was economic, not cognitive. Standing
   forever still kills (the immortal loophole stays closed); the change just
   makes deliberation a purchasable good.
2. **Perception vignette on the game camera** (`Forage.view(fade=0.7)`): the
   egocentric 65x65 window now dims with distance under a Gaussian, matching
   how the brain's own senses attenuate; the player cell never fades.
3. **Hunger is visible on the brain**: the split render's blue channel floods
   with the hunger fraction (green = motor, red = avoid, blue = temperature) —
   literally the uniform field the brain feels. A brain that goes deep blue and
   then finds food reads like a story.
4. **Dense-brain rendering**: `--render-cap 64` (default) block-averages any
   brain (512^2 included) to a watchable GIF; `--view-radius 32` shows 64 map
   cells around the player.

**Desperation (hunger-gated commitment melt), first cut.** The user saw the
mem brain die in a terminal loop — commitment's own trap: a target the
geometry won't yield gets REINFORCED by the very hysteresis that fixed the
two-food tie. `laws/forage_brain_desperate.nomos` melts `mem` and injects
exploration specks when temperature > 0.7 (a nice DSL idiom: the `melt` rule's
`set` simply overrides `commit`'s under its `when` — last matching set wins).
Result (3 seeds): mem 4 meals / 689 alive -> desperate 4 / 701. The hatch
opens (s0: +12 moves alive) but too late to matter — by temperature 0.7 the
brain has ~36 moves of gamble left, not enough to random-walk to a new room.
Threshold/noise tuning is exactly a brain_lab question now; the mechanism and
its measurement are in place. Idle-cost note: the three laws' scores were
unchanged by the metabolism change on these seeds — none of the current brains
ever stands still (commitment always commands); the pressure is there for
FUTURE brains to exploit.

---

## 2026-07-17 (evening) — The 512 mystery and fog of war

**A dense brain is a different mind, and it's not about diffusion speed.** The
512^2 showcase run (same desperate law, 32 ticks) scored 0/120 where 128^2
scores 3/363. First hypothesis: avg8 spreads 1 cell/tick, so a 4x bigger grid
gets 4x less relative "thinking spread" per move — the user's "no time to
think", literally. FALSIFIED by the cheap reverse experiment: if relative
diffusion were the cause, 128^2 @ 8 ticks (same ratio as 512 @ 32) should fail
the same way. It doesn't: ticks {8: 2 meals, 16: 1, 32: 3, 64: 0}. The scan is
non-monotonic, and MORE thinking time kills the 128 brain exactly like the
dense grid did. Reading: the commitment channel's constants (0.99 decay, 0.02
feed, clamp 2) are a PSYCHOLOGY calibrated in ticks — at 64 ticks/move the
integrator accumulates twice as much per decision, saturates early, and the
brain over-commits into a trap. A law is not scale-invariant in grid OR time;
"how long should a thought take" is itself a parameter of mind. Proper
constant-vs-cadence sweeps are brain_lab/evolve territory now.

**Fog of war (user's call: no seeing through walls).** The brain was already
honest — food is SMELL (geodesic, flows around walls, sealed pockets are
scentless) and the wall sense is 4 rays (line of sight by construction). The
only cheat was the VIEWER's: the game camera showed everything in the window.
`Forage.view(fog=True)` (default in play) now masks the window by Bresenham
line of sight, cached per position: the blocking wall surface stays lit,
everything behind it is black, visible floor gets a faint glow so the lit room
reads against the fog. The GIF now shows what the player actually perceives —
a small lamp of a room with rays escaping through doorways — which makes the
case for MEMORY brains visually self-evident: almost everything the player
knows at any moment is remembered, not seen.

---

## 2026-07-17 (night) — Two live forage lab runs; sight GENERALIZES, a random walk OVERFITS

Ran the DeepSeek brain lab on the forage world twice, 8 brains each: once with
the original senses (smell/wall/temperature), once after adding the `sight`
sensor (the user's idea — food in unbroken line of sight painted strong and
straight, `sight*3` in the winner). Then a train-vs-holdout check (seeds 0-2
trained on, 3-7 held out) across every brain, hand-built and LLM-built:

```
                              train (0-2)   holdout (3-7)
reactive baseline             0.67/166      0.80/143
mem (commitment)              1.33/230      1.20/168
desperate (hunger-gated melt) 1.33/234      1.60/237
LLM no-sight (random walk)    3.00/245      0.80/146   <- collapses
LLM sight champion            4.33/253      5.20/286   <- generalizes
```

Two findings, one expected and one not.

**Expected: sight is a strong, real signal.** The champion (journal
brain-forage-sight.md #2, saved `laws/forage_brain_llm.nomos`) mixes four
things and the model weighted them well by hand: a fast leaky smell integrator
(`mem += 0.2*(food-mem)`), exploration noise PROPORTIONAL to hunger (our binary
desperation threshold reinvented as a smooth gradient), plain wall fear, and
`sight*3` — seen food dominates everything, exactly the "certainty beats
inference" the user was after. Best mean score in the project on this world.

**Unexpected: the no-sight winner was an overfit.** The best brain WITHOUT
sight (run 1 #7, `laws/forage_brain_llm_nosight.nomos`) is a correlated random
walk — inertial compass channels, smell overrides near food, hunger reroll to
escape. It scored 3.00 on its three training boards and I nearly filed it as
"exploration beats commitment". The holdout says 0.80 — it collapses to below
the reactive baseline. The 3.00 was a random walk whose seed happened to comb
three specific mazes well; it carries no navigational skill to new ones. The
sight brain, by contrast, IMPROVED out of sample (4.33 -> 5.20): a real
perceptual signal transfers, luck does not. This is why the holdout split
matters — it's the whole difference between a method and a coincidence, and
here it flipped the ranking (no-sight #7 looked like the discovery; it was the
trap). The champion generalizing UP while the random walk collapses is the
cleanest single demonstration the forage world has produced.

Lab mechanics held up: both runs deterministic bar the LLM, GIFs per brain,
the model read its leaderboard and iterated (it explicitly diagnosed zero-food
boards, tried commitment, tried random walk, tried sight-gated approach).
`laws/forage_brain_llm.nomos` is the new forage champion to beat; obvious next
step is evolving its four constants (the hybrid design+tune loop, now on the
harder world) and a wider holdout to be sure 5.20 isn't its own lucky five.

## 2026-07-18 — aqua: honest collisions, no tunnelling, and eyes that evolve

Three mechanics fixes to the continuous `aqua` world after watching the render:
agents were poking through walls, wedging against them, and everyone had the
same fixed eye.

**Collisions now match the drawn body.** The old wall test was a single POINT
sample of the rounded centre cell (`_cell_blocked`), so an agent's centre could
sit in a free cell while its drawn circle overlapped the wall — the body looked
smaller than its collision box, and food/corpses/walls were all judged at that
one point. Replaced with `_blocked_circle`: a proper circle-vs-grid test (each
cell is the unit square on its integer coords, closest-point distance < radius).
Agent–agent opacity already used the sum-of-radii circle test; now walls and
corpses do too, so what you see is what collides. A 600-step run has ZERO agent
bodies overlapping a wall (was common before).

**Movement is marched, so nothing tunnels or gets skipped.** `_advance` used to
jump straight to the destination and test only the endpoint — a fast/diagonal
move could hop over a thin wall or a food cell without registering it, and a
blocked move dead-stopped (agents wedged into walls, the "stuck" bug). Now the
move is walked in sub-steps of ≤ `substep` (0.5 cell) with a circle collision
AND a food-eat check at every hop, so the whole path is resolved: the world
always recomputes what was hit and eaten no matter the speed (this is what makes
`max_speed` safe to raise). `substep 0.5 < 2*radius 0.7` means consecutive body
circles overlap, so nothing thin can slip between two samples. A wall hit now
SLIDES along the free axis instead of stopping, so agents stop wedging. (The
opacity test now asserts "marches up to touching, never overlaps" instead of the
old all-or-nothing "did not move".)

**Vision is a heritable gene now (`vivarium/vision.py`).** Ray COUNT (`n_rays`,
3–12) and ray REACH (`ray_len`, 6–24 cells) are per-genome traits with drift +
crossover, added to BOTH brain genomes (neat, nca) with backward-compatible
defaults (they ride harmlessly on the grid world, steer aqua). The brain input
is sized for `n_rays_max` slots; an agent fans its own count into the first
slots and leaves the rest zero, so a variable eye plugs into a fixed-width brain.
Longer rays are cheap because signal already falls off with distance
(`1 - k/rl`), so a far ray just makes distant food register faintly — a
far-sighted-but-blurry vs sharp-but-short trade-off evolution can settle. And it
does: over 800 steps different seeds converge on different eyes — seed 2 grew
far-sighted 11-ray/22-reach vision at pop 219, seed 4 settled ~7-ray/10-reach,
seed 0 ~5-ray/9-reach. The FOV cone in the app is drawn from the selected
agent's own genes, so you can watch an individual's actual sight.

**A regression the wide input caused, and the fix.** Sizing the brain input for
n_rays_max (12 rays -> 57 inputs, up from 29) quietly BROKE the NEAT brain: fresh
`random_genome` fully-connected every input to the outputs with N(0,1) weights, so
the drive into each output — a SUM over all inputs — had variance ~n_in, saturating
tanh. Wider sensor -> more saturated -> fresh brains twitch erratically and die.
A/B is stark: NEAT summed-pop over 4 seeds x 1500 ticks fell 354 (n_rays_max 5) ->
64 (8) -> 19 (12), extinctions 4 -> 6 -> 10. And the churn is why the user saw NEAT
nets "stuck at one layer" — add_node fires at 0.015/mutation, so a lineage must
survive HUNDREDS of reproductions to grow hidden nodes; a population that keeps going
extinct never compounds any structure (the mutation operator was fine — a genome
hammered 2000x still grows 21 hidden nodes). Fix: scale the initial weights by
1/sqrt(n_in) (Xavier), so the drive stays ~O(1) at any input width. NEAT pop recovered
to 519 with 4 extinctions at n_rays_max 12 — healthier than the old narrow world. Only
FRESH random brains change; loaded/evolved pools keep their weights.

**Render alignment (integer coord = cell CENTRE).** The physics reads the grid with
`round(y)`, so cell k owns world [k-0.5, k+0.5]; walls/food were drawn consistent with
that, but AGENTS, the FOV cone and click-picking were drawn at `x*cell` — half a cell
up-left of where the body actually is — so bodies looked like they crept onto the wall
above/left of them even though collision (correct) never let the circle overlap.
Everything visual now maps world (y,x) -> pixel ((x+0.5)*cell, (y+0.5)*cell), so the
drawn body sits exactly on its collision circle. Also: `aqua.app --load-genomes` now
starts fresh (with a note) if the save doesn't exist yet, and there's an F5 config
"aqua: watch continuous NCA (continue from session)" that loads/saves
`aqua/pools/session.json` (press W to save the pool).

## 2026-07-18 (later) — making aqua actually continuous, stage 1: walls

Agreed framing with the user: the aqua world was only half continuous — agents
float, but walls/food/corpses were a grid that everything collided against and
saw. That grid is now the thing limiting honesty (quantised food, blocky walls,
ray-marched vision). Plan (branch `aqua-continuous`): go fully continuous by
DEMOTING the grid from source-of-truth to an acceleration index, staged
walls -> food -> corpses, determinism held throughout. Scope is A (continuous
geometry), NOT B (mass/momentum/pushing) — if we ever want more, we change brains
or world structure, not the physics.

**Stage 1 — walls are float rectangles now.** The strokes were ALREADY generated
as segments and then rasterised; we just stop rasterising for the physics. Each
stroke becomes one or more axis-aligned float RECTANGLES (a run of cells split by
door / interior-clip / centre-safe-zone), `wall_thickness` (=1.0) thick, plus four
border bars — `World.wall_rects`. Collision is circle-vs-AABB (`_wall_blocks_circle`,
distance to the rect's closest point < radius) and VISION is analytic ray-vs-AABB
(`_ray_wall_dist`, slab method) — no more per-cell blockiness and no ray-march
sampling; the FOV cone in the app draws exact hits. Kept the old `self.walls` grid
as a BRIDGE cache that still backs food PLACEMENT queries (both derive from the same
strokes, so they can't disagree — verified: 0 cell/rect mismatches over the whole
grid); stage 2 removes it with the food grid. Only axis-aligned rects (the user
doesn't want angled walls — "so that it's simple to compute"); capsules/segments
would be the drop-in if that ever changes. Verified: analytic ray hits a border at
exactly its geometric distance (22.5 from centre, not a marched multiple of
ray_step); a body shoved 16 cells at a wall in one call stops at the face and never
overlaps; 0 bodies overlap walls over a run; determinism preserved; +4 tests, full
suite 189 green. Also closed a predation edge case (a carnivore jumping onto prey
now respects the wall test, so it can't land its body inside a wall). Walls now
render as smooth continuous bars with clean door gaps. Next: stage 2, food as
float entities in a bucket index (this is the bulk — re-plumbing the ecology).

## 2026-07-18 (later still) — stage 2: food is float morsels + sense/size tuning

**Food is a cloud of float MORSELS now, not a grid.** `self.food` (the int8 grid)
is gone; food is parallel arrays `food_xy` (N,2 float) + `food_type` (N,), planted
at true float positions (halo + gaussian, no rounding), EATEN when the agent body
CIRCLE overlaps the morsel circle (`radius + food_radius`), and removed on eat. All
queries are vectorised over the arrays — no bucket needed at these counts (~80
morsels): smell = nearest by true distance; VISION treats each morsel like a nearby
agent (a bearing within the ray's angular half-spacing + food_radius/dist), FOV-pruned
to the cone so a ray never scans food behind the creature. The `self.walls` grid
survives ONLY as a coarse open-AREA count for the food-density target (an aggregate,
not per-cell truth). Corpses are still grid (stage 3). Re-plumbed regrow / placement
validity / `_clear_under_walls` / snapshot onto entities. Determinism preserved;
+2 tests (food is genuinely off-grid and never inside a wall; eating needs body
overlap and removes the morsel); full suite 191 green. Morsels render at their true
radius (0.25 cell) — small dots, as the user wanted.

**Three tuning changes the user asked for, same pass.** (1) SMELL is now an
INVERSE-SQUARE field — `amp = diet_eff / (1 + (dist/sharpness)^2)` (sharpness 6
cells) instead of the old linear `1 - dist/range`. Same whole-map reach but faint
across the field and rising STEEPLY on approach (half strength at 6 cells, ~1 on the
morsel) — the "weak everywhere, quadratically strong up close" the user described,
so a creature always has a weak bearing to the nearest food and homes in sharply.
(2) BODY radius 0.35 -> 0.45, about as wide as a wall bar (wall_thickness 1.0) as the
user wanted — but under 0.5 so it still threads a one-cell doorway (0.5 wedges: A/B
pop 84 vs 145 at 0.45, healthiest of the three). (3) food drawn at half its old
size. Perf note: at high population (~150+) a tick is ~130ms, dominated by the
PRE-EXISTING O(n^2) agent-agent scan (`_agent_near` per substep + near-agents
vision), not food (the FOV prune bounds that). A spatial bucket for AGENTS is the
deferred cross-cutting optimisation; correctness/determinism are unaffected. Next:
stage 3, corpses as float circles (small, reuses the food/agent machinery), then the
sensor-input colour-coding the user asked for at the end.

## 2026-07-18 (evening) — stage 3: corpses go float, and the cross-cutting speed pass

**Stage 3 — corpses are float obstacle CIRCLES.** `self.corpse` (the int64 grid) is
gone; a death drops a body into `corpse_xy`/`corpse_decay` arrays (radius 0.4), which
block a moving body (circle-vs-circle) and register as an obstacle in vision (treated
like a nearby agent, FOV-pruned), decrementing each tick and vanishing at 0. That
removes the LAST grid from the physics — walls, food, corpses are all continuous now;
`self.walls` survives only as a coarse open-area count for food density. +2 tests
(corpses appear on death / block / decay; no body overlaps a corpse over a run).

**The cross-cutting optimisation the user asked for — a spatial bucket for agents
(`aqua/space.py`).** The last O(n^2) was agent-agent: `_agent_near` (collision every
sub-step), the mate search, and vision's near-agent list all scanned the whole
population. Now a deterministic uniform-BUCKET index of agent ids answers "who's near
(y,x)?" in local work — rebuilt O(n) each tick and kept live as agents move/spawn/die.
It holds ids only (positions read live), and every caller selects by NEAREST, so the
result is order-independent and runs still reproduce (verified: bucket `_agent_near`
matches a brute-force scan on every agent, 0 mismatches; determinism a==b holds). The
mate search now takes the NEAREST eligible mate (was first-in-dict-order) — a
deliberate, order-independent change.

**A fresh look at speed (profiled, not guessed).** cProfile at pop ~100 said `_sense`
was ~60% and its top cost was `_ray_wall_dist` looping ALL wall rects per ray. Two
correctness-preserving prunes: (1) per agent, keep only the wall rects whose closest
point is within ray reach (`_rects_within`) — most agents see 0-3 walls, not 26; (2)
FOV-prune the near-AGENT list to the cone (food/corpses already were), since a ray can
only see its arc. Net: a tick at pop ~127 went 130ms (pre-bucket) -> 75ms (bucket) ->
64ms (prunes), ~2x, determinism intact throughout. What's left is the NCA brain's own
relaxation (`act`) and the irreducible ray loop — the brain cost is the point, not
overhead. Deferred if ever needed: a numba kernel for the ray test, or a food/corpse
bucket. Full suite green.

**Sensor-input colour-coding (the last request).** The brain inspector's input strip
was a wall of green; now each slot is coloured by sensor MODALITY, matching the
`_sense` layout: per ray food=green / obstacle=grey / kin=blue / non-kin=red, then
the interoception block=amber, then smell=magenta. Done without hardcoding aqua's
layout into the shared vivarium panel: `strip()` gained an optional per-index colour
list, threaded through `_draw_panel/_draw_inspector/_draw_nca_brain` (default None =
the old green, so the grid world is unchanged), and aqua computes the palette from
its own `n_rays_max/RAY_CH/INTERO/SMELL`. A compact legend sits in the world's
bottom-left corner. So the input strip now reads as coloured bands you can decode at
a glance — you can watch which modality is driving the CA. Full suite 193 green.

That closes the aqua continuous-physics arc (branch `aqua-continuous`): walls, food
and corpses are all float geometry, collisions/vision are analytic, senses are
heritable AND legible, and the agent-agent O(n^2) is gone. Open follow-ups if wanted:
a numba/bucket pass if very large populations are needed; angled walls (the AABB ->
capsule swap is isolated); merging the branch.

## 2026-07-21 — cortex: the design, then the world you can walk around

New arc, new playground. Two design notes first (`docs/cortex-design.md`,
`docs/cortex-world.md`), agreed over a long session with the user and written down
so we argue with them instead of re-deriving them. The thesis: **one player, one
big brain, and a genome that stops BEING the brain and becomes a program for
GROWING one.** Cells get morphogenetic outputs (divide / apoptose / retype /
become-organ / sprout-axon) on the same per-cell rule they already think with,
applied on a rare development tick; symmetry is broken by morphogen gradients,
growth is bounded by a global cell cost plus a local diffusing resource, migration
falls out of divide+apoptose for free.

Three things settled that were worth the argument. (1) "One long life with a
dead-end trigger" IS (1+lambda) with lambda=1 — something must change after the dead
end, that something is the genome, and regrowing from it is a new life; so the real
axes are lambda, who proposes the next genome, and what is inherited. (2)
Inheritance: **genome only**. Carrying the grown SHAPE forward is rejected because
the growth program would stop being expressed, stop being selected, and rot
unnoticed. (3) The task ladder is an **observer**, not a curriculum: probes run on a
full copy of the brain after death, in prepared arenas, with plasticity frozen — the
open world stays open, the probes are the thermometer. And the payoff that falls out
for free: growth reads sensory state, so **genome x world grows different brains**;
one program in two worlds is the headline experiment.

**Then stage 1: the world, playable, before any brain.** `cortex/` is written from
scratch (only the lessons come from aqua, not the code). Tiles STORE the map, floats
SIMULATE it: chunked generation from a seed, a float agent, circle-vs-tile collision,
sub-step marched movement with wall sliding. `python -m cortex.app` and you walk
around it with W/A/D — and the keyboard drives `World.step(thrust, turn, interact)`,
the exact call the brain will use, because playtesting a different interface than the
agent lives in is how you fool yourself.

Chunk generation is seam-free by construction: a chunk rasterises the features of the
3x3 block of chunks around it, each feature drawn from its OWN chunk's rng, so a rock
mass or a cave straddling an edge is identical from both sides regardless of visit
order (there is a test that generates the map in reverse and compares). Features:
rock masses with hollow rooms and doors, caves bored by rim-anchored random walks (so
a cave always has mouths and is never a sealed pocket), veins with gaps, breakable
rubble, tree clumps, pushable blocks. Open fraction lands ~0.69 across seeds —
structured but walkable. Material colours live in ONE table shared by the renderer and
(next stage) the vision sensor, so what the user sees is literally what the agent will
sense.

The world is bounded (512) by default even though the generator is infinite-capable
(`--size 0` unbounds it, verified). Not laziness: a geodesic scent needs a boundary,
and — the real reason — in an endless world MEMORY never pays, because "there is
always more ahead" beats "go back to the patch I remember", which would delete a whole
line of the economy before we ever test it.

Numbers: 0.016 ms/step (62k steps/s), 18 new tests, full suite 211 green. Verified:
same seed same map, visit order irrelevant, the body never overlaps a solid tile over
a 600-step random walk, a 40-cell move in ONE call stops at the wall instead of
tunnelling, a diagonal into a wall slides instead of sticking, moving costs more than
standing and standing is not free, death is by energy and not by age.

Next chunk: senses (DDA vision with hue/sat/brightness over 64 rays, odour channels,
touch) and the radar sensor panel — which doubles as a completeness test, because if a
human cannot play the world from the sensor panel alone, the brain has no chance.

**Same day, second chunk — food and the one verb.** The user played stage 1, said the
generator and the map are fine, and asked for the thing that was obviously missing:
food you can actually eat, plus doors and breaking, all on `E`. Built, and one design
point came out of it that is better than what was planned:

**Eating is a DECISION, not a collision.** Food is consumed by INTERACT, never by
walking over it (aqua ate on contact). The consequence matters for the whole poison
arc: an agent can walk straight across a nightcap and simply decline to eat it, so
"do not eat the purple one" is a perception problem instead of "never step there" —
a much cleaner thing to learn, and it keeps the motor and the appetite separate.

The rest as designed: ONE contextual verb, resolved in priority order — eat what is
in reach, else open/close the door in front, else chip the breakable, else shove the
pushable. So the brain will only ever decide WHETHER to act, never which of five
verbs. Breaking takes several hits (rubble 3, trunk 5) on a 6-tick cooldown, so a
shortcut costs real energy and time; a door refuses to close onto the agent's own
body; interacting costs energy even when it achieves nothing.

Food is float entities planted per chunk from a chunk-seeded rng (deterministic,
lazy, infinite-capable), with the KIND read off the surroundings: berries near
foliage, fruit by trunks, mushrooms in enclosed cave cells (rare — 16 of 625 spots
in a 200-cell radius, and worth ~3 berries). 70% of spots are RENEWABLE on a
1200-tick timer, the rest one-shot: the first is the only reason remembering a place
can ever pay, the second rewards exploration. A food kind is a colour + an odour
mixture + an energy value — never an "edible" flag, so the senses can report it
honestly next chunk.

One real bug the tests caught: food was validated at its CELL but planted at a
jittered off-grid offset, so morsels could end up inside a wall — unreachable food is
worse than no food. Now the jittered position is what gets validated.

HUD grew the parts a player needs: an energy BAR (energy is life, so it gets the most
legible widget), a live "E: eat berry / open door / break rubble (1/3)" prompt that is
computed by the same context rule the verb uses (a test asserts prompt and effect can
never disagree), a last-event line, the diet so far, the eat radius drawn around the
body and an outline on the cell E would act on. 13 more tests, 31 for cortex, suite
224 green.

**Playtest 2 — the fixes the player asked for.** The user walked the world and came
back with seven things, one of which was a genuine bug: **seed 1 spawned him inside a
hollow rock with no door.** A coffin, not a world. Fixed with a primitive that turned
out to answer three questions at once — `World.enclosure(y, x, limit)` flood-fills the
walkable region and reports (cells, has_door, bounded), which distinguishes a ROOM
(bounded, with a door) from a SEALED POCKET (bounded, no door) from the open world
(unbounded). Spawn now refuses sealed pockets; there is a test over 12 seeds.

The same primitive fixed the complaint that rooms behind doors were usually empty — a
door that leads to nothing teaches an agent that doors are not worth opening. Rooms
are now stocked with berry LARDERS, so opening one pays.

Each food kind now has a HABITAT, which is the economy working as designed: berries in
rooms and open ground, fruit only at a TRUNK (planted on actual trunks, since random
scatter almost never lands next to one), mushrooms only in walled-in cave cells (again
planted deliberately — random scatter finds tunnels far too rarely, which is why they
were 2.6% of food before and are 12-16% now). So colour and odour will correlate with
PLACE, and "go to the trees / go into the cave" become learnable strategies rather than
noise.

Food was halved as asked: one spot per ~200 cells^2 became one per ~375, by cutting
scattered spots from 9 to 1 and letting the habitat passes carry the rest.

**Damage is now visible, and visible to BOTH.** A chipped tile darkens toward its break
point via `World.damage_frac`, which the renderer multiplies into the material colour —
and vision will call the same function next chunk. That is the shared-colour-table
decision paying off: there is no way for the player and the agent to see different
worlds, because there is only one fact.

Felling a TRUNK now drops its crown (8-connected foliage flood fill, capped at 12 so
two touching trees do not chain into a clear-cut). A canopy floating over a stump reads
as a bug, and to an eye the leaves and the trunk are one object anyway. The first
version removed radius 1 only, matching the generator exactly — the test that placed a
wider crown caught it, and the flood fill is the version that survives a generator
change.

0.0047 ms/step, 38 cortex tests, suite 231 green.

**The coffin, take two.** The user was still shut in on seed 1 — and his diagnosis was
the one that mattered: the room *had* a door, but there was solid rock behind it (two
generated masses overlapping). The spawn guard accepted "bounded, but has a door" as an
exit, which is a claim about the wall, not about the world beyond it.

Fixed by making the enclosure fill walk what the agent can OPEN or BREAK, not only what
is walkable this instant: `enclosure(..., openable=True)` treats doors and breakables as
passable, so "is there a way out" is answered by actually going through them. The two
modes are both wanted — `openable=False` is still the ROOM test that decides where to
stock a larder, `openable=True` is the CAN I EVER GET OUT test the spawn uses. Spawns
now also require a region of at least 600 reachable cells, tested over 16 seeds.

**A second problem found while looking at the first, and an A/B that changed the fix.**
A one-cell doorway is 1.0 wide and the body is 0.9 across, leaving ±0.05 of clearance.
Measured: threading works from an offset of 0.00 and fails from 0.10 onward — passable,
as the user said he managed, but only with sub-cell precision, and hopeless for a brain
steering by heading. First instinct was to shrink the body to 0.30. The A/B said
otherwise: **corridor centring alone fixes it at the original 0.45**, from anywhere in
the right row, and at 0.03 assist as well as 0.06. So the body keeps its designed size
(about a wall's width) and gains `_centre_up` — when a hop is fully blocked, drift
`assist` toward the centre line of the cell across the direction of travel and retry.
Every candidate is still collision-tested, so nothing walks through a wall (there is a
test that drives at a gapless wall for 300 ticks and stays outside), and it only fires
when the move failed outright, so ordinary walking is untouched.

Worth recording as method: the instinct (shrink the body) and the measurement disagreed,
and the measurement kept a design decision the note had reasoned out properly. Aiming at
the right ROW is now enough — which is exactly the granularity a brain will have.

41 cortex tests, suite 234 green, 0.0038 ms/step.

## 2026-07-22 — cortex: the sensorium, a radar you can play blind by, poison, baselines

Four commits closing out the world. Design agreed first (docs/cortex-world.md), then
built in order, each with its own commit.

**Senses (`cortex/senses.py`), 412 slots, 0.21 ms per step+sense.** Vision is 64 rays
over 120 degrees marched by exact grid DDA — 1.9 degrees per ray, so a 2x2 tree ten
cells away spans about six rays and its shape is legible; at aqua's 3-12 rays it would
be one ray and a guess. Per ray: hue as (cos, sin), saturation, brightness, nearness,
solid. Hue is put on the CIRCLE deliberately: a raw 0..1 hue has a seam between the
berry (~0.97) and the wrap, which is exactly where the poison discrimination has to
happen. OPAQUE became separate from SOLID so leaves stop a ray but not a body.
Brightness is multiplied by the same `damage_frac` the renderer darkens with — one
fact, so the player and the agent cannot see different worlds.

Smell is GEODESIC: a BFS from the agent's four neighbours flows around walls, so a
morsel sealed behind rock smells of NOTHING (not faintly), and each odorant reports
intensity plus the egocentric direction of the shortest WALKABLE path. Recomputed only
when the agent changes cell or the map version changes, so smashing a wall re-opens the
scent.

**The panel is drawn from the sensor vector alone**, and TAB blanks the map. That makes
the completeness test from the design note literal — if a human cannot play from the
panel, the brain has no chance. Two tests keep it honest: move the agent without
re-sensing and the panel must not notice; zero the vector and it must go blank. The
radial scale is sqrt(d/reach), because a true-scale scope crushes the first few cells
into the apex and the first few cells are where everything happens.

**Poison.** A nightcap only ever replaces a BERRY: same habitat, identical odour,
different hue. It drains 30 energy at 0.15/tick rather than striking, because a morsel
that kills outright teaches nothing — the dead do not learn. Combined with eating being
a DECISION rather than a collision, avoiding it is a perception problem ("do not eat the
purple one") and not a navigation one ("never step there").

**Baselines — and three bugs they found.** `random` (correlated walk) is the floor;
`reflex` (twenty lines: follow the nose, avoid walls, eat what is under it) is the bar.
Over 8 seeds x 30k ticks:

```
who        runs   ticks  meals  gained  cells   diet
reflex        8   15038   18.5   286.5   2410   berr46 frui78 mush12 nigh12
random        8    7035    3.4    71.8   1580   berr7  frui15 mush5
```

Getting the reflex there took three fixes, and each was a real defect rather than
tuning:

1. **Escaping a wall needs COMMITMENT.** The first version re-decided "turn to the
   emptier side" every tick, and in a corner the two sides trade places, so it
   oscillated against the wall and walked 208 cells to random's 1275. Holding the
   escape turn for 14 ticks fixed it — the same reactive limit cycle, and the same
   cure, that the forage world taught us a week ago.
2. **Approach slowly.** At full speed the body covers 2.1 cells between the 6-tick
   interact attempts while the eat reach is 1.1, so a confident sprinter flies past
   its dinner. Easing off near the source took meals from 3.0 to 8.8 — more than any
   steering change.
3. **Use the `solid` channel.** The policy was fleeing its own food: a morsel two cells
   ahead is a near return like any other, so the escape branch fired every time it
   closed in and it ORBITED the meal. Weighting nearness by solidity took meals 8.8 ->
   18.5. That channel now has an earned reason to exist.

And a sensor bug the baseline exposed, which would have hit the brain identically: the
smell direction was quantised to four cardinal directions, useless in the last two
cells — forever told to go "east" when east is 40 degrees off. The true bearing is now
blended in as the morsel gets near, which is not cheating: at that range the shortest
walkable path IS the straight line.

The reflex ate 12 nightcaps out of 58 berry-like morsels — about the ambient rate,
i.e. it cannot tell poison from berry at all. That is the specific, measurable gap the
brain is supposed to close, and it is now on the record before the brain exists.

Sessions (human, baseline or brain) append one JSON line to `out/cortex/sessions.jsonl`
with the same schema, so they are directly comparable; the app logs a life on death and
on quit. Suite 266 green.

## 2026-07-22 (later) — cortex: the brain is in, and it does not beat the reflex yet

Stage 0 of `docs/cortex-design.md`, in three commits: the brain, the HUD column that
lets you look at it, and the lineage that carries a genome forward.

**The brain (`cortex/brain.py`).** Forked from `vivarium/nca.py` — the ideas were
earned there — but one BIG brain for one agent. A bounded lattice with a heritable
alive mask, cells carrying a TYPE that points at one of a few layer-stack RULES of
evolvable depth and width, ORGANS whose modality says what they hear and whose COLUMN
says which RAY they hear (retinotopy), effectors where the command is read, a hormone
channel gating every rule, and reward-modulated Hebbian plasticity on the readout and
on each rule's output layer. Signal must travel organ -> tissue -> effector, so the
shape has a job. State persists between ticks: the medium is the memory.

Two decisions worth recording. The lattice is 64 wide because retinotopy hands out ray
slots by column and 64 rays cannot be resolved in a narrower sheet. And `P_in` is
Xavier-scaled, because 412 inputs summed into 12 channels at N(0, 0.5) saturates tanh
on the first tick — the same regression that broke NEAT in the vivarium when its
sensor vector widened, and it would have been invisible here as "the brain twitches".

Two clocks, deliberately separate: `subticks` is how far a signal travels inside one
thought, `think_every` is how many world ticks pass per thought with the command held
in between. Seed brain 81 cells / 32 organs / 7.9k params / 0.98 ms per act; after 80
mutations 107 cells, 5 types, depth 5, 29k params.

**The HUD column.** The sheet in four views (type, activity, hormone, organs), the
command bars straight off the effectors, the reward, |dW|, and the rule output layers
as red/blue heat maps with their learned deltas folded in. T animates the sub-ticks so
you watch a thought cross the tissue — with a test that a watched brain and an
unwatched one end in identical states, because observing must not perturb. B hands the
wheel between the player and the brain mid-life, same world, same action tuple.

**The lineage (`cortex/lineage.py`).** (1+lambda) with lambda=1, which is what "one
long life plus a dead-end trigger" already is. Only the GENOME crosses the boundary;
lives rotate over world seeds so a lineage cannot learn one map (the `lamarck.py`
lesson). Each generation returns a PROFILE rather than a number — ticks, meals,
gained, distance, poison eaten, command entropy, |dW| — because at lambda=1 there is
one sample per turn and an LLM will eventually have to reason about it. Dead ends are
NAMED, not merely detected: `frozen-behaviour` (entropy collapsed — the reactive limit
cycle), `not-eating`, `not-learning`, `stalled`; a stalled lineage gets a structural
shove rather than another nudge, which is `scientist.py`'s basin escape pointed at
brains.

**The honest result.** 8 generations, 2 world seeds, 8000-tick cap: fitness 0 -> 56.1,
meals 0 -> 1.5 per life. The reflex baseline is 15038 ticks and 18.5 meals. So the
brain is nowhere near it, which is exactly why the baseline was measured first — this
is a 30k-parameter sheet steered by eight random mutations, and without the reflex
number on the record it would have been very easy to call "fitness went up 56x" a
result. The machinery is in place, reproducible and watchable; making it actually
forage is the next arc (and per the design note, the levers are development, the slow
signal layers, and the LLM writing the growth rule rather than rng nudging it).

Suite 296 green.

## 2026-07-22 (evening) — cortex: morphology, three signal layers, and an A/B that says no

The user played, confirmed the world is playable from the SENSE PANEL ALONE (the
completeness test passes), and put a human line in the table:

```
who       runs   ticks  meals  gained  cells   diet
reflex       8   15038   18.5   286.5   2410   berr46 frui78 mush12 nigh12
human        1   17276    9.0   106.0    930   berr5 frui2 mush1 nigh1
random       8    7035    3.4    71.8   1580   berr7 frui15 mush5
```

Worth staring at: the twenty-line reflex FEEDS BETTER than the human (18.5 meals to
9.0) while the human lives slightly longer on a third of the walking. Different
strategies, not different skill — and both are now on the record for the brain to be
read against.

**Morphology: the square is gone.** The seed brain was a rectangle with organs on its
perimeter, which is the worst possible starting shape — a rectangle has no morphology
to select on, and its organs inherit its symmetry. Now the seed is a blob GROWN by
accretion from one cell (every seed a different body), with interior LACUNAE punched
through it so signal has something to route around, and organs placed
morphologically: only some BORDER cells become senses (the rest are skin) and
effectors sit in DEEP tissue, so a command is read from cells no sense touches
directly. 340 cells, filling ~54% of its bounding box.

**Three signal layers, as designed.** ELECTRIC is the 3x3 neighbourhood every
sub-tick. CHEMICAL is three tail channels that diffuse into their neighbours and
fade — the only carrier with a spatial GRADIENT, which is what a growing tissue will
later read its position from and what an axon would grow along. HORMONAL is a global
vector from the body (energy, poison, last reward) that reaches every cell whether or
not it is an organ — a bloodstream, not a wire — and multiplies every rule's gain, so
a starving brain is physically not the same computer as a fed one. Each is separately
switchable, because a layer you cannot remove is a layer you can never justify.

**The HUD gained the cell-type readout the user asked for**, and it says more than
"there are five types": per type, cell count, depth, width, and two numbers read
straight off its rule — `self` (how much of its input weight is on its OWN state
rather than its neighbours) and `keep` (linearise own-state -> hidden -> own-output
and compare the diagonal against the mixing). Above 1 on both is a type that HOLDS
what it has: a memory cell. These are descriptions, not labels — nothing forces such
a type to exist, and the point of the numbers is to see whether one appears. Plus a
`chemistry` view (three channels as RGB, normalised per frame) so plumes are visible.

**And the A/B, reported as it came out.** Five generations, two seeds, 5000-tick
lives, all three layers versus electric only:

```
all three layers   fitness 26.0   meals 0.5   ticks 5000   H 1.17
electric only      fitness 43.0   meals 1.5   ticks 5000   H 0.94
```

The control arm won. This is NOT evidence that the layers hurt — 0.5 meals against
1.5 is one meal against three, from one brain seed and five random mutations, which
is noise wearing a number's clothes. What it IS is evidence that **the experiment is
underpowered and the layers are not yet paying for themselves**, and it goes in the
journal now, before the temptation to re-run it until it agrees. The layers were
argued for from biology and from what MORPHOGENESIS will need (a gradient to read
position from); that argument is still standing and still unpaid. Settling it needs
the probe suite from the design note — a skill profile per life instead of one noisy
number — and more replicates than a laptop evening.

Suite 304 green.

## 2026-07-22 (night) — cortex: the genome stops being a body

Stage 2 of `docs/cortex-design.md`. Until now the genome CARRIED the brain — a 340-cell
mask, a type map, organ flags. Now it carries a **24-cell seed and a program**, and the
tissue grows itself to 400-600 cells over a life. That is the thing the whole design
note was pointing at: what is inherited is small enough to be read and written, by
evolution now and by an LLM later.

**How it works.** A rule's output is C state channels plus six MORPHOGENETIC ones —
grow, direction y, direction x, die, be-a-sense, be-an-effector — computed by the very
same little network the cell thinks with, from the very same 3x3 perception. There is
no separate growth brain. They accumulate over the sub-ticks of a thought and are read
on a rare DEVELOPMENT TICK (every 8 world ticks in the embryo, every 120 after). The
rarity is not a compromise: a shape change rebuilds the phenotype's index arrays, so a
per-tick version would cost more than the thinking.

**The living body is the phenotype's, never the genome's.** A life can grow, kill and
re-purpose cells all it likes; `Genome.mask` and friends are untouched, with a test that
says so. That is what makes "only the genome crosses the boundary" true rather than
aspirational — if a life could edit its genome, the phenotype would silently become the
heredity and the growth program would rot unselected.

**Four things went wrong, and each was informative.**

1. *Nothing grew at all.* A fresh rule's morph outputs sit near zero, so no cell ever
   asked for anything and "development" was a no-op that had merely made the brain
   smaller. Fixed with an INNATE BIAS in the seed program: an embryo grows unless told
   not to. Mutation shapes it from the first generation.
2. *Everything grew, then everything churned.* Unlimited, a 24-cell seed hit 2344 cells;
   with an aggressive death threshold it turned over 13611 births and 11572 deaths in one
   life and settled nothing. A body has to change slowly enough to mean something, so
   births and deaths are rate-limited per development tick — with deaths on a smaller
   budget than births, because with equal budgets the sheet sat at its seed size forever.
3. *The resource was a signal channel, and the rule flooded it.* Every cell wanted to
   divide and two had the means. A neuron does not write its own blood supply: RESOURCE
   is now metabolism, outside the cell's channels entirely — fed at the BORDER in
   proportion to how full the agent is, seeping inward, spent by division. And it had to
   seep only a LITTLE: at three passes of half-mixing the fresh supply was carried off
   the border and pooled in the interior, which is precisely where there is no room to
   divide, so the cells that could grow had no means and the cells with means had no
   room.
4. *Ranking by urge before checking feasibility grew nothing.* The loudest cell was
   usually one buried in the middle with nowhere to put a daughter; it burned the whole
   budget failing. Feasibility first, then rank.

**What it produces.** Bodies of 400-600 cells from a 24-cell seed, and — the part worth
looking at — different programs grow visibly different bodies: across four fresh genomes
the sensory share came out 60%, 32%, 10% and 5%, with effector counts from 7 to 429. The
`resource` view shows the metabolic gradient plainly: bright at the rim, dark in the
depths. That gradient is why a body is not a disc.

Organ FATE is read from the chemistry where a cell stands — `organ_map` says which
chemical means which sense — so a cell's position decides what it becomes. That is the
French-flag model, and it is what makes an eye REGION something that can form rather
than something we place. It also finally gives the chemical layer a job that the failed
A/B could not: development reads position from it.

Also this session: the sub-tick animation now runs on its own clock with `[`/`]` (tied
to the world's it flashed a whole thought past in a third of a second), the sheet grew
into the space available, and CLICKING A CELL opens the perceptron behind it — role,
every channel it is holding coloured by signal layer, and its whole layer stack as heat
maps. An inserted near-identity layer shows up as a literal diagonal.

14 new tests for development alone, suite 319 green.

## 2026-07-23 — cortex: several cell types in the seed, and the observer

**The seed carries three cell types now, laid out as REGIONS.** One type meant every
cell ran the same network until a 4%-per-mutation split happened to fire, so a fresh
brain was uniform tissue and clicking any two cells showed the same thing — for a good
reason nobody had written down. Types are placed by a small Voronoi over the seed
cells, which matters because a daughter inherits her mother's type: patches grow
outward with the tissue rather than staying a speckle. It surfaced a phenomenon worth
naming — TYPES COMPETE DURING DEVELOPMENT. On two fresh genomes one type ended with 416
and 421 of ~465 cells, because growth is driven by each type's own rule and whichever
asks to divide hardest takes the tissue.

**Two HUD corrections, both the user's, and one of them mine to own.** Clicking cells
showed identical panels because a layer stack belongs to a TYPE — true, but the panel
said nothing about it and showed nothing per-cell. It now says "SHARED by its N cells"
outright, and shows what genuinely varies: what the cell is ASKING FOR (its six
morphogenetic outputs, plus its resource) and what is DRIVING it (neighbourhood weights
times what those neighbours actually hold). And I had written that aqua draws NEAT and
that a node-and-edge picture would not work here. aqua's brain is the NCA, and
`vivarium/app.py::_draw_rule_net` DOES draw a cell rule as a net — it makes a dense
stack legible by averaging the nine neighbour slots down to C and drawing only the
strongest edges. Both tricks are now copied, and the net sits beside the heat maps
rather than instead of them.

**The observer (`cortex/probes.py`).** Six fixed exams a brain sits at the end of its
life, on a DEEP COPY, with plasticity FROZEN, each probe getting its own fresh copy so
order cannot matter, and the live world never touched. `open` (food over open ground),
`corner` (food behind a wall with one gap), `door`, `rubble`, `poison` (a nightcap
under your nose and a berry further off), `patience` (no food at all — standing still
is half price, so the score is energy kept).

The calibration is the point, and it immediately earned its keep:

```
             open  corner  door  rubble  poison  patience | mean
reflex       1.00   0.23   0.10   0.10    0.30    0.17    | 0.32
random       0.15   0.17   0.07   0.07    0.60    0.00    | 0.18
fresh brain  0.18   0.19   0.65   0.66    0.60    0.00    | 0.36   (an earlier run)
```

Read the MEAN and the fresh brain beats the reflex. Read the PROFILE and it obviously
does not: it scores on `door` and `rubble` only because it holds the interact button
down, and it is a fifth of the reflex on `open`, which is the actual skill. That is
exactly the failure mode the suite was built to catch, and it caught it on the first
run. The `poison` number is the other one to watch: the reflex scores 0.30 because it
eats BOTH morsels — it follows its nose and the nightcap is the nearer smell — which is
the same blindness the live world measured (12 nightcaps in 58 berry-like meals), now
isolated into one number.

Three things had to be fixed for the probes to measure anything. The first `poison`
arena put the two morsels equidistant and everything scored exactly 0.50 by eating
neither: two equal scent sources average to a bearing straight between them and the
agent oscillates — the forage limit cycle, reproduced in a test tube. True, useless as
a measurement, so the temptation is asymmetric now. The `corner` detour was longer than
the scent radius, so there was no cue at all. And the big one: **a shut door was a hard
scent wall**, which makes food behind one undetectable and the door mechanic
unlearnable in principle. A door is not airtight and neither is rubble — scent now
seeps through both with a distance penalty (7 and 11 cells), so you can tell there is
something back there and it smells further away than it is.

The strip render is what made all three obvious: you can see the reflex walk straight
to its food in `open`, circle in the left half of `corner` never finding the gap, and
go directly to the nightcap in `poison`. A probe that can only be read as a number is
a probe we would not have trusted.

The lineage now reports a skill profile per generation, and fitness leans on the LIFE
with the exam as a tie-break, so a brain cannot win by being good at tests and bad at
living. 19 new tests, suite 342 green.

**Same day, the suite rebuilt as a ladder.** The user looked at the strip and said what
the picture was already saying: start with tests where the player just EATS — no doors,
no poison — only different surroundings so it learns terrain and smell. He was right,
and the reason is worth keeping: the first suite led with door, rubble and poison, and
every player scored ~0.1 on five of six probes. A whole suite carrying one probe's
worth of signal. **A measurement has to have resolution where its subject actually is.**

The default suite is now a foraging ladder — `near` (four cells dead ahead), `open`
(far), `side` (off the line, it has to turn), `behind` (out of sight entirely: smell
only), `corner` (a wall with one gap), `maze` (a few short walls), `patience` (no food
at all) — and the arenas differ only in where the food is and what stands in the way.
door / rubble / poison are kept in `ADVANCED`, worth running the day something can
forage and not before.

```
             near  open  side  behind  corner  maze  patience | mean
reflex       1.00  1.00  1.00   1.00    0.23   0.23    0.17   | 0.66
random       0.35  0.15  0.32   0.35    0.17   0.13    0.00   | 0.21
fresh brain  0.30  0.16  0.03   0.00    0.11   0.21    0.00   | 0.11
```

*(2026-07-24: re-measured. `reflex` reproduces exactly; `random` does not — it is
0.35 0.23 0.32 0.35 0.24 **1.00** 0.00 | 0.36, because a blind stagger ACES `maze`.
The arena is small enough that wandering finds the food, so `maze` is not measuring
what it is named for and `corner` is the honest terrain probe of the two. Both
players are deterministic; `python -m cortex.probes --player reflex|random`
re-measures either in one command.)*

That is a ladder with rungs. The reflex aces open ground and stumbles on terrain, so
beating 0.66 means doing what a nose-follower does AND coping with a wall. A fresh
brain sits at 0.11, below a coin flip — and the generation strips show why in one
glance: it circles in place in all six arenas.

Also this session, so a run can be LOOKED at rather than only read: `python -m
cortex.probes --player reflex|random|brain` prints the profile and writes the strip;
the lineage writes one strip per generation into `out/cortex/probes/genNNN.png`; and
`python -m cortex.app --probe corner` drops you (or a brain) into a probe arena LIVE
with the whole HUD, so the exam can be walked by hand.

One real bug on the way: `import os` inside a branch of `lineage.run` made `os` a local
for the whole function, and the genome save at the end died with UnboundLocalError. The
module already imported it.

The next milestone is written into `docs/cortex-design.md` and is deliberately narrow:
an LLM writes a growth program, the body develops from it, moves, and eats.

Suite 344 green.

## 2026-07-23 (later) — the diagnostic, and axons

**The diagnostic first, because it was cheap and it decided the design.** Grow ONE body,
then sit that same body at the `open` probe with different think-times — which isolates
"can the signal get there" from "did it grow well":

```
brain 0 (470 cells, span 30, 38 sensory / 242 effector)
   subticks   10:0.09   20:0.16   40:0.26   80:0.26   160:0.34
brain 4 (488 cells, span 33, 128 sensory / 222 effector)
   subticks   10:0.12   20:0.18   40:0.23   80:0.24   160:0.24
```

Two things fall out. **The relay bottleneck is real**: eight times the think-time
roughly doubles the score, which is what you would expect if the problem is that a
signal crosses one cell per sub-tick and the body is thirty cells wide. And **it is not
the only problem**: even at 160 sub-ticks — eight times the cost — the brain sits at a
third of the reflex's 1.00. The other number in that line says where to look next:
**242 effectors out of 470 cells.** The readout pools over ALL of them, so whatever does
arrive is averaged with half a body that heard nothing. That fix is three lines and is
deliberately NOT bundled with this one, because changing two things at once measures
neither.

**Axons (`cortex/brain.py`).** A seventh morphogenetic output, `SPROUT`. A cell that
asks for one, and can pay in resource, grows a GROWTH CONE: it walks up the guidance
chemical — the same gradient development already reads its position from — and wherever
it ends up, if that is far enough and alive, becomes the far end. The wire lands as an
additive drive exactly like a sense, so nothing about a cell's rule has to change to
receive one. Weights are reward-gated Hebbian (fired together, and the reward landed),
decay every development tick, and are pruned below a floor: use it or lose it, so a
topology is lost as well as grown. Axons are stored as CELL COORDINATES and re-pointed
on every reindex, dropping any whose end died — a wire that silently re-pointed at
whatever cell inherited its row number would be worse than no wire.

Two blockers, both the same shape as earlier ones. `sprout_bias` at 0.10 against a 0.30
threshold meant no cell ever asked, so the whole mechanism was dead code that mutation
could not even find — a wire never grown is never selected for or against. And the cone
stopped at the first local peak, which in a chemical the cells write themselves is two
cells away; it now always MOVES, to the best neighbour it has not visited.

**The A/B, reported as it came out:**

```
brain 0  axons on  0.40   off  0.24      (behind 1.00 vs 0.11, open 0.28 vs 0.15)
brain 4  axons on  0.10   off  0.14
```

One clear win, one small loss, n = 2. That is not evidence. What IS established is the
diagnostic above, and that the mechanism works: wires grow, carry signal (cutting them
all changes the state), learn, prune, die with their cells, and are visible as a green
web on the sheet with `axons N mean length 7.4 cells` in the HUD. Whether they pay is
an open question needing more genomes than an evening.

One regression I caused and the tests caught: the new axon header line only appeared
once a wire existed, which pushed the fixed sheet box down by its own height — the exact
twitch that box exists to prevent. It is drawn always now, zero included.

8 new tests, suite 352 green.

## 2026-07-24 — cortex: the read-out was muffling the turn, and the growth program gets names

Three things this session, in the order they had to happen: a diagnostic that changed
what "fix the read-out" meant, the refactor that made a growth program something you
can write, and the loop that lets an LLM write one.

**The diagnostic, and how it nearly lied.** The previous session left two suspects for
why a brain circles: the command is pooled as a mean over ALL effector cells (242 of
470 in one body), and a silent sheet SPRINTS, because sigmoid rests at 0.5 and the seed
added another +0.5 of bias on top. The obvious test is to run the probe ladder with
each changed. I did, on three grown bodies, and got this:

```
brain 0   pool 1.00 -> 0.23   pool 0.10 -> 0.39     (better)
brain 1   pool 1.00 -> 0.34   pool 0.10 -> 0.48     (better, but 0.25 gave 0.14)
brain 2   pool 1.00 -> 0.34   pool 0.10 -> 0.13     (worse)
```

Which settles nothing. The ladder moved as much BETWEEN SETTINGS ON ONE BRAIN as it
moved between brains — the signature of a measurement reading luck. These brains have
never been selected for anything; what a probe scores is mostly whether this particular
random tissue happens to drift somewhere useful.

So measure the MECHANISM instead of its consequence. The claim was "a mean over
everything attenuates the TURN command", and that is a claim about the command stream,
not about food. Drive one grown body through an arena and look at the spread of the
turns it issues:

```
             n_eff      pool 1.0     pool 0.25      pool 0.1     (turn std / peak-to-peak)
brain 0         71   0.050/0.470   0.104/0.656   0.177/0.791
brain 1        168   0.077/0.495   0.154/0.930   0.249/1.336
brain 2        137   0.094/0.425   0.152/0.962   0.174/1.061
brain 3         37   0.027/0.272   0.061/0.710   0.060/0.737
```

Every brain, same direction, two to three times. THRUST survives the mean because it is
unsigned and a whole sheet can lean on it together; TURN does not, because it has to
swing and a handful of cells pulling right gets averaged with three hundred holding
zero. `pool_frac` is 0.25 now, the loudest slice.

The honest report is that this does NOT move the ladder yet, and that is written into
the commit rather than smoothed over. Steering authority is a precondition, not a
policy. There is nothing yet that wants to steer.

The sprint half came out differently and is worth recording as a lesson about the
ladder itself. Removing it entirely (`thrust_rest` 0.15) collapses every score to 0.11
mean — below random. Which does not mean the sprint was good; it means `near` and
`behind` were handing out free marks to any brain that merely existed, and with the
free marks gone there is nothing underneath. Default is 0.50: the arbitrary +0.5 nudge
gone, the sigmoid's natural rest kept, and it is a named knob now so a lineage can
argue with the choice.

**The growth program gets names (`cortex/program.py`).** Writing that diagnostic
surfaced something nobody had written down. Every constant governing how a body
DEVELOPS — the thresholds a cell's request is judged against, the innate urges it
starts from, the pace of development, the metabolism paying for it, the chemistry, the
axon rules, the learning rates — lived in `brain.Config`. Shared by every lineage,
identical in every generation, and untouched by `mutate`. So:

> **Evolution could not reach development at all.** Eight generations of a lineage
> jittered weights, and every brain in every one of them grew under exactly the same
> developmental physics.

That is a substantial part of the reason the lineage does not improve, and it was
invisible because the constants were *right there*, well commented, and being read
every development tick. Being read is not being selected.

Forty named scalars with a range and a sentence each. `Config` still says which
mechanisms exist and how big the machine is; a `Program` is the heritable override over
it and `Config.merged(program)` is what a phenotype runs on, so all the existing
`cfg.X` reads work untouched. Mutation jitters relative to each field's RANGE rather
than its value — an absolute-fraction jitter pins any knob that ever reaches zero,
which is the quiet way a search space collapses.

The innate urges had to move out of the weights to get here. They were added into every
rule's output bias at birth: identical arithmetic, since the morph tail is a raw mean,
but smeared across every cell type and thereafter indistinguishable from a weight. Same
numbers, one name each, and `urge = what the cell asked for + the program's bias` is
now a line you can read.

This is the adapter `docs/cortex-design.md` names as the one thing missing before an
LLM can write a growth program.

**The lab (`cortex/lab.py`).** The model gets a skill profile, what the life did, and a
summary of the body that grew — cells, sensory, effector, axons, born, died — and
writes back a program. It never sees a weight, and there is a test asserting the prompt
contains no `P_in`/`P_out`/`b_out`/rule matrix, because that is the sort of thing that
leaks in later by accident. Every candidate grows from the SAME fixed random weights,
so a score difference is attributable to the program. Generation 0 is the default
program on identical terms, because without it "better" has no referent. `--proposer
rng` is the control arm: is an LLM actually better than chance at this?

The smoke run made the previous commit's case by accident. Default program: 345 cells.
One random mutation: 998 cells, 649 of them sensory. Another: 329 cells with FOUR
sensory cells and a single effector. The program reaches morphology, plainly.

**The track, in the live HUD (R).** The probe strips have always drawn one and it is
the most legible thing on them. The live world had no equivalent, so the same failures
were invisible in the world they actually happen in. It fades by alpha AND hue so
direction survives a still screenshot — and the very first lab screenshot shows the
agent walking a closed loop, which is the whole diagnosis in one glance.

Each lab iteration writes that screenshot plus a probe strip into the run directory and
rebuilds a contact sheet as they arrive, so a run can be watched while it runs.

40 new tests, suite 392.

## 2026-07-24 (later) â€” the brain was never being taught, and the exam was throwing away what it learned

The user's read: "we are not letting the brain learn â€” you do not even need an LLM for
this." Correct, and the measurements went further than expected. Four findings, each of
which invalidated part of the previous session's picture.

**1. The reward stream is one-sided.** Count the feedback rather than the score. Over a
life of 4000 ticks:

```
brain    ticks  meals   r>0    r<0   sum r+    |dW|
0         4000      0     0   4000    0.000   0.0346
1         4000      0     0   4000    0.000   0.0440
2         4000      0     0   4000    0.000   0.0269
reflex    4000   8-15
```

Zero positive reward events out of four thousand. Living costs energy every tick and the
agent never eats, so the three-factor rule was never failing â€” it was working perfectly
and faithfully teaching the tissue that **everything it does is bad**, uniformly, with
no positive example anywhere to point at. `|dW|` is not zero, which is the tell: the
machinery ran, on a signal that only ever said "stop".

Fix: reward is SURPRISE about energy, not energy. A running mean subtracted
(`reward_baseline`, the standard actor-critic trick) makes the predictable drain worth
nothing to learn from and leaves a meal â€” a spike the mean cannot track â€” as the only
thing that moves anything. Three lines.

**2. The cradle, and why it had to come before reflexes.** The user's other idea was to
put the brain somewhere it only has to walk forward to eat, and their better one was to
give it innate reflexes â€” hardwired but retunable. Both are right, but there is an order
argument: you cannot tell a broken learner from a starved one by looking at what it
learned, and a reflex would score well while hiding an inert learner forever. So
`cortex/cradle.py`: a small world thick with food and no poison, a bench and not a
benchmark, with exactly one question â€” *given reward that actually arrives, does
within-life learning contribute anything?*

Asking it needed care, because a brain both GROWS and LEARNS during a life and "it got
better" cannot be attributed to either. The ablation splits them: one snapshot, two
copies at the same exams, one with the within-life weight deltas zeroed and everything
else identical â€” same body, same cell states, same wiring. The difference is learning
with growth held fixed. (Axon weights are deliberately not reset; a wire is topology
that was grown, not a delta that was learned, and cutting the wiring is a different
question.)

**3. The exam had been throwing the learning away â€” every probe ever run.** The first
cradle run returned a delta of exactly 0.00 on all seven probes. Exactly zero is not a
result, it is a symptom, and it was:

```python
eff = g.P_out + self.dW if self.cfg.plastic else g.P_out
```

`cfg.plastic` was gating two different things â€” "go on learning" and "use what you
learned" â€” and `probes.snapshot` turns plasticity off for an exam, as it must. So every
probe ever run examined a brain with its whole life's learning discarded, while
`docs/cortex-design.md` says in as many words that the exam is of *the developed,
TRAINED mind*. The same held for the per-type rule deltas. One flag doing two jobs, and
the wrong job won.

**4. And the two time constants of learning were wrong by one to two orders of
magnitude.** With the exam fixed, the delta became measurable and was still ~0. The
arithmetic says why, and it is not subtle:

```
plastic_elig 0.5, decayed once per THOUGHT  -> half-life 1.0 ticks
                                                10 ticks back: weight 0.001
plastic_forget 0.002, per TICK              -> over 2500 ticks: 0.007 of it left
measured gap between meals                  -> 312 to 833 ticks
```

A meal is the END of an approach; the turns that aimed the agent happened tens of ticks
earlier. The actions that earned the reward were outside the eligibility trace
**entirely** â€” the only thing ever credited was pressing E while already standing on the
food. And a delta kept 19-53% of itself between meals, so no lesson survived to meet the
next one. Both defaults were set for a dense-reward regime; this world rewards once in
several hundred ticks.

Swept in the cradle, scored by ablation so growth could not take the credit:

```
elig 0.5   forget 0.002    delta  0.00     credit never reaches the cause
elig 0.95  forget 0.0002   delta +0.12     near +0.54, behind +0.24
elig 0.98  forget 5e-05    delta -0.12     credits everything, learns noise
```

An optimum rather than a trend: credit has to survive as long as an approach and no
longer. New defaults are the middle row.

**Where that leaves it, stated more carefully than the sweep suggests.** Re-run at n=3
with the new defaults and the upkeep folded into the reward, the per-brain deltas are
-0.00, +0.07, -0.01 â€” mean +0.02, which is noise. But individual probes swing by up to
0.9 either way. So two separate things are now true and only one of them is good:

> Learning is **not inert** â€” wiping it changes behaviour substantially. Learning is
> **not yet useful** â€” the mean contribution is zero and its sign varies by brain.

Which is a much sharper diagnosis than "it does not learn", and it points somewhere
specific. The credit now reaches the behaviour; the problem is that the behaviour and
the meals are **uncorrelated**. The agent bumps into food incidentally, so what gets
reinforced is whatever it happened to be doing, and a gradient over noise is noise.

That is the real argument for the user's innate reflexes, and a better one than "it
would raise the score": an innate approach makes eating a CONSEQUENCE of a behaviour,
which is the precondition for reinforcing that behaviour at all. Reflexes do not replace
learning here â€” they manufacture the correlation learning needs. And with the ablation
harness built, "the tissue improved on its reflexes" can now be told apart from "the
tissue rode its reflexes", which was the thing to be afraid of.

Also this session, from reading a failing test rather than the code: the brain's own
upkeep was charged AFTER the reward was measured, so a brain never felt the cost of its
own tissue. Upkeep is now inside the tick's ledger.


## 2026-07-24 (later still) â€” innate reflexes, and the brain finally forages

The user's idea, and their words for what it had to be: "hardwired, but retunable with
effort â€” so the brain already does the basic things at the start". Built as
`cortex/reflex.py`. The result is the best thing this playground has produced, and the
route to it went through a blocker nobody had noticed.

**Why reflexes, stated so we do not misremember it.** Not to raise the score. The
previous entry ends with the measurement that matters: credit now reaches the behaviour
that earned a meal, and learning still contributes nothing, because **what the agent
does and whether it eats are uncorrelated**. It blunders into food, so what gets
reinforced is whatever it happened to be doing. An innate approach makes eating a
CONSEQUENCE of a behaviour, which is the precondition for reinforcing that behaviour at
all. Reflexes do not replace learning. They manufacture the correlation learning needs.

**An arc, not a shortcut.** The ten-line version is `command = policy(sensors) + sheet`,
and it would make the tissue a decoration on a hand-written policy while every number
improved. So a reflex here is assembled out of what the brain already is:

1. a SENSORY PROJECTION â€” each odorant's forward / lateral / intensity component drives
   a reserved state channel, so the organs that smell are holding the signal;
2. a PIONEER TRACT â€” axons from those nose organs to the effectors, grown at the first
   development ticks. Not `_sprout`'s growth cones wandering up a chemical: a pioneer
   has a target and is guided to it, which is how a species is born already able to do
   something. Both end up as ordinary axons in the same list;
3. a COHERENT READ-OUT â€” that channel wired to the command it means.

All three are ordinary weights and ordinary wires. Mutation jitters them, plasticity
moves them, and the cell rules write to the very same channels, so the tissue can
suppress, amplify or repurpose any of it. The one asymmetry is decay: an innate tract
loses 0.001 per development tick against a grown axon's 0.02, and is never pruned. That
asymmetry is the whole of "retunable with effort" â€” the tissue can argue with a reflex,
it simply cannot let one rot by neglect.

Only SMELL gets an arc. Smell is the one sense here that says which way to GO â€” it is
geodesic, so its direction is a walkable path around walls rather than a bearing through
rock. Vision would require the agent to already know which hue is food, and that is the
poison experiment's entire question; an innate arc must not answer it in advance.

**The blocker, which was worth the whole detour.** The first tracts refused to grow: 0,
3 and 3 of a requested 16 across three brains. `organ_map` decides which chemical region
becomes which sense and it is drawn at random from nine modalities â€” so **most fresh
genomes had no odour organ anywhere and could not smell at all.** Every brain we have
ever measured was, with high probability, noseless. A reflex has to bring the organ it
needs: `install` now claims ONE chemical region for smell and leaves the rest to decide
their own fates.

**The measurement, three arms, eight brains.** `brain` as it lived; `no reflex`, the
same brain with the whole arc removed â€” sensory drive, tracts and read-out â€” and
everything it grew and learned left alone; and `reflex only`, the twenty-line baseline
policy with no brain at all. The second says whether the arcs are load-bearing. The
third is what stops us congratulating a decorative brain for riding one.

```
brain   tracts   near  open  side  behind  corner  maze  patience | mean
0            7   1.00  0.24  0.35   0.33    0.24   0.11    0.34   | 0.37
1           15   0.33  0.32  0.09   0.26    0.24   0.06    0.00   | 0.18
2           16   1.00  1.00  1.00   1.00    0.34   0.01    0.30   | 0.66
3           13   1.00  1.00  0.35   1.00    1.00   0.02    0.41   | 0.68
4           16   0.35  0.32  0.35   0.35    0.13   0.08    0.16   | 0.25
5           16   1.00  1.00  1.00   1.00    0.22   0.19    0.29   | 0.67
6           16   1.00  0.27  0.23   0.00    0.24   0.11    0.00   | 0.26
7           16   1.00  0.31  0.15   0.32    0.22   0.12    0.00   | 0.30

delta from the arcs, mean of 8:
            +0.39 +0.46 +0.23  +0.14   +0.20  +0.01    +0.06   | +0.21

reflex POLICY (no brain at all):
             1.00  1.00  1.00   1.00    0.23   0.23    0.14   | 0.66
```

Read it in three passes.

**The arcs are load-bearing.** +0.21 mean, and positive on every one of the seven
probes. Brain 5 goes 0.05 -> 0.67 when its arc is restored. This is not a tuning gain;
without the arc these brains circle.

**Three brains now forage.** 2, 3 and 5 take all four foraging rungs at 1.00 â€” reach
food dead ahead, far over open ground, off to the side, and out of sight behind them on
smell alone. No brain in this project has ever done that. The milestone in
`docs/cortex-design.md` â€” *an LLM writes a growth program; the body develops from it,
moves, and eats* â€” has its second and third clauses for the first time.

**And the tissue is doing something the policy cannot.** Brain 3 scores **1.00 on
`corner`** against the reflex policy's 0.23 â€” food behind a wall with one gap, where a
nose-follower walks into the wall and stays there. Its overall 0.68 edges the policy's
0.66. That is one brain out of eight and I am not going to call it a trend, but it is
the specific thing the three-arm design was built to be able to see: not "the brain
scores well", but "the brain scores well WHERE THE REFLEX DOES NOT", which cannot be
explained by riding it.

**What is not true.** The mean brain is 0.42, well under the policy's 0.66, and the
spread is enormous (0.18 to 0.68) across nothing but the weight seed. `maze` stays noise
â€” a random walk aces it, as recorded earlier. And meals in the real world are still 1-3
per 2500 ticks against the reflex's 8-15, so scoring on the ladder has run ahead of
living in the world; the probes are small arenas and the world is not.

The obvious next question is now finally worth asking, because there is variation with a
cause: with reflexes making meals correlate with behaviour, does within-life learning
start contributing? The ablation harness for that already exists â€” `--unlearn` against
`--no-reflex`, two knobs, four arms.

Watchable, since that is the rule here: `cortex: WATCH a REFLEX brain play` and `WATCH
the same brain WITHOUT reflexes` side by side, the brain column counting innate tracts
apart from grown ones, and R for the track â€” a forager draws a line to its food where a
circler draws a closed loop.

16 tests, suite 423.


## 2026-07-24 (last) â€” learning contributes now, and it needed the reflexes to

The question left open by the previous entry, answered at n = 8. Two experiments,
because the first one turned out to be asking something slightly different from the
claim, and noticing that was the useful part.

**The claim, restated so it can fail.** Reflexes were not built to raise a score. They
were built because learning contributed nothing while *what the agent does and whether
it eats were uncorrelated* â€” it blundered into food, so what got reinforced was whatever
it happened to be doing. If that diagnosis is right, then adding innate arcs should make
LEARNING better, not merely make the agent better. If learning's contribution is
unchanged, the arcs bought foraging and the correlation story was wrong.

**First experiment (`--matrix`): two knobs, four arms.** One life per seed, and the
learning delta measured twice â€” once with the arcs intact at exam time, once with them
cut.

```
                              near  open  side  behind  corner  maze  patience | mean
learning delta WITH arcs     +0.10 +0.24 +0.14  +0.25   +0.15  +0.02   +0.05   | +0.14
learning delta WITHOUT arcs  -0.09 -0.15 +0.03  +0.19   -0.02  -0.02   +0.02   | -0.01
```

Against +0.02 before the reflexes existed. Positive on all seven probes with the arcs,
and around zero with a wandering sign without them. So what the tissue learned is a
MODULATION OF ITS REFLEX: it pays only when the reflex is there to be modulated, which
is a good deal more interesting than a general improvement, and it is what you would
expect of tissue that spent its life working with an arc.

**But that is the exam-time question, not the life-time one**, and the claim was about
the life. Cutting the arcs at exam time asks "is this learned delta useful with or
without reflex support NOW". The argument is that arcs make the meals a life collects
point at something, so learning has a consistent thing to reinforce â€” which can only be
tested by living twice.

**Second experiment (`--two-lives`).** Same weight seed, two whole lives: once with its
innate arcs, once with `reflex_gain = 0` â€” a genome born knowing nothing â€” each life's
learning delta measured by ablation inside its own world.

```
                              near  open  side  behind  corner  maze  patience | mean
LIVED, born knowing           0.83  0.56  0.44   0.53    0.33   0.09    0.19   | 0.42
LIVED, born blank             0.36  0.10  0.16   0.30    0.08   0.05    0.13   | 0.17
learning delta, born knowing +0.10 +0.24 +0.14  +0.25   +0.15  +0.02   +0.05   | +0.14
learning delta, born blank   -0.16  0.00 +0.03  +0.03   -0.03  -0.01   -0.02   | -0.02
```

And the single line that explains the whole table â€” meals per life, all eight brains:

```
brain          0    1    2    3    4    5    6    7
born knowing   3    1    3    3    1    1    1    1
born blank     3    0    0    0    0    0    0    0
```

**A brain born blank does not eat.** Seven of eight took zero meals in 2500 ticks, so
there was no reward, so there was nothing to learn from, so the delta is -0.02 â€” the
same "learning is not inert but not useful" that this whole thread started from. A brain
born knowing eats, and its learning contributes +0.14.

That is the argument confirmed end to end, and by the mechanism it named rather than by
the outcome it hoped for. The reflexes did not teach the tissue anything. They made the
world teachable.

Note brain 0, the honest exception: it ate three meals either way and is the one brain
whose learning delta barely differs between the arms. One data point, but it points the
right way â€” where the meals arrived anyway, the arcs mattered less.

**What is still not true.** The mean brain lives at 0.42 against the reflex policy's
0.66. Real-world meals are 1-3 per 2500 ticks against the policy's 8-15, so scoring on
the ladder is still running ahead of living in the world, and the probes are small
arenas while the world is not. `maze` remains noise. And "learning contributes +0.14"
is a claim about a profile mean at n = 8 with per-brain spread from -0.02 to +0.35;
it is a direction, not a number to quote.

The next thing worth doing is now obvious and was not before: put the LINEAGE and the
LLM lab back on top of this. Both were measuring a brain that could not be taught â€”
`cortex/lab.py`'s eight generations optimised body size while learning ran on a dead
signal â€” and neither has been run since any of it worked.

6 tests, suite 429.


## 2026-09-14 — cortex-atlas: why the brain stopped growing, and the sheet as a map

New branch `cortex-atlas`, after a long pause. The user's read on coming back: "we
over-complicated the brain from the previous iteration and did not adapt the rest to it,
so it fell apart; the earlier things probably worked because the perceptrons did all the
work while the cellular automaton was never really doing its share." The measurements
below say the same thing, and point at what to do about it.

**Diagnosis, in three numbers that were not on the record.**

1. *Selection is deaf.* Take brain 2 (one of the three foragers from 2026-07-24), live
   it 6000 ticks on eight worlds, and separately live eight of its children on world 0:

   ```
   same genome, 8 worlds:    fitness 214  9  70  8 174 102 100  92   sd 67
   8 children, world 0:      fitness 110 114 32 148 160 212 120 128  sd 48
   parent on world 0: 214    mean child - parent: -86  (8 of 8 worse)
   ```

   The spread of ONE genome across worlds is larger than the spread between its
   children, and every child is worse than its parent on the same map. A (1+1) lineage
   living one life per generation cannot hear an improvement through that, and the
   mutation operator — thirty thousand weights jittered at once — mostly breaks things.
   The lineage code had already been fixed four times for exactly these symptoms (paired
   comparison, rotating worlds, stepping mutation); the symptoms were the substrate's.

2. *The tissue did not compute; the arcs did.* Everything that foraged went through the
   innate tracts, nose organs -> effectors, which BYPASS the sheet. With the arcs the
   mean brain scored 0.42 against the reflex policy's 0.66 (journal, 2026-07-24): the
   sheet's own contribution was negative. A random per-type perceptron over a 3x3
   neighbourhood with tanh is a random recurrent medium, and nothing selected it into a
   signal path, because (1).

3. *The I/O had no spatial meaning.* A 1D retina was painted onto the sheet by column, a
   command was read as a mean over effector cells, and in between the 2D lattice was used
   as a slow, noisy MLP. The one thing a cellular automaton is actually GOOD at — local
   rules producing global SPATIAL structure: wavefronts, distance fields, gradients — was
   never asked of it. In that arrangement a perceptron beats a CA, which is the opposite
   of the thesis.

(And a control, run in the background on 8 cores while the rest was built: the old brain
under a (1+15) population instead of (1+1). Twelve generations, a child "accepted" in
eleven of them — and the accepted parent's score on the NEXT pair of worlds fell back to
50-150 every time. That is the winner's curse, not adaptation: the best of fifteen on two
maps is the luckiest of fifteen. More selection pressure does not rescue this substrate.)

**The fix: the sheet IS a map (`cortex/atlas.py`).** The lattice is an allocentric map
of the agent's surroundings, agent at the centre, scrolled under it as it walks (the
sense of its own motion is the one extra sense it gets: proprioception). Every sense
paints WHERE IT REFERS TO — a ray paints "solid" at the cell it hit and "free" along the
cells it crossed; a seen morsel paints a bright thing at its cell, with its hue; a smell
paints a source out along the walkable bearing it reports, at the distance its intensity
implies; touch paints solid one cell out. Nothing is labelled: a cell holds solidity,
freeness, hue, brightness. Evidence fades slowly, so the map is a MEMORY.

The rule is one local update, run six times per thought, and it is the whole
computation: a value wave spreads from every source through cells that are not solid,
losing 6% per cell of travel. That is Bellman / brushfire / BFS done by cells that look
only at their eight neighbours — the thing [Pathfinding NCA (Earle et al. 2023)](https://arxiv.org/abs/2301.06820)
shows a neural CA can learn. The command is read at ONE cell, the centre: which
neighbour holds the most value is which way to step; turn toward it, walk, press
INTERACT when a thing is within reach or a smell says it is under the nose. Two things
fall out for free that the old brain never had: EXPLORATION (unknown cells bordering
known-free ones are faint sources, so with nothing to smell it walks toward what it has
not seen), and a kind of PATIENCE (no source, no wave, no movement).

Three lessons from making it play:

* *A wall must leak.* Seed 4 spawned in a room with a door; with walls as absolute
  barriers the wave from the smell outside never reached the centre, the agent stood
  still and starved. A solid cell now passes 12% of the wave, and the read-out presses
  INTERACT when the wave leads it INTO a solid — which opens a door, chips rubble, and
  does nothing to stone. Which hues open is the door experiment, and a per-hue leak is
  where it will live.
* *A far smell must not be faint.* Painted at its intensity, a smell sixteen cells off
  was worth 0.09 and lost to the 0.12 of an unexplored cell beside the body: the agent
  went sightseeing with dinner in its nose. A painted smell is now floored at 0.35; how
  far it is has already been said by WHERE it is painted. Swept over 16 worlds: floor 0
  -> 27.4 meals, 0.35 -> 32.2, 1.0 -> 18.8 (chases every far smell to the map's edge).
* *The nose was reporting compass points.* The smell's direction was the first cardinal
  step of the BFS, which for a morsel two rows down and sixteen columns over says SOUTH,
  because a south-first path is as short as an east-first one and the search grew that
  tree first. A reflex can live with that; a map cannot be painted from it. The bearing
  is now continuous: the straight line when the walk is about as long as the straight
  line, else the bearing to the point on the walk six cells out. Same information, less
  quantised, and it lifted the REFLEX too — 14.2 -> 21.9 meals over 16 worlds — which is
  the right kind of improvement: it is a sensor, and every player gets it.

**Results, 16 worlds, 20000-tick cap, the new nose for everyone.**

```
                                  ticks   meals   gained  immortal
random walk                        6499     2.9     59      -
reflex policy (no brain, no cost) 14515    21.9    376     6/16
atlas r=12, paying 625 cells      14230    30.2    598     7/16
atlas r=12, map free              16466    31.1    611    10/16
atlas r=9 / r=16, paying          15467 / 13304   27.3 / 29.6   8/16 / 7/16

probe ladder     near  open  side  behind  corner  maze  patience | mean
reflex           1.00  1.00  1.00   1.00    0.23   0.23    0.14   | 0.66
atlas            1.00  1.00  1.00   1.00    1.00   1.00    0.05   | 0.86
best cortex brain (07-24)                                          0.68
```

`corner` and `maze` — the two rungs the reflex fails, where the straight line is not the
walk — go to 1.00, because the wave routes around what the map knows. `patience` is
the honest miss: an explorer does not stand still in an empty room, it goes to look at
the corners, and the arena is bigger than its memory so it never runs out of corners.
Whether that is a fault is an economic question, not a scoring one.

**The size of the brain is an economic question, measured.** Radius swept under upkeep
(the same 1.5e-5 per cell per tick a cortex brain cell pays), 16 worlds, before the nose
change: r6 12.6 meals, r9 20.1, r12 28.1, r16 20.2, r24 17.9. An optimum, because a map
bigger than the walk between meals is memory nobody reads, paid for every tick, and r24
pays more for its map than for walking. This is the first "genome x world" number the
playground has produced — the right radius depends on how far apart the food is — and
radius is the obvious first gene for a lineage.

**Stage 2, started: can the rule be FOUND? (`cortex/wave.py`).** The hand rule settles
that a CA with the right local rule out-forages the reflex; it does not settle whether
such a rule can be grown. So: the same map, but the wave replaced by a per-cell
perceptron — fixed perception (each channel's 3x3, its neighbourhood max and mean, the
way an NCA fixes Sobel), one ReLU layer of 12, outputs the cell's value and two hidden
channels of its own; 711 weights. `wave.exact()` builds the weights that ARE the wave
(`V = relu(max(src, k*max8 V) - 2*solid)` is two hidden units), so the target is known to
be in the space: it scores 0.86 on the ladder, identical to the hand rule. Then a
(1+31)-ES over the weights, fitness = the ladder (exact, so no re-measuring), 12 cores.
From random weights: gen 0 = 0.11, gen 1 = 0.55 (near/side/behind at 1.00 — a rule
that lets the read-out follow a local source is one lucky jitter away). Running.

`python -m cortex.atlas --probes | --seeds 0-15 --jobs 16 | --gif ...` (world and map
side by side: walls grey where the rays put them, the wave in blue, sources yellow, the
chosen step green). 9 tests.

## 2026-09-14 (later) — the rule is FOUND, not written; and the atlas learns what not to eat

Two results in the same session, both on the atlas, both of the kind the old brain never
produced.

**1. The wave can be found (`cortex/wave.py`).** The claim to settle: the local rule that
makes the map plan is not something a person has to write. Three searches for it, all
over the same 711 weights of a per-cell perceptron (fixed perception — each channel's
3x3, neighbourhood max and mean; ReLU 12; value + 2 hidden channels), all starting from
random weights:

```
search                                          ladder    corner  maze   open-world (16 worlds)
(1+31)-ES on the LADDER, 34 gens                 0.64      0.20    0.14   —
(1+32)-ES on the TEACHER (MSE vs wave), 300 gens 0.66      0.22    0.10   —
GRADIENT on the teacher, 3000 steps (Adam)       0.88      1.00    1.00   24.6 meals / 15228 t / 6 immortal
hand-written wave                                0.86      1.00    1.00   25.6 / 13986 / 5
built target `exact()` (the wave as weights)     0.86      1.00    1.00   —
reflex policy                                    0.66      0.23    0.22   21.9 / 14515 / 6
```

The teacher is imitation of the wave on random maps (walls with gaps, blobs, a couple of
sources), a fresh batch every step, scored on the last four of fourteen passes — the
game is never in the loop. The gradient-trained rule then plays the game as well as the
rule that was written for it: every foraging rung at 1.00, and in the open world the
same meals per life as the hand rule. **The rule was learned from what a wave IS, and
it transferred to what a wave is FOR.** That is Earle et al.'s Pathfinding NCA result,
reproduced on this playground's own map and then made to eat.

The two failures are as informative. Evolution over the weights plateaus at exactly the
reflex's level — the four smell-following rungs at 1.00, the two terrain rungs at
nothing — whether the objective is the game or the teacher's smooth MSE. On the game
there is a valley: pushing into the wall four cells from the food scored more than
walking toward the gap, so the probes now score the geodesic WALK closed rather than
the crow's flight, and it still plateaued (sigma had also collapsed to 0.007 under the
1/5 rule; it is floored at 0.02 now). On the teacher the ES got the loss from 0.080 to
0.021 in 300 generations while the exact rule sits at 0.000 and gradient reaches 0.003
in a few minutes. A (1+lambda) with isotropic Gaussian steps over seven hundred
coupled weights is the wrong tool; that is a statement about the optimiser, not about
the CA, and it is the same lesson the old brain taught at thirty thousand weights.
Torch is a `train` extra used by this one function; the rule it returns is the same
numpy object the game runs.

An ES fine-tune of the found rule ON THE GAME is running (`--load trained.npy
--evolve`); if the game can improve a rule the teacher found, that is the two-stage
loop — taught, then selected — and if it cannot, the teacher was enough.

**2. Appetite: the atlas learns not to eat the purple one.** Poison in this world is a
nightcap that smells exactly like a berry and differs only in hue, so the only way to
learn it is to eat one and feel the burn — interoception's poison channel rising after
a meal. The atlas now has ONE learned thing: an appetite perceptron over hue, a
population code of sixteen bumps round the colour circle (colour-tuned cells) plus a
fixed innate bias of "anything bright is dinner". A seen thing's pull is its brightness
times its appetite; a thing within reach with appetite under 0.5 is REFUSED — no press,
the cell is marked bad, and every source within two cells of a bad mark (the smell that
led there included) is hushed and the wave there zeroed, so the agent leaves instead of
standing over the one thing it will not eat. The verdict on a meal is +1 after sixty
ticks with no burn, -1 the tick the poison rises; one Hebbian step on the bumps, the
bias untouched, weights clipped so the brain stays teachable.

Three things had to be found out on the way, each by a measurement that came out wrong:

* A perceptron over plain (cos, sin) of hue CANNOT do the job — the nightcap (288 deg)
  sits between the berry (349) and the mushroom (222) on the circle, so no half-plane
  keeps both foods and drops the poison; it learned to hate mushrooms instead
  (appetite 0.00, they are worth 32). Hence the population code.
* With the bias learnable it saturated at +4 after a dozen good meals and outvoted any
  burn. The bias is the innate prior; it is not taught.
* Eight bumps of ~30 degrees overlapped enough that one berry after a burn forgave the
  nightcap (0.14 -> 0.56). Sixteen bumps at ~20 degrees share almost nothing.
* And the press on smell alone would eat a refused thing that had drifted out of view:
  a remembered bad mark within reach now blocks the press, sight or no sight.

The exam is `python -m cortex.atlas --poison`: the poison arena twice with the same
brain, a nightcap dead ahead and a berry off to the side, the map wiped between visits
and only the appetite carried over:

```
                first visit           second visit
learning        berry 1  nightcap 1   berry 1  nightcap 0     (appetite: nightcap 0.14, berry 1.00)
no learning     berry 1  nightcap 1   berry 1  nightcap 1
```

One burn is enough. In the open world over 16 lives: nightcaps eaten 12 -> 5, meals
26.8 -> 25.6, lifetime 13196 -> 13986 ticks. It eats slightly less and lives slightly
longer, which is what refusing dinner should do; and it is the first within-life
learning in this project whose effect is visible in the count of a specific behaviour
rather than in a probe mean.

**Also this session.** The atlas plays in the live HUD (`python -m cortex.app --atlas`,
map column on the right: walls grey where the rays put them, the wave in blue, smell
yellow, a seen thing in its own hue, a bad mark magenta, the chosen step green).
`cortex/atlas_lineage.py` got a CONFIRMATION step: children are screened on four
rotating worlds and the finalist is re-lived against the parent on all eight, because
the first version accepted a child in fourteen of fifteen generations while the
parent's score went nowhere — the best of fifteen on four maps is the luckiest of
fifteen, the same winner's curse the old brain showed under a population. The sparse /
rich arms are re-running with it. 21 new tests this session, suite 460.

## 2026-09-14 (last) — the ladder Goodharted, the LLM beat rng on the holdout, and the map grew to fit its world

Three closing measurements, each the answer to a question the day had left open.

**1. The exam, used as a fitness, was gamed within forty generations.** The found rule
(gradient on the teacher) was handed to the (1+31)-ES on the ladder to see whether the
GAME could improve what the TEACHER found. On the ladder it could: 0.88 -> 0.927, with
`patience` 0.18 -> 0.49 and every foraging rung still at 1.00. In the open world the
tuned rule eats 2.1 meals a life and dies at 5765 ticks (the rule it started from: 24.6
meals, 15228 ticks).

```
rule                          ladder   patience    open world (16): meals  ticks  immortal
found by gradient (teacher)   0.88     0.18        24.6   15228   6/16
...then tuned by ES on ladder 0.927    0.49         2.1    5765   0/16
```

The ES found the one thing the ladder rewards that the world punishes — sitting still —
and bought patience marks with the foraging the small arenas do not test at length.
`docs/cortex-design.md` says the ladder is an OBSERVER, not a curriculum, and the
lineage code refused to let the exam select for exactly this reason; here is the
measurement that says why. It also retroactively explains the ES-on-the-ladder runs
from random weights: a plateau at the reflex's level on an objective that was never
the point. The rule that lives in the world was taught by a teacher and selected by
nothing; anything that selects on the atlas from now on selects on LIVES.

**2. The LLM writes the map's growth program (`cortex/atlas_lab.py`).** Same loop as
`cortex/lab.py`, on a brain that works: DeepSeek writes the twelve atlas knobs, every
candidate lives the same eight worlds for 8000 ticks (exact score, no re-measuring),
the reflex's meals on those worlds are the bar in the prompt, `--proposer rng` is the
control. Ten iterations each:

```
                           train (worlds 0-7)          HOLDOUT (worlds 8-15)
program                    fitness  meals               fitness  meals   ticks   radius
default                     326     15.1                 298     14.0    7274    12
LLM best (iteration 4)      417     21.0                 391     19.2    7391    17
rng best (iteration 1)      472     21.9                 321     15.6    7765    12
```

On the training worlds rng "won" — with one lucky jitter at its first try. On worlds
neither had seen, the LLM's program is +37% meals over the default and rng's is +11%.
The model's edits were SYSTEMATIC and monotone across iterations — a bigger map
(12 -> 17), slower value decay, more exploration, a lower smell floor, faster thinking —
with one overshoot (radius 18, iteration 5) that it walked back; rng's win was a draw.
This is the forage lab's result again (the sight champion generalised, the random walk
overfit), which is what a lab that measures on a holdout is for.

**3. Genome x world, first numbers.** `cortex/atlas_lineage.py` with the confirmation
step (children screened on four rotating worlds, the finalist re-lived against the
parent on all eight), one program in two economies:

```
world    gens  accepted  parent fitness       radius        other moves
sparse    23      3       121 -> 210 (+74%)   12 -> 9       thing_gain 1 -> 1.16, leak up
rich      17      7       579 -> 935 (+61%)   12 -> 12      decay 0.06 -> 0.077, thing_gain -> 1.5,
                                                            leak 0.12 -> 0.056, smell_floor -> 0.43
```

Where food is scarce the map SHRANK — memory nobody can afford — and where it is
everywhere it stayed, with the wave tuned shorter (the nearest thing is always near)
and walls trusted less. The direction matches the radius sweep of the morning (in a
sparse world r20 starves on its own upkeep; in a rich one it eats most) and the LLM's
default-world programme (17, generalised). It is a modest divergence, 9 against 12, on
one seed each; it is also the first time in this project that a heritable size settled
at a different value in a different world for a reason the economy explains.

And the acceptance counts are the real news about the lineage: three of twenty-three
and seven of seventeen, against fourteen of fifteen before confirmation. A lineage that
mostly says no and is right when it says yes is what selection is supposed to look
like.

**Where the atlas stands, end of day.** Ladder 0.86-0.88 (reflex 0.66). Open world, 16
seeds: 25-31 meals against the reflex's 21.9, at the same lifetime, paying for its map.
The rule can be written or learned. The program can be evolved or written by a model.
It learns not to eat the purple one. It is watchable. The old sheet (`cortex/brain.py`)
stays as reference. Suite 465.

## 2026-09-17 — the rule as a law of nature, and the law finds what the wave lacked

**The third form of the rule.** `cortex/atlas.py` hard-codes the wave in numpy;
`cortex/wave.py` found it as a perceptron; `cortex/atlas_law.py` runs it as a `.nomos`
LAW — the project's own DSL, the language the scientist writes physics in and the brain
lab wrote snake brains in. The map's lattice is a nomos `World`: each thought the harness
paints the senses into read-only channels (`source solid free thing bright scent bad
here`), steps the world `subticks` times, reads `value` back; every other channel the law
declares is its own hidden state, kept between thoughts and scrolled with the map. Two
rings of padding — the horizon (free, sourceless) and an outer ring forced to zero after
every step — keep the engine's torus from wrapping a wave across the map.
`laws/atlas_wave.nomos` is the wave transcribed in one line:

```
set value = max(source, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - 0.88 * solid)
```

(the max of a von-Neumann and a Moore max with different discounts IS the eight-way wave
with a diagonal that costs sqrt 2). It reproduces `Atlas._wave` to 1e-6 (test), the
ladder 0.87 vs 0.86, and 8 worlds x 8000 ticks 15.1 meals / 7311 ticks against the
default program's 15.1 / 7274. ~1.5x slower than numpy. A rule the model can WRITE, not
knobs of one, validated by the engine's own parser.

**The lab (`cortex/law_lab.py`)**: same loop as `atlas_lab`, but the proposal is the whole
law; a law that fails to load goes back with the parser's error; `--proposer rng` jitters
the best law's constants (structure frozen); `--proposer files` runs a physicist's own
laws through the same loop and holdout. Three generations of DeepSeek before the user
asked to stop spending the API (a local model is coming): it forgot to declare an input
channel two proposals out of two (the harness now declares inputs a law reads), built a
fear field from `bad` (no change: nightcaps are 0.1 a life), then multiplied the wave by
1.2 on unknown cells — 0.94 x 1.2 > 1, the wave EXPLODES to the clip and the agent runs
into the unknown for ever: 4.2 meals. The rng control found the same cliff by accident
(every jitter that put a factor above 1 scored 4.5) and its "best" (walls leaking 45%,
18.9 meals on train) was exactly the reference on the holdout, 14.0 = 14.0 — the third
time in this project that rng's win was luck.

**Where the meals come from.** Ablation with the default program, 8 worlds x 8000:

```
                          meals   immortal   per world
smell + frontier (default) 15.1     5/8      [5, 33, 31, 14, 13, 7, 17, 1]
frontier only              14.4     7/8      [17, 6, 10, 19, 18, 26, 7, 12]
smell only                  5.5     3/8      [22, 0, 10, 0, 6, 2, 4, 0]
neither                     0.0     0/8
```

Exploration feeds this agent; the smell adds NOTHING on average — it doubles the take on
worlds 1-2 and costs worlds 0, 5, 7 two thirds of theirs. World 7 on a contact sheet:
eleven frames out of twelve the agent stands on the same spot above a wall with the food
in the room beneath, walking (920 cells in 4776 ticks) and going nowhere, while the
painted smell hops from the right of it to the lower left and back. The world's smell
bearing is the geodesic waypoint; from a spot between two ways round a wall it flips as
the agent takes a step each way. The forage brain's reactive limit cycle, now at the
level of the sensor.

**Four laws against it**, lived through the lab (8 train worlds, holdout 8-15):

```
law                                    train meals  immortal   holdout meals
reference (Bellman)                       15.1        5/8         14.0
atlas_ghost — sources LINGER              22.9        8/8         17.5
atlas_habit — a place goes boring         17.5        6/8
atlas_ghost_habit — both                  22.2        6/8
atlas_halfscent — control, smell x 0.5    16.5        8/8
```

`atlas_ghost`: a `ghost` channel keeps 99% of a painted source per pass after the paint
moves on (~94% per thought, half-life ~35 ticks), erased in the 3x3 under the agent
(`here`: it is standing there, there is nothing); the wave runs from the ghost. Per world
on train: [29, 35, 22, 15, 15, 21, 17, 29] — the three trap worlds went 5 -> 29, 7 -> 21,
1 -> 29 and none is under 15. The hypothesis was confirmed by its own mechanism: the
worlds it fixed are the worlds the smell was trapping. On world 7 the agent now crosses
twelve different scenes in 4800 ticks and eats 19. Sweep and ablation on all 16 worlds:

```
                              meals   immortal
reference                      14.6    11/16
ghost 0.98                     18.2    14/16
ghost 0.99                     20.2    14/16      (+38%)
ghost 0.995                    16.4    15/16
ghost 0.99, NO erasure         13.5     9/16      worse than none
```

The erasure is load-bearing: without it the ghost of an eaten morsel keeps the agent
circling the spot, as the law's own comment predicted. The ladder does not see any of
this (0.87 either way): it is an observer of skills the open world does not test at
length, again.

**Back into the engine.** `Config.linger` (default 0, the plain wave) is the same rule in
numpy, pass for pass (test: 1e-6 against the law — snapshot semantics bit once: the hand
version read the ghost it had just written), and `linger` is the thirteenth gene of the
atlas genome, so a lineage can find commitment on its own. The engine gave the DSL a rule;
the DSL gave it back a better one.

The headline bench, 16 worlds x 20000 ticks, the hand wave either way:

```
                          meals   ticks   immortal   per world
plain wave                 25.6   13986     5/16     [5, 81, 50, 22, 14, 7, 36, 1, 21, 1, 4, 13, 31, 57, 21, 46]
linger 0.99                37.8   17274    11/16     [77, 57, 48, 38, 19, 24, 35, 48, 21, 5, 5, 43, 25, 63, 52, 45]
reflex (2026-09-14)        21.9   14515     6/16
```

+48% meals, a quarter longer lives, two worlds under ten meals instead of five. Worlds
9 and 10 stay at 5 for both — a trap of another kind, the next one to look at.

**Method notes.** Four sanity anchors carried the day: the reference law's numbers
matched the journal's to the decimal before anything new was tried; every hypothesis
had a control (`halfscent` for "is it just less smell?", `no erasure` for "is the erase
doing anything?"); train and holdout were kept apart; and the ladder was consulted only
as an observer (it saw nothing). `multiprocessing` on Windows cannot spawn workers from
a `python -` stdin script — a bench run that way computed nothing for half an hour; run
pools from a file. Suite 478.

**World 9, looked at.** Not a smell trap: a labyrinth of walled rooms with the food
inside them. The agent walks the corridors (1443 cells in 6280 ticks, twelve different
scenes on the sheet) and never opens a door (`door: 0`, six idle presses). The wave leaks
12% through any solid, which is enough to route round a wall and not enough to make a
door worth pressing when a frontier is free next to it. This is the door experiment the
`leak` comment in `Config` has been promising — a per-hue leak the body teaches, the way
appetite is taught by poison — and it is the next trap, not a law of the wave.

## 2026-09-17 (later) — the door experiment: four things that had to be true before a door could be learned

The `leak` comment in `Config` had promised for months that "which hues actually open is
the door experiment, and a per-hue leak is where it lives". Built today: `Config.leak_lr`
turns the leak into a perceptron over the hue population code (the appetite's twin —
bias = logit(leak), the innate guess that any wall might open), taught by the body: a
press that opens a door, chips rubble or shoves a block is +1 for the hue pressed on,
a press that does nothing ("nothing to do here") is -1. `laws/*.nomos` receive it as
the `leak` input channel, so the reference law is still the hand wave under any leak.

Then the measurement said the learner had learned NOTHING in 24000 ticks over three
worlds: zero presses on solids ever. Four separate reasons, found in order, each one
hiding the next:

1. **The read-out charged the leak twice.** The wave already multiplies a solid cell's
   value by its leak; the read-out multiplied the solid neighbour by it again, so a
   door held 1.4% of a value and lost to any free cell. "Led into a solid" was a
   branch that never ran. Dropping the second charge: world 9 went 5 -> 13 meals
   before anything else was touched.
2. **The smell died on the last step.** The scent BFS skipped solids on its first
   ring and charged them through `_passing` on every later ring, so a larder's smell
   was 0.129 a cell from the door and exactly 0 beside it — the step where pressing
   is the whole question. One cost rule for every ring.
3. **The nose could not point at a door.** The bearing is to the walk's waypoint,
   "the last cell within 6 steps"; a door costs 7, so the waypoint was always the cell
   BEFORE the door, and standing in that cell the bearing degenerated to a sub-cell
   offset and flipped up and down along the wall (the door probe: `nearest` 13 for
   ever, the agent pacing the wall past the door both ways). Now "the first cell out
   at least 6 away" — beside a door, the door.
4. **The horizon beats every wall at leak 0.12.** The padded ring round the map is a
   free way round; on a radius-12 map that walk is ~26 cells, 0.94^26 = 0.2, more
   than 0.12 through the wall — so the wave routed round the outside of the map rather
   than through ANY door, always. Innate leak must be CURIOUS: 0.3 solves the door
   probe (0.08 -> 1.00, opened and ate), 0.5 with the horizon; rubble (smell cost
   higher) needs 0.7 or learning.

And two more once presses happened at all: the verb acts on the cell the nose faces,
and a press thrown while still turning landed on the stone beside the door and blamed
the door's hue — learning drove the door's leak DOWN (0.5 -> 0.36); now a solid is
pressed only when faced (`press_angle`). And a cell that has just yielded stayed solid
on the map until a ray happened to look, so the next press hit bare ground and the
-1 went to the hue of the thing that had just given (rubble and door share a bump of
the population code): the body now tells the map the cell gave (`Atlas.gave`).

**Numbers**, 16 worlds x 8000 ticks, all with linger 0.99 (the meal means differ by
less than their noise — per-world sd ~10, so SE ~2.5 — the DEEDS are the result):

```
innate leak  learning   meals  immortal  doors/life  breaks/life  idle presses/life
0.12         off         19.2   14/16       0.0         1.1           54
0.12         on          18.3   12/16       0.1         5.7           41
0.5          off         22.5   16/16       1.6         1.4          127
0.5          on          21.8   15/16       1.8        22.9           72     door hue 0.50 -> 0.57, stone -> 0.23
0.7          off         18.4   14/16       0.8         3.4          233
0.7          on          17.0   13/16       1.1        28.2           58     door 0.73, stone 0.29
```

At 0.12 no door is ever opened, whatever the learner. A curious prior opens doors
(0 -> ~1.5 a life) at the price of pressing on stone (127 idle presses a life); the
learner halves that price (127 -> 72, 233 -> 58) and moves the hues the right way, and
costs no meals. Doors are simply not where this world's meals are: ~1.5 doors a life
against ~20 meals. The mechanism works — probe 1.00, hues taught, presses cut — and the
economy does not reward it yet. New defaults: `linger 0.99, leak 0.5, leak_lr 1.0`
(and the genome's defaults with them); ladder 0.87, `door` 1.00, `rubble` 0.01
(needs the learner, which the probes bypass).

**Caveat on the noise.** Sixteen worlds tell 14.6 from 20.2 apart (the ghost) and do
not tell 19 from 22 apart. Anything under ~5 meals at n=16 is a shrug; deeds and
probes are the honest read-out for a mechanism like this one.

**Genome x world, three seeds each** (the background lineages from this morning, on the
code as it was at 19:16 — one pool per run, so the day's edits never reached them):

```
world    seed  gens  kept   fitness            radius
sparse    0     25    3     121 -> 210 (+73%)   12 -> 9
sparse    1     25    3     121 -> 244 (+101%)  12 -> 11
sparse    2     25    6     121 -> 236 (+95%)   12 -> 11
rich      0     25    7     579 -> 935 (+62%)   12 -> 12
rich      1     20+   5     579 -> 947 (+64%)   12 -> 12   (still running)
rich      2     25    4     579 -> 899 (+55%)   12 -> 13
```

Sparse never keeps 12, rich never goes below it: 9/11/11 against 12/12/13. A small,
consistent split — the map shrinks where food is scarce and holds where it is not —
and three seeds is what turns "9 vs 12" from an anecdote into a direction. The
acceptance rate stayed where confirmation put it (3-7 of 25).

**Headline, 16 worlds x 20000 ticks, new defaults** (linger 0.99, leak 0.5, learner on):
42.6 meals / 16708 ticks / 9 immortal, against 32.6 / 15576 / 8 for linger with the old
0.12 leak on the same code (morning: 37.8; the plain wave: 25.6). World 9 — the room
world — 5 -> 30. World 1 fell 47 -> 5, and it did so on ONE learned verdict (stone
0.5 -> 0.21, then a different route, then a different life): the world is
deterministic and chaotic, and one world's swing of 40 is 2.5 on a mean of sixteen.
The paired difference is +10 with an SE of 6. Suite 484.

## 2026-09-23 — a local model, and a lab built for its speed

The user brought up a local model on their own hardware (llama.cpp, Qwen3.6-35B-A3B
Q8 on a 20-vCPU VM, `http://llm.pgs.dom:8080/v1`; infra notes in
`phil-architecture/docs/llm.md`). No API bill, so the labs can finally run for hours —
but it is a CPU box with one slot, shared with the user's chat UI. Measured first, on
the law lab's own first turn (~1.5k prompt tokens):

```
thinking off   40-95 s a turn (prompt ~53 tok/s cold, ~all of it prefix-cached after;
               generation ~9.7 tok/s). 1 law of 4 loaded before the prompt named the
               syntax traps (`let`, `if/then/else`, `pass`), 2 of 3 after.
thinking on    ~10-13 min a turn (5.5k hidden tokens). Loads; the idea was timid.
a law lived    16 worlds x 8000 ticks in ~16 s on this machine.
vision         a 600-px PNG + question in ~7-40 s.  embeddings (bge-m3) 0.8 s.
```

So the MODEL is the scarce resource and the simulator is nearly free: one idea is
worth dozens of evaluations. That, and two old lessons (the snake's best brain was an
LLM architecture with evolved constants; rng's law-lab win was luck on holdout), shaped
`cortex/night_lab.py`: the model designs STRUCTURE, rng tunes CONSTANTS of each
structure between the model's turns, a CONTROL arm tunes the hand-written ghost with the
same number of evaluations, and the model keeps a short THEORY (carried in every prompt
instead of a chat history it could not afford) and now and then OBSERVES — it is shown
the champion's worst life as a picture, track drawn from cyan to red, and asked what the
agent does wrong.

**The probe (25 min, 4 design turns) taught five things before the night:**

1. The model writes explanations INTO the law (a 250-s turn of comments). "One comment
   line per rule, under 70 lines" halved the turn.
2. Its first law summed neighbour terms and blew the wave up — the known cliff — and a
   later one was FLAT. It cannot see its field, so now it gets a mirror: `diagnose()`
   lives the law on a 25x25 open map with one source and reports max/mean/at-the-cap and
   whether the best neighbour points at the source; an unhealthy law is sent back like
   a parse error, inside the same turn.
3. It chains `set` lines on one channel as if they ran in order (they read the snapshot;
   the last one wins), so its new channel changed nothing: a law with the ghost's meals
   on every world, to the digit. Caught now as a BEHAVIOURAL twin, and the trap is in
   the prompt. `max(a, b, c)` failed three repairs running — `min`/`max` are variadic in
   the DSL now.
4. **Tuning overfits.** The model's third structure scored holdout 23.1 as written
   (ghost 16.4); thirteen accepted jitters later it was train 21.7, holdout 16.9. The
   tuned ghost (the control) went 19.6 -> 23.4 on train and 16.4 -> 16.2 on holdout.
   Sixteen chaotic worlds are a small target for 1+1 jitter. Hence a 2% acceptance
   margin, 20 training worlds, and a third set — 32 FINAL worlds that neither selection
   nor the model (who sees holdout numbers in its prompt) ever touches — lived once at
   the end by the finalists, including the holdout champions AS WRITTEN.
5. The observer is good. Shown a life that starved in the bottom-left corner it said:
   "it enters a small area ... begins pacing in tight circles ... starves ... needs a
   mechanism to detect when it is trapped and prioritise unvisited cells" — which is
   exactly what the picture shows. With the design system prompt it answered with a law
   instead; it has its own short system prompt now and is given the law it watches.

**The night** (`out/cortex/night/2026-09-23/`): 6 hours, train 0-19, holdout 20-35,
final 36-67, 8000 ticks, tune 5 evaluations per new structure then round-robin over the
best three, a deep (thinking) turn every 6th design turn, theory every 6, observer every
8 (or on a new champion).

## 2026-09-24 — the night's results, and what it says about the loop

### The port experiment (branch `atlas-confinement`): the dose is the finding

#35's mechanism went into `cortex/atlas.py` as `Config.conf_pen` (a source hemmed in
by solid is divided by `1 + conf_pen * sum8(solid)`; also a lineage gene). Probe
ladder unchanged at every dose tried (0.87-0.88). But the headline, 16 worlds x 20000
ticks, is NON-MONOTONE in the dose:

```
conf_pen   meals   immortal
0.0        42.6    9/16      (the current default: linger 0.99, leak 0.5)
0.15       43.7    9/16
0.3        52.9    12/16
0.5        37.4    6/16
```

At the law's own 0.5 the mechanism LOSES — suppression so strong the agent starves
waiting (immortal halves). At 0.3 it looked like the best headline the atlas had
posted: +10 meals, 12 immortal. **So we confirmed on sixteen worlds nobody had used
(seeds 16-31): base 32.6 meals / 9 immortal, conf 0.3: 31.2 / 6 — the win did not
replicate.** Combined over all 32 worlds the paired difference is +4.4 meals with an
SE near 5: a shrug. The 0-15 sweep is a textbook winner's curse on chaotic worlds —
four doses tried, one world swinging 40 meals on a single learned verdict, and the
best-looking dose is the luckiest one. The night's own lesson about 1+1 jitter applies
to a manual sweep of four. `conf_pen` stays in the code and the genome at default 0.0 —
a mechanism the lineage or the next night can reach for, not a default.

And the model's final theory, wrong about its own champion at the law's dose ("the
confinement penalty is too aggressive"), at least named the right tension: suppression
strong enough to matter is strong enough to starve.

### What the night itself said

42 design turns, 38 islands, 6 theory rewrites, 5 observations. **FINAL worlds** (32,
seen by nobody until the end; the champion was only tuned 5x — the structure, not the
constants, carried it):

```
reference wave (#0)                 18.8 meals  immortal 28/32
ghost, hand-written (#1)            14.6 meals  immortal 20/32
CONTROL: ghost tuned 138x           17.1 meals  immortal 26/32
#35 tuned 5x                        24.5 meals  immortal 28/32
#10 tuned 12x                       21.6 meals  immortal 29/32
#3 tuned 24x                        20.6 meals  immortal 26/32
#10 as written                      19.6 meals  immortal 29/32
```

**The model beat the control with an idea.** Champion #35 divides the injection by
confinement — `injection = max(thing, source) / (1 + 0.5 * sum8(solid))` — local food
in tight spots stops dominating the field, the frontier wins there. It did not drop
between sets at all (train 25.7 = holdout 25.7 -> final 24.5), while everything
heavily tuned did (#3: holdout 23.5 -> final 20.6; control: 21.4 -> 17.1). Second
confirmation of the division of labour: **structure generalises, constants don't.**
The control lost to the hand-written wave on finals too (17.1 vs 18.8) — 138 accepted
jitters bought nothing the world didn't already give for free.

**And three findings about the night lab itself:**

1. **World 16 ate the night.** It scored 0 for EVERY law, champion included, and the
   observer showed the model the same frozen life there four times (identical metrics:
   59 cells, box 1x1) — so the model wrote ~15 variations of one idea (commit /
   confinement / visited, embedding similarity 0.76-0.92) aimed at a single world.
   The win came from a different direction. Fixes for night 2: the observer rotates
   over the champion's several worst lives instead of one, and an embedding deduper
   pushes the model onto a new axis instead of the 10th commit channel.
2. **Thinking did not pay.** Deep turns (~17 min each) produced 0.6 and 3.0 meals; the
   champion came from a plain 75-second turn. Worse, the final THEORY contradicted the
   result ("the confinement penalty is too aggressive" — the champion IS a confinement
   penalty): the theory tracked the conversation, not the evidence.
3. Two turns lost to parse errors, one behavioural twin slipped through (a chained-set
   law scoring the ghost's exact meals on every world).

The night validated the design split (model = structure, rng = constants, control arm
to keep both honest) and left a concrete continuation: port the confinement mechanism
back into `cortex/atlas.py` like `linger` before it, and seed night 2 from the
champion rather than the ghost.

## 2026-11-04 — the champion replicates as a law; a lab whose model is the session's agent

**Replication on 128 fresh worlds** (seeds 100-227, 8000 ticks, `cortex.atlas_law --lives`,
same harness for all three laws; no seed here was ever seen by selection or by a model):

```
reference wave   17.2 meals  immortal  98/128
ghost            20.2 meals  immortal 104/128
confine (#35)    22.4 meals  immortal 108/128
```

Night 1's champion holds as a LAW: +2.2 over the ghost and +5.2 over the wave on worlds
nobody selected on — the same ordering as the 32 FINAL worlds (24.5 / 17.1 tuned control /
18.8). What did not replicate on 2026-09-24 was the PORT (`Config.conf_pen` in
`cortex/atlas.py`, where it sits next to linger, the frontier and the bad-thing veto);
the mechanism inside the bare wave law is real. It is now `laws/atlas_confine.nomos` and
the bar for the next run.

**The model is the agent.** `cortex/mailbox.py` + `night_lab --agent`: the lab writes each
question (design, repair, observe with its PNG, theory) to a file and waits; the agent of a
Claude Code session answers into a file. Everything else — tuning, the control arm,
holdout, sealed FINAL worlds — is unchanged. Recipe and rules: `docs/cloud-lab.md`. Three
night-1 lessons are built in: the observer ROTATES over the champion's three worst lives
(night 1 showed one frozen life on world 16 four times); novelty without an embedding
model = Jaccard over the law's vocabulary (channel + builtin names), >= 0.8 to an earlier
island is called a variation; `--control-law` makes confine anchor #1, so the control arm
tunes the best known structure and a win means beating IT. New with a fast model: PACING —
the next design waits until the last idea had its tuning budget, otherwise a model that
answers in a minute out-runs a tuner that needs ~2 min per evaluation on 4 vCPU.

A local smoke run (4 train worlds, 2000 ticks, Claude answering by hand) went through
every path: a design that became an island and was tuned, an observe turn on its PNG, a
theory, a design answer without a law sent back as REPAIR, an unanswered request
abandoned after the grace period, FINAL table and summary written.

## 2026-11-04 — cloud study FULL-1

Arm FULL (the session's agent answered every request itself, one mind all night, observer
on), replicate 1 (`--train 0-19 --holdout 20-35 --seed 0`), `runs/cloud-lab/full-1/`,
branch `cloud-lab-full-1`. 12.4 h, 43 design turns (all parsed first try, 1 constants
twin), 7 theories, 6 observations (worlds 16 x4, 2, 18).

**FINAL** (32 sealed worlds, seeds 36-67; paired difference vs the control over the 32):

```
reference wave (#0)            18.8 meals  immortal 28/32   -7.0 (SE 2.6)
confine, hand-written (#1)     24.5        28/32            -1.3 (SE 2.0)
CONTROL: #1 tuned 212x         25.8        28/32
#19 tuned 5x  (CHAMPION)       26.9        30/32            +1.1 (SE 1.5)
#35 tuned 5x                   29.8        31/32            +4.0 (SE 1.7)
#30 tuned 6x                   28.2        29/32            +2.4 (SE 2.1)
#44 tuned 1x                   27.2        28/32            +1.4 (SE 2.2)
#32 tuned 5x                   26.9        28/32            +1.1 (SE 2.5)
#19 as written                 27.1        30/32            +1.2 (SE 1.8)
```

**Champion vs control:** the champion (picked by holdout, #19) beats tuned confinement by
+1.1 meals with SE 1.5 — not a win by itself. All five model laws that reached FINAL beat
the control (+1.1 to +4.0) and the hand-written confine (+2.4 to +5.3); the best, #35, is
+4.0 (SE 1.7), but it is the holdout's SECOND pick (30.8 vs #19's 31.2), so that number is
chosen after the fact.

**train -> holdout -> final:** #19 27.1 -> 31.2 -> 26.9; #35 28.8 -> 30.8 -> 29.8;
#32 30.5 -> 30.5 -> 26.9; #30 31.4 -> 27.2 -> 28.2; control 30.7 -> 24.2 -> 25.8. The
control arm overfit its 212 jitters the most (-4.9 train -> final); the model laws were
tuned only 5x and moved within about +-3, in both directions — the 16-world holdout cannot
rank laws this close (its gaps are ~1 SE).

**Prediction error** (|predicted - untuned train meals|): first half 4.8, second half 3.3.

**Mechanism.** One idea carried the night: price every STEP of the wave by what the map
knows. Unknown ground costs extra (#18, 25.6 -> 27.7 untuned), unknown cells continuing
a seen wall line are guessed wall and pass half (#23, 29.6), cells continuing a line of
open floor are guessed floor and pay no unknown price (#32, 30.5 untuned = holdout 30.5),
walls pass by DOOR CONTRAST (taught per-hue leak minus the mean leak of the walls around
it). On top: confinement, darkness-weighted frontier (frontier mark x smoothed unknown
around it), a fading footprint trail that dims the read-out. #35 = #32 + the trail also
charged as a step price. Ablations: without darkness + trail 24.6, without door contrast
26.7, without the read-out dimming 25.1 — all load-bearing. Settled negatives (untuned):
any new attracting source (leaves from `bright` 18.3, lingering food 21.5, compass 21.4,
breach 23.2, larder 24.6, a room source that changed no decision at all), the same
knowledge used as source WORTH instead of step price (22.1), changing distance pricing
either way (additive 16.9, distance-steepened 21.9), the plan's speed either way (wipe each
thought 12.8, express lanes 26.7, slower 28.4), harmonic relaxation 14.2, a momentum
read-out 13.7. Train world 16 (a sealed pocket) scored 0 for every law; the observer described it
four times and no law fixed it. Two FINAL worlds (seeds 61, 62) scored 0 for nearly every
law (#35 got 18 on seed 62, the reference wave 30 on seed 61).

**The theory** ended as "knowledge goes into step prices, not into source worth; every
piece of #32 is load-bearing; tuning overfits" — consistent with its evidence and with
FINAL (every knowledge-priced law beat the control). It called trail-as-price "neutral"
(28.8 train, 30.8 holdout); FINAL makes that law the best, which the theory could not
have known.

**What this arm's agent did differently.** It kept one line of inquiry and climbed it:
four successive knowledge priors, each built on the last, plus three explicit ablations to
find what was load-bearing (night 1's entry records no ablation). It read harness code
(`cortex/atlas.py`, `world.py`, `tiles.py`) for facts the prompt does not give (food
regrows after 1200 ticks, the map forgets at 0.0015/tick, foliage brightness 0.58 vs
ground 0.21) — these facts motivated the larder, seen-memory and leaves laws, and all
three lost. Disclosure on rule 2: besides `mailbox check` it stepped `LawRule` by hand on
a few hand-built test maps (one source, a sealed box, a wall line; no world lived, no
score) to see a field; that is how it found a one-pass lag in its own early laws (a
helper channel read before it was written injects a stale full-strength source for one
pass). Compared with night 1 (deepseek, 6 h, ghost as control): night 1's champion IS this
night's hand-written anchor (24.5 on the same FINAL worlds); this arm's champion adds
+2.4 and its best law +5.3 on top, against a control that was itself the stronger
structure.

## 2026-11-05 — cloud study FULL-2

Arm FULL (the session's agent answers every request, observer on), replicate 2
(`--train 100-119 --holdout 120-135 --seed 1`), 12 h + grace, 30 design turns (6 deep),
4 observations, 5 theories; all 30 laws parsed first try, 0 constants-only twins. Run
directory `runs/cloud-lab/full-2/` (branch `cloud-lab-full-2`).

**FINAL worlds** (36-67, the same 32 as night 1):

```
                            train   holdout   FINAL   immortal (FINAL)
reference wave (#0)          20.4    15.5     18.8    28/32
confine, hand-written (#1)   23.2    23.6     24.5    28/32
CONTROL: #1 tuned 153x       29.8    25.1     23.4    30/32
#28 champion, tuned 5x       34.6    35.7     33.1    31/32
#21 tuned 5x                 35.1    34.6     32.8    31/32
#18 tuned 6x                 33.7    34.8     34.3    31/32
#26 tuned 6x                 34.7    31.4     30.5    29/32
#7  tuned 7x                 34.1    28.4     27.8    30/32
```

**Champion vs control on FINAL: 33.1 vs 23.4 (+9.7)**, and every course-law finalist
beat the control by 7-11. Drops train -> holdout -> FINAL: the course laws hold
(#28 34.6 -> 35.7 -> 33.1, #18 33.7 -> 34.8 -> 34.3, #21 35.1 -> 34.6 -> 32.8); the
pre-course train leader #7 did not (34.1 -> 28.4 -> 27.8), nor did the control
(29.8 -> 25.1 -> 23.4, below the untuned confine law's 24.5 — 153 rounds of constant
tuning bought a train number and lost a meal on the exam). Night 1's lesson repeats:
structure generalises, tuned constants do not. Against night 1 (deepseek, 6 h, same
FINAL worlds): its champion 24.5, this one 33.1.

**Prediction error** (|predicted - actual| before tuning): first half 2.8, second
half 3.4 meals — no gain in prediction; I systematically over-predicted additions to a
good law.

**The champion's mechanism.** A LONG COURSE: a heading vector averaged over ~250 ticks,
stored at the agent's own cell and gathered back from the ring after each scroll
(`here * (c + sum8(c))`), normalised to a unit direction, broadcast over the map with
max8, and used to weight frontier GOALS (`1 + ahead`, 0..2). It broke a plateau: every
structure before it (memory, confinement, roominess, trail, frustration, momentum,
geometry, area-restricted search) landed at 25-28 untuned, within noise of each other;
the course alone took it to 33 (#15 on the step, #16 on goals), the unit version to 35.1
(#21). The reason, from the observer (world 100: orbiting the home block for half a
life): the 25-cell map forgets what scrolls off, so a frontier-greedy walker re-sweeps
known districts; a persistent heading is the only large-scale memory this brain can
keep. The stack under it matters (stripping #21 to course + roominess + memory, #25:
28.8, -5 on ordinary worlds, 0 in the pocket world 110): confinement (night 1),
frustration (dwell raises wall leak; with a symmetry breaker it escapes sealed pockets,
110: 0 -> 29), leak only in thin walls (room town 117: 6 -> 33), doors-as-frontier.
#28 = #21 + frustration fading the course's say + a 0.2 floor for frontier behind;
per world nearly identical to #21.

**Negative results worth keeping:** the course pays at the level of goals, not paths
or steps — anisotropic discount #22 32.2, re-aiming by an information gradient #27 32.8,
a chiral (spiralling) course #29 30.7, a hunger clock gating it #30 31.6, and BOTH
levels at once #31 19.1 (they interfere badly). Commitment hurts: target lock via
argmax8 ids #24 29.7, step momentum #9. A harmonic (mean-field) exploration potential
#14 18.9. Larder clock, tree/canopy prior, area-restricted search: neutral or worse.

**The theory** ended (9.2 h) as "one structural gain dominates: the course, at the level
of goals; the stack is not insurance; commitment hurts; non-max exploration hurts" —
consistent with the champion and the FINAL table (#28 is #21 plus a refinement the
theory did not need). Unlike night 1, the last theory agrees with its champion.

**What this arm's agent did differently.** The eyes contributed two of the champion's
mechanisms directly: the world-110 picture (pacing a sealed slot) -> frustration, the
world-117 picture (circling rooms full of food, 42 idle presses) -> the thin-wall leak;
the world-100 picture confirmed the course's rationale after the fact. The course itself
came from the NUMBERS: a theory line ("why do open worlds plateau at ~28?") plus the
map's size. One disclosure that matters for the cross-arm comparison: this agent also
READ the harness and world source (`cortex/atlas.py`, `atlas_law.py`, `world.py`,
`tiles.py`, `senses.py`, `nomos/engine.py`) for facts the system text does not give —
the 1200-tick regrowth, the frontier's exact definition (so `source - 0.12 * front`
isolates food), material brightness (stone 0.50, door 0.77, trunk 0.41), `shift`/`posx`
orientation and the lattice centre 14/29 = 0.4828. It lived no worlds itself (only
`mailbox check`). A FRESH subagent is told to read nothing but the request; part of
FULL's edge may be this, not memory.

**What the run taught about the lab.** (1) The tuner jitters `shift()` offsets and the
engine truncates them with `int()` (`shift(here, 0.998, 0)` = `shift(here, 0, 0)`), so
tuning can silently break a law's geometry (tuned #15: holdout 33.0* -> 28.8); it also
jitters `when` constants (harmless). Not fixed — it is the exam. (2) The vocabulary-twin
note fired on 14 laws in a row once the course existed: a productive axis needs its
substrate, so the note measures reuse, not stuckness. (3) The session's worker process
was restarted twice and took the lab with it; the same command resumed from
`state.json` with no lost turn ("resumed 2x").

## 2026-11-05 — cloud study BLIND-1

Arm BLIND (the session's own agent answers every request, observer OFF:
`--observe-every 0`), replicate 1 (`--train 0-19 --holdout 20-35 --seed 0`), 12 h,
control = confine (`laws/atlas_confine.nomos`) tuned by rng, anchor = ghost. Run:
`runs/cloud-lab/blind-1/` (branch `cloud-lab-blind-1`). 31 design turns (6 deep), 31/31
parsed first try, 0 constants-only twins, 5 theory rewrites, no OBSERVE turns.

**FINAL worlds** (seeds 36-67, seen by nobody until the end):

```
reference wave (#0)                 18.8 meals  immortal 28/32
confine, hand-written (#1)          24.5 meals  immortal 28/32
CONTROL: #1 tuned 159x              25.9 meals  immortal 29/32
#26 tuned 5x  (champion)            33.0 meals  immortal 32/32
#29 tuned 5x                        32.8 meals  immortal 32/32
#33 tuned 2x                        31.9 meals  immortal 32/32
#30 tuned 5x                        30.7 meals  immortal 32/32
```

**Champion vs control on FINAL: 33.0 vs 25.9 (+7.1, +27%), 32/32 immortal vs 29/32.**
Every law of the model that reached FINAL beat the tuned control by >= 4.8.

**Drops train -> holdout -> final:** champion #26 33.7 -> 32.5 -> 33.0 (none); #29
33.5 -> 32.2 -> 32.8; #33 32.5 -> 31.3 -> 31.9; #30 33.2 -> 30.6 -> 30.7; CONTROL
29.2 -> 28.2 -> 25.9 (-3.3, the only real drop — 159 accepted jitters, again). The
champion was tuned 5x and as-written = tuned (33.7): the structure carried it.

**Prediction error** (|predicted - actual| meals before tuning): first half 2.4, second
half 2.7 over 31 predictions — no gain in prediction across the night.

**The champion** (#26) is a stack of six mechanisms, each added in its own turn and kept
only while it held up: honest appetite (morsels pull only through the want-weighted
`source`, never `thing`=1) + a halo round refusals; ballistic exploration (frontier
muted by `exp(-k*behind)`, a screened diffusion of the walked path); diagonals only where
the body fits (both corner cells open; a diagonal neighbour the body cannot reach reads
0); frustration (visit density under the agent rises -> smell-only lures muted for ~100
thoughts); fog-discounted wave (unknown cells pass less) + sharpened wall trust (untaught
walls pass ~1/3, stone ~0); and the last, largest step: **pure frontier marks weighted by
the blurred fog mass around them** (+2.9 train). An ablation turn (#28: weight by plain
distance from the agent instead) fell back to 30.1, so the gain is information, not reach.
Two Occam turns showed parts do real work: confinement on seen morsels (#21, -2.4 without)
and the aversion halo (#31, -2.3 without). Failures were as informative: invented lures on
landmarks are catastrophic (foliage "groves" #25: 19.9, -10.9; probing every wall #12:
-5.2), anything weakening ballistic search loses (area-restricted search #18), and costs on
cells the body can still use lose (squeezes #19, turn cost #16).

**The theory** ended as "meals come from exploring for information and from honest
prices", with every claim pinned to a turn number; it agrees with the champion and the
FINAL table. It corrected itself once on evidence: the first version read #8's -6.4
(two-cell express wave) as "a slow wave is useful inertia"; #14 (a slower wave, -3.2)
refuted that, and #20 (express hops in straight lines only, neutral) showed #8's loss was
corner-cutting, which #15 had meanwhile fixed.

**What this arm's agent did differently** (no pictures): with no lives to look at, it
reasoned from the per-world meal vectors on the board and from reading the instrument's
source (`cortex/world.py`, `cortex/atlas.py` — read only, never lived): that `thing` is 1
on poison too, that `ate` counts nightcaps as meals, that sight reaches 22 cells while the
map holds 12, that leak starts at 0.5 for every hue, that `source` is already hushed round
`bad`. Most of its gains came from that reading — finding what the wave *misprices* (the
body at corners, unknown ground, untaught walls, poison) rather than from watching a
failure. It also used ablations and controlled re-tests as experiments (#20 vs #8, #28 vs
#26, #21, #31). What it could not do: it never saw world 16 (fixed blind by #9's learned
doors, then lost and won back by later laws without knowing why) and it had no way to tell
a trap from a slow approach except through its own frustration channel. Night 1 (deepseek,
6 h, with pictures): champion 24.5 on the same FINAL worlds; here 33.0.

Instrument notes (bookkeeping only, nothing changed): `mailbox check` on a bare `.nomos`
file goes through `extract_law`'s fallback, which starts the law at the first occurrence
of the word "channel" — a header comment containing "channel" therefore truncates the law
("expected channel name, got 'and'"). Replies are fenced, so the lab itself is unaffected.
The tuner also perturbs `when` constants and `shift` offsets (e.g. `shift(open, 1.028, 0)`),
knobs with no effect that dilute its 5-try budget.

## 2026-11-05 — cloud study BLIND-2

Arm BLIND (the session's own agent answered every request, no observer: `--observe-every 0`),
replicate 2 (`--train 100-119 --holdout 120-135 --seed 1`), 12 h, `--control-law
laws/atlas_confine.nomos`. Branch `cloud-lab-blind-2`, run `runs/cloud-lab/blind-2/`.
38 design turns (7 deep), 1 repair-after-bounce, 1 behavioural twin, 0 parse failures,
6 theory rewrites, 40 islands.

**FINAL** (seeds 36-67, lived once at the end):

```
reference wave (#0)            18.8 meals  immortal 28/32
confine, hand-written (#1)     24.5        28/32
CONTROL: #1 tuned 191x         26.9        30/32
#25 tuned 5x (CHAMPION)        26.9        30/32
#25 as written                 27.5        30/32
#5 tuned 7x                    27.0        30/32
#4 as written / tuned          26.5 / 26.2 29/32
#10 tuned 8x                   26.2        29/32
#30 tuned 5x                   25.6        29/32
```

**Champion vs control on FINAL: a tie, 26.9 = 26.9.** The model's structures did not beat
night 1's confinement tuned 191x; every stack law sits within ~1.5 meals of it on FINAL. Both
beat the hand-written confinement (24.5, the same number night 1 measured on these worlds)
and the reference wave (18.8). The best FINAL number is the champion AS WRITTEN (27.5) —
its tuning added nothing that transferred.

**Drops, train -> holdout -> final:** champion #25 (area-restricted search) 32.2 -> 29.4 ->
26.9; control 29.8 -> 24.6 -> 26.9; #10 (the sweep, best on train all night) 34.7 -> 25.4 ->
26.2. Train ranking predicted nothing: the train leader was 8 meals ahead of the control on
train and 0.7 behind it on FINAL. Holdout (16 worlds) moved the right way but its 29.4 was
also mostly luck.

**Prediction error:** 3.7 meals first half, 5.3 second half (38 predictions) — no gain; I
predicted ~31 for almost everything and most laws landed 22-31.

**What this arm's agent did.** Without pictures it designed from the per-world meal vectors,
the code (world.py's economy, atlas.py's painting and read-out) and ablations. The first five
turns stacked onto confinement: frontier weighted by the unknown space behind it (`vast`,
+5 on train), dwell-frustration lighting known wall faces (rescued a 0-meal world, 0 -> 32),
distrust of unknown cells in the wave. From then on every addition landed 26-31 on train,
within the ~2-meal noise of a 20-world mean, so the second half turned to ablations and
mirror tests, which produced the night's firm findings (each a loss when removed or bent):
the max() wave (a mean-based potential field: 11.3), `vast` (removed from the sweep law:
34.7 -> 21.7), the far smell (weighing it by its own faint strength: -10; boosting it: 0),
the raw pull of seen morsels (`source` only: -4 — and a nightcap eaten still counts as a
meal), the plain wave discount (far-sighted and near-sighted frontiers both lost 5-8), the
8-way wave (orthogonal turn-aware and Manhattan waves lost). Distrust of unknown cells
starves exactly one world (112: 0 meals with it, 26 without) but helps the rest; lifting it
while frustrated made lives even (20/20 immortal, worst world 16) without raising the mean.
The theory followed the evidence (each claim cites its island) and its last version agrees
with the champion's mechanism, but the theory's honest conclusion — "nothing separates on
holdout" — was also the morning's.

Disclosures: I read the control's tuned law in `state.json["control"]` once at 4.8 h (its
diagonal discount had been tuned to 0.76, below two straight steps; that motivated two
designs, both of which lost), and checked one law's heading channel with `LawRule.step` on
a hand-made test map (no world lived). The VM's worker was restarted twice near the end; the
lab was restarted with the same command and resumed from `state.json` (one turn re-asked).

## 2026-11-05 — cloud study FRESH-1

Arm FRESH (a NEW general-purpose subagent per mailbox request, never reused; it read only
the system text, the request and the PNG, and could run `mailbox check`), replicate 1
(`--train 0-19 --holdout 20-35 --seed 0`), 12 h, control = `atlas_confine` tuned.
Run: `runs/cloud-lab/fresh-1/` (branch `cloud-lab-fresh-1`). 29 design turns, all
valid on the first try, 0 twins, 6 observations, 4 theories.

FINAL (32 sealed worlds, seeds 36-67; paired vs the control, SE of the paired mean):

```
                               FINAL  vs control   SE   wins/losses
reference wave (#0)            18.8     -7.4      2.3     10/20
confine, hand-written (#1)     24.5     -1.7      1.8     10/17
CONTROL: #1 tuned 158x         26.2       —        —        —
#6 tuned 7x  (summary champ)   28.6     +2.4      2.0     18/12
#16 tuned 6x                   29.7     +3.4      2.0     19/12
#11 tuned 6x                   30.2     +4.0      2.5     20/9
#31 tuned 2x                   29.1     +2.8      2.1     18/12
#21 tuned 7x                   27.8     +1.6      1.7     15/13
#6 as written                  27.2     +1.0      2.1     16/12
```

**Champion vs control on FINAL: the model's structure wins, modestly.** #6 (chosen by
holdout) beats the tuned confinement by +2.4 meals (~1.2 SE). Every model law on the
board, tuned or not, beats the control, but they are all descendants of #6, so that is
one result seen several times, not five. Night 1's bar was the ghost (17.1). This
night's bar was night 1's champion, tuned 158x (26.2), and it was still cleared.

Train -> holdout -> final: #6 28.5 -> 33.6 -> 28.6; #16 28.8 -> 32.6 -> 29.7; #11
26.9 -> 30.1 -> 30.2; #31 29.9 -> 28.5 -> 29.1; #21 29.1 -> 27.8 -> 27.8; control
28.5 -> 27.0 -> 26.2. The holdout-picked champion lost 5 meals on FINAL. 16 holdout
worlds, looked at for every candidate, are themselves a selection surface: winner's
curse one level up. The train leaders (#21, #31) did not drop on FINAL.

Prediction error: 3.2 (first half) -> 3.3 (second half), no gain. 27 of 29
predictions were too high, mean +3.2. Almost every prediction is "best so far + 1"
(29.5 or 30.5 from turn 6 on). A fresh mind has no calibration memory, and the lab's
prompt does not show it its own past errors.

Champion mechanism (#6): a per-cell step price. Known-free ground is cheap (0.95),
unknown dear (0.89), and wall-hugging dearer. Diagonals cost `step^sqrt2` and pass only if
neither corner they squeeze past is solid. Frontier marks are weighted by the unknown
behind them, on top of #1's confinement divisor. Theory (last at 9.8 h): "how the wave
SPREADS helps; anything that outvalues real food hurts". That agrees with the board (#6
holdout champion; #24/#26 grove/second-field losses), and unlike night 1's theory it does not
contradict its own champion. Its "dig commitment helps" claim (#16 holdout 32.6) also held
up on FINAL (29.7, second-best by holdout choice).

What this arm's agent did differently: with no memory, every subagent rebuilt its
picture from the request alone. The lab's own notes were enough for the agents to stay
on topic for the whole night. From turn 7 on they marched down one line: sealed pocket
on world 16 -> trapped gates (#9-#12) -> dig commitment (#13, #16) -> claim with a
clock (#21) -> food veto (#31). The observer kept handing them world 16 (4 of 6
pictures). Night 1's failure mode came back in a milder form: world 16 was not a 0
for everyone (#5/#7/#13/#16/#23-#25/#31 ate 15-35 there), but ~20 turns went to
escape/commit variants. The fresh minds were good AUDITORS: three separate subagents
found from the tuned text alone that #21's `spent` habituation could never fire (age
capped at 1, threshold 1.187), a dead term that tuning had created. Bookkeeping:
(1) for request 0004 my dispatch prompt named the DESIGN system file instead of the
observer one the request named. The reply was in shape and was kept; from 0010 on the
prompt used the file the request names. (2) Three OBSERVE replies ran to six sentences.
They were left as written, per protocol. (3) The session's worker restarted at 11.88 h
and killed the lab. It was resumed with the same command (`resumed 1x`), and no answer
was lost.

## 2026-11-05 — cloud study FRESH-2

Arm FRESH, replicate 2 (`--train 100-119 --holdout 120-135 --seed 1`), branch
`cloud-lab-fresh-2`, run `runs/cloud-lab/fresh-2/`. 12 h (12.9 with the FINAL lives), 40
requests: 30 DESIGN, 5 OBSERVE, 5 THEORY. Every request was answered by a NEW subagent that
saw only the system text, that one request and its PNG; the dispatcher passed nothing on.
The session's worker was restarted once at 10.9 h, which killed the lab; the same command
resumed it from `state.json` ("resumed 1x" in the summary), and no request was lost.

**FINAL** (seeds 36-67, the same 32 worlds for all six arms):

```
reference wave (#0)            18.8 meals  immortal 28/32
confine, hand-written (#1)     24.5        28/32
CONTROL: #1 tuned 152x         25.8        30/32
#30 tuned 5x (champion)        30.7        30/32
#9  tuned 6x                   30.2        31/32
#28 tuned 7x                   30.2        30/32
#23 tuned 7x                   29.9        30/32
#9  as written                 31.8        32/32
```

**Champion vs control on FINAL: 30.7 vs 25.8 (+4.9, +19%).** All five model laws on the
FINAL list beat the tuned control by 4.1-6.0 meals, so this is not one lucky law. The
champion by holdout is #30 (a walk-only `hope` wave that silences wall-face probes while
anything is still reachable on foot), built on #23 (a fixed random `grain` per cell,
anchored to the world, picks which wall face wins).

**Train -> holdout -> FINAL:** control 29.7 -> 24.6 -> 25.8; #30 34.5 -> 31.4 -> 30.7;
#23 34.8 -> 29.9 -> 29.9; #28 34.4 -> 29.9 -> 30.2; #9 tuned 31.7 -> 29.8 -> 30.2. Every
law loses 3-5 meals from train to holdout and then holds on FINAL, so holdout was an honest
estimate. The night-1 lesson repeats: **#9 untuned (27.9 on train) is the best law on
FINAL, 31.8 and 32/32 immortal**; six rounds of tuning took 1.6 meals off it. The
constants overfit and the structure is what carries.

**Prediction error** (|predicted - actual| before tuning): 4.9 in the first half, 4.7 in
the second, over 30 predictions. 25 of the 30 were too high, typically by 1-4 meals, and
the worst misses were +31.5 (a law that ate 3.0) and +13.7 (one that ate 20.3). Fresh
minds carried no calibration from turn to turn, and it shows.

**What this arm's agent did differently.** With no memory, every subagent rebuilt its
picture from the request: the board, the theory the lab carries, and the champion's text.
So the lab's own notes WERE the mind. The result was a strongly incremental chain: nearly
every design was "the current best law + one or two channels" (19/30 were flagged as
vocabulary twins of earlier islands), and the arm never made a clean break. The
observer's world-110 picture (a slot-born agent chipping two faces) set the agenda for 25
turns; the turns that answered it moved the score (#7 effort on the aimed face, #9 crown,
#13 thin walls, #23 grain tie-break, #30 hope gate). Three fresh subagents in a row (0016,
0017, 0018) independently found the same bug in the champion, a futility exit that the
tuned constants had made unreachable (`work` saturating near 0.09 against a 0.9
threshold). Each re-derived it from the law's text alone, because none could know that
the previous one had found it. Zero repairs, zero laws that failed to load, zero
constants-only twins: the request format alone was enough for a cold model to write a
valid law first time, 30/30. The theory stayed consistent with the board (it names #23
and #30 as the leaders and lists the losers correctly), because each THEORY writer read
only the evidence in front of it and had no conversation to drift from. Cross-arm
comparison: to be written from all six branches.

## 2026-11-05 — cloud study: the six runs compared, and an exam on fresh worlds

Each run's FINAL (32 worlds, shared) was too coarse to compare runs: paired SE ~2.3, and
finalists were looked at there. So the champions, controls and a few FINAL leaders lived
128 worlds no run ever saw (seeds 1000-1127, `python -m cortex.study_exam`, SE ~1.1):

```
law                        meals  immortal  vs confine     vs own control
reference                   18.5   99/128    -5.1 +- 1.2   
confine                     23.5  108/128                  
full-1 control              25.7  112/128    +2.2 +- 1.1   
full-1 champion #19         26.5  113/128    +2.9 +- 1.1    +0.8 +- 0.9
full-1 #35                  28.5  117/128    +4.9 +- 1.2    +2.8 +- 1.0
full-2 control              27.8  118/128    +4.3 +- 1.0   
full-2 champion #28         32.3  126/128    +8.7 +- 1.1    +4.4 +- 1.1
full-2 #18                  32.8  125/128    +9.3 +- 1.1    +5.0 +- 1.2
blind-1 control             26.7  115/128    +3.1 +- 1.1   
blind-1 champion #26        31.0  122/128    +7.4 +- 1.1    +4.3 +- 1.1
blind-2 control             26.2  115/128    +2.7 +- 1.1   
blind-2 champion #25        25.2  118/128    +1.7 +- 1.2    -1.0 +- 1.1
fresh-1 control             26.0  117/128    +2.4 +- 1.0   
fresh-1 champion #6         27.2  115/128    +3.7 +- 1.0    +1.2 +- 1.0
fresh-1 #11                 30.8  127/128    +7.2 +- 1.1    +4.8 +- 1.1
fresh-2 control             24.6  111/128    +1.1 +- 1.0   
fresh-2 champion #30        29.8  125/128    +6.2 +- 1.2    +5.1 +- 1.1
fresh-2 #9 as written       28.9  124/128    +5.4 +- 1.2    +4.3 +- 1.1
```

**The arm question is not answered.** Champion means per arm: FULL 26.5 / 32.3, BLIND
31.0 / 25.2, FRESH 27.2 / 29.8 — replicates differ by up to 5.8, arms by at most 1.3. A
night's result looks like whether it hits a good idea, not which arm it is in. Two
replicates cannot resolve this, and the arms were not clean either: FULL and BLIND agents
read the world's source, FRESH subagents could not (`cloud-study.md` did not forbid it).

**What is solid.** The agent found structures far above night 1's champion: `laws/
atlas_course.nomos` 32.8 vs confine 23.5 (+9.3 +- 1.1). Three champions of six beat their
tuned control by 4-5, three did not. FINAL over-stated the two biggest wins (FULL-2 +9.7
-> +4.4, BLIND-1 +7.1 -> +4.3). Holdout picks imperfectly: FRESH-1's champion #6 27.2,
its #11 30.8.

**A correction: tuning does not hurt.** The pilot and several runs concluded that tuned
constants do not generalise. On fresh worlds all six tuned controls beat untuned confine
(+1.1 to +4.3, mean +2.7). Tuning over-states its gain on train, about half survives.
"As written beats tuned" for single laws (BLIND-2 #25, FRESH-2 #9) was not re-tested.

**The model does not learn to predict.** Prediction error fell clearly in one run of six
(FULL-1 4.8 -> 3.3); all six over-predict, by +2.4 to +4.4 meals. By the project's own
measure, this is engineering by trial and error, not a theory of the system.

**Left open:** the tuner jitters `when` and `shift()` arguments (wasted tries, sometimes
broken geometry); `mailbox check` truncates a bare law if its header comment contains
"channel". Whole runs (mailboxes, laws, pictures) stay on branches `cloud-lab-*`.

## 2026-11-05 — the spin exam: memory carried by the body's motion, and growth that pays

Question: can a brain of perceptron cells get smarter by GROWING, the same weights on a
bigger sheet? Exam (`cortex/pred/spin.py`, validation worlds only, test worlds still closed):
stand, turn twice, second lap with the eyes closed, predict the view. Score = share of the
road from MEAN (knows nothing, 0%) to a hand-made PANORAMA (100%).

**Why the sheet stood still.** Sheet 16x16, 2 048 numbers: 15.8%, as good as "the average look
of the place" (17%). GRU16 14%, GRU128 32%, GRU512 37% (1 029 000 weights). `cortex/pred/watch.py`
looks inside: 81% of the sheet's change is shared by all cells at the end of training (99.9%
at step 500), the read-out listens to nearly all of them, and over the closed lap its belief
about a ray moves 4% of the truth (GRU512 54%). Every cell hears one code and the read-out
averages, so there is nothing for size to do.

**The idea.** A ring of columns around the body (`FlowBrain`, 4 rows, 6 381 weights). A ray
lands in and is predicted from the columns it looks along: fixed wiring, like a retina. A wall
stays put in the world while the body turns, so memory must move: the action sets how much a
cell takes from each of its nine neighbours (a multiplicative input). Passes = columns // 8, so
a pass moves the same number of cells at any size. Hand-made ceilings, no learning
(known turn): world-fixed address 17/51/76/90/95% at 8/16/32/64/128 columns, upwind flow
23/40/50/60/68%, Lax-Wendroff 28/46/57/67/76%. On this exam size can pay.

**Each step, and what it taught** (share at 32 columns unless noted):

| arm | what it adds | result |
|---|---|---|
| `ring` | columns, no multiplicative input | 16.5% at 16 |
| `flow` | action sets the kernel | 14.9 / 9.2 / 0.4% at 8/16/32: WORSE WITH SIZE |
| `flowc` | kernel hears only the action, sums to 1 | 25.4, 25.6% at 16; 37.4, 36.9% at 32 |
| `flowc --sides 8,16,32` | trained across sizes | 36.7, 21.5, 32.1%: seeds disagree |
| `flowm` | left turn = mirror of right turn | 38.5, 38.9, 38.7% |
| `flowo` | cell rule once a frame, transport each pass | 36.9, 38.5%; EXPLODES beyond training |
| `flows` | all kernel shares >= 0 | 43.6, 41.1, 42.2% |

Findings, in order:
1. `flow` learned to carry memory with the eyes open, then ERASED it when they closed (kernel
   sum ~0.2): early on memory is unreliable and the mean is safer. Hence `flowc`.
2. Two predictions written beforehand failed outright: `flow` lost to its control and got
   worse with size. Everything that worked was found after looking at data.
3. Weights from 16 alone did not transfer (25% -> 5% at 32 -> -25% at 64). I then credited
   multi-size training; `flowc-32`, trained on 32 alone, transfers as well (16.5/29/37.4/39.7%
   at 8/16/32/64), so that credit was wrong.
4. **The prediction bench found the physical law itself.** Ideal shift = 0.92 cells per pass at
   full turn, proportional to the turn. Runs that took off had channels at slope 0.94-0.95 with
   offset ~0; runs that stalled stopped at 0.64-0.82, mostly with an offset. With the mirror,
   both seeds examined grew the same anatomy: 3 channels at 0.93-0.96 (carry the world), 4 near 0.78
   (role unknown), 1 near 0 (body-fixed).
5. Too many cell-rule passes wear memory out: `flowm` fell at 128 columns (27%, 32%); with the
   rule once a frame one seed rose to 49.0% at 128, unseen. But the other seed gave 10.9% at 64
   and -230% at 128, and the good one later fell too (39.8% at 256, -80% at 512). All the learned
   kernels amplify some frequency by 21-70% per pass at full turn, and bigger rings run more
   passes. Non-negative shares make that impossible, and they also score better at every trained size.

**`flows`, the same weights at every size (trained on 8/16/32):**

| columns | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| seed 0 | 30.0 | 38.5 | 43.6 | 46.5 | 46.7 | 44.8 | 34.6 |
| seed 1 | 29.9 | 37.9 | 41.1 | 38.6 | 30.5 | 23.2 | 17.5 |
| seed 2 | 29.8 | 37.9 | 42.2 | 43.6 | 41.5 | 39.5 | 32.7 |

Written beforehand: every seed rises to 128 and scores at least 40% there. **Not met**: only
seed 0, and its 64->128 step is a tie. What holds: nothing explodes up to 512, trained sizes
agree closely and rise on all three seeds, and at 32 columns all three beat GRU512 (error
-0.039 to -0.064, se 0.005) on 161 times fewer weights. Beyond training it degrades gently, and
the peak varies by seed (32, 64, 128). Why it declines is not known.

**Honest limits.** The body stands still and the turn is felt. Wiring, conservation, mirror and
non-negativity are design, all chosen after seeing data; only the speed and what cells do with
memory are learned. The ceiling (upwind 60-68% at 64-128) is still far off. Nothing here acts in
a world yet. Test worlds stay closed for a final exam.

**Training up to 64 (`flows --sides 8,16,32,64`):**

| columns | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| seed 0 | 27.7 | 36.5 | 42.2 | 45.5 | 46.0 | 44.1 | 37.8 |
| seed 1 | 27.8 | 36.4 | 41.0 | 43.0 | 41.4 | 36.2 | 27.3 |
| seed 2 | 28.3 | 38.1 | 43.5 | 47.4 | 50.0 | 51.6 | 46.3 |

Written beforehand: on every seed 64 beats 32 and 128 is no worse than 64. **Not met**: 64 beats
32 on all three, but seed 1 falls at 128 (and seed 0's gain there is -0.005 +- 0.002). What moved:
the peak sizes went from 128/32/64 to 128/64/256, and the worst seed's best score from 41.1% to 43.0%,
at a price of about two points at 8 columns. Growth now holds on every seed across the whole
trained range; at 64 columns all three beat GRU512 (error -0.058 to -0.100, se ~0.007). Every
seed still falls by 512. Best single figure 51.6%, still below the upwind ceiling.

**On the walking data (`cortex/pred/bench.py`), one arm, one size, 3000 steps.** The ring can only
carry rotation here: it has no way to express the body moving. Skill = 1 - error / best simple
rival, 0 = no better than the best trick.

| brain | weights | eyes open | h1 | h8 | h8, turned | h16 | h32 |
|---|---|---|---|---|---|---|---|
| `flows` 32, seed 0 | 6 381 | 0.573 | 0.222 | 0.093 | 0.091 | 0.059 | 0.006 |
| `flows` 32, seed 1 | 6 381 | 0.563 | 0.203 | 0.097 | 0.080 | 0.050 | 0.011 |
| `flows` 16 | 6 381 | 0.547 | 0.175 | 0.087 | 0.081 | 0.055 | 0.010 |
| GRU128 | 117 000 | 0.473 | 0.072 | 0.077 | 0.076 | 0.049 | 0.042 |
| wide128, 20 000 steps | 142 000 | 0.539 | 0.155 | 0.076 | 0.058 | 0.080 | 0.022 |
| sheet 16 | 49 000 | 0.361 | -0.123 | 0.056 | 0.066 | 0.026 | 0.006 |

Written beforehand: both 32-column runs above 0.076 on h8 turned. **Met, weakly**: errors 0.889
and 0.899 against GRU128's 0.903 on 88 episodes, with no paired test, so memory at h8 may be
a tie. The clear gain is SHORT range: eyes open and h1 beat every earlier brain, on far fewer
weights. At h32 nothing beats simple rivals by much, and GRU128 is ahead. So walking memory
is still almost absent, as expected of a ring that cannot express translation.

The paired oracle check (`cortex/pred/oracle.py`, rays whose answer is a wall already seen) settles
it: at h8 flows vs gru16 is +0.002 to +0.012 +- 0.013, no gain. Error there: mean 0.463, all
brains 0.43-0.46, hand memory with the true pose 0.172. Nobody remembers while walking, and there
is 2.5x to win.

**The polar sheet (`PolarBrain`, `polar`).** Columns = directions, rows = distances (columns // 4),
rays land on whole columns, a ray is read through attention over the rows. A step moves near things
across many columns and far ones across few, so the shares depend on the cell's place as well as
on the motion. Hand ceilings with the true pose (scratch, crude read) did not separate polar from
a north-aligned square, both improve with size; polar kept as agreed, half the cells.

Written beforehand: at 32 columns every seed under 0.38 on answerable rays at h8. **Not met,
narrowly**: 0.389, 0.404, 0.385. Against gru16 -0.035, -0.018, -0.036 (+- 0.014), so one seed is
inside the noise. Yet it is the best brain so far at every horizon up to h8, on 7 119 weights:

| answerable rays | h1 | h2 | h4 | h8 | h16 | h32 |
|---|---|---|---|---|---|---|
| polar, three seeds | 0.243-0.259 | 0.268-0.292 | 0.305-0.320 | 0.385-0.404 | 0.383-0.412 | 0.487-0.511 |
| flows 32 | 0.268-0.287 | 0.308-0.331 | 0.356-0.368 | 0.430-0.435 | 0.399-0.407 | 0.476-0.502 |
| GRU128 | 0.340 | 0.363 | 0.392 | 0.431 | 0.405 | 0.499 |
| hand memory | 0.120 | 0.110 | 0.131 | 0.172 | 0.185 | 0.240 |

Skill rises with size on all seeds at h8: 0.078-0.096 / 0.112-0.130 / 0.127-0.143 at 8/16/32.
**Is the law found?** Turning, yes: 6 of 8 channels at 0.90-0.92 against the ideal 0.92, all seeds.
The step only partly: shape correlates (columns ~0.8, up to 0.97 on rows), size is 0.1-1.5 of the
physics, some channels flow the opposite way, and the memory drifts at rest. (My first read of
this was wrong: I fed the speed sense in cells per tick, it is a fraction of the top speed.)

The memory-pressure reaper stopped three parallel runs at step 500 on 2026-11-06; resumed one at
a time, resumable as designed.

**Three more shapes for the flow, same bar, none met it (answerable rays, h8, 32 columns):**
`polarl` (flow = motion x a learned map of the place, nothing flows at rest, learned sharpness)
0.398-0.413. It did NOT raise the sharpness: a cell keeps only 83-90% per pass at rest. I took
that blur for the culprit, so `polarv` learns a SPEED and derives the exact shares, keeping 100% at
rest: 0.401-0.421, slightly worse. That hypothesis was wrong. Many channels flow along rows at
correlation about -1 to the physics. Perhaps rows have no fixed meaning and a channel may hold
distance upside down; not tested.

**A false verdict, caught.** `polarp`: true physics by hand, writing and reading learned: 0.399-0.429.
I concluded "transport is not the bottleneck". Then the same grid and flow entirely by hand scored
0.71 at 16 columns and 2.64 at 64, far worse than the mean and worse with size. So that flow was
not a fair stand-in: it clamps at one cell a pass, and near the body a passing wall needs many
(the polar singularity, a guess that fits). The ablation answered nothing.

**The real finding: smoothness itself loses the memory.** Hand-made memory with the TRUE pose and the
same crude read, answerable rays:

| h8 error | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| polar, smooth rigid resampling | 0.798 | 0.569 | 0.651 | 0.612 |
| compass square, smooth | 0.815 | 0.785 | 0.677 | 0.564 |
| compass square, WHOLE-CELL jumps | **0.326** | **0.263** | not finished | not finished |

Mean 0.463, dictionary memory 0.172. Every bilinear step blurs; a jump does not. The dictionary and
the atlas's scroll never had this problem. My earlier plan "replace the atlas's scroll with a
smooth learned flow" would have been a step back.

**`north`: compass square, whole-cell jumps by hand, writing and reading learned.** A cell knows its
direction relative to the body, so which ray it hears changes as the body turns (live wiring). A ray
is read by marching along it, cells may stop it. 5 167 weights, path counted from commands and felt
heading. Written beforehand: every seed under 0.30 at 32 and better than at 16. **Not met**: 0.371,
0.371, 0.353; 16 gives 0.381, 0.372, 0.359, so size barely pays. 64 (unseen) is worse (0.400-0.409).
But it is reliably the best learned memory so far, skill at h8 0.204-0.211 (polar 0.127-0.154, GRU128
0.077), h32 0.092-0.094 (GRU128 0.042), against gru16 at h8 -0.036 to -0.055 (+- 0.018).

**Counting the path.** Turning is exact, speed is not: commands promise 1.27 cells a frame, the body
moves 1.01, and about 9% of moving frames go nowhere. Path error at 8 / 32 frames: commands 2.44 /
7.35, true distance along the heading 0.57 / 1.30. A small perceptron on commands and body senses
gets 1.64 / 3.96, with the current view 1.16 / 2.68, but it was fitted on the true pose, which a
brain does not have. So the signal is there; how to get it without a teacher is open.

`north --true-walk` (the real distance as the thrust, an ablation, 2 seeds): answerable h8 still
0.348, but long range improves a lot: answerable h16 0.331-0.339 (was 0.374-0.396), h32 0.412-0.458
(was 0.513-0.532); all rays skill h8 0.261-0.276, h32 0.151-0.158. So the path count limits the long
range, and writing and reading limit h8 (hand read with a true pose: 0.263).

**Where this leaves the question.** Learned perceptron cells hold a memory that beats every other
learned brain here, on 5 167 weights. But what makes it work is structure the atlas already had by
hand: the compass grid and the jump. Learning a smooth flow, which is what paid on the spin ring,
did not carry over to walking. Growth paid clearly on the ring and barely here.

## 2026-11-07 — the sheet gets a body: a map nobody showed it feeds it

Goal: a brain that acts. `cortex/pred/act.py` hands the atlas's idea of SOLID to the sheet and changes
nothing else: wave, frontier, appetite, FREE and the other channels stay hand-made. Wall belief =
`NorthBrain.believed_solid`, the chance a ray stops within a cell, learned from predicting senses only.

**The belief was not a map at first.** `north`: 0.167 on walls, 0.135 off, AUC 0.65-0.69 (hand
dictionary with the true pose 0.78-0.81). Nothing forced it: distance could be read out of the state
without ever stopping the ray. `northr` forbids that: nearness and hit come ONLY from where the
cells stopped the ray, as in a renderer. Bar written first: out-of-view AUC at least 0.75. Met,
0.758-0.771 on all three seeds. Memory on answerable rays jumped: h8 0.267-0.292 (was 0.353-0.371; hand
read with a true pose 0.263), -0.161 to -0.183 (+-0.015) against gru16. The price: over ALL rays it is
worse, h8 0.88 against 0.77, eyes-open skill 0.35 against 0.60. A renderer cannot hedge about the unseen.

**In the world, three wrong turns** (32 worlds, 8000 ticks; hand 18.4, blank 16.2):
1. Belief alone 12.0-14.7, worse than BLANK, and the ladder drops at `maze` (0.17-0.22, others 1.00).
   Written bar missed. Stricter cut (false walls at 10%, set on training lives) worse still, 7.1-10.0.
2. I concluded "knowing walls is worth almost nothing, even a perfect map". WRONG. Wiping parts of the
   hand atlas every thought: forget solid 4.0 (-14.4 +- 2.0), forget free 16.3, forget things 19.8,
   forget ghost 17.8, forget all 4.8. Remembering walls is essential; BLANK was never blank, it keeps
   every wall it bumped. Seeing walls by rays is what adds little.
3. So my belief arm was unfair: each frame it replaced the map and forgot what the body had bumped.

**Fixed:** `beliefV` = belief + remembered touch + no wall where a ray has been seen to pass. 32 worlds:
17.3-18.1, bar missed vs blank (+1.1 to +1.9, +-2). Chosen after looking, so confirmed on 128 FRESH worlds
(1032-1159), same bar written before running:

| arm | meals | immortal | vs blank | vs hand |
|---|---|---|---|---|
| hand | 19.5 | 105/128 | +5.2 +- 1.0 | |
| `beliefV` s0 | 17.6 | 102/128 | +3.4 +- 0.8 | -1.8 +- 0.9 |
| `beliefV` s1 | 18.2 | 104/128 | +4.0 +- 0.9 | -1.2 +- 1.0 |
| `beliefV` s2 | 17.8 | 103/128 | +3.6 +- 0.9 | -1.7 +- 0.9 |
| blank | 14.2 | 82/128 | | -5.2 +- 1.0 |

**Met on all seeds.** A wall map no one showed it, learned only by predicting senses, earns 65-77% of
what hand-painted walls add over bumping, and lives about as long as hand. First learned sheet that
acts and is paid for it.

**What is NOT learned here:** the compass grid, the whole-cell jump, reading by marching, the wave,
frontier, appetite, FREE (hand-painted, and it vetoes the belief, so part of this credit is the eyes'),
touch memory, the cut (set on training lives against the true map, which is a hand calibration). The
atlas's pose for its own scroll is true; the sheet counts its own. It was never tried on the best law
(`atlas_course`, 32.8 with hand walls). Nothing of this has met the test worlds.

A latent crash surfaced on the way: `Atlas._read` faced `last_target` when food lay exactly under the
body and the wave pointed nowhere (None). Fixed for that case only, every other path is unchanged.

## 2026-11-11 — tissue: a restart, and rung 0: 76 numbers hold a pair per interneuron at sizes never seen

**Why a restart.** The user's read: the evolving brains of vivarium/aqua improved by growing their
perceptrons, not their tissue, and everything since followed from that. The saved pools fit it: rule
depth at its cap of 4 almost everywhere, hidden width 16 -> 23-32 (cap 32), cell types 4 of 4, cells
48-55 (aqua) and 11-39 (vivarium) of 144. Not proof: depth and width were free, mutation was biased
to grow, only cells cost energy, and no ablation was ever run. New package `tissue/`, design and
rules in `docs/tissue-design.md`. Decided with the user: the goal is growth that pays plus explicit
memory and thinking; gradient may search for the genome, learning in a life stays local; the
perceptron in a cell is small and fixed; exams before worlds; one mechanism per step.

**Exam 0, associative memory** (`tissue/assoc.py`). A life = K random pairs, cue 12 bits -> answer
4 bits, each shown once, then every cue asked. K90 = pairs held at 90% of the bits. Hand-made
memories (`tissue/hand.py`, M threshold interneurons with random input wires, best sparseness each)
are the yardstick, in the table below. Direct sensory -> motor Hebb with no interneurons: 7.9.

**Kernel v0** (`tissue/kernel.py`). Sensory cells, M interneurons, motor cells. Sensory -> inter
wires innate (random in each individual, the genome sets a gain); inter -> motor wires plastic from
zero. One perceptron per cell type, 8 hidden. A wire changes by (source activity) x (target's
learning signal). Genome: 76 numbers. Search: gradient through whole lives, sizes 8-64, 1/8 to 2
pairs per interneuron, 6000 steps, 5 minutes a seed.

**Bars written before the search** (in the design note): B1 frozen wires at chance (0.5 +- 0.03);
B2 K90 rises at every doubling 8 -> 512 on every seed and K90(512) >= 2.5 x K90(64); B3 at least
hand Hebb from 16 up. My prediction: within 20% of hand delta at 8-64, behind it at 256-512, B2
narrow or missed.

**Result, K90** (`python -m tissue.exam`; * = a size the search never saw):

| M | seed 0 | seed 1 | seed 2 | hand hebb | hand delta | least squares |
|---|---|---|---|---|---|---|
| 8 | 8.2 | 7.7 | 7.8 | 4.3 | 5.9 | 12.0 |
| 16 | 16.5 | 16.2 | 17.0 | 7.4 | 8.9 | 25.6 |
| 32 | 33.0 | 30.4 | 31.3 | 12.6 | 16.2 | 51.3 |
| 64 | 67.8 | 63.4 | 66.1 | 20.9 | 28.9 | 102.3 |
| 128* | 128.6 | 121.5 | 124.3 | 34.3 | 46.4 | 210.4 |
| 256* | 251.4 | 227.0 | 245.4 | 52.9 | 78.2 | 419.9 |
| 512* | 461.2 | 390.6 | 450.7 | 78.5 | 123.1 | 847.7 |

- **B2 met**: rises at every doubling on all three; K90(512) / K90(64) = 6.8, 6.2, 6.8 (8 would be
  perfectly linear). Pairs per interneuron: about 1.0 up to 128, 0.89-0.98 at 256, 0.76-0.90 at 512.
- **B3 met**, by 2x at 16 and 5-6x at 512.
- **B1 missed by the letter**: 0.465 .. 0.515. With frozen wires the answer is one constant, so
  this is the luck of the answers drawn at K = 4 (512 bits, se 0.022), the same figure at every
  size and seed. The control itself is exact: a test asserts the answers do not change when
  every taught answer is flipped. The bar was set tighter than the exam can resolve.
- **My prediction failed.** It beat hand delta by 1.4-2.3x at trained sizes and 3.2-3.7x at 512.
  I assumed the search would keep a threshold code and only find the rule.

**What was found** (`python -m tissue.inside`), the same on all three seeds:
1. *A code that is a hash.* An interneuron's activity is NOT monotonic in its dendrite: a narrow
   peak near zero with a dip beside it, input gain 3.4-5.6. Codes of two cues one bit apart
   correlate 0.28-0.37 (hand threshold units 0.46-0.63); three bits apart 0.04-0.09 (0.15-0.33).
   The exam never asks for likeness between cues, so the best code keeps every cue apart.
2. *A rule between Hebb and delta.* The learning signal has the teacher's sign and shrinks when the
   dendrite already agrees (+2.4 at 0, +1.5 at +2, +2.6 at -1): error correction that saturates.
3. *Something nobody wrote:* with NO teacher the signal is not zero but opposes the cell's own
   answer (-0.3 at +0.5, -0.6 at +2). The tick after a pair is taught the motor cell still hears
   that pair's code, so each lesson is a large step and a partial retraction sized by what the
   step produced. Its worth is not measured yet (ablation owed).

**Whose merit, the code or the rule** (found code under the hand rules):

| M | hand code + hand delta | found code + hand delta | found code + found rule | least squares, either code |
|---|---|---|---|---|
| 64 | 28.9 | 46-50 | 63-68 | 102-104 |
| 512 | 123.1 | 263-335 | 391-461 | 827-848 |

The code is the larger part (x1.7 at 64, x2.1-2.7 at 512), the rule adds about x1.35 on top. The
ceiling does not move: a better code does not hold more, it lets a local one-pass rule reach more
of what is there. The found tissue reaches 62-66% of that ceiling at 64 and 46-55% at 512.

**An ordinary net, same exam** (GRU, one seed each, 6000 steps, not tuned, one tick per pair):
64 hidden (16 k numbers) K90 = 1.4; 256 hidden (213 k numbers) K90 = 5.6. A weak baseline, still
improving when stopped, so only the shape counts: its capacity is one figure that more weights
buy slowly, while 76 numbers hold 8 or 460 pairs depending on how big they are grown.

**Limits.** By hand: the cell types, which wires exist, the average on a dendrite (this is what
makes a cell's input independent of size), the product form of the rule, the teacher's timing. The
theory says a delta rule on separated codes scales with size, so growth paying here is expected,
not a discovery; what is new is that a search over 76 numbers at 8-64 cells found the code and
the rule, and that they hold at 8x the largest size it saw with a 10-25% loss per cell. "Smarter"
here means capacity only. Nothing yet about thinking, recurrence, or wiring from the genome. Seed 1
found a blunter code and loses most at 512. Why any of them lose there is not known.

Next: rung 1, chains (A->B, B->C, asked A->C), where the answer must be fed back and time to answer
should grow with the chain. 8 tests in `tests/test_tissue.py`.

**The retraction, switched off** (same day; `python -m tissue.exam ... --taught-only`: motor wires
change only on a tick the teacher is on; the same three genomes, nothing searched again). My guess
was a 10-30% loss, held in my head and not written down first, which breaks my own rule.

| K90 | seed 0 | seed 1 | seed 2 | hand delta on the found code |
|---|---|---|---|---|
| 64, as found | 67.8 | 63.4 | 66.1 | |
| 64, no retraction | 43.9 | 39.6 | 42.5 | 46-50 |
| 512, as found | 461.2 | 390.6 | 450.7 | |
| 512, no retraction | 290.1 | 207.6 | 287.9 | 263-335 |

A loss of 35-38% at 64 and 36-47% at 512. Without the retraction the found rule holds what hand
delta holds on the same code, so the retraction IS the found rule's edge: teach with a large step,
then, one tick later and with no teacher, take part of it back in proportion to what the step
produced. Caveat: this removes a part the rest of the genome was tuned around; a genome searched
without it might recover some of the loss.

## 2026-11-11 (later) — tissue rung 1a: a hop through the brain's own wires, found by one seed of three

**The mechanism added** (`RUNG1` in `tissue/kernel.py`): one block of plastic wires among the
interneurons, from zero. An interneuron reads two dendrites (innate from the senses, plastic from
the other interneurons) and one global signal, whether the senses are on. The wire rule is the
same product. Genome 101 numbers. I had planned to feed the motor answer back to the sensory cells
over hand-drawn one-to-one wires and dropped that before building: wires among interneurons are
the tissue's own recurrence, need no teacher, grow as M^2, and are what rung 2 will have to shape.

**Exam 1** (`tissue/chain.py`). Items of 12 bits. A link a->b is a 3-tick episode (a; b; silence),
cells silenced between episodes. The teacher teaches the motor cells only the ECHO of what was
just seen, so what follows what can sit only in the wires among interneurons. Test: the first item
for one tick, then silence; hop h is read at tick h + 2. Search on single links, sizes 8-64, 1/8 to
1 link per interneuron, 6000 steps (11 minutes a seed).

**K90 at hop 1, single links** (* = never seen by the search):

| M | seed 0 | seeds 1, 2 | hand delta hop | least squares |
|---|---|---|---|---|
| 8 | 3.7 | 0 | 4.0 | 7.3 |
| 16 | 8.7 | 0 | 7.7 | 19.8 |
| 32 | 20.6 | 0 | 12.5 | 48.7 |
| 64 | 47.5 | 0 | 24.8 | 113.6 |
| 128* | 83.6 | 0 | 47.7 | 270.8 |
| 256* | 115.2 | 0 | 98.5 | 598.1 |
| 512* | 134.3 | 0 | 183.2 | 1337.1 |

- **Two seeds of three found nothing.** All three reach accuracy 0.64 within 500 steps; seed 0
  leaves at step ~1200, the others never. The flat spot is a real answer of a degenerate kind:
  the interneuron's code no longer depends on the cue (its activity is -0.35 .. -0.6 whatever the
  senses say), so every walk recalls the same blend, the bitwise majority of everything that ever
  came second. Measured 0.74 / 0.69 / 0.63 / 0.59 at 2 / 4 / 8 / 16 links; a majority vote gives
  0.75 / 0.69 / 0.64 / 0.60. It pays most in the smallest lives and needs nothing to cooperate,
  while a real hop needs the code, the wire rule and the decoder to fit each other.
- **C1 met** on seed 0: with the wires among interneurons frozen, hop 1 is 0.472 .. 0.506 at 16
  links and more, every size.
- **C2 (growth) met by the letter on seed 0, missed as written** (the bar said every seed). It
  rises at every doubling and K90(512) / K90(64) = 2.8, but links per interneuron go 0.74 at 64
  -> 0.65 -> 0.45 -> 0.26 at 512. Growth pays to twice the largest size seen and then fades.
- **C3 met on seed 0** at 16-256 (1.1-1.9x the hand hop); at 512 it is BELOW the hand hop.
- My predictions: 0.4-0.8 per interneuron at 8-64, right (0.47-0.74). K90(512) / K90(64) between 4
  and 6, wrong (2.8). A code less like a hash, right: two cues one bit apart correlate 0.45
  (rung 0: 0.28-0.37).

**Inside seed 0** (`python -m tissue.inside`). The code is a sawtooth with one sharp jump at zero
(input gain 10). With the senses silent an interneuron passes its other dendrite through a
saturating curve. And the trick: with the senses on, what it teaches its wires is MINUS what the
senses make it say, and the motor cell stores the echo with the sign turned over as well. Two
inversions cancel for one hop. Nothing scored the echo, so nothing objected.

**So along a chain the sign alternates.** The same genome, never shown a chain, on chains of 4,
bit accuracy by hop at M/4 links:

| M | hop 1 | hop 2 | hop 3 | hop 4 |
|---|---|---|---|---|
| 16 | 0.992 | 0.083 | 0.948 | 0.057 |
| 64 | 0.995 | 0.087 | 0.888 | 0.152 |
| 128* | 0.993 | 0.103 | 0.857 | 0.193 |
| 512* | 0.971 | 0.156 | 0.777 | 0.311 |

Read as distance from 0.5 it walks four links it was never asked to walk: 0.92, 0.91 (inverted),
0.95, 0.94 (inverted) at 16 cells; 0.97, 0.84, 0.78, 0.69 at 512. My prediction (hop 2 about 0.85)
was wrong as written and right about the knowledge being there. The links are composable; the
sign convention is not.

**The decoder is the bottleneck at large sizes.** In chains of 4, K links are 1.25 K items to
decode instead of 2 K, and K90 at hop 1 rises to 54 / 104 / 177 / 259 at 64 / 128 / 256 / 512
(single links: 48 / 84 / 115 / 134).

**What this says about the method.** The exam taught one thing (the echo) and scored another (the
hop), and both failures live in that gap: a code that cannot echo, and a decoder that echoes
upside down. Next, one change to the exam and none to the brain: the echo is scored too. Bars in
the design note. 13 tests.
