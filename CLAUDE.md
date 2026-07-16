# CLAUDE.md — nomos-engine

Research playground for **LLM-generated artificial physics**. An LLM (or a
human) writes *laws of nature* in a small DSL; a grid world of multi-channel
cells lives by them; an observer reports what emerged. The real goal is a
**scientific loop** — hypothesis → law → simulation → observation → new
hypothesis — with the model keeping a lab journal, building a *theory* of its
own physics, not optimizing a score.

Start from `README.md` (the DSL spec + layout) and `docs/lab-journal.md` (what
we've actually tried and learned). This file is operational memory for working
sessions.

## Working rules

- **Language**: chat in Russian; all code, comments, docs, commits in English.
- **Venv**: always `.venv` in the project root. Never install into global
  Python. Run things as `.\.venv\Scripts\python.exe -m ...`.
- **Shell**: Windows PowerShell. `2>$null` to drop numpy warnings from output.
- **This is a fast research playground** — favour small playable iterations
  and visible results (GIF/PNG) over heavy architecture. No CI yet.
- **Reproducibility is load-bearing**: a run is a pure function of
  `(law file, size, seed, steps)`. Keep it that way — it's what makes causal
  experiments ("change one law, hold everything else") and parallel execution
  possible.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q                    # sanity: gliders must glide
.\.venv\Scripts\python.exe -m nomos run laws/<law>.nomos --help  # all run flags
```

Rendering: one `--channels` -> colormap, two or three -> RGB. `--every N`
samples frames; `--gif/--png/--json` write artifacts to `out/` (gitignored).

## Architecture (one screen)

```
nomos/dsl.py      tokenizer + Pratt parser -> tuple AST; static validation
nomos/engine.py   World.step(): SNAPSHOT semantics (synchronous CA), builtins,
                  regularization (NaN->0, clip 1e6). Torus grid, float64 channels.
nomos/metrics.py  the observer: activity, entropy, gzip ratio, verdict line
nomos/render.py   frames -> GIF/PNG
nomos/archive.py  SQLite archive (laws + reports + hypotheses); memory() feeds
                  the scientist's first turn
nomos/elites.py   MAP-Elites behavior map: activity x structure x order,
                  100 regime cells computed from stored reports on the fly.
                  structure axis = spatial coherence (metrics.spatial_autocorr,
                  neighbour autocorrelation), NOT gzip — it separates coherent
                  waves/blobs from noise; legacy runs fall back to gzip
nomos/game.py     deterministic toroidal Snake (the external world)
nomos/embodied.py brain<->game coupling: food sensor -> world, motor field ->
                  move; `play()` loop; CLI `python -m nomos play`
nomos/__main__.py CLI `python -m nomos run`
laws/*.nomos      law files (the "physics")
tests/            pytest; GoL glider/blinker = engine ground truth
docs/lab-journal.md  chronological record of experiments + conclusions
```

Semantics that bite if forgotten: all rule expressions in a step read the
world snapshot from the *start* of the step; rules apply in declaration order;
per channel the last matching `set` wins, `add`s accumulate. Full contract in
README.

## Key builtins for evolution

`argmax8(key, value)` = value of the neighbour with the largest key — the
**inheritance** primitive. `argmax8(rand()*alive, g)` copies the gene of a
random living neighbour; `argmax8(fitness*alive, g)` copies the fittest. For
*linked* inheritance of several genes from one parent, use a shared random
parent-key channel: `argmax8(pk*alive, geneA)` and `argmax8(pk*alive, geneB)`
pick the same neighbour (see `laws/evolvability_moving.nomos`).
`posx()/posy()` = normalized coordinates; `time()` = step number; `sin/cos/pi()`
— together these let a law drive a changing environment.

## State of play

- Done: engine + DSL + GoL sanity check + rendering.
- Done (Exp 3): **evolution** — heredity + variation + selection → adaptation
  (`laws/evolution.nomos`).
- Done (Observer v2): channel-pair correlations, axis-gradient & connected
  components per channel, activity periodicity (FFT), temporal dynamics, SQLite
  archive (`--archive runs.db`, `nomos/archive.py`). The observer now *notices*
  adaptation unprompted.
- Done (Exp 4): **evolvability evolves** — mutation rate as a heritable gene
  rises ~5× in a moving environment, stays flat/declines in a static one
  (`laws/evolvability_{moving,static}.nomos`). Uses linked inheritance via `pk`.
- Done (automated loop): `nomos/scientist.py` — an LLM physicist proposes a
  hypothesis + law + prediction, the engine runs it, the observer reports, the
  model concludes and proposes the next; logged to `journal/` + SQLite archive.
  `nomos/llm.py` provider-agnostic (WaveSpeed unified OpenAI-compatible API,
  default `deepseek/deepseek-v4-flash`); `nomos/sim.py` = `simulate`+`digest`.
  Run: `.\.venv\Scripts\python.exe -m nomos.scientist --iterations N`
  (`--dry-run` uses a mock physicist, no API). Its first live run already
  surfaced and fixed an observer bug (activity is now magnitude-based, not
  fraction-of-cells-changed). Key config lives in `.env` (gitignored):
  `WAVESPEED_API_KEY` reused from the ava project; `NOMOS_LLM_BASE_URL`,
  `NOMOS_LLM_MODEL`. Note: WaveSpeed's LLM endpoint needs an LLM-enabled key
  (the image-gen key 401s). deepseek-v4 defaults to thinking mode — don't set a
  tiny max_tokens or `content` comes back empty.
- Done (memory + behavior map): the scientist's first turn now carries the
  archive — past laws deduped by hash, with hypotheses (extracted from the
  HYPOTHESIS: line and stored in a new `hypothesis` column, auto-migrated) and
  outcomes (`--memory N`, default 10, 0 = off). Every run is binned into a
  MAP-Elites behavior map (`nomos/elites.py`): activity(5, same thresholds as
  the verdict) x spatial-coherence structure(5) x temporal-order recurrence(4)
  = 100 cells; the follow-up message tells the model its cell and NEW/visited,
  the memory block includes coverage. Cells are computed from stored reports, so
  binning can change without touching the DB. The structure axis was originally
  mean-gzip but that couldn't tell a spiral-wave medium from noise (both
  compress poorly); it's now `metrics.spatial_autocorr` (neighbour correlation:
  ~1 coherent, ~0 noise, <0 checkerboard). Scientist auto-renders one GIF per
  experiment (`--gif-dir`, `--gif-fps` default 6) with a bottom progress strip.
- Done (illumination): `--mode illuminate` actively steers. Each turn it picks
  an empty behavior-map cell on the frontier (adjacent to an occupied one via
  `elites.frontier`/`choose_target`, deterministic under a seeded rng, prefers
  well-supported least-attempted targets), hands the model the neighbour's law
  as a SEED plus which single axis to move and a lever hint for that axis
  (`target_prompt` + `_LEVER_HINTS`), and reports HIT/miss. A miss just lands in
  another neighbour — still illumination. Basin escape: two identical landings
  in a row -> the next prompt asks for a STRUCTURAL change (add/remove a channel
  or rule) instead of a coefficient tweak. `--mode explore` (default) keeps the
  old free hypothesis chain. Live runs grew coverage 9 -> 19 over the whole
  memory+coherence+illumination arc.
- The closed-world scientific cycle is considered COMPLETE (memory, coherence-
  honest map, illumination + basin escape, watchable GIFs).
- Done (game Phase A — brain<->game coupling): `nomos/game.py` (deterministic
  toroidal Snake) + `nomos/embodied.py` couple the grid to the game. The world
  senses via a read-only `food` channel (Gaussian bump egocentric to the head,
  re-asserted each physics step) and acts via a `motor` channel (harness reads
  the motor field's centre of mass -> a move). Game ticks once per `--ticks`
  physics steps. `python -m nomos play laws/snake_brain.nomos --gif ...` renders
  the SPLIT view (life left, snake right in the fire palette, progress strip).
  Baseline `laws/snake_brain.nomos` (diffuse food into motor) already scores ~22
  at 48/16, seed 0 — the sense->act loop demonstrably drives the snake to food.
  Reproducible: `play()` is a pure function of (law, sizes, seed, cadence, moves).
  Phase B (NOT started) = learning: selection with fitness = game score over the
  argmax8 inheritance primitives.
- The manual physicist (Claude-in-session) still runs the deeper experiments;
  every one is written up in `docs/lab-journal.md` (keep doing this).
- Next: a smarter model setting direction over DeepSeek's cheap mutations;
  environment-coupled worlds. Preferred framing (user's, 2026-07-16): the cell
  grid is a *brain* that self-organizes and drives an external world through
  emitted SIGNALS — e.g. cell clustering in a region raises a "move" signal read
  back as a game action (Snake direction) — rather than the world merely feeding
  read-only channels in. Data flows both ways via physical channels
  (temperature/energy); game ticks can be rare relative to physics ticks. Boxes-
  you-can-push style direct interaction was considered and set aside — cells are
  too simple to manipulate objects; signalling/organizing is the right level.
  Render plan (user's): a SPLIT GIF — life field on the left, the game (snake) on
  the right styled to match the life field, progress strip below; aggregate cells
  for the life render once populations grow. "Learning to play" is a step beyond
  mere coupling: needs selection (fitness = game score) over the argmax8
  inheritance primitives (Exp 3-4 evolution with an external reward). Phase A =
  build the world<->game coupling loop; phase B = add that selection.
- Deployment intent: runs are pure CLI -> trivially parallel as k8s Jobs on the
  user's cluster later; GPU only if we move to Lenia-style continuous physics.
