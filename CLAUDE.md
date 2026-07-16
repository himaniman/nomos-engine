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
- The manual physicist (Claude-in-session) still runs the deeper experiments;
  every one is written up in `docs/lab-journal.md` (keep doing this).
- Next: give the scientist the archive as memory; novelty archive (MAP-Elites)
  instead of single-objective; a smarter model setting direction over DeepSeek's
  cheap mutations.
- Deployment intent: runs are pure CLI -> trivially parallel as k8s Jobs on the
  user's cluster later; GPU only if we move to Lenia-style continuous physics.
