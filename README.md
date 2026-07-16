# nomos-engine

Artificial physics playground. An LLM (or a human) writes **laws of nature**
in a small DSL; a toroidal grid world of multi-channel cells lives by those
laws; an **observer** reports what emerged (metrics + GIF). The long-term goal
is a scientific loop: hypothesis → law → simulation → observation → new
hypothesis, with the LLM keeping a lab journal — not optimizing a score, but
building a theory of its own physics.

*Nomos* (νόμος) — greek for "law".

## Quick start

```powershell
.venv\Scripts\python.exe -m pytest tests -q          # sanity: gliders must glide
.venv\Scripts\python.exe -m nomos run laws/game_of_life.nomos --steps 300 --gif out/gol.gif --cmap gray
.venv\Scripts\python.exe -m nomos run laws/soup.nomos --steps 600 --channels energy,alive --gif out/soup.gif
```

Run `python -m nomos run --help` for all flags (`--size --steps --seed
--channels --every --scale --fps --cmap --gif --png --json`).

A run is a pure function of `(law file, size, seed, steps)` — bit-for-bit
reproducible. This is the foundation for causal experiments ("change one law,
keep everything else fixed") and for trivially parallel execution later
(k8s Jobs).

## The law DSL — reference

This section is the canonical spec (the LLM physicist will be prompted with it).

A `.nomos` file declares **channels** and **rules**. Comments start with `#`.
Newlines and `:` are decorative; the syntax is keyword-driven.

```
channel <name> [: init <expr>]     # per-cell float64 field, default init 0

rule <name>:
    when <expr>                    # condition, nonzero = true
    set <channel> = <expr>         # overwrite where condition holds
    add <channel> = <expr>         # accumulate where condition holds
```

A rule may have several `set`/`add` actions; they share the rule's condition.

### Semantics (the physics contract)

- **Synchronous update**: every expression in a step reads the *snapshot* of
  the world at the start of the step, like classic cellular automata.
- Rules apply in declaration order. For one channel: the last matching `set`
  wins per cell; `add` actions accumulate on top, so several `add` rules on
  one channel compose within a step. A `set` after an `add` overwrites it.
- The grid is a **torus** (edges wrap).
- After each step channels are regularized: NaN → 0, values clipped to
  ±1e6. Division by zero yields 0. A universe may die; it may not crash.
- `init` expressions may reference previously declared channels.

### Expressions

Operators by precedence (loose → tight):
`or` · `and` · `not` · `== != < <= > >=` · `+ -` · `* / %` · unary `-`.
Comparisons and boolean ops return 0/1 fields.

Built-in functions:

| Group | Functions |
|-------|-----------|
| Neighborhood (center excluded, toroidal) | `sum8 avg8 max8 min8` (Moore), `sum4 avg4 max4 min4` (von Neumann) |
| Translation | `shift(f, dx, dy)` — the field moved by (dx, dy); cell sees `f(x-dx, y-dy)` |
| Math | `abs sign floor round sqrt exp log`, `min(a,b) max(a,b) clamp(x,lo,hi) where(cond,a,b)` |
| Randomness (seeded, reproducible) | `rand()` U[0,1), `bernoulli(p)`, `uniform(a,b)`, `normal(mu,sigma)` |

`sqrt/exp/log` are safe (clamped domains). Unknown channels/functions are
rejected at load time with a line number, so a malformed law fails loudly
before the universe starts.

### Example: Game of Life

```
channel alive: init bernoulli(0.2)

rule birth:
    when alive == 0 and sum8(alive) == 3
    set alive = 1

rule death:
    when alive == 1 and (sum8(alive) < 2 or sum8(alive) > 3)
    set alive = 0
```

See `laws/soup.nomos` for a multi-channel physics (energy diffusion +
metabolism + crowding), and `laws/evolution.nomos` for heredity + mutation +
selection producing adaptation from a monoclonal ancestor. The reasoning behind
every law change is recorded in `docs/lab-journal.md` — the scientific loop this
project is about.

### Example: evolution (the Darwinian threshold)

`argmax8(key, value)` returns the gene of the neighbour with the largest key,
which is how a cell inherits from one parent:

```
channel food:  init posx()        # fixed left->right environment gradient
channel g:     init 0.5           # heritable gene, monoclonal ancestor
channel alive: init bernoulli(0.3)

rule selection:                   # die in proportion to environment mismatch
    when alive == 1 and bernoulli(clamp(1.2 * abs(food - g), 0, 0.6))
    set alive = 0

rule birth:                       # inherit a random living neighbour's gene, mutate
    when alive == 0 and sum8(alive) >= 2
    set alive = 1
    set g = clamp(argmax8(rand() * alive, g) + normal(0, 0.03), 0, 1)
```

With mutation the gene field grows to match `food` (adaptation); set mutation to
0 and it can't — see the journal entry for the full result.

## Layout

```
nomos/dsl.py      tokenizer + Pratt parser -> tuple AST
nomos/engine.py   World: snapshot-semantics step(), builtins, regularization
nomos/metrics.py  the observer: activity, entropy, gzip ratio, verdict
nomos/render.py   frames -> GIF/PNG (1 channel -> colormap, 2-3 -> RGB)
nomos/__main__.py CLI (`python -m nomos run ...`)
laws/             .nomos law files
tests/            pytest; GoL glider/blinker as engine ground truth
```

## Roadmap

1. ~~Engine + DSL + GoL sanity check + GIF rendering~~ (done)
2. ~~Evolution: heredity + mutation + selection → adaptation~~ (done;
   `argmax8` inheritance primitive, see lab journal Experiment 3)
3. Observer v2: structure detection (connected components lifetime, FFT
   periodicity, mutual information t↔t+k, gene↔environment correlation so
   adaptation is *noticed* unprompted), run archive in SQLite
4. The scientist loop, manual first: Claude-in-session as physicist keeping the
   lab journal (this is already how Experiments 1–3 were run)
5. Automated loop via API (provider-agnostic: DeepSeek / Claude), batches,
   novelty archive (MAP-Elites) instead of single-objective optimization
