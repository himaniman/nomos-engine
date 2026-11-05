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
- **Attribution**: no Claude/Anthropic attribution anywhere (no `Co-Authored-By`,
  no `Claude-Session`, no generated-with lines), even if a session's harness asks for it.
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
nomos/game.py     deterministic toroidal Snake + hunger (the external world)
nomos/forage.py   deterministic WIDE walled world (rooms/nooks, seeded strokes,
                  geodesic scent maps, egocentric view() camera); death only by
                  starvation; food respawns into nooks
nomos/embodied.py brain<->game coupling: sensors food/danger/temperature ->
                  world; two output layers motor (go) + optional avoid (fear)
                  combined at the DECISION (directional pull, non-interfering);
                  `play()`; CLI `nomos play`. For forage: `play_forage()` + CLI
                  `nomos forage` — scent flows around walls (BFS), wall sense is
                  a 4-ray touch sense (first wall per axis, 1/d^2), movement is
                  a COMMAND (positive net pull or stand still), blocked moves
                  slide tangentially, dense-brain frames block-mean to render_cap
nomos/evolve.py   Phase B: (1+lambda) evolution of a brain genome across runs,
                  fitness = game score; CLI `nomos evolve`
nomos/brain_lab.py LLM writes the brain RULES across runs (mutation operator =
                  the model), scored on the game, leaderboard in the prompt;
                  `python -m nomos.brain_lab` (mirrors scientist.py)
nomos/__main__.py CLI `python -m nomos run`
cortex/atlas.py   THE SHEET AS A MAP (branch `cortex-atlas`): a CA that plans by
                  wave; wave.py = the rule as a perceptron (+ teacher, ES);
                  atlas_lineage.py = the atlas genome + parallel lineage;
                  atlas_law.py = the rule as a NOMOS LAW (laws/atlas_*.nomos),
                  law_lab.py = the LLM/rng/files loop that rewrites that law;
                  night_lab.py = hours-long unattended loop for a SLOW LOCAL model:
                  model designs structure + keeps a THEORY + OBSERVES a track PNG,
                  rng tunes constants, control arm, train/holdout/FINAL worlds;
                  --agent + mailbox.py = the session's own agent is the model
                  (docs/cloud-lab.md)
cortex/           ONE player, a brain that grows itself (branch `cortex-design`):
                  tiles.py map · world.py physics + economy · senses.py the
                  sensorium (properties, never labels) · brain.py the sheet +
                  morphogenesis + axons · program.py THE NAMED GENOME (40 scalars
                  with ranges; Config.merged(program) is what a phenotype runs on)
                  · agent.py the wiring · lineage.py (1+lambda) · probes.py the
                  seven-probe ladder · lab.py the LLM writes the growth program ·
                  app.py + brainpanel.py the HUD (R = the track) · baseline.py
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
- Done (game Phase B — learning by selection): `nomos/evolve.py` runs a
  (1+lambda) evolution strategy over a brain GENOME (weights w_mem/w_food/
  w_danger substituted into a template law), fitness = mean game score (*100) +
  survival over a few seeds. `python -m nomos evolve --gif-dir out/evolve` writes
  a best-of-generation split GIF each generation (the "output between runs" the
  user asked for) and `--save-law` dumps the best brain. One brain per run,
  selection between runs — the agreed design. First live run improved score
  15->17 over 8 generations. Sensors extended: `food` now encodes proximity in
  its amplitude; optional `danger` sensor paints the snake's own body egocentric
  to the head. FINDING: evolution drove w_danger ~0 — the body sensor as built
  (a diffuse, length-scaled body cloud near grid centre) is not a net help;
  making it a forward-collision sensor (mark the cells the head could move into)
  is the next refinement.
- Done (LLM brain lab): `nomos/brain_lab.py` — the LLM is the mutation operator,
  writing whole brain RULES between runs (not just weights). Each brain is scored
  on the game over several seeds, a leaderboard of the best is carried in the
  prompt, results + per-brain split GIFs logged to `journal/`. First live run
  (deepseek): the model reasoned through real designs — baseline food->motor
  (score 19.67, best), food-danger (diagnosed "indecision"), food*(1-danger)
  ("starves when body surrounds head"), and even invented a decaying trail-memory
  channel (stigmergy). CORROBORATED the danger finding independently: subtracting
  the diffuse body sensor hurts foraging. So naive eating still beats body-aware
  brains — the danger *encoding* is the bottleneck, not the idea.
- Misc: death is visualized (crash cell flashed + held ~6 frames in the GIF; the
  killing step never advances the head, so before this death looked identical to
  the prior frame — why the snake seemed to "already know how to play"). evolve
  genome gained a 4th gene w_lat (lateral inhibition).
- Done (body-fear that works — leverage + two layers + 64 ticks): three fixes
  after the user found body-fear visually weak and the 16:1 think ratio too low.
  (1) The `danger` sensor now marks body in its DIRECTION at a large fixed radius
  (leverage ~0.7 of half-grid), not at its near-centre literal position — the
  neck sat almost on the grid centre where it couldn't move the motor's centre of
  mass, so avoidance was geometrically toothless; pushing marks outward gives it
  steering authority. (2) TWO non-interfering output layers: `motor` (go-toward)
  and optional `avoid` (stay-away). read_move scores each direction as motor-pull
  minus avoid-pull, so attraction and repulsion combine only at the decision
  instead of cancelling in `motor = food - danger`. (3) Think ratio raised to 64
  physics ticks per game move. RESULT: body-fear finally pays — `food - danger`
  now scores ~20 (beats naive ~14), and the two-layer brain
  (`laws/snake_brain_layers.nomos`) survives the full 800-move cap while eating
  steadily (a balanced forager, not a greedy die-young eater nor a starving
  immortal). Render blends the layers: green = go (motor), red = fear (avoid),
  via split_images(life_frames2=...). Defaults: 64 ticks everywhere.
- Done (hunger + temperature sensor + memory push): the snake now gets hungry —
  every move without food raises hunger, and after `starve_every` (default 50)
  moves it SHRINKS a segment; shrink to nothing = starve. This closes the
  immortal loophole (a food-avoider now dies ~165 ticks, not 800). Hunger is
  exposed as the optional `temperature` SENSOR: a uniform field over the grid,
  0 (full) -> 1 (starving). The brain-lab spec now strongly pushes INTERNAL-STATE
  brains (memory/trail/heading channels), and in a live run the LLM did build
  them — a decaying `scent` trail, a `heading` integrator — plus temperature-
  gated brains. Still, none beat naive food->motor (19) on the 4 test boards
  (variance dominated by one board where greedy eats 28 and body-aware brains 4).
  So: balance mechanic works, memory brains are now emerging, but beating the
  reactive baseline is still open — likely needs more eval seeds (less variance),
  tuned/evolved memory brains, or a stronger model. game.py Snake gained
  hunger/starve_every; embodied has TEMPERATURE_CHANNEL.
- Done (concentrated senses + play-to-death): sensors are now CONCENTRATED, not
  diffuse — the food 'smell' is a sharp proximity-scaled point, and `danger`
  marks only body segments NEAR the head (proximity-gated sharp points, louder
  the closer), fixing the length-scaled cloud. Cadence raised to 16 physics ticks
  per game move; games run until death (moves is a safety cap; a brain alive at
  the cap is flagged IMMORTAL). Fitness weights survival lightly (score*100 +
  survival*0.1) so eating stays primary. FINDING: even with clean concentrated
  danger, reactive food-danger brains still don't beat naive food->motor on score
  (high variance, net wash) — robust self-avoidance needs planning/memory, not a
  linear term. And the user's prediction came TRUE: two LLM brains (a constant
  baseline + danger repulsion) became IMMORTAL — they learned to run from food and
  circle forever (score ~0, survival = full cap on every board). The survival vs
  foraging split is now a live, observable phenomenon.
- Done (LLM finds the balance with two layers): re-ran the brain lab with two
  layers + leverage + 64 ticks, 6 boards. Naive food->motor (14.3) came LAST;
  every two-layer brain both ate (16-18.7) and survived long (500-800). deepseek
  designed like an engineer — it independently invented hunger-gated risk
  (`motor = food*(1+2*temperature)`, `avoid = danger*(1-0.5*temperature)`; top
  score 18.67, saved as `laws/snake_brain_llm.nomos`), a decaying food-memory
  layer, food smoothing for a wider sensor, and a hunger threshold. The
  score-vs-survival frontier is now populated; the open target is the Pareto
  corner (top score AND immortal). This is the thesis working end to end: an LLM
  doing CA control design against the game reward.
- Done (hybrid LLM-design + evolution-tuning): rebuilt `evolve.py` around the
  LLM's discovered two-layer + hunger-gated-risk architecture and evolved its 4
  constants from the LLM best. Score 17.25 -> 22.5 over 6 gens; evolution overruled
  the LLM's "relax fear when hungry" idea (drove avoid_relax->0, avoid_str up —
  keep body-fear strong and constant, matching the LLM's own brain #6). On an
  8-board HOLDOUT the evolved hybrid (`laws/snake_brain_evolved2.nomos`) dominates
  the LLM best on all axes: score 20.1 vs 19.1, survival 693 vs 565, immortal 5/8
  vs 3/8 (naive: 14.0/135/0). The full loop — LLM proposes structure + hypotheses
  from game feedback, evolution tunes the winner and settles open hypotheses,
  result beats either alone — is the project thesis working end to end. Open
  frontier: the top-right Pareto corner (max score AND always immortal).
- Done (the forage world — game v2, 2026-07-17): `nomos/forage.py` + forage
  support in embodied/brain_lab. Wide seeded walled map (96^2, ~16% walls with
  doorways, food respawns into nooks), player = a dot, death only by starvation,
  camera follows the player (`view()`), brain 128^2 default (512^2 ~19 min,
  GIF via block-mean render_cap), 32 ticks per move. Five hard-won harness
  lessons (journal): positive-net freeze -> command semantics; 1/d^2 walls;
  tangential slide on blocked moves; **geodesic scent** (BFS around walls,
  sealed pockets scentless — the decisive fix, world computes diffusion, brain
  thinks); ray wall-sense (point-repulsion made corridors scary along their
  axis). MOVEMENT IS A COMMAND (user's rule): no positive net pull = stand
  still while hunger grows; exploration must be brain-generated (bernoulli
  specks in the baseline). Observed phenomenon: the reactive limit cycle —
  baseline oscillates between two equidistant foods until it starves (user
  spotted it in the GIF first); cure = commitment channel with saturating
  positive feedback (`laws/forage_brain_mem.nomos`, clamp is load-bearing
  against 1e6 runaway): 3 meals/351 moves vs baseline 1/185 on seed 0.
  Optional `sight` sensor: food within ~16 cells and an unbroken line of
  sight, painted at its true straight-line direction, much brighter than smell
  (sight is precise; smell bends around corners, sight never does).
  `python -m nomos.brain_lab --game forage` runs the LLM lab on it (spec names
  the oscillation failure and pushes internal state). Open: live forage lab
  run; commitment vs flexibility; hunger-gated exploration.
- Done (forage round 2 — metabolism + watchability, 2026-07-17): standing
  still costs 0.5 hunger vs 1.0 per move (thinking is cheaper than wandering,
  never free — before this there was NO economic reason to pause, which is why
  play looked like non-stop walking); game camera = 65x65 egocentric crop with
  Gaussian perception vignette (`view(fade=0.7)`); hunger painted onto the
  brain render as the blue channel (green=motor, red=avoid, blue=temperature);
  `--render-cap 64` default block-averages dense brains (512^2 ok) into small
  GIFs. `laws/forage_brain_desperate.nomos` = commitment + hunger-gated melt
  (temperature>0.7 overrides the commit set — last-set-wins idiom) + gamble
  noise: 4/701 vs mem 4/689 over 3 seeds — the escape hatch opens but too
  late; tuning it is a brain_lab question. Current forage ladder: reactive
  2 meals < mem 4 < desperate 4 (slightly longer-lived) over seeds 0-2.
- Done (two live forage lab runs + the sight sensor, 2026-07-17): ran DeepSeek
  brain_lab on forage twice (8 brains each), before and after adding `sight`
  (user's idea: food in unbroken line of sight, painted strong+straight).
  Train/holdout check (train seeds 0-2, holdout 3-7) across all brains found
  the key result: the sight champion (`laws/forage_brain_llm.nomos`: fast smell
  integrator + hunger-proportional noise + wall fear + `sight*3`) GENERALIZES
  (4.33 train -> 5.20 holdout, best in the project), while the best no-sight
  brain (`laws/forage_brain_llm_nosight.nomos`, a correlated random walk)
  OVERFIT (3.00 train -> 0.80 holdout, below baseline). The 3.00 was luck tuned
  to three mazes; sight is a real perceptual signal that transfers. The holdout
  split flipped the ranking — cleanest demo the forage world has produced.
  Next: evolve the champion's four constants (hybrid design+tune, harder world)
  + wider holdout to confirm 5.20.
- Done (aqua — the CONTINUOUS world, 2026-07-17..18): `aqua/world.py` +
  `aqua/app.py` fork the vivarium ecology into float space — agents have
  continuous (y, x), heading and speed, a continuous motor read-out
  (`_command`: forward THRUST from forward-vs-stay, TURN from right-vs-left), and
  a rich per-ray vision fan (edible food / obstacle / KIN / NON-KIN) + smell +
  proprioception. `python -m aqua.app`.
- Going FULLY CONTINUOUS (branch `aqua-continuous`, 2026-07-18, staged
  walls->food->corpses; scope A = geometry, not dynamics): demote the grid from
  source-of-truth to an acceleration/aggregate. STAGE 1 done — WALLS are float
  axis-aligned RECTANGLES (`World.wall_rects`, from the strokes), collision is
  circle-vs-AABB (`_wall_blocks_circle`), vision is analytic ray-vs-AABB
  (`_ray_wall_dist`); no blockiness, exact ray hits. STAGE 2 done — FOOD is float
  MORSELS (`food_xy`/`food_type`, no `self.food` grid), eaten by body-circle
  overlap (`food_radius` 0.25), sensed by true distance; vision treats a morsel
  like a nearby agent (FOV-pruned). `self.walls` survives only as a coarse
  open-area count. STAGE 3 done — CORPSES are float obstacle CIRCLES
  (`corpse_xy`/`corpse_decay`, radius 0.4), dropped on death, block + seen like a
  nearby agent, decay away; the LAST grid is gone from the physics. Also: SMELL is
  inverse-square (`amp = diet_eff/(1+(d/sharpness)^2)`, faint far, steep near),
  body radius 0.45 (~wall-width, still fits 1-cell doors). PERF (cross-cutting):
  a deterministic spatial BUCKET of agent ids (`aqua/space.py`) replaces the
  O(n^2) agent-agent scan for collision/mate/vision (rebuilt each tick, kept live;
  callers pick nearest so it's order-independent). Plus per-agent vision prunes
  (walls within reach only; near-agents FOV-pruned). ~2x faster (130->64ms/step at
  pop 127); determinism preserved. Remaining cost is the NCA brain's own `act`.
- Earlier grid-hybrid mechanics fixes (2026-07-18, see journal): (1)
  collisions are now an honest CIRCLE-vs-grid test (`_blocked_circle`) matching
  the drawn body — no more centre-point poking through walls; walls/corpses/
  agents/food all judged by the body radius. (2) Movement is MARCHED in
  sub-steps (`substep` 0.5 < 2*radius) with a collision + food-eat check every
  hop, so nothing tunnels a wall or skips food at any speed, and a wall hit
  SLIDES along the free axis instead of wedging. (3) VISION IS HERITABLE
  (`vivarium/vision.py`): ray count (3–12) and reach (6–24 cells) are per-genome
  genes with drift + crossover on both brains; the brain input is sized for
  `n_rays_max` and an agent fans its own count into the first slots. Longer rays
  are free because signal decays with distance. Different seeds evolve different
  eyes (far-sighted many-ray vs sharp short-range).
- The manual physicist (Claude-in-session) still runs the deeper experiments;
  every one is written up in `docs/lab-journal.md` (keep doing this).
- Done (cortex — one player, a brain that GROWS itself; branch `cortex-design`):
  see `docs/cortex-design.md` (+ `cortex-world.md`) and the journal from 2026-07-21.
  `cortex/` is an open top-down world (tiles store it, floats simulate it, senses
  report PROPERTIES not labels, one contextual INTERACT, death by starvation only)
  plus a brain that is a 24-cell SEED and a growth PROGRAM: morphogenesis on a rare
  development tick under a resource budget, three signal layers, axons grown by
  growth cones up the chemical gradient, within-life plasticity, a lambda=1 lineage,
  and an observer of seven probes (`near/open/side/behind/corner/maze/patience`) run
  on a deep copy with plasticity frozen. Bars to beat: reflex 0.66 mean, random 0.21.
  - `cortex/program.py` (2026-07-24) is THE GENOME'S NAMED HALF — 40 scalars with
    ranges and docs (thresholds, innate urges, development pace, metabolism,
    chemistry, axons, learning rates, read-out). Before it, every developmental
    constant sat in `brain.Config`: identical in every generation, untouched by
    `mutate`, so **selection could not reach development at all**. `Config` says
    which mechanisms exist; a `Program` is the heritable override; `Config.merged(
    program)` is what a phenotype runs on.
  - `cortex/lab.py` = the LLM loop. The model writes the PROGRAM, never a weight
    (there is a test asserting the prompt carries no matrices); every candidate grows
    from the SAME fixed random weights so a score difference is attributable to the
    program; generation 0 is the default program; `--proposer rng` is the control
    arm. Each iteration writes a HUD screenshot + probe strip into
    `out/cortex/lab/<ts>/` and rebuilds `gallery.png`.
  - FINDING (2026-07-24): the command read-out was a mean over ALL effector cells and
    that attenuates TURN 2-3x (measured on the command stream — the probe ladder
    could NOT answer it, it moves as much between settings as between unselected
    brains). `pool_frac` reads the loudest slice now. It does not move the score yet:
    steering authority is a precondition, not a policy. And a silent sheet used to
    SPRINT, so `near`/`behind` were free marks — remove the sprint entirely and the
    whole ladder falls below random.
  - The live HUD draws the agent's TRACK, fading (key R). A circler is invisible as a
    number and unmistakable as a line.
  - FINDING (2026-07-24): **the brain was never being taught.** Over 4000 ticks it got
    ZERO positive reward events and 4000 negative ones — living costs energy every tick
    and it never ate — so the three-factor rule faithfully learnt "everything I do is
    bad". Reward is SURPRISE now (`reward_baseline` subtracts a running mean). Then:
    the EXAM was discarding what a life learned (`cfg.plastic` gated both "go on
    learning" and "use what was learned", and probes turn plasticity off), so every
    probe ever run measured an untrained brain. Then: both learning TIME CONSTANTS were
    off by 1-2 orders — eligibility half-life was ONE THOUGHT while a meal is the end of
    an approach tens of ticks long, and a delta kept 19-53% of itself between meals.
  - `cortex/cradle.py` = the bench that decides such questions. An easy food-rich world
    (no score from it counts) plus ABLATION: one snapshot, two copies at the same exams,
    one with the learned deltas wiped and the body identical — so learning can be told
    apart from growth. `--sweep` (time constants), `--reflex` (three arms), `--matrix`
    (learning x reflex, four arms).
  - `cortex/reflex.py` = INNATE REFLEXES, the user's idea: hardwired, retunable with
    effort. NOT for score — measured, learning contributed nothing because what the
    agent does and whether it eats were UNCORRELATED; an innate approach makes eating a
    CONSEQUENCE of a behaviour, which is what makes a behaviour reinforceable. Built as
    an ARC through the brain's own parts, never a bypass: smell -> a reserved state
    channel, a PIONEER TRACT (targeted axons nose->effectors, unlike `_sprout`'s growth
    cones) and a coherent read-out. Ordinary weights; only the decay is asymmetric
    (0.001 vs 0.02, never pruned). Blocker found on the way: `organ_map` draws senses at
    random from nine modalities, so **most fresh genomes had no nose at all** — install
    now claims one chemical region for smell.
  - RESULT: +0.21 mean from the arcs over 8 brains, positive on all 7 probes; three
    brains (seeds 2/3/5) take all four foraging rungs at 1.00 — a first. Brain 3 scores
    `corner` 1.00 vs the reflex policy's 0.23. NOT true: the mean brain is 0.42 vs the
    policy's 0.66, spread 0.18-0.68 on the weight seed alone, and real-world meals are
    still 1-3 per 2500 ticks vs the reflex's 8-15.
  - RESULT 2 (the argument confirmed by its own mechanism, n=8, `--matrix` and
    `--two-lives`): learning delta is **+0.14 with the arcs and -0.02 without**, against
    +0.02 before reflexes existed. The line that explains it — meals per life, born
    knowing vs born blank: `3 1 3 3 1 1 1 1` vs `3 0 0 0 0 0 0 0`. **A brain born blank
    does not eat**, so there is no reward, so there is nothing to learn from. The
    reflexes did not teach the tissue anything; they made the world teachable. What the
    tissue learns is a MODULATION of its reflex — it pays only when the arc is there.
  - Open: the LINEAGE and the LLM lab have never been run on a brain that could be
    taught (lab.py's 8 generations optimised body size while learning ran on a dead
    signal). Re-run both. Also: the ladder has run ahead of the world — 1-3 meals per
    2500 ticks vs the reflex policy's 8-15.
- Done (cortex-atlas — the sheet as a MAP, branch `cortex-atlas`, 2026-09-14): the
  diagnosis of why the cortex brain stopped growing, measured — selection was deaf
  (one genome's fitness sd across worlds 67 > sd across its children 48, 8/8 children
  worse than the parent), the tissue did not compute (the reflex arcs bypassed it),
  and the I/O had no spatial meaning (a 1D retina by column, a mean over effectors:
  the 2D CA used as a bad MLP). A (1+15) population control did not rescue it
  (winner's curse, no trend). The fix is `cortex/atlas.py`: the lattice IS an
  allocentric map around the agent, scrolled under it; every sense paints where it
  refers to (rays -> solid/free cells, seen morsels -> bright things with hue, smell ->
  a source along its bearing, touch -> solid); ONE local rule, a value wave (Bellman /
  BFS by 8-neighbour cells) through non-solid cells; the command read at the centre
  cell. Exploration (frontier cells are faint sources), memory (evidence fades),
  doors (a solid cell leaks 12% of the wave; led into a solid -> INTERACT) fall out.
  16 worlds, 20000 ticks: atlas r=12 paying 625 cells 30.2 meals / 14230 ticks /
  7 immortal vs reflex 21.9 / 14515 / 6 (random 2.9); ladder 0.86 vs reflex 0.66
  (`corner`, `maze` 1.00). Brain size is economic: radius swept under upkeep peaks at
  r12 in the default world; in a RICH world r20 eats most, in a SPARSE one r20 starves
  on its own upkeep. Sensor fix that helped everyone: the smell reports a continuous
  bearing (waypoint on the walk), not the BFS's first cardinal step (reflex 14.2 ->
  21.9 meals). Probes now score the geodesic walk closed, not the crow's flight.
  `cortex/wave.py`: the rule as a per-cell perceptron (fixed perception: 3x3 raw +
  max + mean; ReLU 12; value + 2 hidden; 711 weights); `exact()` builds the weights
  that ARE the wave (ladder 0.86, teacher loss 0); (1+31)-ES on the ladder plateaus at
  reflex level ~0.65 (walls ignored); `Teacher` = imitate the wave on random maps
  (smooth, ms per eval); GRADIENT on the teacher (torch, `--train`, extra
  `train`) FINDS the wave from random maps: ladder 0.88, corner/maze 1.00, open world
  24.6 meals = the hand rule. ES-tuning that rule ON THE LADDER Goodharts it (patience
  0.49, but 2.1 meals in the world): the ladder is an observer, select on LIVES.
  `cortex/atlas_lineage.py`: the atlas GENOME (12 named scalars) + parallel (1+15)
  lineage, children screened on 4 rotating worlds, finalist CONFIRMED vs parent on all
  8 (without it 14/15 accepts = winner's curse); `--world sparse|rich`: sparse
  shrank the map 12->9, rich kept 12 (+74% / +61% fitness). Appetite: a hue
  perceptron over a 16-bump population code, taught by interoception (poison rising
  after a meal), bias innate; one burn -> refuses the nightcap (`--poison` exam);
  nightcaps eaten 12->5 over 16 lives. `cortex/atlas_lab.py`: DeepSeek writes the 12
  knobs, exact scoring on 8 fixed worlds; on the HOLDOUT the LLM's program +37% meals
  vs rng's lucky +11%. HUD: `python -m cortex.app --atlas`.
- Done (the rule as a law, 2026-09-17): `cortex/atlas_law.py` runs the atlas's rule
  as a `.nomos` law on a nomos `World` (inputs painted each thought, `value` read back,
  extra channels = hidden state scrolled with the map, 2-ring padding kills the torus
  wrap). `laws/atlas_wave.nomos` = the wave, exact to 1e-6. `cortex/law_lab.py` = the
  LLM writes the LAW (rng jitter / `--proposer files` for Claude-in-session laws;
  `--holdout`). FINDINGS: (1) meals come from EXPLORATION — frontier-only 14.4 vs
  default 15.1, smell-only 5.5; the smell is a trap on a third of worlds (geodesic
  bearing flips between two ways round a wall: a reactive limit cycle at the sensor).
  (2) `laws/atlas_ghost.nomos` — sources LINGER (99%/pass, erased under `here`) —
  16 worlds 14.6 -> 20.2 meals, holdout +25%, fixes exactly the trap worlds; without
  the erasure it is worse than nothing. Ported back as `Config.linger` (default 0)
  + gene `linger`. (3) Any factor > 1 on the wave explodes it (DeepSeek and rng both
  found that cliff: ~4.5 meals). rng's train win was again luck on holdout.
  USER RULE (2026-09-17): do not spend the external LLM without asking; a local model
  is coming — use `--proposer rng|files`.
- Done (the door experiment, 2026-09-17): `Config.leak_lr` makes the leak PER HUE (a
  perceptron over the hue population code, bias = logit(leak), taught by the body: a
  press that opens/chips/shoves = +1 for that hue, "nothing to do here" = -1; laws get
  it as the `leak` input). Four blockers had to fall before any door was ever pressed:
  the read-out charged the leak twice (fixed: world 9 5->13 alone); the scent BFS
  skipped solids on its first ring (smell died beside the door); the waypoint was "last
  cell within 6" so it could never be a door costing 7 (now "first cell at least 6
  out"); and at leak 0.12 the horizon ring round the map beats leaking through ANY
  wall (innate leak must be CURIOUS: 0.5). Plus: press only when facing
  (`press_angle`), and the map learns a cell gave (`Atlas.gave`). RESULT: door probe
  0.08 -> 1.00, doors opened in the wild 0 -> ~1.5/life, learner halves idle presses
  and moves hues the right way; meals at n=16 are noise (SE ~2.5) — read DEEDS and
  probes. New defaults linger 0.99 / leak 0.5 / leak_lr 1.0; headline 16x20000: 42.6
  meals vs 32.6 (old leak) vs 25.6 (plain). Genome x world with 3 seeds: sparse
  radius 9/11/11, rich 12/12/13. Worlds are deterministic-chaotic: one verdict can
  swing one world by 40 meals.
- Done (a LOCAL model + the night lab, 2026-09-23): `.env` now points `nomos/llm.py`
  at the user's llama.cpp box (`http://llm.pgs.dom:8080/v1`, `qwen3.6-35b-a3b`, see
  `C:\gitlocal\phil-architecture\docs\llm.md`; ONE CPU slot shared with Open WebUI —
  `client.busy()` waits for other users). `chat(thinking=...)` toggles Qwen thinking per
  call (~60 s/turn off, ~10-13 min on); `client.last` = reasoning, tokens, seconds;
  `llm.embed()` = bge-m3 on :8081. The WaveSpeed lines are commented out in `.env`.
  DSL: `min`/`max` are variadic now (every model writes `max(a, b, c)`).
  `cortex/night_lab.py` is the loop built for it (journal 2026-09-23): the model
  designs STRUCTURE (a design turn, every 6th with thinking), rewrites a <=150-word
  THEORY carried instead of chat history, and OBSERVES a PNG of the champion's worst
  life with the track drawn (vision; accurate on the first try); rng tunes constants;
  a control arm tunes the ghost with the same budget; a law's value field is
  `diagnose()`d on a one-source test map and bounced if flat/saturated/aimless; twins
  (same structure, or same meals on every world) are caught. Probe finding: (1+1)
  tuning OVERFITS the training worlds badly (holdout 23.1 as written -> 16.9 after 13
  accepts), hence `--margin 0.02` and 32 FINAL worlds nobody sees until the end.
  Run: `python -m cortex.night_lab --hours 6 --dir out/cortex/night/<name>` (same
  command again = resume). First 6-hour night: `out/cortex/night/2026-09-23/`.
  RESULT: the model's structure won the 32 never-seen FINAL worlds — champion #35
  (injection divided by confinement `sum8(solid)`) 24.5 meals vs reference wave 18.8,
  control (ghost rng-tuned 138x) 17.1 — and showed no train->final drop, while every
  heavily-tuned law dropped (structure generalises, constants don't). The night's
  failure modes: world 16 ate the search (0 meals for EVERY law; the observer kept
  showing that one life, and ~15 islands were variations on commit/confinement);
  thinking turns (~17 min) paid nothing (0.6/3.0 meals) vs the champion's plain 75-s
  turn; the final theory contradicted the champion. PORTED back as `Config.conf_pen`
  + lineage gene (branch `atlas-confinement`): the law's 0.5 loses (37.4 vs 42.6),
  0.3 looked like +10 meals on seeds 0-15 and DID NOT REPLICATE on 16-31 (32.6 ->
  31.2); over 32 worlds +4.4 with SE ~5 = a shrug, a textbook winner's curse on
  chaotic worlds. Default stays 0.0 — available to the lineage and the next night.
  As a LAW it does replicate (2026-11-04, 128 fresh worlds): confine 22.4 / ghost 20.2
  / wave 17.2 — now `laws/atlas_confine.nomos`, the bar for the next run.
- Done (cloud lab, 2026-11-04, branch `cloud-night`): `night_lab --agent` +
  `cortex/mailbox.py` — the lab's model is the coding agent of a Claude Code (cloud)
  session answering files, no API. Recipe + rules: `docs/cloud-lab.md`. Observer
  rotates over the 3 worst lives, vocabulary novelty (no embeddings), `--control-law`
  / `--anchor`, pacing for a fast model. Runs live in TRACKED `runs/cloud-lab/<date>/`.
- Done (cloud study, 2026-11-05): six 12-h runs (full/blind/fresh x 2), compared in
  `docs/lab-journal.md`, verdict `python -m cortex.study_exam` on 128 fresh worlds. Best
  `laws/atlas_course.nomos` 32.8 meals vs confine 23.5. Eyes vs memory NOT answered.
  Tuning does help (+2.7 on fresh worlds); predictions do not improve. Summaries in
  `runs/cloud-lab/`, whole runs on branches `cloud-lab-*`.
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
