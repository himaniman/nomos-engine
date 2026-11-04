# night lab — 2026-11-04T11:35:16

Model `agent-in-session (mailbox)` | 12.0 h | world default | train [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] | holdout [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] | 8000 ticks/life | tune 5 | deep every 5 | theory every 6 | observe every 0

- anchor #0 `laws/atlas_wave.nomos`: train 14.8 meals [28, 27, 10, 35, 13, 4, 5, 5, 33, 12, 5, 7, 23, 23, 18, 8, 0, 28, 5, 6], holdout 19.6

- anchor #1 `laws/atlas_confine.nomos`: train 25.7 meals [37, 39, 18, 30, 23, 22, 13, 34, 26, 13, 33, 36, 33, 24, 29, 30, 0, 34, 14, 26], holdout 25.7

- anchor #2 `laws/atlas_ghost.nomos`: train 19.2 meals [28, 5, 25, 11, 13, 18, 16, 33, 29, 12, 4, 24, 21, 32, 16, 26, 0, 28, 23, 20], holdout 18.6

> 11:41 model job `design` started

---

## Turn 0 -> island #3

**Hypothesis:** Keep confinement suppression but add a MEMORY of the walked path (a fading trail channel that scrolls with the map) and damp the wave across it. The cell the agent came from always looks worse than the cells ahead, so equidistant targets and two-way smell bearings stop producing back-and-forth limit cycles and the agent commits to moving on, exploring more ground per tick.

Prediction 27.0, got **24.6** meals, fitness 477, per world [27, 21, 20, 34, 43, 17, 19, 34, 24, 22, 36, 26, 25, 27, 20, 33, 0, 31, 9, 23]; model 32.1 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Inhibition of return on top of confinement suppression: the path just walked
# is remembered and the wave is damped across it, so ties break forward.
channel source
channel solid
channel thing
channel here
channel value
channel confinement
channel injection
channel trail

# Local solidity: a source hemmed in by walls counts less.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# The walked path, fading; it scrolls with the map so it lies behind the agent.
rule remember_path:
  when 1
  set trail = max(here, 0.97 * trail)

rule inject:
  when 1
  set injection = max(thing, source) / (1 + 0.5 * confinement)

# Bellman wave, damped where the agent has just been.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.3 * trail)
```

> 11:54 model job `design` started

---

## Turn 1 -> island #4

**Hypothesis:** Exploration is where meals come from, but every frontier mark currently weighs the same 0.12 whether it opens onto a wide unexplored region or a one-cell sliver between walls. Here pure frontier marks (no smell, no seen thing) are scaled by a blurred mass of unknown cells around them, so the wave pulls the agent toward large unexplored areas where new food is likely, while food sources stay as in the confinement law.

Prediction 27.0, got **25.8** meals, fitness 505, per world [42, 39, 19, 25, 20, 16, 23, 35, 34, 29, 28, 30, 17, 35, 30, 35, 0, 15, 18, 26]; model 30.1 s, repairs 0

value on the test map: max 0.50, mean 0.26, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Frontier weighted by how much unexplored space lies behind it, on top of
# confinement suppression: big unknown regions pull, slivers between walls don't.
channel source
channel solid
channel free
channel thing
channel scent
channel value
channel confinement
channel unknown
channel openness
channel injection

rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Neither seen free nor seen solid.
rule unknown_cells:
  when 1
  set unknown = (1 - free) * (1 - solid)

# A blurred mass of unknown cells around each cell (bounded diffusion).
rule open_area:
  when 1
  set openness = 0.5 * avg8(unknown) + 0.5 * avg8(openness)

# Food (seen or smelt) as before; pure frontier marks scaled by the unknown mass.
rule inject:
  when 1
  set injection = where(max(thing, scent) > 0, max(thing, source), source * (0.5 + 1.5 * openness)) / (1 + 0.5 * confinement)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

### Holdout checkpoint at 0.54 h

```
  #4   train 25.8 meals  holdout 27.6  (tuned 3x) | Exploration is where meals come from, but every frontier mark currently weighs the same 0.12 whether it opens 
  #3   train 24.6 meals  holdout 26.8  (tuned 5x) | Keep confinement suppression but add a MEMORY of the walked path (a fading trail channel that scrolls with the
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 8x: train 25.7 meals, holdout 25.7
```

> 12:12 model job `design` started

---

## Turn 2 -> island #5

**Hypothesis:** Food is not uniformly spread: a morsel with other food around it is worth more, because the next meal is then a few steps away instead of a long search. A blurred food-density channel (seen or smelt food, minus remembered refusals in `bad`) multiplies each food source, so the agent prefers patches over lone morsels; the open-frontier weighting and confinement suppression of #4 stay.

Prediction 27.0, got **24.6** meals, fitness 492, per world [31, 19, 25, 26, 23, 18, 30, 24, 29, 27, 29, 22, 31, 38, 31, 40, 0, 22, 2, 26]; model 34.1 s, repairs 0; vocabulary twin of #4

value on the test map: max 0.50, mean 0.26, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Patches: food that has other food around it is worth more, because the next
# meal is then a step away. On top of #4 (open-frontier weighting + confinement).
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel value
channel confinement
channel unknown
channel openness
channel food
channel patch
channel injection

rule calc_conf:
  when 1
  set confinement = sum8(solid)

rule unknown_cells:
  when 1
  set unknown = (1 - free) * (1 - solid)

rule open_area:
  when 1
  set openness = 0.5 * avg8(unknown) + 0.5 * avg8(openness)

# What is wanted and known to be food (seen or smelt), minus remembered refusals.
rule food_cells:
  when 1
  set food = max(thing, scent) * (1 - bad)

# A blurred density of food around each cell (bounded diffusion).
rule patch_density:
  when 1
  set patch = 0.6 * avg8(food) + 0.4 * avg8(patch)

# Food boosted by the patch it sits in; frontier scaled by unknown mass.
rule inject:
  when 1
  set injection = where(max(thing, scent) > 0, max(thing, source) * (1 - bad) * (1 + 3 * patch), source * (0.5 + 1.5 * openness)) / (1 + 0.5 * confinement)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 12:31 model job `design` started

---

## Turn 3 -> island #6

**Hypothesis:** Distrust old evidence. A boredom channel grows slowly on every map cell that keeps holding a lure (seen thing, smell waypoint or frontier mark) and relaxes where none is, and it scales that lure down; a reachable target is reached before boredom matters, but food behind a wall the agent cannot open, or a frontier it cannot get to, fades out and releases the agent to go elsewhere instead of pressing against the same obstacle for thousands of ticks.

Prediction 26.5, got **24.9** meals, fitness 484, per world [48, 16, 31, 37, 14, 38, 16, 22, 30, 19, 30, 23, 20, 31, 34, 28, 0, 40, 6, 14]; model 26.1 s, repairs 0

value on the test map: max 0.92, mean 0.50, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Habituation (boredom): a source that has been on the map a long time without
# being reached loses its pull, so unreachable lures and dead frontiers fade.
channel source
channel solid
channel thing
channel here
channel value
channel confinement
channel lure
channel boredom
channel injection

rule calc_conf:
  when 1
  set confinement = sum8(solid)

rule lure_cells:
  when 1
  set lure = max(thing, source)

# Grows slowly while a lure sits on a cell, relaxes where none is, reset under the agent.
rule habituate:
  when 1
  set boredom = clamp(0.998 * boredom + 0.003 * min(lure * 8, 1), 0, 1) * (1 - here)

rule inject:
  when 1
  set injection = lure * (1 - 0.8 * boredom) / (1 + 0.5 * confinement)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

### Holdout checkpoint at 1.12 h

```
  #5   train 26.6 meals  holdout 28.6  (tuned 5x) | Food is not uniformly spread: a morsel with other food around it is worth more, because the next meal is then 
  #4   train 26.1 meals  holdout 24.4  (tuned 5x) | Exploration is where meals come from, but every frontier mark currently weighs the same 0.12 whether it opens 
  #6   train 25.6 meals  holdout 23.5  (tuned 2x) | Distrust old evidence. A boredom channel grows slowly on every map cell that keeps holding a lure (seen thing,
  #3   train 24.6 meals  holdout 26.8  (tuned 5x) | Keep confinement suppression but add a MEMORY of the walked path (a fading trail channel that scrolls with the
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 16x: train 25.7 meals, holdout 25.7
```

> 12:50 model job `deep` started

---

## Turn 4 (DEEP) -> island #7

**Hypothesis:** The map forgets everything more than 12 cells away, so ground the agent covered a while ago comes back as fresh frontier and it re-explores it; the efficient search on a forgetful map is ballistic — keep going the way you were going. A long-lived trail of the walked path is diffused into a wide, screened "behind" field (high around and behind the path, low ahead), and pure frontier marks are muted by exp(-k*behind), so the frontier ahead always beats the frontier behind and the agent sweeps new ground in long runs instead of dithering between frontiers on opposite sides; food sources and confinement suppression are unchanged.

Prediction 28.0, got **28.2** meals, fitness 552, per world [42, 38, 27, 27, 28, 22, 16, 39, 17, 26, 38, 36, 22, 38, 30, 52, 0, 23, 12, 31]; model 44.1 s, repairs 0

value on the test map: max 0.96, mean 0.51, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Ballistic exploration: the map forgets everything 12 cells away, so the way to
# cover new ground is to keep going. A wide blurred "behind" field grows around
# the walked path; frontier marks inside it are muted, so the frontier AHEAD wins
# and the agent sweeps in long straight-ish runs instead of dithering between
# frontiers on opposite sides. Food is untouched.
channel source
channel solid
channel thing
channel scent
channel here
channel value
channel confinement
channel trail
channel behind
channel injection

rule calc_conf:
  when 1
  set confinement = sum8(solid)

# The walked path, long-lived; it scrolls with the map.
rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)

# Screened diffusion of the path: high near and behind it, low far ahead (bounded by 1).
rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail

# Food as in confinement; pure frontier marks muted where the agent has been.
rule inject:
  when 1
  set injection = where(max(thing, scent) > 0, max(thing, source), source * exp(-40 * behind)) / (1 + 0.5 * confinement)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 13:08 model job `design` started

---

## Turn 5 -> island #8

**Hypothesis:** The harness steps the law only 6 times per thought, so the wave moves at most 6 cells per thought: a freshly seen morsel 12 cells away needs two thoughts before the agent feels it, and an eaten one leaves a stale ridge that keeps pulling for several thoughts after it is gone. Adding two-cell hops (discounted as two steps, gated by the passability of the cell jumped over so walls still block) doubles the wave's speed, so the field the agent climbs is closer to the true current plan and less time is lost walking along stale gradients; exploration stays ballistic as in #7.

Prediction 29.0, got **21.8** meals, fitness 429, per world [7, 31, 21, 33, 15, 32, 46, 43, 20, 2, 11, 34, 13, 36, 5, 46, 0, 6, 16, 18]; model 48.1 s, repairs 0; vocabulary twin of #7

value on the test map: max 0.96, mean 0.51, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Express wave: the harness steps the law only 6 times per thought, so a new
# source needs several thoughts to reach the agent and an eaten one leaves a
# stale ridge for several more. Two-cell hops (gated by the passability of the
# cell jumped over) double the wave speed. Exploration as in #7 (ballistic).
channel source
channel solid
channel thing
channel scent
channel here
channel value
channel confinement
channel trail
channel behind
channel injection
channel pass

rule calc_conf:
  when 1
  set confinement = sum8(solid)

rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)

rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail

rule inject:
  when 1
  set injection = where(max(thing, scent) > 0, max(thing, source), source * exp(-40 * behind)) / (1 + 0.5 * confinement)

# How much wave a cell lets through.
rule passability:
  when 1
  set pass = 1 - solid * (1 - leak)

# One-step Bellman terms plus two-step hops over a passable middle cell.
rule wave:
  when 1
  set value = max(injection, 0.94 * max4(value), 0.916214 * max8(value), 0.8836 * max(shift(value, 2, 0) * shift(pass, 1, 0), shift(value, -2, 0) * shift(pass, -1, 0), shift(value, 0, 2) * shift(pass, 0, 1), shift(value, 0, -2) * shift(pass, 0, -1)), 0.8394 * max(shift(value, 2, 2) * shift(pass, 1, 1), shift(value, -2, 2) * shift(pass, -1, 1), shift(value, 2, -2) * shift(pass, 1, -1), shift(value, -2, -2) * shift(pass, -1, -1))) * pass
```

> 13:12 model job `theory` started

---

## Theory (after 6 design turns)

Helps: where the agent explores. Ballistic exploration (#7: frontier muted inside a wide field around the walked path, so the frontier ahead wins) is the best idea, 28.2 untuned vs 25.7 for confinement (#1). Weighting frontier by unknown mass behind it (#4, +1.2) also helped on train, but its holdout fell after tuning (27.6 -> 24.4), so treat it as weak. Confinement suppression (#1) is a solid base.
Hurts: damping the wave itself across the path (#3, -1.1); boredom on long-held lures (#6, -1.8); and above all a FASTER wave (#8, two-cell hops, -6.4). That last one suggests the slow, laggy wave is itself useful inertia: a stale field keeps the agent on its current target. Patch weighting (#5) was -1.4 on train but best holdout 28.6 — ambiguous, maybe its (1-bad) term.
Untried: explicit commitment to a target, poison avoidance alone, near-vs-far food trade, door/corridor geometry.
Rests on single runs; per-world spread is large (world 16 always 0, world 18 2-18).

### Holdout checkpoint at 1.67 h

```
  #7   train 28.2 meals  holdout 25.4  (tuned 5x) | The map forgets everything more than 12 cells away, so ground the agent covered a while ago comes back as fres
  #5   train 26.6 meals  holdout 28.6  (tuned 5x) | Food is not uniformly spread: a morsel with other food around it is worth more, because the next meal is then 
  #4   train 26.1 meals  holdout 24.4  (tuned 5x) | Exploration is where meals come from, but every frontier mark currently weighs the same 0.12 whether it opens 
  #6   train 25.6 meals  holdout 23.5  (tuned 5x) | Distrust old evidence. A boredom channel grows slowly on every map cell that keeps holding a lure (seen thing,
  #3   train 24.6 meals  holdout 26.8  (tuned 5x) | Keep confinement suppression but add a MEMORY of the walked path (a fading trail channel that scrolls with the
  #8   train 21.8 meals  holdout  -    (tuned 1x) | The harness steps the law only 6 times per thought, so the wave moves at most 6 cells per thought: a freshly s
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 25x: train 28.2 meals, holdout 26.6
```
