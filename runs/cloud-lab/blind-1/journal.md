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

> 13:25 model job `design` started

---

## Turn 6 -> island #9

**Hypothesis:** Rooms behind doors are stocked larders, but the frontier marks only unknown cells touching known-free ground, so a closed room is invisible to exploration unless food inside happens to be smelt. A solid cell whose leak has been taught above the innate guess (its hue opened when pressed) and that faces known-free ground becomes a curiosity source itself, muted behind the agent like the frontier in #7, so once the agent has learned what a door looks like it opens the doors it passes and harvests the rooms behind them.

Prediction 29.0, got **27.3** meals, fitness 536, per world [42, 38, 24, 27, 22, 24, 16, 39, 17, 26, 38, 36, 17, 23, 29, 41, 21, 23, 12, 31]; model 58.1 s, repairs 0; vocabulary twin of #7

value on the test map: max 0.96, mean 0.51, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Doors are frontier. Rooms behind doors are larders, but the frontier only marks
# unknown cells that touch known-free ones, so a closed room is never worth
# exploring unless something inside is smelt. A solid cell whose leak has been
# taught ABOVE the innate guess (a hue that opened when pressed) and that faces
# known-free ground becomes a curiosity source itself. Exploration as in #7.
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel confinement
channel trail
channel behind
channel door
channel injection

rule calc_conf:
  when 1
  set confinement = sum8(solid)

rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)

rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail

# A known solid of a hue taught to give way, standing next to known-free ground.
rule find_doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * clamp(sum4(free), 0, 1)

# Food as before; frontier marks and doors muted where the agent has been.
rule inject:
  when 1
  set injection = max(where(max(thing, scent) > 0, max(thing, source), source * exp(-40 * behind)) / (1 + 0.5 * confinement), 0.3 * door * exp(-40 * behind))

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 13:39 model job `design` started

---

## Turn 7 -> island #10

**Hypothesis:** Object permanence. A morsel is painted only while it is inside the field of view, so turning toward one target drops the others from the map and a corner hides the very morsel being chased, which makes the agent dither and abandon approaches. Seen, wanted morsels (appetite-weighted, minus refusals) are written into a slowly fading memory channel that keeps injecting after they leave view and is erased once the agent stands on or beside the spot; unlike the ghost law, smell waypoints and frontier marks are not remembered. Base is #9 (ballistic exploration + learned doors as curiosity), which was the first law to eat on world 16.

Prediction 29.0, got **26.9** meals, fitness 506, per world [32, 29, 27, 33, 22, 25, 10, 33, 30, 16, 41, 27, 27, 32, 23, 51, 11, 19, 10, 40]; model 26.1 s, repairs 0; vocabulary twin of #9

value on the test map: max 0.96, mean 0.51, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Object permanence: a morsel is painted only while it is in the field of view,
# so turning toward one target drops the others and a corner hides the one being
# chased. Seen, wanted morsels are remembered on the map (fading slowly, erased
# under and beside the agent) and keep injecting after they leave view. Smell and
# frontier are NOT remembered (the ghost's mistake). Base: #9 (ballistic + doors).
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel here
channel value
channel confinement
channel trail
channel behind
channel door
channel seen
channel injection

rule calc_conf:
  when 1
  set confinement = sum8(solid)

rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)

rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail

rule find_doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * clamp(sum4(free), 0, 1)

# Where a wanted morsel was last seen; gone once the agent stands on or beside it.
rule remember_things:
  when 1
  set seen = max(thing * min(source, 1) * (1 - bad), 0.995 * seen) * (1 - here) * (1 - max8(here))

rule inject:
  when 1
  set injection = max(where(max(thing, scent) > 0, max(thing, source), source * exp(-40 * behind)) / (1 + 0.5 * confinement), 0.3 * door * exp(-40 * behind), seen / (1 + 0.5 * confinement))

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

### Holdout checkpoint at 2.20 h

```
  #7   train 28.2 meals  holdout 25.4  (tuned 5x) | The map forgets everything more than 12 cells away, so ground the agent covered a while ago comes back as fres
  #9   train 27.3 meals  holdout 25.8  (tuned 5x) | Rooms behind doors are stocked larders, but the frontier marks only unknown cells touching known-free ground, 
  #5   train 26.6 meals  holdout 28.6  (tuned 5x) | Food is not uniformly spread: a morsel with other food around it is worth more, because the next meal is then 
  #4   train 26.1 meals  holdout 24.4  (tuned 5x) | Exploration is where meals come from, but every frontier mark currently weighs the same 0.12 whether it opens 
  #10  train 26.9 meals  holdout  -    (tuned 1x) | Object permanence. A morsel is painted only while it is inside the field of view, so turning toward one target
  #6   train 25.6 meals  holdout 23.5  (tuned 5x) | Distrust old evidence. A boredom channel grows slowly on every map cell that keeps holding a lure (seen thing,
  #3   train 24.6 meals  holdout 26.8  (tuned 5x) | Keep confinement suppression but add a MEMORY of the walked path (a fading trail channel that scrolls with the
  #8   train 21.8 meals  holdout  -    (tuned 5x) | The harness steps the law only 6 times per thought, so the wave moves at most 6 cells per thought: a freshly s
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 36x: train 28.2 meals, holdout 26.6
```

> 13:58 model job `design` started

---

## Turn 8 -> island #11

**Hypothesis:** Every law so far injected max(thing, source), and `thing` is 1 on every seen morsel, poison included, so it overrode the appetite the brain learns (source carries the want-weighted morsel); the agent keeps walking to nightcaps it has learned to refuse and stands by them. Here morsels pull only through `source`, and a remembered refusal (`bad`) casts a decaying aversion halo that hushes lures near it and bends the wave around the burnt place, so after one burn the agent stops spending time and energy on poison. Base is #9 (ballistic exploration + learned doors).

Prediction 28.5, got **28.9** meals, fitness 553, per world [37, 38, 29, 42, 15, 24, 20, 51, 33, 25, 32, 35, 6, 24, 27, 33, 26, 31, 18, 32]; model 46.2 s, repairs 0; vocabulary twin of #9, #10

value on the test map: max 0.96, mean 0.51, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Honest appetite + aversion. `thing` is 1 on EVERY seen morsel, poison included,
# and the old max(thing, source) let it override the appetite the brain has
# learned (source = want-weighted). Here a morsel pulls only by `source`, and a
# remembered refusal casts a decaying aversion halo that both hushes lures and
# bends the wave around the burnt place. Base: #9 (ballistic + doors).
channel source
channel solid
channel free
channel scent
channel thing
channel bad
channel here
channel value
channel confinement
channel trail
channel behind
channel door
channel aversion
channel injection

rule calc_conf:
  when 1
  set confinement = sum8(solid)

rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)

rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail

rule find_doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * clamp(sum4(free), 0, 1)

# A halo around remembered refusals, falling off with distance.
rule fear:
  when 1
  set aversion = max(bad, 0.7 * max8(aversion))

# Appetite-weighted lures only; frontier and doors muted behind; all hushed by aversion.
rule inject:
  when 1
  set injection = max(where(max(thing, scent) > 0, source, source * exp(-40 * behind)) / (1 + 0.5 * confinement), 0.3 * door * exp(-40 * behind)) * (1 - aversion)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.5 * aversion)
```

> 14:14 model job `deep` started

---

## Turn 9 (DEEP) -> island #12

**Hypothesis:** Most of this world's food sits in closed places — larder rooms behind doors and caves behind rubble — but the frontier only marks unknown cells that touch known-FREE ground, so an unexplored room behind a wall is worth nothing to exploration, and the door rule of #9 only fires after a door's hue has already been taught. Here unknown cells directly behind a known wall face become a faint curiosity source (frontier strength, muted behind the agent); the wave reaches them only through the wall's leak, so walls are probed mainly when the open frontier is far away, and the existing leak learner (pressing opens = up, stone = down) quickly confines the probing to door- and rubble-coloured walls, op

Prediction 30.0, got **23.7** meals, fitness 454, per world [24, 36, 25, 42, 15, 30, 15, 24, 33, 9, 32, 26, 21, 26, 21, 46, 0, 6, 15, 28]; model 90.2 s, repairs 0; vocabulary twin of #11, #9, #10

value on the test map: max 0.96, mean 0.51, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# What lies behind a wall. Larders and caves are closed rooms, and the frontier
# only marks unknown cells touching known-FREE ground, so a walled-off unknown is
# worth nothing until the leak of its door's hue has been taught. Here unknown
# cells just behind a known wall face are a faint curiosity source; the wave
# reaches it only through the wall's leak, so the agent probes walls when the open
# frontier is far, and the leak learner (press: opens = +, stone = -) soon keeps
# the probing on door-coloured walls. Base: #11 (appetite, aversion, ballistic, doors).
channel source
channel solid
channel free
channel scent
channel thing
channel bad
channel here
channel value
channel confinement
channel trail
channel behind
channel door
channel aversion
channel wallface
channel beyond
channel injection

rule calc_conf:
  when 1
  set confinement = sum8(solid)

rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)

rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail

rule find_doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * clamp(sum4(free), 0, 1)

rule fear:
  when 1
  set aversion = max(bad, 0.7 * max8(aversion))

# A known wall that faces known-free ground.
rule find_wallface:
  when 1
  set wallface = solid * clamp(sum4(free), 0, 1)

# Unknown cells right behind such a wall: a room nobody has looked into.
rule find_beyond:
  when 1
  set beyond = clamp((1 - free) * (1 - solid), 0, 1) * clamp(sum4(wallface), 0, 1)

rule inject:
  when 1
  set injection = max(where(max(thing, scent) > 0, source, source * exp(-40 * behind)) / (1 + 0.5 * confinement), 0.3 * door * exp(-40 * behind), 0.12 * beyond * exp(-40 * behind)) * (1 - aversion)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.5 * aversion)
```

### Holdout checkpoint at 2.78 h

```
  #11  train 28.9 meals  holdout 28.1  (tuned 5x) | Every law so far injected max(thing, source), and `thing` is 1 on every seen morsel, poison included, so it ov
  #7   train 28.2 meals  holdout 25.4  (tuned 7x) | The map forgets everything more than 12 cells away, so ground the agent covered a while ago comes back as fres
  #9   train 27.3 meals  holdout 25.8  (tuned 5x) | Rooms behind doors are stocked larders, but the frontier marks only unknown cells touching known-free ground, 
  #5   train 26.6 meals  holdout 28.6  (tuned 5x) | Food is not uniformly spread: a morsel with other food around it is worth more, because the next meal is then 
  #4   train 26.1 meals  holdout 24.4  (tuned 5x) | Exploration is where meals come from, but every frontier mark currently weighs the same 0.12 whether it opens 
  #10  train 26.9 meals  holdout  -    (tuned 5x) | Object permanence. A morsel is painted only while it is inside the field of view, so turning toward one target
  #6   train 25.6 meals  holdout 23.5  (tuned 5x) | Distrust old evidence. A boredom channel grows slowly on every map cell that keeps holding a lure (seen thing,
  #3   train 24.6 meals  holdout 26.8  (tuned 5x) | Keep confinement suppression but add a MEMORY of the walked path (a fading trail channel that scrolls with the
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 46x: train 28.2 meals, holdout 26.6
```

> 14:38 model job `design` started

---

## Turn 10 -> island #13

**Hypothesis:** Food and exploration have been one wave with one discount, so how a far frontier trades against a near one, and exploration against food, was fixed by distance alone. Splitting them into two waves with their own horizons — food (appetite-weighted things and smell, confinement-suppressed) with the usual steep discount, exploration (frontier and learned doors, muted behind the agent) with a much slower one — lets the far frontier straight ahead keep its pull over a near frontier at the side, making exploration more ballistic, while nearby food still wins; value is the larger of the two.

Prediction 29.0, got **25.1** meals, fitness 496, per world [40, 24, 35, 35, 25, 26, 20, 33, 28, 21, 15, 34, 30, 13, 22, 40, 0, 7, 22, 32]; model 38.1 s, repairs 0

value on the test map: max 0.96, mean 0.70, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Two waves, two horizons. Food and exploration were one wave with one discount,
# so the trade between a far frontier and a near one was fixed by distance alone.
# Here food (appetite-weighted things, smell) spreads in its own wave with the
# usual 0.94, and exploration (frontier, learned doors, muted behind the agent)
# in a second, slower-decaying wave, so the far frontier ahead keeps its pull over
# the near one at the side; `value` is the larger of the two.
channel source
channel solid
channel free
channel scent
channel thing
channel bad
channel here
channel value
channel trail
channel behind
channel door
channel lure
channel roam
channel foodwave
channel roamwave
channel pass

rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)

rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail

rule find_doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * clamp(sum4(free), 0, 1)

rule passability:
  when 1
  set pass = (1 - solid * (1 - leak)) * (1 - bad)

# Split the senses: what is food, and what is only worth looking at.
rule split_sources:
  when 1
  set lure = where(max(thing, scent) > 0, source, 0) / (1 + 0.5 * sum8(solid))
  set roam = max(where(max(thing, scent) > 0, 0, source), 0.3 * door) * exp(-40 * behind)

rule food_wave:
  when 1
  set foodwave = max(lure, 0.94 * max4(foodwave), 0.916214 * max8(foodwave)) * pass

rule roam_wave:
  when 1
  set roamwave = max(roam, 0.97 * max4(roamwave), 0.957847 * max8(roamwave)) * pass

rule readout:
  when 1
  set value = max(foodwave, roamwave)
```

### Holdout checkpoint at 3.28 h

```
  #11  train 28.9 meals  holdout 28.1  (tuned 5x) | Every law so far injected max(thing, source), and `thing` is 1 on every seen morsel, poison included, so it ov
  #7   train 28.2 meals  holdout 25.4  (tuned 7x) | The map forgets everything more than 12 cells away, so ground the agent covered a while ago comes back as fres
  #9   train 27.3 meals  holdout 25.8  (tuned 5x) | Rooms behind doors are stocked larders, but the frontier marks only unknown cells touching known-free ground, 
  #5   train 26.6 meals  holdout 28.6  (tuned 5x) | Food is not uniformly spread: a morsel with other food around it is worth more, because the next meal is then 
  #4   train 26.1 meals  holdout 24.4  (tuned 5x) | Exploration is where meals come from, but every frontier mark currently weighs the same 0.12 whether it opens 
  #10  train 26.9 meals  holdout  -    (tuned 5x) | Object permanence. A morsel is painted only while it is inside the field of view, so turning toward one target
  #6   train 25.6 meals  holdout 23.5  (tuned 5x) | Distrust old evidence. A boredom channel grows slowly on every map cell that keeps holding a lure (seen thing,
  #13  train 25.1 meals  holdout  -    (tuned 4x) | Food and exploration have been one wave with one discount, so how a far frontier trades against a near one, an
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 55x: train 28.2 meals, holdout 26.6
```

> 14:55 model job `design` started

---

## Turn 11 -> island #14

**Hypothesis:** The theory's strongest negative result is that a FASTER wave (#8) cost 6 meals, which suggests the lag of a slow wave works as commitment: the field the agent climbs changes its mind slowly, so the agent finishes approaches instead of flipping between lures. This tests that directly: the Bellman target is computed separately and value only relaxes part of the way toward it each step, so a newly painted or flickering lure must persist for a few thoughts before it can steal the agent from its current target, while the fixed point is the same wave as #11. If inertia helps, this beats #11; if it loses, the express-wave result had another cause.

Prediction 29.5, got **25.7** meals, fitness 509, per world [30, 11, 29, 20, 17, 38, 29, 23, 29, 26, 35, 28, 24, 36, 29, 33, 0, 28, 15, 34]; model 36.1 s, repairs 0; vocabulary twin of #11, #12, #9, #10

value on the test map: max 0.97, mean 0.32, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Inertia, tested directly. The two-cell express wave (#8) cost 6 meals, which
# suggests the lag of a slow wave is useful: the field the agent climbs changes
# its mind slowly. Here the Bellman target is computed separately and
# `value` RELAXES toward it (a blend of the old field and the new), so a newly
# painted lure must persist for a few thoughts before it can steal the agent from
# its current target, while the fixed point is still the same wave. Base: #11.
channel source
channel solid
channel free
channel scent
channel thing
channel bad
channel here
channel value
channel confinement
channel trail
channel behind
channel door
channel aversion
channel injection
channel target

rule calc_conf:
  when 1
  set confinement = sum8(solid)

rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)

rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail

rule find_doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * clamp(sum4(free), 0, 1)

rule fear:
  when 1
  set aversion = max(bad, 0.7 * max8(aversion))

rule inject:
  when 1
  set injection = max(where(max(thing, scent) > 0, source, source * exp(-40 * behind)) / (1 + 0.5 * confinement), 0.3 * door * exp(-40 * behind)) * (1 - aversion)

# What the wave would be now, built from the current field.
rule bellman:
  when 1
  set target = max(injection, 0.94 * max4(value), 0.916214 * max8(value)) * (1 - solid * (1 - leak)) * (1 - 0.5 * aversion)

# The field the agent climbs moves only part of the way toward it each step.
rule relax:
  when 1
  set value = 0.6 * value + 0.4 * target
```

> 14:59 model job `theory` started

---

## Theory (after 12 design turns)

Gains come only from WHAT is injected, never from changing how the wave moves. Helps: ballistic exploration (#7, frontier muted around the walked path: 28.2 vs 25.7 for #1); honest appetite + aversion (#11, morsels pull only by want-weighted source, halo around bad: 28.9 train, 28.1 holdout vs control 26.6); learned doors as curiosity (#9-#11 are the only laws that ever ate on world 16: 21, 11, 26).
Hurts: every change to wave dynamics — faster two-cell hops (#8, -6.4), slower relaxation (#14, -3.2: refutes my old "inertia" reading of #8), two waves with different discounts (#13, -3.8), damping along the path (#3). Plain Bellman at 6 subticks is a sweet spot. Also hurts: probing every wall for rooms behind it (#12, -5.2, wastes presses, world 16 back to 0); boredom (#6).
Untried: weighing food by kind or brightness, smell reliability, corridor geometry. Noise is ~2 meals.

> 15:14 model job `design` started

---

## Turn 12 -> island #15

**Hypothesis:** The wave has never respected the body. max8 carries value diagonally between two walls that touch at a corner, and past any single wall corner, but a body of radius 0.45 moving cell-centre to cell-centre diagonally brushes both corner cells, so the agent is led into gaps it cannot fit and slides or wedges, losing ticks in exactly the cluttered places (caves, rubble, room corners) where food sits. Here a diagonal step only carries the wave when both orthogonal cells it brushes are open, and a diagonal neighbour of the agent that its body cannot reach directly reads as nothing, so routes bend orthogonally around corners; injection is #11's (not a change to the wave's speed or discount, only to

Prediction 29.5, got **29.3** meals, fitness 554, per world [35, 31, 29, 34, 23, 31, 25, 32, 21, 13, 28, 33, 39, 39, 26, 32, 26, 22, 35, 32]; model 52.2 s, repairs 0; vocabulary twin of #11, #14, #10, #12

value on the test map: max 0.96, mean 0.51, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# The wave must respect the body. max8 lets the wave cut diagonally between two
# walls that touch at a corner, and past any wall corner, but a body of radius
# 0.45 cannot squeeze through such a pinch: it is led into a gap it cannot fit,
# and slides or sticks. Here a diagonal step only carries the wave when BOTH
# orthogonal cells it passes are open, and a diagonal neighbour the agent
# cannot reach directly reads as nothing. Injection as in #11.
channel source
channel solid
channel free
channel scent
channel thing
channel bad
channel here
channel value
channel confinement
channel trail
channel behind
channel door
channel aversion
channel injection
channel open
channel pinched

rule calc_conf:
  when 1
  set confinement = sum8(solid)

rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)

rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail

rule find_doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * clamp(sum4(free), 0, 1)

rule fear:
  when 1
  set aversion = max(bad, 0.7 * max8(aversion))

# How clear a cell is for a body to pass beside.
rule clearance:
  when 1
  set open = 1 - clamp(solid, 0, 1)

# A diagonal neighbour of the agent that its body cannot reach directly.
rule pinch:
  when 1
  set pinched = max(shift(here, 1, 1) * (1 - min(shift(open, 1, 0), shift(open, 0, 1))), shift(here, -1, 1) * (1 - min(shift(open, -1, 0), shift(open, 0, 1))), shift(here, 1, -1) * (1 - min(shift(open, 1, 0), shift(open, 0, -1))), shift(here, -1, -1) * (1 - min(shift(open, -1, 0), shift(open, 0, -1))))

rule inject:
  when 1
  set injection = max(where(max(thing, scent) > 0, source, source * exp(-40 * behind)) / (1 + 0.5 * confinement), 0.3 * door * exp(-40 * behind)) * (1 - aversion)

# Orthogonal steps as before; each diagonal gated by the two corner cells it brushes.
rule wave:
  when 1
  set value = max(injection, 0.94 * max4(value), 0.916214 * max(shift(value, 1, 1) * min(shift(open, 1, 0), shift(open, 0, 1)), shift(value, -1, 1) * min(shift(open, -1, 0), shift(open, 0, 1)), shift(value, 1, -1) * min(shift(open, 1, 0), shift(open, 0, -1)), shift(value, -1, -1) * min(shift(open, -1, 0), shift(open, 0, -1)))) * (1 - solid * (1 - leak)) * (1 - 0.5 * aversion) * (1 - pinched)
```

### Holdout checkpoint at 3.81 h

```
  #15  train 29.6 meals  holdout 29.6  (tuned 1x) | The wave has never respected the body. max8 carries value diagonally between two walls that touch at a corner,
  #11  train 28.9 meals  holdout 28.1  (tuned 5x) | Every law so far injected max(thing, source), and `thing` is 1 on every seen morsel, poison included, so it ov
  #7   train 28.2 meals  holdout 25.4  (tuned 7x) | The map forgets everything more than 12 cells away, so ground the agent covered a while ago comes back as fres
  #9   train 27.3 meals  holdout 25.8  (tuned 5x) | Rooms behind doors are stocked larders, but the frontier marks only unknown cells touching known-free ground, 
  #5   train 26.6 meals  holdout 28.6  (tuned 5x) | Food is not uniformly spread: a morsel with other food around it is worth more, because the next meal is then 
  #4   train 26.1 meals  holdout 24.4  (tuned 5x) | Exploration is where meals come from, but every frontier mark currently weighs the same 0.12 whether it opens 
  #14  train 25.7 meals  holdout  -    (tuned 5x) | The theory's strongest negative result is that a FASTER wave (#8) cost 6 meals, which suggests the lag of a sl
  #10  train 26.9 meals  holdout  -    (tuned 5x) | Object permanence. A morsel is painted only while it is inside the field of view, so turning toward one target
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 62x: train 28.2 meals, holdout 26.6
```

> 15:37 model job `design` started

---

## Turn 13 -> island #16

**Hypothesis:** Turning has a price the wave does not know: the body turns 0.12 rad per tick and crawls while more than a radian off, so turning round costs ~26 ticks (~9 cells of travel), yet every neighbour is priced alike and near-ties between a lure ahead and one behind are settled by noise. The cell the agent came from (the freshest trail cell on its ring) is charged a turn cost and the two cells flanking it half that, so the agent reverses only when what lies behind is clearly better — continuing #15's line that the field should respect the body (it changes only the agent's own ring, not the wave's speed or discount).

Prediction 30.0, got **28.1** meals, fitness 539, per world [38, 30, 30, 20, 19, 36, 22, 33, 29, 24, 29, 28, 28, 48, 13, 27, 24, 26, 25, 34]; model 98.2 s, repairs 0; vocabulary twin of #15

value on the test map: max 0.96, mean 0.51, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Turning has a price. The body turns 0.12 rad per tick and crawls while facing
# more than a radian off, so turning round costs ~26 ticks, ~9 cells of travel,
# but the wave prices every neighbour alike. The cell the agent came from (the
# freshest trail cell beside it) and the two beside that one are charged a turn
# cost, so it reverses only when what lies behind is clearly better.
# Injection and body-honest diagonals as in #15.
channel source
channel solid
channel free
channel scent
channel thing
channel bad
channel here
channel value
channel confinement
channel trail
channel behind
channel door
channel aversion
channel injection
channel open
channel pinched
channel ring
channel recent
channel freshest
channel back
rule calc_conf:
  when 1
  set confinement = sum8(solid)
rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)
rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail
rule find_doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * clamp(sum4(free), 0, 1)
rule fear:
  when 1
  set aversion = max(bad, 0.7 * max8(aversion))
rule clearance:
  when 1
  set open = 1 - clamp(solid, 0, 1)
rule pinch:
  when 1
  set pinched = max(shift(here, 1, 1) * (1 - min(shift(open, 1, 0), shift(open, 0, 1))), shift(here, -1, 1) * (1 - min(shift(open, -1, 0), shift(open, 0, 1))), shift(here, 1, -1) * (1 - min(shift(open, 1, 0), shift(open, 0, -1))), shift(here, -1, -1) * (1 - min(shift(open, -1, 0), shift(open, 0, -1))))
# The agent's eight neighbours, and how recently each was walked.
rule neighbours:
  when 1
  set ring = max8(here)
  set recent = trail * max8(here)
# The best recency around the agent, carried back out to its ring.
rule freshest_mark:
  when 1
  set freshest = max(here * max8(recent), max8(here * max8(recent)) * ring)
# The cell it came from, and half for the two cells flanking it on the ring.
rule backward:
  when 1
  set back = max(ring * (recent >= 0.99 * freshest) * (recent > 0.01), 0.5 * ring * max4(ring * (recent >= 0.99 * freshest) * (recent > 0.01)))
rule inject:
  when 1
  set injection = max(where(max(thing, scent) > 0, source, source * exp(-40 * behind)) / (1 + 0.5 * confinement), 0.3 * door * exp(-40 * behind)) * (1 - aversion)
rule wave:
  when 1
  set value = max(injection, 0.94 * max4(value), 0.916214 * max(shift(value, 1, 1) * min(shift(open, 1, 0), shift(open, 0, 1)), shift(value, -1, 1) * min(shift(open, -1, 0), shift(open, 0, 1)), shift(value, 1, -1) * min(shift(open, 1, 0), shift(open, 0, -1)), shift(value, -1, -1) * min(shift(open, -1, 0), shift(open, 0, -1)))) * (1 - solid * (1 - leak)) * (1 - 0.5 * aversion) * (1 - pinched) * (1 - 0.4 * back)
```

### Holdout checkpoint at 4.32 h

```
  #15  train 29.6 meals  holdout 29.6  (tuned 5x) | The wave has never respected the body. max8 carries value diagonally between two walls that touch at a corner,
  #11  train 28.9 meals  holdout 28.1  (tuned 5x) | Every law so far injected max(thing, source), and `thing` is 1 on every seen morsel, poison included, so it ov
  #7   train 28.2 meals  holdout 25.4  (tuned 8x) | The map forgets everything more than 12 cells away, so ground the agent covered a while ago comes back as fres
  #16  train 28.1 meals  holdout  -    (tuned 3x) | Turning has a price the wave does not know: the body turns 0.12 rad per tick and crawls while more than a radi
  #9   train 27.3 meals  holdout 25.8  (tuned 5x) | Rooms behind doors are stocked larders, but the frontier marks only unknown cells touching known-free ground, 
  #5   train 26.6 meals  holdout 28.6  (tuned 5x) | Food is not uniformly spread: a morsel with other food around it is worth more, because the next meal is then 
  #4   train 26.1 meals  holdout 24.4  (tuned 5x) | Exploration is where meals come from, but every frontier mark currently weighs the same 0.12 whether it opens 
  #14  train 25.7 meals  holdout  -    (tuned 5x) | The theory's strongest negative result is that a FASTER wave (#8) cost 6 meals, which suggests the lag of a sl
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 70x: train 28.2 meals, holdout 26.6
```

> 16:00 model job `deep` started

---

## Turn 14 (DEEP) -> island #17

**Hypothesis:** Every trap this agent can fall into — a smell whose bearing flips between two ways round a wall, a smell from behind stone, a wall pressed for nothing — looks the same from the inside: the body stays within a few cells for a long time. The law counts its own visits (a slowly fading per-cell tally under the agent); when the visit density around the agent is high a FRUSTRATION level rises, floods the map and fades over ~100 thoughts, and while it lasts smell-only lures are muted (seen morsels never are), so the frontier and sight carry the agent away from the trap and it comes back to smell only once it is elsewhere. This is self-monitoring rather than per-lure boredom (#6), and it leaves norm

Prediction 30.5, got **30.5** meals, fitness 558, per world [37, 20, 23, 33, 21, 46, 31, 32, 23, 23, 31, 23, 35, 28, 27, 39, 38, 31, 37, 32]; model 140.1 s, repairs 0; vocabulary twin of #15, #16

value on the test map: max 0.96, mean 0.51, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Noticing you are stuck. Every trap seen so far (two-way smell bearings, a smell
# behind stone, a wall pressed for nothing, a pinch) looks the same from inside:
# the body stays in a few cells for a long time. Visits pile up where the agent
# stands; when the visit density around it is high, a FRUSTRATION level rises,
# floods the map and fades slowly, and while it lasts smell-only lures are muted
# (seen morsels are not), so frontier and sight carry the agent out of the trap.
# Base: #15 (appetite, aversion, ballistic, doors, body-honest diagonals).
channel source
channel solid
channel free
channel scent
channel thing
channel bad
channel here
channel value
channel confinement
channel trail
channel behind
channel door
channel aversion
channel injection
channel open
channel pinched
channel visits
channel loiter
channel frust
rule calc_conf:
  when 1
  set confinement = sum8(solid)
rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)
rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail
# Time spent on each cell, slowly forgotten; and its 3x3 sum.
rule count_visits:
  when 1
  set visits = clamp(0.998 * visits + 0.02 * here, 0, 10)
  set loiter = visits + sum8(visits)
# Rises when the agent loiters, spreads over the map, fades over ~100 thoughts.
rule frustration:
  when 1
  set frust = clamp(max(here * (loiter - 2) / 3, 0.999 * frust, 0.999 * max8(frust)), 0, 1)
rule find_doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * clamp(sum4(free), 0, 1)
rule fear:
  when 1
  set aversion = max(bad, 0.7 * max8(aversion))
rule clearance:
  when 1
  set open = 1 - clamp(solid, 0, 1)
rule pinch:
  when 1
  set pinched = max(shift(here, 1, 1) * (1 - min(shift(open, 1, 0), shift(open, 0, 1))), shift(here, -1, 1) * (1 - min(shift(open, -1, 0), shift(open, 0, 1))), shift(here, 1, -1) * (1 - min(shift(open, 1, 0), shift(open, 0, -1))), shift(here, -1, -1) * (1 - min(shift(open, -1, 0), shift(open, 0, -1))))
# Seen morsels pull fully; smell-only lures are muted by frustration.
rule inject:
  when 1
  set injection = max(where(max(thing, scent) > 0, source * where(thing > 0.5, 1, 1 - frust), source * exp(-40 * behind)) / (1 + 0.5 * confinement), 0.3 * door * exp(-40 * behind)) * (1 - aversion)
rule wave:
  when 1
  set value = max(injection, 0.94 * max4(value), 0.916214 * max(shift(value, 1, 1) * min(shift(open, 1, 0), shift(open, 0, 1)), shift(value, -1, 1) * min(shift(open, -1, 0), shift(open, 0, 1)), shift(value, 1, -1) * min(shift(open, 1, 0), shift(open, 0, -1)), shift(value, -1, -1) * min(shift(open, -1, 0), shift(open, 0, -1)))) * (1 - solid * (1 - leak)) * (1 - 0.5 * aversion) * (1 - pinched)
```
