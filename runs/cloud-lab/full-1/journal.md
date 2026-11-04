# night lab — 2026-11-04T11:14:39

Model `agent-in-session (mailbox)` | 12.0 h | world default | train [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] | holdout [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] | 8000 ticks/life | tune 5 | deep every 5 | theory every 6 | observe every 8

- anchor #0 `laws/atlas_wave.nomos`: train 14.8 meals [28, 27, 10, 35, 13, 4, 5, 5, 33, 12, 5, 7, 23, 23, 18, 8, 0, 28, 5, 6], holdout 19.6

- anchor #1 `laws/atlas_confine.nomos`: train 25.7 meals [37, 39, 18, 30, 23, 22, 13, 34, 26, 13, 33, 36, 33, 24, 29, 30, 0, 34, 14, 26], holdout 25.7

- anchor #2 `laws/atlas_ghost.nomos`: train 19.2 meals [28, 5, 25, 11, 13, 18, 16, 33, 29, 12, 4, 24, 21, 32, 16, 26, 0, 28, 23, 20], holdout 18.6

> 11:20 model job `design` started

---

## Turn 0 -> island #3

**Hypothesis:** Every frontier mark is worth the same 0.12 today, so a one-cell unknown pocket beside a wall pulls as hard as the edge of a large unexplored region. Weighting each frontier mark by the smoothed amount of unknown around it (information gain) sends the agent toward big unseen areas where new food is likely, while smell and seen morsels keep their value and the confinement suppression is kept.

Prediction 27.0, got **24.2** meals, fitness 472, per world [21, 26, 25, 5, 14, 29, 4, 32, 25, 28, 32, 33, 25, 35, 28, 37, 0, 37, 29, 20]; model 42.1 s, repairs 0

value on the test map: max 0.30, mean 0.16, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel value
channel dark
channel front
channel injection

# how much unknown lies around a cell: a smoothed field of not-yet-known cells
rule darkness:
  when 1
  set dark = 0.5 * (1 - free) * (1 - solid) + 0.5 * avg8(dark)

# the frontier part of source: faint marks where no smell and no morsel are painted
rule split:
  when 1
  set front = where(scent + thing < 0.05, source, 0)

# food counts as told; a frontier counts by the darkness behind it; tight places count less
rule inject:
  when 1
  set injection = max(source - front, thing * (1 - bad), front * (0.3 + 2.0 * dark)) / (1 + 0.5 * sum8(solid))

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 11:31 model job `design` started

---

## Turn 1 -> island #4

**Hypothesis:** A reactive wave lets the agent step back onto the cell it just left whenever two targets are nearly equal, which wastes ticks in back-and-forth limit cycles. A fading footprint channel (trail = max(here, decay*trail), scrolling with the map) dims what the agent READS on its own recent path while the plan wave itself stays untouched, so it commits forward and sweeps instead of dithering, but can still return when the way back is clearly better.

Prediction 27.0, got **25.6** meals, fitness 509, per world [22, 41, 20, 26, 14, 20, 29, 34, 29, 33, 23, 38, 17, 28, 30, 37, 0, 31, 14, 25]; model 30.1 s, repairs 0; vocabulary twin of #3

value on the test map: max 0.30, mean 0.16, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel here
channel value
channel dark
channel front
channel injection
channel wave
channel trail

# where the agent has recently stood (persists and scrolls with the map, fades)
rule track:
  when 1
  set trail = max(here, 0.97 * trail)

rule darkness:
  when 1
  set dark = 0.5 * (1 - free) * (1 - solid) + 0.5 * avg8(dark)

rule split:
  when 1
  set front = where(scent + thing < 0.05, source, 0)

rule inject:
  when 1
  set injection = max(source - front, thing * (1 - bad), front * (0.3 + 2.0 * dark)) / (1 + 0.5 * sum8(solid))

# the plan itself: an ordinary Bellman wave in its own channel
rule plan:
  when 1
  set wave = max(injection, max(0.94 * max4(wave), 0.916214 * max8(wave))) * (1 - solid * (1 - leak))

# what the agent reads: the plan, dimmed on its own fresh footprints (no turning back)
rule read:
  when 1
  set value = wave * (1 - 0.4 * trail)
```

### Holdout checkpoint at 0.54 h

```
  #3   train 25.8 meals  holdout 29.3  (tuned 5x) | Every frontier mark is worth the same 0.12 today, so a one-cell unknown pocket beside a wall pulls as hard as 
  #4   train 25.9 meals  holdout 30.0  (tuned 4x) | A reactive wave lets the agent step back onto the cell it just left whenever two targets are nearly equal, whi
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 9x: train 25.7 meals, holdout 25.7
```

> 11:47 model job `design` started

---

## Turn 2 -> island #5

**Hypothesis:** One discount serves two different promises: food (certain, worth a detour only when near) and the frontier (uncertain, but only informative if the agent commits to a long walk). Splitting the wave into an exploitation wave with a steep discount and an exploration wave with a shallow discount, read as their max, lets far frontiers keep a coherent long-range pull while nearby food still wins when close.

Prediction 27.0, got **21.6** meals, fitness 418, per world [25, 20, 19, 11, 32, 24, 22, 21, 15, 33, 21, 26, 19, 26, 27, 22, 0, 33, 16, 19]; model 28.1 s, repairs 0; vocabulary twin of #4

value on the test map: max 0.30, mean 0.22, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel here
channel value
channel dark
channel front
channel food
channel explore
channel trail

rule track:
  when 1
  set trail = max(here, 0.97 * trail)

rule darkness:
  when 1
  set dark = 0.5 * (1 - free) * (1 - solid) + 0.5 * avg8(dark)

rule split:
  when 1
  set front = where(scent + thing < 0.05, source, 0)

# exploitation wave: smell and morsels, steep discount (food is a near promise)
rule eat:
  when 1
  set food = max(max(source - front, thing * (1 - bad)) / (1 + 0.5 * sum8(solid)), max(0.9 * max4(food), 0.86 * max8(food))) * (1 - solid * (1 - leak))

# exploration wave: frontier weighted by darkness, shallow discount (worth a long walk)
rule roam:
  when 1
  set explore = max(front * (0.3 + 2.0 * dark) / (1 + 0.5 * sum8(solid)), max(0.97 * max4(explore), 0.958 * max8(explore))) * (1 - solid * (1 - leak))

# read the stronger promise, dimmed on fresh footprints
rule read:
  when 1
  set value = max(food, explore) * (1 - 0.4 * trail)
```

> 12:04 model job `design` started

---

## Turn 3 -> island #6

**Hypothesis:** When frontiers on several sides look alike, the agent's target can switch from thought to thought and it sweeps nothing completely. A frozen random preference per place (seeded once, inherited by cells scrolling in, smoothed into regions, and carried with the map) breaks such ties the same way every thought, so the agent commits to one side of the unknown and explores it through; the frontier/food split is also computed inline now, removing a one-pass lag that injected a stale full-strength peak.

Prediction 27.0, got **24.4** meals, fitness 486, per world [22, 24, 17, 33, 33, 32, 4, 30, 32, 25, 26, 37, 23, 31, 29, 33, 0, 20, 15, 21]; model 130.1 s, repairs 0

value on the test map: max 0.15, mean 0.08, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel here
channel value
channel dark
channel injection
channel wave
channel trail
channel whim: init uniform(0.01, 1)
channel taste

rule track:
  when 1
  set trail = max(here, 0.97 * trail)

# a frozen random preference per place; cells scrolling in inherit a random neighbour's
rule seed_whim:
  when whim <= 0
  set whim = clamp(argmax8(rand() * (whim > 0), whim) + uniform(-0.05, 0.05), 0.01, 1)

# smoothed into regions: one side of the map is simply preferred, and stays preferred
rule smooth_whim:
  when 1
  set taste = 0.5 * whim + 0.5 * avg8(taste)

rule darkness:
  when 1
  set dark = 0.5 * (1 - free) * (1 - solid) + 0.5 * avg8(dark)

# frontier (faint marks, no smell or morsel) pull = darkness behind it times the region's standing preference
rule inject:
  when 1
  set injection = max(where(scent + thing < 0.05, 0, source), thing * (1 - bad), where(scent + thing < 0.05, source, 0) * (0.3 + 2.0 * dark) * (0.5 + taste)) / (1 + 0.5 * sum8(solid))

rule plan:
  when 1
  set wave = max(injection, max(0.94 * max4(wave), 0.916214 * max8(wave))) * (1 - solid * (1 - leak))

rule read:
  when 1
  set value = wave * (1 - 0.4 * trail)
```

> 12:09 model job `observe` started

---

## Observer: #3 on world 16

![](observe/obs00_island3_world16.png)

`{'meals': 0, 'ticks': 3747, 'alive': False, 'distance': 283.6, 'cells_visited': 60, 'last500_box': '4x12', 'doors': 3, 'breaks': 5, 'idle_presses': 8, 'nightcaps': 0}`

The agent spends its whole life inside one small walled pocket (a 4x12 box, start and death in the same spot), sweeping back and forth along it and never leaving, so it never even reaches the food lying a short walk away in the neighbouring rooms.
It does press and break things (3 doors, 5 breaks), but after each press it drifts back into the pocket instead of walking through, so the openings it makes do not turn into a route out.
The failure is that inside a sealed pocket every boundary cell is known solid, so there are no frontier marks (unknown cells touching known-free ones) and the value field has nothing to point at except leak through walls, which is weak and spread evenly around the whole box.
A fix: give solid cells that have unknown space behind them (a wall with darkness on its far side, especially one whose leak was taught high by a door opening) a source of their own when no frontier exists, so the wave leads the agent into one specific wall and through it, and keep that target (a lingering source) until it is passed.

### Holdout checkpoint at 1.09 h

```
  #3   train 26.7 meals  holdout 27.3  (tuned 6x) | Every frontier mark is worth the same 0.12 today, so a one-cell unknown pocket beside a wall pulls as hard as 
  #6   train 25.1 meals  holdout 24.9  (tuned 3x) | When frontiers on several sides look alike, the agent's target can switch from thought to thought and it sweep
  #4   train 25.9 meals  holdout 30.0  (tuned 5x) | A reactive wave lets the agent step back onto the cell it just left whenever two targets are nearly equal, whi
  #5   train 21.6 meals  holdout  -    (tuned 5x) | One discount serves two different promises: food (certain, worth a detour only when near) and the frontier (un
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 18x: train 25.7 meals, holdout 25.7
```

> 12:26 model job `deep` started

---

## Turn 4 (DEEP) -> island #7

**Hypothesis:** Exploration is driven only by frontier marks, which exist only where unknown ground touches known-free ground; once the agent has seen every wall of a room, cave or pocket, the value field has nothing to aim at and it sweeps the box until it starves (the observer's world-16 life). A BREACH source on every reachable wall cell with darkness behind it, weighted by the darkness and passed through the wall's taught leak, is weak enough that any food or frontier beats it in the max, but when nothing else calls it leads the agent into the most promising wall, where pressing opens doors and chips rubble; stone that never gives is taught a low leak and drops out, so the agent tries walls in turn. The

Prediction 27.5, got **23.2** meals, fitness 460, per world [26, 30, 27, 34, 40, 23, 4, 28, 31, 26, 30, 34, 11, 19, 2, 33, 0, 22, 24, 20]; model 48.1 s, repairs 0; vocabulary twin of #4

value on the test map: max 0.30, mean 0.16, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel here
channel leak
channel value
channel dark
channel breach
channel injection
channel wave
channel trail

rule track:
  when 1
  set trail = max(here, 0.97 * trail)

rule darkness:
  when 1
  set dark = 0.5 * (1 - free) * (1 - solid) + 0.5 * avg8(dark)

# a wall the agent can reach with darkness behind it is a way out, weighted by how it gave before
rule breach_mark:
  when 1
  set breach = where((solid > 0.5) * (max8(free) > 0.5), dark * leak, 0)

# food as told, frontier by darkness, breaches as a weak fallback the max lets win only when nothing else calls
rule inject:
  when 1
  set injection = max(max(where(scent + thing < 0.05, 0, source), thing * (1 - bad), where(scent + thing < 0.05, source, 0) * (0.3 + 2.0 * dark)) / (1 + 0.5 * sum8(solid)), 0.1 * breach)

rule plan:
  when 1
  set wave = max(injection, max(0.94 * max4(wave), 0.916214 * max8(wave))) * (1 - solid * (1 - leak))

rule read:
  when 1
  set value = wave * (1 - 0.4 * trail)
```

> 12:41 model job `design` started

---

## Turn 5 -> island #8

**Hypothesis:** Food in this world grows on fixed spots that regrow on a timer, yet every law so far forgets a spot as soon as its sighting fades, so the agent re-finds food only by luck. A long LARDER memory of where good morsels were seen, multiplied by a RIPENESS clock that is reset whenever the agent stands on or beside the place and climbs slowly back, injects a source that calls the agent back to a patch about when it should have regrown: a trapline over the remembered map, on top of the darkness-weighted frontier and the footprint trail.

Prediction 27.5, got **24.6** meals, fitness 491, per world [26, 30, 27, 34, 40, 23, 4, 28, 31, 26, 30, 34, 11, 27, 23, 33, 0, 22, 24, 20]; model 40.1 s, repairs 0; vocabulary twin of #7, #4

value on the test map: max 0.30, mean 0.16, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel here
channel value
channel dark
channel larder
channel ripe
channel injection
channel wave
channel trail

rule track:
  when 1
  set trail = max(here, 0.97 * trail)

rule darkness:
  when 1
  set dark = 0.5 * (1 - free) * (1 - solid) + 0.5 * avg8(dark)

# where good food has been seen: a long memory that outlives the fading sighting
rule remember:
  when 1
  set larder = max(thing * (1 - bad), 0.9998 * larder)

# a remembered place ripens with time since the agent last stood on or beside it
rule ripen:
  when 1
  set ripe = where(max(here, max8(here)) > 0, 0, min(1, ripe + 0.0004))

# food and frontier as before; a ripe remembered patch calls the agent back (a trapline)
rule inject:
  when 1
  set injection = max(where(scent + thing < 0.05, 0, source), thing * (1 - bad), where(scent + thing < 0.05, source, 0) * (0.3 + 2.0 * dark), 0.5 * larder * ripe) / (1 + 0.5 * sum8(solid))

rule plan:
  when 1
  set wave = max(injection, max(0.94 * max4(wave), 0.916214 * max8(wave))) * (1 - solid * (1 - leak))

rule read:
  when 1
  set value = wave * (1 - 0.4 * trail)
```

> 12:45 model job `theory` started

---

## Theory (after 6 design turns)

1. Exploration is the lever; frontier quality matters. Darkness-weighted frontier (#3) is the train leader (26.7 tuned, holdout 27.3) over plain confinement (#1, 25.7/25.7).
2. Do not walk back. The footprint trail (#4) is the best holdout (30.0) with equal train (25.9); it rescued worlds 3 and 6 (5->26, 4->29), the dithering worlds.
3. Splitting food and frontier into two waves with different discounts hurts (#5, 21.6): one shared discount keeps food and frontier comparable.
4. Extra pull terms layered on top so far cost meals untuned: random regional preference (#6, 24.4, holdout 24.9), wall breach (#7, 23.2 raw, 26.2 tuned, world 16 still 0), larder/trapline (#8, 24.6 raw, untuned). Each new source competes with the frontier and distracts.
5. Per-world scores swing 2 <-> 28 between near-identical laws: one world is not evidence; trust holdout.
6. World 16 (sealed pocket) starves every law, breach included.

### Holdout checkpoint at 1.62 h

```
  #3   train 26.7 meals  holdout 27.3  (tuned 6x) | Every frontier mark is worth the same 0.12 today, so a one-cell unknown pocket beside a wall pulls as hard as 
  #7   train 26.2 meals  holdout 26.4  (tuned 5x) | Exploration is driven only by frontier marks, which exist only where unknown ground touches known-free ground;
  #6   train 25.1 meals  holdout 24.9  (tuned 7x) | When frontiers on several sides look alike, the agent's target can switch from thought to thought and it sweep
  #4   train 25.9 meals  holdout 30.0  (tuned 5x) | A reactive wave lets the agent step back onto the cell it just left whenever two targets are nearly equal, whi
  #8   train 24.6 meals  holdout  -    (tuned 1x) | Food in this world grows on fixed spots that regrow on a timer, yet every law so far forgets a spot as soon as
  #5   train 21.6 meals  holdout  -    (tuned 5x) | One discount serves two different promises: food (certain, worth a detour only when near) and the frontier (un
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 29x: train 25.7 meals, holdout 25.7
```

> 13:01 model job `design` started

---

## Turn 6 -> island #9

**Hypothesis:** My theory says extra competing pulls distract; modulating the frontier helps (darkness weighting, footprint trail). The smell is painted at a single point along a bearing that flips around walls, so following it directly is fragile, but it says roughly where food is. A broad slowly fading AROMA halo around recent smell points multiplies frontier marks inside it, so exploration is biased toward unknown ground in the smelled direction (finding the actual path around the wall) instead of toward any dark region, with no new competing source.

Prediction 27.0, got **25.9** meals, fitness 521, per world [37, 29, 23, 33, 15, 26, 25, 20, 36, 23, 27, 34, 22, 36, 26, 28, 0, 26, 19, 32]; model 46.1 s, repairs 0; vocabulary twin of #7, #4, #8

value on the test map: max 0.30, mean 0.16, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel here
channel value
channel dark
channel aroma
channel injection
channel wave
channel trail

rule track:
  when 1
  set trail = max(here, 0.97 * trail)

rule darkness:
  when 1
  set dark = 0.5 * (1 - free) * (1 - solid) + 0.5 * avg8(dark)

# a broad, slowly fading halo around where the nose has pointed (crosses walls: it is a belief, not a path)
rule halo:
  when 1
  set aroma = max(scent, 0.85 * max8(aroma), 0.95 * aroma)

# frontier marks are worth more inside the halo: explore toward what is smelled, not just what is dark
rule inject:
  when 1
  set injection = max(where(scent + thing < 0.05, 0, source), thing * (1 - bad), where(scent + thing < 0.05, source, 0) * (0.3 + 2.0 * dark) * (1 + 3.0 * aroma)) / (1 + 0.5 * sum8(solid))

rule plan:
  when 1
  set wave = max(injection, max(0.94 * max4(wave), 0.916214 * max8(wave))) * (1 - solid * (1 - leak))

rule read:
  when 1
  set value = wave * (1 - 0.4 * trail)
```

> 13:18 model job `design` started

---

## Turn 7 -> island #10

**Hypothesis:** Every law so far is a max-wave, which follows the single best target and ignores how much is there: one frontier cell and a whole dark region, one berry and a patch of five pull alike once discounted. Replacing the race with a RELAXATION (each free cell becomes the mean of its neighbours, sources fixed, zero at the outer ring and in walls) gives a harmonic potential with no local maxima away from sources, where many sources in one direction add up, so the agent heads for where most food and most unknown lie and cannot be trapped between two equal targets.

Prediction 22.0, got **14.2** meals, fitness 298, per world [11, 14, 12, 10, 6, 18, 15, 21, 15, 13, 0, 14, 17, 34, 18, 17, 0, 24, 16, 10]; model 34.0 s, repairs 0

value on the test map: max 1.00, mean 0.04, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel thing
channel bad
channel value
channel inflow

# what calls: senses as painted, plus seen good morsels; hemmed-in sources count less
rule inject:
  when 1
  set inflow = max(source / (1 + 0.5 * sum8(solid)), thing * (1 - bad))

# a pressure field, not a race: each cell relaxes to the MEAN of its neighbours (bounded by
# the sources, zero at the outer ring), so many sources in one direction pull harder than one
rule relax:
  when 1
  set value = max(inflow, 0.6 * avg8(value) + 0.4 * avg4(value)) * (1 - solid * (1 - leak))
```

### Holdout checkpoint at 2.13 h

```
  #3   train 26.7 meals  holdout 27.3  (tuned 6x) | Every frontier mark is worth the same 0.12 today, so a one-cell unknown pocket beside a wall pulls as hard as 
  #7   train 26.2 meals  holdout 26.4  (tuned 5x) | Exploration is driven only by frontier marks, which exist only where unknown ground touches known-free ground;
  #6   train 25.1 meals  holdout 24.9  (tuned 7x) | When frontiers on several sides look alike, the agent's target can switch from thought to thought and it sweep
  #9   train 25.9 meals  holdout  -    (tuned 5x) | My theory says extra competing pulls distract; modulating the frontier helps (darkness weighting, footprint tr
  #4   train 25.9 meals  holdout 30.0  (tuned 5x) | A reactive wave lets the agent step back onto the cell it just left whenever two targets are nearly equal, whi
  #8   train 24.6 meals  holdout  -    (tuned 5x) | Food in this world grows on fixed spots that regrow on a timer, yet every law so far forgets a spot as soon as
  #5   train 21.6 meals  holdout  -    (tuned 5x) | One discount serves two different promises: food (certain, worth a detour only when near) and the frontier (un
  #10  train 14.2 meals  holdout  -    (tuned 1x) | Every law so far is a max-wave, which follows the single best target and ignores how much is there: one fronti
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 39x: train 25.7 meals, holdout 25.7
```

> 13:28 model job `design` started

---

## Turn 8 -> island #11

**Hypothesis:** Exploration so far is local and isotropic: the darkness-weighted frontier wave keeps the agent re-sweeping the same neighbourhood, while food regrows on fixed spots spread over the whole world. A slowly turning world-fixed COMPASS (a life clock carried by the map, and each cell's projection on a heading that rotates once every few thousand ticks) multiplies frontier marks that lie ahead, so the agent makes long straight excursions into new land and comes round to every side over a life, while food and smell are untouched and still win when present.

Prediction 27.0, got **21.4** meals, fitness 414, per world [27, 17, 15, 17, 31, 26, 31, 23, 25, 23, 1, 18, 24, 18, 26, 17, 0, 34, 26, 29]; model 40.1 s, repairs 0

value on the test map: max 0.39, mean 0.21, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel here
channel value
channel dark
channel clock
channel compass
channel injection
channel wave
channel trail

rule track:
  when 1
  set trail = max(here, 0.97 * trail)

rule darkness:
  when 1
  set dark = 0.5 * (1 - free) * (1 - solid) + 0.5 * avg8(dark)

# a life-long clock, shared by the whole map (cells scrolling in take their neighbours' time)
rule tick:
  when 1
  set clock = max(clock, max8(clock)) + 1

# a slowly turning world-fixed heading: how far each cell lies along it from the agent
rule turn:
  when 1
  set compass = clamp(1 + 2.0 * ((posx() - 0.5) * cos(0.0008 * clock) + (posy() - 0.5) * sin(0.0008 * clock)), 0.1, 3)

# frontier marks ahead along the heading count more: long straight excursions into new land
rule inject:
  when 1
  set injection = max(where(scent + thing < 0.05, 0, source), thing * (1 - bad), where(scent + thing < 0.05, source, 0) * (0.3 + 2.0 * dark) * compass) / (1 + 0.5 * sum8(solid))

rule plan:
  when 1
  set wave = max(injection, max(0.94 * max4(wave), 0.916214 * max8(wave))) * (1 - solid * (1 - leak))

rule read:
  when 1
  set value = wave * (1 - 0.4 * trail)
```

> 13:43 model job `deep` started

---

## Turn 9 (DEEP) -> island #12

**Hypothesis:** The map forgets a wall at the same rate as everything else, so a wall seen a few hundred ticks ago becomes half transparent to the wave and then "unknown" again: the agent is led through stale walls into presses that do nothing and back to re-inspect stone as if it were frontier. A MASONRY memory keeps walls nearly permanent (cleared only where the cell is seen free, so opened doors and chipped rubble disappear), and confinement, darkness, frontier and the wave's passability all read it, while free ground and food are still forgotten so regrowth gets revisited.

Prediction 27.0, got **25.6** meals, fitness 499, per world [18, 27, 33, 30, 24, 38, 25, 36, 40, 10, 23, 33, 23, 26, 26, 27, 0, 19, 16, 38]; model 24.1 s, repairs 0; vocabulary twin of #9, #7, #4, #8

value on the test map: max 0.30, mean 0.16, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel here
channel value
channel wall
channel dark
channel injection
channel wave
channel trail

# walls are permanent: keep a wall far longer than the fading map does, until the cell is seen free (an opened door)
rule masonry:
  when 1
  set wall = max(solid, 0.9995 * wall) * (1 - free)

rule track:
  when 1
  set trail = max(here, 0.97 * trail)

rule darkness:
  when 1
  set dark = 0.5 * (1 - free) * (1 - wall) + 0.5 * avg8(dark)

# confinement and passability read the remembered walls, not the fading evidence
rule inject:
  when 1
  set injection = max(where(scent + thing < 0.05, 0, source), thing * (1 - bad), where(scent + thing < 0.05, source, 0) * (0.3 + 2.0 * dark) * (1 - wall)) / (1 + 0.5 * sum8(wall))

rule plan:
  when 1
  set wave = max(injection, max(0.94 * max4(wave), 0.916214 * max8(wave))) * (1 - wall * (1 - leak))

rule read:
  when 1
  set value = wave * (1 - 0.4 * trail)
```

### Holdout checkpoint at 2.66 h

```
  #3   train 26.7 meals  holdout 27.3  (tuned 7x) | Every frontier mark is worth the same 0.12 today, so a one-cell unknown pocket beside a wall pulls as hard as 
  #7   train 26.2 meals  holdout 26.4  (tuned 5x) | Exploration is driven only by frontier marks, which exist only where unknown ground touches known-free ground;
  #6   train 25.1 meals  holdout 24.9  (tuned 7x) | When frontiers on several sides look alike, the agent's target can switch from thought to thought and it sweep
  #9   train 25.9 meals  holdout  -    (tuned 5x) | My theory says extra competing pulls distract; modulating the frontier helps (darkness weighting, footprint tr
  #4   train 25.9 meals  holdout 30.0  (tuned 5x) | A reactive wave lets the agent step back onto the cell it just left whenever two targets are nearly equal, whi
  #12  train 25.6 meals  holdout  -    (tuned 3x) | The map forgets a wall at the same rate as everything else, so a wall seen a few hundred ticks ago becomes hal
  #8   train 24.6 meals  holdout  -    (tuned 5x) | Food in this world grows on fixed spots that regrow on a timer, yet every law so far forgets a spot as soon as
  #5   train 21.6 meals  holdout  -    (tuned 5x) | One discount serves two different promises: food (certain, worth a detour only when near) and the frontier (un
  #0   train 14.8 meals  holdout 19.6  (hand-written) | the reference Bellman wave
  #1   train 25.7 meals  holdout 25.7  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 19.2 meals  holdout 18.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 51x: train 27.3 meals, holdout 25.4
```

> 14:00 model job `design` started

---

## Turn 10 -> island #13

**Hypothesis:** The smell is repainted from scratch every thought at a waypoint whose bearing flips between the two ways round a wall, so food-directed walks can dither even with the footprint trail; the plain ghost helped the bare wave by +4 meals but was never tried on top of darkness-weighted exploration. Here only FOOD evidence (smell and seen morsels) lingers in its own NOSE channel, erased when the agent reaches the spot or when it is marked bad, while frontier marks stay instantaneous, so a food target once chosen is held through the flip but exploration still follows the live map.

Prediction 27.0, got **21.5** meals, fitness 424, per world [36, 27, 16, 29, 4, 19, 21, 34, 29, 13, 15, 27, 23, 23, 23, 20, 0, 31, 28, 12]; model 38.1 s, repairs 0; vocabulary twin of #12, #9, #7, #4, #8

value on the test map: max 0.30, mean 0.16, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel here
channel value
channel dark
channel nose
channel injection
channel wave
channel trail

rule track:
  when 1
  set trail = max(here, 0.97 * trail)

rule darkness:
  when 1
  set dark = 0.5 * (1 - free) * (1 - solid) + 0.5 * avg8(dark)

# smell and seen morsels LINGER where they were painted (a flipping bearing keeps its first target);
# reaching the place, or learning it is bad, erases the ghost
rule linger:
  when 1
  set nose = max(scent, thing * (1 - bad), 0.99 * nose) * (1 - max(here, max8(here))) * (1 - bad)

# frontier by darkness as before; food is the lingering nose, so a target once chosen is held
rule inject:
  when 1
  set injection = max(nose, where(scent + thing < 0.05, source, 0) * (0.3 + 2.0 * dark)) / (1 + 0.5 * sum8(solid))

rule plan:
  when 1
  set wave = max(injection, max(0.94 * max4(wave), 0.916214 * max8(wave))) * (1 - solid * (1 - leak))

rule read:
  when 1
  set value = wave * (1 - 0.4 * trail)
```

> 14:14 model job `design` started
