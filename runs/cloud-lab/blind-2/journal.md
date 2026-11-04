# night lab — 2026-11-04T12:47:58

Model `agent-in-session (mailbox)` | 12.0 h | world default | train [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119] | holdout [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135] | 8000 ticks/life | tune 5 | deep every 5 | theory every 6 | observe every 0

- anchor #0 `laws/atlas_wave.nomos`: train 20.4 meals [21, 36, 36, 18, 25, 34, 20, 12, 2, 26, 13, 15, 28, 35, 33, 10, 21, 18, 0, 5], holdout 15.5

- anchor #1 `laws/atlas_confine.nomos`: train 23.2 meals [20, 30, 20, 31, 25, 35, 45, 25, 17, 30, 0, 18, 21, 9, 44, 26, 3, 37, 0, 28], holdout 23.6

- anchor #2 `laws/atlas_ghost.nomos`: train 22.7 meals [21, 27, 29, 40, 21, 7, 14, 5, 12, 23, 16, 27, 28, 23, 34, 12, 12, 30, 25, 48], holdout 17.9

> 12:53 model job `design` started

---

## Turn 0 -> island #3

**Hypothesis:** Meals come mostly from exploration, but every frontier mark is worth the same 0.12, so a crack between two walls pulls as hard as the edge of a big unexplored plain. A leaky blur of the unknown (`vast`) weights each frontier mark by how much unexplored space lies behind it (information gain), on top of confinement suppression; the agent should open new ground faster and waste fewer trips into pockets.

Prediction 24.5, got **28.1** meals, fitness 544, per world [41, 35, 40, 46, 34, 40, 30, 35, 17, 30, 0, 32, 21, 26, 37, 30, 34, 8, 0, 25]; model 56.1 s, repairs 0

value on the test map: max 0.50, mean 0.26, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel value
channel unknown
channel vast
channel confinement
channel injection

# what is neither known solid nor known free
rule unknown:
  when 1
  set unknown = (1 - free) * (1 - solid)

# a leaky blur of the unknown: how much unexplored space lies around a cell
rule vast:
  when 1
  set vast = 0.3 * unknown + 0.7 * avg8(vast)

# a source hemmed in by solid counts less (night 1's champion)
rule confine:
  when 1
  set confinement = sum8(solid)

# frontier marks are worth more where they open onto a large unknown; food and smell untouched
rule inject:
  when 1
  set injection = max(thing, source * where(thing + scent > 0, 1, 0.5 + 1.5 * vast)) / (1 + 0.5 * confinement)

# the Bellman wave from the injection
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 13:08 model job `design` started

---

## Turn 1 -> island #4

**Hypothesis:** An agent that finds nothing worth walking to stands still or rocks between two cells and starves (worlds where every law eats 0 look like that); the wave has no notion of being stuck. A `dwell` channel counts time spent under the agent, its broadcast `frust` turns known wall faces (solid touching known free, weighted by their learned leak) into targets, so a stuck agent goes pressing walls and opens doors/rubble out of a sealed or exhausted region instead of waiting.

Prediction 28.5, got **27.3** meals, fitness 538, per world [28, 17, 38, 46, 30, 44, 36, 35, 34, 30, 32, 36, 16, 15, 37, 20, 32, 8, 4, 8]; model 46.1 s, repairs 0; vocabulary twin of #3

value on the test map: max 0.50, mean 0.26, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel injection

# what is neither known solid nor known free; its blur = unexplored space around a cell
rule unknown:
  when 1
  set unknown = (1 - free) * (1 - solid)

rule vast:
  when 1
  set vast = 0.3 * unknown + 0.7 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

# time spent on a cell grows under the agent and fades elsewhere (scrolls with the map)
rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005), 0.98 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1, 0.5 + 1.5 * vast)) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

### Holdout checkpoint at 0.59 h

```
  #3   train 27.1 meals  holdout 21.9  (tuned 5x) | Meals come mostly from exploration, but every frontier mark is worth the same 0.12, so a crack between two wal
  #4   train 27.9 meals  holdout 27.5  (tuned 3x) | An agent that finds nothing worth walking to stands still or rocks between two cells and starves (worlds where
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 7x: train 28.2 meals, holdout 25.6
```

> 13:29 model job `design` started

---

## Turn 2 -> island #5

**Hypothesis:** The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to be free, so a smell painted far out in unexplored land, or a path that only exists if the unknown is open, pulls as hard as a route it knows. Making the wave dearer through unknown cells (trust in known ground) should favour reachable targets and the frontier at the edge of known space over far guesses through possible walls, cutting wasted walks into dead ends.

Prediction 28.0, got **30.3** meals, fitness 612, per world [29, 17, 25, 46, 26, 38, 25, 30, 43, 32, 32, 48, 0, 26, 40, 43, 23, 37, 24, 22]; model 34.1 s, repairs 0; vocabulary twin of #4, #3

value on the test map: max 0.08, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel injection

# what is neither known solid nor known free; its blur = unexplored space around a cell
rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 0.9921
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

# time spent on a cell grows under the agent and fades elsewhere (scrolls with the map)
rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, 0.4275 + 1.5 * vast)) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 13:45 model job `design` started

---

## Turn 3 -> island #6

**Hypothesis:** The smell is painted along the world's geodesic bearing, which flips between the two ways round a wall as the agent steps, a sensor-level limit cycle (night lab: the ghost fixed exactly those trap worlds). Here only the SMELL gets a memory: its painted spot lingers as a `ghost` after the bearing moves on and is erased where the agent stands, so the agent commits to the first way round instead of rocking; the frontier, which never flips, stays memoryless.

Prediction 31.0, got **27.2** meals, fitness 539, per world [25, 26, 38, 22, 33, 31, 24, 35, 27, 23, 37, 14, 33, 34, 24, 19, 14, 29, 31, 25]; model 22.1 s, repairs 0; vocabulary twin of #5, #4

value on the test map: max 0.08, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel ghost
channel injection

# what is neither known solid nor known free; its blur = unexplored space around a cell
rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 0.9921
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1.176
  set confinement = sum8(solid)

# time spent on a cell grows under the agent and fades elsewhere (scrolls with the map)
rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# the smell's painted spot lingers after the bearing moves on, erased where the agent stands
rule ghost:
  when 1
  set ghost = max(scent, 0.99 * ghost) * (1 - here) * (1 - max8(here))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, ghost, source * where(thing + scent > 0, 1.358, 0.4275 + 1.5 * vast)) / (1 + 0.4454 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 1.16 h

```
  #5   train 31.6 meals  holdout 26.7  (tuned 5x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #3   train 27.1 meals  holdout 21.9  (tuned 5x) | Meals come mostly from exploration, but every frontier mark is worth the same 0.12, so a crack between two wal
  #4   train 27.9 meals  holdout 27.5  (tuned 5x) | An agent that finds nothing worth walking to stands still or rocks between two cells and starves (worlds where
  #6   train 27.2 meals  holdout  -    (tuned 3x) | The smell is painted along the world's geodesic bearing, which flips between the two ways round a wall as the 
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 17x: train 28.2 meals, holdout 25.6
```

> 14:03 model job `deep` started

---

## Turn 4 (DEEP) -> island #7

**Hypothesis:** When two targets are about equally good the agent can turn back and forth between them, and a reactive wave has no reason to prefer going on; the ghost (#6) fixed rocking but by committing to the smell, which cost the rich worlds. Here the commitment is in the movement instead: a fading `trail` of where the agent has just been makes the wave dearer back over its own fresh tracks, so it keeps heading on, sweeps new ground and backtracks only when the target behind is clearly better.

Prediction 31.0, got **30.2** meals, fitness 601, per world [39, 16, 48, 26, 28, 38, 23, 32, 33, 33, 35, 28, 11, 26, 46, 25, 34, 34, 30, 20]; model 30.1 s, repairs 0; vocabulary twin of #5, #4, #6

value on the test map: max 0.08, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel trail
channel injection

# what is neither known solid nor known free; its blur = unexplored space around a cell
rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 0.9921
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1.176
  set confinement = sum8(solid)

# time spent on a cell grows under the agent and fades elsewhere (scrolls with the map)
rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# a fading trail of where the agent has just been (scrolls with the map)
rule trail:
  when 1
  set trail = max(here, 0.995 * trail)

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.358, 0.4275 + 1.5 * vast)) / (1 + 0.4454 * confinement), frust * solid * max8(free) * leak)

# the wave trusts known ground, and going back over the fresh trail costs extra (momentum)
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free)) * (1 - 0.3 * trail)) * (1 - solid * (1.028 - leak))
```

> 14:21 model job `design` started

---

## Turn 5 -> island #8

**Hypothesis:** In one Bellman wave every goal competes through max(), so the agent serves exactly one goal at a time and is indifferent between two routes to a meal even if one of them opens new ground; between two near-equal targets it can rock. Here food (morsels + smell) and exploration (frontier + frustrated wall faces) run as two separate discounted waves and the read-out ADDS them, so among routes to food the one that also explores wins, ties break toward new ground, and with no food in sight the agent just explores as before.

Prediction 31.0, got **26.6** meals, fitness 523, per world [17, 36, 32, 38, 27, 40, 31, 27, 17, 32, 23, 31, 4, 37, 35, 32, 21, 1, 21, 31]; model 56.1 s, repairs 0; vocabulary twin of #5, #4, #7, #6

value on the test map: max 0.04, mean 0.03, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel food
channel explore

rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

# unexplored space around a cell
rule vast:
  when 1
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

# time under the agent, broadcast as frustration
rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# FOOD wave: seen morsels and smell only, confinement-suppressed, trusting known ground
rule food:
  when 1
  set food = max(where(thing + scent > 0, max(thing, 1.358 * source), 0) / (1 + 0.4454 * confinement), max(0.94 * max4(food), 0.916214 * max8(food)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))

# EXPLORE wave: frontier weighted by the unknown behind it, plus wall faces when frustrated
rule explore:
  when 1
  set explore = max(where(thing + scent > 0, 0, source * (0.4275 + 1.5 * vast)) / (1 + 0.4454 * confinement), frust * solid * max8(free) * leak, max(0.94 * max4(explore), 0.916214 * max8(explore)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))

# goals ADD instead of competing: the way to food that also opens new ground wins
rule value:
  when 1
  set value = food + 0.5 * explore
```

> 14:25 model job `theory` started

---

## Theory (after 6 design turns)

Exploration is the main food source, and how the frontier is weighted matters most. Helps: weighting frontier marks by the unknown behind them (#3, +4.9 train, but holdout only 21.9, so partly luck); turning wall faces into targets when the agent dwells (#4, holdout 27.5, the best honest number; it lifted a 0-meal world to 32); making the wave dearer through unknown cells (#5, +3.2 train, holdout 26.7). Hurts: giving the smell a lingering memory (#6, -3 raw: it commits to the smell and loses the rich worlds, though no world starved); splitting food and exploration into two waves and adding them (#8, -5: goals must compete, not blend). Neutral: momentum via a trail penalty (#7, about equal, all 20 immortal). Holdout gaps of 1-2 meals are noise; worlds range 0-48. Untried: food clusters, revisiting regrowing spots, poison avoidance, distance-adaptive discount, door-specific planning.

### Holdout checkpoint at 1.69 h

```
  #5   train 31.6 meals  holdout 26.7  (tuned 5x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #7   train 30.2 meals  holdout 27.1  (tuned 5x) | When two targets are about equally good the agent can turn back and forth between them, and a reactive wave ha
  #3   train 27.1 meals  holdout 21.9  (tuned 5x) | Meals come mostly from exploration, but every frontier mark is worth the same 0.12, so a crack between two wal
  #4   train 27.9 meals  holdout 27.5  (tuned 5x) | An agent that finds nothing worth walking to stands still or rocks between two cells and starves (worlds where
  #6   train 27.2 meals  holdout  -    (tuned 5x) | The smell is painted along the world's geodesic bearing, which flips between the two ways round a wall as the 
  #8   train 26.6 meals  holdout  -    (tuned 1x) | In one Bellman wave every goal competes through max(), so the agent serves exactly one goal at a time and is i
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 26x: train 28.2 meals, holdout 25.6
```

> 14:40 model job `design` started

---

## Turn 6 -> island #9

**Hypothesis:** The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways round a wall, so as a target it traps the agent; yet the ablation says smell doubles the take on some worlds. Here smell stops being a target (only a weak direct pull is left) and becomes a COMPASS: a wall-blind halo around its painted spot multiplies the frontier marks that lie its way, so the agent always walks to real, reachable unknown ground, but chooses the unknown in the direction of the food.

Prediction 32.0, got **30.9** meals, fitness 615, per world [17, 26, 41, 29, 31, 30, 22, 34, 34, 33, 33, 39, 16, 35, 35, 35, 35, 33, 23, 37]; model 16.1 s, repairs 0; vocabulary twin of #5, #4, #7, #6, #8

value on the test map: max 0.08, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel compass
channel injection

rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 1
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# the smell as a broad halo around its painted spot, ignoring walls (a direction, not a place)
rule compass:
  when 1
  set compass = max(scent, 0.85 * max8(compass))

# smell no longer pulls on its own: it brightens the frontier that lies its way
rule inject:
  when 1
  set injection = max(max(thing, where(thing > 0, 1.358 * source, 0), 0.3 * scent, where(thing + scent > 0, 0, source * (0.4275 + 1.5 * vast) * (1 + 3 * compass))) / (1 + 0.4454 * confinement), frust * solid * max8(free) * leak)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 14:59 model job `design` started

---

## Turn 7 -> island #10

**Hypothesis:** The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its frontier choice can lead it back into ground it just emptied of food. A slowly fading `trail` of the walked path, blurred wide into `been`, discounts frontier marks on the side the agent came from, so exploration becomes a sweep that keeps moving away from its own past into fresh ground where uneaten food is.

Prediction 32.0, got **34.7** meals, fitness 684, per world [32, 29, 31, 32, 40, 40, 41, 44, 14, 40, 33, 34, 32, 34, 44, 27, 33, 39, 33, 42]; model 36.1 s, repairs 0; vocabulary twin of #7, #5, #4, #9, #6, #8

value on the test map: max 0.08, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel trail
channel been
channel injection

rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 1
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# the path walked, slowly fading; scrolls with the map
rule trail:
  when 1
  set trail = max(here, 0.998 * trail)

# a wide blur of the path: how much of the agent's own past lies around a cell
rule been:
  when 1
  set been = 0.2 * trail + 0.8 * avg8(been)

# frontier on the side the agent came from counts less: the sweep goes on away from its past
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.358, (0.4275 + 1.5 * vast) * (1 - min(1, 3 * been)))) / (1 + 0.4454 * confinement), frust * solid * max8(free) * leak)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 2.32 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 0x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #5   train 31.6 meals  holdout 26.7  (tuned 5x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #7   train 30.2 meals  holdout 27.1  (tuned 5x) | When two targets are about equally good the agent can turn back and forth between them, and a reactive wave ha
  #8   train 27.3 meals  holdout  -    (tuned 5x) | In one Bellman wave every goal competes through max(), so the agent serves exactly one goal at a time and is i
  #3   train 27.1 meals  holdout 21.9  (tuned 5x) | Meals come mostly from exploration, but every frontier mark is worth the same 0.12, so a crack between two wal
  #4   train 27.9 meals  holdout 27.5  (tuned 5x) | An agent that finds nothing worth walking to stands still or rocks between two cells and starves (worlds where
  #6   train 27.2 meals  holdout  -    (tuned 5x) | The smell is painted along the world's geodesic bearing, which flips between the two ways round a wall as the 
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 35x: train 28.2 meals, holdout 25.6
```

> 15:22 model job `design` started

---

## Turn 8 -> island #11

**Hypothesis:** Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has just eaten is worth coming back to later, and the plain wave forgets it the moment the morsel is gone. A `larder` memory marks each place a morsel was taken and a `ripe` clock refills it on the regrowth timescale, so ripe larders inside the map join the food as sources: the agent harvests a home range of regrowing spots instead of relying only on finding new ones.

Prediction 32.0, got **31.6** meals, fitness 631, per world [35, 30, 33, 41, 42, 21, 20, 36, 44, 31, 38, 33, 12, 28, 45, 32, 23, 31, 24, 33]; model 84.1 s, repairs 0; vocabulary twin of #5, #4, #9, #7, #6, #10

value on the test map: max 0.08, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel larder
channel ripe
channel injection

rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 1
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# a place where the agent took a morsel is remembered (it may be a renewable spot)
rule larder:
  when 1
  set larder = where(thing * max(here, max8(here)) > 0, 1, 0.9998 * larder)

# ...and ripens again on the regrowth timescale, reset at each meal there
rule ripe:
  when 1
  set ripe = where(thing * max(here, max8(here)) > 0, 0, min(1, ripe + 0.0004 * larder))

# ripe larders join the food as sources
rule inject:
  when 1
  set injection = max(max(thing, 0.5 * larder * ripe, source * where(thing + scent > 0, 1.358, 0.4275 + 1.5 * vast)) / (1 + 0.4454 * confinement), frust * solid * max8(free) * leak)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 2.89 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 6x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #5   train 31.6 meals  holdout 26.7  (tuned 5x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 2x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #7   train 30.2 meals  holdout 27.1  (tuned 5x) | When two targets are about equally good the agent can turn back and forth between them, and a reactive wave ha
  #8   train 27.3 meals  holdout  -    (tuned 5x) | In one Bellman wave every goal competes through max(), so the agent serves exactly one goal at a time and is i
  #3   train 27.1 meals  holdout 21.9  (tuned 5x) | Meals come mostly from exploration, but every frontier mark is worth the same 0.12, so a crack between two wal
  #4   train 27.9 meals  holdout 27.5  (tuned 5x) | An agent that finds nothing worth walking to stands still or rocks between two cells and starves (worlds where
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 43x: train 28.3 meals, holdout 27.1
```

> 15:49 model job `deep` started

---

## Turn 9 (DEEP) -> island #12

**Hypothesis:** The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and turn; a shortest path hugs walls and cuts diagonals past corners, where the body collides, slides and loses time on every trip. A `clearance` channel (lower beside any solid, lowest orthogonally) makes the wave dearer along walls, so paths keep a cell of berth where the space allows and still thread doorways and corridors where it does not, and every trip gets faster.

Prediction 31.0, got **31.4** meals, fitness 628, per world [22, 34, 41, 28, 26, 32, 31, 29, 39, 32, 31, 29, 21, 34, 46, 28, 34, 24, 33, 34]; model 42.1 s, repairs 0; vocabulary twin of #5, #4, #9, #7, #6, #11

value on the test map: max 0.08, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel clearance
channel injection

# what is neither known solid nor known free; its blur = unexplored space around a cell
rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 0.9921
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

# time spent on a cell grows under the agent and fades elsewhere (scrolls with the map)
rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# how far a cell is from touching any wall: 1 in the open, lower beside solid
rule clearance:
  when 1
  set clearance = 1 - max(max4(solid), 0.5 * max8(solid))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, 0.4275 + 1.5 * vast)) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

# the wave trusts known ground and prefers cells clear of walls (no corner-cutting for a round body)
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free)) * (0.9 + 0.1 * clearance)) * (1 - solid * (1.028 - leak))
```

> 16:08 model job `design` started

---

## Turn 10 -> island #13

**Hypothesis:** Every law since #5 lands at about 31 meals on train and 25-27 on holdout whatever mechanism is added, so the stack may be carrying parts that do nothing or only fit the training worlds. This is the ablation: night 1's confinement plus only the trust-in-known-ground wave (#5's one new idea), without the unknown-weighted frontier or the frustration channels. If it eats as much, trust is the mechanism and the rest is noise, and a smaller law should generalise better to the holdout.

Prediction 29.0, got **23.9** meals, fitness 474, per world [25, 33, 24, 43, 22, 30, 20, 12, 28, 7, 0, 23, 21, 30, 44, 35, 28, 26, 0, 27]; model 68.0 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel value
channel confinement
channel injection

# a source hemmed in by solid counts less
rule confine:
  when 1
  set confinement = sum8(solid)

rule inject:
  when 1
  set injection = max(thing, source) / (1 + 0.5 * confinement)

# the wave trusts what is known: a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1 - leak))
```

### Holdout checkpoint at 3.41 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 6x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #5   train 31.6 meals  holdout 26.7  (tuned 5x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 5x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #7   train 30.2 meals  holdout 27.1  (tuned 5x) | When two targets are about equally good the agent can turn back and forth between them, and a reactive wave ha
  #8   train 27.3 meals  holdout  -    (tuned 5x) | In one Bellman wave every goal competes through max(), so the agent serves exactly one goal at a time and is i
  #3   train 27.1 meals  holdout 21.9  (tuned 5x) | Meals come mostly from exploration, but every frontier mark is worth the same 0.12, so a crack between two wal
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 51x: train 28.3 meals, holdout 27.1
```

> 16:22 model job `design` started

---

## Turn 11 -> island #14

**Hypothesis:** The body turns only ~20 degrees per thought, so a U-turn costs about nine thoughts of standing and swinging, yet the wave treats a frontier behind the agent the same as one ahead of it and the nearest unknown pocket can drag it round again and again. The law now estimates the HEADING (away from the freshest trail cell next to the agent, smoothed, copied outward over the map) and scales every frontier mark by how far it lies ahead of that heading, so exploration keeps going forward and turns only for food or a clearly better frontier.

Prediction 32.0, got **30.2** meals, fitness 606, per world [34, 33, 37, 28, 24, 22, 27, 39, 27, 40, 34, 32, 29, 30, 40, 27, 39, 30, 0, 33]; model 112.2 s, repairs 0

value on the test map: max 0.08, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel trail
channel hx
channel hy
channel ahead
channel injection

# what is neither known solid nor known free; its blur = unexplored space around a cell
rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 0.9921
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

# time spent on a cell grows under the agent and fades elsewhere (scrolls with the map)
rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# the walked path; the freshest neighbour of the agent's cell is where it came from
rule trail:
  when 1
  set trail = max(here, 0.99 * trail)

# heading = away from the previous cell, smoothed at the agent, copied outward from it
rule hx:
  when 1
  set hx = where(here > 0, 0.8 * hx + 0.2 * where(max8(trail) > 0.01, 29 * (posx() - argmax8(trail, posx())), 0), argmax8(2 - abs(posx() - 0.4828) - abs(posy() - 0.4828), hx))

rule hy:
  when 1
  set hy = where(here > 0, 0.8 * hy + 0.2 * where(max8(trail) > 0.01, 29 * (posy() - argmax8(trail, posy())), 0), argmax8(2 - abs(posx() - 0.4828) - abs(posy() - 0.4828), hy))

# cosine between a cell's bearing from the agent and the heading: +1 dead ahead, -1 behind
rule ahead:
  when 1
  set ahead = ((posx() - 0.4828) * hx + (posy() - 0.4828) * hy) / (0.001 + sqrt((posx() - 0.4828) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828)) * sqrt(hx * hx + hy * hy))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4275 + 1.5 * vast) * (1 + 0.5 * clamp(ahead, -1, 1)))) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 16:30 model job `theory` started

---

## Theory (after 12 design turns)

Measured on train, as-written: confinement alone 23.2; plus trust-in-known-ground alone 23.9 (#13), so trust by itself is nothing. Plus unknown-weighted frontier 28.1 (#3) and wall-face frustration (#4, fixed world 110 from 0; without it #13 is back to 0 there). The full stack (#5) sits near 31, and every single addition to it since lands at 30-32: trail momentum (#7), smell as compass (#9), regrow memory (#11), wall clearance (#12), heading-ahead frontier (#14). Only the sweep away from the blurred past path (#10) reached 34.7, but its holdout is 25.4, no better than the rest. Hurt: lingering smell (#6, 27.2), adding food and explore waves (#8, 26.6). On holdout nothing separates: every stack law 25-27.5, tuned confinement control 27.1. Conclusion: the stack's effect is real versus confinement on train but not proven on holdout. Untried: door/rubble-specific planning, food clusters, adaptive discount.

### Holdout checkpoint at 3.95 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 6x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #5   train 31.6 meals  holdout 26.7  (tuned 6x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 5x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #14  train 30.2 meals  holdout  -    (tuned 3x) | The body turns only ~20 degrees per thought, so a U-turn costs about nine thoughts of standing and swinging, y
  #7   train 30.2 meals  holdout 27.1  (tuned 5x) | When two targets are about equally good the agent can turn back and forth between them, and a reactive wave ha
  #8   train 27.3 meals  holdout  -    (tuned 5x) | In one Bellman wave every goal competes through max(), so the agent serves exactly one goal at a time and is i
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 59x: train 28.3 meals, holdout 27.1
```

> 16:53 model job `design` started

---

## Turn 12 -> island #15

**Hypothesis:** The world stocks most of its berries in larder rooms behind doors (up to three per chunk), and the only things that make the agent open one are a faint smell seeping through it or frustration. Once the body has learned that a hue gives (its leak taught above the innate guess), every known wall face of that hue becomes a source in its own right, a promise of a stocked room, so the agent opens the doors it walks past instead of only the ones a smell happens to point at.

Prediction 32.0, got **26.8** meals, fitness 538, per world [29, 17, 25, 46, 26, 38, 25, 30, 43, 32, 17, 48, 0, 13, 40, 14, 23, 37, 11, 22]; model 18.1 s, repairs 0; vocabulary twin of #5, #4, #12, #9, #7, #6

value on the test map: max 0.08, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel door
channel injection

# what is neither known solid nor known free; its blur = unexplored space around a cell
rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 0.9921
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

# time spent on a cell grows under the agent and fades elsewhere (scrolls with the map)
rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# a known wall face whose hue has been TAUGHT to give (leak above the innate guess) is a door:
# rooms behind doors are where the world stocks its larders
rule door:
  when 1
  set door = solid * max8(free) * max(0, leak - 0.55)

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, 0.4275 + 1.5 * vast)) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak, 2 * door)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 17:12 model job `design` started

---

## Turn 13 -> island #16

**Hypothesis:** One discount serves two jobs with different needs: food should be taken nearest-first, but for exploration the wave's 6% per step makes a two-cell crack next to the agent beat a wide unknown at the map's edge, so the agent pokes into every pocket and shadow nearby. Frontier marks are now paid back part of the distance discount by their distance from the agent (exp of distance), so exploration becomes far-sighted while food stays near-sighted, within one wave.

Prediction 31.0, got **25.6** meals, fitness 504, per world [31, 34, 4, 19, 25, 35, 26, 34, 20, 43, 0, 17, 8, 39, 38, 33, 26, 24, 17, 38]; model 96.2 s, repairs 0; vocabulary twin of #5, #4

value on the test map: max 0.10, mean 0.07, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel far
channel injection

# what is neither known solid nor known free; its blur = unexplored space around a cell
rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 0.9921
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

# time spent on a cell grows under the agent and fades elsewhere (scrolls with the map)
rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# distance from the agent in cells (the agent sits at 14/29 of the lattice)
rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.4828) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frontier marks are paid back part of the wave's distance discount: exploration is far-sighted,
# food stays near-sighted; frustrated, known wall faces become targets
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4275 + 1.5 * vast) * exp(0.04 * far))) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 4.51 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 6x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #5   train 31.6 meals  holdout 26.7  (tuned 6x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 6x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #14  train 30.2 meals  holdout  -    (tuned 5x) | The body turns only ~20 degrees per thought, so a U-turn costs about nine thoughts of standing and swinging, y
  #7   train 30.2 meals  holdout 27.1  (tuned 5x) | When two targets are about equally good the agent can turn back and forth between them, and a reactive wave ha
  #8   train 27.3 meals  holdout  -    (tuned 5x) | In one Bellman wave every goal competes through max(), so the agent serves exactly one goal at a time and is i
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 67x: train 28.3 meals, holdout 27.1
```

> 17:33 model job `deep` started

---

## Turn 14 (DEEP) -> island #17

**Hypothesis:** The planner and the body disagree about what a path costs: the wave charges distance only, but the body turns ~20 degrees a thought and slows hard above ~57 degrees of error, so a route with many bends costs far more time than its length (the tuned control drifted the same way: its diagonal step is now dearer than two straight ones). This law plans in position x heading: four directional waves (entered moving E, W, N, S) where going on straight costs one step and a 90-degree turn costs extra, U-turns are not a move; each orthogonal neighbour of the agent reads the layer of the move that enters it times the turn from the current heading (estimated from the trail), diagonals read nothing. The 

Prediction 33.0, got **22.3** meals, fitness 435, per world [26, 27, 25, 21, 18, 28, 20, 31, 23, 17, 27, 14, 24, 15, 30, 26, 21, 22, 8, 23]; model 130.2 s, repairs 0

value on the test map: max 0.08, mean 0.04, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel injection
channel pass
channel ox
channel oy
channel trail
channel hx
channel hy
channel ve
channel vw
channel vn
channel vs

rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 1
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(solid)

rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.358, 0.4275 + 1.5 * vast)) / (1 + 0.4454 * confinement), frust * solid * max8(free) * leak)

# what a cell lets through: known ground at full price, unknown dearer, solid by its leak
rule pass:
  when 1
  set pass = (0.85 + 0.15 * min(1, free)) * (1 - solid * (1.028 - leak))

# a cell's offset from the agent in cells (the agent sits at 14/29 of the lattice)
rule ox:
  when 1
  set ox = round(29 * (posx() - 0.4828))

rule oy:
  when 1
  set oy = round(29 * (posy() - 0.4828))

# heading = away from the freshest trail cell beside the agent, copied outward
rule trail:
  when 1
  set trail = max(here, 0.99 * trail)

rule hx:
  when 1
  set hx = where(here > 0, 0.5 * hx + 0.5 * where(max8(trail) > 0.01, 29 * (posx() - argmax8(trail, posx())), 0), argmax8(2 - abs(posx() - 0.4828) - abs(posy() - 0.4828), hx))

rule hy:
  when 1
  set hy = where(here > 0, 0.5 * hy + 0.5 * where(max8(trail) > 0.01, 29 * (posy() - argmax8(trail, posy())), 0), argmax8(2 - abs(posx() - 0.4828) - abs(posy() - 0.4828), hy))

# TURN-AWARE wave: v<d> = value of a cell entered moving in direction d; going on straight
# costs the step, turning 90 degrees costs extra (the body turns ~20 degrees a thought)
rule ve:
  when 1
  set ve = max(injection, 0.94 * max(shift(ve, -1, 0), 0.85 * shift(vn, 0, 1), 0.85 * shift(vs, 0, -1)) * pass)

rule vw:
  when 1
  set vw = max(injection, 0.94 * max(shift(vw, 1, 0), 0.85 * shift(vn, 0, 1), 0.85 * shift(vs, 0, -1)) * pass)

rule vn:
  when 1
  set vn = max(injection, 0.94 * max(shift(vn, 0, 1), 0.85 * shift(ve, -1, 0), 0.85 * shift(vw, 1, 0)) * pass)

rule vs:
  when 1
  set vs = max(injection, 0.94 * max(shift(vs, 0, -1), 0.85 * shift(ve, -1, 0), 0.85 * shift(vw, 1, 0)) * pass)

# read-out: each orthogonal neighbour shows the layer of the move that enters it, times the
# turn from the current heading; diagonals show nothing (moves are orthogonal)
rule value:
  when 1
  set value = where(abs(ox) + abs(oy) == 1, where(ox == 1, ve * (1 - 0.1 * (1 - clamp(hx, -1, 1))), where(ox == 0 - 1, vw * (1 - 0.1 * (1 + clamp(hx, -1, 1))), where(oy == 1, vs * (1 - 0.1 * (1 - clamp(hy, -1, 1))), vn * (1 - 0.1 * (1 + clamp(hy, -1, 1)))))), where(abs(ox) + abs(oy) == 2 and abs(ox) == 1, 0, max(ve, vw, vn, vs)))
```
