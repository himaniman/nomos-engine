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

### Holdout checkpoint at 5.07 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
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
  CONTROL: #1's law, constants tuned by random search 75x: train 28.3 meals, holdout 27.1
```

> 18:10 model job `design` started

---

## Turn 15 -> island #18

**Hypothesis:** Paying frontier marks back for distance (#16, far-sighted exploration) cost ~5 meals, so the trade-off has a direction: if far-sighted exploration hurts, near-sighted should help. Frontier marks now pay an EXTRA discount by their distance from the agent (food keeps the plain wave discount), so the agent finishes the unknown around it, shadows and corners within a few cells, before walking off to the map's edge, and food hidden close by is found before it is walked past.

Prediction 31.0, got **22.4** meals, fitness 441, per world [9, 8, 21, 24, 32, 41, 17, 22, 11, 21, 15, 27, 28, 40, 32, 6, 3, 35, 20, 37]; model 64.2 s, repairs 1; vocabulary twin of #16, #5, #4, #15, #12, #9

value on the test map: max 0.29, mean 0.15, 0% at the cap -> healthy: the best neighbour points at the source

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
  set unknown = max(0, 0.8696 - free) * (1 - solid)

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

# frontier marks pay an EXTRA distance discount: exploration finishes the ground nearby first
# (food keeps the plain discount); frustrated, known wall faces become targets
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4275 + 1.5 * vast) / (1 + 0.1 * far))) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 5.59 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
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
  CONTROL: #1's law, constants tuned by random search 83x: train 28.3 meals, holdout 27.1
```

> 18:24 model job `design` started

---

## Turn 16 -> island #19

**Hypothesis:** Every stack law (and night 1's confinement) injects raw `thing`, which is 1 for any seen morsel whatever the appetite says, while `source` carries the learned appetite and the hush around a remembered poison. So a nightcap the body has learned to refuse stays the brightest target on the map: the agent walks to it, refuses it, and hovers there until frustration pulls it away, or starves (the 0-meal worlds). Feeding the wave only what the appetite wants should remove those traps and cost nothing elsewhere.

Prediction 32.0, got **26.1** meals, fitness 533, per world [17, 17, 25, 35, 28, 27, 8, 27, 43, 20, 31, 33, 0, 26, 40, 43, 17, 36, 24, 24]; model 56.1 s, repairs 0; vocabulary twin of #5, #4, #15, #12, #9, #7

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

# food is what the APPETITE wants: source carries the learned appetite and the hush around a
# remembered poison, raw thing does not (a refused nightcap can stay the brightest target on the
# map for ever); frustrated, known wall faces become targets
rule inject:
  when 1
  set injection = max(source * where(thing + scent > 0, 1.027, 0.4275 + 1.5 * vast) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 18:38 model job `design` started

---

## Turn 17 -> island #20

**Hypothesis:** Dropping raw `thing` (#19) cost ~4 meals, so the full-strength pull of every SEEN morsel is load-bearing (and a nightcap eaten is still a meal). Yet confinement divides that pull too, and seen food is often hemmed in by solid: fruit grows at tree trunks, larder berries sit by room walls, mushrooms live in caves. Here a seen morsel is never suppressed; confinement still hushes the guesses (smell and frontier) in tight spots, which is where night 1 said its benefit comes from.

Prediction 32.0, got **29.9** meals, fitness 611, per world [28, 29, 32, 28, 38, 38, 19, 41, 22, 18, 32, 36, 0, 36, 36, 39, 31, 33, 28, 35]; model 18.1 s, repairs 0; vocabulary twin of #19, #5, #4, #15, #12, #9

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

# a SEEN morsel is real and is never suppressed; confinement hushes only the guesses (smell,
# frontier) in tight spots; frustrated, known wall faces become targets
rule inject:
  when 1
  set injection = max(where(thing > 0, max(thing, 1.027 * source), source * where(scent > 0, 1.027, 0.4275 + 1.5 * vast) / (1 + 0.5 * confinement)), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 18:42 model job `theory` started

---

## Theory (after 18 design turns)

The working core (#5, ~31 train as written) is: confinement suppression, frontier weighted by the unknown behind it (#3, +5 over confinement), wall faces as targets when the agent dwells (#4, rescued 0-meal world 110), and the full-strength pull of any seen morsel. Trust in known ground adds nothing alone (#13, 23.9). Clear losses, each a structural test: reshaping the frontier's distance either way (#16 far-sighted 25.6, #18 near-sighted 22.4) - the plain wave discount sits at the optimum; orthogonal turn-aware planning (#17, 22.3, uniformly slower); taught doors as sources (#15, 26.8); food only via the appetite-weighted source (#19, 26.1 - raw thing is load-bearing, and a nightcap still counts as a meal). Neutral additions (30-32): regrow memory, clearance, heading bias, unsuppressed seen food (#20). Only the sweep away from the past (#10, 34.7) stood out, but holdout 25.4. Holdout: all stack laws 25-27.5, control 27.1 - not separated.

### Holdout checkpoint at 6.11 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #5   train 31.6 meals  holdout 26.7  (tuned 6x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 6x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #20  train 29.9 meals  holdout  -    (tuned 4x) | Dropping raw `thing` (#19) cost ~4 meals, so the full-strength pull of every SEEN morsel is load-bearing (and 
  #14  train 30.2 meals  holdout  -    (tuned 5x) | The body turns only ~20 degrees per thought, so a U-turn costs about nine thoughts of standing and swinging, y
  #7   train 30.2 meals  holdout 27.1  (tuned 5x) | When two targets are about equally good the agent can turn back and forth between them, and a reactive wave ha
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 93x: train 28.3 meals, holdout 27.1
```

> 18:55 model job `design` started

---

## Turn 18 -> island #21

**Hypothesis:** The harness's map forgets a seen cell in ~1000 ticks, after which the same ground comes back as frontier and pulls the agent to re-explore it, and forgotten walls come back as frontier the agent walks into. The law keeps its OWN slower memory of what was ever seen free (`seen`) and solid (`wall`): a frontier mark on ground the law still remembers is hushed, the unexplored-space blur and confinement read the remembered map, and trust in known ground uses it too, so exploration is spent on ground truly never seen.

Prediction 31.0, got **26.4** meals, fitness 540, per world [25, 30, 27, 34, 27, 44, 26, 28, 36, 34, 0, 28, 21, 34, 52, 7, 24, 21, 0, 30]; model 8.1 s, repairs 0; vocabulary twin of #20, #19, #5, #4, #15, #12

value on the test map: max 0.43, mean 0.23, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel seen
channel wall
channel unknown
channel vast
channel confinement
channel dwell
channel frust
channel injection

# the law's own, slower memory of the map: what was ever seen free, what was ever seen solid
rule seen:
  when 1
  set seen = max(free, 0.9998 * seen)

rule wall:
  when 1
  set wall = max(solid, 0.9998 * wall)

# truly unexplored = neither remembered free nor remembered solid
rule unknown:
  when 1
  set unknown = max(0, 0.8696 - seen) * (1 - wall)

rule vast:
  when 1
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

rule confine:
  when 1
  set confinement = sum8(wall)

rule dwell:
  when 1
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# a frontier mark the harness raised only because ITS map forgot is hushed by the law's memory
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4275 + 1.5 * vast) * (1 - min(1, 2 * max(seen - free, wall - solid))))) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, max(free, seen)))) * (1 - solid * (1.028 - leak))
```

> 19:08 model job `deep` started

---

## Turn 19 (DEEP) -> island #22

**Hypothesis:** In open or symmetric ground many frontier marks are worth exactly the same, the agent's choice among them is settled by tiny differences that change as it moves, and a deterministic tie-break can lock it into rocking between equal options (the classic reactive limit cycle). Each map cell now draws one random preference when it first scrolls into view and keeps it while in view, multiplying its frontier weight: a fixed, world-anchored symmetry breaker, so among equal unknowns the agent consistently prefers the same one and commits without any memory of its own.

Prediction 31.0, got **26.9** meals, fitness 530, per world [32, 25, 36, 31, 26, 33, 17, 30, 32, 28, 28, 16, 24, 29, 42, 27, 22, 32, 0, 29]; model 20.1 s, repairs 0; vocabulary twin of #20, #19, #5, #4, #15, #12

value on the test map: max 0.07, mean 0.05, 0% at the cap -> healthy: the best neighbour points at the source

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
channel pref
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

# every cell draws ONE random preference when it scrolls into view and keeps it: a fixed,
# consistent tie-breaker among otherwise equal frontier marks (no flicker)
rule pref:
  when pref == 0
  set pref = uniform(0.6, 1)

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4275 + 1.5 * vast) * pref)) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 19:25 model job `design` started

### Holdout checkpoint at 6.62 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #5   train 31.6 meals  holdout 26.7  (tuned 6x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 6x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #20  train 29.9 meals  holdout  -    (tuned 5x) | Dropping raw `thing` (#19) cost ~4 meals, so the full-strength pull of every SEEN morsel is load-bearing (and 
  #14  train 30.2 meals  holdout  -    (tuned 5x) | The body turns only ~20 degrees per thought, so a U-turn costs about nine thoughts of standing and swinging, y
  #7   train 30.2 meals  holdout 27.1  (tuned 5x) | When two targets are about equally good the agent can turn back and forth between them, and a reactive wave ha
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 103x: train 28.3 meals, holdout 27.1
```

---

## Turn 20: behavioural twin of #5

**Hypothesis:** Giving the law a longer memory of the map (#21) cost ~4 meals and brought back two 0-meal worlds, so re-visiting old ground is valuable: 70% of food spots regrow after ~1200 ticks, about when the map forgets. The opposite test: ground whose free-belief has HALF faded (`stale`, peaking midway through forgetting) becomes a faint source of its own, so the agent drifts back over ground it saw a few hundred ticks ago, where eaten spots are ripening, instead of waiting for it to be fully forgotten and re-raised as frontier.

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
channel stale
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

# ground seen long ago: still believed free, but the belief has faded (half-forgotten);
# food regrows on about the timescale the map forgets on, so stale ground is worth a look
rule stale:
  when 1
  set stale = where(free > 0.2, 4 * free * (1 - free), 0) * (1 - solid) * (1 - max(here, max8(here)))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, 0.4275 + 1.5 * vast), 0.05 * stale) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 19:28 model job `design` started

---

## Turn 21 -> island #23

**Hypothesis:** Every law so far plans with a max() wave: each cell sees only its single best target, so one berry pulls exactly as hard as a cluster of three, one frontier crack as hard as a long open edge, and between two near-equal targets the agent can rock. This law replaces the wave with a POTENTIAL FIELD: each cell relaxes toward the slightly discounted MEAN of its eight neighbours (walls absorb by their leak), so the pulls of all sources add up, the field is smooth with no plateaus or ties, and the agent heads to where the most food and unknown lie together. Same senses and injection as the best law; only the planning physics changes.

Prediction 28.0, got **11.3** meals, fitness 234, per world [4, 15, 18, 20, 18, 17, 8, 0, 4, 22, 0, 13, 13, 4, 13, 11, 23, 11, 0, 13]; model 48.1 s, repairs 0; vocabulary twin of #20, #19, #5, #4, #15, #12

value on the test map: max 0.08, mean 0.00, 0% at the cap -> healthy: the best neighbour points at the source

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

# a POTENTIAL FIELD instead of a wave: each cell relaxes to the (slightly discounted) MEAN of
# its neighbours, walls absorb by their leak; sources add up, so many marks pull harder than one
rule potential:
  when 1
  set value = max(injection, 0.995 * avg8(value) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 19:45 model job `design` started

---

## Turn 22 -> island #24

**Hypothesis:** The worlds that still end with 0 meals look like a sealed or exhausted start: the wave cannot see past solid (it leaks only the learned leak, low for stone and for any hue that once refused), so nothing beyond the wall pulls and frustration alone only lights the wall faces. Here frustration also opens the PLAN: the wave's leak through solid rises with frustration, so the unknown and the smell beyond the walls flow in and lead the agent into pressing the wall that stands between it and them; when it moves again, frustration fades and walls close.

Prediction 31.0, got **27.4** meals, fitness 549, per world [24, 20, 28, 26, 26, 35, 30, 33, 43, 28, 24, 48, 2, 17, 40, 19, 21, 15, 32, 36]; model 30.2 s, repairs 0; vocabulary twin of #20, #19, #5, #4, #15, #12

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

# frustration makes walls transparent to the PLAN: a stuck agent's wave leaks through solid as
# if every wall might give, so unknown ground beyond a wall pulls it into pressing that wall
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - max(leak, min(1, 2 * frust))))
```

### Holdout checkpoint at 7.13 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #5   train 31.6 meals  holdout 26.7  (tuned 7x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 6x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #20  train 29.9 meals  holdout  -    (tuned 5x) | Dropping raw `thing` (#19) cost ~4 meals, so the full-strength pull of every SEEN morsel is load-bearing (and 
  #14  train 30.2 meals  holdout  -    (tuned 5x) | The body turns only ~20 degrees per thought, so a U-turn costs about nine thoughts of standing and swinging, y
  #7   train 30.2 meals  holdout 27.1  (tuned 5x) | When two targets are about equally good the agent can turn back and forth between them, and a reactive wave ha
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 112x: train 28.3 meals, holdout 27.1
```

> 20:01 model job `design` started

---

## Turn 23 -> island #25

**Hypothesis:** Food in this world comes in clumps (one to two berries per larder room, two fruits per tree, mushrooms in the same caves), so a meal is evidence of more food close by; foraging animals answer that with area-restricted search. A meal (a morsel at the agent's ring) starts a pulse `ars` that fills the map and fades over ~100 ticks; while it lasts, frontier marks within a few cells of the agent are boosted, so after eating the agent combs the nearby nooks before moving on, and otherwise explores as before (unlike the always-near-sighted #18, which lost).

Prediction 31.0, got **28.2** meals, fitness 558, per world [20, 33, 25, 29, 27, 32, 30, 38, 19, 32, 35, 31, 0, 27, 38, 28, 35, 30, 19, 36]; model 36.1 s, repairs 0; vocabulary twin of #16, #18

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
channel ars
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

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.4828) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4275 + 1.5 * vast) * (1 + 2 * ars * exp(0 - far / 4)))) / (1 + 0.5 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 20:06 model job `theory` started

---

## Theory (after 24 design turns)

Core that works (#5, 30.3 as written; ablation #13 shows confinement plus trust alone is 23.9): frontier weighted by the unknown behind it (+5) and wall faces lit by dwell-frustration (rescues the 0-meal starts). The max() wave itself is essential: a mean-based potential field fell to 11.3 (#23). The plain wave discount is a local optimum: far-sighted (#16, 25.6) and near-sighted (#18, 22.4) frontiers both lost; so did area-restricted search after meals (#25, 28.2). Map memory: a slower memory than the harness's lost 4 (#21, 26.4) and a faint pull to half-forgotten ground never won a max (#5 twin): re-exploring forgotten ground is needed, not added. Raw seen-morsel pull is load-bearing (#19, 26.1). Neutral or noise (26.9-29.9): random tie-breaker (#22), frustration-opened walls (#24), unsuppressed seen food (#20). Single train means differ by ~2 from noise alone. Only the sweep away from the past (#10, 34.7) stands out on train; holdout 25.4, unresolved.

> 20:23 model job `deep` started

---

## Turn 24 (DEEP) -> island #26

**Hypothesis:** Weighting each frontier mark by the unknown space behind it was the largest single gain of the night (#3, +5), but that blur passes straight through walls, so a crack whose far side is the next sealed room counts the room's unknown as if walking there would reveal it. Here the unknown-space blur stops at solid cells (information you can actually see from that frontier), on top of the best law, the sweep away from the agent's own past (#10): both the target choice and the direction of exploration should spend fewer trips on frontier that opens onto nothing.

Prediction 34.0, got **31.6** meals, fitness 637, per world [36, 20, 25, 52, 28, 32, 6, 49, 28, 38, 39, 18, 35, 32, 47, 28, 28, 32, 23, 35]; model 10.1 s, repairs 0; vocabulary twin of #10, #7, #24, #20, #19, #5

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

# unexplored space a frontier opens onto, blurred only through what is not solid: the unknown
# in the next room, behind a wall, is not what walking to this frontier will reveal
rule vast:
  when 1
  set vast = (0.3 * unknown + 0.8306 * avg8(vast)) * (1 - solid)

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

### Holdout checkpoint at 7.71 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #25  train 32.2 meals  holdout 29.4  (tuned 5x) | Food in this world comes in clumps (one to two berries per larder room, two fruits per tree, mushrooms in the 
  #26  train 31.6 meals  holdout 22.6  (tuned 0x) | Weighting each frontier mark by the unknown space behind it was the largest single gain of the night (#3, +5),
  #5   train 31.6 meals  holdout 26.7  (tuned 7x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 7x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #20  train 29.9 meals  holdout  -    (tuned 5x) | Dropping raw `thing` (#19) cost ~4 meals, so the full-strength pull of every SEEN morsel is load-bearing (and 
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 121x: train 28.3 meals, holdout 27.1
```

> 20:45 model job `design` started

---

## Turn 25 -> island #27

**Hypothesis:** The two laws that stand out are opposites: area-restricted search after a meal (#25, best holdout 29.4) combs the neighbourhood, the sweep away from the agent's own past (#10, best train 34.7) leaves it. Foraging animals switch between exactly these two modes, intensive search after a find and extensive search between finds. Here both run on the ARS law: the blurred past path discounts frontier near where the agent has been, but only to the extent no recent meal (`ars`) says the neighbourhood is worth combing.

Prediction 33.0, got **25.6** meals, fitness 511, per world [28, 6, 9, 8, 30, 24, 22, 35, 33, 32, 34, 40, 14, 26, 33, 27, 26, 24, 30, 31]; model 38.1 s, repairs 0; vocabulary twin of #25, #16, #18

value on the test map: max 0.11, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

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
channel ars
channel far
channel trail
channel been
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
  when 0.7408
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

# EXTENSIVE search between meals: a fading trail of the walked path, blurred wide; frontier near
# the agent's own past counts less unless a recent meal (ars) says to comb the neighbourhood
rule trail:
  when 1
  set trail = max(here, 0.998 * trail)

rule been:
  when 1
  set been = 0.2 * trail + 0.8 * avg8(been)

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.5524) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4611 + 1.5 * vast) * (0.8222 + 2 * ars * exp(0 - far / 4)) * (1 - min(1, 3 * been) * (1 - min(1, ars))))) / (0.9063 + 0.6266 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 0.8649
  set value = max(injection, max(0.94 * max4(value), 0.6968 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 8.23 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #25  train 32.2 meals  holdout 29.4  (tuned 5x) | Food in this world comes in clumps (one to two berries per larder room, two fruits per tree, mushrooms in the 
  #26  train 31.6 meals  holdout 22.6  (tuned 5x) | Weighting each frontier mark by the unknown space behind it was the largest single gain of the night (#3, +5),
  #27  train 30.7 meals  holdout  -    (tuned 4x) | The two laws that stand out are opposites: area-restricted search after a meal (#25, best holdout 29.4) combs 
  #5   train 31.6 meals  holdout 26.7  (tuned 7x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 7x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 129x: train 28.3 meals, holdout 27.1
```

> 21:04 model job `design` started

---

## Turn 26 -> island #28

**Hypothesis:** The random-search tuner, run independently on two different laws (the control and #25), drove the diagonal discount from 0.916 down to ~0.70-0.76, below two orthogonal steps (0.884), so it keeps saying the same thing: value should not travel diagonally. Read as structure, that is a Manhattan wave: value spreads only through the four orthogonal neighbours, so the field takes the grid's own geometry (walls, doorways, corridors are axis-aligned), corner-cutting through a wall's diagonal gap gets no value, and the agent's diagonal steps are only those two straight steps would justify. On the best-train law (#10, the sweep), with the max8 term removed.

Prediction 34.0, got **27.0** meals, fitness 549, per world [32, 34, 43, 28, 23, 37, 30, 34, 30, 32, 0, 40, 26, 10, 40, 27, 32, 33, 0, 9]; model 16.1 s, repairs 0; vocabulary twin of #26, #10, #7, #24, #20, #19

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

# a MANHATTAN wave: value travels only orthogonally, so a diagonal neighbour is worth what two
# straight steps are; the plan is in the grid's own geometry (walls and doors are axis-aligned)
rule wave:
  when 1
  set value = max(injection, 0.94 * max4(value) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 21:21 model job `design` started

---

## Turn 27 -> island #29

**Hypothesis:** The painted smell's strength encodes its distance: a far smell is floored at 0.35, painted at the map's edge in a coarse cardinal direction (a phantom that flips round walls), a near one is strong and exact. Every law so far weighs them alike. Here the smell is weighed by its own strength again (quadratic), so a far phantom counts about like one unexplored cell and stops dragging the agent into limit cycles, while a near, exact smell still pulls like food. Built on the best-holdout law (#25, area-restricted search).

Prediction 32.0, got **22.2** meals, fitness 449, per world [21, 7, 11, 34, 18, 21, 20, 32, 23, 27, 24, 23, 16, 28, 34, 27, 25, 19, 18, 17]; model 40.1 s, repairs 0; vocabulary twin of #25, #16, #27, #18

value on the test map: max 0.11, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

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
channel ars
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
  when 0.7408
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.5524) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
# a smell's painted strength says how far it is (faint = far, floored, cardinal, a phantom):
# weigh it by itself again, so a far smell counts like an unexplored cell, a near one like food
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing > 0, 1.027, where(scent > 0, 1.027 * scent, (0.4611 + 1.5 * vast) * (0.8222 + 2 * ars * exp(0 - far / 4))))) / (0.9063 + 0.6266 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 0.8649
  set value = max(injection, max(0.94 * max4(value), 0.6968 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 8.74 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #25  train 32.2 meals  holdout 29.4  (tuned 5x) | Food in this world comes in clumps (one to two berries per larder room, two fruits per tree, mushrooms in the 
  #26  train 31.6 meals  holdout 22.6  (tuned 5x) | Weighting each frontier mark by the unknown space behind it was the largest single gain of the night (#3, +5),
  #27  train 30.7 meals  holdout  -    (tuned 5x) | The two laws that stand out are opposites: area-restricted search after a meal (#25, best holdout 29.4) combs 
  #5   train 31.6 meals  holdout 26.7  (tuned 7x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 7x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #9   train 30.9 meals  holdout 25.2  (tuned 5x) | The painted smell is a phantom place: right in direction, wrong in position, and it flips between the two ways
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 138x: train 28.3 meals, holdout 27.1
```

> 21:39 model job `design` started

---

## Turn 28 -> island #30

**Hypothesis:** Weighing the smell by its own strength (#29, far smells reduced to a frontier mark's worth) cost 10 meals, so the far smell, phantom or not, is the agent's main long-range lead to food: it is the only sense that reaches past the 12-cell map. The opposite test: a far smell is painted at the map's edge where the wave's distance discount already halves it before it reaches the agent, so its weight is now boosted back as it gets fainter (faint smells pull more, near ones as before), and the nearest frontier crack should no longer outbid a real meal beyond the map.

Prediction 33.0, got **28.5** meals, fitness 582, per world [18, 12, 31, 54, 31, 36, 17, 41, 42, 32, 27, 14, 23, 22, 44, 25, 26, 17, 26, 32]; model 22.1 s, repairs 0; vocabulary twin of #29, #25, #16, #27, #18

value on the test map: max 0.11, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

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
channel ars
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
  when 0.7408
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.5524) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
# a smell's painted strength says how far it is; a far smell is painted at the map's edge and the
# wave's discount already halves it, so it is boosted back: faint smells pull MORE, not less
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing > 0, 1.027, where(scent > 0, 1.027 * (2 - scent), (0.4611 + 1.5 * vast) * (0.8222 + 2 * ars * exp(0 - far / 4))))) / (0.9063 + 0.6266 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 0.8649
  set value = max(injection, max(0.94 * max4(value), 0.6968 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 21:56 model job `deep` started

---

## Turn 29 (DEEP) -> island #31

**Hypothesis:** The max() wave values a lone berry exactly like a berry with two more beside it, yet food grows in clumps (one or two berries per larder, two fruits per tree) and a trip to a clump pays several meals for one walk. A `clump` blur of the seen morsels raises each morsel's pull by how many others lie around it, so between two seen targets at similar distances the agent goes for the group; area-restricted search (#25, the best holdout law, which this builds on) then combs the rest of it.

Prediction 32.0, got **29.1** meals, fitness 584, per world [35, 11, 29, 32, 29, 47, 30, 33, 22, 28, 28, 32, 8, 24, 42, 39, 26, 21, 31, 35]; model 40.1 s, repairs 0; vocabulary twin of #30, #29, #25, #16, #27, #18

value on the test map: max 0.11, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

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
channel ars
channel far
channel clump
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
  when 0.7408
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.5524) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
# how many seen morsels lie around a cell: food grows in clumps (larders, trees), and a morsel
# with others beside it is worth more than one alone, the trip pays several meals
rule clump:
  when 1
  set clump = 0.3 * thing + 0.6 * avg8(clump)

rule inject:
  when 1
  set injection = max(max(thing * (1 + 2 * clump), source * where(thing + scent > 0, 1.027, (0.4611 + 1.5 * vast) * (0.8222 + 2 * ars * exp(0 - far / 4)))) / (0.9063 + 0.6266 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 0.8649
  set value = max(injection, max(0.94 * max4(value), 0.6968 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 22:00 model job `theory` started

---

## Theory (after 30 design turns)

Load-bearing, each shown by a loss when removed or weakened: the max() wave (mean-based potential field #23: 11.3); the far smell, the only sense past the map (weighing it by its own faint strength, #29: 22.2, -10 against its base); raw pull of seen morsels (#19: 26.1); frontier weighted by unknown behind it (#3: +5); dwell-frustration on wall faces (removing or bypassing it brings back the 0-meal worlds 110/118, e.g. Manhattan wave #28: 27.0). The plain discount and the 8-way wave sit at an optimum: reshaping frontier distance either way (#16, #18) and pure orthogonal waves (#17, #28) lost. Neutral, within the ~2-meal noise of one 20-world mean: boosting far smell (#30: 28.5), clump bonus (#31: 29.1), sweep plus wall-aware unknown (#26: 31.6). Honest numbers: area-restricted search after meals (#25) holds 29.4 on holdout against the tuned control's 27.1; the sweep (#10) is best on train, 25.4 holdout.

### Holdout checkpoint at 9.28 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #30  train 32.2 meals  holdout 25.7  (tuned 5x) | Weighing the smell by its own strength (#29, far smells reduced to a frontier mark's worth) cost 10 meals, so 
  #25  train 32.2 meals  holdout 29.4  (tuned 5x) | Food in this world comes in clumps (one to two berries per larder room, two fruits per tree, mushrooms in the 
  #26  train 31.6 meals  holdout 22.6  (tuned 5x) | Weighting each frontier mark by the unknown space behind it was the largest single gain of the night (#3, +5),
  #27  train 30.7 meals  holdout  -    (tuned 5x) | The two laws that stand out are opposites: area-restricted search after a meal (#25, best holdout 29.4) combs 
  #5   train 31.6 meals  holdout 26.7  (tuned 7x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 7x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 147x: train 28.3 meals, holdout 27.1
```

> 22:19 model job `design` started

---

## Turn 30 -> island #32

**Hypothesis:** Trust in known ground (unknown cells dearer for the wave) did nothing on its own (#13), and since it entered the stack world 112 ends at or near 0 meals in most laws (#5, #19, #20, #25, #31) where earlier laws ate 16-21: when the only way on runs through cells the map has not confirmed free, the trust term starves that route of value. This ablation removes trust from the best-holdout law (#25, area-restricted search, as tuned), keeping everything else: if trust is the cause, world 112 comes back and nothing else is lost.

Prediction 32.0, got **24.9** meals, fitness 478, per world [13, 24, 18, 26, 28, 40, 28, 34, 35, 28, 22, 17, 26, 20, 32, 14, 27, 23, 22, 22]; model 18.1 s, repairs 0; vocabulary twin of #30, #29, #25, #31, #16, #27

value on the test map: max 0.11, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

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
channel ars
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
  when 0.7408
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.5524) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4611 + 1.5 * vast) * (0.8222 + 2 * ars * exp(0 - far / 4)))) / (0.9063 + 0.6266 * confinement), frust * solid * max8(free) * leak)

# NO trust term: an unknown cell carries the wave at the same price as a known one
rule wave:
  when 0.8649
  set value = max(injection, max(0.94 * max4(value), 0.6968 * max8(value))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 9.80 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #30  train 32.2 meals  holdout 25.7  (tuned 5x) | Weighing the smell by its own strength (#29, far smells reduced to a frontier mark's worth) cost 10 meals, so 
  #25  train 32.2 meals  holdout 29.4  (tuned 5x) | Food in this world comes in clumps (one to two berries per larder room, two fruits per tree, mushrooms in the 
  #26  train 31.6 meals  holdout 22.6  (tuned 5x) | Weighting each frontier mark by the unknown space behind it was the largest single gain of the night (#3, +5),
  #27  train 30.7 meals  holdout  -    (tuned 5x) | The two laws that stand out are opposites: area-restricted search after a meal (#25, best holdout 29.4) combs 
  #5   train 31.6 meals  holdout 26.7  (tuned 7x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 7x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 154x: train 28.3 meals, holdout 27.1
```

> 22:37 model job `design` started

---

## Turn 31 -> island #33

**Hypothesis:** Removing trust in known ground (#32) brought world 112 back from 0 to 26 meals but cost ~7 meals elsewhere: distrust of unknown cells is right in general and wrong exactly when the only way on runs through unconfirmed ground. So distrust is made conditional: dwell-frustration, already the law's sign of being stuck, lifts it, and a stuck agent's wave values routes through unknown cells at full price until it moves again. Built on the best-holdout law (#25 as tuned).

Prediction 32.0, got **27.7** meals, fitness 553, per world [29, 25, 34, 16, 37, 25, 26, 23, 31, 37, 26, 33, 24, 24, 43, 18, 28, 32, 24, 19]; model 18.1 s, repairs 0; vocabulary twin of #32, #30, #29, #25, #31, #16

value on the test map: max 0.11, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

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
channel ars
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
  when 0.7408
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.5524) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4611 + 1.5 * vast) * (0.8222 + 2 * ars * exp(0 - far / 4)))) / (0.9063 + 0.6266 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known (unknown cells dearer) - unless the agent is stuck: frustration
# lifts the distrust, so a way on through unconfirmed ground gets its full value back
rule wave:
  when 0.8649
  set value = max(injection, max(0.94 * max4(value), 0.6968 * max8(value)) * (0.85 + 0.15 * min(1, max(free, 3 * frust)))) * (1 - solid * (1.028 - leak))
```

> 22:54 model job `design` started

---

## Turn 32 -> island #34

**Hypothesis:** Far-sighted exploration everywhere lost 5 meals (#16) and combing near a meal won the best holdout (#25), so the right horizon depends on how the ground has been paying. A slow meal pulse (`fed`, ~700 ticks) measures a drought; as it runs dry, frontier marks are paid back for distance, so an agent in a barren patch strikes out for far unknown instead of picking over the near cracks, while after a meal it still combs the neighbourhood. Built on #25 as tuned.

Prediction 31.0, got **27.6** meals, fitness 548, per world [28, 27, 25, 19, 23, 29, 33, 30, 24, 35, 26, 31, 20, 30, 42, 28, 29, 25, 22, 27]; model 32.2 s, repairs 0; vocabulary twin of #33, #32, #30, #29, #25, #31

value on the test map: max 0.12, mean 0.07, 0% at the cap -> healthy: the best neighbour points at the source

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
channel ars
channel far
channel fed
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
  when 0.7408
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

# a SLOW meal pulse (~700 ticks): how recently this region fed the agent at all; a long drought
# says the ground around is barren
rule fed:
  when 1
  set fed = max(thing * max(here, max8(here)), 0.9995 * max8(fed))

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.5524) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
# in a drought, frontier is paid back for distance (leave the barren patch); after meals, comb near
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4611 + 1.5 * vast) * (0.8222 + 2 * ars * exp(0 - far / 4)) * exp(0.04 * far * (1 - min(1, fed))))) / (0.9063 + 0.6266 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 0.8649
  set value = max(injection, max(0.94 * max4(value), 0.6968 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 10.32 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #30  train 32.2 meals  holdout 25.7  (tuned 5x) | Weighing the smell by its own strength (#29, far smells reduced to a frontier mark's worth) cost 10 meals, so 
  #25  train 32.2 meals  holdout 29.4  (tuned 5x) | Food in this world comes in clumps (one to two berries per larder room, two fruits per tree, mushrooms in the 
  #26  train 31.6 meals  holdout 22.6  (tuned 5x) | Weighting each frontier mark by the unknown space behind it was the largest single gain of the night (#3, +5),
  #27  train 30.7 meals  holdout  -    (tuned 5x) | The two laws that stand out are opposites: area-restricted search after a meal (#25, best holdout 29.4) combs 
  #5   train 31.6 meals  holdout 26.7  (tuned 7x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 7x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 163x: train 29.8 meals, holdout 24.6
```

> 23:15 model job `design` started

---

## Turn 33 -> island #35

**Hypothesis:** The wave values each target alone, as if the life ended on reaching it, but a forager's next meal starts where this one ends. Here a target's injection is raised by the value field around it (max8 of value), so a target with more food or unknown just beyond it beats an equally near dead end, and routes that string meals together win. Part of the raise is the target's own echo, which makes strong targets (seen food, near smell) superlinearly stronger than faint frontier marks; it stays bounded (about 1.6x for a full morsel). Built on #25 as tuned.

Prediction 30.0, got **26.4** meals, fitness 523, per world [19, 31, 33, 36, 24, 34, 14, 15, 34, 33, 28, 43, 28, 29, 37, 21, 3, 21, 26, 18]; model 32.1 s, repairs 0; vocabulary twin of #33, #32, #30, #29, #25, #34

value on the test map: max 0.11, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

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
channel ars
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
  when 0.7408
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.5524) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4611 + 1.5 * vast) * (0.8222 + 2 * ars * exp(0 - far / 4)))) / (0.9063 + 0.6266 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
# CHAINING: a target is worth more when the field around it is high, i.e. when something else is
# worth having just beyond it, so routes that string meals together beat a lone dead-end target
rule wave:
  when 0.8649
  set value = max(injection * (1 + 0.4 * max8(value) * where(injection > 0.05, 1, 0)), max(0.94 * max4(value), 0.6968 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 23:36 model job `deep` started

---

## Turn 34 (DEEP) -> island #36

**Hypothesis:** The sweep away from the agent's own past (#10) is the best law on train but loses 9 meals on holdout, the signature of a few worlds where it gets stuck; on #25, lifting the distrust of unknown ground while frustrated (#33) gave the most even result of the night (all 20 immortal, worst world 16, world 112 back from 0). The same conditional trust on the sweep: distrust unknown cells while things go well, value routes through them at full price while stuck, so the sweep keeps its gains and loses its dead ends.

Prediction 33.0, got **28.7** meals, fitness 553, per world [34, 12, 11, 5, 31, 23, 6, 32, 29, 32, 45, 27, 37, 41, 47, 34, 31, 39, 33, 25]; model 18.1 s, repairs 0; vocabulary twin of #28, #26, #10, #7, #24, #20

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

# trust in known ground, lifted while the agent is stuck (frustration): a way on through
# unconfirmed cells gets its full value back exactly when it is the only way on
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, max(free, 3 * frust)))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 10.86 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 7x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #30  train 32.2 meals  holdout 25.7  (tuned 5x) | Weighing the smell by its own strength (#29, far smells reduced to a frontier mark's worth) cost 10 meals, so 
  #25  train 32.2 meals  holdout 29.4  (tuned 5x) | Food in this world comes in clumps (one to two berries per larder room, two fruits per tree, mushrooms in the 
  #26  train 31.6 meals  holdout 22.6  (tuned 5x) | Weighting each frontier mark by the unknown space behind it was the largest single gain of the night (#3, +5),
  #27  train 30.7 meals  holdout  -    (tuned 5x) | The two laws that stand out are opposites: area-restricted search after a meal (#25, best holdout 29.4) combs 
  #5   train 31.6 meals  holdout 26.7  (tuned 7x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 7x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 171x: train 29.8 meals, holdout 24.6
```

> 23:55 model job `design` started

---

## Turn 35 -> island #37

**Hypothesis:** Two mechanisms in the best-train law (#10) both decide WHICH frontier to walk to: the unknown-space weighting (#3, +5 over confinement) and the sweep away from the blurred past path. If the sweep already sends the agent toward big unexplored ground (what lies away from where it has been is mostly unknown), the unknown weighting is redundant there. This ablation removes `unknown`/`vast` from #10 as tuned: if it eats about the same, the sweep subsumes the information-gain weight; if it drops ~5, they are separate effects.

Prediction 30.0, got **21.7** meals, fitness 431, per world [22, 4, 19, 39, 30, 30, 20, 8, 24, 39, 30, 26, 1, 29, 25, 8, 23, 33, 0, 24]; model 18.1 s, repairs 0; vocabulary twin of #36, #28, #26, #10, #7, #24

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel confinement
channel dwell
channel frust
channel trail
channel been
channel injection

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
# ablation: every frontier mark weighs the same again; only the sweep (been) shapes exploration
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.358, (1 - min(1, 3 * been)))) / (1 + 0.4454 * confinement), frust * solid * max8(free) * leak)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

> 23:58 model job `theory` started

---

## Theory (after 36 design turns)

Load-bearing, each shown by a loss on removal: weighting frontier by the unknown space behind it (removed from the sweep law, #37: 21.7 vs 34.7; added to confinement, #3: +5), the sweep does not replace it; the far smell (#29 weakening it: 22.2); the max() wave (#23 potential field: 11.3); raw seen-morsel pull (#19: 26.1); dwell-frustration on wall faces (rescues worlds 110/118). Trust in known ground helps overall but starves world 112 (#32 removing it: 112 back, total -7); lifting it while frustrated evens lives out (#33, #36: all or most immortal) without raising the mean. Reshaping the frontier's distance (#16, #18, #34), orthogonal waves (#17, #28), memory variants (#21, #22) and additions to food value (#30 boost smell, #31 clumps, #35 chaining) were neutral or worse. Honest numbers: area-restricted search after meals (#25) holds 29.4 on holdout, tuned control 24.6; sweep (#10) is best on train only.

> 00:10 model job `design` started

> 00:11 model job `design` started

---

## Turn 36 -> island #38

**Hypothesis:** The unknown-space weighting of the frontier is the night's most load-bearing mechanism (+5 when added, -13 when removed from the sweep law), but its blur only reaches a few cells: it says what a frontier opens onto right there, nothing about which side of the map is unexplored as a whole. A second, WIDE blur of the unknown (clamped at 0, reaching most of the 25-cell map) is added to the weight, so between two frontiers that open onto similar pockets the agent takes the one on the emptier side of its map. Built on #25 (area-restricted search) as tuned.

Prediction 30.0, got **26.4** meals, fitness 534, per world [31, 31, 34, 32, 31, 43, 19, 20, 29, 33, 27, 10, 0, 17, 39, 14, 35, 36, 21, 25]; model 40.1 s, repairs 0; vocabulary twin of #35, #33, #32, #30, #29, #25

value on the test map: max 0.11, mean 0.06, 0% at the cap -> healthy: the best neighbour points at the source

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
channel ars
channel far
channel wide
channel injection

# what is neither known solid nor known free; its blur = unexplored space around a cell
rule unknown:
  when 1
  set unknown = (0.8696 - free) * (1 - solid)

rule vast:
  when 0.9921
  set vast = 0.3 * unknown + 0.8306 * avg8(vast)

# a second, WIDE blur of the unknown (reaches most of the map): which side of the map is still
# unexplored as a whole, beside `vast`, which says what a frontier opens onto right there
rule wide:
  when 1
  set wide = 0.05 * max(0, unknown) + 0.95 * avg8(wide)

rule confine:
  when 1
  set confinement = sum8(solid)

# time spent on a cell grows under the agent and fades elsewhere (scrolls with the map)
rule dwell:
  when 0.7408
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.5524) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4611 + 1.5 * vast + 1.5 * wide) * (0.8222 + 2 * ars * exp(0 - far / 4)))) / (0.9063 + 0.6266 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 0.8649
  set value = max(injection, max(0.94 * max4(value), 0.6968 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

### Holdout checkpoint at 11.42 h

```
  #10  train 34.7 meals  holdout 25.4  (tuned 8x) | The map only spans 25x25 cells and forgets, so the agent has no sense of where it has already been and its fro
  #30  train 32.2 meals  holdout 25.7  (tuned 5x) | Weighing the smell by its own strength (#29, far smells reduced to a frontier mark's worth) cost 10 meals, so 
  #25  train 32.2 meals  holdout 29.4  (tuned 5x) | Food in this world comes in clumps (one to two berries per larder room, two fruits per tree, mushrooms in the 
  #26  train 31.6 meals  holdout 22.6  (tuned 5x) | Weighting each frontier mark by the unknown space behind it was the largest single gain of the night (#3, +5),
  #27  train 30.7 meals  holdout  -    (tuned 5x) | The two laws that stand out are opposites: area-restricted search after a meal (#25, best holdout 29.4) combs 
  #5   train 31.6 meals  holdout 26.7  (tuned 7x) | The Bellman wave prices a step through an unknown cell the same as a step through a cell the agent has seen to
  #11  train 31.6 meals  holdout 25.9  (tuned 7x) | Most food sits at fixed spots and 70% of them regrow on a timer (~1200 ticks), so a place where the agent has 
  #12  train 31.4 meals  holdout  -    (tuned 5x) | The wave plans for a point but the agent is a round body of nearly a cell's width that steers with thrust and 
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 181x: train 29.8 meals, holdout 24.6
```
