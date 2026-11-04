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
