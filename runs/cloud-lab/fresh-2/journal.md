# night lab — 2026-11-04T12:48:30

Model `agent-in-session (mailbox)` | 12.0 h | world default | train [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119] | holdout [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135] | 8000 ticks/life | tune 5 | deep every 5 | theory every 6 | observe every 8

- anchor #0 `laws/atlas_wave.nomos`: train 20.4 meals [21, 36, 36, 18, 25, 34, 20, 12, 2, 26, 13, 15, 28, 35, 33, 10, 21, 18, 0, 5], holdout 15.5

- anchor #1 `laws/atlas_confine.nomos`: train 23.2 meals [20, 30, 20, 31, 25, 35, 45, 25, 17, 30, 0, 18, 21, 9, 44, 26, 3, 37, 0, 28], holdout 23.6

- anchor #2 `laws/atlas_ghost.nomos`: train 22.7 meals [21, 27, 29, 40, 21, 7, 14, 5, 12, 23, 16, 27, 28, 23, 34, 12, 12, 30, 25, 48], holdout 17.9

> 12:54 model job `design` started

---

## Turn 0 -> island #3

**Hypothesis:** Add inhibition of return. A slowly fading TRAIL channel marks the cells the agent has stood on, and it scrolls with the map. Smell and frontier sources are injected weaker on that trail, while seen morsels still inject at full strength. A smell bearing that flips between two ways round a wall, or a nook already searched, stops pulling the agent back, so the wave turns to unvisited frontier instead of a limit cycle. The 0-meal worlds look like this kind of trap. Remembered poison (`bad`) also damps the wave, so the agent does not walk back over a burn.

Prediction 25.0, got **25.6** meals, fitness 515, per world [41, 30, 29, 31, 18, 31, 12, 25, 37, 19, 25, 21, 30, 23, 45, 18, 24, 19, 0, 35]; model 30.1 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Inhibition of return: a decaying trail of where the agent has stood suppresses
# re-injection there, so smell traps and two-way limit cycles burn themselves out.

channel source
channel solid
channel thing
channel here
channel bad
channel value
channel trail
channel confinement
channel injection

# Trail of visited cells (centre + a faint ring), slow fade, bounded in [0, 1].
rule walk:
  when 1
  set trail = max(here, 0.6 * max8(here), 0.997 * trail)

# Local solidity: dead-ends and nooks.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Seen morsels inject fully; other sources fade where the agent already went.
rule inject:
  when 1
  set injection = max(thing, source * (1 - 0.8 * trail)) * (1 - bad) / (1 + 0.5 * confinement)

# Bellman wave; remembered poison blocks passage like solid.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.9 * bad)
```

> 13:07 model job `design` started

---

## Turn 1 -> island #4

**Hypothesis:** Every frontier mark is painted at the same 0.12 now, so a one-cell sliver of unknown pulls as hard as the mouth of a big unexplored region. A smoothed `dark` channel (unknown mass around a cell) weights each frontier mark by how much map lies behind it, so the agent explores where the information is. A `ghost` channel also remembers seen morsels after they leave sight and erases them where the agent stands, so a turn of the head does not lose a meal. Both sit on top of #3's inhibition of return, so exploration does not double back. Fewer wasted walks should mean more food found per life, and the starving worlds are the ones where exploring well matters most.

Prediction 28.0, got **19.8** meals, fitness 405, per world [19, 25, 15, 11, 15, 29, 31, 16, 32, 29, 0, 13, 14, 24, 29, 27, 20, 23, 0, 24]; model 42.1 s, repairs 0

value on the test map: max 0.30, mean 0.16, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Information-gain frontier + remembered morsels, on top of inhibition of return.

channel source
channel solid
channel free
channel thing
channel scent
channel here
channel bad
channel value
channel trail
channel confinement
channel dark
channel ghost
channel injection

# Trail of visited cells, slow fade, bounded in [0, 1].
rule walk:
  when 1
  set trail = max(here, 0.6 * max8(here), 0.997 * trail)

# Local solidity: dead-ends and nooks.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Smoothed unknown mass: how much unexplored map lies around a cell, in [0, 1].
rule calc_dark:
  when 1
  set dark = 0.5 * (1 - max(solid, free)) + 0.5 * avg8(dark)

# Seen morsels are remembered after they leave sight; erased where the agent stands or got burned.
rule remember:
  when 1
  set ghost = max(thing, 0.998 * ghost) * (1 - here) * (1 - bad)

# Frontier marks (no scent, no thing) are weighted by the darkness behind them; trail inhibits re-injection.
rule inject:
  when 1
  set injection = max(0.8 * ghost, thing, where(scent + thing > 0.01, source, source * (0.4 + 1.6 * dark)) * (0.7387 - 0.7824 * trail)) * (1.029 - bad) / (1 + 0.5 * confinement)

# Bellman wave; remembered poison blocks passage like solid.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.9 * bad)
```

### Holdout checkpoint at 0.57 h

```
  #3   train 26.9 meals  holdout 22.3  (tuned 5x) | Add inhibition of return. A slowly fading TRAIL channel marks the cells the agent has stood on, and it scrolls
  #4   train 23.6 meals  holdout 21.6  (tuned 3x) | Every frontier mark is painted at the same 0.12 now, so a one-cell sliver of unknown pulls as hard as the mout
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 7x: train 27.6 meals, holdout 23.2
```

> 13:27 model job `design` started

---

## Turn 2 -> island #5

**Hypothesis:** The agent gets a global internal state, FRUSTRATION. A wide blur of the trail, read at the centre cell, measures how much it is circling over old ground. That mood spreads across the map from the centre and fades slowly. As it rises, smell sources are damped and frontier marks are amplified, so a smell trap or limit cycle (the 0-meal worlds) flips the wave from exploiting to exploring until the agent reaches fresh ground. There, the trail thins and the smell takes back over.

Prediction 27.5, got **19.9** meals, fitness 384, per world [18, 19, 31, 18, 21, 21, 23, 27, 23, 23, 3, 26, 21, 17, 28, 19, 24, 16, 0, 19]; model 42.1 s, repairs 0

value on the test map: max 1.46, mean 0.77, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Frustration-gated exploration: a blurred trail measures how much the agent is
# circling; that mood is broadcast from the centre and flips injection from smell
# (exploit) toward frontier marks (explore) until the agent breaks out.

channel source
channel solid
channel thing
channel scent
channel here
channel bad
channel value
channel trail
channel crowd
channel mood
channel confinement
channel injection

# Trail of visited cells (centre + a faint ring), slow fade, bounded in [0, 1].
rule walk:
  when 1
  set trail = max(here, 0.6 * max8(here), 0.997 * trail)

# Wide blur of the trail: how revisited the neighbourhood of a cell is, in [0, 1].
rule blur:
  when 1
  set crowd = 0.3 * avg8(trail) + 0.7 * avg8(crowd)

# Mood = crowding read at the agent, broadcast over the map, fading slowly.
rule frustrate:
  when 1
  set mood = max(here * clamp(3 * crowd - 0.6, 0, 1), 0.995 * max8(mood))

# Local solidity: dead-ends and nooks.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Smell sources weaken and frontier marks strengthen as frustration builds; seen morsels inject fully.
rule inject:
  when 1
  set injection = max(thing, source * (0.7387 - 0.7824 * trail) * where(scent + thing > 0.012, 1 - 0.6 * mood, 1 + 2 * mood)) * (1.029 - bad) / (1 + 0.5 * confinement)

# Bellman wave; remembered poison blocks passage like solid.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.9 * bad)
```

> 13:46 model job `design` started

---

## Turn 3 -> island #6

**Hypothesis:** When the open map is spent (no frontier, no smell worth following), every law so far goes flat and the agent stands still and starves, which is my guess for worlds like 110 and 118. A new PROBE source makes the faces of known walls (solid touching known-free) faint sources weighted by their taught leak, kept below the 0.12 frontier and erased by the trail once pressed. That way a sealed agent goes and presses the most door-like walls one after another, opens a way out, and then the frontier and smell outside take over again, on top of #3's inhibition of return.

Prediction 25.0, got **27.4** meals, fitness 544, per world [31, 25, 29, 37, 39, 31, 15, 20, 30, 47, 0, 29, 24, 40, 36, 23, 31, 30, 0, 30]; model 64.1 s, repairs 0; vocabulary twin of #3

value on the test map: max 0.76, mean 0.40, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Wall probing: when the open map is spent, the faces of walls become faint sources weighted
# by their taught leak, so the agent presses doors (likely ones first) instead of standing still.

channel source
channel solid
channel free
channel thing
channel here
channel bad
channel leak
channel value
channel trail
channel confinement
channel injection
channel probe

# Trail of visited cells (centre + a faint ring), slow fade, bounded in [0, 1].
rule walk:
  when 1
  set trail = max(here, 0.6 * max8(here), 0.997 * trail)

# Local solidity: dead-ends and nooks.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Seen morsels inject fully; other sources fade where the agent already went.
rule inject:
  when 1
  set injection = max(thing, source * (0.7387 - 0.7824 * trail)) * (1.029 - bad) / (1 + 0.5 * confinement)

# Known wall faces (solid touching known-free) are faint sources, scaled by leak, spent once pressed.
rule calc_probe:
  when 1
  set probe = 0.06 * solid * max4(free) * leak * clamp(1 - 1.5 * trail, 0, 1) * (1 - bad)

# Bellman wave over both kinds of source; remembered poison blocks passage like solid.
rule wave:
  when 1
  set value = max(injection, probe, 0.94 * max4(value), 0.916214 * max8(value)) * (1 - solid * (1 - leak)) * (1 - 0.9 * bad)
```

> 13:49 model job `observe` started

---

## Observer: #6 on world 110

![](observe/obs00_island6_world110.png)

`{'meals': 0, 'ticks': 4103, 'alive': False, 'distance': 333.4, 'cells_visited': 27, 'last500_box': '7x2', 'doors': 0, 'breaks': 22, 'idle_presses': 26, 'nightcaps': 0}`

The agent is born in a sealed slot about two cells wide and seven tall, between two rock masses in the middle of the map, and it never leaves it: the whole track, cyan through red, is a short up-and-down shuttle inside that slot, and it starves a cell below where it started. It is not idle. It hits the rock 22 times and the brown chipped cells show it working on two different faces, the cap above the slot and the wall to its upper right, but neither gives way, so it never makes a door and never sees any of the food growing in the open ground just above and to the left. The failing is that it doesn't stick with one face: its own trail covers every cell of so small a room, and the probe term is cut by that trail, so all the wall faces fade together. None of them stays brighter than the rest, so the wave sends the agent back and forth between the top and the side and it chips a little at each. The fix I would make first is commitment to a face: a wall cell that has already given (chipped) should become a stronger source that does not fade, exempt from the trail spending, so the agent keeps striking the same spot until it breaks through instead of spreading its blows over every wall of

### Holdout checkpoint at 1.12 h

```
  #6   train 28.5 meals  holdout 25.8  (tuned 2x) | When the open map is spent (no frontier, no smell worth following), every law so far goes flat and the agent s
  #3   train 26.9 meals  holdout 22.3  (tuned 5x) | Add inhibition of return. A slowly fading TRAIL channel marks the cells the agent has stood on, and it scrolls
  #4   train 23.6 meals  holdout 21.6  (tuned 5x) | Every frontier mark is painted at the same 0.12 now, so a one-cell sliver of unknown pulls as hard as the mout
  #5   train 19.9 meals  holdout  -    (tuned 5x) | The agent gets a global internal state, FRUSTRATION. A wide blur of the trail, read at the centre cell, measur
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 16x: train 27.6 meals, holdout 23.2
```

> 14:04 model job `deep` started
