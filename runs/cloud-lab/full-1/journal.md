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
