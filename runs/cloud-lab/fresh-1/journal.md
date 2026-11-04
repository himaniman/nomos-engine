# night lab — 2026-11-04T11:35:33

Model `agent-in-session (mailbox)` | 12.0 h | world default | train [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] | holdout [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] | 8000 ticks/life | tune 5 | deep every 5 | theory every 6 | observe every 8

- anchor #0 `laws/atlas_wave.nomos`: train 14.8 meals [28, 27, 10, 35, 13, 4, 5, 5, 33, 12, 5, 7, 23, 23, 18, 8, 0, 28, 5, 6], holdout 19.6

- anchor #1 `laws/atlas_confine.nomos`: train 25.7 meals [37, 39, 18, 30, 23, 22, 13, 34, 26, 13, 33, 36, 33, 24, 29, 30, 0, 34, 14, 26], holdout 25.7

- anchor #2 `laws/atlas_ghost.nomos`: train 19.2 meals [28, 5, 25, 11, 13, 18, 16, 33, 29, 12, 4, 24, 21, 32, 16, 26, 0, 28, 23, 20], holdout 18.6

> 11:41 model job `design` started

---

## Turn 0 -> island #3

**Hypothesis:** The map gets a FOOTPRINT memory: a slowly fading `trail` of where the agent has stood, scrolling with the map. Remembered sources (a `ghost` that lingers at 0.99) are wiped wherever the agent has walked, and smelled sources painted onto trodden ground lose most of their weight, while seen morsels keep full weight. Poisoned cells (`bad`) block the wave. This breaks the smell trap, where the bearing flips and leads the agent back and forth over the same cells. Ground the agent has already searched stops calling, so the wave goes to sources it has not checked yet and to the frontier, and the confinement damping keeps nooks from dominating.

Prediction 27.5, got **24.2** meals, fitness 473, per world [41, 42, 15, 24, 12, 27, 16, 36, 30, 2, 36, 25, 32, 33, 31, 33, 0, 17, 12, 21]; model 42.1 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel thing
channel bad
channel here
channel leak
channel value
channel trail
channel ghost
channel confinement
channel injection

# footprint memory: 1 where the agent stands, fading slowly, scrolls with the map
rule walk:
  when 1
  set trail = max(here, 0.995 * trail)

# sources linger, but are wiped wherever the agent has recently walked
rule linger:
  when 1
  set ghost = max(source, 0.99 * ghost) * (1 - trail)

rule calc_conf:
  when 1
  set confinement = sum8(solid)

# seen morsels stay true; remembered/smelled sources lose weight on trodden ground and in nooks
rule inject:
  when 1
  set injection = max(thing * (1 - bad), max(source, ghost) * (1 - 0.7 * trail)) * (1 - bad) / (1 + 0.5 * confinement)

# Bellman wave; a burned cell does not pass the wave
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.9 * bad)
```

> 11:51 model job `design` started

---

## Turn 1 -> island #4

**Hypothesis:** Every frontier mark is painted at the same 0.12, so a crack in a wall that leads nowhere pulls the agent as hard as the mouth of a big unexplored region. I add an `openness` field (unknown territory, smoothed a few rings out) and weight frontier sources by it, so exploration flows toward where the most unseen ground is, while food sources keep confinement suppression and burned cells stop the wave. On worlds where meals come from exploring, the agent should reach new food faster and waste fewer trips into small pockets.

Prediction 27.0, got **27.2** meals, fitness 539, per world [38, 30, 17, 38, 21, 26, 25, 31, 31, 27, 27, 24, 25, 26, 30, 31, 0, 42, 22, 34]; model 48.1 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel leak
channel value
channel unknown
channel openness
channel frontier
channel confinement
channel injection

# what is not yet known (neither seen free nor seen solid)
rule known:
  when 1
  set unknown = (1 - free) * (1 - solid)

# how much unknown lies around a cell, spread a few rings out (bounded below 1)
rule spread:
  when 1
  set openness = 0.5 * avg8(unknown) + 0.45 * avg8(openness)

# the faint frontier marks: a source that is neither smell nor a seen morsel
rule mark:
  when 1
  set frontier = where((source > 0) and (source < 0.13) and (thing < 0.01) and (scent < 0.01), 1, 0)

rule calc_conf:
  when 1
  set confinement = sum8(solid)

# frontier counts by the unknown territory behind it; food as before; burned cells inject nothing
rule inject:
  when 1
  set injection = where(frontier > 0, source * (0.3 + 2 * openness), max(thing, source)) * (1 - bad) / (1 + 0.5 * confinement)

# Bellman wave; a burned cell does not pass the wave
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.9 * bad)
```
