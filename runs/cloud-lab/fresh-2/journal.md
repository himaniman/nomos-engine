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
