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
