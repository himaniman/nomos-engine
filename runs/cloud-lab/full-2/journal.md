# night lab — 2026-11-04T12:47:59

Model `agent-in-session (mailbox)` | 12.0 h | world default | train [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119] | holdout [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135] | 8000 ticks/life | tune 5 | deep every 5 | theory every 6 | observe every 8

- anchor #0 `laws/atlas_wave.nomos`: train 20.4 meals [21, 36, 36, 18, 25, 34, 20, 12, 2, 26, 13, 15, 28, 35, 33, 10, 21, 18, 0, 5], holdout 15.5

- anchor #1 `laws/atlas_confine.nomos`: train 23.2 meals [20, 30, 20, 31, 25, 35, 45, 25, 17, 30, 0, 18, 21, 9, 44, 26, 3, 37, 0, 28], holdout 23.6

- anchor #2 `laws/atlas_ghost.nomos`: train 22.7 meals [21, 27, 29, 40, 21, 7, 14, 5, 12, 23, 16, 27, 28, 23, 34, 12, 12, 30, 25, 48], holdout 17.9

> 12:53 model job `design` started

---

## Turn 0 -> island #3

**Hypothesis:** The confinement law has no memory: when the geodesic smell bearing flips between the two ways round a wall, its source hops and the agent oscillates (the trap the ghost law fixed), while the ghost law lacks confinement and is lured into nooks. Here a lingering ghost of every source (erased under the agent) is injected through the confinement divisor, and remembered refusals (`bad`) mute injection, so the target is both stable over time and discounted where it is hemmed in.

Prediction 25.0, got **25.8** meals, fitness 510, per world [22, 37, 31, 25, 17, 43, 22, 17, 34, 20, 0, 36, 33, 6, 43, 25, 35, 36, 19, 14]; model 26.1 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel thing
channel solid
channel here
channel bad
channel value
channel ghost: init 0
channel confinement
channel injection

# memory: a source lingers after the paint moves on, erased where the agent stands
rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))

# local solidity around a cell: high in nooks and dead ends
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# remembered sources, discounted by confinement and muted where something burned
rule inject:
  when 1
  set injection = ghost / (1 + 0.5 * confinement) * (1 - bad)

# discounted Bellman wave through passable cells
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 13:06 model job `design` started

---

## Turn 1 -> island #4

**Hypothesis:** Exploration is a flat 0.12 on every frontier cell, so a crack into a one-cell pocket pulls as hard as an opening onto a wide unexplored region; most meals come from exploration, so the frontier should be worth what lies beyond it. This law keeps #3's lingering ghost + confinement, and adds a relaxed "roominess" field (smoothed density of unknown cells) that scales a recomputed frontier source, so the agent heads for large unknown areas (and the map edge, beyond which everything is unknown) rather than probing pockets.

Prediction 27.0, got **26.2** meals, fitness 521, per world [37, 30, 27, 17, 40, 30, 18, 34, 7, 36, 0, 37, 28, 23, 44, 26, 36, 31, 0, 23]; model 40.1 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel thing
channel solid
channel free
channel here
channel bad
channel value
channel ghost: init 0
channel confinement
channel unk
channel room
channel front
channel injection

# what is not yet known: neither seen free nor seen solid
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)

# smoothed density of the unknown around a cell (relaxes each pass, bounded in [0,1])
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)

# frontier: an unknown cell touching a known-free one
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)

# memory: a source lingers after the paint moves on, erased where the agent stands
rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))

# local solidity: high in nooks and dead ends
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# frontier worth what lies beyond it: open unknown beats a crack into a pocket
rule inject:
  when 1
  set injection = max(ghost, 0.3 * front * room * room) / (1 + 0.5 * confinement) * (1 - bad)

# discounted Bellman wave through passable cells
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```
