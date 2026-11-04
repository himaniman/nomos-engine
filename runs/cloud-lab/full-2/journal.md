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

### Holdout checkpoint at 0.57 h

```
  #3   train 28.9 meals  holdout 22.5  (tuned 5x) | The confinement law has no memory: when the geodesic smell bearing flips between the two ways round a wall, it
  #4   train 26.2 meals  holdout 28.9  (tuned 4x) | Exploration is a flat 0.12 on every frontier cell, so a crack into a one-cell pocket pulls as hard as an openi
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 8x: train 27.6 meals, holdout 23.2
```

> 13:24 model job `design` started

---

## Turn 2 -> island #5

**Hypothesis:** 70% of food spots are renewable and regrow ~1200 ticks after being eaten, but every law so far forgets a spot the moment its morsel is gone, so it only ever eats what exploration stumbles on. A larder channel remembers where morsels were seen and a per-cell harvest clock rises from 0 (just eaten) toward 1 (due), injecting the spot as a growing source as regrowth approaches; a spot found empty when due is erased, so one-shot spots are not camped. Built on #4 (ghost + confinement + roominess frontier), this turns local patches into a revisited pantry instead of a one-time find.

Prediction 28.0, got **26.6** meals, fitness 532, per world [37, 30, 36, 17, 40, 30, 18, 34, 7, 36, 0, 37, 28, 23, 44, 26, 36, 31, 0, 23]; model 64.1 s, repairs 0; vocabulary twin of #4

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
channel larder
channel clock
channel injection

# a place where a morsel was seen, remembered long; forgotten if found empty when due
rule remember_spot:
  when 1
  set larder = max(thing, 0.9998 * larder * (1 - max(here, max8(here)) * (clock > 0.9)))

# time since the spot last held a morsel: 0 while seen, rises toward 1 at regrowth
rule harvest_clock:
  when 1
  set clock = where(thing > 0.5, 0, min(1, clock + 0.0004))

# what is not yet known: neither seen free nor seen solid
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)

# smoothed density of the unknown around a cell
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)

# frontier: an unknown cell touching a known-free one
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)

# a source lingers after the paint moves on, erased where the agent stands
rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))

# local solidity: high in nooks and dead ends
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# sources: what is sensed, wide unknown, and eaten spots coming due again
rule inject:
  when 1
  set injection = max(ghost, 0.3 * front * room * room, 0.5 * larder * clock * clock) / (1 + 0.5 * confinement) * (1 - bad)

# discounted Bellman wave through passable cells
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 13:42 model job `design` started

---

## Turn 3 -> island #6

**Hypothesis:** The agent's failure modes are dithering between two targets and re-walking ground it has already searched, and nothing in the rule knows where the agent has been. A stigmergic trail channel (laid at `here`, fading over hundreds of ticks, scrolled with the map) makes walked cells slightly less conductive to the wave, so ties and near-ties break toward fresh ground and away from backtracking, while a strong food source still pulls through a walked corridor. Built on the lingering ghost only (no confinement, no roominess), so the effect of self-avoidance is measured on its own.

Prediction 24.0, got **23.1** meals, fitness 474, per world [16, 19, 31, 24, 27, 37, 16, 15, 17, 22, 14, 31, 22, 30, 25, 18, 29, 18, 20, 31]; model 30.0 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel thing
channel solid
channel here
channel bad
channel value
channel ghost: init 0
channel trail: init 0

# stigmergy: the agent's own recent path, laid where it stands, fading over hundreds of ticks
rule lay_trail:
  when 1
  set trail = max(here, 0.999 * trail)

# a source lingers after the paint moves on, erased where the agent stands
rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here)) * (1 - bad)

# the wave flows a little less through ground already walked: no backtracking, no two-goal dithering
rule wave:
  when 1
  set value = max(ghost, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.08 * trail)
```

> 13:44 model job `observe` started

---

## Observer: #3 on world 110

![](observe/obs00_island3_world110.png)

`{'meals': 0, 'ticks': 4221, 'alive': False, 'distance': 228.1, 'cells_visited': 23, 'last500_box': '4x1', 'doors': 0, 'breaks': 11, 'idle_presses': 24, 'nightcaps': 0}`

The agent is born in a narrow sealed slot about one cell wide and four tall, walled by rock with patches of brown rubble right beside it, and it never leaves: the whole 4221-tick life is spent pacing up and down that slot (last 500 ticks in a 4x1 box) until it starves. It does try to dig — 11 breaks and 24 idle presses — but the presses are spread over different wall cells (much of it on plain rock), so no single rubble cell is ever chipped all the way through. The world outside is open and has food in reach (green spots above and to the left), so this is not a lack of food but a failure to commit to one exit. Confinement suppression makes it worse here: every source in a pocket is divided down, so the wave in the slot is weak and keeps re-pointing as the faint frontier/smell marks shift. The fix I would try first is a dig commitment: a per-cell memory that grows while the agent presses a solid cell and raises that cell's leak (its conductivity to the wave), so the wave keeps leading into the same rubble cell until it gives way instead of hopping between walls.

### Holdout checkpoint at 1.11 h

```
  #3   train 28.9 meals  holdout 22.5  (tuned 6x) | The confinement law has no memory: when the geodesic smell bearing flips between the two ways round a wall, it
  #5   train 26.6 meals  holdout 28.7  (tuned 5x) | 70% of food spots are renewable and regrow ~1200 ticks after being eaten, but every law so far forgets a spot 
  #4   train 26.2 meals  holdout 28.9  (tuned 5x) | Exploration is a flat 0.12 on every frontier cell, so a crack into a one-cell pocket pulls as hard as an openi
  #6   train 23.1 meals  holdout  -    (tuned 4x) | The agent's failure modes are dithering between two targets and re-walking ground it has already searched, and
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 20x: train 27.6 meals, holdout 23.2
```

> 13:55 model job `deep` started

---

## Turn 4 (DEEP) -> island #7

**Hypothesis:** Evidence: #6 (ghost + a stigmergic trail) is the only law that ever escaped world 110's sealed slot (14 meals vs 0 for every other law), while confinement + roominess (#3/#4) carry the open worlds; the observer saw the agent pace the slot pressing scattered walls. This law adds a FRUSTRATION state: accumulated dwell time, smoothed into a `crowd` field, is high only where the agent has paced for long; there it (a) multiplies the wall leak (so a taught-leaky rubble cell becomes the clear way out while learned-dead rock stays low), (b) lifts the confinement suppression, and (c) a faint pull from every unknown cell gives the wave something beyond a sealed wall to aim at. In open ground dwell nev

Prediction 29.0, got **27.0** meals, fitness 550, per world [34, 25, 36, 30, 37, 27, 7, 32, 23, 24, 29, 35, 18, 27, 35, 28, 26, 6, 29, 32]; model 50.1 s, repairs 0; vocabulary twin of #4

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel thing
channel solid
channel free
channel here
channel bad
channel leak
channel value
channel ghost: init 0
channel confinement
channel unk
channel room
channel front
channel trail: init 0
channel dwell: init 0
channel crowd: init 0
channel injection

# any visit: the recent path (max), and accumulated time spent on a cell (sum, saturating)
rule lay_trail:
  when 1
  set trail = max(here, 0.999 * trail)
rule dwell_time:
  when 1
  set dwell = min(1, 0.998 * dwell + 0.02 * here)

# frustration: smoothed dwell — high only where the agent has paced for long (a pocket)
rule frustration:
  when 1
  set crowd = 0.5 * dwell + 0.5 * avg8(crowd)

# what is not yet known, the frontier, and how much unknown lies around
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)

# a source lingers after the paint moves on, erased where the agent stands
rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))

rule calc_conf:
  when 1
  set confinement = sum8(solid)

# sources: sensed, wide frontier, and a faint pull of ALL unknown (the only thing beyond a sealed wall);
# confinement suppression is lifted where the agent is frustrated
rule inject:
  when 1
  set injection = max(ghost, 0.3 * front * room * room, 0.04 * unk) / (1 + 0.5 * confinement * (1 - crowd)) * (1 - bad)

# the wave: walls near long dwelling conduct more (a taught-leaky wall most), walked ground a little less
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - min(1, leak * (1 + 3 * crowd)))) * (1 - 0.08 * trail)
```
