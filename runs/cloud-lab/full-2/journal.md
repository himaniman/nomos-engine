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

> 14:10 model job `design` started

---

## Turn 5 -> island #8

**Hypothesis:** Axis: the far-vs-near trade-off. One wave gives every source the same 0.94/cell horizon, so a wide frontier ten cells away competes with a narrow one beside the agent and it zig-zags across the map instead of sweeping it. Here food (the source with the harness's 0.12 frontier marks subtracted, lingering) and exploration run as two separate waves with separate horizons — food far-sighted at 0.94/cell, exploration short-sighted at 0.85/cell — and the agent follows the larger; #7's frustration (dwell-opened walls, lifted confinement) and trail are kept as the shared conductance, so nearest-frontier sweeping is the only change.

Prediction 31.0, got **25.7** meals, fitness 518, per world [20, 27, 22, 33, 21, 33, 10, 25, 23, 48, 26, 28, 26, 18, 33, 26, 26, 18, 12, 39]; model 38.1 s, repairs 0; vocabulary twin of #7

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
channel conduct
channel fwave
channel xwave

rule lay_trail:
  when 1
  set trail = max(here, 0.999 * trail)
rule dwell_time:
  when 1
  set dwell = min(1, 0.998 * dwell + 0.02 * here)
rule frustration:
  when 1
  set crowd = 0.5 * dwell + 0.5 * avg8(crowd)
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)
# the harness's own frontier, recomputed: unknown touching known-free
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)
rule calc_conf:
  when 1
  set confinement = sum8(solid)
# how much wave a cell lets through (frustration opens walls, walked ground resists a little)
rule conductance:
  when 1
  set conduct = (1 - solid * (1 - min(1, leak * (1 + 3 * crowd)))) * (1 - 0.08 * trail)

# FOOD alone: the source with the harness's 0.12 frontier marks taken out, lingering
rule linger:
  when 1
  set ghost = max(source - 0.12 * front, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here)) * (1 - bad)

# two waves with their own horizons: food carries far (0.94/cell) ...
rule food_wave:
  when 1
  set fwave = max(ghost / (1 + 0.5 * confinement * (1 - crowd)), max(0.94 * max4(fwave), 0.916214 * max8(fwave))) * conduct
# ... exploration is short-sighted (0.85/cell): sweep the nearest wide frontier, not a far one
rule explore_wave:
  when 1
  set xwave = max(max(0.3 * front * room * room, 0.04 * unk) / (1 + 0.5 * confinement * (1 - crowd)), max(0.85 * max4(xwave), 0.7943 * max8(xwave))) * conduct

rule combine:
  when 1
  set value = max(fwave, xwave)
```

> 14:15 model job `theory` started

---

## Theory (after 6 design turns)

Helps: memory + confinement together (#3 untuned 25.8 vs confine 23.2, ghost 22.7), but tuned #3 fell to 22.5 on holdout, so its tuned constants overfit. Weighting the frontier by how much unknown lies beyond it (#4, #5) generalises best: holdout 28.9/28.7, above train. Escaping sealed pockets is a real, separate failure: every law scored 0 on world 110 until the trail (#6: 14) and dwell-driven frustration that opens nearby walls (#7: 29 there, best train 32.2, holdout pending).
Neutral: the larder/regrowth clock (#5 = #4 on 19 of 20 worlds; spots leave the 25-cell map before they regrow). Trail alone costs meals in open worlds (#6 23.1).
Hurts: short-sighted exploration as a separate fast-decaying wave (#8 25.7 vs #7 27.0 untuned): far frontiers matter.
Untried: poison/appetite use, commitment to one target over thoughts, reading corridor geometry, why #7 lost worlds 116/117.

### Holdout checkpoint at 1.64 h

```
  #7   train 32.2 meals  holdout 27.2  (tuned 5x) | Evidence: #6 (ghost + a stigmergic trail) is the only law that ever escaped world 110's sealed slot (14 meals 
  #3   train 28.9 meals  holdout 22.5  (tuned 7x) | The confinement law has no memory: when the geodesic smell bearing flips between the two ways round a wall, it
  #5   train 26.6 meals  holdout 28.7  (tuned 5x) | 70% of food spots are renewable and regrow ~1200 ticks after being eaten, but every law so far forgets a spot 
  #4   train 26.2 meals  holdout 28.9  (tuned 5x) | Exploration is a flat 0.12 on every frontier cell, so a crack into a one-cell pocket pulls as hard as an openi
  #8   train 25.7 meals  holdout  -    (tuned 3x) | Axis: the far-vs-near trade-off. One wave gives every source the same 0.94/cell horizon, so a wide frontier te
  #6   train 23.1 meals  holdout  -    (tuned 5x) | The agent's failure modes are dithering between two targets and re-walking ground it has already searched, and
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 30x: train 27.6 meals, holdout 23.2
```

> 14:29 model job `design` started

---

## Turn 6 -> island #9

**Hypothesis:** Axis: commitment in the agent's own motion. Every law so far re-decides from scratch each thought, so near-ties between two targets or two frontiers flip the step back and forth (the trail in #6/#7 only fought this indirectly by damping walked ground). Here a fast-fading `recent` memory of the last cells walked gives each neighbour of the agent a `momentum` score (how freshly the cell opposite it, across the agent, was left), and the read-out boosts the planned value there by up to 10%: the agent keeps its heading unless another way is clearly better, which turns dithering into sweeps. The plan itself is #7's wave (frustration, roominess frontier, ghost, confinement) with the trail term remo

Prediction 29.0, got **25.4** meals, fitness 518, per world [23, 24, 31, 30, 27, 20, 28, 25, 26, 31, 29, 25, 26, 10, 32, 24, 23, 20, 28, 26]; model 56.1 s, repairs 0; vocabulary twin of #7

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
channel dwell: init 0
channel crowd: init 0
channel injection
channel recent: init 0
channel momentum
channel plan

# a short memory of the last few cells walked: the cell just left is bright, older ones fade fast
rule recency:
  when 1
  set recent = max(here, 0.95 * recent)

# momentum: a neighbour of the agent scores how freshly the cell OPPOSITE it (across the agent) was left
rule heading:
  when 1
  set momentum = max(shift(here, 1, 0) * shift(recent, 2, 0), shift(here, -1, 0) * shift(recent, -2, 0), shift(here, 0, 1) * shift(recent, 0, 2), shift(here, 0, -1) * shift(recent, 0, -2), shift(here, 1, 1) * shift(recent, 2, 2), shift(here, -1, -1) * shift(recent, -2, -2), shift(here, 1, -1) * shift(recent, 2, -2), shift(here, -1, 1) * shift(recent, -2, 2))

rule dwell_time:
  when 1
  set dwell = min(1, 0.998 * dwell + 0.02 * here)
rule frustration:
  when 1
  set crowd = 0.5 * dwell + 0.5 * avg8(crowd)
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)
rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))
rule calc_conf:
  when 1
  set confinement = sum8(solid)
rule inject:
  when 1
  set injection = max(ghost, 0.3 * front * room * room, 0.04 * unk) / (1 + 0.5 * confinement * (1 - crowd)) * (1 - bad)

# the plan: #7's wave without the trail term (momentum replaces it)
rule wave:
  when 1
  set plan = max(injection, max(0.94 * max4(plan), 0.916214 * max8(plan))) * (1 - solid * (1 - min(1, leak * (1 + 3 * crowd))))

# the read-out: keep going the way you were going unless another way is clearly better
rule commit:
  when 1
  set value = plan * (1 + 0.1 * momentum)
```

> 14:43 model job `design` started

---

## Turn 7 -> island #10

**Hypothesis:** Axis: reading the map's geometry. The wave treats a route through never-seen cells exactly like a route over known ground, so it plans paths through unknown territory that turn out walled, and it cannot tell a doorway (a gap flanked by solid on opposite sides) from a nook, so confinement suppression also mutes the frontier at doors that open new rooms. This law adds a `gateway` field (gap detection with shift) that doubles a frontier source at a gap and exempts it from confinement, and a risk discount on cells not known free, so plans prefer known ground and exits; #7's frustration/trail/roominess substrate is kept unchanged so the geometry is the only thing under test.

Prediction 30.0, got **27.9** meals, fitness 558, per world [25, 37, 31, 25, 34, 21, 33, 34, 37, 38, 31, 18, 1, 15, 47, 22, 28, 19, 24, 39]; model 58.1 s, repairs 0; vocabulary twin of #7, #8

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
channel gateway
channel conduct
channel injection

rule lay_trail:
  when 1
  set trail = max(here, 0.999 * trail)
rule dwell_time:
  when 1
  set dwell = min(1, 0.998 * dwell + 0.02 * here)
rule frustration:
  when 1
  set crowd = 0.5 * dwell + 0.5 * avg8(crowd)
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)

# GEOMETRY: a gap — a cell flanked by solid on two opposite sides (a doorway or corridor mouth)
rule gaps:
  when 1
  set gateway = max(shift(solid, 1, 0) * shift(solid, -1, 0), shift(solid, 0, 1) * shift(solid, 0, -1)) * (1 - solid)

rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# a frontier at a gap opens a new room: worth more; a gap is not a nook, so it escapes confinement
rule inject:
  when 1
  set injection = max(ghost, 0.3 * front * room * room * (1 + gateway), 0.04 * unk) / (1 + 0.5 * confinement * (1 - crowd) * (1 - gateway)) * (1 - bad)

# DISTRUST: a route through cells never seen free may be walled — it conducts a little less than known ground
rule conductance:
  when 1
  set conduct = (1 - solid * (1 - min(1, leak * (1 + 3 * crowd)))) * (1 - 0.08 * trail) * (1 - 0.04 * (1 - free) * (1 - solid))

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * conduct
```

> 14:47 model job `observe` started

---

## Observer: #7 on world 117

![](observe/obs01_island7_world117.png)

`{'meals': 13, 'ticks': 8000, 'alive': True, 'distance': 1490.7, 'cells_visited': 902, 'last500_box': '33x24', 'doors': 0, 'breaks': 3, 'idle_presses': 42, 'nightcaps': 0}`

This world is a town of many small walled rooms with food scattered both outside and inside them; the agent lives the whole 8000 ticks, eats 13 times, and spends most of its life walking loop after loop around the OUTSIDE of room walls (the blue and red tracks trace room perimeters two and three times over). Food dots sit inside rooms whose walls it walks right along, yet it opens no door (doors 0) while pressing walls 42 times for nothing, so the wave is leading it into plain wall cells rather than the rooms' doors. Late in life (red) it is confined to one block of rooms in the upper right, circling the same corridors, so it is not starving for lack of food but for lack of a way in. The trail and frustration terms make walls it has lingered beside more conductive, which spreads the leak over whichever stone it happens to be next to, instead of the one cell that actually opens. The mechanism I would try is a thin-wall detector: a solid cell with free (or a seen thing) on both opposite sides is a door candidate and keeps its leak, while solid cells backed by more solid get almost none, so the wave pours through doorways and the presses land on doors.

### Holdout checkpoint at 2.17 h

```
  #7   train 32.2 meals  holdout 27.2  (tuned 5x) | Evidence: #6 (ghost + a stigmergic trail) is the only law that ever escaped world 110's sealed slot (14 meals 
  #3   train 28.9 meals  holdout 22.5  (tuned 7x) | The confinement law has no memory: when the geodesic smell bearing flips between the two ways round a wall, it
  #10  train 27.9 meals  holdout 28.2  (tuned 4x) | Axis: reading the map's geometry. The wave treats a route through never-seen cells exactly like a route over k
  #5   train 26.6 meals  holdout 28.7  (tuned 5x) | 70% of food spots are renewable and regrow ~1200 ticks after being eaten, but every law so far forgets a spot 
  #4   train 26.2 meals  holdout 28.9  (tuned 5x) | Exploration is a flat 0.12 on every frontier cell, so a crack into a one-cell pocket pulls as hard as an openi
  #8   train 25.7 meals  holdout  -    (tuned 5x) | Axis: the far-vs-near trade-off. One wave gives every source the same 0.94/cell horizon, so a wide frontier te
  #9   train 25.4 meals  holdout  -    (tuned 5x) | Axis: commitment in the agent's own motion. Every law so far re-decides from scratch each thought, so near-tie
  #6   train 23.1 meals  holdout  -    (tuned 5x) | The agent's failure modes are dithering between two targets and re-walking ground it has already searched, and
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 41x: train 27.6 meals, holdout 23.2
```

> 15:01 model job `design` started

---

## Turn 8 -> island #11

**Hypothesis:** The observer saw #7 on world 117 circle the outsides of small rooms full of food, pressing plain wall 42 times and opening no door: the leak is spread evenly over every solid cell, so the wave pours into thick rock as readily as into a one-cell room wall where a door can be. A `thinwall` detector (a solid cell with non-solid on both sides along an axis) keeps the taught leak only on thin walls and cuts thick rock to a third, so plans lead into room walls and doors (where the per-hue leak learning then picks the door) instead of rock masses; frustration still multiplies both, so a sealed pocket can still be dug out. Built on #10 (best untuned, 27.9), the wall-thickness rule is the one change.

Prediction 29.0, got **27.9** meals, fitness 560, per world [41, 31, 39, 28, 24, 35, 32, 35, 18, 33, 27, 20, 1, 18, 29, 29, 27, 33, 24, 34]; model 36.1 s, repairs 0; vocabulary twin of #10, #7

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
channel gateway
channel thinwall
channel conduct
channel injection

rule lay_trail:
  when 1
  set trail = max(here, 0.999 * trail)
rule dwell_time:
  when 1
  set dwell = min(1, 0.998 * dwell + 0.02 * here)
rule frustration:
  when 1
  set crowd = 0.5 * dwell + 0.5 * avg8(crowd)
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)
rule gaps:
  when 1
  set gateway = max(shift(solid, 1, 0) * shift(solid, -1, 0), shift(solid, 0, 1) * shift(solid, 0, -1)) * (1 - solid)

# DOOR CANDIDATE: a solid cell only one cell thick along some axis (a room wall), not a mass of rock
rule thin_walls:
  when 1
  set thinwall = solid * max((1 - shift(solid, 1, 0)) * (1 - shift(solid, -1, 0)), (1 - shift(solid, 0, 1)) * (1 - shift(solid, 0, -1)))

rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))
rule calc_conf:
  when 1
  set confinement = sum8(solid)
rule inject:
  when 1
  set injection = max(ghost, 0.3 * front * room * room * (1 + gateway), 0.04 * unk) / (1 + 0.5 * confinement * (1 - crowd) * (1 - gateway)) * (1 - bad)

# the leak goes where a wall can open: thin walls keep their taught leak, rock mass keeps a third of it;
# frustration still raises both; known ground is trusted over unknown
rule conductance:
  when 1
  set conduct = (1 - solid * (1 - min(1, leak * (0.3 + 0.7 * thinwall) * (1 + 3 * crowd)))) * (1 - 0.08 * trail) * (1 - 0.04 * (1 - free) * (1 - solid))

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * conduct
```

> 15:20 model job `deep` started

---

## Turn 9 (DEEP) -> island #12

**Hypothesis:** Bold axis: AREA-RESTRICTED SEARCH, the classic optimal-foraging rule. Food in this world is clumped by habitat (stocked rooms hold two berries, trees two fruits, caves several mushrooms), yet every law weighs exploration the same everywhere and walks off after a meal. Here a meal (the agent beside a wanted morsel) leaves a LUCK mark pinned to that world spot (it scrolls with the map) and fading over a few hundred ticks; its PATCH (a few cells, falling off with distance) triples the frontier there and lifts confinement suppression, so the agent combs the clump — including the nooks and rooms beside it — before ranging out, and far from luck the law is #7 as before. I also drop #7's trail term

Prediction 29.5, got **26.9** meals, fitness 543, per world [20, 32, 31, 34, 36, 38, 23, 31, 26, 34, 0, 28, 25, 32, 34, 28, 21, 21, 21, 23]; model 58.1 s, repairs 0; vocabulary twin of #7

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
channel dwell: init 0
channel crowd: init 0
channel feast
channel luck: init 0
channel patch: init 0
channel injection

# a MEAL is happening where the agent stands next to a wanted morsel
rule meal:
  when 1
  set feast = here * max(thing, max8(thing)) * (1 - bad)

# LUCK: where meals happened, pinned to the world (it scrolls with the map), fading over a few hundred ticks
rule remember_luck:
  when 1
  set luck = max(feast, 0.998 * luck)

# the PATCH around a lucky spot: a few cells of radius, falling off with distance
rule patch_extent:
  when 1
  set patch = max(luck, 0.8 * max8(patch))

rule dwell_time:
  when 1
  set dwell = min(1, 0.998 * dwell + 0.02 * here)
rule frustration:
  when 1
  set crowd = 0.5 * dwell + 0.5 * avg8(crowd)
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)
rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# AREA-RESTRICTED SEARCH: food grows in patches, so unexplored ground near a recent meal is worth
# up to 3x (and its nooks are not suppressed); far from luck the frontier is weighed as before
rule inject:
  when 1
  set injection = max(ghost, 0.3 * front * room * room * (1 + 2 * patch), 0.04 * unk) / (1 + 0.5 * confinement * (1 - crowd) * (1 - patch)) * (1 - bad)

# #7's wave without the trail term
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - min(1, leak * (1 + 3 * crowd))))
```

### Holdout checkpoint at 2.71 h

```
  #7   train 32.2 meals  holdout 27.2  (tuned 5x) | Evidence: #6 (ghost + a stigmergic trail) is the only law that ever escaped world 110's sealed slot (14 meals 
  #3   train 28.9 meals  holdout 22.5  (tuned 7x) | The confinement law has no memory: when the geodesic smell bearing flips between the two ways round a wall, it
  #11  train 28.2 meals  holdout 25.5  (tuned 5x) | The observer saw #7 on world 117 circle the outsides of small rooms full of food, pressing plain wall 42 times
  #10  train 27.9 meals  holdout 28.2  (tuned 5x) | Axis: reading the map's geometry. The wave treats a route through never-seen cells exactly like a route over k
  #12  train 26.9 meals  holdout  -    (tuned 2x) | Bold axis: AREA-RESTRICTED SEARCH, the classic optimal-foraging rule. Food in this world is clumped by habitat
  #5   train 26.6 meals  holdout 28.7  (tuned 5x) | 70% of food spots are renewable and regrow ~1200 ticks after being eaten, but every law so far forgets a spot 
  #4   train 26.2 meals  holdout 28.9  (tuned 5x) | Exploration is a flat 0.12 on every frontier cell, so a crack into a one-cell pocket pulls as hard as an openi
  #8   train 25.7 meals  holdout  -    (tuned 5x) | Axis: the far-vs-near trade-off. One wave gives every source the same 0.94/cell horizon, so a wide frontier te
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 48x: train 27.6 meals, holdout 23.2
```

> 15:40 model job `design` started

---

## Turn 10 -> island #13

**Hypothesis:** A synthesis test of the two mechanisms that each fixed a hard world without breaking another: heading momentum (#9: worlds 116/117 at 28/20 where the trail law #7 had 7/6, and still out of the world-110 pocket at 29, 20/20 immortal) and thin-wall leak (#11: world 117 6 -> 33 by leading presses into room walls instead of rock), on #7's frustration/roominess/ghost/confinement substrate — with the trail, gateway and unknown-route risk dropped (the last two are the only common factor of #10/#11's collapse to 1 meal on world 112). If momentum + thin walls are additive, the law should keep #9's floor and #11's town worlds.

Prediction 29.0, got **26.1** meals, fitness 511, per world [31, 27, 23, 30, 29, 27, 15, 29, 12, 34, 30, 19, 31, 27, 30, 31, 21, 23, 26, 28]; model 70.1 s, repairs 0; vocabulary twin of #9, #11

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
channel dwell: init 0
channel crowd: init 0
channel recent: init 0
channel momentum
channel thinwall
channel plan
channel injection

rule recency:
  when 1
  set recent = max(here, 0.95 * recent)
# a neighbour of the agent scores how freshly the cell opposite it was left
rule heading:
  when 1
  set momentum = max(shift(here, 1, 0) * shift(recent, 2, 0), shift(here, -1, 0) * shift(recent, -2, 0), shift(here, 0, 1) * shift(recent, 0, 2), shift(here, 0, -1) * shift(recent, 0, -2), shift(here, 1, 1) * shift(recent, 2, 2), shift(here, -1, -1) * shift(recent, -2, -2), shift(here, 1, -1) * shift(recent, 2, -2), shift(here, -1, 1) * shift(recent, -2, 2))
rule dwell_time:
  when 1
  set dwell = min(1, 0.998 * dwell + 0.02 * here)
rule frustration:
  when 1
  set crowd = 0.5 * dwell + 0.5 * avg8(crowd)
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)
# a solid cell one cell thick along some axis: a room wall where a door can be, not rock mass
rule thin_walls:
  when 1
  set thinwall = solid * max((1 - shift(solid, 1, 0)) * (1 - shift(solid, -1, 0)), (1 - shift(solid, 0, 1)) * (1 - shift(solid, 0, -1)))
rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))
rule calc_conf:
  when 1
  set confinement = sum8(solid)
rule inject:
  when 1
  set injection = max(ghost, 0.3 * front * room * room, 0.04 * unk) / (1 + 0.5 * confinement * (1 - crowd)) * (1 - bad)
# the leak lives in thin walls; frustration raises it; no trail term
rule wave:
  when 1
  set plan = max(injection, max(0.94 * max4(plan), 0.916214 * max8(plan))) * (1 - solid * (1 - min(1, leak * (0.3 + 0.7 * thinwall) * (1 + 3 * crowd))))
# keep going the way you were going unless another way is clearly better
rule commit:
  when 1
  set value = plan * (1 + 0.1 * momentum)
```

> 16:01 model job `design` started

### Holdout checkpoint at 3.22 h

```
  #7   train 32.2 meals  holdout 27.2  (tuned 5x) | Evidence: #6 (ghost + a stigmergic trail) is the only law that ever escaped world 110's sealed slot (14 meals 
  #3   train 28.9 meals  holdout 22.5  (tuned 7x) | The confinement law has no memory: when the geodesic smell bearing flips between the two ways round a wall, it
  #11  train 28.2 meals  holdout 25.5  (tuned 5x) | The observer saw #7 on world 117 circle the outsides of small rooms full of food, pressing plain wall 42 times
  #10  train 27.9 meals  holdout 28.2  (tuned 5x) | Axis: reading the map's geometry. The wave treats a route through never-seen cells exactly like a route over k
  #13  train 29.3 meals  holdout  -    (tuned 5x) | A synthesis test of the two mechanisms that each fixed a hard world without breaking another: heading momentum
  #12  train 26.9 meals  holdout  -    (tuned 5x) | Bold axis: AREA-RESTRICTED SEARCH, the classic optimal-foraging rule. Food in this world is clumped by habitat
  #5   train 26.6 meals  holdout 28.7  (tuned 5x) | 70% of food spots are renewable and regrow ~1200 ticks after being eaten, but every law so far forgets a spot 
  #4   train 26.2 meals  holdout 28.9  (tuned 5x) | Exploration is a flat 0.12 on every frontier cell, so a crack into a one-cell pocket pulls as hard as an openi
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 56x: train 27.6 meals, holdout 23.2
```

---

## Turn 11 -> island #14

**Hypothesis:** Bold axis: change the PHYSICS of exploration, not its weights. Every law so far plans exploration with a max-wave, which only ever sees the single best frontier cell, so roominess had to be bolted on to say how much unknown lies beyond it. Here exploration is a HARMONIC potential — each cell takes the discounted MEAN of its neighbours (0.99*avg8, bounded: weights sum below one, no saturation cliff) clamped at frontier sources — so many frontier cells together pull harder than one crack and the field's gradient points at the mass of the unknown; food keeps its own Bellman max-wave (sources with the frontier marks subtracted, lingering), and the agent follows the larger of the two. Frustration

Prediction 28.0, got **18.9** meals, fitness 358, per world [22, 23, 21, 26, 20, 22, 32, 18, 21, 29, 0, 15, 14, 25, 21, 11, 22, 13, 0, 22]; model 84.2 s, repairs 0

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
channel unk
channel front
channel confinement
channel dwell: init 0
channel crowd: init 0
channel recent: init 0
channel momentum
channel conduct
channel morsel: init 0
channel foodplan: init 0
channel lure: init 0
channel plan

rule recency:
  when 1
  set recent = max(here, 0.95 * recent)
rule heading:
  when 1
  set momentum = max(shift(here, 1, 0) * shift(recent, 2, 0), shift(here, -1, 0) * shift(recent, -2, 0), shift(here, 0, 1) * shift(recent, 0, 2), shift(here, 0, -1) * shift(recent, 0, -2), shift(here, 1, 1) * shift(recent, 2, 2), shift(here, -1, -1) * shift(recent, -2, -2), shift(here, 1, -1) * shift(recent, 2, -2), shift(here, -1, 1) * shift(recent, -2, 2))
rule dwell_time:
  when 1
  set dwell = min(1, 0.998 * dwell + 0.02 * here)
rule frustration:
  when 1
  set crowd = 0.5 * dwell + 0.5 * avg8(crowd)
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)
rule calc_conf:
  when 1
  set confinement = sum8(solid)
rule conductance:
  when 1
  set conduct = 1 - solid * (1 - min(1, leak * (1 + 3 * crowd)))

# FOOD only (the harness's 0.12 frontier marks taken out), lingering, erased under the agent
rule food_memory:
  when 1
  set morsel = max(source - 0.12 * front, thing, 0.99 * morsel) * (1 - here) * (1 - max8(here)) * (1 - bad)
# food is planned by Bellman: the single best target, discounted by distance
rule food_wave:
  when 1
  set foodplan = max(morsel / (1 + 0.5 * confinement * (1 - crowd)), max(0.94 * max4(foodplan), 0.916214 * max8(foodplan))) * conduct

# EXPLORATION is a harmonic potential: a cell takes the discounted MEAN of its neighbours, so many
# frontier cells together pull harder than one — the mass of the unknown, not its nearest crack
rule lure_field:
  when 1
  set lure = max(0.3 * front / (1 + 0.5 * confinement * (1 - crowd)), 0.04 * unk, 0.99 * avg8(lure)) * conduct

rule combine:
  when 1
  set plan = max(foodplan, lure)
rule commit:
  when 1
  set value = plan * (1 + 0.1 * momentum)
```

> 16:07 model job `theory` started

---

## Theory (after 12 design turns)

Untuned, every Bellman-based structure since #3 lands at 25-28 meals (#3 25.8 ... #10/#11 27.9): within noise of each other; tuning adds 1-5 on train and mostly gives it back on holdout (#3 28.9->22.5, #7 32.2->27.2). Generalises: roominess-weighted frontier (#4/#5 holdout 28.9/28.7). Hard worlds are mechanism-specific: pocket 110 needs frustration (dwell opens walls) plus a symmetry breaker, trail or momentum (#7, #9: 29; #12 with neither: 0); town 117 needs leak only in thin walls (#11: 33 vs #7 6). Momentum raises the floor (#9, #13: 20/20 immortal, min 10-12) but not the mean. Gateway + unknown-route risk broke world 112 (#10/#11: 1 meal).
Hurts: any non-max exploration physics — harmonic mean field #14 18.9, fast-decaying explore wave #8 25.7. Neutral: larder clock, area-restricted search (#12 26.9).
Untried: why open worlds plateau at ~28 — speed/turning, what eats time between meals.

> 16:22 model job `design` started

---

## Turn 12 -> island #15

**Hypothesis:** Axis: the SCALE of exploration (my theory's open question: why open worlds plateau near 28). The map is only 25 cells and forgets what scrolls off, so a frontier-greedy walker keeps turning back toward nearby cracks and re-sweeps the same blocks (the observer's room-perimeter loops). This law keeps a LONG COURSE — a heading vector averaged over hundreds of ticks, stored at the agent's own cell and gathered back from the ring after each scroll — and, only while exploring (nothing worth 0.3 in play), boosts neighbours that keep that course and damps those that turn back, a Lévy-like straight sweep across new ground; once food is in play the plan alone decides. The substrate is #13's minus its 

Prediction 28.5, got **33.4** meals, fitness 656, per world [28, 27, 37, 27, 31, 34, 33, 41, 31, 32, 33, 46, 30, 41, 29, 31, 39, 32, 32, 34]; model 98.2 s, repairs 0; vocabulary twin of #13, #9

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
channel dwell: init 0
channel crowd: init 0
channel thinwall
channel recent: init 0
channel coursex: init 0
channel coursey: init 0
channel align
channel plan
channel injection

rule recency:
  when 1
  set recent = max(here, 0.95 * recent)
# the agent's LONG COURSE, a vector kept at the agent's own cell: gathered back from the ring after
# the map scrolls (sum8), slowly averaged with the step just taken (away from the freshest cell left)
rule course_x:
  when 1
  set coursex = here * (0.998 * (coursex + sum8(coursex)) + 0.002 * (shift(recent, 1, 0) - shift(recent, -1, 0)))
rule course_y:
  when 1
  set coursey = here * (0.998 * (coursey + sum8(coursey)) + 0.002 * (shift(recent, 0, 1) - shift(recent, 0, -1)))
# each neighbour of the agent: how well stepping onto it keeps the long course
rule alignment:
  when 1
  set align = shift(here, 1, 0) * shift(coursex, 1, 0) - shift(here, -1, 0) * shift(coursex, -1, 0) + shift(here, 0, 1) * shift(coursey, 0, 1) - shift(here, 0, -1) * shift(coursey, 0, -1) + 0.7 * (shift(here, 1, 1) * (shift(coursex, 1, 1) + shift(coursey, 1, 1)) - shift(here, -1, -1) * (shift(coursex, -1, -1) + shift(coursey, -1, -1)) + shift(here, 1, -1) * (shift(coursex, 1, -1) - shift(coursey, 1, -1)) + shift(here, -1, 1) * (shift(coursey, -1, 1) - shift(coursex, -1, 1)))

rule dwell_time:
  when 1
  set dwell = min(1, 0.998 * dwell + 0.02 * here)
rule frustration:
  when 1
  set crowd = 0.5 * dwell + 0.5 * avg8(crowd)
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)
rule thin_walls:
  when 1
  set thinwall = solid * max((1 - shift(solid, 1, 0)) * (1 - shift(solid, -1, 0)), (1 - shift(solid, 0, 1)) * (1 - shift(solid, 0, -1)))
rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))
rule calc_conf:
  when 1
  set confinement = sum8(solid)
rule inject:
  when 1
  set injection = max(ghost, 0.3 * front * room * room, 0.04 * unk) / (1 + 0.5 * confinement * (1 - crowd)) * (1 - bad)
rule wave:
  when 1
  set plan = max(injection, max(0.94 * max4(plan), 0.916214 * max8(plan))) * (1 - solid * (1 - min(1, leak * (0.3 + 0.7 * thinwall) * (1 + 3 * crowd))))
# EXPLORING (nothing worth 0.3 in play): hold the long course, so the walk sweeps far instead of
# turning back for every nearby crack; with food in play the plan alone decides
rule steer:
  when 1
  set value = plan * (1 + 0.5 * clamp(align, -1, 1) * (plan < 0.3))
```

### Holdout checkpoint at 3.86 h

```
  #7   train 34.1 meals  holdout 28.4  (tuned 6x) | Evidence: #6 (ghost + a stigmergic trail) is the only law that ever escaped world 110's sealed slot (14 meals 
  #15  train 33.4 meals  holdout 33.0  (tuned 1x) | Axis: the SCALE of exploration (my theory's open question: why open worlds plateau near 28). The map is only 2
  #3   train 28.9 meals  holdout 22.5  (tuned 7x) | The confinement law has no memory: when the geodesic smell bearing flips between the two ways round a wall, it
  #11  train 28.2 meals  holdout 25.5  (tuned 6x) | The observer saw #7 on world 117 circle the outsides of small rooms full of food, pressing plain wall 42 times
  #10  train 27.9 meals  holdout 28.2  (tuned 5x) | Axis: reading the map's geometry. The wave treats a route through never-seen cells exactly like a route over k
  #13  train 29.3 meals  holdout  -    (tuned 5x) | A synthesis test of the two mechanisms that each fixed a hard world without breaking another: heading momentum
  #12  train 26.9 meals  holdout  -    (tuned 5x) | Bold axis: AREA-RESTRICTED SEARCH, the classic optimal-foraging rule. Food in this world is clumped by habitat
  #5   train 26.6 meals  holdout 28.7  (tuned 5x) | 70% of food spots are renewable and regrow ~1200 ticks after being eaten, but every law so far forgets a spot 
  #0   train 20.4 meals  holdout 15.5  (hand-written) | the reference Bellman wave
  #1   train 23.2 meals  holdout 23.6  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 22.7 meals  holdout 17.9  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 64x: train 29.8 meals, holdout 25.1
```

> 16:56 model job `design` started

---

## Turn 13 -> island #16

**Hypothesis:** #15's long course broke the plateau (33.4 untuned vs 25-28, 20/20 immortal, worst world 27), but it acts on the STEP, only when nothing worth 0.3 is in play. Question: is the gain from holding a heading step by step, or from choosing far GOALS along a course? Here the same long-course vector is broadcast over the map (signed parts carried outward by max8) and weights the frontier SOURCES by where they lie along it — frontier ahead up to 2x, behind down to 0 — while the step read-out is the plain wave again. If the goal-level course matches #15, the mechanism is "sweep in one direction at large scale"; if it falls back toward 28, the step-level hold (not turning back) is what pays.

Prediction 31.0, got **33.0** meals, fitness 655, per world [29, 35, 42, 41, 34, 37, 27, 31, 31, 28, 7, 30, 32, 43, 41, 29, 36, 43, 35, 30]; model 54.1 s, repairs 0

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
channel dwell: init 0
channel crowd: init 0
channel thinwall
channel recent: init 0
channel coursex: init 0
channel coursey: init 0
channel beamxp: init 0
channel beamxn: init 0
channel beamyp: init 0
channel beamyn: init 0
channel ahead
channel injection

rule recency:
  when 1
  set recent = max(here, 0.95 * recent)
# the LONG COURSE at the agent's cell, gathered back after each scroll (as #15)
rule course_x:
  when 1
  set coursex = here * (0.998 * (coursex + sum8(coursex)) + 0.002 * (shift(recent, 1, 0) - shift(recent, -1, 0)))
rule course_y:
  when 1
  set coursey = here * (0.998 * (coursey + sum8(coursey)) + 0.002 * (shift(recent, 0, 1) - shift(recent, 0, -1)))
# BROADCAST the course over the whole map (signed parts carried by max, 3% loss per cell)
rule beam_xp:
  when 1
  set beamxp = where(here, max(coursex, 0), 0.97 * max8(beamxp))
rule beam_xn:
  when 1
  set beamxn = where(here, max(0 - coursex, 0), 0.97 * max8(beamxn))
rule beam_yp:
  when 1
  set beamyp = where(here, max(coursey, 0), 0.97 * max8(beamyp))
rule beam_yn:
  when 1
  set beamyn = where(here, max(0 - coursey, 0), 0.97 * max8(beamyn))
# how far a cell lies along the course, as seen from the agent (cosine x course strength)
rule along_course:
  when 1
  set ahead = ((posx() - 0.4828) * (beamxp - beamxn) + (posy() - 0.4828) * (beamyp - beamyn)) / (sqrt((posx() - 0.4828) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828)) + 0.01)

rule dwell_time:
  when 1
  set dwell = min(1, 0.998 * dwell + 0.02 * here)
rule frustration:
  when 1
  set crowd = 0.5 * dwell + 0.5 * avg8(crowd)
rule unknown:
  when 1
  set unk = (free < 0.2) * (solid < 0.5)
rule roominess:
  when 1
  set room = 0.5 * unk + 0.5 * avg8(room)
rule frontier:
  when 1
  set front = unk * (max8(free) > 0.5)
rule thin_walls:
  when 1
  set thinwall = solid * max((1 - shift(solid, 1, 0)) * (1 - shift(solid, -1, 0)), (1 - shift(solid, 0, 1)) * (1 - shift(solid, 0, -1)))
rule linger:
  when 1
  set ghost = max(source, thing, 0.99 * ghost) * (1 - here) * (1 - max8(here))
rule calc_conf:
  when 1
  set confinement = sum8(solid)
# the course chooses the GOAL, not the step: frontier ahead along the course worth up to 2x, behind down to 0
rule inject:
  when 1
  set injection = max(ghost, 0.3 * front * room * room * clamp(1 + 2 * ahead, 0, 2), 0.04 * unk) / (1 + 0.5 * confinement * (1 - crowd)) * (1 - bad)
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - min(1, leak * (0.3 + 0.7 * thinwall) * (1 + 3 * crowd))))
```
