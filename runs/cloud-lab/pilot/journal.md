# night lab — 2026-11-04T09:18:50

Model `agent-in-session (mailbox)` | 1.0 h | world default | train [0, 1, 2, 3, 4, 5, 6, 7] | holdout [20, 21, 22, 23, 24, 25, 26, 27] | 4000 ticks/life | tune 5 | deep every 5 | theory every 6 | observe every 8

- anchor #0 `laws/atlas_wave.nomos`: train 10.9 meals [18, 14, 10, 23, 8, 4, 5, 5], holdout 11.0

- anchor #1 `laws/atlas_confine.nomos`: train 16.1 meals [22, 24, 13, 16, 15, 9, 13, 17], holdout 17.2

- anchor #2 `laws/atlas_ghost.nomos`: train 10.4 meals [13, 5, 10, 7, 10, 10, 13, 15], holdout 10.6

> 09:20 model job `design` started

---

## Turn 0 -> island #3

**Hypothesis:** Confinement judged by the 8 neighbours sees a nook but not a dead-end region a few cells deep. A persistent ROOM channel — known free space blurred over the map, kept between thoughts — measures how open the region around each cell is, so sources and frontier marks inside pockets and corridors fade while those in open country (where more food and more frontier lie beyond) win. The agent commits to open regions and stops being pulled into dead pockets.

Prediction 17.0, got **12.5** meals, fitness 255, per world [22, 18, 17, 8, 8, 4, 12, 11]; model 26.0 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel here
channel value
channel room: init 0
channel injection

# Regional openness: known-free space, diffused over the map and remembered.
rule openness:
  when 1
  set room = clamp(0.6 * avg8(room) + 0.4 * free * (1 - solid), 0, 1)

# A source counts in proportion to how open its region is.
rule inject:
  when 1
  set injection = max(thing, source) * (0.3 + 0.7 * room)

# The Bellman wave over the weighted sources.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 09:22 model job `design` started

---

## Turn 1 -> island #4

**Hypothesis:** The agent remembers where it has walked: a TRAIL channel stamped under `here`, scrolling with the map and fading slowly. Cells on the trail pass on less of the wave and their sources count less, so stepping back onto the cell it just left is never the brightest choice — this breaks the two-cell limit cycle at a flipping smell bearing and pushes the agent onto fresh ground, while confinement suppression keeps nooks from dominating.

Prediction 18.0, got **14.5** meals, fitness 281, per world [16, 13, 12, 17, 24, 6, 11, 17]; model 18.1 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel thing
channel here
channel value
channel trail: init 0
channel confinement
channel injection

# Where I have been: stamped under the agent, fading slowly, scrolls with the map.
rule walk:
  when 1
  set trail = clamp(max(here, 0.97 * trail), 0, 1)

# Local confinement, as in the best law.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Sources hemmed in, or on ground already walked, count less.
rule inject:
  when 1
  set injection = max(thing, source * (1 - 0.6 * trail)) / (1 + 0.5 * confinement)

# Bellman wave; a walked cell passes on less, so backtracking is never the best step.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.3 * trail)
```

> 09:25 model job `design` started

---

## Turn 2 -> island #5

**Hypothesis:** All frontier marks are equally faint, so the agent explores the nearest edge of the known map even when only a sliver of unknown lies behind it. A persistent VOID channel diffuses "unknown" (neither free nor solid) over the map, so each frontier mark is weighted by how much undiscovered country lies behind it; the agent heads for big unexplored regions, where unseen food is, instead of nibbling the borders of what it already knows. Confinement suppression is kept.

Prediction 17.0, got **16.2** meals, fitness 316, per world [18, 17, 15, 13, 16, 21, 14, 16]; model 14.1 s, repairs 0

value on the test map: max 0.50, mean 0.26, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel value
channel void: init 0
channel confinement
channel injection

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# Local confinement, as in the best law.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# A source is worth more where a lot of unknown lies behind it; nooks count less.
rule inject:
  when 1
  set injection = max(thing, source * (0.5 + 1.5 * void)) / (1 + 0.5 * confinement)

# The Bellman wave.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 09:28 model job `design` started

---

## Turn 3 -> island #6

**Hypothesis:** The wave crosses unknown cells as cheaply as walked ones, so a smelled source behind unexplored ground draws the agent straight across terrain that is often wall, and it wastes ticks bumping and re-routing. Here the wave pays an extra toll to cross cells that are neither known free nor known solid (distrust of the unseen), so routes run through known-free ground and the agent reaches food on paths that exist, while the frontier itself (a source on the unknown edge) still pulls through the void weighting of the best law.

Prediction 18.5, got **13.5** meals, fitness 250, per world [18, 12, 12, 18, 14, 9, 9, 16]; model 20.1 s, repairs 0; vocabulary twin of #5

value on the test map: max 0.38, mean 0.20, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel value
channel void: init 0
channel confinement
channel injection
channel toll

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# Local confinement, as in the best law.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# The fraction of the wave a cell passes on: an unknown cell is distrusted.
rule distrust:
  when 1
  set toll = 1 - 0.08 * (1 - free) * (1 - solid)

# A source is worth more where a lot of unknown lies behind it; nooks count less.
rule inject:
  when 1
  set injection = max(thing, source * (0.5 + 1.5 * void)) / (1.3 + 0.5 * confinement)

# The Bellman wave, slower through the unknown.
rule wave:
  when 1
  set value = max(injection, toll * max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 09:29 model job `observe` started

---

## Observer: #5 on world 7

![](observe/obs00_island5_world7.png)

`{'meals': 12, 'ticks': 4000, 'alive': True, 'distance': 881.2, 'cells_visited': 452, 'last500_box': '23x50', 'doors': 0, 'breaks': 0, 'idle_presses': 24, 'nightcaps': 0}`

The agent walks south from the start, finds a patch of regrowing food around two brown rubble mounds west of centre, and spends roughly three quarters of its life circling those two mounds on the same loops, re-eating at the same four or five spots.
It is not stuck or pacing a wall — it eats — but it never leaves a region of about 15x15 cells while the map is full of food dots to the north, east and south that it never sees.
Only in the last quarter (the red track) does it break east through the doorway and eat once more on open ground, which shows the frontier pull exists but is drowned out by the smell of the known patch.
The void weighting cannot help here because the trouble is not the frontier marks but the nearby smell sources, which keep full weight however long the agent has been grazing there.
One mechanism: a slowly fading visit-density ("boredom") channel stamped where the agent walks, which discounts smell sources in heavily grazed ground so that after a few laps the frontier outweighs the local patch.

> 09:32 model job `deep` started

---

## Turn 4 (DEEP) -> island #7

**Hypothesis:** One wave mixes two different wants — food it smells or sees, and unknown ground to explore — so a regrowing patch nearby always outshines the faint frontier and the agent grazes one loop for thousands of ticks (observer, world 7). Here they are two separate waves, FOOD and EXPLORE, and a slow GRAZE field (stamped where the agent walks, diffused, fading over hundreds of ticks) discounts the food wave only where the agent has already been grazing; after a few laps of a patch the explore wave wins and the agent leaves for unseen country, and returns once the boredom has faded and the patch has regrown.

Prediction 19.0, got **15.4** meals, fitness 308, per world [17, 17, 23, 20, 10, 13, 4, 19]; model 40.1 s, repairs 0

value on the test map: max 0.50, mean 0.26, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel void: init 0
channel graze: init 0
channel confinement
channel food
channel explore

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# Boredom: stamped where the agent walks, spread over the patch, fading slowly.
rule grazing:
  when 1
  set graze = clamp(0.9995 * (0.6 * graze + 0.4 * avg8(graze)) + 0.03 * here, 0, 1)

# Local confinement, as in the best law.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# The food wave: only what is smelled or seen.
rule food_wave:
  when 1
  set food = max(where(scent + thing > 0, source, 0) / (1 + 0.5 * confinement), max(0.94 * max4(food), 0.916214 * max8(food))) * (1 - solid * (1 - leak))

# The explore wave: only frontier marks, weighted by the unknown behind them.
rule explore_wave:
  when 1
  set explore = max(where(scent + thing > 0, 0, source) * (0.5 + 1.5 * void) / (1 + 0.5 * confinement), max(0.94 * max4(explore), 0.916214 * max8(explore))) * (1 - solid * (1 - leak))

# Food is worth less where I have been grazing; the stronger want leads.
rule decide:
  when 1
  set value = max(food * (1 - 0.8 * graze), explore)
```

> 09:36 model job `design` started

---

## Turn 5 -> island #8

**Hypothesis:** The wave values every morsel alike, so a lone berry one step away beats a rich patch of five a few steps further, and the agent spends its walking on singletons. A persistent CLUSTER field diffuses food evidence (seen morsels and smell) over the map and remembers it for a while, so a source counts in proportion to how much food lies around it: the agent travels to dense patches where one trip buys several meals (far-but-rich over near-but-poor), while confinement still keeps nooks from dominating.

Prediction 17.0, got **13.8** meals, fitness 286, per world [16, 14, 10, 19, 16, 7, 7, 21]; model 36.1 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel thing
channel scent
channel value
channel cluster: init 0
channel confinement
channel injection

# Food density around a cell: morsels and smell, diffused and remembered.
rule density:
  when 1
  set cluster = clamp(0.85 * (0.5 * cluster + 0.5 * avg8(cluster)) + 0.3 * max(thing, scent), 0, 1)

# Local confinement, as in the best law.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# A source inside a rich patch counts more; nooks count less.
rule inject:
  when 1
  set injection = max(thing, source) * (1 + 2 * cluster) / (1 + 0.5 * confinement)

# The Bellman wave.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 09:37 model job `theory` started

---

## Theory (after 6 design turns)

Helps: suppressing sources hemmed in by solid (confinement, #1: 16.1 vs plain wave 10.9) and weighting each source by how much unknown lies behind it (void, #5: 16.2 raw, 17.9 tuned, the only law above #1). Both re-rank WHERE to go, toward open, unexplored country; meals come from exploration. All laws so far are immortal on all 8 worlds, so the score is foraging rate, not survival.
Hurts or does not help: blurred openness of known free ground (#3, 12.5), a walking trail that damps the wave (#4, 14.5), extra toll on unknown cells (#6, 13.5), boosting dense food patches (#8, 13.8) — anything that makes known food or known ground more attractive, or slows the wave, loses. Splitting food and explore waves with grazing boredom (#7, 15.4) helped worlds 2-3 but collapsed world 6 (4).
Untried: commitment to one target, using `bad`, reading doors/leak, far-vs-near discount shape. Caveat: no holdout yet for #5; its +1.7 from tuning may not transfer.

> 09:39 model job `design` started

---

## Turn 6 -> island #9

**Hypothesis:** The Bellman discount makes the nearest frontier mark always win, so the agent explores by dithering into every small unseen pocket beside it — pockets its eyes would clear anyway — instead of striding out. A PROXIMITY field (a short wave from `here`, through non-solid cells) marks ground close to the agent, and frontier marks are discounted there while smell and seen food are not: near food is still taken at once, but exploration becomes long straight strides toward distant unknown, which uncovers more food per tick. Void weighting and confinement are kept.

Prediction 18.5, got **14.5** meals, fitness 293, per world [18, 19, 11, 16, 10, 10, 7, 25]; model 24.1 s, repairs 0

value on the test map: max 0.30, mean 0.16, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel void: init 0
channel proximity
channel confinement
channel injection

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# How close a cell is to the agent, by a short wave from `here` around solids.
rule nearness:
  when 1
  set proximity = clamp(max(here, 0.8 * max8(proximity)) * (1 - solid), 0, 1)

# Local confinement, as in the best law.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Food counts as is; a frontier mark counts by the unknown behind it and less when near.
rule inject:
  when 1
  set injection = max(thing, where(scent > 0, source, source * (0.5 + 1.5 * void) * (1 - 0.7 * proximity))) / (1.3 + 0.5 * confinement)

# The Bellman wave.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 09:43 model job `design` started

---

## Turn 7 -> island #10

**Hypothesis:** Every seen morsel enters the wave at full strength whatever it looks like, although the food differs fourfold in worth (berry 12, fruit 22, mushroom 32) and the eye reports a brightness for each. Here a seen thing is weighted by its BRIGHTNESS (a quality cue, sharpened by a square so differences matter) and silenced where a refusal is remembered (`bad`, and its ring), so the agent walks past a dim berry to a bright fruit or mushroom and never returns to a poison it has burned on; frontier marks keep the void weighting and confinement of the best law.

Prediction 18.0, got **15.5** meals, fitness 286, per world [19, 16, 18, 15, 15, 17, 12, 12]; model 26.1 s, repairs 0; vocabulary twin of #5

value on the test map: max 0.38, mean 0.20, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel bright
channel bad
channel value
channel void: init 0
channel quality
channel confinement
channel injection

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# What a seen morsel is worth: its brightness, sharpened; nothing where I was burned.
rule appraise:
  when 1
  set quality = thing * (0.3 + 1.2 * bright * bright) * (1 - max(bad, max8(bad)))

# Local confinement, as in the best law.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Seen food by its quality; smell and frontier by the unknown behind them.
rule inject:
  when 1
  set injection = max(quality, source * (0.5 + 1.5 * void) * (1 - bad)) / (1.3 + 0.5 * confinement)

# The Bellman wave.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 09:46 model job `design` started

---

## Turn 8 -> island #11

**Hypothesis:** When two goals pull about equally (a smell bearing flipping round a wall, two equidistant frontiers) the agent steps back and forth and loses ticks; my earlier trail law (#4) fought this by damping the whole wave on walked ground, which slowed planning everywhere. Here planning is untouched — the wave runs in its own channel — and only the READ-OUT is biased: a short-lived RECENT mark under the agent's last few cells makes the step back onto them look worse, so the agent keeps going unless the way back is clearly better (momentum, a commitment held for a few steps).

Prediction 18.5, got **13.0** meals, fitness 246, per world [18, 12, 11, 17, 16, 13, 12, 5]; model 30.1 s, repairs 0; vocabulary twin of #5

value on the test map: max 0.38, mean 0.20, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel here
channel value
channel void: init 0
channel recent: init 0
channel confinement
channel injection
channel wave

# The last few cells I stood on: stamped under the agent, gone within a few thoughts.
rule footprint:
  when 1
  set recent = clamp(max(here, 0.9 * recent), 0, 1)

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# Local confinement.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Sources: void-weighted, confinement-suppressed.
rule inject:
  when 1
  set injection = max(thing, source * (0.5 + 1.5 * void)) / (1.3 + 0.5 * confinement)

# Planning: the plain Bellman wave, never damped by where I walked.
rule plan:
  when 1
  set wave = max(injection, max(0.94 * max4(wave), 0.916214 * max8(wave))) * (1 - solid * (1 - leak))

# Read-out: the cells I just left look worse, so I keep going (momentum).
rule readout:
  when 1
  set value = wave * (1 - 0.5 * recent * (1 - here))
```

### Holdout checkpoint at 0.52 h

```
  #5   train 17.9 meals  holdout 12.9  (tuned 7x) | All frontier marks are equally faint, so the agent explores the nearest edge of the known map even when only a
  #10  train 16.0 meals  holdout 10.6  (tuned 6x) | Every seen morsel enters the wave at full strength whatever it looks like, although the food differs fourfold 
  #7   train 15.4 meals  holdout 16.0  (tuned 6x) | One wave mixes two different wants — food it smells or sees, and unknown ground to explore — so a regrowing pa
  #6   train 14.5 meals  holdout  -    (tuned 6x) | The wave crosses unknown cells as cheaply as walked ones, so a smelled source behind unexplored ground draws t
  #8   train 16.2 meals  holdout  -    (tuned 5x) | The wave values every morsel alike, so a lone berry one step away beats a rich patch of five a few steps furth
  #9   train 14.5 meals  holdout  -    (tuned 5x) | The Bellman discount makes the nearest frontier mark always win, so the agent explores by dithering into every
  #4   train 14.5 meals  holdout  -    (tuned 5x) | The agent remembers where it has walked: a TRAIL channel stamped under `here`, scrolling with the map and fadi
  #3   train 12.5 meals  holdout  -    (tuned 8x) | Confinement judged by the 8 neighbours sees a nook but not a dead-end region a few cells deep. A persistent RO
  #0   train 10.9 meals  holdout 11.0  (hand-written) | the reference Bellman wave
  #1   train 16.1 meals  holdout 17.2  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 10.4 meals  holdout 10.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 51x: train 17.4 meals, holdout 11.9
```

> 09:51 model job `deep` started

---

## Turn 9 (DEEP) -> island #12

**Hypothesis:** Food REGROWS where it grew, yet the wave forgets a place the moment the morsel is gone, so the agent either circles a patch blindly (observer, world 7) or abandons it for good. A trap-line forager: a LARDER channel remembers for a long time every cell where a morsel was seen, and a RIPE clock on each cell is reset where the agent walks and grows back slowly, so a remembered place calls again only after it has had time to regrow. This feeds a FOOD wave kept separate from the EXPLORE wave (the split that held up on holdout, #7: 16.0), so the agent explores new country while nothing is ripe and returns along a route of ripened spots when it is.

Prediction 17.0, got **12.4** meals, fitness 244, per world [12, 12, 19, 17, 8, 11, 7, 13]; model 34.1 s, repairs 0; vocabulary twin of #7

value on the test map: max 0.50, mean 0.26, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel here
channel value
channel void: init 0
channel larder: init 0
channel ripe: init 0
channel confinement
channel food
channel explore

# Where food has been seen, remembered for a long time.
rule remember:
  when 1
  set larder = clamp(max(thing, 0.9995 * larder), 0, 1)

# Regrowth clock: reset where I walk (I ate or saw it bare), grows back slowly.
rule ripen:
  when 1
  set ripe = where(max(here, max8(here)) > 0, 0, clamp(ripe + 0.003, 0, 1))

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# Local confinement.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Food wave: seen and smelled food, plus remembered places that have ripened.
rule food_wave:
  when 1
  set food = max(max(thing, where(scent > 0, source, 0), 0.6 * larder * ripe) / (1 + 0.5 * confinement), max(0.94 * max4(food), 0.916214 * max8(food))) * (1 - solid * (1 - leak))

# Explore wave: frontier marks weighted by the unknown behind them.
rule explore_wave:
  when 1
  set explore = max(where(scent + thing > 0, 0, source) * (0.5 + 1.5 * void) / (1 + 0.5 * confinement), max(0.94 * max4(explore), 0.916214 * max8(explore))) * (1 - solid * (1 - leak))

# The stronger want leads.
rule decide:
  when 1
  set value = max(food, explore)
```

> 09:54 model job `design` started

---

## Turn 10 -> island #13

**Hypothesis:** Doors are the cheapest way into new country — a closed room behind one is full of unseen food (mushrooms live in caves) — but the wave treats a door only as a partly passable wall and so routes round it to the nearest open frontier. Here a solid cell whose LEAK has been taught above the innate guess (the body learned that this hue gives) becomes a DOOR source itself, worth more the more unknown lies behind it, so the agent seeks out learned doors and opens them; stone that taught leak downward gets nothing. Void weighting and confinement are kept for the rest.

Prediction 17.0, got **17.9** meals, fitness 338, per world [21, 16, 17, 27, 17, 13, 12, 20]; model 18.1 s, repairs 0; vocabulary twin of #5, #6

value on the test map: max 0.38, mean 0.20, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel value
channel void: init 0
channel door
channel confinement
channel injection

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# A learned door: solid whose leak was taught above the innate guess, with unknown beyond.
rule doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * max8(void)

# Local confinement.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Food and frontier as in the best law; a learned door is a source too.
rule inject:
  when 1
  set injection = max(thing, source * (0.5 + 1.5 * void), 0.8 * door) / (1.3 + 0.5 * confinement)

# The Bellman wave; the door cell keeps its own injection.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (1 - solid * (1 - leak)))
```

> 09:58 model job `design` started

---

## Turn 11 -> island #14

**Hypothesis:** A deterministic wave over a symmetric map leaves the agent like Buridan's ass: two frontiers or two smells of equal worth pull it back and forth, and on the same map it always makes the same choice, so a bad route is repeated. Here every map cell carries a WHIM — a random preference, resampled only rarely so it stays attached to the place as the map scrolls — and frontier and smell sources are scaled by it: ties are broken by a lasting arbitrary preference (a commitment that costs nothing to compute), and different parts of the frontier get tried instead of always the geometrically nearest. Void weighting and confinement are kept.

Prediction 17.5, got **13.1** meals, fitness 249, per world [15, 16, 13, 12, 15, 8, 11, 15]; model 20.1 s, repairs 0; vocabulary twin of #5

value on the test map: max 0.21, mean 0.11, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel value
channel void: init 0
channel whim: init rand()
channel confinement
channel injection

# A lasting arbitrary preference per place: resampled rarely, scrolls with the map.
rule fancy:
  when bernoulli(0.002)
  set whim = rand()

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# Local confinement.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Seen food as is; smell and frontier by the unknown behind them and by whim.
rule inject:
  when 1
  set injection = max(thing, source * (0.5 + 1.5 * void) * (0.4 + 0.6 * whim)) / (1.3 + 0.5 * confinement)

# The Bellman wave.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```

> 09:59 model job `theory` started

---

## Theory (after 12 design turns)

Biggest fact is on holdout: tuning on 8 train worlds overfits. #5 tuned 17.9 train -> 12.9 holdout; control (#1 tuned) 17.4 -> 11.9, while #1 untuned holds 17.2; #10 16.0 -> 10.6. Only #7 (separate food and explore waves) held: 15.4 -> 16.0. Train gaps under ~2 meals are noise.
Helps on train, before tuning: confinement (#1 16.1 vs wave 10.9), void weighting of frontier (#5 16.2), learned doors as sources of unknown country (#13 17.9, best raw, holdout pending). All re-rank where to explore.
Hurts: anything that damps or tolls the wave or biases the step (trail #4 14.5, unknown toll #6 13.5, nearness #9 14.5, momentum #11 13.0), boosting known food (cluster #8 13.8, regrowth memory #12 12.4), random whims (#14 13.1).
Untried: door + two-wave split together; `bad`; judging laws by holdout, not train.

> 09:59 model job `observe` started

---

## Observer: #5 on world 2

![](observe/obs01_island5_world2.png)

`{'meals': 13, 'ticks': 4000, 'alive': True, 'distance': 876.9, 'cells_visited': 601, 'last500_box': '37x36', 'doors': 0, 'breaks': 0, 'idle_presses': 21, 'nightcaps': 1}`

The agent spends the first third exploring well to the north-west and eats about nine times, then in the middle third (pink track) it circles round and round two small walled rooms just west of its start, a few cells across, which have food sitting inside them.
It never gets in: doors 0 and breaks 0, so it paces the outside of the room walls, pulled by the food it sees or smells inside, while the wave routes it round the room instead of into the wall.
Only late (red track) does it give up and walk south-east into open country, where it eats twice more; several other closed rooms with food dots inside are passed by the same way.
The failure is that a wall between the agent and visible food is treated as an obstacle to route around, not as something to press on.
One mechanism: make a solid cell that borders a room holding seen food or smell a source of its own (a door candidate), so the wave leads the agent straight into that wall where it presses INTERACT and opens the room.

> 10:02 model job `design` started

---

## Turn 12 -> island #15

**Hypothesis:** The observer saw the agent circle small closed rooms with food inside for a third of its life (world 2, doors 0): the wave only seeps through a wall at the leak fraction, so going round always looks cheaper and the wall is never pressed. Here a LURE field spreads seen and smelled food a few cells through open ground, and a solid cell that the lure reaches becomes a BREACH source at the lure's strength, so the wave leads straight into the wall between the agent and the food and it presses INTERACT. Food (with breaches) and exploration (frontier weighted by unknown, plus learned doors) run as two separate waves — the split that held on holdout (#7).

Prediction 18.0, got **14.2** meals, fitness 302, per world [10, 24, 12, 18, 15, 10, 13, 12]; model 28.1 s, repairs 0

value on the test map: max 0.50, mean 0.26, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel value
channel void: init 0
channel lure
channel breach
channel food
channel explore

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# Food spreads a short lure through open ground; walls do not pass it on.
rule spread_lure:
  when 1
  set lure = max(thing, scent, 0.7 * max8(lure)) * (1 - solid)

# A wall the lure presses on is a breach: the way in to the food.
rule find_breach:
  when 1
  set breach = solid * max8(lure)

# Food wave: seen and smelled food, and any wall the lure presses on (a breach).
rule food_wave:
  when 1
  set food = max(max(thing, where(scent > 0, source, 0), 0.9 * breach), max(0.94 * max4(food), 0.916214 * max8(food)) * (1 - solid * (1 - leak)))

# Explore wave: frontier by the unknown behind it, and learned doors.
rule explore_wave:
  when 1
  set explore = max(max(where(scent + thing > 0, 0, source) * (0.5 + 1.5 * void), 0.8 * solid * clamp(2 * (leak - 0.5), 0, 1) * max8(void)) / (1 + 0.5 * sum8(solid)), max(0.94 * max4(explore), 0.916214 * max8(explore)) * (1 - solid * (1 - leak)))

# The stronger want leads.
rule decide:
  when 1
  set value = max(food, explore)
```

> 10:06 model job `design` started

---

## Turn 13 -> island #16

**Hypothesis:** Some frontier marks never go away — unknown cells inside sealed rooms or behind rock, seen from outside but unreachable — and they keep pulling the agent round the same walls (the circling of world 2). Here each cell keeps a STALE clock that runs while it stays a bare frontier mark (no smell, no thing) and resets the moment it stops being one, and a frontier mark loses weight as it goes stale: fresh edges of the map pull, old unreachable ones fade, so the agent stops orbiting what it cannot reach. Learned doors, void weighting and confinement (#13, best raw) are kept.

Prediction 18.5, got **16.1** meals, fitness 308, per world [24, 10, 17, 18, 19, 9, 13, 19]; model 22.1 s, repairs 0

value on the test map: max 0.36, mean 0.19, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel value
channel void: init 0
channel stale: init 0
channel door
channel injection

# How long a cell has been a bare frontier mark; reset when it stops being one.
rule age_frontier:
  when 1
  set stale = where(source * (1 - thing) * (1 - min(scent * 100, 1)) > 0, clamp(stale + 0.002, 0, 1), 0)

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# A learned door: solid whose leak was taught above the innate guess, with unknown beyond.
rule doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * max8(void)

# Food as is; frontier by the unknown behind it, fading as it goes stale; doors.
rule inject:
  when 1
  set injection = max(thing, source * (0.5 + 1.5 * void) * (1 - 0.8 * stale), 0.8 * door) / (1.3 + 0.5 * sum8(solid))

# The Bellman wave; a door cell keeps its own injection.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (1 - solid * (1 - leak)))
```

> 10:10 model job `deep` started

---

## Turn 14 (DEEP) -> island #17

**Hypothesis:** The two best-supported pieces never met: separate FOOD and EXPLORE waves (the only structure that held on holdout, #7) and learned DOORS as sources of new country (#13, best raw). Here they are joined, without #7's grazing boredom (which collapsed world 6), and the two wants are kept honest in different ways: the food wave is a SINK at remembered refusals (`bad` and its ring pass no food wave, so a poison patch cannot draw the agent back), while the explore wave carries doors and void-weighted frontier and ignores food altogether. Each wave has one clear job, so tuning has fewer cross-coupled constants to overfit.

Prediction 18.0, got **13.5** meals, fitness 246, per world [13, 13, 12, 14, 13, 15, 18, 10]; model 18.1 s, repairs 0; vocabulary twin of #15

value on the test map: max 0.50, mean 0.26, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel bad
channel value
channel void: init 0
channel door
channel food
channel explore

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# A learned door: solid whose leak was taught above the innate guess, with unknown beyond.
rule doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * max8(void)

# Food wave: seen and smelled food; a remembered refusal and its ring pass nothing.
rule food_wave:
  when 1
  set food = max(max(thing, where(scent > 0, source, 0)) / (1 + 0.5 * sum8(solid)), max(0.94 * max4(food), 0.916214 * max8(food)) * (1 - solid * (1 - leak))) * (1 - max(bad, max8(bad)))

# Explore wave: frontier by the unknown behind it, and learned doors.
rule explore_wave:
  when 1
  set explore = max(max(where(scent + thing > 0, 0, source) * (0.5 + 1.5 * void), 0.8 * door) / (1 + 0.5 * sum8(solid)), max(0.94 * max4(explore), 0.916214 * max8(explore)) * (1 - solid * (1 - leak)))

# The stronger want leads.
rule decide:
  when 1
  set value = max(food, explore)
```

> 10:14 model job `design` started

---

## Turn 15 -> island #18

**Hypothesis:** One discount rate serves two horizons badly: food is worth chasing only when near (a far smell is often a lie round a wall), while unexplored country is worth a long walk. Here two waves with DIFFERENT horizons: a SHORT wave from smelled and seen food that dies within a few cells, and a LONG wave from the frontier and void that crosses the whole map almost undiminished, so near food is always taken, but beyond a few cells the agent commits to the long-range plan of where unknown country lies instead of bending toward every distant smell.

Prediction 17.0, got **11.8** meals, fitness 222, per world [16, 11, 11, 9, 11, 14, 10, 12]; model 14.1 s, repairs 0

value on the test map: max 0.50, mean 0.36, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel value
channel void: init 0
channel near
channel far

# How much undiscovered country surrounds a cell, diffused and remembered.
rule unknown_mass:
  when 1
  set void = clamp(0.7 * avg8(void) + 0.3 * (1 - free) * (1 - solid), 0, 1)

# Short horizon: food, steeply discounted.
rule near_wave:
  when 1
  set near = max(max(thing, where(scent > 0, source, 0)) / (1 + 0.5 * sum8(solid)), max(0.8 * max4(near), 0.75 * max8(near))) * (1 - solid * (1 - leak))

# Long horizon: frontier by the unknown behind it, gently discounted.
rule far_wave:
  when 1
  set far = max(where(scent + thing > 0, 0, source) * (0.5 + 1.5 * void) / (1 + 0.5 * sum8(solid)), max(0.97 * max4(far), 0.958 * max8(far))) * (1 - solid * (1 - leak))

# The stronger want leads.
rule decide:
  when 1
  set value = max(near, far)
```

> 10:17 model job `design` started

---

## Turn 16 -> island #19

**Hypothesis:** A frontier mark that touches one known-free cell is usually a crack or a corner glimpsed past a wall, while one with a broad front of free cells behind it is the open edge of the map, which the agent can actually walk into. Here each frontier mark is weighted by the WIDTH of the known-free front it sits on (free neighbours, counted), so the agent walks into broad openings rather than probing cracks; seen food and smell stay at full weight, and learned doors are kept as sources.

Prediction 17.0, got **17.5** meals, fitness 338, per world [30, 20, 10, 23, 13, 8, 17, 19]; model 12.1 s, repairs 0

value on the test map: max 1.15, mean 0.61, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
channel source
channel solid
channel free
channel thing
channel scent
channel value
channel width
channel injection

# How broad the known-free front behind a cell is.
rule front:
  when 1
  set width = clamp(sum8(free) / 4, 0, 1)

# Food as is; a frontier mark by the width of its front; learned doors.
rule inject:
  when 1
  set injection = max(thing, where(scent > 0, source, source * (0.3 + 1.2 * width)), 0.8 * solid * clamp(2 * (leak - 0.5), 0, 1)) / (1.3 + 0.5 * sum8(solid))

# The Bellman wave; a door cell keeps its own injection.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value)) * (1 - solid * (1 - leak)))
```

### Holdout checkpoint at 1.02 h

```
  #5   train 17.9 meals  holdout 12.9  (tuned 9x) | All frontier marks are equally faint, so the agent explores the nearest edge of the known map even when only a
  #19  train 17.5 meals  holdout 12.4  (tuned 2x) | A frontier mark that touches one known-free cell is usually a crack or a corner glimpsed past a wall, while on
  #13  train 17.9 meals  holdout 13.8  (tuned 7x) | Doors are the cheapest way into new country — a closed room behind one is full of unseen food (mushrooms live 
  #10  train 16.0 meals  holdout 10.6  (tuned 7x) | Every seen morsel enters the wave at full strength whatever it looks like, although the food differs fourfold 
  #16  train 16.1 meals  holdout 17.9  (tuned 5x) | Some frontier marks never go away — unknown cells inside sealed rooms or behind rock, seen from outside but un
  #7   train 15.4 meals  holdout 16.0  (tuned 7x) | One wave mixes two different wants — food it smells or sees, and unknown ground to explore — so a regrowing pa
  #6   train 14.5 meals  holdout  -    (tuned 6x) | The wave crosses unknown cells as cheaply as walked ones, so a smelled source behind unexplored ground draws t
  #15  train 14.2 meals  holdout  -    (tuned 5x) | The observer saw the agent circle small closed rooms with food inside for a third of its life (world 2, doors 
  #0   train 10.9 meals  holdout 11.0  (hand-written) | the reference Bellman wave
  #1   train 16.1 meals  holdout 17.2  (hand-written) | Confinement suppression (night lab 2026-09-23, champion #35): a source hemmed in by solid counts less, so loca
  #2   train 10.4 meals  holdout 10.6  (hand-written) | sources linger (ghost), erased under the agent
  CONTROL: #1's law, constants tuned by random search 96x: train 17.4 meals, holdout 11.9
```

### FINAL worlds (never seen by selection or by the model)

```
reference wave (#0)                 12.6 meals  immortal 8/8
confine, hand-written (#1)          15.2 meals  immortal 8/8
CONTROL: #1 tuned 96x               13.1 meals  immortal 8/8
#16 tuned 5x                        11.8 meals  immortal 8/8
#7 tuned 7x                         14.0 meals  immortal 8/8
#13 tuned 7x                        12.4 meals  immortal 8/8
#5 tuned 9x                         13.1 meals  immortal 8/8
#19 tuned 2x                        15.9 meals  immortal 8/8
```
