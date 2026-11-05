# Night lab summary — 2026-11-04T12:47:58
Model `agent-in-session (mailbox)`, 12.45 h, 38 design turns (7 deep), model busy 0.45 h, waited for other users 0 min, resumed 2x.
```
                                   train  immortal   holdout  immortal
reference wave (#0)                 20.4  17/20    15.5  12/16
confine, hand-written (#1)          23.2  17/20    23.6  13/16
CONTROL: #1 tuned 191x              29.8  19/20    24.6  14/16
#10 (turn 7, tuned 8x)              34.7  20/20    25.4  15/16
#30 (turn 28, tuned 5x)             32.2  20/20    25.7  15/16
#39 (turn 37, tuned 5x)             31.3  19/20    25.2  14/16
#25 (turn 23, tuned 5x)             32.2  20/20    29.4  16/16
#26 (turn 24, tuned 5x)             31.6  20/20    22.6  14/16
```

**FINAL worlds** (32 worlds nobody selected on, not even the model through the prompt):

```
reference wave (#0)                 18.8 meals  immortal 28/32
confine, hand-written (#1)          24.5 meals  immortal 28/32
CONTROL: #1 tuned 191x              26.9 meals  immortal 30/32
#25 tuned 5x                        26.9 meals  immortal 30/32
#4 tuned 5x                         26.2 meals  immortal 29/32
#5 tuned 7x                         27.0 meals  immortal 30/32
#10 tuned 8x                        26.2 meals  immortal 29/32
#30 tuned 5x                        25.6 meals  immortal 29/32
#25 as written                      27.5 meals  immortal 30/32
#4 as written                       26.5 meals  immortal 29/32
```
Design turns: 38 valid (37 first try, 1 after repair), 0 never loaded, 1 constants-only twins. Evaluations: {'llm': 230, 'control': 191, 'holdout': 14}.

Prediction error (|predicted - actual| meals, before tuning): first half 3.7, second half 5.3 over 38 predictions — no gain in prediction.

## Timeline

```
   0.59 h  turns   2  champion #3 train  27.1 hold 21.9375  |  control train  28.2 hold 25.625
   1.16 h  turns   4  champion #5 train  31.6 hold 26.6875  |  control train  28.2 hold 25.625
   1.69 h  turns   6  champion #5 train  31.6 hold 26.6875  |  control train  28.2 hold 25.625
   2.32 h  turns   8  champion #10 train  34.7 hold 25.4375  |  control train  28.2 hold 25.625
   2.89 h  turns   9  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   3.41 h  turns  11  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   3.95 h  turns  12  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   4.51 h  turns  14  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   5.07 h  turns  15  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   5.59 h  turns  16  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   6.11 h  turns  18  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   6.62 h  turns  20  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   7.13 h  turns  23  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   7.71 h  turns  25  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   8.23 h  turns  26  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   8.74 h  turns  28  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   9.28 h  turns  30  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
   9.80 h  turns  31  champion #10 train  34.7 hold 25.4375  |  control train  28.3 hold 27.125
  10.32 h  turns  33  champion #10 train  34.7 hold 25.4375  |  control train  29.8 hold 24.5625
  10.86 h  turns  35  champion #10 train  34.7 hold 25.4375  |  control train  29.8 hold 24.5625
  11.42 h  turns  37  champion #10 train  34.7 hold 25.4375  |  control train  29.8 hold 24.5625
  11.98 h  turns  38  champion #10 train  34.7 hold 25.4375  |  control train  29.8 hold 24.5625
  12.04 h  turns  38  champion #10 train  34.7 hold 25.4375  |  control train  29.8 hold 24.5625
```

## Champion #25

**Hypothesis:** Food in this world comes in clumps (one to two berries per larder room, two fruits per tree, mushrooms in the same caves), so a meal is evidence of more food close by; foraging animals answer that with area-restricted search. A meal (a morsel at the agent's ring) starts a pulse `ars` that fills the map and fades over ~100 ticks; while it lasts, frontier marks within a few cells of the agent are boosted, so after eating the agent combs the nearby nooks before moving on, and otherwise explores as before (unlike the always-near-sighted #18, which lost).

As written: 28.2 meals; tuned 5x: 32.2.

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
channel ars
channel far
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
  when 0.7408
  set dwell = where(here > 0, min(1, dwell + 0.005203), 0.7418 * dwell)

# frustration: the dwell under the agent, broadcast over the map with a discount
rule frust:
  when 1
  set frust = max(here * dwell, 0.95 * max8(frust))

# AREA-RESTRICTED SEARCH: a meal (a morsel at the agent's ring) starts a pulse that fills the
# map and fades over ~100 ticks; food grows in clumps (larders, trees), so search near a meal
rule ars:
  when 1
  set ars = max(thing * max(here, max8(here)), 0.997 * max8(ars))

rule far:
  when 1
  set far = 29 * sqrt((posx() - 0.5524) * (posx() - 0.4828) + (posy() - 0.4828) * (posy() - 0.4828))

# frustrated: known wall faces (solid touching known free) become targets, likelier doors first
rule inject:
  when 1
  set injection = max(max(thing, source * where(thing + scent > 0, 1.027, (0.4611 + 1.5 * vast) * (0.8222 + 2 * ars * exp(0 - far / 4)))) / (0.9063 + 0.6266 * confinement), frust * solid * max8(free) * leak)

# the wave trusts what is known: it crosses a known-free cell at full price, an unknown one dearer
rule wave:
  when 0.8649
  set value = max(injection, max(0.94 * max4(value), 0.6968 * max8(value)) * (0.85 + 0.15 * min(1, free))) * (1 - solid * (1.028 - leak))
```

## Theory, first version (1.64 h)

Exploration is the main food source, and how the frontier is weighted matters most. Helps: weighting frontier marks by the unknown behind them (#3, +4.9 train, but holdout only 21.9, so partly luck); turning wall faces into targets when the agent dwells (#4, holdout 27.5, the best honest number; it lifted a 0-meal world to 32); making the wave dearer through unknown cells (#5, +3.2 train, holdout 26.7). Hurts: giving the smell a lingering memory (#6, -3 raw: it commits to the smell and loses the rich worlds, though no world starved); splitting food and exploration into two waves and adding them (#8, -5: goals must compete, not blend). Neutral: momentum via a trail penalty (#7, about equal, all 20 immortal). Holdout gaps of 1-2 meals are noise; worlds range 0-48. Untried: food clusters, revisiting regrowing spots, poison avoidance, distance-adaptive discount, door-specific planning.

## Theory, last version (11.15 h)

Load-bearing, each shown by a loss on removal: weighting frontier by the unknown space behind it (removed from the sweep law, #37: 21.7 vs 34.7; added to confinement, #3: +5), the sweep does not replace it; the far smell (#29 weakening it: 22.2); the max() wave (#23 potential field: 11.3); raw seen-morsel pull (#19: 26.1); dwell-frustration on wall faces (rescues worlds 110/118). Trust in known ground helps overall but starves world 112 (#32 removing it: 112 back, total -7); lifting it while frustrated evens lives out (#33, #36: all or most immortal) without raising the mean. Reshaping the frontier's distance (#16, #18, #34), orthogonal waves (#17, #28), memory variants (#21, #22) and additions to food value (#30 boost smell, #31 clumps, #35 chaining) were neutral or worse. Honest numbers: area-restricted search after meals (#25) holds 29.4 on holdout, tuned control 24.6; sweep (#10) is best on train only.

GIF: `champion.gif` (world 101, first 4000 ticks).
