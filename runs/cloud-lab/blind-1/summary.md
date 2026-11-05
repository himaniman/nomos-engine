# Night lab summary — 2026-11-04T11:35:16
Model `agent-in-session (mailbox)`, 12.47 h, 31 design turns (6 deep), model busy 0.44 h, waited for other users 0 min, resumed 0x.
```
                                   train  immortal   holdout  immortal
reference wave (#0)                 14.8  11/20    19.6  15/16
confine, hand-written (#1)          25.7  19/20    25.7  14/16
CONTROL: #1 tuned 159x              29.2  19/20    28.2  15/16
#26 (turn 23, tuned 5x)             33.7  20/20    32.5  16/16
#30 (turn 27, tuned 5x)             33.2  20/20    30.6  15/16
#32 (turn 29, tuned 5x)             31.9  19/20    29.0  16/16
#29 (turn 26, tuned 5x)             33.5  20/20    32.2  16/16
#33 (turn 30, tuned 2x)             32.5  20/20    31.3  16/16
```

**FINAL worlds** (32 worlds nobody selected on, not even the model through the prompt):

```
reference wave (#0)                 18.8 meals  immortal 28/32
confine, hand-written (#1)          24.5 meals  immortal 28/32
CONTROL: #1 tuned 159x              25.9 meals  immortal 29/32
#26 tuned 5x                        33.0 meals  immortal 32/32
#29 tuned 5x                        32.8 meals  immortal 32/32
#33 tuned 2x                        31.9 meals  immortal 32/32
#30 tuned 5x                        30.7 meals  immortal 32/32
```
Design turns: 31 valid (31 first try, 0 after repair), 0 never loaded, 0 constants-only twins. Evaluations: {'llm': 191, 'control': 159, 'holdout': 20}.

Prediction error (|predicted - actual| meals, before tuning): first half 2.4, second half 2.7 over 31 predictions — no gain in prediction.

## Timeline

```
   0.54 h  turns   2  champion #4 train  25.8 hold 27.625  |  control train  25.7 hold 25.6875
   1.12 h  turns   4  champion #5 train  26.6 hold 28.625  |  control train  25.7 hold 25.6875
   1.67 h  turns   6  champion #7 train  28.2 hold 25.375  |  control train  28.2 hold 26.625
   2.20 h  turns   8  champion #7 train  28.2 hold 25.375  |  control train  28.2 hold 26.625
   2.78 h  turns  10  champion #11 train  28.9 hold 28.0625  |  control train  28.2 hold 26.625
   3.28 h  turns  11  champion #11 train  28.9 hold 28.0625  |  control train  28.2 hold 26.625
   3.81 h  turns  13  champion #15 train  29.6 hold 29.5625  |  control train  28.2 hold 26.625
   4.32 h  turns  14  champion #15 train  29.6 hold 29.5625  |  control train  28.2 hold 26.625
   4.89 h  turns  15  champion #15 train  29.6 hold 29.5625  |  control train  28.2 hold 26.625
   5.46 h  turns  17  champion #15 train  29.6 hold 29.5625  |  control train  28.2 hold 26.625
   5.98 h  turns  18  champion #15 train  29.6 hold 29.5625  |  control train  28.2 hold 26.625
   6.50 h  turns  19  champion #15 train  29.6 hold 29.5625  |  control train  28.2 hold 26.625
   7.05 h  turns  20  champion #22 train  30.8 hold 30.125  |  control train  28.2 hold 26.625
   7.60 h  turns  21  champion #22 train  30.8 hold 30.125  |  control train  28.2 hold 26.625
   8.15 h  turns  23  champion #24 train  30.8 hold  31.0  |  control train  28.2 hold 26.625
   8.73 h  turns  24  champion #26 train  33.7 hold  32.5  |  control train  29.2 hold 28.25
   9.25 h  turns  25  champion #26 train  33.7 hold  32.5  |  control train  29.2 hold 28.25
   9.83 h  turns  26  champion #26 train  33.7 hold  32.5  |  control train  29.2 hold 28.25
  10.39 h  turns  27  champion #26 train  33.7 hold  32.5  |  control train  29.2 hold 28.25
  10.96 h  turns  28  champion #26 train  33.7 hold  32.5  |  control train  29.2 hold 28.25
  11.54 h  turns  30  champion #26 train  33.7 hold  32.5  |  control train  29.2 hold 28.25
  12.05 h  turns  31  champion #26 train  33.7 hold  32.5  |  control train  29.2 hold 28.25
```

## Champion #26

**Hypothesis:** Every frontier mark weighs the same 0.12 whether it borders a wide unexplored region or a one-cell crack between walls, so the agent spends walks checking slivers. A blurred mass of fog (cells neither seen free nor seen solid) around each cell scales the pure frontier marks, so of the frontiers ahead the agent heads for the one behind which most unknown ground — and so most unfound food — lies. #4 tried this on the old confinement base and had the best UNTUNED holdout of its time (27.6 vs 25.7) before tuning overfit it; it never met ballistic search, the fog-discounted wave or wall trust. Base is #24. (After #25: invented lures on landmarks are dangerous, so this only reweights the honest fr

As written: 33.7 meals; tuned 5x: 33.7.

```nomos
# Frontier worth by what it opens onto. Every frontier mark weighs 0.12 whether it
# borders a wide unexplored region or a one-cell crack between walls. A blurred
# mass of fog (unknown cells) around each cell scales the pure-frontier marks, so
# the agent heads for the frontier behind which most unknown ground lies. (#4 did
# this on an old base: best untuned holdout of its time, 27.6.) Base: #24.
channel source
channel solid
channel free
channel scent
channel thing
channel bad
channel here
channel value
channel confinement
channel trail
channel behind
channel door
channel aversion
channel injection
channel open
channel pinched
channel visits
channel loiter
channel frust
channel fog
channel give
channel vista
rule calc_conf:
  when 1
  set confinement = sum8(solid)
rule remember_path:
  when 1
  set trail = max(here, 0.995 * trail)
rule spread_behind:
  when 1
  set behind = 0.99 * avg8(behind) + 0.01 * trail
rule count_visits:
  when 1
  set visits = clamp(0.998 * visits + 0.02 * here, 0, 10)
  set loiter = visits + sum8(visits)
rule frustration:
  when 1
  set frust = clamp(max(here * (loiter - 2) / 3, 0.999 * frust, 0.999 * max8(frust)), 0, 1)
rule uncertainty:
  when 1
  set fog = clamp((1 - free) * (1 - solid), 0, 1)
rule wall_trust:
  when 1
  set give = clamp((leak - 0.3) / 0.6, 0, 1)
# Fog mass around each cell (bounded screened diffusion).
rule unexplored_mass:
  when 1
  set vista = 0.5 * avg8(fog) + 0.5 * avg8(vista)
rule find_doors:
  when 1
  set door = solid * clamp(2 * (leak - 0.5), 0, 1) * clamp(sum4(free), 0, 1)
rule fear:
  when 1
  set aversion = max(bad, 0.7 * max8(aversion))
rule clearance:
  when 1
  set open = 1 - clamp(solid, 0, 1)
rule pinch:
  when 1
  set pinched = max(shift(here, 1, 1) * (1 - min(shift(open, 1, 0), shift(open, 0, 1))), shift(here, -1, 1) * (1 - min(shift(open, -1, 0), shift(open, 0, 1))), shift(here, 1, -1) * (1 - min(shift(open, 1, 0), shift(open, 0, -1))), shift(here, -1, -1) * (1 - min(shift(open, -1, 0), shift(open, 0, -1))))
rule inject:
  when 1
  set injection = max(where(max(thing, scent) > 0, source * where(thing > 0.5, 1, 1 - frust), source * exp(-40 * behind) * (0.5 + 1.5 * vista)) / (1 + 0.5 * confinement), 0.3 * door * exp(-40 * behind)) * (1 - aversion)
rule wave:
  when 1
  set value = max(injection, 0.94 * max4(value), 0.916214 * max(shift(value, 1, 1) * min(shift(open, 1, 0), shift(open, 0, 1)), shift(value, -1, 1) * min(shift(open, -1, 0), shift(open, 0, 1)), shift(value, 1, -1) * min(shift(open, 1, 0), shift(open, 0, -1)), shift(value, -1, -1) * min(shift(open, -1, 0), shift(open, 0, -1)))) * (1 - solid * (1 - give)) * (1 - 0.5 * aversion) * (1 - pinched) * (1 - 0.15 * fog)
```

## Theory, first version (1.63 h)

Helps: where the agent explores. Ballistic exploration (#7: frontier muted inside a wide field around the walked path, so the frontier ahead wins) is the best idea, 28.2 untuned vs 25.7 for confinement (#1). Weighting frontier by unknown mass behind it (#4, +1.2) also helped on train, but its holdout fell after tuning (27.6 -> 24.4), so treat it as weak. Confinement suppression (#1) is a solid base.
Hurts: damping the wave itself across the path (#3, -1.1); boredom on long-held lures (#6, -1.8); and above all a FASTER wave (#8, two-cell hops, -6.4). That last one suggests the slow, laggy wave is itself useful inertia: a stale field keeps the agent on its current target. Patch weighting (#5) was -1.4 on train but best holdout 28.6 — ambiguous, maybe its (1-bad) term.
Untried: explicit commitment to a target, poison avoidance alone, near-vs-far food trade, door/corridor geometry.
Rests on single runs; per-world spread is large (world 16 always 0, world 18 2-18).

## Theory, last version (11.49 h)

Meals come from exploring for information and from honest prices. Exploration: frontier muted around the walked path (#7, +2.5); frontier weighted by short-range fog mass (#26, 33.7 train, 32.5 holdout, best). It is information, not reach: distance weighting instead fell to 30.1 (#28); a second, regional fog scale added nothing (#32, 31.9). Honest prices: diagonals only where the body fits (#15); unknown cells pass less wave (#22); untaught walls pass less (#24, 31.0 holdout); morsels by learned appetite (#11); confinement kept on seen morsels (#21, -2.4 without); the aversion halo does real work (#31, -2.3 without). Traps: muting smell when loitering (#17).
Hurts: weakening smell's direct pull (#27, -3.1); invented lures (groves #25, -10.9; walls #12); weakening ballistic search (#18); costs on usable cells (#19, #16); slower waves. Neutral: speed (#20), doors by fog (#29), smell out of walls (#30).
Noise ~2.

GIF: `champion.gif` (world 13, first 4000 ticks).
