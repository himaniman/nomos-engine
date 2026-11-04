# night lab — 2026-11-04T11:35:16

Model `agent-in-session (mailbox)` | 12.0 h | world default | train [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] | holdout [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] | 8000 ticks/life | tune 5 | deep every 5 | theory every 6 | observe every 0

- anchor #0 `laws/atlas_wave.nomos`: train 14.8 meals [28, 27, 10, 35, 13, 4, 5, 5, 33, 12, 5, 7, 23, 23, 18, 8, 0, 28, 5, 6], holdout 19.6

- anchor #1 `laws/atlas_confine.nomos`: train 25.7 meals [37, 39, 18, 30, 23, 22, 13, 34, 26, 13, 33, 36, 33, 24, 29, 30, 0, 34, 14, 26], holdout 25.7

- anchor #2 `laws/atlas_ghost.nomos`: train 19.2 meals [28, 5, 25, 11, 13, 18, 16, 33, 29, 12, 4, 24, 21, 32, 16, 26, 0, 28, 23, 20], holdout 18.6

> 11:41 model job `design` started

---

## Turn 0 -> island #3

**Hypothesis:** Keep confinement suppression but add a MEMORY of the walked path (a fading trail channel that scrolls with the map) and damp the wave across it. The cell the agent came from always looks worse than the cells ahead, so equidistant targets and two-way smell bearings stop producing back-and-forth limit cycles and the agent commits to moving on, exploring more ground per tick.

Prediction 27.0, got **24.6** meals, fitness 477, per world [27, 21, 20, 34, 43, 17, 19, 34, 24, 22, 36, 26, 25, 27, 20, 33, 0, 31, 9, 23]; model 32.1 s, repairs 0

value on the test map: max 1.00, mean 0.53, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Inhibition of return on top of confinement suppression: the path just walked
# is remembered and the wave is damped across it, so ties break forward.
channel source
channel solid
channel thing
channel here
channel value
channel confinement
channel injection
channel trail

# Local solidity: a source hemmed in by walls counts less.
rule calc_conf:
  when 1
  set confinement = sum8(solid)

# The walked path, fading; it scrolls with the map so it lies behind the agent.
rule remember_path:
  when 1
  set trail = max(here, 0.97 * trail)

rule inject:
  when 1
  set injection = max(thing, source) / (1 + 0.5 * confinement)

# Bellman wave, damped where the agent has just been.
rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak)) * (1 - 0.3 * trail)
```

> 11:54 model job `design` started

---

## Turn 1 -> island #4

**Hypothesis:** Exploration is where meals come from, but every frontier mark currently weighs the same 0.12 whether it opens onto a wide unexplored region or a one-cell sliver between walls. Here pure frontier marks (no smell, no seen thing) are scaled by a blurred mass of unknown cells around them, so the wave pulls the agent toward large unexplored areas where new food is likely, while food sources stay as in the confinement law.

Prediction 27.0, got **25.8** meals, fitness 505, per world [42, 39, 19, 25, 20, 16, 23, 35, 34, 29, 28, 30, 17, 35, 30, 35, 0, 15, 18, 26]; model 30.1 s, repairs 0

value on the test map: max 0.50, mean 0.26, 0% at the cap -> healthy: the best neighbour points at the source

```nomos
# Frontier weighted by how much unexplored space lies behind it, on top of
# confinement suppression: big unknown regions pull, slivers between walls don't.
channel source
channel solid
channel free
channel thing
channel scent
channel value
channel confinement
channel unknown
channel openness
channel injection

rule calc_conf:
  when 1
  set confinement = sum8(solid)

# Neither seen free nor seen solid.
rule unknown_cells:
  when 1
  set unknown = (1 - free) * (1 - solid)

# A blurred mass of unknown cells around each cell (bounded diffusion).
rule open_area:
  when 1
  set openness = 0.5 * avg8(unknown) + 0.5 * avg8(openness)

# Food (seen or smelt) as before; pure frontier marks scaled by the unknown mass.
rule inject:
  when 1
  set injection = where(max(thing, scent) > 0, max(thing, source), source * (0.5 + 1.5 * openness)) / (1 + 0.5 * confinement)

rule wave:
  when 1
  set value = max(injection, max(0.94 * max4(value), 0.916214 * max8(value))) * (1 - solid * (1 - leak))
```
