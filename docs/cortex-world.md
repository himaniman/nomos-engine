# Design note — the `cortex` world: tile-stored, float-physics, playable first

Status: DESIGN, nothing built. Companion to `cortex-design.md` (which covers the
brain, the lineage and the LLM loop). Written 2026-07-21. This note is only about
the world, its senses, its economy and its HUD.

Three principles everything below follows from:

1. **The world is built and PLAYABLE before any brain exists.** A human at the
   keyboard is the calibration and the fastest detector of a broken economy.
2. **Senses carry physical properties, never semantic labels.** The world says
   "green, smells sweet, 4 cells away", never "food". Edibility is a conclusion the
   brain must reach, not an input it is handed.
3. **Every capability must unlock something nothing else unlocks.** Otherwise the
   brain finds the cheap route and the capability stays decoration — the lesson
   aqua taught (walls, corpses, pheromones, predators, and evolution still produced
   plain foragers, because nothing else PAID).

## Geometry: tiles store it, floats simulate it

Not a choice between the two — a split by role.

| role | representation | why |
|---|---|---|
| storage / authoring | **tile array** `tiles[y][x] -> material id` | seed generation, destruction (`tiles[y][x] = 0`), placement, serialisation are all trivial |
| agent state | **float** (y, x), heading, speed | smooth movement, no cell-hopping, same as `aqua` |
| collision | circle vs the few tiles the body overlaps | the drawn body IS the collision body (aqua's `_blocked_circle` lesson) |
| vision | **DDA ray march** (Amanatides-Woo) over the tile grid | exact hit point and face, and FASTER than aqua's rect list |

This is strictly better than aqua's `wall_rects`: a tile grid is already a perfect
O(1) spatial index, so there is no `_rects_within` pruning to do, and destructible
geometry costs one array write instead of splitting rectangles. Movement stays as
smooth as aqua's — the blockiness aqua removed lived in the PHYSICS reading a grid,
not in the storage being one.

Carried over from aqua without change: sub-step marched movement (nothing tunnels
a wall or skips a pickup at any speed), sliding along the free axis on a wall hit,
the deterministic spatial bucket for entities.

## Build the world first, and play it yourself

Stage 0 is the world, its renderer and keyboard control — no brain at all.

- **The keyboard writes to the SAME action interface the brain uses**
  (thrust / turn / interact). One code path, or you are playtesting a different
  game than the one the agent lives in.
- A human run is the **baseline**: "a human collected 12 food in 2000 ticks" is
  what gives the brain's "3" a meaning.
- If the world is boring or trivially solvable for a human, it will be for the
  brain too. This is the cheapest bad-economy detector we have.
- Human play traces are worth recording — they are raw material for designing the
  probe suite later.

## The HUD

Three panels, added in this order as the project grows:

**Left — the world, top-down, agent-centred** (GTA2 camera). This is the
observer's view: everything visible, tiles, entities, the agent in the middle.
This is NOT what the agent perceives.

**Right — the senses and motors.** The playable-honest view:

- **Vision as a radar fan**: a semicircle covering the FOV, one wedge per ray,
  radius = distance to the hit, colour = the perceived hue/brightness of what was
  hit. Reads like an old green-phosphor radar with coloured contacts — near/far and
  colour legible at a glance.
- **The other modalities by category**, grouped and labelled: odour channels,
  touch (direction + material at contact), interoception (energy, speed, heading,
  age), and later whatever else the world grows.
- **Motor readout**: what the agent is commanding right now.

**Right, later — the brain**: the grown lattice, cell types, organs, axon overlay,
plastic-delta panel, the same inspector aqua already has.

### The sensor panel is a completeness test, not just a display

The stated goal is that the user can play **from the sensor panel alone**, ignoring
the map, to play "honestly" — the same information the brain gets, no more. That
makes the panel a hard test of the sensorium:

> **If a human cannot play the world from the sensor panel alone, the brain has no
> chance either.** A modality that is unreadable to a human is almost certainly
> unusable by the tissue, and a failure to solve a situation from the panel means
> the information needed is not in the senses at all.

This is the cheapest sensory-design feedback loop available, and it costs nothing
beyond a panel we want anyway.

## Senses carry properties, not labels

The central decision of this note. In aqua a vision ray returns 4 channels:
**edible food** / obstacle / KIN / NON-KIN — the world hands over conclusions.
In `cortex` a ray returns what is physically there:

| aqua (labels) | cortex (properties) |
|---|---|
| `food = 0.8` | `hue`, `saturation`, `brightness`, `distance` |
| `obstacle = 1.0` | + `material` class channels (surface / texture) |
| `kin` / `non-kin` | `moving / not moving` + colour — kinship becomes an INFERENCE from similarity, not a fact |

Plus:

- **Smell = several ODORANT channels**, not "smell of food". An object emits a
  mixture; the agent senses a mixture. "Smells like A with a note of B" is
  expressible, which is what the poison experiment needs.
- **Touch** — on contact, direction + material. Nearly free: the collision code
  already computes what was hit. This is the user's "walk right up to it and feel
  the obstacle and its material".

Why this is non-negotiable: **without it the poison experiment is impossible.** If
the world announces "edible", there is nothing to learn. If it reports "purple,
sweet-smelling, 4 away", then "green-sweet is food, purple-sweet poisons you"
becomes a genuine perception-and-memory problem — exactly what the world exists
for.

### Ray count: single-player buys real resolution

aqua's heritable ray count is 3-12 (`vivarium/vision.py`). With one agent the
compute budget supports far more: **64 rays (decided)**, and only at that
resolution do "2x2 trees of slightly different shapes" carry any information at all
— at 8 rays a 2x2 object is one ray and shape is not perceivable in principle. Ray
count stays heritable, with the range raised. Rich perception is a direct
consequence of the one-player decision, not an extra.

**Colour encoding (decided): hue + saturation + brightness per ray**, the physical
option rather than a palette of colour-class channels. Per ray that gives
hue / sat / bright / distance / moving = 5 channels, so vision alone is 64 x 5 =
320 sensor slots; with odorants, touch and interoception the input vector lands
around ~340. Large, but this is a single big brain and `P_in` is only
(n_in, C).

One consequence for the brain: retinotopy in `_build_listen` maps a cell's
horizontal POSITION across the lattice span onto a ray index. With 64 rays a 32-wide
lattice gives two rays per column — workable, but the lattice probably wants to be
wider than aqua's 32 so an "eye" region can resolve individual rays.

## Materials

Start with 6-8, enough for colour and odour to carry information and few enough
that we can tell what the brain learned:

| id | material | solid | breakable | notes |
|---|---|---|---|---|
| 0 | empty | no | — | |
| 1 | ground / floor | no | — | walkable |
| 2 | stone | yes | no | permanent structure, maze walls |
| 3 | soft stone / rubble | yes | **yes** | breaking costs energy + time; shortcuts |
| 4 | tree trunk | yes | yes (slow) | 2x2 objects with foliage |
| 5 | foliage / plant | no | — | edible variants live here |
| 6 | door | yes until opened | no | opens on INTERACT |
| 7 | pushable block | yes | no | moved by INTERACT; makes passages |

Food is not a material but an ENTITY (float position, like aqua's morsels): it has
a colour, an odour mixture, an energy value (possibly negative), and a regrowth
rule. Edible plants sit on foliage tiles; cave food is denser.

## Interaction: one contextual INTERACT

No inventory, no tools, no verb menu — a single output whose effect depends on what
is in front of the agent:

| in front | effect |
|---|---|
| breakable tile | damage it (over several ticks, costing energy) |
| door | open / close |
| pushable block | shove one tile |
| nothing | energy spent for nothing |

So the brain only decides **whether to act now**, never which of five verbs. Vastly
easier to learn, the output space stays tiny, and every behaviour on the wish list
is still reachable.

*Recorded revision*: `CLAUDE.md` carries an old decision that pushable boxes were
set aside because "cells are too simple to manipulate objects; signalling is the
right level". That was about the DSL grid brain in `nomos/`. With a several-
thousand-cell NCA and real motor outputs the question is legitimately reopened —
noting it as an explicit revision rather than pretending the decision never existed.

## The economy

The rule, stated hard:

> **Every capability must unlock something no other capability unlocks.**

| capability | cost | what only it unlocks |
|---|---|---|
| follow an odour gradient | energy per step | baseline food |
| discriminate colour | brain tissue (energy/tick) | not being poisoned |
| **remember a place** | memory tissue | **renewable** food at a fixed spot: returning beats searching |
| wait in place | 0.5 energy (vs 1.0 moving) | a plant regrowing on a timer; letting a threat pass |
| break a tile | energy + several ticks | a shortcut, or food behind a wall |
| push a block | energy | a passage where breaking is more expensive |
| enter a cave | risk | dense food |

Two things this table does not say, and both matter:

**Standing still must sometimes be actively GOOD, not merely cheaper.** forage
already made waiting cheaper (0.5 vs 1.0) so that thinking beats wandering, but
cheap is not a reason. A plant on a regrowth timer, or a threat that loses
interest, is a reason.

**The economy has to be TUNED, not declared.** Memory only pays inside a window:
if food regrows instantly, walk in circles; if it regrows very slowly, keep
exploring. Whether "go back to the patch I remember" beats "search randomly" is a
function of regrow time, world size and travel cost — numbers we must pick and then
verify. The human baseline is the first instrument for that.

Food is both **renewable** (fixed spots on a timer — rewards memory and territory)
and **non-renewable** (rewards exploration), with a tunable ratio. Having both is
what makes the strategy space non-trivial.

## Death

- **Energy is life**: energy reaches 0 -> death. This is the primary stake.
- **Age is a cap**, not the cause — it stops a life dragging on forever.
- Other circumstances can kill (later: animals, hazards).

Correction to an earlier framing: if a life simply ends on a timer, hunger stops
being a stake — not eating costs nothing. The journal already records where that
leads (IMMORTAL brains that learned to run AWAY from food and circle forever,
score ~0). Starvation stays real.

## Poison

Two mechanical conditions, or the experiment is dead on arrival:

1. **Survivable.** Poison DRAINS energy over some number of ticks instead of
   granting it, and does not kill outright. A dead agent learns nothing, and
   in-life plasticity is exactly what we want to measure.
2. **Distinguishable only by a SECONDARY cue.** Same odour, different hue — or the
   reverse. If poison differs on every channel at once the task degenerates.

That yields two clean, separable questions: does the brain learn the cue **within a
life** (plasticity), and does it get written **into the genome** across lives
(evolution / the LLM)?

## Caves and geodesic smell

Caves are just a region generated as a dense maze, with higher food density and
more risk. They break straight-line smell: aqua's inverse-square odour would leak
"through" a wall into a dead end. `nomos/forage.py` already solved this — **BFS
scent maps that flow AROUND walls** (the journal calls it the decisive fix; sealed
pockets are correctly scentless). BFS is native on a tile grid.

New wrinkle destruction adds: **breaking a tile invalidates the scent cache.**
Recompute on a map-change event, rate-limited to at most once every N ticks. The
world computes diffusion; the brain only thinks — that division stays.

## Reproducibility

Unchanged contract: a run is a pure function of (world seed, genome, steps) — the
tile map is generated from the seed and every subsequent modification is a
deterministic consequence of the agent's actions.

One consequence to respect: **probes need a pristine arena.** A probe generates its
small world fresh from its own fixed seed, never from the (possibly demolished)
live world.

## What gets forked

| from | what |
|---|---|
| `aqua/world.py` | float agent state, sub-step marched movement, circle collision, spatial bucket, corpse/entity handling |
| `aqua/app.py` | split render, HUD layout, inspector panels, screenshot keys |
| `nomos/forage.py` | BFS geodesic scent, food respawn into pockets, walled-world generation ideas |
| `vivarium/vision.py` | heritable ray count / reach (range raised) |

New code, no precedent: tile storage + DDA rays, destruction, doors and pushing,
the property-based sensorium, the radar sensor panel, keyboard play.

## Staged plan for the world

1. ~~**Tile world + seed generation + top-down render.**~~ DONE 2026-07-21.
2. ~~**Float agent + collision + keyboard play.**~~ DONE 2026-07-21. Caves arrived
   early, with the generator.
3. **Senses**: DDA vision with colour/material, odour channels, touch. **The
   sensor panel**, playable on its own. ← NEXT
4. **Economy v1**: energy, movement/standing costs, food types renewable and not.
   PARTLY DONE 2026-07-22 (food entities, kinds, regrowth, costs); the numbers are
   still untuned and need a human baseline.
5. ~~**Interaction**: breakable tiles, doors, pushing.~~ DONE 2026-07-22, brought
   forward because a world you cannot eat in is not playable.
6. **Geodesic scent + cache invalidation** (caves already exist).
7. **Poison.** The kind (`nightcap`) and its mechanics are already defined —
   negative energy, berry-coloured odour, distinguishable only by hue — but
   `poison_weight` is 0, so none are planted yet.
8. Only then: plug in the brain (`cortex-design.md` stage 0 onward).

**Eating is a DECISION, not a collision** (settled while building step 5): food is
taken by INTERACT and never by walking over it. That is what turns poison into a
perception problem — the agent may cross a nightcap and decline to eat it — instead
of the much cruder "never step there".

Deferred by agreement: animals, predators, seasons, weather.

## Open questions

- How many odorant channels, and does each material/food emit a fixed mixture or a
  generated one per seed?
- FOV width, and is it heritable like ray count?
- Cave generation: same generator with different parameters, or its own?
- What does a human actually score? Needed before any brain number means anything.
- Does the agent have a body-relative "front" for INTERACT, or does it act on
  whatever it is touching?
