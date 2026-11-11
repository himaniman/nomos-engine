# Design note — `tissue`: a small genome unfolded into a big brain

Status: STARTED 2026-11-11. This note holds what was agreed with the user and what each
exam is expected to show BEFORE it is run. Results go to `docs/lab-journal.md`.

## Why a restart

The user's read of the project (2026-11-11), which the record supports: the evolving
brains of vivarium/aqua got better by growing their PERCEPTRONS, not their tissue. In the
saved pools rule depth sits at its cap (4), hidden width went 16 -> 23-32 (cap 32), cell
types 4 of 4, while cells stayed at 48-55 of 144; depth and width were free, cells cost
energy. Later measurements said the same from the inside: the foragers of `cortex` went
through innate arcs that bypassed the sheet (2026-09-14), and 81% of a sheet's change was
shared by all cells while the read-out averaged them (2026-11-05). Sensors were broadcast
to every cell and the command was a mean, so cells were interchangeable and the only place
to put capacity was the perceptron.

The idea itself (cells with perceptrons of several kinds, axons beyond neighbours, growth
from a genotype) was built once as `cortex/brain.py` and failed for measured reasons: a
selection that could not hear (one genome's spread across worlds 67 > the spread between
its children 48), every A/B at n = 2. So what changes now is the METHOD, not the list.

## The claim

**The same genome, grown bigger, is smarter.** A genome holds rules only: how a cell of
each type computes, how a wire changes, who connects to whom. It is a few hundred numbers
at any brain size. Everything an individual knows lives in the states of its cells and
wires, and those multiply with size. That is the mechanism by which growth must pay.

Three signs, each with the control that could deny it:

| sign | measured as | control |
|---|---|---|
| smarter by growing | one genome, more cells -> a better score, at sizes its search never saw | an ordinary net (GRU) on the same exam |
| memory | the answer depends on what is no longer on the senses | freeze or wipe cell/wire state -> falls to the no-memory level |
| thinking | a harder question needs more inner ticks; the brain says when it is done | cut it short -> wrong |

## Decided with the user (2026-11-11)

1. The goal is research: growth that pays, explicit memory, a visible thinking process.
   The game is a later showcase on the same kernel, not the measuring instrument.
2. Gradient descent may search for the genome if it gives results. Learning within a life
   stays local: a wire sees only the two cells it joins.
3. The perceptron inside a cell is small and FIXED. Capacity grows only by cells and wires.
4. Size is a dial first (grow to M and look). An organism that decides its own growth
   comes last.
5. Exams before worlds: abstract tasks, milliseconds per life, known floor and ceiling.
6. Python first. Rust only for a kernel that has stopped changing, checked bit for bit
   against the Python one. The user's pve host (one free socket: 20 threads, 56 GB, no
   GPU) is for breadth, many seeds for days, not for speed.

## Working rules

- One mechanism per step. The next one waits for a number on at least 3 seeds.
- The bar and my prediction are written here before the run; the journal says met or not.
- Every mechanism has an off-switch, and its ablation is run when it is added.
- A size the search never saw is the only evidence of growth. Trained sizes prove nothing.
- What is by hand and what was found is stated for every result.

## The ladder

| rung | adds | exam | stop if |
|---|---|---|---|
| 0 | plastic wires, one local rule shared by all | associative memory: pairs held vs size | the found rule does not carry to unseen sizes |
| 1 | recurrence: plastic wires among interneurons, taught by no teacher; the senses go silent and the brain runs on its own wires | links A->B, B->C seen apart, asked A->C: hops followed vs ticks, load and size | more ticks do not buy longer chains |
| 2 | wiring from the genome: cell types with affinity, axons by distance | the same exams against random and shuffled wiring | shuffling the wires costs nothing |
| 3 | growth from a seed, size chosen by the organism under a cost | the same exams, with the size it chose | it grows to the cap or not at all |

A map of places is a set of learned chains, so rung 1 is also the road back to an agent in
a world: a map that was learned, not painted by hand as in the atlas.

## Kernel v0 (`tissue/kernel.py`)

Three cell types: S sensory (carry the cue), I interneurons (M of them, the size dial),
O motor (hear the teacher, are read as the answer). Two blocks of wires: S->I innate
(random in every individual, the genome sets only a gain), I->O plastic (start at zero).
Every cell of a type carries the same perceptron, 8 hidden units. A tick, synchronous:
each dendrite AVERAGES its block of wires; the perceptron says the cell's activity and a
learning signal; a plastic wire changes by (source activity) x (target's learning signal).
The genome is 76 numbers.

By hand: the types, the blocks, the average on a dendrite, the product form of the wire
rule. Found: what an interneuron says about its input (its code), what a motor cell does
with its dendrite and the teacher (its answer and its learning signal), the input gain.

## Exam 0: associative memory (`tissue/assoc.py`)

A life is K random pairs, cue (12 bits) -> answer (4 bits), new in every life. Each pair
is shown once (the teacher on the last tick of the cue), then every cue is asked in a
fresh order. Score: bit accuracy, chance 0.5. Headline: **K90**, pairs held at 90% of bits.

Hand-made memories, nothing searched (`python -m tissue.hand`, 2026-11-11):

| interneurons M | hebb | delta, one pass | least squares (not local) |
|---|---|---|---|
| 8 | 4.3 | 5.9 | 12.0 |
| 16 | 7.4 | 8.9 | 25.6 |
| 32 | 12.6 | 16.2 | 51.3 |
| 64 | 20.9 | 28.9 | 102.3 |
| 128 | 34.3 | 46.4 | 210.4 |
| 256 | 52.9 | 78.2 | 419.9 |
| 512 | 78.5 | 123.1 | 847.7 |

No interneurons at all (sensory -> motor Hebb): 7.9, so below about 16 interneurons the
tissue is worth less than a direct wire. The local one-pass rules grow about x1.6 per
doubling, not x2: cues live in 12 dimensions, so their codes stay correlated and more
cells do not average that away. Least squares on the same codes is linear in M (1.6 M):
that gap is what a better code or a better rule could win.

### Bars for rung 0, written before the first search

Search: sizes 8-64, 1/8 to 2 pairs per interneuron, 6000 steps, 3 seeds. Exam: 8-512.

- **B1, control.** Plastic wires frozen: accuracy within 0.5 +- 0.03 at every size, at 4
  pairs and more.
- **B2, growth pays.** On every seed K90 rises at every doubling from 8 to 512, and K90
  at 512 is at least 2.5x K90 at 64. (Hand delta: x4.3. A rule that stops at its trained
  sizes: x1.)
- **B3, the rule is worth finding.** On every seed, at every size from 16 up, K90 is at
  least hand Hebb's.
- GRU (64 and 256 hidden, trained on 1-64 pairs): its K90 is reported beside these. No bar.

My prediction: B1 met. The search finds an error-correcting rule and lands within 20% of
hand delta at 8-64. At 256-512 it falls behind hand delta (K90 per interneuron drops
faster), and B2 is met narrowly or missed at 512. B3 met at trained sizes, open at 512.

**Outcome (journal 2026-11-11, 10:18):** B2 and B3 met on all seeds, about one pair per
interneuron up to 512. B1 missed by the letter (the bar was tighter than the exam resolves;
the control is exact by a test). The prediction was wrong: the search found a hash-like
code and beat hand delta 1.4-3.7x.

## Kernel v1: wires among interneurons (`RUNG1` in `tissue/kernel.py`)

One block added: I->I, plastic, from zero. An interneuron's perceptron now reads two
dendrites (innate from the senses, plastic from the other interneurons) and one global
signal, whether the senses are on; it says its activity and a learning signal for its
I->I wires. The wire rule is unchanged: (source activity a tick ago) x (target's learning
signal now). Nothing outside teaches these wires. Genome: 101 numbers.

The first plan for this rung fed the motor answer back to the sensory cells over
hand-drawn one-to-one wires. Dropped before building: wires among interneurons are the
tissue's own recurrence, they need no teacher, they grow as M^2, and they are what rung 2
(wiring from the genome) will have to shape.

By hand: the global senses-on signal, the rest between episodes (cells fall silent, wires
stay), the teacher's timing. Found: when an interneuron follows its senses and when its
own wires, its code, what it learns from.

## Exam 1: links and chains (`tissue/chain.py`)

Items are 12 bits. A link a->b is its own 3-tick episode: a, then b (the teacher shows the
motor cells `a`), then silence (the teacher shows `b`). The teacher teaches only the ECHO
"say what you just saw"; what follows what is in nobody's signal. Test: the first item for
one tick, then silence; hop h is read at tick h + 2. K = links in a life, K90 per hop.

Step 1a is one hop (single links). Step 1b is chains: links of a chain shown apart, in
random order, asked several hops down.

Hand-made hop (`python -m tissue.hand --hop`): threshold codes, delta rule on the I->I
wires, the recalled code cleaned by the same threshold, a delta-rule decoder.

| M | hand delta, one pass | least squares (not local) |
|---|---|---|
| 8 | 4.0 | 7.3 |
| 16 | 7.7 | 19.8 |
| 32 | 12.5 | 48.7 |
| 64 | 24.8 | 113.6 |
| 128 | 47.7 | 270.8 |
| 256 | 98.5 | 598.1 |

### Bars for step 1a, written before the search

Search: sizes 8-64, 1/8 to 1 link per interneuron, single links only, 6000 steps, 3 seeds.

- **C1, control.** I->I wires frozen: hop-1 accuracy within 0.5 +- 0.03 at 16 links and
  more, every size. (By a test, every walk then says the same thing whatever it began from.)
- **C2, growth pays.** On every seed K90 at hop 1 rises at every doubling from 8 to 512,
  and K90 at 512 is at least 2.5x K90 at 64.
- **C3, worth finding.** On every seed K90 is at least the hand delta hop's at 16-256.

My prediction: C1 met. K90 0.4-0.8 per interneuron at 8-64, above the hand hop. The loss
at 512 is larger than on rung 0 (a recalled code is noisier than a seen one and the
decoder was taught on seen ones): K90(512) / K90(64) between 4 and 6. The code turns less
hash-like and more two-level than on rung 0, because a recalled code has to be cleaned.

A question, no bar: the genome that only ever saw single links, examined on chains of 4.
I expect hop 2 well above chance and a steep fall after: at M/4 links, hop 1 above 0.97,
hop 2 about 0.85, hop 4 about 0.65.

**Outcome (journal 2026-11-11, 10:44):** one seed of three found the hop; two settled on a
constant code that says the bitwise majority of everything that ever came second. On the
seed that worked: C1 met, C2 met narrowly (x2.8; per interneuron 0.74 at 64, 0.26 at 512),
C3 met (and below the hand hop at 512). It stored both the links and the decoder with the
sign turned over, which cancels for one hop and alternates along a chain: hop 2 is right
in 3-16% of the bits, hop 3 in 78-98%.

### Step 1a', the echo is scored: bars written before the search

One change, to the exam and not to the brain: the echo of the first item (hop 0) joins the
error and the score. It is the one thing the teacher teaches, and it was never checked. A
constant code cannot echo, and a decoder with its sign turned over cannot either. Search as
before, 5 seeds.

- **D1, the search is reliable.** Every seed holds links: K90 at hop 1 above zero at every
  size from 16 up.
- **D2 = C2** (growth), on every seed.
- **D3, the hop composes.** The same genomes, never shown a chain, on chains of 4 at M/4
  links: hop 2 at least 0.8 at sizes 16-128, on every seed.

My prediction: 4 or 5 seeds of 5 leave the plateau (another flat spot may exist: echo
right, hop still the majority). D3 met wherever D1 is. K90 per interneuron no better than
this run's good seed, and the fall at 256-512 stays.

**Outcome (journal 2026-11-11, 11:20):** D1, D2, D3 met on five seeds of five, the
prediction held. Links per interneuron 0.68-0.74 at 64, 0.34-0.40 at 512. A chain of 4
never shown is walked without the sign alternating; depth of a walk grows only weakly
with size. A lesson for every later exam: score what the teacher teaches.

### Step 1b, the search is on chains: bars written before the search

Nothing changes in the brain or the exam; the search now sees chains of 1-4 links (echo
scored), sizes 8-64, 1/8 to 1 link per interneuron, 6000 steps, 5 seeds. Exam: chains of 4
and chains of 8 (a depth never seen), sizes to 512. The 1a' genomes sit the same exam as
the baseline.

- **E1, asked for, a walk gets longer.** 64 cells, chains of 4 at M/4 links: hop 4 at
  least 0.92 on every seed (1a': 0.83-0.85).
- **E2, a depth never seen.** 64 cells, chains of 8 at M/4 links: hop 8 at least 0.75 on
  every seed.
- **E3, size buys depth.** Links walked 4 hops at 90% (K90 at hop 4): at 512 cells at
  least 4x that at 64, on every seed (1a': 2.3-2.5x).

My prediction: E1 met. E2 open, hop 8 somewhere in 0.7-0.8. E3 missed: the examiner sets
the pace, one tick a hop, so a brain of any size has no time to clean a recalled code
before it must hand it on.

**Outcome (journal 2026-11-11, 11:44):** E1 missed by one seed of five (0.913; the others
0.92-0.93, up from 0.83-0.85). E2 met (0.82-0.87). E3 missed (x1.8-2.7), as predicted.
Searching on chains doubles the links walked four hops and carries to twice the depth
seen; it does not make size buy depth.

## A brain that never stops (decided with the user, 2026-11-11)

The plan had a step "the brain says when it is done". The user objected: a question in,
an answer out, then stop, is how a perceptron or an LLM is used; the idea here is a brain
whose thinking never stops. The brains already found agree: left alone for 60 silent
ticks, four of five keep running and wander along learned links (`tissue.alone`). And the
objection named two habits of the exams themselves: every cell was silenced between
episodes, by hand, and the examiner set the pace of a walk.

Not stopping is different from acting: a brain that runs for ever still commits to an act
at some moment, and the time it takes is the measurable sign of thinking. And not stopping
is not a virtue in itself: thinking that wears out what was learned is worse than rest.
So "says when it is done" is replaced by THE STREAM, four properties, each an exam:

1. No resets. A life is one stream, and the score holds on streams longer than the
   search saw.
2. In silence the brain is neither dead nor noise.
3. Idle thinking does not wear memory out; better, it mends it.
4. Its own pace: the order of what is said is scored, not the tick.

Order agreed: 1 and 3 first.

## Exam 2: the stream (`tissue/stream.py`)

The world is a few loops of items; after an item comes the next on its loop, for ever.
Study: eyes open, each loop walked round `laps` times, loops in random order (leaving one
loop for another shows a pair that is not a link, once). Test: `blinks` times the world
jumps, the eyes see 3 items and close for `closed` ticks while the world goes on. Every
tick the motor cells are read as "what the world showed a tick ago"; while the eyes were
open a tick ago the teacher shows exactly that, so the only teacher is the senses a tick
late. The brain is never reset. K = items = links. A perfect memory of the stream (the
most frequent successor of each item) scores 1.000 at every hop with two laps; with one
lap a link and a coincidence have each been seen once and nothing tells them apart.

The brain is the same `RUNG1`; only the life changes. The 1a' genomes, made for rested
episodes, in the stream at 64 cells and 16 items: echo 0.94-0.99, hops 1-4 0.93 / 0.84-0.88
/ 0.77-0.83 / 0.71-0.76 (rested: 0.99 / 0.91 / 0.89 / 0.84). After an idle closing of 32
ticks hop 1 is 0.89.

### Bars for the stream, written before the search

Search: `--arm stream`, sizes 8-64, 1/16 to 1/2 item per interneuron, loops of 4 or 8,
2-3 laps, eyes closed 4 ticks, no idle closing, 6000 steps, 5 seeds. Exam: loops of 8, 2
laps, sizes to 512.

- **F1, made for the stream, it lives in it.** 64 cells, 16 items: hop 1 at least 0.97 and
  hop 4 at least 0.90, on every seed.
- **F2, growth pays.** K90 at hop 1 rises at every doubling 8 to 512 and K90(512) is at
  least 2.5x K90(64), on every seed.
- **F3, idleness does no harm (never asked for).** After an idle closing of 64 ticks, hop 1
  at 64 cells and 16 items falls by no more than 0.02, on every seed.
- **F4, a closing longer than any it saw.** Eyes closed 16 ticks (the search: 4), 64 cells,
  16 items: hop 16 at least 0.75 on every seed.

My prediction: F1 met. F2 met with the usual fade (x3 to x5). F3 missed: nothing asked the
brain to hold its wires still while it wanders; I expect a fall of 0.03-0.08. F4 open,
my guess is near the bar: a loop closes on itself, so a walk round it could correct itself,
but nothing in a 4-tick closing rewards that.

**Outcome (journal 2026-11-11, 11:58):** three seeds of five found the hop; two settled on a code
that is a linear copy of the cue (echo right, hops near chance). So F1, F2 and F4 are
missed as written ("every seed"). On the three that worked: F1 met by two (hop 4 0.93-0.94)
and missed by one (0.87); F2 met, and for two seeds growth is LINEAR for the first time on
this rung (K90 at hop 1 x8.8-9.3 from 64 to 512, no fade); F3 met on every seed, idleness
costs nothing, my prediction was wrong; F4 met (0.75-0.85), with the caveat that hop 16 on
a loop of 8 is the item last seen.

### The search starts from the previous step's genome: bars written before the run

Twice now a search from random numbers has settled on a degenerate CODE (constant on rung
1a, linear here) that some seeds never leave. One change, to the search only: the stream
search starts from the five chain genomes of step 1b (`--init`), the same brain plan, so
five lineages instead of five seeds. Everything else as above.

- **G1, every lineage lives in the stream.** F1 on all five (64 cells, 16 items: hop 1 at
  least 0.97, hop 4 at least 0.90).
- **G2, growth without a fade is the rule.** K90 at hop 1: 512 cells at least 6x 64 cells,
  on all five.

My prediction: G1 met by four or five of five. G2 met wherever G1 is.

**Outcome (journal 2026-11-11, 12:15):** no lineage got stuck, which was the point. By the letter
both bars are missed: G1 by two lineages that sit 0.002 and 0.005 under hop 4 = 0.90 (hop
1 is 0.986-0.996 on all five), G2 by two lineages (x5.0 and x6.0 against x8.9-9.5 on the
other three). Growth without a fade is frequent, 5 of the 8 genomes that live in the
stream, not the rule. On loops of 16, never seen, with the eyes closed 12 ticks, hop 12 is
0.67-0.79 at 64 cells: the clean figure for a long closing.

**Kept as method:** a search for a new life starts from the genome of the life before.

## A thought that holds (chosen with the user, 2026-11-11)

With the eyes closed the walk drifts within 8-16 ticks. The user chose this over "own
pace" as the next thing to mend. Diagnosis first, on the genomes in hand (128 cells, a
closing of 64 ticks, per lap of a loop of 8):

- It drifts at EVERY load, even with one loop of 8 items in 128 cells (lap 1: 1.00, lap 8:
  0.63-0.83). Not a matter of capacity.
- In silence the brain WHISPERS: interneurons' mean |activity| is about 1 with the eyes
  open, 0.37 in the first closed lap, 0.14-0.22 from the third lap on, and stays there.
- It is the pattern that goes, not the loudness: the match between the interneurons'
  pattern and the code of the item the world is really at falls 0.97 -> 0.31-0.76 over 8
  laps. At a whisper a cell works on the straight part of its curve, where nothing
  saturates, so nothing pulls a slightly wrong pattern back.

Why whisper? A guess: a loud thought is a liability when the eyes reopen, because a wire
changes by (source activity) x (learning signal) and a cell cannot tell that the source
activity was its own thought and not the world. Whispering keeps those false links small.

### Step: ask for long closings, no new mechanism. Bars written before the search

The cheapest thing first, as on step 1b: ask. The stream search, started from the five
lineage genomes, now sees closings of 4, 8, 16 or 32 ticks (`--hops 32 --vary`). Nothing
else changes. Exam: a closing of 64 ticks (twice the longest seen) on loops of 8, where a
stretch of 8 ticks is one lap, so every lap holds the same items.

- **H1, the thought holds.** 64 cells, 8 items: lap 8 at least 0.95, on every lineage
  (now 0.63-0.79).
- **H2, and not by unlearning.** K90 at hop 1 at 64 and at 512 cells is at least 0.8 of
  the lineage's own figure before.
- **H3, a bigger brain holds more in thought.** The number of items whose lap 8 is at
  least 0.9 rises at every doubling from 64 to 512 cells, on every lineage.

My prediction: H1 missed, or met at the price of H2. Asked, the brain will think louder
and lap 8 will rise to 0.8-0.9, but capacity will fall, because of the false links at
reopening. If so the next step is ONE mechanism that lets a cell tell its own thought
from the world: private state in a cell, or a global signal a tick late.

**Outcome (journal 2026-11-11, 12:49):** H1, H2, H3 met on all five lineages; my
prediction was wrong on every count. No new mechanism was needed: asked, the brain stops
whispering (loudness 0.57 through the closing, pattern 0.98 at every lap), loses no
capacity, and the best lineage walks its loop for 256 ticks, eight times the longest
closing it was searched on. Items held in thought grow x6-7 from 64 to 512 cells.

**Kept as method:** ask before adding a mechanism. The private state of a cell stays
unbuilt until an exam needs it.

## Rung 2: wiring (started 2026-11-11, the user's choice)

Until now every interneuron was wired from every other one (M^2 wires) and there was one
kind of interneuron. The user's idea is the opposite: cells of several kinds, joined by
axons that a genotype lays down. Three things hide in that, taken one at a time:

| step | adds | the question |
|---|---|---|
| 2a | a few wires per cell (`Plan.fan`), drawn at random at birth | with the wires per cell fixed, does a bigger brain still hold more? |
| 2b | a second kind of cell | can a kind of cell make growth pay again where 2a says it does not? |
| 2c | cells in a space, wires by distance and by kind, their statistics in the genome | does wiring laid down by the genome beat wiring at random with the same number of wires? (the shuffle control) |

Why 2a first: it is the floor every later wiring is read against, and it is what lets a
brain be big at all. All-to-all costs M^2 numbers per individual; `fan` wires per cell
cost M x fan, so 4096 cells with 32 wires each are cheaper than 512 cells all-to-all.

### Step 2a: a few wires per cell. Question and prediction, written before any run

No search yet: the five "hold" genomes are grown with `fan` wires into each interneuron
from random other interneurons (sensory and motor wires stay as they are) and sit the
stream exam with a closing of 64 ticks. Sizes 64 to 2048, fan 8 to 128.

What is asked: K90 at hop 1 and the items held in thought, against the size at a fixed
fan, and against the fan at a fixed size.

My prediction, from the theory of diluted associative memories: **what a brain holds
follows the wires per cell, not the cells.** At a fixed fan both figures stay flat from
256 to 2048 cells (within a quarter), and both roughly double when the fan doubles. A
code in which about half the cells speak for every item needs every cell to learn every
item, so more cells with the same wires each add nothing. If that is what comes out, it
is the real obstacle to "grown bigger, smarter" under wiring a body could afford, and step
2b has its job: a kind of cell that keeps few cells speaking per item, so that different
items live in different cells.

**Outcome (journal 2026-11-11, 13:21):** half right. Recall one step ahead still grows with the
cells at a fixed fan, about x1.3 per doubling: the motor cells read every interneuron, so
recall is a vote. A thought that HOLDS does not grow with the cells at all: about 0.55
items per wire (18 at 32 wires, 35 at 64, 70 at 128) at 512, 1024 and 2048 cells alike.
Under affordable wiring the claim "grown bigger, smarter" is true for what is recalled at
once and false for what is kept in mind.

### Step 2a', a search that lives with few wires: bars written before the search

Ask before adding. The stream search (closings 4-32 as before), started from the five
"hold" genomes, now lives in brains of 64, 128 and 256 cells with 8, 16 or 32 wires into
each interneuron, and 1/4 to 2 items per wire (`--fan 16 --sizes 64 128 256`). No new
mechanism. Exam at 32 wires (and 16, 64), 64 to 2048 cells, closing of 64 ticks.

- **K1, a search that knows about few wires uses them better.** At 32 wires and 512 cells
  the items held in thought are at least 1.5x the unsearched figure (17-18), every lineage.
- **K2, and growth returns.** At 32 wires the items held in thought at 2048 cells are at
  least 1.5x those at 512 cells, every lineage (unsearched: x1.0-1.1).

My prediction: K1 met, a gain by a constant factor from a code better suited to few
wires. K2 missed: one kind of cell cannot make fewer cells speak per item as the brain
grows, because a cell has no way to know how big the brain is.

**Outcome (journal 2026-11-11, 15:22):** K1 missed (x0.69-1.05, no gain), K2 missed
(x1.05-1.26). Recall one step ahead did improve (+15-43%). But the search asked the wrong
thing: 62% of its lives carried more than any of these brains can hold in thought (0.55
items per wire), so only recall was left to earn. K1 is not a fair verdict.

### Step 2a', asked properly: the same bars, written before the second search

One change, to the lives of the search: 1/8 to 1 item per wire (71% of lives at or under
what can be held), and bigger brains in the search, 128-512 cells with 8, 16 or 32 wires,
so that most of its lives are in brains with many more cells than wires per cell
(`--fan 16 --sizes 128 256 512 --load 0.125 1`, 4000 steps). Five lineages from the
"hold" genomes, on the VM. Bars K1 and K2 as above, unchanged; 64 wires per cell is a
fan the search never saw.

My prediction: K1 met modestly, x1.2-1.5, now that holding is what the lives pay for. K2
missed again: the reason I gave does not depend on how well the search is asked.

**Outcome (journal 2026-11-11, 16:15):** K1 missed, x1.00-1.13: asked properly the search
loses nothing and gains nothing in holding; my prediction was wrong. K2 missed, x1.04-1.15,
as predicted. Recall one step ahead improves again (+11-15% at 512 cells, +23-51% at 2048).
Half an item per wire (0.51-0.60 at 32 wires, 0.43-0.57 at 64 wires never searched) is what
this rule gives with wires drawn at random. Genomes: `runs/tissue/few2/`.

**A lesson for every search:** put the loads where the thing being asked for is possible.

### Step 2b, decided by a yardstick before any search: no kind of cell for sparse codes

The plan for 2b was a second kind of cell that keeps few cells speaking per item, "so that
different items live in different cells" and holding grows with the brain. Three cheap
checks came first (journal 2026-11-11, 16:20). They say not to build it.

1. **The found brains use a dense code.** The share of interneurons that speak for an item
   (`python -m tissue.drift`, the row "speak") is 0.81-0.89 in every brain looked at: eyes
   open or closed, all-to-all or 32 wires per cell, 128 to 2048 cells.
2. **The ceiling by hand** (`python -m tissue.hand --think`). Fixed codes; every cell fits
   its own K wires for as long as it likes; an inhibition that never errs picks who speaks.
   Items held for 64 ticks alone:

   | wires per cell | half the cells speak | 1/4 | 1/8 | 1/16 | cells |
   |---|---|---|---|---|---|
   | 32 | 32 | 32 | 24-32 | 0-8 | the same at 64, 128, 512, 2048 |
   | 64 | 72 | 72 | 72 | 48 | the same at 128, 512, 2048 |
   | 128 | 168 | - | 168 | 256 | 1024 |

   About one item per wire, whatever the share of cells that speak and whatever the number
   of cells. The found brains hold 0.55 per wire: half of that ceiling, with a rule that
   sees each link two or three times. Few cells speaking per item pays only from about 128
   wires per cell (x1.5), and never with the cells.
3. **What a wire buys.** At the same number of wires in all, a small brain wired
   all-to-all beats a big one with 32 wires per cell on both counts (five lineages each):

   | wires in all | all-to-all: cells | held | K90 hop 1 | 32 per cell: cells | held | K90 hop 1 |
   |---|---|---|---|---|---|---|
   | 4 096 | 64 | 13-24 | 40-42 | 128 | 10-11 | 36-38 |
   | 16 384 | 128 | 24-50 | 90-98 | 512 | 12-18 | 65-80 |
   | 65 536 | 256 | 63-95 | 120-213 | 2048 | 16-19 | 75-112 |

Why, by counting. A cell can be taught only through its own K wires. If every cell speaks
for every item, every cell must learn every link: about K of them, however many cells there
are. If few cells speak per item, a cell has fewer links to learn but hears fewer of its
sources speak, and the two cancel until K is large. And in a brain wired at random two
small groups of cells have hardly a wire between them (K x group / cells), so "different
items in different cells" could not be linked at all. My expectation in step 2a was wrong.

So with wires drawn at random, what a thought can hold is a matter of wires per cell and of
nothing else. Three roads are left:

- **More wires per cell as the brain grows.** That is rung 1, all-to-all: it works, and it
  costs cells squared.
- **Hubs.** A few cells that hear everyone outvote the noise; the motor cells already do,
  which is why recall grows with the cells. A thought could pass through them ("the brain
  hears its own voice": a kind of cell that hears the motor cells and is read back by
  them). Counted through, not built: such a loop takes two ticks, so it must learn links
  two steps apart, and the stream shows enough to tell such a link from a coincidence only
  from the third lap on. It would also move the thought out of the tissue into twelve
  cells. An option for the user, not the plan.
- **Places.** Wires to neighbours, and each loop kept in one small patch of cells. By
  arithmetic on the table above, where a patch of 64 cells with 32 wires each holds as
  much as 2048 cells do: 2048 cells in 32 patches could hold about 1000 items where they
  now hold 18. What it needs is something that says which patch. That is what step 2c,
  cells in a space, is really about.

### Step 2c: cells in a space. Plan, bars and predictions, written before any run

Interneurons sit on a ring (the simplest space; a sheet comes later). Two dials, both set by
hand for now, both statistics of the wiring that a genome could carry:

- `Plan.reach`: a cell's few wires come from cells within `reach` places on either side.
  `reach` 0 is step 2a, wires from anywhere, and it is the SHUFFLE control: the same number
  of wires, drawn without regard to place.
- `Plan.field`: a sense of place. The world is somewhere along the ring and every
  interneuron feels how near that is to itself (a bell `field` cells wide). In the stream
  every loop then lies at a place of its own, drawn at random, felt on every tick with the
  eyes open or closed. The sense says where, never what.

By hand: the ring, wires to neighbours, the sense of place laid out along the ring. Found:
what a cell does with the sense (whether it speaks, whether it learns) and everything else.

**2c-1, neighbours alone. No search, no sense of place.** The five genomes of step 2a'
grown with 32 wires per cell from within 32 places, against 32 wires from anywhere, at 512
and 2048 cells. Prediction: nothing changes. Holding stays within a quarter of the
shuffled figure, recall one step ahead is the same or a little lower (neighbours share
sources, so their errors agree and a vote gains less). A code that knows no place cannot
use wiring that does.

**Outcome (journal 2026-11-11, 18:05):** nothing changes, as predicted. Held x0.93-1.02
of the shuffled figure at 512 and at 2048 cells, recall one step ahead x0.99-1.05. (A
preview with wires drawn with repeats had shown a loss; a third of those wires were
duplicates, and the wiring now takes every wire from a different neighbour.)

**2c-2, a sense of place. A search.** From the five genomes of step 2a' (the new input of
a cell starts at zero, so each begins exactly as it lived): `--fan 16 --reach 32 --field 12
--sizes 128 256 512`, closings up to 32 ticks as before. Loads are counted per wire and
per patch of the ring (a patch = 2 x reach cells), 1/8 to 1 item per wire per patch, so
that about 70% of the lives are at or under what a brain that kept each loop in its own
patch could hold. (Such a life is up to a thousand ticks long, so the search lives fewer
of them at once: `ROOM` in `tissue/train.py`.) Exam: 32 wires, reach 32, 128 to 4096 cells, a closing of 64 ticks.

- **P1, growth returns.** Items held in thought at 2048 cells are at least 2x those at
  512 cells, on every lineage (placeless: x1.0-1.3). 1024 and up are sizes never searched.
- **P2, above the ceiling of a placeless brain.** At 2048 cells at least 100 items are
  held (the hand ceiling with 32 wires and no places is 32; the found brains hold 16-19).
- **P3, the shuffle.** The same genomes with their wires drawn from anywhere (`--reach 0`)
  hold less than half of that at 2048 cells: it is the wiring by place that pays.
- **P4, the sense off.** With nothing felt (`--nowhere`) holding falls back to what a
  placeless brain holds (at most 32).

My prediction: P1-P4 met on at least three lineages of five, with 0.1-0.25 items per cell
at 2048 (200-500 items). By arithmetic a patch holds what its wires allow and patches
multiply with the ring, so like rung 0 this would be an instrument, not a discovery. What
I am least sure of is the search: every cell now speaks for every item, and a cell that
falls silent away from its place costs recall before it pays in holding.

**Amended before any result** (a first start was stopped at step 200 of 4000). Counting
through what a brain with places would look like, I found a by-hand choice of the kernel
that would have decided the outcome. A plastic dendrite is an average over its wires, and a
motor cell is wired from every interneuron. If only a patch speaks, what the motor cells
read, and how far one lesson moves them, shrink as patch / cells: 0.29 at 128 cells, 0.07
at 512, 0.018 at 2048. A genome found at 128-512 cells could not carry to 2048 for that
reason alone, and the bars would have tested my normalisation, not places.

`Plan.spoken`: a plastic dendrite is scaled by what was SAID to it (the squares of what its
sources said, summed, plus one) instead of by the number of its wires. Then neither what
it reads nor how far one lesson moves it depends on how many of its sources speak. For a
code where every cell speaks it is the old average up to a constant, and the 2a' genomes
still live under it before any search (lineage 0 at 256 cells: 9 items held against 14, K90
at hop 1 52 against 54). The search is the one above plus `--spoken`. Bars P1-P4 unchanged.
The shuffle (P3) and the sense off (P4) are run on the same genomes, so they carry the same
dendrite; if the bars are met I owe one more control, a search with `spoken` alone, to show
what the new dendrite does without places.

**The ceiling by hand**, computed while the search ran and before any result
(`python -m tissue.hand --think --patch 36 --reach 32`): a sense of place that never errs,
every cell fitting its own 32 wires, each loop spoken for by the 36 cells at its place.

| cells | 128 | 512 | 2048 | 4096 |
|---|---|---|---|---|
| items held by hand, with places | 72 | 256 | 864 | 1296 |
| by hand, without places | 32 | 32 | 32 | - |

A third to a half of an item per cell (with a patch of 64 cells: 48, 168, 576). So the
arithmetic holds by simulation too, places drawn at random and overlapping included.

**Outcome (journal 2026-11-11, 18:05):** P1, P2, P3 missed on all five lineages; my
prediction was wrong and my fear was right. The search did not find a use for the sense of
place: its genomes hold 10/14/0/11/17 items at 2048 cells (the 2a' genomes they started
from: 18/19/18/16/19), the same with the wires shuffled or the sense off, and inside a life
(`python -m tissue.where`) a cell is as loud and learns as much far from the place as at
it. Recall one step ahead is where it was. The lives were loaded for a brain that keeps
each loop in its own patch, far above what the brain it started as can hold, so there was
nothing to climb: the lesson of step 2a' again, in a new coat.

### Step 2c-3: the gate by hand. Plan, bars and predictions, written before any run

The search did not find "a cell away from its place is silent". Before asking again, the
question it could not answer: IF cells are silent away from their place, does holding grow
with the brain, with the rule found so far? `Plan.gate`: an interneuron whose nearness to
the place is under the gate says nothing and its wires do not change (by hand; 0: off;
with nothing felt there is no gate). At 0.3 with a field of 12 cells the patch that speaks
is about 37 cells.

**(a) No search.** The five 2a' genomes with `--field 12 --reach 32 --spoken --gate 0.3`,
32 wires, 128 to 4096 cells. They were found for a brain in which every cell speaks, so
this asks what gating alone buys. Prediction: holding at 2048 cells above the placeless 18
but far under the hand ceiling of 864: between 30 and 150.

**Outcome of (a) (journal 2026-11-11, evening):** not what I predicted, in both directions.
Holding is 0 at every size: with the gate on, the rule found for a brain where every cell
speaks lets a walk fade (0.94, 0.87, 0.83, 0.79 over four hops, even at a light load). But
recall one step ahead grows in proportion to the cells, with no search at all:

| cells | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| K90 hop 1, gate by hand | 20-32 | 16-53 | 95-125 | 171-255 | 357-507 | 707-1009 |
| K90 hop 1, no places (2a') | 35-38 | 52-60 | 75-91 | 91-126 | 99-169 | - |

0.17-0.25 items per cell at 2048 and at 4096 cells, on all five lineages, at sizes no search
ever saw and with a gate no search ever saw. This is the first time on rung 2 that
something grows with the cells at a fixed 32 wires per cell otherwise than as a vote.

Two controls (lineages 0 and 4, K90 at hop 1 at 2048 cells) say what does it:

| | lineage 0 | lineage 4 |
|---|---|---|
| no places, the dendrite as it was (2a') | 118 | 169 |
| the new dendrite alone (`--spoken`) | 154 | 195 |
| + a sense of place, the gate, wires from neighbours | 442 | 507 |
| the same with wires from anywhere (`--reach 0`) | 0 | 0 |

The gate without wiring by place is worse than no gate at all (a cell of the patch then
hears next to none of its own patch: hop 1 at 0.73 at 512 cells and 0.61 at 2048). It is
the two together that pay.

**(b) A search with the gate on**, from the same genomes: `--fan 16 --reach 32 --field 12
--spoken --gate 0.3 --sizes 128 256 512`, loads of 1/16 to 1/2 item per wire per patch
(what a patch of that many cells can hold with the rule as it is, so that the lives start
where the brain already works). Bars, on 32 wires, exam sizes 128 to 4096:

- **Q1, growth returns.** Held at 2048 cells at least 2x held at 512, every lineage.
- **Q2, above the ceiling of a placeless brain.** At least 100 items held at 2048 cells.
- **Q3, the shuffle.** With wires from anywhere (`--reach 0`) less than half of that.
- **Q4, the gate off** (nothing felt, `--nowhere`): at most 32 items.

Prediction: Q1-Q4 met on at least four lineages; 100-400 items at 2048 cells. If so the
honest reading is: places make growth pay, the gate is by hand, and finding the gate is an
open task for the search.

**Outcome of (b) (journal 2026-11-11, 19:45):** by the letter Q1 and Q2 are missed on four
lineages of five, and my prediction with them. What the search did buy is recall, and it
grows in proportion to the brain to 4096 cells (1024 and up never searched):

| cells | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| K90 hop 1, searched with the gate | 40-53 | 97-113 | 157-183 | 347-413 | 669-850 | 1500-1835 |
| K90 hop 1, gate, no search | 20-32 | 16-53 | 95-125 | 171-255 | 357-507 | 707-1009 |
| K90 hop 1, no places (2a') | 35-38 | 52-60 | 75-91 | 91-126 | 99-169 | - |

0.33-0.45 items per cell at 2048 and at 4096 cells, on all five lineages. Q3: with the
wires shuffled recall is 0 at every size. Q4: with the sense off the same genomes are
placeless brains again (K90 hop 1 65-127 and 13-18 items held at 2048 cells).

Holding: a thought no longer drifts to chance, it walks on a plateau, but the plateau lies
at 0.82-0.95 of the bits, around the bar and mostly under it, even with a single loop in
the whole brain. "Held" as defined (the last lap at 0.9 from the lightest load up) is
therefore 0 on four lineages (lineage 1: 9, 11, 58 at 512, 1024, 2048 cells). The largest
load whose last lap is still at 0.8 does grow with the brain: 48-128 items at 512 cells,
184-512 at 2048, 256-728 at 4096, where a placeless brain stops at 16 at any size. But
that is a looser bar than the one I set, read off after the fact, and the closing is
measured on four lives at those loads. It is a lead, not a result.

**Looking for where the plateau comes from (exploration, after the fact).** Two exams of
the same five genomes at 512 and 2048 cells, nothing searched anew:

| at 2048 cells | held | K90 hop 1 | last lap at 8-48 items |
|---|---|---|---|
| as searched: field 12, two laps of study | 0/58/0/0/0 | 669-850 | 0.85-0.94 |
| three laps of study (`--laps 3`) | 178/221/0/55/175 | 956-1216 | 0.93-1.00 |
| a wider field (`--field 24`: a patch of about 75 cells, not 37) | 249/250/230/270/239 | 739-880 | 0.96-1.00 |

At 512 cells the wider field gives 48/43/48/26/48. So the plateau was the patch: 37 cells
are too few to say a 12-bit item cleanly after two lessons. With a patch twice as wide the
same genomes hold in thought thirteen times what a placeless brain holds at 2048 cells and
five times more at 2048 cells than at 512.

That was a dial turned after seeing a result, on two sizes. It has to earn its keep on
things not yet looked at.

### Step 2c-4: the wider field, confirmed or not. Bars written before the runs

The five genomes of step 2c-3b, unchanged, examined with `--field 24` (32 wires, reach 32,
gate 0.3). Seen so far: 512 and 2048 cells in the usual worlds, nothing else.

- **R1, growth goes on.** Held at 4096 cells at least 1.5x held at 2048, every lineage.
- **R2, in worlds never used.** A second set of worlds (`--worlds 1`): at least 150 items
  held at 2048 cells and held at 2048 at least 2x held at 512, every lineage.
- **R3, the shuffle.** Wires from anywhere (`--reach 0`): less than half at 2048 cells.
- **R4, the sense off** (`--nowhere`): at most 32 items at 2048 cells.

Prediction: all four met on all five lineages; 400-550 items held at 4096 cells.

## Where the ladder stands

| rung | state |
|---|---|
| 0 pairs | done, 3 seeds: about 1 pair per interneuron to 512 |
| 1a one hop through own wires | done, 5 seeds: 0.7 links per interneuron at 64, fading to 0.35 at 512 |
| 1b search on chains | done, 5 seeds: links walked 4 hops doubled (17-19 at 64 cells), depth carries to 8 hops; size still does not buy depth |
| 1c the stream: no resets, idleness | done: 8 genomes live an unbroken life (3 of 5 random starts, 5 of 5 lineages); on 5 of them capacity keeps pace with size to 512 (0.7 items per interneuron); idleness costs nothing; a walk holds 8-16 ticks, then drifts |
| 1d a thought that holds | done, 5 lineages, no new mechanism: a stable cycle through learned items for 64 ticks on all five and 256 on the best; items held in thought x6-7 from 64 to 512 cells; capacity kept |
| own pace; a world with forks (a choice, then a goal) | open, the user decides the order |
| 2a a few wires per cell, drawn at random | done, 5 lineages, no search: recall one step ahead still grows with the cells (a vote, x1.3 per doubling); a thought that holds follows the wires per cell alone, 0.55 items per wire, flat from 512 to 2048 cells |
| 2a' a search that lives with few wires | done twice, the second asked properly: recall +11-51%, holding unchanged (K1, K2 missed) |
| 2b a second kind of cell | not built. A hand-made ceiling decided it: one item per wire whatever the share of cells that speak and whatever the number of cells. Roads left: wires that grow with the brain, hubs, places |
| 2c cells in a space | running: wires from neighbours on a ring (alone: predicted to change nothing), then a sense of place so that each loop can live in its own patch |
| 3 growth from a seed | not started |

Open questions carried along: what separates the lineages that scale without a fade from
those that do not (lineage 3 is the weakest on every exam); why a little idleness seemed
to help hop 4; whether a thought survives a world with forks, where an item has more than
one successor.
