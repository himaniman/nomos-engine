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
- Paid for on rung 2: build the ceiling by hand before the mechanism. Put the loads where
  the brain in hand can begin to earn, not only where the wanted brain could live. When a
  search does not find a thing, put it in by hand once to learn what it would be worth.
  A load measured on four lives is not measured. Check the clock before dating a page.

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

**Outcome of (a) (journal 2026-11-11, 19:45):** not what I predicted, in both directions.
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

**Outcome (journal 2026-11-11, 20:05):** R3 and R4 met; R1 missed on all five lineages and
R2 half missed. My prediction was wrong, and the 230-270 of the exploration did not
replicate at that height.

| cells | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| held, the usual worlds | 12-15 | 24-27 | 26-48 | 56-76 | 230-270 | 230-347 |
| held, other worlds | - | - | 28-58 | - | 85-148 | 142-200 |
| held, no places (2a') | 10-11 | 10-17 | 14-18 | 16-19 | 16-19 | - |
| K90 hop 1, the usual worlds | 46-51 | 95-113 | 176-223 | 385-434 | 739-880 | 1594-1815 |

- R1 (held at 4096 at least 1.5x held at 2048): x1.40 / 0.96 / 1.40 / 0.85 / 1.24.
- R2 (other worlds): held at 2048 is 129/148/138/85/113 against a bar of 150, missed;
  held at 2048 over held at 512 is x2.0-3.4, met.
- R3: wires shuffled, nothing is held and nothing recalled. R4: the sense off, 13-18.

What stands: with places a thought holds more in a bigger brain, about tenfold from 128 to
4096 cells where a placeless brain is flat from 512 on; and recall one step ahead is in
proportion to the cells in both sets of worlds. What does not stand: the size of it. The
figure at 2048 cells in the usual worlds was a lucky one, and my measure is to blame: a
heavy load is measured on four lives, a long closing is one loop per life, and a loop
either holds or derails, so "the last lap at 0.9" there means "all four loops held".

### Step 2c-5: the same exam measured on sixteen lives. Bars written before the runs

Nothing about the brains changes: the five genomes of 2c-3b, `--field 24`, a third set of
worlds (`--worlds 2`), and every load measured on at least sixteen lives (`--lives 16`),
256 to 4096 cells.

- **S1, growth at every doubling.** Held grows by at least x1.3 at each doubling of the
  cells from 256 to 4096, on at least four lineages of five.
- **S2.** At least 150 items held at 4096 cells on every lineage.

Prediction: S1 met on three or four lineages (one doubling in four will fall short
somewhere), S2 met; 150-300 items at 4096 cells.

**Outcome (journal 2026-11-11, 20:35):** S2 met, S1 missed (met on one lineage of five).

| cells | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| held in thought, sixteen lives | 24/28/27/25/30 | 47/48/32/30/44 | 88/104/92/89/95 | 108/114/105/106/203 | 217/160/220/268/287 |
| K90 hop 1 | 93-115 | 193-213 | 386-432 | 785-898 | 1530-1779 |
| held, no places (2a') | 10-17 | 14-18 | 16-19 | 16-19 | - |

From 256 to 4096 cells, sixteen times the cells, a thought holds x9.0 / 5.7 / 8.1 / 10.7 /
9.6 as much: x1.5-1.8 per doubling on average, but not at every doubling (1024 to 2048 is
x1.1-1.2 on four lineages). Per cell it thins out, 0.09-0.12 items at 256 cells and
0.04-0.07 at 4096. Three lineages also walk four hops cleanly and that grows in proportion
(K90 at hop 4: 31, 63, 124, 231, 477 on lineage 1).

**Where step 2c ends.** With a place for every loop, wires from neighbours and cells that
are silent away from their place, 32 wires per cell are enough for a bigger brain to be
smarter: recall one step ahead in proportion to the cells (0.4 items per cell to 4096),
a thought held about nine times more at sixteen times the cells. A placeless brain with
the same wires is flat at 18 items; the hand ceiling with places is 864 at 2048 cells, so
what was found is an eighth to a third of what the wires allow. Take away the wiring by
place and nothing is recalled; take away the sense and the same genome is a placeless
brain. By hand: the ring, wires from neighbours, the sense of place and the width of its
bell (set after seeing a first result, then tested in two sets of new worlds), the gate,
the scaling of a dendrite by what was said. Found: the cells' rule, in brains of at most
512 cells. Open: a search that finds the gate itself (2c-2 did not); the dials (reach,
field, gate) as numbers of the genome; a search that lives with the wider field.

### Step 2c-6: a search that lives with the wider field. Bars written before the run

The exam's field (24) is not the one the rule was searched with (12). One change: the
search of step 2c-3b again, from its own genomes, with `--field 24`. Exam as in 2c-5:
sixteen lives, the third set of worlds, 256 to 4096 cells.

- **T1.** Held at 4096 cells at least 1.5x the figure of 2c-5 (217/160/220/268/287), on at
  least four lineages.
- **T2.** Held at 4096 cells at least 8x held at 256, on at least four lineages (2c-5:
  x9.0 / 5.7 / 8.1 / 10.7 / 9.6).

Prediction: an even chance for T1. Every search today that I expected to gain in holding
did not (K1 twice, P, Q); what speaks for it is that three laps of study and four clean
hops both showed headroom. T2 met if T1 is.

**Outcome (journal 2026-11-11, 22:13):** T1 and T2 missed. Living with the wider field made
holding worse on four lineages of five, 176/165/113/185/204 items at 4096 cells against
217/160/220/268/287 (x0.81 / 1.03 / 0.51 / 0.69 / 0.71), and recall a little worse. The
genomes of step 2c-3b stay the best with a gate, and 2d and 3a start from them. A fourth
search today that did not buy holding.

## After rung 2: the user's vector (2026-11-11, evening)

Step 2c left three crutches: the gate is by hand, the world tells the brain where it is, and
"grown" has only ever meant "born bigger". The user wants each of them tried, and to be
gone on with in his absence. The order is mine:

| step | what | the question |
|---|---|---|
| 2d | the search finds the gate | can the cells learn to keep silent away from their place, once the hand gate is taken away step by step? |
| 3a | growth within one life, the size turned by the examiner | does a brain that is given new cells in mid-life keep what it knew and use the new room? |
| 2e | the brain chooses the place | in rungs: the place is not told with the eyes closed (hold it); told only during study (recall where a thing lives); not told at all |
| 3b | growth decided by the tissue | does a brain that grows where it is crowded do better than one grown evenly? |

2e may need a second kind of cell, an index that knows what lies where. That is the user's
first idea, cells of several kinds, with a job to do at last.

### Step 2d: the search finds the gate. Plan, bars and prediction, written before any run

Nothing new in the brain. A cell has felt its nearness to the place all along (one of its
inputs); in step 2c-2 that was not enough, because there was no slope to climb. Now there
is a brain that works with the gate, and the gate can be a scaffold: `Plan.lean` is how
much of the hand gate is left (1: all, 0: none), and a search with `--anneal 3000` takes it
away evenly over its first 3000 steps and lives the last 1000 with none. As the scaffold
goes, speech away from the place costs at once, and the only one who can stop it is the
cell.

From the best genomes with a gate (step 2c-6 if they are no worse than 2c-3b's on the exam
of 2c-5, else 2c-3b's), the same lives. Exam with NO hand gate: field 24, 32 wires, reach
32, sixteen lives, the third set of worlds, 256 to 4096 cells.

- **U1, recall without the scaffold.** K90 at hop 1 at 4096 cells is at least half of what
  the same lineage recalls with the hand gate, on at least four lineages.
- **U2, holding without the scaffold.** At least 100 items held at 4096 cells on at least
  four lineages (placeless: 18).
- **U3, the gate is the cell's own.** Inside a life at 2048 cells (`python -m tissue.where`)
  cells further than 64 from the place are at most a tenth as loud as cells within 6 of it,
  eyes open and closed, and learn at most a tenth as much.
- **U4, the controls.** Wires shuffled: less than half is recalled at 2048 cells. The sense
  off: at most 32 items held.

Prediction: U3 and U4 met. U1 and U2 met on three lineages or more, with a plain loss
against the hand gate at 4096 cells: a whisper of the far cells that does no harm among 512
cells is eight times as many voices among 4096, and no search has seen that size.

**Outcome (journal 2026-11-12, 00:01):** every bar missed on every lineage; my prediction
was wrong. As the scaffold went, the genomes did not take the gate over, they went back to
being placeless brains:

| cells | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| K90 hop 1, no hand gate | 53-64 | 66-91 | 70-123 | 73-153 | 79-191 |
| K90 hop 1, the gate by hand (2c-5) | 93-115 | 193-213 | 386-432 | 785-898 | 1530-1779 |
| held, no hand gate | 9-12 | 9-13 | 9-17 | 9-17 | 9-17 |

U1: 0.04-0.11 of the hand gate's recall at 4096 cells (bar 0.5). U2: 9-17 items held (bar
100). U3: a cell is as loud at any distance from the place. U4: shuffling the wires or
taking the sense away changes nothing, since nothing uses them. Lineage 3 died half way
(its error stopped being a number as the gate weakened; the search now skips such a step,
159 of its 4000).

Why, looked into before asking again:

1. **The lives did not make the gate pay.** The error of the search did not rise as the
   gate went (0.35 with it, 0.30-0.41 without). At the sizes searched, 128 to 512 cells,
   a placeless brain recalls nearly as much as one with places (35-38 against 40-53 items
   at 128 cells, 77-91 against 157-183 at 512), and most lives were loaded under that. The
   loads lesson a third time: a life has to pay for the wanted thing more than for what
   the brain can do without it.
2. **A cell can say its own gate, roughly.** Its perceptron (8 hidden) fitted to its own
   speech under the hand gate (`python -m tissue.imitate`) leaves a cell that should be
   asleep at a tenth of an awake cell's loudness and gets 8-16% of an awake cell's speech
   wrong (23-33% of its learning signal). So the gate is within the cell's reach, but
   imitation alone is not a brain: without the hand gate and without a search the five
   taught genomes recall 0-39 items at 512 cells.

### Step 2d', asked again with what 2d lacked. Bars U1-U4 unchanged

Two changes, and they are two, said plainly. (1) The cell is first taught its own gate by
imitation, so the search starts with the gate already in the genome and has to keep it and
mend the rest. (2) The lives make places pay: 256, 512 and 1024 cells, 1/8 to 3/4 item per
wire per patch, so that about half the lives lie beyond what a placeless brain recalls and
within what a brain with places does. The hand gate is taken away over the first 3000
steps of 4000 as before. If it works, one more search, with the loads and without the
imitation, says which of the two was needed.

Prediction: U3 met, the gate is the cell's own from the first step and now it pays to keep
it. U4 met. U1 met on three lineages or more. U2, holding, on fewer than U1: no search
today has bought holding, and a tenth of a voice left in every far cell is four hundred
voices among 4096.

**Outcome (journal 2026-11-12, 02:28):** U1, U2 and U3 missed; my prediction was wrong
again. But it is not the plain failure of the first attempt.

| cells | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| K90 hop 1, no hand gate | 62/48/52/48/43 | 110/64/74/68/71 | 160/103/85/85/95 | 163/9/88/91/115 | 0/0/88/87/127 |
| K90 hop 1, the gate by hand (2c-5) | 93-115 | 193-213 | 386-432 | 785-898 | 1530-1779 |
| held, no hand gate | 17/0/0/0/0 | 26/0/0/0/0 | 31/0/0/0/0 | 52/0/0/0/0 | 47/0/0/0/0 |

| at 2048 cells, K90 hop 1 | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| as found | 163 | 9 | 88 | 91 | 115 |
| wires shuffled | 0 | 0 | 79 | 53 | 129 |
| the sense off | 0 | 0 | 80 | 0 | 109 |

- Lineages 0 and 1 kept a gate of their own: without the sense, or with the wires
  shuffled, they recall nothing. Lineage 0 holds 17, 26, 31, 52 items at 256 to 2048
  cells, more than a placeless brain (18) and growing, the first holding that grew with a
  brain whose gate is its own; it breaks at 4096. Lineage 1 breaks from 2048 on.
- Lineages 2 and 4 are placeless brains again (neither control changes them).
- Lineage 3 stopped at step 2204 of 4000 (401 steps without a finite slope, the cap), with
  a quarter of the hand gate still on; lineages 2 and 4 skipped 388 and 206 steps.
- U3: the gate the cells were taught has leaked. A cell more than 64 from the place is 0.26
  to 0.40 as loud as one at it (it was 0.10 after the imitation), and learns 0.06-0.22 as
  much. Among 4096 cells that is too many voices.

So the gate can be the cell's own, in part and on some lineages, and what it does then is
of the right kind; but a perceptron of 8 hidden units says it too loosely, a search living
without the hand gate loosens it further, and the search itself is unstable there. The gate
stays by hand. What would mend it is most likely not asking a third time but a cell that
can say it sharply: a wider perceptron, or a kind of cell whose only work is to put its
neighbours to sleep. That is a change to what a cell is, and the user's to decide.

With the hand gate put back (`--lean 1`) these genomes recall 448-745 items at 2048 cells
and 900-1435 at 4096 (2c-5: 785-898 and 1530-1779): the search cost the rule itself
something as well.

### Step 3a: growth within one life. Plan, bars and prediction, written before the run

Until now "grown bigger" has meant "born bigger". Exam 3 (`tissue/grow.py`): one unbroken
life in two halves. The first loops are learnt, the brain is given new cells
(`Grown.more`: born after the last cell, innate wires of their own, nothing learnt; every
old cell keeps its number, its wires and what they hold), the second loops are learnt, and
then the eyes close on loops of either half. A place is counted in cells, so it does not
move when the ring grows. Three arms live the same world tick for tick: GROWN (512 cells,
then 2048), BORN BIG (2048 from the first tick), SMALL (512 to the end). 96 items before
the growth, 480 after, on sixteen lives; closings of 8 ticks. No search: the best genomes
with a gate, as they are (field 24, 32 wires, reach 32). By hand: when the brain grows and
by how much.

- **G1, nothing is lost by growing.** On the first loops the grown brain is within 0.03 of
  the brain born big, at hop 1 and at hop 8, on every lineage.
- **G2, the new room is used.** The same on the second loops.
- **G3, the growth was worth it.** On the second loops the grown brain is at least 0.10
  above the small one at hop 1, on every lineage.

Prediction: all three met on all five lineages. With places an old cell is not touched by
a newcomer (a unit test shows the grown brain says, on its next tick, what the small one
would have said), so the grown brain and the one born big differ only at the seam where the
ring opened. That makes it close to a consequence of step 2c rather than a discovery; what
it tests is that nothing in the kernel quietly depends on the size a brain was born with.

(Said plainly: to test the code I had already run the exam once, 128 to 512 cells on
lineage 0, and seen the grown brain level with the one born big, 0.84 and 0.87 at hop 1 on
the first loops, 0.93 and 0.93 on the second, the small one at 0.60 and 0.72. The bars are
for sizes and lineages not yet looked at.)

**Outcome (journal 2026-11-11, 22:15):** G1, G2 and G3 met on all five lineages, as
predicted. Accuracy at hop 1 (and at hop 8), five lineages:

| | the first loops (96 items) | the second loops (480 items) |
|---|---|---|
| grown, 512 then 2048 cells | 0.82-0.84 (0.59-0.64) | 0.93-0.94 (0.75-0.77) |
| born big, 2048 cells | 0.83-0.85 (0.60-0.64) | 0.93-0.94 (0.76-0.77) |
| small, 512 cells | 0.65-0.67 (0.52-0.54) | 0.73-0.78 (0.53-0.54) |

The grown brain is within 0.02 of the one born big everywhere; on the second loops it is
0.16-0.20 above the small one at hop 1. Growing in mid-life costs nothing and buys what
being born big buys. (The first loops are recalled worse than the second by both big
brains: the exam puts them all on the first 512 cells and a quarter of the second loops
land there too, so that stretch of the ring is the crowded one. Not an effect of growth.)

### Step 2e: the brain chooses the place. The rungs

The world has told the brain where it is on every tick. It is taken away in three rungs
(`--told` in the exam and the search says when the place is told):

| rung | the place is told | what the brain has to do |
|---|---|---|
| 2e-1 | only while the eyes are open (`--told seen`) | keep the place through a closing |
| 2e-2 | only while a loop is first walked round (`--told study`) | in the test, find where a thing lives from what it sees |
| 2e-3 | never | choose where a new thing shall live |

### Step 2e-1, asked first: no search, no mechanism. Prediction written before the run

The five genomes of step 2c-3b (the gate by hand, field 24), examined with `--told seen`
at 512 cells on sixteen lives in the third set of worlds. With the eyes closed nothing is
felt, and by the gate's own rule there is then no gate: every cell is free to speak.

- **V1.** K90 at hop 1 and the items held in thought are each at least half of what the
  same genome has when it is told the place throughout (2c-5 at 512 cells: K90 193-213,
  held 30-48), on at least four lineages.

Prediction: missed. A cell far from the place hears nothing with the eyes closed, and
its rule was never asked what to say then; I expect the far cells to hum and the motor
cells to drown. If so, the next thing is to ask (a search that lives with `--told seen`),
and only if that fails a mechanism: a cell that remembers its last nearness, the private
state held back since rung 1.

**Outcome (journal 2026-11-11, 22:16):** V1 missed, as predicted: K90 at hop 1 is 0 on four
lineages (65 on the fifth) and nothing is held. But not the way I pictured it. The far
cells do not drown the thought at once: through the first lap of the long closing, at 64
items, the brain is right on 0.96-0.99 of the bits (lineages 0-3), then 0.90-0.95, 0.78-0.85,
and it is down to 0.55-0.66 by the eighth lap. The thought stays where it was to begin
with and is lost by degrees, as cells that nothing silences join in and, with no gate on
their learning either, write into their wires. Every later closing then starts from a
brain already muddied, which is why hop 1 fails even at a light load. So a brain can keep
its place for a while with nothing but its own wires; what it lacks is something that
keeps the others out. That is a thing a search can be asked for.

Inside such a life (`python -m tissue.where ... --told seen`, lineage 0, 512 cells): the
moment nothing is felt, cells more than 64 away from the place are as loud as the patch
(0.82 against 0.92 in the first lap of the closing, 0.87 against 0.86 in the last) and
learn at two thirds of the patch's rate in the dark. A cell that hears nothing, with the
eyes closed, says something: its rule was never asked to be silent there, the hand gate
did that for it.

### Step 2e-1, asked of a search. Bars written before the run

No mechanism. The search of step 2c-3b again, from its own genomes, living with
`--told seen`: the place is felt, and the hand gate works, only while the eyes are open.
In the dark every cell is on its own. Exam with `--told seen`, field 24, sixteen lives, the
third set of worlds, 256 to 4096 cells.

- **V2.** K90 at hop 1 and the items held are each at least half of what the genomes of
  2c-3b have when told the place throughout (2c-5), at 4096 cells, on at least four
  lineages: K90 at least 765-890, held at least 80-143.
- **V3.** In the dark, cells more than 64 away from the place are at most a tenth as loud
  as cells within 6 of it, in the first lap of a closing and in the last (2048 cells).

Prediction: both met on at least four lineages. It is a small thing to ask, silence when
nothing is heard and nothing is seen, and the thought already keeps its place for a lap
without it. My doubt is the usual one: a hum that is harmless among 512 cells may not be
among 4096.

### Step 2e-2, asked first: no search, no mechanism. Prediction written before the run

The five genomes of step 2c-3b (the gate by hand, field 24) examined with `--told study`
at 512 cells: the place is told while a loop is first walked round and never again. In the
test the world jumps to a loop, the eyes see three items of it, and nobody says where it
lives; with nothing felt there is no gate.

Prediction: nothing is recalled (K90 at hop 1 is 0 on every lineage), and worse than in
2e-1, because here the cells are ungated with the eyes OPEN as well: every cell answers what
is seen, writes it into its wires, and the three items seen wake the whole ring and not the
patch that knows them. What I want to see is whether anything in the numbers says the
right patch answers differently from the rest, since that is what a search would have to
build on.

**Outcome (2026-11-12, 00:21):** as predicted, K90 at hop 1 is 0 on four lineages (8 on
the fifth) and nothing is held. At 128 items the first hop is right on 0.74-0.79 of the bits
and the walk fades from there: about what a placeless brain of that size does. Nothing in
these figures tells the patch that knows a loop from the rest of the ring; that will take
a look inside, and most likely a mechanism (a kind of cell that says where a thing lives:
see the plan when 2e-1 is done).

### Step 2e-2a: an index by hand. Plan and prediction, written before the run

Before any new kind of cell: what would it be worth if the brain had a memory of what lies
where? `hand.Index`: threshold cells with random innate wires from the senses (the code of
exam 0), and from them a few cells that learn by the delta rule the place an item was seen
at, written in a Gray code as fine as 16 cells of the ring. It is taught while a loop is
first walked round, when the world tells the place. In the test the eyes see an item and
the index says its place; with the eyes closed the brain goes on feeling the place it felt
last (a latch, by hand). The store is the five genomes of step 2c-3b with their hand gate,
unchanged (`--told index`). Everything about the index is by hand; nothing is searched.

- The yardstick by itself (`python -m tissue.hand --index`): the most items whose place the
  index says within 12 cells nine times in ten, by the number of its own cells.
- **W1.** With the place told by the index, K90 at hop 1 at 512 cells is at least half of
  what the same genome recalls when the world tells it the place throughout (2c-5:
  193-213), on at least four lineages, for an index of four times as many cells as the ring.

Prediction: by itself the index places 0.1-0.2 items per cell of its own (the delta rule
of rung 0 held 0.24 pairs per interneuron at nine bits in ten; here all five to eight bits
of a place must be right). W1 met with an index four times the ring, not with one of the
ring's size. An index cell has 12 innate wires and 5-8 plastic ones, so an index several
times the ring still costs fewer wires than the store.

**Outcome (journal 2026-11-12, 00:40):** W1 met on all five lineages, and with an index no
bigger than the ring, which I had predicted it would not be.

The index by itself, first: as I had written it, with a threshold code, it placed about a
hundred items whatever its size (72-112 with 512 to 32768 cells). Twelve-bit items look
alike to a threshold, and one pass of the delta rule cannot tell them apart. With a wave of
the dendrite for a code (cos of four times the drive: the non-monotonic code the search of
rung 0 found by itself) it places 0.45 items per cell of its own: 256, 864, 1944 items for
an index the size of a ring of 512, 2048, 4096 cells. I had predicted 0.1-0.2, for the code
that did not work.

The brain told its places by that index (`--told index`), against the same brain told by
the world on every tick:

| | K90 hop 1 | held in thought |
|---|---|---|
| 512 cells, told by the world (2c-5) | 193/213/207/207/208 | 47/48/32/30/44 |
| 512 cells, told by an index of 2048 cells | 167/188/179/170/181 | 48/48/49/29/44 |
| 512 cells, told by an index of 512 cells | 166/187/177/171/181 | 48/48/49/29/44 |
| 2048 cells, told by the world (lineages 0, 4) | 785, 867 | 108, 203 |
| 2048 cells, told by an index of 2048 cells | 642, 717 | 108, 196 |

Recall is 0.82-0.88 of what the world's telling gives, holding the same. So a memory of
what lies where is enough for the brain to find, from what it sees, the patch that knows
it, and a latch is enough to stay there in the dark. All of it by hand: the index, its
code, the Gray code of a place, the latch. What is asked of the world now is only to say
the place while a thing is first learnt.

### Step 2e-3a: the brain chooses the place, by hand. Bars written before the run

The last rung: the world never says a place (`--told self`). By hand, a tape: a thing the
index does not know is put a few cells further along the ring than the last new thing
(`--pace`, 8 cells) and the index is taught it there; a thing the index is sure of (every
bit of its answer beyond a half) is where the index says; with the eyes closed the brain
stays where it is. So the items of a loop met for the first time lie side by side, and
when the loop comes round again the index takes the brain back to them. The store is the
five genomes of step 2c-3b with their hand gate, unchanged; the index has four times as
many cells as the ring. Nothing is searched.

- **X1.** With no place ever told, K90 at hop 1 at 512 cells is at least half of what the
  same genome recalls when the world tells it the place on every tick (2c-5: 193-213), on
  at least four lineages.
- **X2.** The same for the items held in thought (2c-5: 30-48).

Prediction: X1 met, at 0.5-0.8 of the world-told figure. X2 missed on some lineages.
The tape lays loops end to end, so the stretch a loop is written on is shared with its
neighbours on both sides and the chance link from the last item of one loop to the first of
the next is now always between neighbouring cells, where with places drawn at random it
seldom was. Both cost holding more than recall.

**Outcome (journal 2026-11-12, 01:00):** X1 met on all five lineages, X2 missed on all five
(I had predicted "on some").

| 512 cells, an index of 2048 cells | K90 hop 1 | held in thought |
|---|---|---|
| told by the world on every tick (2c-5) | 193/213/207/207/208 | 47/48/32/30/44 |
| never told, a new thing 8 cells further on | 138/139/136/145/142 | 0/0/0/0/0 |
| never told, 4 cells further on | 148/152/150/157/153 | 10/9/0/9/12 |
| never told, 2 cells further on | 58/72/74/67/73 | 10/10/9/10/12 |
| 4 cells, and in the dark the place goes after the thing just said | 129/129/128/132/130 | 9/0/0/8/9 |

A brain that is never told a place recalls 0.65-0.78 of what a brain told by the world
recalls. It holds next to nothing in thought: on a tape every item of a loop has a place of
its own, so a place that stands still in the dark fits only part of the loop, and letting
the place go after the thought makes it worse, not better.

### Step 2e-3b: new things in a row share a place. Prediction written before the run

One change to the rule by hand (`--runs`): the tape moves on only when a new thing comes
right after a known one. New things in a row are put at the same place; something new
after something familiar opens the next place, `--pace` cells on. In this world a loop
walked round for the first time is a run of new things that ends on a known one (the item
it began with), so each loop gets one place, as when the world told it, with the places
laid side by side instead of drawn at random. That fit to the shape of this world is by
hand and is said here: the rule is "a new thing after a familiar one starts something
new", and whether it is a good rule in other worlds is not shown by this.

- **X3.** With no place ever told and `--runs`, at 512 cells K90 at hop 1 and the items
  held are each at least half of the world-told figures, on at least four lineages.

Prediction: met, at 0.8-0.9 of the world-told figures for recall and about the same for
holding, at a pace of 20 cells (about the spacing of the world's places at the load it
recalls): it rebuilds what the world's telling gave.

**Outcome (journal 2026-11-12, 01:00):** half met, and my prediction wrong: recall
126/149/139/126/141 (0.61-0.70 of the world-told figure), held 16/11/11/13/15 (0.23-0.43).
One place per loop is not enough when the places lie side by side: a loop then shares its
cells with its neighbours on both sides, and the chance link from the end of one loop to
the start of the next is learnt, since the two patches overlap. The world's places, drawn
at random, had kept consecutive loops far apart without my noticing it was a help.

**Exploration, after the fact:** the same rule with the next place opened far from the
last (`--pace 196` on a ring of 512, a share of 0.382 of the ring, which fills a ring
evenly however long one goes on). Recall 157/163/158/160/168, and HELD 95/78/91/88/96, two
to three times what the same genomes hold when the world tells them places drawn at random
(47/48/32/30/44); four hops walked cleanly at 92-99 items on four lineages. A dial turned
after seeing results, on one size and one set of worlds.

### Step 2e-3c: places chosen far apart, confirmed or not. Bars written before the runs

The five genomes of step 2c-3b with their hand gate, unchanged. No place is ever told.
`--told self --runs`, the next place 0.382 of the ring further on (196, 782, 1565 cells on
rings of 512, 2048, 4096), an index of four times the ring. A fourth set of worlds
(`--worlds 3`), sixteen lives. Seen so far: 512 cells in the third set of worlds only.

- **X4, recall.** K90 at hop 1 is at least 0.7 of what the same lineage recalls when the
  world tells it the place on every tick (2c-5), at 512, 2048 and 4096 cells, on at least
  four lineages.
- **X5, holding.** The items held in thought are at least the world-told figure (2c-5:
  30-48, 105-203, 160-287), at each of the three sizes, on at least four lineages.

Prediction: X4 met. X5 met at 512 and 2048 cells; at 4096 I am less sure, since the index
then has to place some thousands of items to within 12 cells and one wrong place moves a
whole loop.

**Outcome so far (journal 2026-11-12, 01:04; 4096 cells still to run):**

| never told a place, the fourth set of worlds | K90 hop 1 | held in thought |
|---|---|---|
| 512 cells | 151/159/157/160/160 | 63/86/74/75/84 |
| 512 cells, told by the world (2c-5) | 193/213/207/207/208 | 47/48/32/30/44 |
| 2048 cells | 541/549/546/543/553 | 308/298/293/269/296 |
| 2048 cells, told by the world (2c-5) | 785/853/845/898/867 | 108/114/105/106/203 |

- **X5, holding: met at 512 and at 2048 cells on all five lineages**, x1.3-2.5 and
  x1.5-2.9 the world-told figure. From 512 to 2048 cells it grows x3.5-4.9.
- **X4, recall: met at 512 cells (0.75-0.78), missed at 2048 (0.60-0.69, bar 0.7).** The
  five lineages stop at the same load there, which points at the hand-made part and not at
  the genomes. My guess, not checked: the index's test of being sure. At 545 items in an
  index of 8192 cells a known item has about one chance in five of having some bit of its
  place under a half, and is then taken for a new thing and written somewhere else.

So a brain that chooses its own places holds more in thought than one told places by the
world, and the reason is plain once seen: the world drew its places at random, so loops
fell on top of one another here and there, while a rule that opens each new place far from
the last spreads them evenly. The cost is some recall. All of it is by hand: the index,
the latch, the rule of runs, the stride.

**4096 cells, added 01:49:** K90 at hop 1 is 981/1002/986/986/1014 (0.55-0.64 of the
world-told 1530-1779: X4 missed there too), and 588/627/573/589/590 items are held (the
world-told brain: 217/160/220/268/287: X5 met, x2.1-3.9). So X5 is met at all three sizes
on all five lineages and X4 at 512 cells only. Never told a place, a thought holds 63-86
items at 512 cells, 269-308 at 2048 and 573-627 at 4096: 0.14-0.15 items per cell, in
proportion to the brain, which is the first time holding has done that on this rung.

### Step 3b: a brain that decides to grow, by hand. Bars written before the run

Exam 3 with `--alone`. Nothing is told: neither where a thing lies nor when to grow. Loops
come in 8 equal batches, each walked round twice; then the eyes close on every loop. By hand
(`grow.Places`): a run of new things is put at the stretch of the ring least in use; a
known thing is found by the index; after a batch, if every stretch of the ring is in use at
least twice (its neighbours counted), the brain gives itself 256 more cells, up to 2048. It
starts with 256. Two worlds, a poor one (24 loops, 192 items) and a rich one (96 loops, 768
items). Three arms: GROWS, BORN BIG (2048 cells from the first tick), SMALL (256 to the
end). The store is the five genomes of step 2c-3b with their hand gate; eight lives.
Nothing is searched; the rule of growth is by hand.

- **Y1, it grows as much as the world asks.** In the poor world the brain ends with at most
  1024 cells, and in the rich world with more than in the poor one, on every lineage.
- **Y2, and loses little by it.** In both worlds its accuracy at hop 1 is within 0.05 of
  the brain born big, on at least four lineages.
- **Y3, growth pays.** In the poor world it costs at most half the cell-ticks of the brain
  born big; in the rich world it is at least 0.10 above the small brain at hop 1.

Prediction: Y1 and Y3 met. Y2 met in the poor world; in the rich one I expect the grown
brain a little under the one born big, 0.03-0.08: what it learnt while small it learnt
crowded, and nothing moves a thing out once there is room.

**Outcome (journal 2026-11-12, 01:21):** Y1 and Y3 met on all five lineages; Y2 met in the
rich world and missed in the poor one, the opposite of what I predicted.

| five lineages | hop 1 | hop 8 | cells at the end | cost, million cell-ticks |
|---|---|---|---|---|
| poor world (192 items), grows by itself | 0.93-0.94 | 0.76-0.79 | 768 | 0.6 |
| poor world, born big | 0.99-1.00 | 0.98-1.00 | 2048 | 2.0 |
| poor world, small | 0.78-0.81 | 0.55-0.57 | 256 | 0.2 |
| rich world (768 items), grows by itself | 0.82-0.84 | 0.65-0.66 | 2048 | 6.3 |
| rich world, born big | 0.83-0.84 | 0.67-0.69 | 2048 | 7.9 |
| rich world, small | 0.58-0.61 | 0.50-0.53 | 256 | 1.0 |

The brain grows to three times its size in the poor world and eight times in the rich
one, by its own rule. In the rich world it is level with a brain born big (within 0.01) for
four fifths of the cost. In the poor world it stops at 768 cells and is 0.06-0.07 under the
brain born big, which has ten cells for every item and costs three times as much. The rule
stops growing the moment the ring is no longer crowded; it does not buy comfort. How much
room is "enough" is the one number in the rule (`--full`), and it is by hand.

## Hand-made parts into cells (from 2026-11-12, the user's word to go on)

Only the cells' rule is found; the gate, the index, the latch, the rule of places and the
rule of growth are by hand. They are to become cells of the genome one at a time. The user
restated the aim the same day: a brain that thinks, develops and in the end acts in a
simulated world. So this work does not stand in front of the road to a world (forks, an
action, a goal: the open rung of the ladder); the two go side by side.

### Step 2f: a cell that puts its neighbours to sleep. Plan, bars and predictions

Step 2d' ended on a fork that was the user's: a wider perceptron in the cell, or a second
kind of cell. The second is taken: the cell stays small and the same at every size, and
kinds of cells were the user's first idea.

`Plan.sleep = 8`: a SLEEP CELL beside every 8 interneurons along the ring, an eighth more
cells. It carries a perceptron of the same size as any cell (8 hidden). It feels how near
the place is to itself and whether the eyes are open, hears no cell, and says one number.
Its interneurons are awake as much as the sigmoid of that number: what they say and what
their wires learn is scaled by it. By hand: that there is such a kind of cell, where it
sits, what it feels, and the product form of sleep (as the wire rule's product is by hand).
Found: when it says "sleep". Why a cell of its own can do what the interneuron could not: an
interneuron of 8 hidden units had to say its speech AND multiply it by a gate, and a small
perceptron multiplies badly (a tenth of a voice left at best); here the product is the
mechanism and the cell has one thing to say.

**Built by hand first, and what that showed (before any search, lineage 0, 512 cells).**
`Tissue.sleep_by_hand(0.3)` sets the sleep cell's perceptron to the hand gate (one hidden
unit). With no hand gate in the brain:

| lineage 0, 512 cells | K90 hop 1 | held |
|---|---|---|
| the gate by hand (2c-5) | 193 | 47 |
| sleep cells by hand, speaking within the tick | 191 | 51 |
| sleep cells by hand, heard a tick late (`--late`) | 109 | 9 |

My first build had the sleep cell heard a tick late, as one cell hears another. That costs
nearly half the recall and four fifths of the holding: when the world jumps to another
loop, the patch it has just left is still awake for the first thing seen there and learns
"my last item, then that thing", once, in the very lap in which the new patch, not yet
awake, misses its own first link. One stretch of eight interneurons per sleep cell costs
nothing (191 against 193 with one each). So a sleep cell speaks WITHIN the tick, as a sense
organ does (S cells carry the cue as it is): it hears senses only, so no cell waits on
another. It is the same computation as a brain whose sleep cells lag a tick and whose
other senses are delayed a tick to match, shifted by that tick; the kernel takes the
simpler of the two. `Plan.late` keeps the other as a control.

**The exams**, as in 2c-5 and 2d: field 24, 32 wires, reach 32, sixteen lives, the third
set of worlds, 256 to 4096 cells, NO hand gate (`--gate 0`). The figures to stand against
are the hand gate's (2c-5): K90 at hop 1 of 93-115, 193-213, 386-432, 785-898, 1530-1779 at
256 to 4096 cells; held 30-48 at 512, 105-203 at 2048, 160-287 at 4096.

**(H) By hand, nothing searched.** The five genomes of 2c-3b with sleep cells set by hand.
- **W0.** At every size recall within a tenth of the hand gate's and at least 100 items
  held at 4096 cells, on all five lineages. And heard a tick late: recall under 0.7 of it.
Prediction: met. It is the hand gate said by a cell; this arm only says so at every size.

**(A) The search finds the gate, with a scaffold.** As 2d: the hand gate is taken away
evenly over the first 3000 steps of 4000 (`--gate 0.3 --anneal 3000`), the lives of 2d'
(256, 512, 1024 cells, 1/8 to 3/4 item per wire per patch). The sleep cell is born saying
"awake" to everyone, so as the scaffold goes somebody has to say "sleep", and now there is
a cell whose word is enough. From the genomes of 2c-3b; every number of the genome is free.

**(B) The search finds the gate, with none.** The same, but no hand gate from the first
step (`--gate 0`): a rule made for a gate lives without one until the sleep cells learn it.

Bars for (A) and for (B), each on its own:
- **W1, recall.** K90 at hop 1 at 4096 cells at least half the hand gate's, on at least
  four lineages.
- **W2, holding.** At least 100 items held at 4096 cells on at least four lineages.
- **W3, the gate is sharp and the cells' own.** At 2048 cells (`python -m tissue.where`)
  cells further than 64 from the place are at most a tenth as loud as cells within 6 of
  it, eyes open and closed, and learn at most a tenth as much.
- **W4, the controls.** With the sleep cells taken out (`--sleep 0`) and with the wires
  shuffled (`--reach 0`): less than half the recall at 2048 cells.

Predictions. (A): W1-W4 met on four lineages or more, recall at 4096 cells at 0.8 of the
hand gate's or better; the nearness at which the cell says "sleep" need not be my 0.3. (B):
the gate is found on three lineages or more; the rest stay brains with everyone awake or
stop on an endless slope, since they start where step 2d ended. I have been wrong about
every search on this ladder so far, each time on the hopeful side.

**Outcome (journal 2026-11-12, 14:35 and 18:50).** (H) W0 met in its first half on all five
(a sleep cell set by hand is the hand gate at every size); heard a tick late recall is
0.54-0.63 of it from 512 cells up and 0.72-0.77 at 256. (A) **W1 and W2 missed, met by two
lineages of five** (1 and 4): there the search has a gate of the cells' own, K90 at hop 1 of
1221 and 1154 at 4096 cells (0.69-0.80 of the hand gate's) and 330 and 289 items held (more
than the hand gate's 160-287), sharp (W3: far cells 0.06 as loud, learning a hundredth) and
its own (W4: 0.18-0.19 of the recall without the sleep cells, 0.08-0.10 with the wires
shuffled). Lineage 0 keeps a leaky gate, 2 and 3 are placeless again. (B) **missed on all
five**: with no scaffold every genome stays a placeless brain. Both predictions wrong on
the hopeful side. It is the first gate found by a search that is sharp and scales; it took
a kind of cell whose one job is to say "asleep", and the scaffold.

## Exam 4: forks, the road to a world (`tissue/fork.py`, from 2026-11-12)

What the brain can do so far is say what follows what when only one thing can follow. To
act in a world it needs three more things, in this order: FORKS (what comes next depends
on what the body does), a CHOICE (the brain says the way itself), a GOAL (it chooses the
way that leads to a wanted thing, by walking its learned links in the dark before it
acts: thinking as planning on a map that was learned).

The world of exam 4 is rooms, each a distinct item, with `ways` ways out of each; a way
out of a room always leads to the same room (each way is a shuffle of the rooms, the
first one a single round through them all). The eyes see the room's item and may close.
The body feels the way it has just gone and never stops feeling it. Study: the body is
led out of every room by every way, twice (a round that takes every link once, so no
jump). Test: a jump, three rooms seen, then the eyes close and the body walks on by ways
drawn at random; hop j is right only if the brain went on by itself AND the way its body
went. K = links = rooms x ways. One way is one loop of exam 2. Yardsticks: a perfect
memory is right at every hop; a memory that does not feel the body is right at 0.75,
0.63, 0.57, 0.55 over four hops with two ways.

### Step 4a: the body's ways, asked first with no search. Bars written before the run

Three ways for the body's sense to reach the brain, the genomes those of step 1d
(`runs/tissue/hold/`, wired from every cell, no places), nothing searched:

- `cells`: the way gone sits on sensory cells of its own beside the eyes'. No mechanism;
  what an interneuron makes of them is up to a rule that never met them.
- `dendrites`, BY HAND: an interneuron has one dendrite from the other interneurons per
  way, and on a tick only the dendrite of the way just gone is heard and learns. The cell
  and its rule are untouched; it reads "my dendrite" and never knows there are several.
  (Why not groups of cells per way: a room's code would then depend on the way it was
  entered by, and a link would have to be learnt once for every pair of ways, in and out.)
- `numb`: the control.

Seen before writing the bars (lineage 0 only, 64 and 256 cells): with `cells` hop 1 is at
0.58-0.60, under the numb brain's 0.77-0.82 (the body's cells speak in the dark, where
this rule has only known silence). With `dendrites` K90 at hop 1 is 46 and 217 links with
one way, 98 and 422 with two, and at the top of the grid with four.

Bars, for the five lineages at 64 to 512 cells (128 and up never searched), sixteen lives:
- **F1, the dark walk goes where the body goes.** `dendrites`, two ways, 256 cells: K90 at
  hop 1 at least 300 links and at hop 4 at least 150, on at least four lineages; `numb`
  holds nothing at any load (hop 1 under 0.9).
- **F2, growth pays.** K90 at hop 1 at 512 cells at least 1.8 times that at 256, on at
  least four lineages.
- **F3, a way is paid for in wires, not cells.** At 256 cells the links held with two ways
  are at least 1.7 times those with one, and with four at least 3 times.

Prediction: all three met on all five lineages. It is one lineage and two sizes that I
have seen, so the doubt is lineage 3, the weakest on every exam, and 512 cells.

## Exam 5: a goal (`tissue/seek.py`)

The forked world, studied as in exam 4. Then, sixteen times in the same unbroken life,
the world jumps somewhere, the eyes see the room and a WANTED room is named, 1 to `depth`
steps away by the shortest road. The brain walks, eyes open, until it stands there or has
taken 2 x `depth` + 2 steps.

### Step 5a: thinking by hand on a map that was learned. Bars written before any run

All the brain knows of the world is in its wires, and the only way to ask them is to walk
them in the dark. BY HAND: before each step a copy of the brain is walked with its eyes
closed along every string of `depth` ways, its body's sense (the hand-made dendrites of
4a) fed as if it had gone them. A string along which the copy says the wanted room, all
bits but one at most, is a plan; the shortest wins and its first way is taken by the real
body; no plan, a way at random. What a copy learns in thought is thrown away. The map is
the brain's own: the rule found for loops and the wires of this life. The search through
it, the holding of the wanted room, the comparing and the choosing are mine. Nothing is
searched. This asks one thing: is the map a brain learns good enough to plan on?

The genomes of step 1d, two ways, thoughts 4 deep, sixteen lives of sixteen wanted rooms.
`straight`: the share of wanted rooms reached by a shortest road. A perfect map with the
same thinking goes straight every time.

- **G1, it plans on its own map.** 32 rooms (64 links), 256 cells: straight at least 0.9
  at every distance from 1 to 4, on at least four lineages.
- **G2, thinking is what does it.** The same brain not thinking (`blind`, ways at random)
  and the same thinking in a brain that does not feel its body (`numb`): straight under
  0.5 at distance 3 and 4.
- **G3, a thought is only as good as the map.** 128 rooms (256 links) at 256 cells, beyond
  what that brain walks four hops deep: straight at distance 4 under 0.7.
- **G4, and a bigger brain has the better map.** The same 128 rooms at 512 cells: straight
  at least 0.9 at every distance, on at least four lineages.

Prediction: G1, G2 and G4 met; G3 met, with distance 1 still above 0.9 in the crowded
brain (a thought one hop deep needs only the first hop). The one I am least sure of is
G1 at distance 4: a string of four ways is right only if four hops in a row are, and the
wanted room has to be said to within a bit.

### Step 5b: thinking with no copy. Bar written before the run

In 5a a COPY of the brain walks in thought and what it learns there is thrown away, which
no brain can do. `ponders`: the brain itself closes its eyes, walks a string of ways in
thought, opens its eyes on the room it stands in (which is how it comes back), and so for
every string; whatever its wires learn in the dark stays learnt. A step then costs 96
ticks of thought (sixteen strings, four hops and a tick to hear the last, a tick to look
again), and a life of sixteen wanted rooms is several thousand dark ticks on a map that
was studied for 129. Seen before writing this: G1 and G2 on five lineages, nothing of 5b.

- **G5, thinking does not wear the map out.** 32 rooms, 256 cells, the five lineages:
  straight at least 0.9 at every distance over the sixteen wanted rooms, and over the last
  eight of them no more than 0.05 under the first eight.

Prediction: not met. In the dark the rule goes on learning: where a thought is right it
only says again what the wires hold, but every string is also walked past the point where
it is right, and what is said there is learnt as well. I expect the first wanted rooms
near 5a's figures and a slide after that, to under 0.8 at distance 4 in the second half.
The stream said idleness costs nothing over 64 ticks; this is fifty times that.

**Outcomes of 5a and 5b (journal 2026-11-12, 13:10 and after).** G1 and G2 met on all five
lineages (thinks 1.00 / 0.94-0.97 / 0.92-0.98 / 0.92-0.96 at distances 1 to 4; blind 0.53 /
0.28 / 0.17 / 0.15; numb the same as blind). G3 met on all five (128 rooms at 256 cells:
0.21-0.48 at distance 4), though distance 1 is above 0.9 on three lineages only. **G4
missed**, my prediction wrong: at 512 cells the same world is walked straight 0.92-1.00 /
0.79-1.00 / 0.53-0.97 / 0.43-0.94, at least 0.9 at every distance on one lineage of five.
The bigger brain does have the better map (at distance 4 from 0.21-0.48 to 0.43-0.94) but
256 links are too many for 512 cells to be walked four hops deep to within a bit. **G5
missed** as predicted, and by more: 0.40-0.59 at distance 4, the last eight wanted rooms
worse than the first eight, and the first eight already well under the copy's figures.

### Step 5c: which half of thinking in one's own head does the harm. Prediction first

Two arms between `thinks` and `ponders`, 32 rooms, 256 cells, five lineages, no bar:
`ponders-still` (the brain walks in thought itself and opens its eyes again, but its
wires are still while it thinks) and `thinks-learns` (a copy walks, but what its wires
learn in thought is kept). Prediction: the learning does the harm. `ponders-still` within
0.05 of `thinks` at every distance (one look at the room is enough to come back), and
`thinks-learns` as bad as `ponders`. If so, what the road needs next is a rule that does
not learn from its own thoughts as it learns from its eyes, and the cell already hears
whether the eyes are open.

**Outcome (journal 2026-11-12, 13:20):** as predicted, on all five lineages.
`ponders-still` 1.00 / 0.95-0.97 / 0.93-0.98 / 0.94-0.99 with no slide through a life;
`thinks-learns` 0.76-0.92 / 0.69-0.77 / 0.47-0.62 / 0.45-0.58, as bad as `ponders`. The
learning does all of the harm; the copy was never needed.

### Step 5d: wires still in the dark, in the kernel. Bars written before the run

In 5c the stillness was the examiner's: he put the wires back after every thought.
`Plan.still`, BY HAND, puts it into the brain: on a tick with the eyes closed no wire
changes. It reaches further than exam 5, since every exam of the stream let the brain
learn in the dark, so it is asked of the old exams too. The genomes of 1d, nothing
searched, sixteen lives.

- **S1, a thought that is not learnt still holds.** Exam 2 at 64 and 256 cells with
  `--still`: K90 at hop 1 and the items held through a closing of 64 ticks at least 0.9 of
  the same genome's figures without it, on at least four lineages.
- **S2, and the brain thinks in its own head.** Exam 5, `ponders` with `--still`, 32 rooms
  at 256 cells: straight at least 0.9 at every distance on at least four lineages, the
  last eight wanted rooms no more than 0.05 under the first eight.

Prediction: S2 met (it is 5c again, bar the one echo a closing loses). S1: recall met;
holding I do not know. A walk in the dark that learns what it says may be mending the
loop as it goes or wearing it, and the stream never told the two apart. My guess is that
holding is kept or better, since the same learning wears a map in exam 5.

**Outcome (journal 2026-11-12, 13:25):** S2 met on four lineages (straight 1.00 /
0.95-0.97 / 0.95-0.98 / 0.94-0.99 on all five; lineage 3 slides 0.06 at distance 4 from
the first eight wanted rooms to the last, a hundredth over the bar). **S1 missed on
holding**, and my guess had the wrong sign: recall at hop 1 is untouched (0.97-1.02 of the
figure without `still`), but the items held through 64 dark ticks are 0.80-0.96 of it at
64 cells on four lineages (1.17 on the fifth) and 0.85-0.98 at 256, at least 0.9 at both
sizes on one lineage only; K90 at hop 4 is 7-12% lower as well. So on a loop, where there
is one road, what the brain learns in the dark is the road said again, and it mends a
little; in thought, where it walks every branch, the same learning wears the map out.
Stillness by hand is the blunt cure: it buys thinking in the brain's own head and costs
a tenth of the holding. The fine one is a rule that tells rehearsing from imagining.

## Exam 6: a world (`tissue/world.py`, the user's choice, 2026-11-12)

Offered a fork (a small world now, with the thinking still by hand, or first the hand-made
parts of thinking into cells) the user chose the world, "though it is important to remove
the hand-made parts too". So the world is built on what exams 4 and 5 have, by hand where
it must be, with the list of what is by hand kept in the open, and the parts are then
taken away one at a time.

What exams 4 and 5 were not: the body was LED through a study (out of every room by every
way, twice, in a round with no jump), then examined in trials, and the world was a shuffle
with no shape. A world has a shape, and nobody leads.

**The land.** A square grid of rooms. Between two rooms that touch stands a wall or none
(a wall in three places of ten, and the land hangs together). Every room has a look of its
own, a random item of 12 bits. The body goes north, east, south or west; into a wall, it
stays. So a land of side 6 is 36 rooms and 144 ways, of which about four in ten lead back
into the room they leave. One room in six has a TREE; one tree at a time bears a fruit; a
body that comes to it eats and another tree, drawn at random, bears the next. A life is 40
ticks to every room of the land, unbroken, from birth: no study, no trial, no jump.

**The brain** is the tissue of exam 5 with its wires still in the dark (`Plan.still`), eyes
that never close and a dendrite per way.

**By hand, all of it listed** (found by a search: the cell's rule, and nothing else):

1. the body's sense of the way gone, as a dendrite per way (exam 4);
2. no wire changes in thought (5d);
3. THE WILL: it wants the room where the fruit is and is TOLD what that room looks like
   (exam 5's naming of a wanted room); it thinks before every step; a road found, it takes
   the first way of it; none found, a way at random. There is no curiosity: it wanders
   because it has no plan, and it learns the land because it wanders;
4. THE THINKING: copies of what the cells say now walk in the dark, the nearest rooms
   first (one hop by every way, then one hop on from every room so reached). The list of
   rooms already thought of, the comparing of what a copy says with the wanted look (every
   bit), the choice among roads of one length (at random), a limit of 24 hops and of 256
   rooms a thought: all mine. In exam 5 a thought was every string of four ways, 16 of
   them; with four ways and roads of ten steps that is a million strings, so the nearest
   rooms go first and a room is thought of once;
5. the world's: no two rooms look alike, and the eyes see the room and nothing beyond it.

First tried and dropped, on lands of a pilot seed: a will that remembers the rooms where it
ate and wants the one it ate at longest ago. A brain that happened to find two trees side
by side and no third ate 1399 times in 1440 ticks, more than one born knowing the land:
with that rule knowing less pays, and what is measured is the luck of the finding. Hence
one fruit at a time and the look of its room told.

**Yardsticks**, the same will with another memory, in the same lands with the fruit going
round in the same order: `blind` never has a plan; `perfect` remembers every way it ever
went and thinks on that; `oracle` was born knowing every way. `perfect` is not a ceiling:
it stops wandering once it has a road, a brain that fails a thought wanders on and comes
to know more, so only `oracle` is above everything.

**Pilot** (seed 1, lineage 0, 256 cells, 8 lives, side 6; it set the rule above and the
limit of 256 rooms a thought, which costs nothing against 512 and halves the time): meals
a life 168 against `perfect` 150, `oracle` 304, `blind` 7; a single life swings by a factor
of five (a fruit on a tree in a dead end the body has never entered is found by luck
alone). So sixteen lives, and bars with room for that noise. The exam's lands (seed
90000) have not been looked at.

### Step 6a: a life in a land. Bars written before the run

Genomes of 1d (`runs/tissue/hold/`), nothing searched, 16 lives, five lineages.

- **V1, it makes a living.** Side 6 at 256 cells: meals a life at least 0.8 of `perfect`
  and at least ten times `blind`, on at least four lineages of five.
- **V2, and it is the map in its wires that feeds it.** `numb` (one dendrite whatever the
  way) at 256 cells, side 6: no more than twice `blind`, on every lineage.
- **V3, what it learned by wandering it thinks with.** In the last third of a life, side 6
  at 256 cells: trips whose shortest road is 3 or 4 steps go by a shortest road at least
  0.8 of the time, on at least four lineages.
- **V4, a bigger land wants a bigger brain.** Side 10 (100 rooms, 400 ways: about what 256
  cells hold at hop 4 with four ways in exam 4, and half of what 512 hold): in the last
  third of a life 512 cells eat at least 1.25 times what 256 eat, on at least three
  lineages; and at side 6 they eat no more than 1.1 times as much (there the small brain
  is enough).

Reported with no bar: side 8 at 256 cells; meals by third of a life (how fast it comes to
know its land); at the end of a life, of the ways it has gone the share it says right one
hop ahead, and how LOUD it speaks of ways gone and of ways never gone (if a brain is
quieter about what it does not know, that is where a curiosity of its own can come from).

Prediction. V1 met on four (lineage 3 the doubt, as everywhere). V2 met: a numb thought
finds the wanted room at some depth and cannot say which way, so it walks at random. V3
met on four. V4 is the open one: I expect 256 cells to fall to about half of `perfect` at
side 10 and 512 cells to be near it, so met; but exam 5's crowded world missed its bar on
just this, and a land has roads of ten steps and more where exam 5 had four, so the depth
of a thought may bind before the size of the map does, and then growing the brain would
buy little. Side 8 at 256 cells: 0.7 of `perfect` or better on four lineages.

**Outcome (journal 2026-11-12, 18:40).** Meals a life: `blind` 5-7 in every land; `perfect`
121 / 182 / 141 and `oracle` 259 / 380 / 476 at sides 6 / 8 / 10.
- **V1 missed**: at side 6 the five lineages at 256 cells eat 130 / 92 / 118 / 86 / 95, which
  is 1.07 / 0.76 / 0.97 / 0.71 / 0.78 of `perfect` (two over the bar, three just under) and 16
  to 24 times `blind` (all over).
- **V2 met** on all five: numb 5.9-6.1.
- **V3 met** on all five: trips of 3-4 steps in the last third straight 0.98-1.00.
- **V4**: the first half met on all five (side 10, last third, 512 cells eat 2.5 to 4.4 times
  what 256 eat); the second not cleanly (side 6: 0.92 to 1.22, over 1.1 on three, which a
  life that swings by a factor of five does not let me read).
A bigger land wants a bigger brain far more sharply than I predicted: 256 cells make 0.71-1.07
of `perfect` at side 6, 0.20-0.52 at side 8, 0.07-0.17 at side 10 (512 cells there:
0.12-0.52). What binds is the DEPTH of a thought and not the map one hop ahead: at side 8
the brain says the right room behind 91-97% of the ways it has gone and still goes straight
on 0.07-0.55 of trips of nine steps and more. Half the copies of a long thought say rooms
that are not there. Two guesses at a cure, tried as pilots and left behind switches that are
off: a copy shown what it has just said (`--imagery`) does next to nothing; a copy that
speaks quietly dropped (`--mumble`) halves the meals. Why a long thought fails is the open
question on this road. Also seen: the brain speaks 2 to 6 times more quietly of ways it has
never gone, from which a curiosity of its own could be read.

### Step 6b: a wider thought. A look first, then bars written before the run

**The look** (`python -m tissue.world GENOME --why --side 8`, four of the exam's lands,
the brain as it is at the end of its life): a copy is carried into every room and asked for
a road to every tree. Shares of the pairs whose shortest road is 9 steps or more:

| | lineage 0 | lineage 4 |
|---|---|---|
| a road exists through ways the body has gone | 1.00 | 0.95 |
| a road exists on the brain's own map, a hop at a time | 1.00 | 0.72 |
| the thought finds a road (its first way on a shortest road) | 0.70 (0.66) | 0.22 (0.15) |
| it finds none: at its limit of 256 rooms / nothing left to think of | 0.30 / 0 | 0.78 / 0 |
| allowed 4096 rooms | 1.00 (0.92) | 0.91 (0.55) |
| shown what it says (`--imagery`), 256 rooms | 0.79 (0.73) | 0.53 (0.40) |
| both | 1.00 (0.88) | 0.95 (0.65) |

So a long thought fails for one reason, and it is mine: the limit of 256 rooms a thought,
set on a pilot at side 6 where it cost nothing, and spent mostly on copies that say rooms
which are not there. The road is in the wires. (At 5-8 steps the same: 0.93 and 0.40 found,
1.00 and 0.88 with 4096 rooms.) METHOD: a dial set by hand where it does not bind was carried
to where it does, unexamined; I then guessed at cures before looking.

**The step**: no new mechanism, the dial `--budget` (rooms a thought may think of) at 1024
and 4096. It costs thinking time and nothing else. Reported beside it with no bar:
`--imagery` at 4096 rooms, which the look says helps a weak lineage and costs a strong one
a little.

Bars: the exam's lands, side 8, 256 cells, 16 lives, five lineages, 4096 rooms a thought,
against 6a (meals 95 / 55 / 88 / 53 / 36; trips of 9+ steps in the last third straight
0.55 / 0.22 / 0.39 / 0.17 / 0.07; `perfect` 182):

- **T1.** Meals a life at least 1.3 times 6a's, on at least four lineages.
- **T2.** Trips of 9+ steps in the last third straight at least 0.6 of the time, on at least
  three.
- **T3.** At least 0.7 of `perfect`, on at least two.

Prediction: T1 met on four (the weak lineages gain most), T2 on three, T3 on two (lineages 0
and 2). The doubt: the look asks a grown brain; in a life the thoughts that fail come mostly
while the map is still being made, and there a thought may be right to find nothing. Then
side 10 at 512 cells with the same dial, reported.

**Outcome (2026-11-13; records `runs/tissue/hold/hold-s*.world8.thinks*.budget*.json`).**

| rooms a thought | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 256 (6a): meals a life | 95 | 55 | 88 | 53 | 36 |
| 1024 | 139 | 87 | 103 | 68 | 62 |
| 4096 | 163 | 69 | 123 | 51 | 62 |
| 4096, shown what it says | 107 | 109 | 165 | 22 | 102 |
| 256: meals by third of the life | 7 / 22 / 66 | 7 / 9 / 40 | 7 / 21 / 60 | 4 / 13 / 36 | 5 / 12 / 19 |
| 4096 | 14 / 70 / 79 | 9 / 30 / 30 | 15 / 64 / 44 | 6 / 25 / 19 | 9 / 29 / 24 |
| 256: trips of 9+ in the last third, straight | 0.55 | 0.22 | 0.39 | 0.17 | 0.07 |
| 1024 | 0.54 | 0.29 | 0.31 | 0.39 | 0.19 |
| 4096 | 0.49 | 0.37 | 0.28 | 0.37 | 0.10 |
| 256: lives of 16 with no meal in the last third | 1 | 1 | 0 | 1 | 3 |
| 1024 | 1 | 5 | 1 | 1 | 2 |
| 4096 | 0 | 6 | 0 | 9 | 1 |
| 4096, shown what it says | 0 | 7 | 0 | 14 | 1 |

- **T1: missed.** At 4096 rooms three lineages eat 1.3 times 6a's or more (x1.71, x1.40,
  x1.71), lineage 1 x1.24 and lineage 3 x0.95. (At 1024: three as well.)
- **T2: missed**, on every lineage: the best is 0.49.
- **T3: missed**: lineage 0 only (163 of the 127 asked); lineage 2 is at 0.68 of `perfect`.

All three predicted met. What the look did not show, because it asked each thought once and
never let the body act on it: **a thought that always finds a road locks the body in.** With
256 rooms a thought often found nothing and the body took a way at random; that was also what
got it out of a bad place. With 4096 a road is found on 0.7-0.8 of the ticks (it was 0.2-0.3),
some of those roads are false (a copy says the wanted look where it is not), and a false road
that leads back whence the body came is thought again on the next tick. A look at such lives
(`stuck`, single lives of the exam with a watch): the body goes to and fro between two rooms
next to each other, a road in its head on every tick, from its last meal to the end of the
life. The wide thought eats two or three times as much in the middle third of a life and then
loses lives one by one: 16 lives of 80 end with an empty last third, where 6 did. Imagery
makes it worse where the map is weak (lineage 3: fourteen lives of sixteen) and better where
it is not (lineage 2: 165, lineage 4: 102).

This is the reactive limit cycle of the forage world again, one floor up: there the cure was a
commitment, here the fault is that the body takes only the first way of a plan and thinks
afresh, so two plans that disagree can pass it between them forever. The limit of rooms is
hand-made and so is this fault; 1024 rooms is the better dial as things stand (139 / 87 / 103
/ 68 / 62), not because a thought should be short. The next asking of depth is a body that
does not undo its own last step on the word of a thought (or keeps a plan while it holds), with
bars of its own. METHOD: a look that asks thoughts one at a time says nothing of what happens
when they are acted on in a row.

### Step 6c: curiosity in place of the dice. Pilot, then bars written before the run

One hand-made part of the will made less arbitrary. With no road in its head the body took a
way at random. A yardstick says what that costs: `seeker`, a perfect memory that with no road
takes a way it has never gone out of the room it is in, eats 172 / 210 / 231 times a life on
the exam's lands of side 6 / 8 / 10 where `perfect` eats 121 / 182 / 141. And the brain has
the signal already: exam 6 found it speaks 2-6 times more quietly of ways it never went.

**The mechanism** (`--curious P`, BY HAND, off unless asked for): with no road, the body takes
the way of which the brain spoke most quietly one hop ahead in the thought it has just had, if
that is quieter than a share P of the loudest of the four; otherwise the dice as before. What
is read is the brain's own voice; that it is read, the rule and the threshold are mine.

**The pilot** (pilot lands, seed 1, side 8, 256 cells, 16 lives, 256 rooms a thought), meals a
life and the share of the land's ways gone by an eighth of the life:

| | no curiosity | P 0.5 | P 0.7 |
|---|---|---|---|
| lineage 0 | 88, 0.46 | 69, 0.51 | 133, 0.64 |
| lineage 4 | 34, 0.46 | | 59, 0.57 |

P 0.5 below no curiosity is not understood (sixteen lives of chaotic walks; reported, not
explained). The threshold is set at 0.7 on these lands and not touched again.

Bars: the exam's lands, side 8, 256 cells, 16 lives, five lineages, 256 rooms a thought,
`--curious 0.7` against the same brain without (rerun beside it, since 6a's records do not
hold the share of ways gone early; 6a: 95 / 55 / 88 / 53 / 36) and against `seeker` (210):

- **C1.** Meals a life at least 1.3 times those without curiosity, on at least four lineages.
- **C2.** A larger share of the ways gone by an eighth of the life, on all five.
- **C3.** At least 0.6 of `seeker` (126 meals), on at least two.

Reported beside it with no bar: the same with 1024 rooms a thought.

Prediction: C1 met on four, C2 on five, C3 on two (lineages 0 and 2). The doubt: the pilot is
two lineages, and its middle threshold went the wrong way.

**Outcome (2026-11-13; records `runs/tissue/hold/hold-s*.world8.thinks*.curious0.7.json`; the
rerun without curiosity gives 6a's figures to the last digit).**

| side 8, 256 cells, 16 lives | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| no curiosity, 256 rooms a thought: meals a life | 95 | 55 | 88 | 53 | 36 |
| curious 0.7 | 124 | 79 | 99 | 71 | 47 |
| times as much | 1.31 | 1.43 | 1.13 | 1.33 | 1.32 |
| no curiosity: ways gone by an eighth of the life | 0.48 | 0.51 | 0.48 | 0.47 | 0.48 |
| curious 0.7 | 0.60 | 0.62 | 0.61 | 0.58 | 0.60 |
| no curiosity, 1024 rooms a thought (6b): meals | 139 | 87 | 103 | 68 | 62 |
| curious 0.7, 1024 rooms | 173 | 97 | 123 | 71 | 80 |
| the same: lives of 16 with no meal in the last third | 0 | 6 | 1 | 9 | 1 |

- **C1: met**, on four lineages (0, 1, 3, 4); lineage 2 gains x1.13.
- **C2: met**, on all five: 0.58-0.62 of the land's ways gone by an eighth of the life where
  the dice gave 0.47-0.51.
- **C3: missed**: the best is lineage 0 at 124 meals, 0.59 of `seeker`, two meals short.

Predicted: C1 and C2 met (right), C3 on two lineages (wrong). The brain's own quiet is a usable
sign of what it has not seen: read by a hand-made rule in place of the dice it is worth a third
more meals, and the gain is in the middle of a life (lineage 0 by thirds: 6 / 29 / 90 against
7 / 22 / 66), where the map is being made. With 1024 rooms a thought beside it the two add up:
173 / 97 / 123 / 71 / 80, the best figures at this side, lineage 0 at 0.95 of `perfect` and 0.83
of `seeker`. They do not mend each other's fault: lives locked between two rooms are as many as
without curiosity (17 of 80), since a body with a road in its head is never curious.

Three more looks at lives of 6b came in (lineage 3 life 1, lineage 4 life 8: the same two rooms
to and fro, from tick 935 and tick 1060 to the end; lineage 0 life 5, a life that goes on
eating: the same to and fro for some tens of ticks at a time, and then out of it). Four dead
lives looked at, four locked between two rooms.

### Step 6d: the bigger land with both dials. Bars written before the run

No new mechanism: the land of side 10 (100 rooms, 400 ways, 4000 ticks) at 512 cells, where 6a
gave 73 / 39 / 42 / 19 / 17 meals a life with a road in the head on 0.08-0.18 of the ticks
(`perfect` 141, `seeker` 231, blind 5), now with the two dials of 6b and 6c as they were set
at side 8 and not touched: `--curious 0.7`, `--budget 1024`, and both. `seeker` over `perfect`
is x1.63 here where it was x1.15 at side 8, so what wandering finds should matter more.

Bars: the exam's lands, 16 lives, five lineages, against 6a's records:

- **L1.** Curiosity alone (256 rooms a thought): at least 1.5 times 6a's meals, on at least
  four lineages.
- **L2.** Both dials: at least twice 6a's meals, on at least three.
- **L3.** Both dials: at least 0.7 of `perfect` (99 meals), on at least two.

Reported with no bar: 1024 rooms alone; lives with no meal in the last third.

Prediction: L1 met on four, L2 on three (0, 1, 4), L3 missed: lineage 0 only. The doubt: the
weak lineages (19 and 17 meals) may be bound by what 512 cells hold of 400 ways, and then
neither dial reaches them.

**Outcome (2026-11-16; records `runs/tissue/hold/hold-s*.world10.thinks.*.json`).**

| side 10, 512 cells, 16 lives | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 6a (256 rooms a thought, a way at random) | 73 | 39 | 42 | 19 | 17 |
| curious 0.7 | 107 | 55 | 77 | 22 | 24 |
| 1024 rooms a thought | 162 | 79 | 135 | 56 | 64 |
| both | 167 | 92 | 109 | 49 | 71 |
| both, times 6a | 2.29 | 2.36 | 2.60 | 2.58 | 4.18 |
| lives with no meal in the last third: 6a / curious / 1024 / both | 2 / 0 / 0 / 1 | 0 / 0 / 4 / 6 | 5 / 2 / 0 / 0 | 0 / 2 / 2 / 4 | 0 / 2 / 1 / 0 |

- **L1: missed.** Curiosity alone gives 1.5 times 6a's meals on one lineage (2: x1.83); the
  rest x1.16-1.47.
- **L2: met**, on all five (x2.3-4.2).
- **L3: met**, on two (167 and 109; `perfect` 141, so 0.7 of it is 99).

Predicted: L1 on four (wrong), L2 on three (met, on five), L3 on one (wrong, two). At side 10
the wider thought does nearly all of it: 1024 rooms alone gives 162 / 79 / 135 / 56 / 64, and
curiosity on top adds to three lineages and takes from two (lineage 2: 135 to 109). Lineage 0
eats more than `perfect` (167 against 141) and 0.72 of `seeker` (231). The locked lives of 6b
are here too where a road is found often (1024 rooms: 7 of 80; with curiosity: 11).

### Step 6e: a road kept. Bars written before the run

The fault 6b found is in my thinking procedure, not in the cells: the body takes the first way
of a road and the brain thinks afresh on the next tick, so a road is never walked and a false
one is never put to the test. Two thoughts that disagree pass the body between two rooms for
the rest of its life.

**The mechanism** (`--keep`, BY HAND, off unless asked for): a road once thought of is kept,
with the room a copy said after each of its ways. The body walks it a way a tick with no new
thought for as long as the room it comes to is the room that was said and the wanted look is
the same. At the first room that is not as said (a wall where a way was thought, another
room, the end of the road with no fruit there) the brain thinks afresh, from what it now sees.
Nothing is added to the cells; what changes is when a thought is thought. The argument for it:
a road walked is a road tested, and where it is false the body sees the truth and its wires
learn it by walking, which a first step taken and forgotten never does. Yardsticks are
untouched by it (their maps are true, and a kept road and a road thought afresh are the same).

Seen before these bars, on the desktop while checking the code: three lives of the exam's lands
of SIDE 6, lineage 0, 256 rooms a thought: 184 / 206 / 227 meals with a road kept where the
record says 162 / 119 / 207, a thought afresh on 0.37 of the ticks. The exam below is at side
8, which nobody has seen with a road kept.

Bars: the exam's lands, side 8, 256 cells, 16 lives, five lineages, 4096 rooms a thought,
`--keep` against 6b with the same rooms (163 / 69 / 123 / 51 / 62 meals; 16 lives of 80 with no
meal in the last third; trips of 9+ in the last third straight 0.49 / 0.37 / 0.28 / 0.37 /
0.10; `perfect` 182, `seeker` 210):

- **K1.** Lives with no meal in the last third: at most 4 of 80.
- **K2.** Meals a life at least 1.3 times 6b's, on at least three lineages.
- **K3.** At least 0.7 of `perfect` (127 meals), on at least three.
- **K4.** Trips of 9+ steps in the last third straight at least 0.6 of the time, on at least two.

Reported with no bar, as the VM has room: a road kept with 256 rooms a thought (against 6a);
with curiosity and 1024 rooms as well.

Prediction: K1 met; K2 met on three (the lineages that locked: 1, 3, 4); K3 missed, two (0 and
2); K4 missed, one (lineage 0). The doubt: a false road walked to its end carries the body
away from the fruit, and a longer round of roads may take the place of the short one.

**Outcome (2026-11-16; records `runs/tissue/hold/hold-s*.world8.thinks.budget4096.keep.json`).**

| side 8, 256 cells, 4096 rooms a thought, 16 lives | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 6b, a thought afresh on every tick: meals a life | 163 | 69 | 123 | 51 | 62 |
| a road kept | 241 | 212 | 214 | 193 | 160 |
| times 6b | 1.48 | 3.07 | 1.74 | 3.78 | 2.58 |
| 6b: lives with no meal in the last third | 0 | 6 | 0 | 9 | 1 |
| a road kept | 0 | 0 | 0 | 0 | 0 |
| 6b: trips of 9+ in the last third, straight | 0.49 | 0.37 | 0.28 | 0.37 | 0.10 |
| a road kept | 0.71 | 0.57 | 0.57 | 0.65 | 0.17 |
| a road kept: meals by third | 24 / 106 / 112 | 18 / 90 / 104 | 20 / 95 / 100 | 15 / 74 / 105 | 19 / 78 / 63 |
| a road kept: a thought thought afresh, per tick | 0.36 | 0.38 | 0.39 | 0.42 | 0.45 |

- **K1: met.** No life of 80 ends locked (6b: 16).
- **K2: met**, on all five (x1.5-3.8).
- **K3: met**, on all five: 160-241 meals against 127 asked; every lineage eats nearly as
  much as `perfect` (182) or more, and three as much as `seeker` (210) or more.
- **K4: met**, on two (lineages 0 and 3: 0.71 and 0.65).

Predicted K1 and K2 (right), K3 and K4 missed (wrong: both met). The largest step of the world
so far, and the weak lineages gain most: the five lineages, 36-95 meals apart in 6a, are now
160-241. What was mended is mine, not the cells': WHEN a thought is thought.

That a brain beats `perfect` says what `perfect` is not: it knows only the ways it has gone,
and once it has a road to every tree it never goes a way it has not gone, so a shortcut
through a way never gone stays unknown for life (its trips of 1-2 steps in the last third are
straight 0.90; the brain's 1.00). The look (journal, 2026-11-16) bears the guess out: by
the end of a life the brain has gone every open way of its land where the perfect memories
leave 0.06-0.21 never gone, and more than half its first goings come on a false road it was
walking toward the fruit. A brain that thinks afresh every tick explores as much and eats a
third as much: keeping a road adds the use of what was found, not the finding.

### Step 6f: the claim of the line in a world. Bars written before the run

The same genome, grown bigger, is smarter: asked now in a world, with the road kept. The land
of side 10 (100 rooms, 400 ways counting walls, 4000 ticks), `--keep`, 4096 rooms a thought,
brains of 256, 512 and 1024 cells (the genomes were searched at 8-64 interneurons, so all three
are sizes no search saw), 16 lives, five lineages. Nothing else changed. Why growth should pay
here: the land has 400 ways to hold and a brain holds about 0.7 links per interneuron (rung 1),
so 256 cells hold less than the land and 1024 more than it.

Against 6d at 512 cells with 1024 rooms a thought (162 / 79 / 135 / 56 / 64) and the yardsticks
of side 10 (`perfect` 141, `seeker` 231, born knowing the land 476):

- **G1.** At 512 cells, at least 1.3 times 6d's meals, on at least four lineages.
- **G2.** Growth pays: 512 cells eat at least 1.3 times what 256 cells eat, on at least four.
- **G3.** Growth goes on: 1024 cells eat at least 1.1 times what 512 cells eat, on at least
  three.
- **G4.** At 512 cells at least 0.7 of `seeker` (162 meals), on at least three.

Reported with no bar: lives with no meal in the last third; the road kept at side 8 with 256
rooms a thought (does the gain of 6e need a wide thought?).

Prediction: G1 on four, G2 on five, G3 on three, G4 on three. The doubt: G3, if 512 cells
already hold the land, 1024 add only what a vote of more cells adds to recall.

**Outcome, in part (2026-11-16; 1024 cells still running).**

| side 10, a road kept, 4096 rooms | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 6d (512 cells, 1024 rooms, a thought afresh) | 162 | 79 | 135 | 56 | 64 |
| 256 cells | 237.9 | 172.8 | 222.6 | 150.7 | 98.9 |
| 512 cells | 301.1 | 267.4 | 289.3 | 224.0 | 192.6 |
| 512 over 256 | 1.27 | 1.55 | 1.30- | 1.49 | 1.95 |
| lives with no meal in the last third, 256 / 512 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| trips of 9+ in the last third straight, 256 / 512 | 0.58 / 0.75 | 0.32 / 0.69 | 0.46 / 0.62 | 0.40 / 0.70 | 0.09 / 0.28 |

- **G1: met**, on five (x1.9-4.0).
- **G2: missed**, three of four: lineage 2 is at 1.2997 (289.31 over 222.6), lineage 0 at 1.27.
- **G4: met**, on five; four lineages at 512 cells eat more than `seeker` (231), lineage 0 0.63
  of born knowing the land (476).
- **G3 (2026-11-17): missed.** 1024 cells eat 1.1 times what 512 eat on two lineages (3: x1.13,
  4: x1.21), not three; every lineage still gains (x1.06-1.21).

| side 10, a road kept, 4096 rooms | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 256 cells | 237.9 | 172.8 | 222.6 | 150.7 | 98.9 |
| 512 cells | 301.1 | 267.4 | 289.3 | 224.0 | 192.6 |
| 1024 cells | 323.8 | 290.2 | 307.9 | 253.4 | 233.8 |
| 512 over 256 | 1.27 | 1.55 | 1.2999 | 1.49 | 1.95 |
| 1024 over 512 | 1.08 | 1.09 | 1.06 | 1.13 | 1.21 |

Growth pays at every step on every lineage and pays less at each: x1.27-1.95, then x1.06-1.21.
The 1024-cell records were overwritten in turn (by the repeated 256-cell runs, written under one
name); `runs/tissue/hold/world10.thinks.budget4096.keep.m1024.printed.json` keeps their printed
lines, and the runs go again under `.m1024` names.

Growth pays on every lineage, x1.27-1.95, and most on the weakest (4: x1.95); long trips get
straighter with size on all five. The 256-cell records were overwritten by the 512-cell runs
(one file per genome and land; the size was not in its name): the figures above are the runs'
own printed lines, and the runs are repeated to restore the records. Names now carry the size
when it is not 256, and the land when it is not of the default make.

Side 8 with a road kept and a NARROW thought (256 rooms; reported, no bar): 90 / 83 / 103 / 75
/ 59 meals (6a 95 / 55 / 88 / 53 / 36; with 4096 rooms 241 / 212 / 214 / 193 / 160); a road is
found on 0.24-0.33 of the ticks, so there is little to keep. Width of thought and a road kept
are worth little alone and much together.

## Breadth (from 2026-11-16, the user's word: many ideas, data for a later reading)

Each item: one change against a known arm, five lineages, 16 lives, a prediction written
before the run that serves as its bar. The point is a wide, comparable record.

### Breadth 1: other lands, and a looser match. Predictions before the run

Side 8, 256 cells, a road kept, 4096 rooms a thought, the exam's seeds; yardsticks rerun for
each land first (`--yardsticks --walls W` / `--trees N`, files named with the land):

| land of side 8 | blind | `perfect` | `seeker` | born knowing |
|---|---|---|---|---|
| walls 0.3, 10 trees (the exam's) | 6.6 | 182 | 210 | 380 |
| walls 0.5, a maze | 2.3 | 20 | 54 | 257 |
| walls 0.1, open | 13.4 | 292 | 363 | 469 |
| 21 trees | 5.9 | 139 | 168 | 376 |

- **B1, the maze** (`--walls 0.5`): roads are long and most ways are walls, so a memory that
  explores by the dice starves (`perfect` 20). Prediction: the brain eats more than `seeker`
  (54) on four lineages and 100 or more on two; its false roads explore toward the fruit.
- **B2, the open land** (`--walls 0.1`): shortcuts everywhere, and a perfect memory finds them
  as it wanders. Prediction: the brain's lead over `perfect` shrinks: below `perfect` (292) on
  three lineages.
- **B3, a looser match** (`--slack 1`, on the exam's land: a copy that says the wanted look
  with one bit of twelve wrong ends the thought): false hits about thirteen times as many, so
  more false roads walked. Prediction: fewer meals than with an exact match (241 / 212 / 214 /
  193 / 160) by a tenth or more, on four lineages.

Next in line, when threads are free: 21 trees; the 256-cell records of 6f again.

**Outcome (2026-11-16; records `runs/tissue/hold/hold-s*.world8.walls0.*.json`,
`*.world8.thinks.slack1.*.json`).**

| side 8, 256 cells, a road kept, 4096 rooms | lineage 0 | 1 | 2 | 3 | 4 | `perfect` | `seeker` | born knowing |
|---|---|---|---|---|---|---|---|---|
| the exam's land (6e) | 241 | 212 | 214 | 193 | 160 | 182 | 210 | 380 |
| B1, the maze (walls 0.5) | 76 | 89 | 97 | 82 | 49 | 20 | 54 | 257 |
| B2, the open land (walls 0.1) | 334 | 308 | 328 | 289 | 268 | 292 | 363 | 469 |
| B3, one bit of slack, the exam's land | 40 | 40 | 38 | 37 | 30 | | | |
| lives locked of 16: maze / open / slack | 5 / 0 / 14 | 2 / 0 / 14 | 0 / 0 / 14 | 0 / 0 / 14 | 0 / 0 / 14 | | | |

- **B1: half met.** Above `seeker` on four lineages (met), 100 or more on none (missed; best
  97). In the maze the brain eats 2.4-4.9 times what a perfect memory eats: the harder the
  land is to explore by the dice, the more the brain's walked mistakes are worth.
- **B2: missed.** Below `perfect` on two lineages, not three; above it on three, below
  `seeker` on all five. In an open land the lead over a perfect memory shrinks, as predicted,
  but does not go.
- **B3: met, and far beyond:** a fifth of the meals, and 70 lives of 80 locked again in spite
  of the road kept. With one bit of slack a road is found on 0.97 of the ticks, and the body
  passes from near-miss to near-miss. The exact match of the wanted look, a hand-made part, is
  load-bearing: what makes a thought end is as much a part of thinking as what it searches.

### Breadth 2: more trees, a smaller brain, and a check. Predictions before the run

Side 8, a road kept, 4096 rooms a thought, five lineages, 16 lives:

- **B4, 21 trees** (`--trees 21`; `perfect` 139, `seeker` 168, born knowing 376): the brain
  above `perfect` on four lineages and above `seeker` on three.
- **B5, 128 cells** (`--sizes 128`): 256 cells eat at least 1.25 times what 128 eat (6e: 241 /
  212 / 214 / 193 / 160), on four lineages.
- **The 256-cell runs of 6f again** (side 10): the same figures to the last digit as the lines
  they printed (237.9 / 172.8 / 222.6 / 150.7 / 98.9). A life is a pure function of its seed.
- **B8, 512 cells at side 8** (written before its run, queued behind breadth 3): 512 cells eat
  at least 1.15 times what 256 eat (241 / 212 / 214 / 193 / 160) on four lineages. With B5
  that gives a curve of growth at one land: 128, 256, 512.

**Outcome (2026-11-17).**

| side 8, a road kept, 4096 rooms | lineage 0 | 1 | 2 | 3 | 4 | `perfect` | `seeker` |
|---|---|---|---|---|---|---|---|
| 128 cells (B5) | 172.7 | 121.8 | 159.0 | 118.8 | 66.6 | 182 | 210 |
| 256 cells (6e) | 241.4 | 211.9 | 214.4 | 193.4 | 160.2 | | |
| 512 cells (B8) | 250.1 | 234.6 | 241.2 | 213.5 | 191.1 | | |
| 256 over 128 | 1.40 | 1.74 | 1.35 | 1.63 | 2.40 | | |
| 512 over 256 | 1.04 | 1.11 | 1.13 | 1.10 | 1.19 | | |
| 21 trees, 256 cells (B4) | 239.3 | 210.3 | 226.1 | 194.4 | 169.3 | 139 | 168 |

- **B4 (21 trees): met**, above `perfect` and `seeker` on all five lineages (predicted four and
  three). The brain's figures hardly move with twice the trees; the yardsticks fall.
- **B5 (128 cells): met**, 256 cells eat 1.35-2.40 times as much, on all five.
- **The 256-cell runs of 6f again: met**, the same to the last digit (237.94 / 172.75 / 222.56 /
  150.69 / 98.94). A life is a pure function of its seed.
- **B8 (512 cells): missed**, 1.15 times on one lineage (4); x1.04-1.19.

A KNEE, read across 6f and breadth 2: growth pays much while the brain has fewer cells than the
land has ways and little after. Side 8 (256 ways): 128 to 256 cells x1.35-2.40, 256 to 512
x1.04-1.19. Side 10 (400 ways): 256 to 512 x1.27-1.95, 512 to 1024 x1.06-1.21. That is what
rung 1 measured in silence (about 0.7 links held per interneuron) seen from a world. A claim to
test, breadth 4.

### Breadth 3: a land that changes under the body. Predictions before the run

`--shift F` (`world.moved`): at the half of a life a share F of the land's walls opens and as
many open places close (each only if the land still hangs together); looks, trees and the
body's room stay. The change has dice of its own, so the land is the exam's up to it. What is
asked: does a brain that learned a land unlearn it? Its wires hold a way that is now a wall,
and a wall that is now a way; only walking tells it.

Yardsticks in the changed lands (side 8; `oracle` is born knowing the land BEFORE the change
and learns the rest by walking, as `perfect` does; meals in the last third, all after the
change, against the same land unchanged):

| last third of a life | `perfect` | `seeker` | born knowing (before) |
|---|---|---|---|
| unchanged | 99 | 107 | 127 |
| shift 0.25 | 73 (0.74) | 76 (0.71) | 96 (0.76) |
| shift 0.5 | 64 (0.65) | 57 (0.53) | 73 (0.58) |

A table never tries a way it holds for a wall, so a wall that opened stays a wall to it for
life; the brain's walked mistakes may find it.

- **B6, shift 0.25** (256 cells, a road kept, 4096 rooms): the brain's last third at least 0.74
  of its own unchanged last third (6e: 112 / 104 / 100 / 105 / 63) on four lineages, as well as
  a perfect memory recovers or better.
- **B7, shift 0.5**: at least 0.65 of it on three lineages.

Prediction: both met; the doubt is that a plastic rule which held a land for 1280 ticks may
hold on to it (the old link louder than one walk can overwrite).

**Outcome (2026-11-17; records `runs/tissue/hold/hold-s*.world8.shift*.json`).**

| side 8, 256 cells, a road kept; meals in the last third (all after the change) | lineage 0 | 1 | 2 | 3 | 4 | `perfect` | `seeker` |
|---|---|---|---|---|---|---|---|
| unchanged (6e) | 111.7 | 103.8 | 100.0 | 104.7 | 62.6 | 98.7 | 106.9 |
| shift 0.25 (B6) | 80.7 | 80.6 | 83.6 | 72.4 | 49.4 | 73.1 | 76.4 |
| over unchanged | 0.72 | 0.78 | 0.84 | 0.69 | 0.79 | 0.74 | 0.71 |
| shift 0.5 (B7) | 41.4 | 49.6 | 54.6 | 38.9 | 23.1 | 64.4 | 56.9 |
| over unchanged | 0.37 | 0.48 | 0.55 | 0.37 | 0.37 | 0.65 | 0.53 |
| shift 0.5: lives locked of 16 | 2 | 0 | 1 | 2 | 2 | | |
| shift 0.5: ways right at the end, of those gone (unchanged land 0.95-0.96), lineages 0 and 4 | 0.92 | | | | 0.86 | | |

- **B6: missed** by a little: 0.74 or more on three lineages, not four (0.69-0.84); in the last
  third above `perfect` on three.
- **B7: missed** outright: 0.37-0.55, every lineage below `perfect` and `seeker`.

The doubt was the truth. A small change the brain takes about as well as a perfect memory; a
large one it does not, though by the end of the life its map is again mostly right (0.86-0.92 of
the ways gone). So it is the time it takes to unlearn: a link held for 1280 ticks is not undone
by one walk. A brain that must live in a changing world needs forgetting it does not have; no
switch for it exists yet.

### Breadth 4: the knee asked at two more lands, and two old switches with a road kept

A road kept, 4096 rooms a thought, five lineages, 16 lives. Records now hold the tick of every
meal (`meal_ticks`), so a reader can draw any curve of a life without running it again.

THE KNEE as a claim: growth pays at least 1.25 times a doubling while the brain has fewer cells
than its land has ways (walls counted: four to a room), and at most 1.15 times once it has more.

- **B9, side 6** (144 ways; `perfect` 121, `seeker` 172, born knowing 259) at 64, 128, 256 and
  512 cells: 64 to 128 at least 1.25 on four lineages; 256 to 512 at most 1.15 on four. 128 to
  256 straddles the knee: reported.
- **B10, side 12** (576 ways, 5760 ticks; `perfect` 107, `seeker` 247, born knowing 601) at 256
  and 512 cells: 512 over 256 at least 1.25 on four.
- **B11, curiosity with a road kept** (side 8, 256 cells, `--curious 0.7`): within a tenth of a
  road kept alone (241 / 212 / 214 / 193 / 160) on four lineages; with a road in the head on
  0.85 of the ticks there is little left for curiosity to steer.
- **B12, imagery with a road kept** (side 8, 256 cells, `--imagery`): reported, no bar. Before,
  imagery helped the weak lineages and ruined lineage 3; a guess: the same.
- **The 1024-cell runs of 6f again**, under `.m1024`: the same figures as their printed lines.

**Outcome (2026-11-17; side 12 at 512 cells on lineages 3 and 4 and three of the 1024 runs
finished that evening and were written in then).**

| a road kept, 4096 rooms; meals a life | lineage 0 | 1 | 2 | 3 | 4 | `perfect` | `seeker` | born knowing |
|---|---|---|---|---|---|---|---|---|
| side 6 (144 ways), 64 cells | 134.6 | 95.6 | 121.4 | 114.2 | 61.6 | 121 | 172 | 259 |
| 128 cells | 159.2 | 137.6 | 151.7 | 136.9 | 113.2 | | | |
| 256 cells | 175.1 | 165.1 | 181.8 | 172.7 | 138.6 | | | |
| 512 cells | 157.6 | 171.2 | 185.2 | 176.9 | 143.3 | | | |
| 128 / 64, 256 / 128, 512 / 256 | 1.18, 1.10, 0.90 | 1.44, 1.20, 1.04 | 1.2497, 1.20, 1.02 | 1.20, 1.26, 1.02 | 1.84, 1.22, 1.03 | | | |
| side 12 (576 ways), 256 cells | 153.4 | 72.4 | 133.5 | 49.0 | 26.7 | 107 | 247 | 601 |
| 512 cells | 333.6 | 274.2 | 297.9 | 192.4 | 154.9 | | | |
| 512 / 256 | 2.17 | 3.79 | 2.23 | 3.93 | 5.80 | | | |

- **B9, side 6: half met.** 64 to 128 cells at least 1.25 on two lineages, not four (lineage 2
  at 1.2497); 256 to 512 at most 1.15 on all five (met). At side 6 even 64 cells eat about as
  much as a perfect memory, and the curve flattens at 256 cells, not 144: the knee is nearer
  one to two cells per way than one.
- **B10, side 12: met**, x2.2-5.8 on all five; no life locked. At 512 cells every lineage is
  above `perfect` (107) and three are above `seeker` (247).

| side 8, 256 cells, a road kept, 4096 rooms | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| a road kept (6e) | 241.4 | 211.9 | 214.4 | 193.4 | 160.2 |
| and curious 0.7 (B11) | 253.3 | 226.1 | 228.3 | 201.7 | 170.6 |
| over a road kept | 1.05 | 1.07 | 1.06 | 1.04 | 1.07 |
| ways gone by an eighth of the life: kept / and curious | 0.57 / 0.68 | 0.55 / 0.72 | 0.58 / 0.71 | 0.56 / 0.63 | 0.54 / 0.68 |
| and imagery (B12) | 230.4 | 156.2 | 223.8 | 207.3 | 176.1 |
| over a road kept | 0.95 | 0.74 | 1.04 | 1.07 | 1.10 |

- **B11: met**, within a tenth on all five, and above on all five (x1.04-1.07): it explores
  faster (0.63-0.72 of the ways by an eighth of a life against 0.54-0.58).
- **B12** (no bar): my guess (the weak helped, lineage 3 ruined) was wrong in its parts: lineage
  1 loses a quarter (three lives locked), lineage 3 and 4 gain.
- The 1024-cell runs again: all five the same to the digit as their printed lines (323.81 /
  290.19 / 307.94 / 253.44 / 233.81).

### Breadth 5, set before leaving: the knee at side 12, unlearning with more cells, a maze with more cells

Queued on the VM with `infra/dev/queue.sh` (side 12 at 1024 cells is a day of running):

- **B13, side 12 at 1024 cells**: 1024 over 512 between 1.05 and 1.30 on four lineages (the
  brain passes the land's 576 ways).
- **B14, shift 0.5 at 512 cells** (side 8): the last third over the same brain's unchanged last
  third (B8: 115.9 / 113.4 / 107.3 / 114.4 / 83.5) within 0.1 of what 256 cells did (0.37 / 0.48
  / 0.55 / 0.37 / 0.37) on four lineages: more cells do not unlearn faster.
- **B15, the maze at 512 cells**: 512 over 256 (76.4 / 88.6 / 97.4 / 82.1 / 48.5) at most 1.2 on
  four lineages, as the knee says for 256 ways; if it is more, the knee is about how hard the
  land is, not how many ways it has.

**Outcome (B14 and B15 on 2026-11-17; B13, which ran for 11 to 16 hours, on the 18th: below).**

| side 8, a road kept, 4096 rooms; meals in the last third (all after the change) | lineage 0 | 1 | 2 | 3 | 4 | `perfect` | `seeker` |
|---|---|---|---|---|---|---|---|
| 512 cells, unchanged (B8) | 115.9 | 113.4 | 107.3 | 114.4 | 83.5 | 98.7 | 106.9 |
| 512 cells, shift 0.5 (B14) | 46.4 | 54.3 | 42.2 | 39.8 | 41.2 | 64.4 | 56.9 |
| over unchanged | 0.40 | 0.48 | 0.39 | 0.35 | 0.49 | 0.65 | 0.53 |
| 256 cells, over unchanged (B7) | 0.37 | 0.48 | 0.55 | 0.37 | 0.37 | | |
| 512 cells less 256 | +0.03 | +0.00 | -0.15 | -0.02 | +0.13 | | |
| lives locked of 16: 256 / 512 cells | 2 / 2 | 0 / 0 | 1 / 2 | 2 / 2 | 2 / 1 | | |
| ways right at the end, of those gone: 256 / 512 cells (unchanged land at 512: 0.98-0.99) | 0.92 / 0.92 | 0.94 / 0.93 | 0.92 / 0.93 | 0.92 / 0.92 | 0.86 / 0.92 | | |

- **B14: missed by the letter, right in what it says.** Within 0.1 of the 256-cell figure on
  three lineages, not four (lineage 2 falls by 0.15, lineage 4 rises by 0.13); the five
  together 0.42 at 512 cells against 0.43 at 256. No lineage reaches `perfect` (0.65), and in
  meals none reaches `seeker`. More cells do not unlearn faster.

A look the records now allow (`meal_ticks`), not a bar: meals per 160 ticks, mean of 16 lives,
512 cells, shift 0.5. The land changes at tick 1280; up to it the life is the unchanged one.

| meals per 160 ticks, from tick | 0 | 160 | 320 | 480 | 640 | 800 | 960 | 1120 | **1280** | 1440 | 1600 | 1760 | 1920 | 2080 | 2240 | 2400 | unchanged, last third |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lineage 0 | 0.2 | 0.8 | 1.6 | 5.6 | 9.8 | 14.9 | 18.8 | 21.6 | 2.6 | 2.9 | 3.2 | 3.6 | 5.7 | 8.9 | 13.5 | 14.2 | 21.7 |
| 1 | 0.4 | 0.1 | 1.4 | 6.0 | 9.3 | 13.7 | 17.8 | 16.8 | 2.1 | 2.5 | 3.4 | 5.7 | 6.8 | 10.6 | 14.6 | 15.2 | 21.3 |
| 2 | 0.2 | 1.5 | 2.9 | 6.5 | 12.6 | 16.1 | 17.2 | 19.5 | 2.5 | 2.6 | 2.7 | 5.3 | 5.9 | 7.6 | 10.2 | 11.7 | 20.1 |
| 3 | 0.4 | 0.6 | 1.4 | 2.1 | 3.6 | 7.6 | 13.1 | 16.1 | 3.0 | 2.1 | 2.6 | 3.3 | 5.1 | 8.1 | 10.4 | 12.0 | 21.5 |
| 4 | 0.6 | 0.4 | 2.1 | 4.2 | 6.2 | 11.4 | 14.4 | 18.5 | 2.8 | 2.5 | 3.2 | 5.0 | 6.6 | 8.5 | 9.9 | 10.2 | 15.7 |

- Right after the change the brain eats 0.12-0.19 of what it ate just before it.
- It climbs back no faster than a newborn learns a land, and on three lineages more slowly (the
  first 160 ticks with ten meals start 640-960 ticks after birth and 800-1120 after the
  change). In the second half of its life it eats 0.63-1.04 of what it ate in the first (54.7 /
  60.8 / 48.6 / 46.5 / 48.7 meals against 73.4 / 65.6 / 76.6 / 44.9 / 57.8).
- At the end of the life it eats 0.52-0.70 of the unchanged pace and is still climbing: the
  figure of the bar, a whole last third, is a mean over a climb.
- Not asked, and the next thing to ask on this road: what the old map is WORTH. A brain whose
  wires are emptied at the change (or one born then) has no false roads to walk; if it eats as
  much in the half-life after, half a land moved has cost the brain all it knew.

| side 8, the maze (walls 0.5), a road kept, 4096 rooms | lineage 0 | 1 | 2 | 3 | 4 | `perfect` | `seeker` | born knowing |
|---|---|---|---|---|---|---|---|---|
| 256 cells (B1) | 76.4 | 88.6 | 97.4 | 82.1 | 48.5 | 20 | 54 | 257 |
| 512 cells (B15) | 93.1 | 101.2 | 96.1 | 87.6 | 76.2 | | | |
| 512 over 256 | 1.22 | 1.14 | 0.99 | 1.07 | 1.57 | | | |
| the same on the exam's land (B8) | 1.04 | 1.11 | 1.13 | 1.10 | 1.19 | | | |
| 512 less 256 in meals, and its error over the 16 lives (the same lands at both sizes) | +17 +- 23 | +13 +- 22 | -1 +- 21 | +6 +- 13 | +28 +- 14 | | | |
| lives of 16 that eat more at 512 cells | 9 | 10 | 10 | 10 | 12 | | | |
| lives with 5 meals or fewer in the last third: at 256 / at 512 (in both) | 6 / 5 (2) | 2 / 3 (0) | 2 / 3 (1) | 1 / 2 (1) | 4 / 3 (1) | | | |
| trips of 9+ steps, straight: 256 / 512 cells | 0.66 / 0.73 | 0.46 / 0.68 | 0.51 / 0.64 | 0.52 / 0.64 | 0.27 / 0.44 | | | |

- **B15: missed by the letter** (at most 1.2 on three lineages, not four: lineage 0 at 1.218,
  lineage 4 at 1.57), **and the bar was finer than a maze allows.** A life in a maze eats or
  nearly starves (spread over lives 45-73 meals at 512 cells against 29-38 on the exam's land),
  and WHICH lives starve changes with the size of the brain (of the 1-6 starved lives of a
  lineage, 0-2 are the same at both sizes). So one lineage's ratio is known to about 0.2-0.3.
  Only lineage 4's gain stands clear of that. The five together: x1.16 in the maze, x1.11 on
  the exam's land.
- Read: a harder land does not move the knee as far as 16 lives can tell; the weakest lineage
  gains most, as everywhere. Long trips are straighter at 512 cells on every lineage and the
  meals do not follow, which suggests that in a maze what binds is finding the ways, not
  holding them (ways gone by an eighth of a life: 0.31-0.37 at both sizes).
- METHOD: look at the spread over lives in a land before setting a bar on a ratio there, and
  compare sizes life by life on the same lands.

**B13, collected 2026-11-18** (`hold-s*.world12.thinks.budget4096.keep.m1024.json`; 11 to 16
hours a run).

| side 12 (576 ways), a road kept, 4096 rooms | lineage 0 | 1 | 2 | 3 | 4 | `explorer` | born knowing |
|---|---|---|---|---|---|---|---|
| 256 cells | 153.4 | 72.4 | 133.5 | 49.0 | 26.7 | 517 | 601 |
| 512 cells | 333.6 | 274.2 | 297.9 | 192.4 | 154.9 | | |
| 1024 cells (B13) | 356.2 | 352.8 | 351.2 | 263.8 | 211.8 | | |
| 1024 over 512 | 1.07 | 1.29 | 1.18 | 1.37 | 1.37 | | |
| 1024 less 512 and its error over the 16 lives | +23 +- 11 | +79 +- 6 | +53 +- 5 | +71 +- 9 | +57 +- 12 | | |
| the first third: 512 / 1024 cells | 21.1 / 18.0 | 13.5 / 17.9 | 15.5 / 22.7 | 6.7 / 7.1 | 8.7 / 14.6 | 129 | 201 |
| the last third: 512 / 1024 cells | 175.1 / 183.3 | 155.1 / 183.2 | 159.3 / 166.2 | 140.3 / 174.7 | 59.2 / 88.7 | 193 | 199 |
| trips of 9+ steps, straight: 512 / 1024 | 0.60 / 0.71 | 0.47 / 0.68 | 0.51 / 0.61 | 0.44 / 0.60 | 0.24 / 0.40 | | |

- **B13: missed.** In the band (1.05-1.30) on three lineages; lineages 3 and 4 are above it
  (1.37). Past the land's ways growth pays more here than it did at sides 8 and 10 (x1.04-1.21):
  the knee is softer in a bigger land, and every difference but lineage 0's stands far clear of
  its error.
- All of the gain is in the later thirds. The first third hardly moves (7-23 meals at 1024
  cells, 7-21 at 512): more cells do not make a young brain know its land sooner. In its last
  third a brain of 1024 cells eats 0.83-0.92 of what one born knowing the land eats, on four
  lineages.

## After the breadth: the order chosen with the user (2026-11-17, evening)

Read off the ledger before choosing: where the meals are lost. Side 8, meals by third of a life:

| | first | second | last |
|---|---|---|---|
| born knowing the land | 127 | 125 | 127 |
| the brain, 512 cells, a road kept (five lineages) | 10-28 | 89-112 | 84-116 |
| `seeker` | 18 | 86 | 107 |
| `perfect` | 13 | 70 | 99 |

A grown brain in a land that does not change eats 0.85-0.91 of what one born knowing it eats
(the weak lineage 0.66). Nearly all of the gap is the first third: getting to know the land. At
side 12 the same (21 / 137 / 175 against 200 a third). No yardstick starts faster, because none
looks for what it does not know farther than the room it stands in.

The order, the user's word on my proposal:

1. **Wanting to know** (the main line, on the road to a world): with no road to the fruit, go to
   the nearest way not known. A yardstick first (step 6g), then the brain (6h). By hand: it is
   the will.
2. **Forgetting** (beside it, on the VM): what the old map is worth after a change, how many
   walks undo a way that became a wall; then the search is asked. The one road where the
   outcome would belong to the cell's rule.
3. **A thought as a wave in the tissue** (later): can a brain hold many rooms at once.

### Step 6g: a yardstick that seeks what it does not know. Prediction before the run

`explorer` (`tissue/world.py`): a perfect memory, as `seeker`; with no road to the fruit it
thinks of the nearest room, by ways it has gone, out of which there is a way it has never gone,
and takes the first way of that road (`seeker` looks no farther than the room it stands in).
With a road to the fruit it never seeks, as the others. The old yardsticks must come out the
same to the digit.

Predictions (the exam's lands, 16 lives):
- side 8 (`seeker` 18 / 86 / 107 = 210; born knowing 127 a third): the first third at least 40,
  a life at least 240; a guess, no bar: the last third no better than `seeker`'s, since both
  stop seeking once a road to every tree is known.
- side 12 (`seeker` 14 / 92 / 142 = 247; born knowing 200 a third): the first third at least
  40, a life at least 330.

**Outcome (2026-11-17; `runs/tissue/world*.yardsticks.json`, every land run again; the old
yardsticks came out the same to the digit in all nine).**

| meals a life | `perfect` | `seeker` | `explorer` (by third) | born knowing | the brain, a road kept, five lineages (cells) |
|---|---|---|---|---|---|
| side 6 | 121 | 172 | 221 (60 / 79 / 81) | 259 | 143-185 (512) |
| side 8 | 182 | 210 | 312 (83 / 114 / 115) | 380 | 191-250 (512) |
| side 10 | 141 | 231 | 412 (106 / 155 / 152) | 476 | 234-324 (1024) |
| side 12 | 107 | 247 | 517 (129 / 195 / 193) | 601 | 155-334 (512) |
| side 8, the maze | 20 | 54 | 228 (57 / 86 / 85) | 257 | 76-101 (512) |
| side 8, the open land | 292 | 363 | 378 (103 / 136 / 138) | 469 | 268-334 (256) |
| side 8, 21 trees | 139 | 168 | 328 (82 / 122 / 124) | 376 | 169-239 (256) |
| side 8, shift 0.25 | 142 | 158 | 256 (83 / 89 / 85) | 326 (127 / 102 / 96) | |
| side 8, shift 0.5 | 120 | 121 | 213 (83 / 70 / 60) | 279 (127 / 78 / 73) | |

- **Both predictions met, and both far too low**: side 8, the first third 83 (40 asked) and a
  life 312 (240); side 12, 129 and 517 (40 and 330). The guess was wrong: the last third is
  above `seeker`'s (115 against 107, 193 against 142), because seeking as far as the nearest
  way not known leaves a fuller map (0.82-0.87 of the ways gone by an eighth of a life against
  0.50-0.59).
- With a perfect memory, wanting to know is worth 0.81-0.89 of being born knowing the land, in
  every land that does not change. The brain's lead over a perfect memory (the maze: x2.4-4.9)
  was a lead over memories that do not seek: in the maze `explorer` eats 228 where the brain
  eats 76-101. `explorer` is the yardstick to hold the brain against from here on.
- A land that changes is NOT mended by it: after half the walls move its last third is 60
  (0.53 of its unchanged 115), no better than `perfect` (64) or one born knowing the old land
  (73). A table never doubts a wall: a wall that opened stays a wall to it for life, and it has
  no way never gone left to seek. Every table unlearns a way in one walk, so what a changing
  land asks for is not that. It is DOUBT: what was not met for long becoming not known again,
  so that it is sought again. That is forgetting in the plain sense, and it is where the first
  two roads meet (a yardstick for it first, when that road is taken).

### Step 6h: the brain wants to know. Pilot, then bars written before the run

`--explore Q` (BY HAND, `Mind.toward` in `tissue/world.py`): a thought that finds no road to
the wanted look no longer ends in the dice. Of the copies it made, the nearest one that spoke
of its room more quietly than Q times how loud the brain says the room it stands in marks a
way the brain does not know; the body takes the first way of the road to it, and with a road
kept it walks that road with no new thought. Nothing is added to the thought itself: the same
copies on the same wires, read once more for how sure they are. The will is given a second
want. A step toward what is not known is not counted as a plan (`sought` counts it).

What the brain has for "not known" is weaker than a table's. A look (lineages 0 and 4, 256
cells, side 8, the exam's lands, copies followed along the true land): the brain says the room
it stands in at 7.0-7.3 (the mean size of a bit), a copy in the dark says a way the body has
gone at 3.8-4.8 and one it has never gone at 2.0-2.5: half as loud, not a whisper. (The 0.0-0.6
that the records hold as `loud_never` is a mean over lives most of which had no way never gone
left.) Under 0.4 of the brain's own loudness fall 0.07-0.14 of the ways gone and 0.67-0.86 of
those never gone; under 0.5, 0.24-0.34 and 0.89-1.00. A way gone once is said right on 0.81
of the askings (lineage 4) and louder with every walk, so a false "not known" mends itself by
being walked.

**Q is set at 0.4 from that look, before any life was lived with it**, and the bars are
written before the exam and before the pilot's lines came back (a pilot on other lands, `--seed
1`, eight lives, lineages 0 and 4, Q 0.3 / 0.4 / 0.5, was started first and is reported beside
the exam: the VM was slower than I reckoned). A road kept, 4096 rooms, 16 lives, the exam's
lands, five lineages:

- **H1, side 8, 256 cells: the first third** (6e: 23.7 / 18.2 / 19.8 / 15.2 / 19.4; `explorer`
  83): at least 40 meals on four lineages.
- **H2, a life** (6e: 241.4 / 211.9 / 214.4 / 193.4 / 160.2; `explorer` 312): at least 1.10
  times 6e's on four lineages.
- **H3, ways gone by an eighth of a life** (6e: 0.54-0.58; `explorer` 0.82): at least 0.70 on
  four lineages.
- **H4, no lock**: at most one life of the 80 with no meal in its last third (6e: none).
- **H5, side 12, 512 cells** (B10: 333.6 / 274.2 / 297.9 / 192.4 / 154.9; `explorer` 517): a
  life at least 1.25 times B10's on four lineages. The bigger the land, the more of a life is
  spent getting to know it.
- **H6, the maze, 256 cells** (B1: 76.4 / 88.6 / 97.4 / 82.1 / 48.5; `explorer` 228): a life at
  least 1.5 times B1's on four lineages (far enough from 1 for a maze's spread; the lives are
  compared one by one as well).
- **The dial**, side 8 at Q 0.3 and 0.5, no bar: a guess that 0.4 is the best of the three on
  three lineages or more (0.3 misses half of what is not known, 0.5 calls a quarter of what is
  known not known).

Prediction: H1-H3 and H6 met; H5 met; H4 is the doubt. A way the brain cannot hold stays quiet
however often it is walked, and 256 cells for 256 ways is at the knee: the body may go back to
such a way for ever, a lock of a new kind.

**Outcome (2026-11-18; records `runs/tissue/hold/hold-s*.world*.explore0.*.keep*.json`, the
pilot's under `.seed1`).**

| side 8, 256 cells, a road kept, 4096 rooms | lineage 0 | 1 | 2 | 3 | 4 | `explorer` | born knowing |
|---|---|---|---|---|---|---|---|
| a road kept (6e), meals a life | 241.4 | 211.9 | 214.4 | 193.4 | 160.2 | 312 | 380 |
| and seeking, Q 0.4 | 263.9 | 236.1 | 245.4 | 223.5 | 177.6 | | |
| over 6e | 1.09 | 1.11 | 1.14 | 1.16 | 1.11 | | |
| the first third: 6e / seeking | 23.7 / 40.6 | 18.2 / 36.6 | 19.8 / 42.6 | 15.2 / 21.6 | 19.4 / 35.1 | 83 | 127 |
| the second third | 106.0 / 110.7 | 89.9 / 95.8 | 94.6 / 102.6 | 73.5 / 91.6 | 78.2 / 88.2 | 114 | 125 |
| the last third | 111.7 / 112.6 | 103.8 / 103.8 | 100.0 / 100.1 | 104.7 / 110.2 | 62.6 / 54.2 | 115 | 127 |
| ways gone by an eighth of a life | 0.57 / 0.79 | 0.55 / 0.79 | 0.58 / 0.79 | 0.56 / 0.71 | 0.54 / 0.76 | 0.82 | |
| steps toward a way not known, of all ticks | 0.07 | 0.08 | 0.08 | 0.10 | 0.11 | | |
| lives of 16 that eat more than in 6e | 14 | 16 | 16 | 14 | 13 | | |
| the dial, meals at Q 0.3 / 0.5 | 264.4 / 265.1 | 231.2 / 234.4 | 247.5 / 240.9 | 226.2 / 218.5 | 176.2 / 173.5 | | |

| the bigger land and the maze, Q 0.4 | lineage 0 | 1 | 2 | 3 | 4 | `explorer` | born knowing |
|---|---|---|---|---|---|---|---|
| side 12, 512 cells: a road kept (B10) | 333.6 | 274.2 | 297.9 | 192.4 | 154.9 | 517 | 601 |
| and seeking | 356.2 | 297.4 | 320.9 | 224.2 | 173.8 | | |
| over B10 | 1.07 | 1.08 | 1.08 | 1.17 | 1.12 | | |
| the first third: B10 / seeking | 21.1 / 29.1 | 13.5 / 22.1 | 15.5 / 25.1 | 6.7 / 9.4 | 8.7 / 15.4 | 129 | 201 |
| ways gone by an eighth of a life | 0.61 / 0.74 | 0.57 / 0.73 | 0.58 / 0.77 | 0.51 / 0.66 | 0.52 / 0.69 | 0.87 | |
| the maze, 256 cells: a road kept (B1) | 76.4 | 88.6 | 97.4 | 82.1 | 48.5 | 228 | 257 |
| and seeking | 164.3 | 140.6 | 145.4 | 125.2 | 81.6 | | |
| over B1 | 2.15 | 1.59 | 1.49 | 1.53 | 1.68 | | |
| ways gone by an eighth of a life | 0.37 / 0.76 | 0.34 / 0.73 | 0.36 / 0.75 | 0.36 / 0.62 | 0.31 / 0.69 | 0.82 | |
| lives of 16 that eat more than in B1 | 16 | 16 | 16 | 14 | 14 | | |
| lives with 5 meals or fewer in the last third: B1 / seeking | 6 / 0 | 2 / 0 | 2 / 0 | 1 / 0 | 4 / 1 | | |

- **H1: missed.** A first third of 40 on two lineages (40.6, 42.6); 36.6 and 35.1 on two more,
  21.6 on lineage 3. That is x1.4-2.2 over 6e's and half the explorer's at best.
- **H2: met**, x1.09-1.16, 1.10 on four (lineage 0 at 1.093); in every lineage 13-16 lives of
  16 eat more.
- **H3: met** on all five, 0.71-0.79 of the ways by an eighth of a life (the explorer 0.82).
- **H4: met.** No life locked, at any Q, in any land. The doubt did not come true.
- **H5: missed** on all five: x1.07-1.17 at side 12, where 1.25 was asked and the explorer is
  x2.1 over `seeker`.
- **H6: met**, x1.49-2.15, 1.5 on four (lineage 2 at 1.494). The lock of the maze is gone (7
  lives of 80 had no meal in their last third, none now).
- **The dial**: no difference between Q 0.3, 0.4 and 0.5 that 16 lives can show (each is best
  on one or two lineages). The guess that 0.4 would stand out was wrong.
- The pilot on other lands (8 lives, written after the bars): lineage 0 x1.09-1.10 (235 to
  256-259), lineage 4 x1.12-1.18 (151 to 169-178), the same picture.

**What it says.** The young brain now gets about its land nearly as fast as the explorer does
(0.71-0.79 of the ways by an eighth of a life against 0.82; at side 12, 0.66-0.77 against 0.87)
and in its first third eats at most half of what the explorer eats at side 8 (22-43 against
83) and at most a quarter at side 12 (9-29 against 129). So walking is no longer what holds a
young brain back. A
table knows a way when it has gone it once. The brain does not: the look of the forgetting
road (step 2 below) says a way gone once is said right on 0.37-0.61 of the askings and one gone
twice on 0.80-0.93, so a road of five hops through ways gone once comes out right about once in
twenty. And this will walks a way once and holds it for known. Where the dice could not find
the ways at all, the maze, seeking alone is worth x1.5-2.2 and ends the lock; where they could,
it is worth a tenth. In the last third of a life nothing has changed: it speeds the start.
NEXT on this road: one walk must write a way (asked of the search, the cell's rule), or the
will must walk a new way until it is known.

### Step 6i: one walk must write a way. Asked first with no search, then of the search. Predictions and bars before the runs

Where the two walks come from. The lineage genomes were found in the stream (exam 2). There
every loop is walked round two or three times, and where one loop is left for another the
stream shows a pair that is not a link, seen once: a coincidence. A rule found there was
taught not to believe a pair it has seen once (the stream's own perfect memory "needs two
laps: after one, a link and a coincidence have each been seen once and nothing tells them
apart"). In a land there are no coincidences: the body goes a way and comes to the room behind
it, every time. No search was ever shown a link once and asked for it.

**Asked first, no search**: exam 4 (forks, `python -m tissue.fork --ways 4 --body dendrites`:
rooms with four ways, a dendrite per way by hand as in a land, a study with no jump in it)
lived with ONE lap (`--laps 1`) and with three, beside the records' two. Prediction: with one
lap the lineage genomes hold less than a quarter of what they hold with two (K90 at hop 1, 256
cells, two laps: 857 / 926 / 935 / 647 / 891 links) on four lineages; with three laps no less
than with two.

**The search** (`tissue.train --arm walk`, new): the stream brain lives exam 4 with the
hand-made dendrite per way, four ways, 1/8 to 2 links per interneuron (a land of side 8 asks a
link per cell of a brain of 256), and each life has one, two or three laps, drawn. Closings of
16 ticks and shorter (`--hops 16 --vary`), brains of 8-64 cells, 4000 steps, started from each
lineage genome (`--init`). One thing is asked that was not: a link shown once. Bars, at sizes
no search saw and in lands no search saw:

- **W1, one lap is enough**: exam 4 with one lap, 256 cells, K90 at hop 1 at least half of
  what the lineage genome holds with two laps (429 / 463 / 467 / 324 / 446), on four lineages.
- **W2, nothing paid for it**: with two laps, 256 cells, K90 at hop 1 and at hop 4 at least
  0.8 of the lineage genome's (hop 4: 417 / 361 / 439 / 202 / 338), on four lineages.
- **W3, in a land** (the look `tissue.unlearn`, side 8, 256 cells): a way gone once since
  birth is said right on at least 0.85 of the askings (0.54 / 0.53 / 0.61 / 0.37 / 0.51), on
  four lineages.
- **W4, a life** (side 8, 256 cells, a road kept, 4096 rooms, seeking at 0.4): a first third
  of at least 55 meals on three lineages (6h: 40.6 / 36.6 / 42.6 / 21.6 / 35.1; `explorer` 83)
  and a life no smaller than 6h's (263.9 / 236.1 / 245.4 / 223.5 / 177.6) on four.
- **W5, the bigger land** (side 12, 512 cells): a life at least 1.15 times 6h's (356.2 / 297.4
  / 320.9 / 224.2 / 173.8) on three lineages.

Prediction: W1 and W3 met, W4 met. W2 is the doubt: a rule that believes at once may hold
less, and a thought may hold less far in the dark when the search's closings are 16 ticks and
not 32. W5 open.

**Outcome, asked first with no search (2026-11-19; `runs/tissue/hold/hold-s*.fork4x4.laps{1,3}.
dendrites.json`, run on the VM).**

| exam 4, four ways, 256 cells, the lineage genomes | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| K90 at hop 1: one lap / two / three | 573 / 857 / 1022 | 520 / 926 / 1120 | 584 / 935 / 1122 | 301 / 647 / 907 | 473 / 891 / 1105 |
| one lap over two | 0.67 | 0.56 | 0.63 | 0.47 | 0.53 |
| K90 at hop 4: one lap / two / three | 201 / 417 / 532 | 156 / 361 / 469 | 230 / 439 / 546 | 75 / 202 / 328 | 139 / 338 / 460 |
| 256 links shown once, bits right at hop 1 / hop 4 | 0.98 / 0.85 | 0.97 / 0.80 | 0.98 / 0.88 | 0.92 / 0.66 | 0.96 / 0.77 |

- **The prediction was wrong.** Shown once, the lineage genomes hold not a quarter but a half to
  two thirds of what they hold shown twice (0.47-0.67 at hop 1; 0.37-0.53 four hops deep).
  Three laps hold x1.19-1.40 of two, as said. So the rule does not wait for a second showing: it
  writes a link a part at a time, and every showing adds.
- Why a land still wanted two walks: exam 4 counts bits (K90 is where nine bits of ten are
  right) and a thought in a land asks for a whole room, bit for bit. With a link per cell shown
  once, 0.92-0.98 of the bits are right one hop on and 0.66-0.88 four hops on: enough for a
  count, not for a road.

**Outcome of the search (2026-11-19; five genomes `runs/tissue/walk/walk-s0..4.pt`, 4000 steps,
48 minutes each on the VM; their records beside them; the lineage genomes' figures in
brackets).**

| sizes and worlds no search saw | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| exam 4, 256 cells, ONE lap: K90 at hop 1 | 665 | 703 | 719 | 444 | 698 |
| over the lineage genome's with two laps (W1 asks 0.5) | 0.78 | 0.76 | 0.77 | 0.69 | 0.78 |
| over the lineage genome's with one lap | 1.16 | 1.35 | 1.23 | 1.47 | 1.47 |
| two laps: K90 at hop 1 / hop 4 | 951 / 558 | 1039 / 566 | 1026 / 600 | 829 / 341 | 1029 / 577 |
| over the lineage genome's (W2 asks 0.8) | 1.11 / 1.34 | 1.12 / 1.57 | 1.10 / 1.37 | 1.28 / 1.69 | 1.15 / 1.70 |
| one lap, 512 cells: K90 at hop 1 | 1236 (1048) | 1331 (965) | 1416 (1052) | 714 (390) | 1285 (808) |
| in a land (side 8, 256 cells): a way gone once / twice since birth, said right | 0.92 / 0.98 (0.54 / 0.92) | 0.92 / 0.97 (0.53 / 0.93) | 0.93 / 0.98 (0.61 / 0.93) | 0.80 / 0.96 (0.37 / 0.80) | 0.91 / 0.98 (0.51 / 0.87) |
| a wall bumped once | 0.96 (0.71) | 0.96 (0.77) | 0.96 (0.80) | 0.93 (0.72) | 0.96 (0.79) |
| a way that closed, gone once / twice since the change | 0.87 / 1.00 (0.23 / 0.47) | 0.88 / 1.00 (0.00 / 0.35) | 0.88 / 1.00 (0.01 / 0.61) | 0.64 / 0.94 (0.43 / 0.68) | 0.85 / 0.99 (0.00 / 0.52) |
| a wall that opened, gone once / twice | 0.78 / 1.00 (0.00 / 0.40) | 0.80 / 1.00 (0.00 / 0.23) | 0.78 / 1.00 (0.01 / 0.27) | 0.47 / 0.97 (0.02 / 0.30) | 0.60 / 0.99 (0.00 / 0.14) |
| walls that opened, gone by the end of the life | 0.56 (0.75) | 0.65 (0.77) | 0.70 (0.68) | 0.73 (0.83) | 0.74 (0.85) |
| how loud a copy says a way never gone / gone once | 3.5 / 5.6 (2.8 / 3.7) | 3.1 / 5.7 (3.1 / 4.2) | 3.1 / 5.5 (3.0 / 3.9) | 4.1 / 5.7 (3.6 / 4.3) | 3.7 / 5.9 (3.2 / 3.9) |
| meals in that life (half the walls moved at half-life, a road kept, no seeking, 8 lives) | 169 (120) | 193 (125) | 177 (130) | 132 (96) | 182 (94) |

| a life, a road kept, 4096 rooms, seeking at 0.4, 16 lives | lineage 0 | 1 | 2 | 3 | 4 | `explorer` | born knowing |
|---|---|---|---|---|---|---|---|
| side 8, 256 cells: the lineage genome (6h) | 263.9 | 236.1 | 245.4 | 223.5 | 177.6 | 312 | 380 |
| the genome of the walk arm | 310.9 | 294.8 | 302.1 | 264.1 | 272.3 | | |
| over 6h | 1.18 | 1.25 | 1.23 | 1.18 | 1.53 | | |
| the first third: 6h / walk | 40.6 / 64.8 | 36.6 / 62.7 | 42.6 / 66.2 | 21.6 / 31.6 | 35.1 / 63.9 | 83 | 127 |
| the second third | 110.7 / 120.6 | 95.8 / 112.8 | 102.6 / 116.6 | 91.6 / 112.5 | 88.2 / 112.9 | 114 | 125 |
| the last third | 112.6 / 125.5 | 103.8 / 119.3 | 100.1 / 119.2 | 110.2 / 120.1 | 54.2 / 95.4 | 115 | 127 |
| ways gone by an eighth of a life | 0.79 / 0.83 | 0.79 / 0.82 | 0.79 / 0.83 | 0.71 / 0.75 | 0.76 / 0.82 | 0.82 | |
| trips of 9 steps and more walked straight, the last third | 0.66 / 0.95 | 0.59 / 0.79 | 0.49 / 0.81 | 0.67 / 0.88 | 0.15 / 0.47 | 0.87 | 1.00 |
| the difference a life and its error; lives of 16 that eat more | +47 +- 2; 16 | +59 +- 3; 16 | +57 +- 3; 16 | +41 +- 3; 16 | +95 +- 4; 16 | | |
| side 12, 512 cells: the lineage genome (6h) | 356.2 | 297.4 | 320.9 | 224.2 | 173.8 | 517 | 601 |
| the genome of the walk arm | 441.2 | 406.2 | 428.2 | 333.1 | 356.1 | | |
| over 6h | 1.24 | 1.37 | 1.33 | 1.49 | 2.05 | | |
| the first third: 6h / walk | 29.1 / 59.9 | 22.1 / 61.8 | 25.1 / 69.9 | 9.4 / 16.7 | 15.4 / 62.3 | 129 | 201 |
| the second third | 150.8 / 187.1 | 117.6 / 167.8 | 143.2 / 180.8 | 63.4 / 128.9 | 101.3 / 171.3 | 195 | 201 |
| the last third | 176.2 / 194.2 | 157.8 / 176.6 | 152.6 / 177.6 | 151.4 / 187.4 | 57.1 / 122.4 | 193 | 199 |
| ways gone by an eighth of a life | 0.74 / 0.82 | 0.73 / 0.81 | 0.77 / 0.81 | 0.66 / 0.71 | 0.69 / 0.81 | 0.87 | |
| lives of 16 that eat more | 16 | 16 | 16 | 16 | 16 | | |

- **W1: met** on all five, 0.69-0.78. Read with the wrong prediction above: unsearched, four
  lineage genomes already stood at the bar (0.47-0.67). What the search added with one lap is
  x1.16-1.47.
- **W2: met** on all five, and the other way round from my doubt: shown twice the new genomes
  hold MORE, x1.10-1.28 one hop on and x1.34-1.70 four hops on. Nothing was paid in exam 4.
- **W3: met** on four: a way gone once is said right on 0.91-0.93 of the askings (lineage 3:
  0.80), where it was 0.37-0.61.
- **W4: met.** A first third of 62.7-66.2 on four lineages (55 asked; lineage 3: 31.6), and a
  life x1.18-1.53 of 6h's: all 80 lives eat more than the same life in 6h.
- **W5: met** on all five, x1.24-2.05; all 80 lives eat more.

**What it says.**
- One walk now writes a way, and one walk undoes one: a way that closed, gone once since the
  change, is said right on 0.85-0.88 (lineage 3: 0.64) where the lineage genomes said neither
  the old room nor the new. Nobody asked for the undoing; it came with the writing.
- Against the `explorer` the brain now stands at 0.85-1.00 of a life at side 8 (lineage 0: 310.9
  against 311.8) and 0.64-0.85 at side 12 with 512 cells. Its first third is 0.75-0.80 of the
  explorer's at side 8 on four lineages and about half at side 12 (60-70 against 129), where
  512 cells are fewer than the land's 576 ways.
- In the last third at side 8 four lineages eat as much as the explorer or more (119-126 against
  115; life by life +11 +- 4 for lineage 0, +5 +- 4 for lineages 1-3), and lineage 0 is at what
  one born knowing the land eats (127). The explorer stops seeking once it has a road to every
  tree and goes on walking the roads of a map it never finished (its long trips are straight
  0.87 of the time); the brain goes on finding ways.
- The lineages keep their order: lineage 3 writes worst by every measure (0.80 gone once, a
  first third of 32 and 17) and gains least early; lineage 4, whose last third was the worst,
  gains most (54 to 95, 57 to 122) and still walks its long trips least straight (0.47).
- Fewer false roads, fewer walls tried: by the end of a life in a land half moved the new
  genomes have gone 0.56-0.74 of the walls that opened, the lineage genomes 0.68-0.85. A brain
  that knows its land better explores less by mistake. It needs the wish to know, and in a land
  that changes it needs doubt, as a table does.
- For the forgetting road: the line between known and not known is drawn after one walk (a copy
  says a way never gone at 3.1-4.1 and one gone once at 5.5-5.9; the lineage genomes 2.8-3.6
  and 3.7-4.3). But a way that closed and has not been gone since is said as it was, at 6.3-6.8,
  as loud as any: nothing in the brain marks how old a wire is.
- **A caution on the cause.** The walk arm differs from the lineage's searches in more than
  the one lap: it is the first search that lived with the hand-made dendrites, in a world of
  forks, at loads up to two links per interneuron. Holding with TWO laps rose as well, and four
  hops deep most of all, which a link shown once does not explain. How much of the gain is the
  one lap is asked by a control that differs in the laps alone (step 6j). Not asked yet either:
  whether the new genomes still live the stream and hold a thought as the lineage did (exam 2).

### Step 6j: what in the walk arm did it, and the new genomes in the old lands. Predictions before the runs

Nothing new in the code. Three questions asked of the genomes of 6i, on the VM (`out/jobs9a.txt`,
`out/jobs9b.txt`, logs `out/jobs/j6-*`; started 2026-11-19).

**The control** (`runs/tissue/walk/twice-s0..4.pt`): the same search, seed for seed, with two or
three laps a life and never one (`--laps 2 3`), then the same chain (exam 4 with one lap and
with two, the look `tissue.unlearn`, a life at side 8 with seeking).

- **J1**: with two laps, K90 at hop 1 and at hop 4 (256 cells) is at least 0.9 of the walk
  genomes' on four lineages. The gain with two laps came from living with the dendrites.
- **J2**: a way gone once since birth is said right on less than 0.80 of the askings on four
  lineages (walk: 0.92 / 0.92 / 0.93 / 0.80 / 0.91; lineage: 0.54 / 0.53 / 0.61 / 0.37 / 0.51).
  The one lap is what makes one walk write.
- **J3**: at side 8 the first third is smaller than the walk genomes' (64.8 / 62.7 / 66.2 / 31.6
  / 63.9) on four lineages, and the life as well.

If J2 and J3 are missed, the seams of the stream are not what 6i mended: any search that lives
with the dendrites would have done.

**The old exam** (exam 2, the stream: `tissue.exam --stream --idle 64 --lives 16`, 64 and 256
cells; the lineage genomes there: K90 at hop 1 40-42 and 148-209, at hop 4 18-24 and 39-81,
items held in thought through 64 dark ticks 13-24 and 63-91). Written after one look (lineage
0, 64 cells, four lives: hop 1 nearly kept, hop 4 not, twice as many items held), so not blind:

- **J4**: K90 at hop 1 at least 0.8 of the lineage's at both sizes on four lineages, and K90 at
  hop 4 under half of it on four. A rule that believes a pair seen once believes the seams
  between loops: the price of 6i is paid in the world the lineage came from, not in a land,
  where nothing is a coincidence.
- **J5**: items held in thought no fewer than the lineage's, on four.

**The new genomes in the old lands** (16 lives, a road kept, 4096 rooms, the exam's lands):

- **J6, no seeking** (side 8, 256 cells; 6e: 241.4 / 211.9 / 214.4 / 193.4 / 160.2): more than
  6e's on four lineages; seeking worth more to a brain that writes at once, with over without
  at least 1.15 on four (lineage genomes: 1.09-1.16); and the first third without seeking no
  more than 1.5 times 6e's (23.7 / 18.2 / 19.8 / 15.2 / 19.4): the rule alone does not find
  the ways.
- **J7, the maze** (walls 0.5, 256 cells, seeking; 6h: 164.3 / 140.6 / 145.4 / 125.2 / 81.6;
  `explorer` 228): at least 1.2 times 6h's on four lineages, at least 182 (0.8 of the
  explorer's) on three, and no life with 5 meals or fewer in its last third.
- **J8, a land half moved at half-life** (shift 0.5, side 8, 256 cells; `explorer` 212.8 with a
  last third of 60, `doubter` at T 640 275.3 and 105). Three arms: the new genomes without
  seeking (B7: 122.4 / 114.3 / 126.1 / 89.3 / 80.8; the look of 6i gave 169 / 193 / 177 / 132 /
  182 on eight of these lives, so not blind), the new genomes with seeking, and the lineage
  genomes with seeking for the pair. Without seeking: at least 1.3 times B7's on four. With
  seeking: a life of at least 200 on three, a last third of at least 60 on four and under 95
  on all five (one walk undoes a way, but nothing sends the body to look at what it holds for
  known); the new genomes over the lineage's, both seeking, at least 1.3 on four.
- **J9, the knee again** (seeking): side 8 at 128 and 512 cells, side 12 at 256 and 1024. Side
  8: 256 over 128 cells at least x1.3 on four; 512 over 256 at most x1.10 on lineages 0-2 (they
  are at the explorer's) and more than that on 3 and 4 (the weaker the lineage, the more a
  doubling pays, as before). Side 12: 512 over 256 at least x1.4 on four; 1024 over 512
  x1.10-1.25 on four, with a first third of at least 85 on three (60-70 now; `explorer` 129)
  and a life of at least 465 (0.9 of the explorer's) on three.

**A look beside it, no bar**: the pilot of "Forgetting, step 4" again with the new genome
(`walk-s0`, the same land and life, fade 0 / 0.001 / 0.003). A guess: a wire that fades still
makes the brain wrong before it makes it quiet (nothing marks how old a wire is), and costs
fewer meals than it cost the lineage genome, because a way said wrongly is written again in
one walk.

**Outcome (2026-11-19, evening; records `runs/tissue/walk/twice-s*`, `walk-s*`, and
`hold-s*.world8.shift0.5.*explore0.4*`; everything ran on the VM, eleven hours).**

*The control.* Lineage genome / walk genome / control, 256 cells:

| | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| exam 4, two laps, K90 at hop 1 | 857 / 951 / 903 | 926 / 1039 / 972 | 935 / 1026 / 908 | 647 / 829 / 842 | 891 / 1029 / 943 |
| two laps, K90 at hop 4 | 417 / 558 / 514 | 361 / 566 / 570 | 439 / 600 / 548 | 202 / 341 / 412 | 338 / 577 / 558 |
| one lap, K90 at hop 1 | 573 / 665 / 534 | 520 / 703 / 595 | 584 / 719 / 544 | 301 / 444 / 465 | 473 / 698 / 575 |
| in a land: a way gone once, said right | 0.54 / 0.92 / 0.92 | 0.53 / 0.92 / 0.94 | 0.61 / 0.93 / 0.91 | 0.37 / 0.80 / 0.90 | 0.51 / 0.91 / 0.90 |
| a way that closed, gone once since | 0.23 / 0.87 / 0.96 | 0.00 / 0.88 / 0.99 | 0.01 / 0.88 / 0.95 | 0.43 / 0.64 / 0.97 | 0.00 / 0.85 / 0.99 |
| a wall that opened, gone once | 0.00 / 0.78 / 0.95 | 0.00 / 0.80 / 0.98 | 0.01 / 0.78 / 0.96 | 0.02 / 0.47 / 0.97 | 0.00 / 0.60 / 0.92 |
| side 8, seeking: meals a life | 263.9 / 310.9 / 304.8 | 236.1 / 294.8 / 309.0 | 245.4 / 302.1 / 306.9 | 223.5 / 264.1 / 292.3 | 177.6 / 272.3 / 309.9 |
| the first third | 40.6 / 64.8 / 59.8 | 36.6 / 62.7 / 61.1 | 42.6 / 66.2 / 59.5 | 21.6 / 31.6 / 45.5 | 35.1 / 63.9 / 62.2 |
| the last third (born knowing: 127) | 112.6 / 125.5 / 124.7 | 103.8 / 119.3 / 125.2 | 100.1 / 119.2 / 125.3 | 110.2 / 120.1 / 125.3 | 54.2 / 95.4 / 125.4 |
| trips of 9 steps and more walked straight, the last third | 0.66 / 0.95 / 0.92 | 0.59 / 0.79 / 0.97 | 0.49 / 0.81 / 0.95 | 0.67 / 0.88 / 0.98 | 0.15 / 0.47 / 0.96 |
| the control less the walk genome, a life, and its error | -6 +- 1 | +14 +- 2 | +5 +- 3 | +28 +- 3 | +38 +- 4 |

- **J1: met.** With two laps the control holds 0.89-1.02 of the walk genomes' at hop 1 (0.9 on
  four) and 0.92-1.21 at hop 4.
- **J2: missed on all five.** Never shown a link once, the control says a way gone once right
  on 0.90-0.94 of the askings, and undoes one better than the walk genomes do (0.95-0.99).
- **J3: missed.** The first third is smaller on four lineages, by 2-7 meals (lineage 3: larger
  by 14), but the life is LARGER on four.
- **So the seams of the stream are not what 6i mended.** Any search that lives with the
  dendrites in a world of forks writes a way in one walk, and the one that is never shown a
  link once does the rest better: every lineage ends its life at 125 meals a third, where one
  born knowing the land eats 127, long trips are straight 0.92-0.98 of the time, and the five
  lineages no longer differ (292-310 meals; 264-311 with the walk genomes, 178-264 before).
  What the one lap in the search buys is holding at heavy loads shown once (exam 4, one lap:
  the control holds 0.76-0.85 of the walk genomes' on four lineages) and 2-7 meals early; what
  it costs is depth.
- What the lineage genomes lacked was a search with the body they were given: the dendrite per
  way was set on them by hand (4a) and they never lived with it. METHOD: a part added by hand
  changes what the rule should be; search again with the part in place before judging either.
  **The control genomes (`twice-s0..4`) are the best in a land and are carried on.**

*The old exam, the stream.* Lineage genome / walk genome:

| | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 256 cells: K90 at hop 1 | 194 / 174 | 209 / 188 | 208 / 187 | 148 / 167 | 199 / 190 |
| K90 at hop 4 | 81 / 0 | 70 / 0 | 78 / 0 | 39 / 0 | 64 / 0 |
| items held in thought through 64 dark ticks | 82 / 274 | 71 / 217 | 91 / 210 | 63 / 151 | 82 / 209 |
| 64 cells: K90 at hop 1; items held | 40 / 37; 24 / 55 | 40 / 37; 17 / 46 | 42 / 36; 24 / 50 | 40 / 41; 13 / 29 | 42 / 40; 23 / 48 |

- **J4: met by the letter** (hop 1 at 0.87-1.13 of the lineage's; at hop 4 nine bits of ten are
  reached at no load), **J5: met** (x2.1-3.4 items held). **The reason I gave is not shown.**
  The walk in the dark loses about six bits in a hundred at every hop even with 8 items in 256
  cells (0.93 / 0.87 / 0.83 / 0.79 of the bits right at hops 1-4; the lineage genome 1.00), so
  it is not seams crowding the wires. The stream has no body, and the new genomes walk with
  the dendrite of the way. The control genomes sit the same exam next.

*The new genomes (walk) in the old lands*, a road kept, 4096 rooms, 16 lives:

| | lineage 0 | 1 | 2 | 3 | 4 | `explorer` | born knowing |
|---|---|---|---|---|---|---|---|
| side 8, NO seeking: the lineage genome (6e) | 241.4 | 211.9 | 214.4 | 193.4 | 160.2 | 312 | 380 |
| the walk genome | 278.6 | 275.3 | 270.5 | 241.7 | 258.4 | | |
| with seeking over without | 1.12 | 1.07 | 1.12 | 1.09 | 1.05 | | |
| the first third without seeking (6e) | 33.9 (23.7) | 44.9 (18.2) | 39.1 (19.8) | 22.1 (15.2) | 43.8 (19.4) | 83 | 127 |
| the maze, seeking: the lineage genome (6h) | 164.3 | 140.6 | 145.4 | 125.2 | 81.6 | 228 | 257 |
| the walk genome | 204.6 | 187.0 | 189.8 | 160.3 | 152.4 | | |
| over 6h | 1.25 | 1.33 | 1.31 | 1.28 | 1.87 | | |
| side 8, cells 128 / 256 / 512, seeking | 255 / 311 / 323 | 213 / 295 / 315 | 242 / 302 / 318 | 199 / 264 / 285 | 214 / 272 / 301 | 312 | 380 |
| a doubling: 256 over 128, 512 over 256 | 1.22, 1.04 | 1.39, 1.07 | 1.25, 1.05 | 1.33, 1.08 | 1.27, 1.10 | | |
| side 12, cells 256 / 512 / 1024, seeking | 306 / 441 / 487 | 188 / 406 / 458 | 282 / 428 / 469 | 140 / 333 / 388 | 230 / 356 / 411 | 517 | 601 |
| a doubling: 512 over 256, 1024 over 512 | 1.44, 1.10 | 2.16, 1.13 | 1.52, 1.10 | 2.38, 1.16 | 1.55, 1.15 | | |
| side 12, 1024 cells: the first third | 90.9 | 78.9 | 92.5 | 21.8 | 84.3 | 129 | 201 |

| a land half moved at half-life (shift 0.5, side 8, 256 cells) | lineage 0 | 1 | 2 | 3 | 4 | `explorer` | `doubter`, T 640 |
|---|---|---|---|---|---|---|---|
| the lineage genome, no seeking (B7) / seeking | 122.4 / 160.0 | 114.2 / 134.2 | 126.1 / 158.2 | 89.3 / 116.1 | 80.8 / 110.7 | 212.8 | 275.3 |
| the walk genome, no seeking / seeking | 175.4 / 195.9 | 185.9 / 200.2 | 161.0 / 204.2 | 118.2 / 158.2 | 179.2 / 197.1 | | |
| the last third: walk genome, no seeking / seeking | 73.9 / 59.8 | 72.5 / 68.9 | 60.1 / 66.6 | 42.9 / 63.6 | 69.2 / 65.2 | 60.3 | 104.8 |
| meals in the half-life after the change, the same | 84.7 / 70.8 | 85.1 / 80.7 | 65.9 / 79.6 | 53.1 / 73.9 | 77.1 / 75.1 | 71.6 | 134.8 |
| lives with no meal in the last third, the same | 2 / 2 | 1 / 1 | 1 / 0 | 6 / 0 | 0 / 2 | 1 | 0 |

- **J6: met in its first part, wrong in the rest.** Without seeking the new genomes eat
  x1.15-1.61 of 6e's. But seeking is worth LESS to them, x1.05-1.12 (I said 1.15 and more),
  and their first third without it is 1.4-2.5 times 6e's (I said no more than 1.5). When one
  walk writes a way, a walk by the dice is not wasted: the rule does most of the finding.
- **J7: met**: the maze x1.25-1.87 of 6h's, three lineages at 0.82-0.90 of the explorer, no
  life starved.
- **J8: met in what it says, two figures missed by a hair.** Without seeking x1.28-2.22 of B7's
  (1.3 on four). With seeking a life of 196 / 200 / 204 / 158 / 197 (200 asked on three: two),
  a last third of 60-69 (at least 60 on four, under 95 on all five: met), and x1.22-1.78 of the
  lineage genomes' with seeking (1.3 on three, four asked). The curve of a life: 20-23 meals
  per 160 ticks before the change, 2.4-4.8 right after it, 13-18 by the end, which is the
  explorer's curve (2.9, then 15.6); the `doubter` is back at 18-20 within 480 ticks. **In a
  land that changes the brain is now a table that never doubts**, 0.74-0.96 of it, and seeking
  adds nothing after the change (it seeks what was never gone, not what has changed).
- **J9: the shape met, three figures missed.** Side 8: 256 over 128 cells x1.22-1.39 (1.3 asked
  on four: two), 512 over 256 x1.04-1.10 with the weaker lineages gaining more (1.08, 1.10:
  not "more than 1.10"). At 512 cells three lineages eat more than the explorer (323 / 315 /
  318 against 312; level within the error). Side 12: 512 over 256 x1.44-2.38 (met), 1024 over
  512 x1.10-1.16 (met); at 1024 cells a first third of 85 and a life of 465 on two lineages,
  not three (84.3 and 457.6 on the third). The knee stands where it stood: a doubling pays
  x1.2-2.4 while cells are fewer than the land's ways and x1.04-1.16 after.

*The two looks.* Fading with the new genome (`walk-s0`, the land and life of "Forgetting, step
4"; right / loud by the ticks since the way was last gone; the brain says its own room at 7.9):

| fade a tick | meals (by third) | 0-80 ticks | 160-320 | 320-640 | 640-1280 | never gone: loud |
|---|---|---|---|---|---|---|
| 0 | 265 (55 / 107 / 103) | 1.00 / 6.8 | 0.98 / 6.5 | 0.96 / 6.2 | 0.95 / 6.2 | 3.4 |
| 0.001 | 174 (39 / 73 / 62) | 1.00 / 6.1 | 0.96 / 5.4 | 0.86 / 4.8 | 0.50 / 4.1 | 3.1 |
| 0.003 | 50 (17 / 15 / 18) | 1.00 / 5.3 | 0.72 / 3.7 | 0.18 / 3.0 | 0.04 / 3.0 | 2.8 |

- The guess was half right. Fading costs the new genome as much as it cost the old (a third of
  the meals at 0.001), and at the line of 0.4 a faded way is wrong long before it is quiet. But
  with this genome loudness falls step by step with the age of a wire: a way is at 0.73 of the
  root's loudness while it is still right on 0.96 of the askings, at 0.66 when on 0.86. The
  line of 0.4 lies under even what was never gone. **A line at 0.6-0.7 would call a fading way
  not known while it is still mostly right**: quiet before wrong, by where the line is put.
- In thought (`walk-s0`, `walk-s4`, two lives each, copies followed along the true land): a
  copy says a way gone at 0.72-0.82 of the root's loudness and a way never gone at 0.25-0.54.
  Under 0.4 fall 0.41-0.51 of the ways never gone (lineage genomes: 0.67-0.86) and 0.01 of
  those gone; under 0.6, 0.93-0.96 and 0.09-0.11; under 0.7, 0.98-1.00 and 0.28-0.32. **For the
  new genomes Q 0.4 misses half of what they do not know; the line is 0.6.**

**What it says.** The gain of 6i came from searching the rule with the body it lives with,
not from the link shown once; my account of the cause was wrong and the control shows it. With
those genomes a land that does not change is nearly done at this size: the brain eats 0.94-0.99
of the explorer at side 8 (above it with 512 cells), its last third is at born-knowing, growth
pays to the knee as before. What is left there is the first third and the bigger land. THE GAP
THAT IS LEFT IS A LAND THAT CHANGES: after the change the brain eats 71-81 meals where a memory
that doubts eats 135. One walk undoes a way now, so the rule is no longer what is slow. Nothing
sends the body to look.

### Step 6k: the control genomes in the old lands. Predictions before the runs

The control genomes of 6j (`twice-s0..4`) are the ones carried on, and all that is known of them
is exam 4, the look `tissue.unlearn` and a life at side 8. The same records as the walk genomes
have (16 lives, a road kept, 4096 rooms, seeking at 0.4 where there is seeking; on the VM,
`out/jobs10b.txt`, logs `out/jobs/k6-*`):

- **K1, the stream** (64 and 256 cells): the control genomes lose the walk in the dark as the
  walk genomes did (K90 at hop 4 under half of the lineage's on four lineages) and hold at
  least twice the lineage's items in thought. It is the dendrite, not the one lap.
- **K2, side 8 with no seeking**: no less than the walk genomes' (278.6 / 275.3 / 270.5 / 241.7
  / 258.4) on four lineages; seeking at 0.4 worth under x1.10 on four (the line of 0.4 catches
  a fifth of what these genomes do not know, see the look below).
- **K3, the maze**: more than the walk genomes' (204.6 / 187.0 / 189.8 / 160.3 / 152.4) on
  four; three lineages at 0.85 of the explorer's 228 (194) or more.
- **K4, shift 0.5 with seeking and without**: with seeking within a tenth of the walk genomes'
  (195.9 / 200.2 / 204.2 / 158.2 / 197.1) on four, a last third under 95 on all five. Undoing a
  way in one walk is not what a changing land lacks.
- **K5, side 12, 512 cells**: more than the walk genomes' (441.2 / 406.2 / 428.2 / 333.1 /
  356.1) on four, lineages 3 and 4 by the most.
- **K6, the knee** (added the same evening, before any of it ran; seeking at 0.4; logs
  `out/jobs/k6-m128-*`, `k6-m512-*`, `k6-s12m256-*`, `k6-s12m1024-*`): side 8 at 128 and 512
  cells, side 12 at 256 and 1024. Side 8: 256 over 128 cells x1.2-1.4 on four lineages, 512
  over 256 at most x1.06 on four (the last third is at born-knowing already; only the first
  third can grow), three lineages at 512 cells at the explorer's 312 or above. Side 12: 512
  over 256 at least x1.4 on four, 1024 over 512 x1.08-1.20 on four; at 1024 cells a life of at
  least 465 (0.9 of the explorer's) on three and a last third at the explorer's 193 or above
  on four.

**Outcome (2026-11-20; `twice-s*` records in `runs/tissue/walk/`; walk genome / control).**

| | lineage 0 | 1 | 2 | 3 | 4 | `explorer` | born knowing |
|---|---|---|---|---|---|---|---|
| the stream, 256 cells, K90 at hop 1: lineage / walk / control | 194 / 174 / 158 | 209 / 188 / 170 | 208 / 187 / 159 | 148 / 167 / 148 | 199 / 190 / 159 | | |
| K90 at hop 4 | 81 / 0 / 0 | 70 / 0 / 0 | 78 / 0 / 0 | 39 / 0 / 0 | 64 / 0 / 0 | | |
| items held in thought through 64 dark ticks | 82 / 274 / 264 | 71 / 217 / 277 | 91 / 210 / 277 | 63 / 151 / 162 | 82 / 209 / 262 | | |
| side 8, no seeking | 278.6 / 272.9 | 275.3 / 286.2 | 270.5 / 292.1 | 241.7 / 262.9 | 258.4 / 274.6 | 312 | 380 |
| the control: with seeking over without | 1.12 | 1.08 | 1.05 | 1.11 | 1.13 | | |
| the maze, seeking | 204.6 / 190.0 | 187.0 / 194.7 | 189.8 / 194.4 | 160.3 / 179.8 | 152.4 / 189.7 | 228 | 257 |
| shift 0.5, seeking | 195.9 / 192.1 | 200.2 / 206.4 | 204.2 / 199.4 | 158.2 / 184.2 | 197.1 / 214.7 | 213 | |
| side 12, 512 cells, seeking | 441.2 / 423.2 | 406.2 / 434.2 | 428.2 / 432.0 | 333.1 / 391.9 | 356.1 / 442.8 | 517 | 601 |
| the control, side 8, cells 128 / 256 / 512 | 252 / 305 / 315 | 270 / 309 / 321 | 265 / 307 / 318 | 236 / 292 / 304 | 266 / 310 / 319 | 312 | 380 |
| a doubling | 1.21, 1.03 | 1.14, 1.04 | 1.16, 1.04 | 1.24, 1.04 | 1.16, 1.03 | | |
| the control, side 12, cells 256 / 512 / 1024 | 297 / 423 / 464 | 330 / 434 / 474 | 320 / 432 / 472 | 226 / 392 / 430 | 341 / 443 / 479 | 517 | 601 |
| a doubling | 1.43, 1.10 | 1.31, 1.09 | 1.35, 1.09 | 1.73, 1.10 | 1.30, 1.08 | | |
| side 12, 1024 cells, by third | 72 / 196 / 196 | 82 / 197 / 195 | 76 / 199 / 198 | 40 / 192 / 197 | 82 / 199 / 198 | 129 / 195 / 193 | 201 / 201 / 199 |

- **K1: met.** The control genomes lose the walk in the dark of the stream just as the walk
  genomes did (0.93 / 0.88 / 0.83 / 0.79 of the bits right at hops 1-4 with 8 items in 256
  cells) and hold x2.0-3.9 of the lineage's items in thought. It is the dendrite, not the one
  lap: a brain searched with a body walks with its body.
- **K2: met in its first part, missed in its second.** Without seeking the control eats no
  less than the walk genomes on four lineages; seeking at 0.4 is worth x1.05-1.13 to it,
  under 1.10 on two lineages and not four. The line that catches a fifth of what is not known
  is enough to lead the body there.
- **K3: met in its first part; the second by a hair.** The maze: more than the walk genomes on
  four; two lineages at 194 and more, two at 190.
- **K4: met.** In a land half moved the control eats within a tenth of the walk genomes on
  four lineages (lineage 3: x1.16), a last third of 61-79.
- **K5: met.** Side 12 with 512 cells: more on four, lineages 3 and 4 by 59 and 87 meals.
- **K6: the shape met, two figures missed.** Side 8: 256 over 128 cells x1.14-1.24 (x1.2-1.4
  said: two lineages), 512 over 256 x1.03-1.04, and four lineages at 512 cells above the
  explorer's 312. Side 12: 512 over 256 x1.30-1.73 (1.4 said on four: two), 1024 over 512
  x1.08-1.10; at 1024 cells three lineages at 465 or more and every last third at 195-198
  (the explorer 193, born knowing 199). I had the small brains too small: 128 cells of the
  control at side 8 eat 236-270, more than 256 cells of the lineage genomes did.
- With the control genomes the lineages are alike in every land asked (the maze 180-195, side
  12 with 512 cells 392-443, with 1024 cells 430-479). Lineage 0 is the one that was better
  served by the search with one lap, by 6 to 23 meals in every land. In a big land all that
  is left of the gap to born-knowing is the first third: 40-82 meals against the explorer's
  129, with the other two thirds at 192-199 of 200.

### Forgetting, step 5: doubt by hand. A look, a pilot, then bars

What a land that changes asks, by the yardsticks: no longer to hold for known a way not gone
for about a quarter of a life, and to go and look (`doubter`, T 640: 135 meals in the half-life
after the change where the explorer and the brain eat 71-81). Three hand-made parts, each with
its off-switch, none new in kind:

1. **Wires that fade** (`--fade F`, step 4): every plastic wire loses the share F of itself on
   a tick the brain sees. Alone it made the brain wrong before quiet. With the new genomes
   loudness falls step by step with the age of a wire (the look of 6j), so age can be read.
2. **The line** (`--explore Q`): what a copy says under the share Q of the brain's own loudness
   is a way not known, to be gone to when no road is found. At 0.4 the line lies under what
   was never gone.
3. **`--sure`** (new, `Mind.sure`): what a copy says quietly, under that same line, is not
   believed: it is no road to the fruit and no copy goes on from it. Without it a thought walks
   on through ways it would call not known, and a faded way that is wrong makes a false road.

Together: a way not gone for long fades, falls under the line, is left out of every road, and
is sought again when the roads through what is sure do not reach the fruit. That is the
doubter, in the brain's own terms; its span is set by F and Q together.

**The look that sets the line** (control genomes `twice-s0`, `twice-s4`, two lives each, copies
followed along the true land; and one life of `twice-s0` under fading, the examiner asking one
hop on). The brain says its own room at 8.7-9.7. In thought a copy says a way gone at 0.76-0.81
of that and a way never gone at 0.42-0.60:

| under the share q of the root's loudness | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|
| of the ways never gone | 0.22 | 0.56-0.60 | 0.80-0.82 | 0.92-0.93 | 0.98-1.00 |
| of the ways gone | 0.00 | 0.01-0.02 | 0.05-0.08 | 0.19-0.21 | 0.47-0.50 |

A way gone once is said at 0.74-0.77 of the root's loudness (right on 0.88-0.92), gone twice at
0.78-0.81, nine times and more at 0.88-0.90. Under fading of 0.0003 a tick (one life; no
fading in brackets): gone within 80 ticks 0.85 of the root's loudness and right on 1.00 (0.88,
1.00); 320-640 ticks ago 0.77, 0.96 (0.81, 0.96); 640-1280 ago 0.73, 0.95 (0.81, 0.96); longer
ago 0.68, 0.87 (0.80, 0.96); never gone 0.46. The life: 231 meals against 265 (thirds 47 / 103
/ 81 against 58 / 105 / 102), with the line at 0.4 and nothing left out of a thought.

So for these genomes the line that tells known from not known is 0.6-0.7, and at 0.7 a way that
has faded for a thousand ticks at 0.0003 falls under it while it is still right nine times in
ten: quiet before wrong. The price is at the same place: at 0.7 a fifth of the ways gone are
called not known as well, most of them gone once.

**The pilot** (other lands than the exam's, `--seed 1`, 8 lives, `twice-s0` and `twice-s4`, 256
cells, side 8, a road kept, 4096 rooms; the land that does not change and shift 0.5; on the VM,
`out/jobs10a.txt`, logs `out/jobs/k5-*`). Eleven settings: the line at 0.4 (the base), 0.6, 0.7;
0.6 and 0.7 with `--sure`; those two with fading at 0.0003; 0.7 with `--sure` and fading at
0.0005 and at 0.001; and fading at 0.0003 with the line at 0.6 and 0.7 and nothing left out.
The yardsticks on the same lands, with the doubter at T 640. No bars on a pilot. Guesses,
written before it: where nothing changes, the line alone adds a little to the first third,
`--sure` alone costs under 3% of a life and with fading at 0.0003 under 5%; in the land half
moved, the three together lift the half-life after the change by a third or more over the
base, and no two of them do. Fading at 0.001 is too fast (the yardstick's T 320).

**The exam, set before the pilot is read** (the user asked for the VM to be given a night's
work at once, 2026-11-19, so the choice the pilot was to make is written as a rule and the exam
is run for every setting the rule can pick). The exam's lands, 16 lives, `twice-s0..4`, 256
cells, side 8, a road kept, 4096 rooms (on the VM, `out/jobs11.txt`, logs `out/jobs/d5-*`):

- the land that does not change and shift 0.5: the line at 0.6 and at 0.7; each with `--sure`;
  the three together at (0.6, 0.0003), (0.7, 0.0003), (0.7, 0.0005); and, at shift 0.5 only,
  those three with `--sure` off. The base (the line at 0.4, nothing else) is step 6k's.
- shift 0.25: the base and the three settings with all three parts.

**The rule that picks.** Of the three settings with all three parts, the one with the most
meals in the half-life after the change in the PILOT (the mean of the two genomes), among those
whose life in the pilot's land that does not change is at least 0.95 of the base's there; if
none is, the one with the largest sum of the two lives. Fading at 0.001 is in the pilot to show
the fast end and cannot be picked. The bars are for the setting picked; the others are
reported as the dial.

- **D1, a land half moved**: meals in the half-life after the change at least 1.3 times the
  base's on four lineages, and at least 100 on three (the `doubter` 135, the `explorer` 72).
- **D2, where nothing changes**: a life at least 0.95 of the base's (304.8 / 309.0 / 306.9 /
  292.3 / 309.9) on four lineages.
- **D3, no lock**: at most 2 lives of 80 with no meal in the last third at shift 0.5.
- **D4, every part is needed**: with the fading off (the same line, `--sure`) and with `--sure`
  off (the same line and fading) the half-life after the change is smaller than with all
  three, each on four lineages.
- **D5, a quarter moved** (shift 0.25; `explorer` 256, `doubter` 280): a life no smaller than
  the base's on four lineages.

Prediction: D1 met in its first half (the three together do lift the half-life after the
change by a third) and missed in its second (100 meals is the doubter's country); D3 and D5
met; D2 is the doubt (one life under fading at 0.0003 lost an eighth of its meals, with
nothing left out of its thoughts); D4 met for the fading, open for `--sure`.

**The pilot (2026-11-20; `twice-s0`, `twice-s4`, lands of `--seed 1`, 8 lives; meals).**

| setting | the land still: a life | shift 0.5: a life | the half-life after the change |
|---|---|---|---|
| the line at 0.4 (the base) | 294 / 301 | 198 / 200 | 84 / 81 |
| the line at 0.6 | 297 / 302 | 187 / 195 | 72 / 74 |
| the line at 0.7 | 295 / 303 | 201 / 207 | 87 / 87 |
| 0.6, `--sure` | 297 / 300 | 208 / 205 | 87 / 82 |
| 0.7, `--sure` | 271 / 295 | 185 / 173 | 79 / 56 |
| 0.6, `--sure`, fading 0.0003 | 278 / 302 | 227 / 226 | 118 / 102 |
| 0.7, `--sure`, fading 0.0003 | 262 / 282 | 212 / 242 | 115 / 126 |
| 0.7, `--sure`, fading 0.0005 | 243 / 268 | 185 / 209 | 107 / 108 |
| 0.7, `--sure`, fading 0.001 | 113 / 156 | 66 / 130 | 32 / 69 |
| 0.6, fading 0.0003, nothing left out | 263 / 273 | 214 / 200 | 108 / 86 |
| 0.7, fading 0.0003, nothing left out | 263 / 272 | 216 / 214 | 111 / 101 |
| `explorer` / `doubter`, T 640 | 302 / 300 | 203 / 274 | 66 / 137 |

**The rule picks the line at 0.6 with `--sure` and fading at 0.0003**: its life where nothing
changes is 0.974 of the base's (the other two: 0.913 and 0.858, not let in) and its half-life
after the change 110 meals against the base's 82. The pick was made before any record of the
exam was opened. The guesses: the line alone does nothing (right); `--sure` alone costs nothing
at 0.6 and 2-8% at 0.7; the three together lift the half-life after the change by x1.30-1.46
(right); "no two of them do" was wrong in part: fading with the line and nothing left out
lifts it x1.18-1.29, but costs a tenth of a life where nothing changes, and `--sure` takes
most of that cost away. Fading at 0.001 is far too fast (right).

**The exam (2026-11-20; the exam's lands, 16 lives, `twice-s0..4`; base / the setting picked).**

| | lineage 0 | 1 | 2 | 3 | 4 | `explorer` | `doubter`, T 640 |
|---|---|---|---|---|---|---|---|
| shift 0.5, meals in the half-life after the change | 72.2 / 110.9 | 84.9 / 120.9 | 78.6 / 116.5 | 79.2 / 104.4 | 90.8 / 111.0 | 71.6 | 134.8 |
| over the base | 1.54 | 1.42 | 1.48 | 1.32 | 1.22 | | |
| the difference a life and its error; lives of 16 that eat more | +39 +- 9; 15 | +36 +- 8; 15 | +38 +- 9; 15 | +25 +- 9; 10 | +20 +- 8; 12 | | |
| the same half-life with the fading off / with `--sure` off | 85.2 / 89.4 | 88.0 / 98.5 | 85.1 / 90.8 | 95.0 / 86.5 | 81.9 / 96.1 | | |
| shift 0.5, a life | 192.1 / 224.7 | 206.4 / 240.1 | 199.4 / 234.6 | 184.2 / 205.3 | 214.7 / 232.7 | 212.8 | 275.3 |
| the last third | 61 / 93 | 72 / 100 | 67 / 99 | 64 / 92 | 79 / 95 | 60 | 105 |
| lives with no meal in the last third | 2 / 0 | 2 / 0 | 1 / 0 | 2 / 0 | 0 / 0 | 1 | 0 |
| the land still, a life | 304.8 / 288.0 | 309.0 / 291.2 | 306.9 / 293.1 | 292.3 / 275.5 | 309.9 / 293.7 | 311.8 | 307.9 |
| over the base | 0.945 | 0.942 | 0.955 | 0.942 | 0.948 | | 0.99 |
| the land still, by third, the setting picked | 56 / 116 / 116 | 62 / 115 / 114 | 61 / 116 / 116 | 43 / 116 / 117 | 64 / 115 / 115 | 83 / 114 / 115 | 83 / 113 / 112 |
| shift 0.25, a life | 255.4 / 249.3 | 251.3 / 255.4 | 256.6 / 254.6 | 240.5 / 230.8 | 248.9 / 256.7 | 256.4 | 280.2 |
| shift 0.25, the half-life after the change | 135.4 / 135.5 | 129.8 / 136.2 | 135.9 / 136.4 | 135.5 / 129.9 | 125.1 / 135.0 | 115.2 | 139.8 |

Meals per 160 ticks from the change on, the mean of the five lineages (21-23 before it): the
base 3.7, 4.8, 6.6, 9.2, 11.2, 13.7, 15.7, 16.3; the setting picked 4.1, 6.6, 10.5, 15.5,
17.6, 19.6, 18.9, 19.9; the `doubter` 6.2, 11.9, 17.9, 20.4, 19.6, 18.7, 19.9, 20.1.

- **D1: met, in both halves** (I had the second missed): x1.22-1.54 over the base, 1.3 on
  four, and 104-121 meals on all five, 0.77-0.90 of what the table that doubts eats.
- **D2: missed by a hair.** Where nothing changes a life is 0.942-0.955 of the base's, 0.95 on
  one lineage. Doubt costs the brain 5-6%, the table 1%: the last third falls from 125 to
  115, which is the explorer's.
- **D3: met.** No life of the 80 ends without a meal (the base: 7).
- **D4: met for both parts, on all five lineages.** Alone, fading lifts the half-life after
  the change x1.06-1.24 and `--sure` x0.90-1.20; together x1.22-1.54.
- **D5: missed by the letter, level in fact**: with a quarter of the walls moved two lineages
  of five eat no less with doubt, and no difference stands clear of its error (-10 to +8, +- 6
  to 11). There the brain with no doubt already eats 125-136 in the half-life after the
  change, above the explorer's 115 and near the doubter's 140: what its own slips find is
  most of what moved.
- The dial, as said it would be reported: the line at 0.7 with fading costs 9-22% where
  nothing changes (at 0.0005: 14-38%) and lineage 3 falls apart after a change (125 and 89
  meals a life): it calls a fifth of what is known not known. The line alone, 0.6 or 0.7,
  changes nothing in either land (0.99-1.01). `--sure` alone costs 0-3% at 0.6 and 3-8% at
  0.7. The pilot's pick is the best of the three on the exam's lands as well.

**What it says.** Doubt by hand works. A brain whose wires fade slowly, which holds for not
known what it says quietly and leaves it out of its thoughts, comes back after half its land
has moved to the rate of a table that doubts (19-20 meals per 160 ticks, 22 before the change)
in 800 ticks, where the table takes 480 and the brain without doubt never does (16 at the end
of its life). Each of the three parts is needed, and no life locks. It costs 5-6% of a life
where nothing changes. All three parts are by hand; by the lesson of 6j the rule should now
be searched with them in place.

### Step 6l: more search with the body. Predictions before the runs

Step 6j says the rule wants to be searched with the body it lives with, and 4000 steps of
that did more than anything since the road kept. Two questions, the same arm as the control
(`tissue.train --arm walk --laps 2 3 --hops 16 --vary`), each followed by exam 4, the look
`tissue.unlearn`, a life at side 8 and one at side 12 with 512 cells (seeking at 0.4, a road
kept; on the VM, logs `out/jobs/l6-*`):

- **`long-s0..4`**: 8000 steps more, from each control genome. Does more of it still pay?
  Prediction: exam 4, two laps, 256 cells, K90 four hops deep at least 1.1 times the control's
  on three lineages; at side 8 nothing a life can show (within 2% of the control's on four:
  it ends its life at born-knowing already); at side 12 with 512 cells, below the knee, at
  least 3% more on three.
- **`fresh-s0..4`**: 8000 steps from random starts, no lineage behind them. Is the history of
  the lineage (chains, the stream, long closings) needed once the search lives with the body?
  Prediction: three of five reach 280 meals at side 8 (the control: 292-310), and at least one
  fails outright (under 150), as two random starts of five failed in the stream.

**Outcome (2026-11-20; `runs/tissue/walk/long-s*`, `fresh-s*`; control / long / fresh).**

| | lineage (or start) 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| exam 4, two laps, 256 cells, K90 four hops deep | 514 / 571 / 438 | 570 / 558 / 415 | 548 / 556 / 327 | 412 / 475 / 302 | 558 / 582 / 484 |
| in a land: a way gone once, said right | 0.92 / 0.93 / 0.83 | 0.94 / 0.94 / 0.91 | 0.91 / 0.93 / 0.77 | 0.90 / 0.91 / 0.76 | 0.90 / 0.93 / 0.93 |
| side 8, seeking: meals a life | 304.8 / 313.3 / 291.7 | 309.0 / 309.3 / 289.6 | 306.9 / 310.6 / 274.4 | 292.3 / 303.4 / 258.1 | 309.9 / 311.8 / 302.6 |
| the first third | 60 / 64 / 46 | 61 / 62 / 49 | 60 / 62 / 39 | 46 / 56 / 30 | 62 / 63 / 56 |
| side 12, 512 cells | 423.2 / 449.4 / 402.4 | 434.2 / 443.1 / 399.6 | 432.0 / 447.1 / 350.1 | 391.9 / 421.4 / 299.6 | 442.8 / 454.5 / 424.5 |
| long over the control: side 8, side 12 | 1.03, 1.06 | 1.00, 1.02 | 1.01, 1.04 | 1.04, 1.08 | 1.01, 1.03 |
| fresh over the control: side 8, side 12 | 0.96, 0.95 | 0.94, 0.92 | 0.89, 0.81 | 0.88, 0.76 | 0.98, 0.96 |

- **`long`: more search pays a little.** Four hops deep x1.1 on two lineages, not three
  (missed). At side 8 every lineage eats more, by 0 to 11 meals (within 2% on three; lineages
  0 and 3 gain 3-4%); at side 12 with 512 cells 2-8% more, 3% on three (met). The gain is
  where the brain is small for its land and in the lineages that were behind; the search's own
  error hardly moves (0.012 to 0.007-0.012). `long-s0..4` are the best genomes in a land by
  that little; the records of doubt are on `twice`, which stays the base.
- **`fresh`: no start fails.** Three reach 280 at side 8 as said (292, 290, 303), and the two
  that do not eat 274 and 258, more than any lineage genome ate before 6i. I had said one
  would fail outright. For the first time five random starts of five give a brain that lives
  in a land. The lineage behind the control (chains, the stream, long closings) is worth
  2-12% at side 8 and 4-24% at side 12, nearly all of it in the first third (30-56 meals
  against 46-62): it helps a way gone once be said right (0.76-0.93 against 0.90-0.94).
  It is not needed; it is a head start.

### Forgetting, step 6: the rule searched with the parts of doubt in place. Asked first, then bars before the runs

Step 6j's lesson, applied: the three parts of doubt were set on genomes that never lived with
them. Here the rule is searched with wires that fade, in a world of forks whose ways change.

**The world** (`fork.Walk.wander`, `hazard`, new): after the study the body walks on by ways
drawn at random with its eyes open, for one to two laps' worth of ticks (a lap: as many ticks
as the land has links), and on every tick each way comes to lead somewhere new with a small
chance. Nobody tells the brain. The test is of the land as it has become, so some of what is
asked in the dark is STALE: changed since the body last went that way, and not to be known.
The error of the search (softplus) punishes a wrong word said loudly, so a brain that speaks
quietly of what is likely stale is paid for it; and what is likely stale is what is old.
**The wires fade** in the search as in a land (`Plan.fade`, by hand). Fading and hazard are
counted a lap, so that the age of a wire means the same in a land of 8 links and one of 128:
0.3 each (a wire loses three tenths of itself in a lap; a way has changed within a lap with
chance 0.3). In a land of side 8 the fading of step 5 is 0.08 a lap: the search's is four
times as fast. BY HAND: the fading, its rate, the hazard, the walk at random.

**Asked first, no search** (the look `python -m tissue.stale`, new: 256 cells, half a link per
cell, a walk of a lap and a half; the control genomes of lineages 0 and 3, 16 lives; bits
right / loudness one hop into the dark):

| | fresh, gone within half a lap | half a lap to a lap ago | a lap to two ago | stale | the error |
|---|---|---|---|---|---|
| no fading | 1.00 / 7.0-7.4 | 0.99-1.00 / 6.9-7.2 | 1.00 / 6.8-7.2 | 0.50 / 6.9-7.3 | 0.50-0.53 |
| fading 0.3 a lap | 1.00 / 6.0 | 0.98-0.99 / 5.4-5.5 | 0.95-0.98 / 4.4-4.6 | 0.50-0.51 / 4.9-5.2 | 0.39 |

With no fading a way that has changed is said as loudly as one that has not. With fading by
hand an old way is quieter, so the error falls by a quarter; a stale way is still said at
about five where a calibrated brain would whisper. That is what the search can mend.

**The searches** (`tissue.train --arm walk --laps 2 3 --hops 16 --vary`, 4000 steps from each
control genome; on the VM, `out/jobs12a.txt`, logs `out/jobs/f6-*`):

- `doubt-s0..4`: `--fade 0.3 --wander 1 2 --hazard 0.3`;
- `fade-s0..4`, a control: the wires fade and the land does not change (`--fade 0.3 --wander
  1 2`);
- `change-s0..4`, a control: the land changes and the wires do not fade (`--wander 1 2
  --hazard 0.3`). What 8000 steps of search with neither do is step 6l's `long`.

Each then sits the look `tissue.stale` (with fading and without), exam 4 with two laps, and
the exam's lands at side 8 (256 cells, 16 lives, a road kept, 4096 rooms; the land still and
shift 0.5): the base (the line at 0.4, nothing else) and the three parts at step 5's setting
(0.6, `--sure`, 0.0003); the `doubt` genomes also with the line at 0.5 and at 0.7, and a
pilot on other lands (`--seed 1`, 8 lives, `doubt-s0` and `doubt-s4`) from which the rule of
step 5 picks their line among 0.5, 0.6 and 0.7. Bars S3 and S4 are for the line picked.

- **S1, in forks**: the `doubt` genomes' error one hop into the dark, under fading at 0.3 a
  lap, is under 0.8 of the control genomes' there, and a way gone within half a lap is still
  right on 0.98 of the bits, on four lineages.
- **S2, nothing paid**: exam 4, two laps, no fading, 256 cells: K90 at hop 1 and four hops
  deep at least 0.9 of the control genome's, on four lineages.
- **S3, in a land doubt costs less**: where nothing changes, the base life is at least 0.98
  of the control genome's (304.8 / 309.0 / 306.9 / 292.3 / 309.9) and the life with the three
  parts at least 0.97 of that base (the control genomes: 0.942-0.955), each on four lineages.
- **S4, and catches more**: half the walls moved, the three parts: at least 120 meals in the
  half-life after the change on three lineages (the control genomes: 110.9 / 120.9 / 116.5 /
  104.4 / 111.0; the `doubter` 135).
- **S5, which part of the search did it**: at step 5's setting, the genomes searched with
  fading alone and those searched with change alone eat less in that half-life than `doubt`,
  each on four lineages.

Prediction: S1 met (it is the error the search sees); S2 met on three or four; S3 met in its
first half, the second open; S4 missed (a land's fading is a quarter as fast a lap as the
search's, and a land asks for whole rooms); S5 missed for the fading: by 6j, living with the
part is what matters, and `fade` will do most of what `doubt` does.

**Outcome (2026-11-21; `runs/tissue/walk/doubt-s*`, `fade-s*`, `change-s*`; 95 minutes a search).
The search did what the error asked and not what I meant, and in a land its genomes are worse
than the ones they started from.**

In forks (the look, 256 cells, five lineages; the error one hop into the dark, and how loudly
a way is said: gone within half a lap / a lap to two ago / stale):

| | the control (`twice`) | `doubt` | `fade` | `change` |
|---|---|---|---|---|
| no fading: the error | 0.46-0.54 | 0.25-0.26 | 0.38-0.47 | 0.25-0.26 |
| loudness | 7.0-8.1 / 6.9-8.0 / 6.9-8.0 | 2.8-3.1 / 2.9-3.1 / 2.8-3.1 | 5.9-6.8 / 5.2-6.8 / 5.2-6.8 | 3.0-3.2 / 2.7-3.1 / 2.8-3.1 |
| fading 0.3 a lap: the error | 0.35-0.39 | 0.24-0.26 | 0.36-0.38 | 0.29-0.32 |
| loudness | 6.0-6.7 / 4.6-5.3 / 4.8-5.6 | 2.5-2.8 / 2.0-2.3 / 2.1-2.4 | 5.3-5.8 / 4.1-5.1 / 4.3-5.1 | 2.0-2.4 / 1.4-1.7 / 1.5-1.8 |
| a way gone since half a lap / a lap to two ago, bits right (fading) | 1.00 / 0.96-0.98 | 1.00 / 0.94-0.99 | 0.99-1.00 / 0.88-0.98 | 0.98-1.00 / 0.82-0.93 |

In a land (side 8, 256 cells, 16 lives; control / `doubt` / `fade` / `change`):

| | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| exam 4, two laps, K90 at hop 1 / hop 4 over the control's: `doubt` | 1.00 / 1.13 | 0.90 / 1.05 | 0.88 / 0.74 | 0.86 / 0.83 | 0.93 / 0.98 |
| the same: `fade`; `change` | 0.71 / 0.52; 0.59 / 0.31 | 0.75 / 0.72; 0.89 / 0.70 | 0.60 / 0.40; 0.67 / 0.31 | 0.47 / 0.41; 0.80 / 0.69 | 0.68 / 0.57; 0.91 / 0.70 |
| a way that closed, gone once since, said right (no fading) | 0.96 / 0.99 / 0.86 / 0.63 | 0.99 / 0.76 / 0.98 / 0.91 | 0.95 / 0.66 / 0.58 / 0.74 | 0.97 / 0.68 / 0.44 / 0.78 | 0.99 / 0.42 / 0.83 / 0.92 |
| nothing changes, the base (no fading): a life | 305 / 314 / 208 / 189 | 309 / 307 / 297 / 231 | 307 / 190 / 96 / 197 | 292 / 218 / 143 / 225 | 310 / 124 / 236 / 233 |
| nothing changes, the three parts at 0.6 | 288 / 285 / 242 / 6 | 291 / 277 / 295 / 7 | 293 / 214 / 179 / 8 | 276 / 248 / 209 / 16 | 294 / 109 / 278 / 12 |
| half the walls moved, the three parts at 0.6: the half-life after | 111 / 119 / 91 / 2 | 121 / 122 / 91 / 3 | 117 / 79 / 71 / 3 | 104 / 91 / 79 / 5 | 111 / 13 / 96 / 5 |
| `doubt` at the line the pilot picked (0.5): a life still; the half-life after | 304; 109 | 298; 118 | 208; 73 | 233; 84 | 123; 16 |

- **S1: met by the letter, not in what it was for.** The error of the `doubt` genomes is
  0.63-0.73 of the control's and a fresh way is still right. But they did not learn to tell
  old from new: they learned to say EVERYTHING in a low voice (2.5-2.8 where the control says
  6.0-6.7), and a stale way is 0.82-0.86 as loud as a fresh one, as it is for the control
  under fading by hand (0.80-0.84). The genomes searched with change and no fading did the
  same. In the world I made a way is stale one asking in five whatever its age, so the least
  error is a low voice for all; the error pays for quietness as such.
- **S2: missed.** K90 at hop 1 and four hops deep is 0.9 of the control's on three lineages
  each, not four; the two controls of the search lose a third to a half.
- **S3: missed where it matters.** The base life is 1.03 / 0.99 / 0.62 / 0.74 / 0.40 of the
  control's. (With the three parts at the line picked, 0.97-1.09 of that base on four
  lineages: the letter of the second half, of a base that has fallen.)
- **S4: missed.** No lineage reaches 120 meals at the line picked (109 / 118 / 73 / 84 / 16;
  at 0.6 two do, by a hair: 119 and 122).
- **S5: met by the letter**: at step 5's setting `fade` eats less after the change than
  `doubt` on four lineages and `change` on all five. All three eat less than the control. The
  `change` genomes with `--sure` plan nothing at all (6-16 meals a life): a brain that says
  everything quietly is sure of nothing.

**What it says.** (1) Score what the teacher teaches, once more: an error that punishes a
loud wrong word, in a world where anything may be wrong, teaches a low voice, not doubt. The
land's own change is of another kind: nothing moves for half a life, then half the walls at
once. (2) A rule searched with wires that fade fast leans on the fading: without it, or with a
quarter of it, a way that closed is no longer undone in one walk (0.42-0.76 where it was
0.95-0.99). The search's fading was 0.3 a lap and the land's 0.08. (3) Lineages 0 and 1 took
no harm (lineage 0 even eats 9 +- 1 meals more with no doubt at all); 2, 3 and 4 lost a
quarter to more than half. Asked again below with those two things mended; if that fails too
the three parts stay by hand, as the gate does.

### Forgetting, step 6', asked again with what it lacked. Bars before the runs

Two changes, nothing else:

- **The land's own kind of change** (`fork.Walk.burst`, new): in every other world, at one
  tick of the walk drawn at random, a share of the ways come to lead somewhere new AT ONCE
  (half of them); in the other worlds nothing changes at all. So a loud word of a fresh way
  is right, and what is old is stale only where there was a change.
- **The land's own fading**: 0.08 a lap (0.0003 a tick at side 8), not 0.3.

`doubt2-s0..4` (`--fade 0.08 --wander 1 2 --burst 0.5`) and a control `fade2-s0..4` (the same
fading and walk, no change), 4000 steps from each control genome; then the look, exam 4, the
look `tissue.unlearn`, and the exam's lands as in step 6 (the base; the three parts with the
line at 0.5, 0.6, 0.7 for `doubt2`, at 0.6 for `fade2`; a pilot on other lands from which
step 5's rule picks the line). On the VM: `out/jobs13a.txt`, `out/jobs13c.txt`, logs
`out/jobs/h6-*`.

- **T1, in forks, and not by a low voice** (the look with the burst, fading 0.08 a lap, 256
  cells): the error one hop into the dark under 0.9 of the control genomes' there, and a way
  gone within half a lap said at least 0.8 as loudly as the control genomes say it, each on
  four lineages.
- **T2, nothing paid**: exam 4, two laps, no fading: K90 at hop 1 and four hops deep at least
  0.9 of the control's on four lineages.
- **T3, nothing lost in a land**: where nothing changes the base life at least 0.98 of the
  control's on four lineages, and with the three parts at the line picked at least 0.96 of
  that base on four (the control genomes: 0.942-0.955).
- **T4, more caught**: half the walls moved, the three parts: at least 120 meals in the
  half-life after the change on three lineages.

Prediction: T2 met; T3 met in its first half; T1 open (the voice may still drop); T4 missed:
I expect the control's 104-121 again, because what the rule can add to fading by hand is
small. If T3 and T4 are both missed, the search is not asked a third time.

A look beside it, the diagnosis of step 6 tested: the `doubt` genomes of step 6 in a land
with the fading they were searched with (0.3 a lap, 0.0012 a tick; the line at 0.4, nothing
left out). If they lean on their fading, lineages 2, 3 and 4 eat more with it than without
(190 / 218 / 124 where nothing changes). Logs `out/jobs/h6-lean-*`.

**Step 6' was NOT RUN: decided by a count before the search (2026-11-21, the same day).** The
world was built (`fork.Walk.burst`, tested, off unless asked) and the control genomes looked
at in it first, as the method asks. Under the land's own fading (0.08 a lap) they say a way
gone within half a lap at 6.9-7.2 and one not gone for two laps and more at 6.3-6.7: within a
life of the search (three to seven laps; a land's is ten) the age of a wire moves the voice by
7%. Now the error. With one voice L for all and a share q of the askings stale, it is
(1 - q) log(1 + e^-L) + q (log(1 + e^L) + log(1 + e^-L)) / 2, least at L = 2.2 when q is 0.17
(a burst in the last lap of the walk) and at 4.5 when q is 0.025 (a burst anywhere in a long
walk); the control genomes speak at 7. So the error pulls the whole voice down to a third
long before a 7% difference between old and new can earn anything, which is what step 6 did
and why. A search of this kind cannot be asked for doubt while age is marked so faintly in a
life as short as the search's. **The three parts of doubt stay by hand**, as the gate does.
What would change the count: an age the cell can read sharply (a mark of its own, not the
strength of the wire it is written in), or a search whose lives are as long as a land's.
The look `h6-lean` is still run.

### Forgetting, step 8: doubt by hand counted a lap. Predictions before the runs

The control genomes, 16 lives, the line at 0.6, `--sure` (on the VM, `out/jobs13.txt`, logs
`out/jobs/h8-*`):

- **Side 12, 512 cells, fading 0.08 a lap** (0.00014 a tick): where nothing changes a life at
  least 0.92 of the base's (423.2 / 434.2 / 432.0 / 391.9 / 442.8) on four lineages (at 0.0003
  a tick: 0.26-0.88); half the walls moved, the half-life after the change at least 1.2 times
  the base's (101 / 94 / 80 / 64 / 91) on three (the `doubter`, T 1440: 217).
- **A maze that changes** (walls 0.5, shift 0.5, side 8, 256 cells, 0.0003 a tick; yardsticks
  with the `doubter` at T 640): the half-life after the change at least 1.2 times the base's
  on three lineages, no life locked with doubt.
- **The best genomes** (`long-s0..4`, side 8, the three parts at step 5's setting): within 3%
  of the control genomes' with doubt on four lineages, in the land still (288 / 291 / 293 /
  276 / 294) and in the half-life after half the walls moved (111 / 121 / 117 / 104 / 111).

**Outcome (2026-11-22; base / doubt).**

| | lineage 0 | 1 | 2 | 3 | 4 | `explorer` | `doubter` |
|---|---|---|---|---|---|---|---|
| side 12, 512 cells, 0.08 a lap: nothing changes, a life | 423 / 419 | 434 / 413 | 432 / 429 | 392 / 354 | 443 / 444 | 517 | 504 (T 1440) |
| over the base (at 0.0003 a tick: 0.26-0.88) | 0.99 | 0.95 | 0.99 | 0.90 | 1.00 | | |
| side 12, half moved: the half-life after | 101 / 151 | 94 / 156 | 80 / 142 | 64 / 91 | 91 / 145 | 78 | 217 |
| side 12, half moved: a life; lives with no meal at its end | 234 / 286; 1 / 0 | 235 / 288; 1 / 0 | 220 / 282; 2 / 0 | 165 / 169; 5 / 0 | 242 / 299; 2 / 0 | 305; 6 | 441; 0 |
| a maze that changes (side 8): the half-life after | 19 / 28 | 8 / 32 | 12 / 24 | 10 / 22 | 8 / 28 | 11 | 102 (T 640) |
| a maze that changes: a life; lives with no meal at its end | 86 / 99; 4 / 2 | 79 / 109; 10 / 1 | 83 / 102; 3 / 2 | 65 / 77; 6 / 3 | 73 / 104; 7 / 1 | 113; 9 | 203; 0 |
| `long` over `twice`, both with doubt: a life still; the half-life after a change | 1.04; 1.06 | 1.01; 0.90 | 1.00; 0.98 | 1.04; 0.98 | 1.01; 1.13 | | |

- **Counted a lap, doubt works in the bigger land: both met.** At 0.08 a lap a life where
  nothing changes is 0.90-1.00 of the base's (0.92 on four lineages), the half-life after half
  the walls moved x1.41-1.78 of the base's on all five, and no life of 80 ends without a meal
  (the base: 11). Still 0.42-0.72 of what the table that doubts eats there.
- **A maze that changes: met in its first part.** Without doubt everything starves there, the
  explorer too (11 meals in the half-life after; 9 lives of 16 locked). Doubt lifts the brain
  x1.5-4.0 (22-32 meals) on all five; 9 lives of 80 still end without a meal (I said none; the
  base: 30). The table that doubts eats 102: in a maze a closed way costs a long way round,
  and the brain's doubt finds a quarter of what the table's does.
- **`long` with doubt**: level with `twice` (a life still x1.00-1.04, within 3% on three
  lineages and above it on two; the half-life after a change x0.90-1.13). By the letter both
  missed; no difference that five lineages show.
- The look `h6-lean` (step 6's `doubt` genomes with the fading they were searched with, 0.0012
  a tick): lineage 4 does lean on it (124 meals to 204 where nothing changes; its map right
  on 0.22 of the ways gone without fading and 0.86 with), the other four eat LESS with it (314
  / 307 / 190 / 218 to 221 / 230 / 156 / 116). The diagnosis of step 6 holds for one lineage
  of the three that broke.

## A thought as a wave (the third road of the order, from 2026-11-21)

Thinking in exam 6 is the largest hand-made part: copies of the brain, one for every room
thought of (up to 4096), walked breadth-first by the examiner, an exact match with the wanted
look, a road kept. The question of this road: can the TISSUE hold a thought, many rooms in one
state, and step it? First a ceiling by hand, before any mechanism.

### Wave, step 1: one state for many rooms. A look by hand, then guesses before the runs

By hand (a scratch look, not yet a module): a thought is FOUR states, one per first way out
of the room the body stands in. Each is stepped a hop at a time in the dark by all four ways
at once; the four states that come of it are averaged into one and brought back to the size
of a single copy's state (a gain, by hand). Such a state is a wave: a superposition of every
room a walk of that many steps may have reached. A room is in a wave when the wave's
interneurons lie along that room's code (the cosine with what a copy's interneurons say when
it has hopped into that room in the dark). The first way whose wave reaches the wanted room
first is taken. Sixteen ticks of the brain a hop, where the copies cost as many as there are
rooms. A second arm, `--refract`, by hand: a wave is kept clear of what the waves before it
were (no way back), so that it stays a front.

One life (`twice-s0`, 256 cells, side 8, asked at ticks 1200 and 2400 for every room of the
land as the wanted one), seen before these guesses were written:

- A room has ONE code, seen or thought of: the cosine between the two is 0.99.
- N codes laid over one another are told from the rest of the land: with N at 8 a member lies
  at 0.38 along the mix and another room at 0.07 (a member above another on 0.995 of the
  pairs); at 16, 0.29 and 0.09 (0.957); at 32, 0.24 and 0.11 (0.892). So one state of 256
  cells holds about 8 to 16 rooms.
- A wave stepped by the brain's own wires: at depth 2 it holds 4 rooms at 0.49 against 0.04
  for the rest, at depth 3 9 rooms at 0.27 against 0.06, at depth 4 16 rooms at 0.20 against
  0.09; from depth 6 on nothing is told apart. The first way is right for a room 1 step away
  on 1.00 of the askings, 2 steps 0.89-1.00, 3 steps 0.62-0.69, 4 steps 0.53 at best, and at
  chance beyond (chance: 0.33-0.45, the share of ways that begin a shortest road).
- Kept clear of where it has been, the wave is a front of 3 to 9 rooms, and it dies at the
  same depth (0.23 against 0.08 at depth 4, 0.11 against 0.04 at depth 6). So it is not the
  number of rooms that ends a wave: stepping a mix through the cells loses it.

**The look in full** (five lineages, 256 / 512 / 1024 cells, 4 lives, both arms; on the VM,
logs `out/jobs/w1-*`, records `runs/tissue/walk/twice-s*.wave*.json`). Guesses:

- growth pays for what one state can hold: the N at which a member is still above another
  room on 0.95 of the pairs doubles with the cells (16 at 256 cells, 32 at 512, 64 at 1024);
- and for depth a little: the last depth at which a wave lies twice as much along its own
  rooms as along the rest (4 at 256 cells) is one hop more at 1024, on four lineages;
- the first way for a room 3 steps away is right on at least 0.8 of the askings at 1024 cells
  (0.62-0.69 at 256) on four lineages, and beyond 6 steps at chance at every size.

**Outcome (2026-11-22; five lineages, 4 lives each; `twice-s*.wave.{plain,front}.m*.json`).**

| cells | 256 | 512 | 1024 |
|---|---|---|---|
| a room's code seen along its code thought of | 0.97-0.98 | 0.96-0.98 | 0.96-0.98 |
| 16 codes over one another: a member / another room along the mix | 0.29-0.32 / 0.07-0.13 | 0.28-0.32 / 0.07-0.14 | 0.28-0.32 / 0.07-0.14 |
| a member above another room, share of pairs: 16 codes; 32 codes | 0.93-0.97; 0.83-0.89 | 0.95-0.99; 0.87-0.94 | 0.96-0.99; 0.89-0.95 |
| the plain wave at depth 3 / 4 / 5: rooms held; along its rooms / the rest | 8: 0.32 / 0.06; 13: 0.23 / 0.07; 19: 0.19 / 0.07 | 8: 0.31 / 0.06; 14: 0.23 / 0.07; 20: 0.18 / 0.07 | 8: 0.31 / 0.07; 14: 0.23 / 0.07; 21: 0.19 / 0.08 |
| the front at depth 3 / 5 / 7: rooms held; along its rooms / the rest | 4: 0.38 / 0.04; 6: 0.19 / 0.04; 7: 0.11 / 0.03 | 4: 0.39 / 0.05; 7: 0.19 / 0.04; 7: 0.10 / 0.03 | 4: 0.38 / 0.05; 7: 0.20 / 0.05; 7: 0.11 / 0.03 |
| the first way right, a room 2 / 3 / 4 / 5 steps away: the plain wave | 0.96-1.00 / 0.44-0.70 / 0.30-0.47 / 0.14-0.31 | 0.95-1.00 / 0.49-0.70 / 0.31-0.45 / 0.12-0.37 | 0.96-1.00 / 0.61-0.70 / 0.24-0.60 / 0.18-0.53 |
| the same: the front (chance 0.28-0.30) | 0.98-1.00 / 0.70-0.87 / 0.48-0.65 / 0.40-0.55 | 0.97-1.00 / 0.79-0.88 / 0.47-0.66 / 0.39-0.49 | 0.96-1.00 / 0.80-0.84 / 0.58-0.75 / 0.43-0.61 |

- **Growth does not pay for what one state can hold: the first guess was wrong.** Sixteen codes
  laid over one another are told from the rest about as well at 1024 cells as at 256; a member
  lies along the mix at 0.3 and another room at 0.1 at every size. Two rooms' codes are as
  alike in a big brain as in a small one: a room's code is its LOOK (12 bits) spread over the
  cells, and more cells spread the same few dimensions wider. It is the figure of rung 2 again
  ("a placeless brain holds 18 items at any size"): a dense code does not use more cells for
  more things at once; places did.
- **Nor for depth: the second guess was wrong.** A wave lies along its own rooms and along the
  rest by the same figures at every size, hop for hop.
- **The third guess: right for the front, wrong for the plain wave.** Kept clear of where it
  has been, a wave gives the first way to a room 3 steps away on 0.80-0.84 of the askings at
  1024 cells on all five lineages (0.70-0.87 at 256), 4 steps 0.58-0.75 (0.48-0.65), 5 steps
  0.43-0.61 (0.40-0.55); the plain wave 0.61-0.70 at 3 steps and near chance at 5. The front
  is the better thought at every size and distance, and a little better with more cells.
- So the ceiling of a wave in THIS tissue is set by its code, not its size: about 16 rooms a
  state, a front that tells the way three or four steps out. Two roads follow. (1) Use it as
  it is: a body that thinks afresh on every tick needs the way to be right when the fruit is
  near and only better than chance when it is far; a life lived on waves is asked next. (2)
  Give the code more dimensions as the brain grows (sparse codes, or places as in rung 2), so
  that a bigger brain holds a bigger wave: the claim of the line, asked of thinking.

### Wave, step 2: a life lived on waves. A pilot, a rule that picks, bars before the exam

`--wave LINE` (BY HAND, `Mind.ripple` in `tissue/world.py`, new): the thought is four waves and
no copies. The brain is shown the wanted look in thought and what its interneurons say then is
the code sought (a room has one code, seen or thought of). From the room it stands in, one
state per first way; at each depth the first wave to lie along the code sought by more than
LINE gives the way; else every wave is stepped by all four ways at once, averaged, kept clear
of what the waves before it were, and brought back to the size of one copy's state. No road is
kept: the brain thinks afresh on every tick, `--depth` waves deep at most (10 asked). With no
wave reaching, the dice or `--curious`. Sixteen ticks of the brain a hop; nothing in it knows
how many rooms the land has. BY HAND: all of it; what is NOT by hand any more is the list of
rooms thought of, one copy each.

The pilot (other lands, `--seed 1`, 8 lives, `twice-s0`, `twice-s4`, side 8, 256 cells): the
line at 0.15, 0.25, 0.35, each with the dice and with `--curious 0.7`. **The rule that picks**:
the setting with the most meals a life (the mean of the two genomes). The exam is run for all
six settings on the exam's lands (five lineages, 16 lives), and at 1024 cells with the line at
0.25 and the dice; the bars are for the setting picked (on the VM, `out/jobs14.txt`, logs
`out/jobs/w2-*`).

- **V1, a life on waves**: at least 140 meals on three lineages (half of what the same genomes
  eat thinking with copies and no seeking: 273-292; `perfect` 182, a blind walker 7).
- **V2, near the fruit it is a thought**: trips of 3-4 steps walked straight on at least 0.8 of
  them in the last third of a life, on four lineages.
- **V3, no lock**: at most 2 lives of 80 with no meal in the last third.
- A guess, no bar: 1024 cells eat no more than 1.15 times 256 (the wave is no wider there).

Prediction: V1 open (my guess is 120-200 meals), V2 met, V3 met (a body that thinks afresh on
every tick and takes the dice when no wave reaches does not stand still).

**Outcome (2026-11-22): it fails at the pilot, and the exam is not run.** Two lives of
`twice-s0` on the pilot's lands, the dice with no wave reaching: with the line at 0.25 a way is
given on 0.92 of the ticks and the body eats 0.5 meals a life (a wave that does not hold the
room passes so low a line, and the body is passed to and fro); at 0.35, 7 meals; at 0.45 and
0.6, 4 meals with a way given on 0.00 of the ticks (a blind walker: 6). In a life lived with
copies the same `ripple` does give the first way to a neighbouring room every time, so the
part works; its reach is too short to live on. No way of reading the four waves mends it (two
lives, by the true distance of the room: the first wave over a line, the best wave at any
depth, within six hops, discounted by depth: 0.94-1.00 at 2 steps, 0.63-0.77 at 3, 0.42-0.67
at 4, and from 6 steps on 0.2-0.5 where chance is 0.35). V1 cannot be met by an order of
size. The pilot is run in full for the record (`out/jobs/w2-*`).

### Wave, step 3: thinking as dreams. A look by hand, a pilot of two lives, then bars

What the tissue does well in the dark is WALK: one state, one room after another, right hop
after hop. So, by hand, a thought with no list of rooms and no copy per room (`--dreams K`,
`Mind.dream` in `tissue/world.py`, new): K walks in the dark from the room the brain stands in,
each by ways drawn at random (never straight back by the way just come), all a step at a
time, `--depth` steps at most. The first walk to come upon the wanted look gives its first way,
and its ways are the road, kept as any road is. With `--explore`, the first walk to speak
quietly marks a way not known. Nothing in it knows what rooms there are or which were thought
of before. BY HAND: the dice of the walks, and the noting of which walk met the look first.
(K walks at once are K copies of the brain's activity, on one set of wires; they could as well
be K walks one after another.)

The look (scratch `dream.py`, one life of `twice-s0`, 256 cells, side 8, walks of 16 steps,
every room of the land as the wanted one; the first way right, of all askings, by the true
distance of the room):

| walks | 2 steps | 3 | 4 | 5 | 6 | 7 | 8 | 9-10 |
|---|---|---|---|---|---|---|---|---|
| 16, never straight back | 0.89 | 0.62 | 0.63 | 0.59 | 0.20 | 0.14 | 0.00 | 0.05 |
| 64 | 1.00 | 1.00 | 0.89 | 0.88 | 0.60 | 0.50 | 0.25 | 0.11 |
| 256 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.64 | 0.58 | 0.17 |
| 256, any way | 1.00 | 1.00 | 1.00 | 0.88 | 0.80 | 0.50 | 0.33 | 0.22 |
| four waves, the front (step 1) | 0.98 | 0.80 | 0.57 | 0.47 | 0.4 | 0.4 | 0.4 | |

Two pilot lives on dreams (`twice-s0`, the pilot's lands, a road kept, seeking at 0.4, walks of
16 steps), seen before the bars: 64 dreams 84 meals, 256 dreams 127 (copies, the same will:
294). The body goes toward a way not known on 0.31-0.38 of its ticks (with copies: 0.07): a
thought that does not find the fruit by chance takes the fruit for out of reach and goes to
look at something else.

**The look in full** (five lineages, 256 / 512 / 1024 cells; `out/jobs/w3-look-*`, records
`twice-s*.dream.m*.json`). **The pilot** (other lands, `--seed 1`, 8 lives, `twice-s0`,
`twice-s4`, 256 cells, a road kept): 64 / 256 / 1024 dreams of 16 and of 32 steps, each with
seeking at 0.4 and with the dice alone. **The rule that picks**: the most meals a life, the
mean of the two genomes, among the five settings the exam is run for: 64, 256 and 1024 dreams
of 32 steps with seeking; 256 of 16 steps with seeking; 256 of 32 steps with the dice. The
exam: those five on the exam's lands, five lineages, 16 lives (`out/jobs14.txt`, logs
`out/jobs/w3-*`). Bars for the setting picked:

- **R1, a life on dreams**: at least 0.6 of the same genome's life on copies with seeking
  (304.8 / 309.0 / 306.9 / 292.3 / 309.9: so 183 / 185 / 184 / 175 / 186) on four lineages.
- **R2, no lock**: at most 2 lives of 80 with no meal in the last third.
- **R3, more dreams, more meals**: with walks of 32 steps and seeking, 256 dreams eat more than
  64 and 1024 more than 256, each on four lineages. A dial the brain can turn with what it
  can afford.

Prediction: R2 and R3 met; R1 open (two lives at 256 dreams of 16 steps gave 0.43; longer
walks and more of them should lift it, my guess is 0.5-0.7).

**Outcome (2026-11-22, evening): R1, R2 and R3 are met on all five lineages. A life can be
lived on dreams.**

The pilot (meals a life, the mean of `twice-s0` and `twice-s4`, 8 lives each on the pilot's
lands; copies with seeking on the same lands: 294 and 301):

| dreams | 16 steps, seeking | 16 steps, the dice | 32 steps, seeking | 32 steps, the dice |
|---|---|---|---|---|
| 64 | 118 | 83 | 130 | 101 |
| 256 | 165 | 139 | 170 | 128 |
| 1024 | 203 | 165 | 207 | 173 |

The rule picks **1024 dreams of 32 steps with seeking** (207; the next of the five, 256 dreams
of 32 steps, 170). The exam (the exam's lands, 16 lives a lineage, records
`twice-s*.world8.thinks.dreams*.depth*...keep.json`):

| | meals a life, five lineages | of the life on copies | first / last third, of the copies' | ticks toward a way not known | lives of 80 with no meal in the last third |
|---|---|---|---|---|---|
| 64 dreams of 32 steps, seeking | 128 / 131 / 135 / 115 / 138 | 0.39-0.45 | 0.32-0.41 / 0.41-0.47 | 0.20-0.33 | 0 |
| 256 of 32, seeking | 172 / 180 / 175 / 158 / 178 | 0.54-0.58 | 0.39-0.51 / 0.58-0.61 | 0.14-0.18 | 0 |
| **1024 of 32, seeking (picked)** | **208 / 212 / 212 / 192 / 219** | **0.66-0.71** | 0.48-0.58 / 0.71-0.74 | 0.08-0.10 | 0 |
| 256 of 16, seeking | 169 / 172 / 169 / 154 / 175 | 0.53-0.56 | 0.50-0.58 / 0.55-0.56 | 0.23-0.34 | 0 |
| 256 of 32, the dice | 131 / 120 / 116 / 111 / 129 | 0.38-0.43 | 0.07-0.16 / 0.50-0.61 | 0 | 4 |
| copies, seeking (step 6j) | 305 / 309 / 307 / 292 / 310 | 1 | 1 / 1 | 0.06-0.08 | 0 |

- **R1 met**: 0.66-0.71 of the life on copies on all five lineages (the bars were 183 / 185 /
  184 / 175 / 186). My guess was 0.5-0.7: it came at its top. No life of the 80 eats more than
  the same life on copies.
- **R2 met**: no life is locked in any arm that seeks (the least life of the picked setting:
  91-127 meals). With the dice alone four lives of 80 eat nothing in the last third and three
  of them 2-5 meals in a whole life (copies with the dice: none): a lock that seeking undoes;
  not looked into yet.
- **R3 met**: on all five lineages 256 dreams eat x1.29-1.38 what 64 do and 1024 x1.18-1.23
  what 256 do (160 and 159 lives of 160 eat more). A dial, and each fourfold buys less.
- Longer walks buy little: 256 dreams of 32 steps eat x1.02-1.05 what 256 of 16 do. Seeking is
  worth more to dreams than to copies: x1.3-1.5 (copies: x1.05-1.12).
- **Where it loses.** The map is as right as with copies (0.995-1.00 of the ways gone said
  right): what is lost is lost in thought. In the last third, the land known, the picked
  setting eats 89-93 where copies eat 125: a trip of 1-4 steps is walked straight every time,
  of 5-8 steps on 0.62-0.66 of them (x1.11-1.14 the shortest road), of 9 and more on 0.04-0.06
  (x1.64-1.77; copies 0.92-0.98 and x1.02). The road a dream finds is a walk by ways at
  random: a far room is reached by a long road or not at all. In the first third it eats half
  of what copies do, and it gets about its land more slowly: 0.68-0.72 of the ways gone by an
  eighth of a life (copies 0.79-0.82), and the MORE dreams the fewer (64 dreams 0.70-0.79, 256
  dreams 0.70-0.75).
- **The look in full** (five lineages, 256 walks of 16 steps, the first way right of all
  askings; chance 0.28-0.33):

  | cells | 3 steps | 4 | 5 | 6 | 7 | 8 | 9-10 | 11-13 |
  |---|---|---|---|---|---|---|---|---|
  | 256 | 1.00 | 0.98 | 0.88 | 0.78 | 0.61 | 0.48 | 0.33 | 0.17 |
  | 512 | 0.99 | 0.95 | 0.88 | 0.77 | 0.65 | 0.50 | 0.34 | 0.17 |
  | 1024 | 0.98 | 0.97 | 0.91 | 0.83 | 0.73 | 0.53 | 0.36 | 0.18 |
  | 256 cells, 64 walks | 0.93 | 0.85 | 0.70 | 0.54 | 0.38 | 0.23 | 0.15 | 0.07 |
  | 256 cells, 16 walks | 0.66 | 0.50 | 0.37 | 0.24 | 0.17 | 0.10 | 0.05 | 0.02 |
  | 256 cells, 256 walks, any way | 0.99 | 0.94 | 0.73 | 0.62 | 0.44 | 0.29 | 0.16 | 0.10 |

  A dream's reach is bought with WALKS, not with cells: a brain of 1024 cells dreams no
  farther than one of 256. Never straight back is the better walk at every distance. The
  road of the soonest walk is 1.0-1.3 times the shortest for the rooms it does reach.
- The pilot of wave step 2, in full for the record (`twice-s{0,4}.world8.thinks.wave*`): the
  line at 0.15, 0.1 and 0.2 meals a life; at 0.25, 0.8 and 1.2; at 0.35, 4.5 and 3.8 (a blind
  walker: 6); 5 to 8 lives of 8 with no meal in the last third. The `--curious` arm was no
  arm: the same lives to the meal (curiosity listens to what copies say one hop ahead, and a
  thought of waves has no copies; my mistake in setting it up).

BY HAND in a life on dreams: the dice of the walks; K walks at once (K copies of the brain's
activity on one set of wires); the noting of the walk that met the look (an exact match); the
road kept; the line of seeking. NOT by hand any more: the list of rooms thought of, the
breadth-first order, a copy for every room.

### Wave, step 4: dreams mended. A look at where a life on dreams loses, three parts by hand, bars

**The look** (scratch `dreamwhy.py`, on the VM as `out/tissue.dreamwhy.py`: every thought
thought afresh in a life is noted, and the road it gives is walked in the true land;
`twice-s0`, two lives on the pilot's lands, 256 dreams of 32 steps with seeking, a road kept;
the two lives are the pilot's own to the meal, 148 and 137). Seen before the bars:

| third of a life | 1 | 2 | 3 |
|---|---|---|---|
| ticks on a road that is TRUE (walked in the land it ends at the fruit) | 0.16 | 0.75 | 0.83 |
| ticks on a road that is FALSE | 0.39 | 0.18 | 0.05 |
| ticks toward a way not known | 0.43 | 0.07 | 0.11 |
| true roads to a room 5-8 steps away: their length over the shortest; with loops cut | 1.27; 1.05 | 1.45; 1.06 | 1.36; 1.06 |
| the same, 9 steps and more away | 1.73; 1.06 | 1.75; 1.10 | 1.85; 1.08 |

- **Loops.** The road of a walk at random is 1.3-1.9 times the shortest. With its loops cut
  (where the walk says a room it has said before, what lay between left out) it is 1.05-1.10
  times, and the shortest on 0.83-0.85 of the askings 5-8 steps away, 0.60-0.78 beyond.
- **False roads.** In the first third the body spends 0.39 of its ticks on roads that do not
  lead to the fruit: a walk comes upon the wanted look where that room is not. Two of three
  such walks never spoke quietly (under 0.4 of loud) on their way; they say the look at
  0.59-0.65 of loud, where a true road says it at 0.73-0.84. A thought of 256 walks of 32
  steps is 8 000 sayings, and a brain half taught says a room it knows behind a way it has
  never gone, in a middling voice, often enough for some to be the wanted look (my reading;
  why it says so was not looked into). A false road is walked 3 steps before a room is not as
  said, later in life 6-11.
- **False leads.** In the last two thirds EVERY way not known that a thought leads to has
  been gone: the walk is 14-20 steps deep in the dark and its voice has sunk under the line.
  The body walks toward nothing on 0.07-0.11 of its ticks.

**Three parts, BY HAND**, each with its off-switch (`Mind.dream` in `tissue/world.py`):

- `--cut`: the road a walk gives has its loops cut.
- `--again`: with a road in hand the brain dreams on, on every tick: walks no longer than what
  is left of the road less a step, and a shorter road takes the place of the one in hand; on
  a road toward a way not known, a road to the fruit takes its place. A thought is never
  done (a brain that never stops, the user's word of 2026-11-11, asked of thinking); the road
  in hand is only the best so far.
- `--sure`, the switch of forgetting step 5, whole for dreams now: what a walk says quietly
  (under the line `--explore`) is not the wanted look, AND THE WALK ENDS THERE: a new one
  begins from the room the brain stands in. A road is only through what the brain is sure
  of. The line at 0.4, 0.5 or 0.6.

New in every record from here on: `hops`, what thinking costs (a tick of one copy or of one
walk in the dark, a tick of life), so that settings are compared at their cost. Lives without
the new switches are the same to the step as before (checked on small lives).

**The pilot** (the pilot's lands, `--seed 1`, 8 lives, `twice-s0`, `twice-s4`, 256 dreams of 32
steps, seeking, a road kept; as they are: 170 meals): cut; again; cut and again; cut and again
and sure with the line at 0.4, at 0.5, at 0.6. By the look cut and again are in; what the
pilot is asked is `sure` and its line. **The rule that picks** the mended setting: of the four
with cut and again (no sure; sure at 0.4, 0.5, 0.6), the most meals a life, the mean of the
two genomes.

**The exam** (the exam's lands, five lineages; `out/jobs16.txt`, logs `out/jobs/w4-*`): at 256
dreams the pilot's six settings, 16 lives; at 64 dreams the four the rule can pick, 16 lives;
at 1024 dreams the same four, 8 lives (the first 8 lands of the exam: the same lives as the
first 8 of step 3's record). Bars, for the setting picked:

- **M1, mended, 256 dreams eat what 1024 did**: more than 1024 dreams as they are (208 / 212 /
  212 / 192 / 219) on four lineages.
- **M2, three quarters of a life on copies**: at 256 dreams at least 0.75 of the life on copies
  (so 229 / 232 / 230 / 219 / 232) on four lineages.
- **M3, each part is worth having** (256 dreams): cut alone eats more than dreams as they are
  (172 / 180 / 175 / 158 / 178), again alone more than dreams as they are, and cut with again
  more than either alone, each on four lineages. (`sure` is the pilot's question: whichever
  way it falls, the setting with it and the one without are both in the exam.)
- **M4, the land known**: in the last third at least 0.85 of what copies eat there (125: so
  106) on four lineages, and trips of 9 steps and more in that third take at most 1.25 steps
  per step of the shortest road (as they are: 2.13-2.32 at 256 dreams).
- **M5, no lock**: at most 2 lives of 80 with no meal in the last third.
- **M6, 64 dreams mended eat what 256 did**: more than 256 dreams as they are, on four
  lineages.

Predictions. The pilot picks `sure` with the line at 0.5 (the false roads speak at 0.6 of
loud, the true ones at 0.73-0.84; at 0.6 too many true roads are lost). M1 met. M2 open, my
guess 0.72-0.85 of copies. M3 met for cut (x1.15-1.3) and for cut with again; again alone
open (it dreams more, so it also comes upon more false roads). M4 met. M5 met. M6 missed by a
little (64 walks seldom reach beyond 5 steps, mended or not: my guess 0.9-1.05 of the 256 as
they are). At 1024 dreams, a guess and no bar: the setting picked eats x1.15-1.3 what 1024
dreams as they are do on the same lives, 0.8-0.9 of copies. The cost, a guess: `again` makes
thinking three times as dear in hops.

**Dreams as they are in the bigger land** (running beside, `out/jobs15.txt`, logs
`out/jobs/w4-big-*`): side 12, 512 cells, `twice-s0`, `twice-s4`, 8 lives, 256 and 1024 dreams
of 32 steps with seeking. A guess: 1024 dreams eat 0.35-0.5 of what copies do on the same
lives (the fruit is more often farther than a walk reaches), 256 dreams 0.25-0.4.

(Forgetting step 8, the look `h6-lean` and wave step 1 ran on the VM on 2026-11-21; wave step 2 next.)

### Forgetting, step 7: doubt by hand in other lands and at a slower fading. Predictions before the runs

Breadth for step 5's setting, the control genomes, 16 lives (on the VM, `out/jobs12b.txt`,
logs `out/jobs/g7-*`):

- **Slower fading** (side 8, the line at 0.6, `--sure`; the land still and shift 0.5): at
  0.0002 a tick a life where nothing changes is at least 0.96 of the base's and the half-life
  after the change at least 1.2 times the base's, each on four lineages; at 0.0001, at least
  0.98 and under 1.2 on four (too slow to matter in a life of 2560 ticks).
- **A bigger land** (side 12, 512 cells, a life of 5760 ticks; the same setting; yardsticks
  with the `doubter` at T 640 and at T 1440, a quarter of that life): the half-life after
  half the walls have moved at least 1.3 times the base's on three lineages; where nothing
  changes a life at least 0.92 of the base's (423.2 / 434.2 / 432.0 / 391.9 / 442.8) on four.
  A bigger land has more ways to keep fresh at the same fading, so doubt should cost more
  there.

**Outcome (2026-11-21; `twice-s*` records; base / the three parts).**

| | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| side 8, nothing changes, a life over the base: fading 0.0001 / 0.0002 / 0.0003 | 0.96 / 1.00 / 0.95 | 0.96 / 0.95 / 0.94 | 0.97 / 0.98 / 0.96 | 0.99 / 0.93 / 0.94 | 0.98 / 0.98 / 0.95 |
| side 8, half moved, the half-life after: 0.0001 / 0.0002 / 0.0003 (the base 72-91) | 94 / 111 / 111 | 104 / 111 / 121 | 89 / 104 / 117 | 100 / 93 / 104 | 90 / 103 / 111 |
| side 12, 512 cells, nothing changes, a life (`explorer` 517, `doubter` T 1440: 504) | 423 / 343 | 434 / 288 | 432 / 346 | 392 / 103 | 443 / 389 |
| side 12, half moved, the half-life after (`explorer` 78; `doubter` T 640: 94, T 1440: 217) | 101 / 111 | 94 / 91 | 80 / 129 | 64 / 37 | 91 / 145 |
| side 12, half moved, lives with no meal in the last third | 1 / 0 | 1 / 0 | 2 / 0 | 5 / 0 | 2 / 0 |

- **Slower fading: both predictions missed.** At 0.0002 a life costs 0-7% (0.96 asked on four
  lineages: three) and the half-life after the change is x1.13-1.53 of the base's (1.2 on
  three, four asked); at 0.0001 it still costs 1-4% and still lifts that half-life x0.99-1.30.
  A slower fading buys back little of the cost and gives up part of the gain: the cost is not
  the fading's speed, it is what is left out of a thought. 0.0003 stays.
- **The bigger land: both missed, and the yardstick says why.** At side 12 the same fading a
  tick costs 12-74% where nothing changes and lifts the half-life after the change on two
  lineages of five. The table that doubts shows the same thing: a span of 640 ticks, right at
  side 8, is worse than no doubt at all at side 12 (194 meals a life against the explorer's
  305), and a quarter of that land's life, 1440 ticks, is right (441). **Doubt has to be
  counted by the land: so much a lap, not so much a tick.** 0.0003 a tick is 0.08 a lap at
  side 8 and 0.17 at side 12; asked next at 0.08 a lap there (0.00013 a tick).

### Forgetting, step 1: what the old map is worth. Predictions before the runs

`--shift 0.5 --reborn` (a look, `Land.reborn`): at the change the body is born again where it
stands, the same make of body with its memory empty (a brain: new wires; a table: no entries;
one born knowing: knowing the land as it now is, the ceiling after a change). What is compared
is the half of a life after the change: the body that kept its memory against the same body
born again. Side 8, the exam's lands, 16 lives.

- Yardsticks first. `perfect` born again eats less in that half than `perfect` that kept its
  table (a table corrected a walk at a time is worth more than none to a body that explores by
  the dice). `explorer` born again eats MORE, at least 1.5 times: a body that seeks would learn
  the land as it is now within a few hundred ticks, and the table it kept holds walls it will
  never doubt.
- The brain (512 cells, a road kept, 4096 rooms; kept its wires, B14: 54.7 / 60.8 / 48.6 / 46.5
  / 48.7 meals in that half): born again it eats at least as much, on four lineages of five.
  The old map is worth nothing to it, or less.

**Outcome for the yardsticks (2026-11-17; `runs/tissue/world8.shift0.5.reborn.yardsticks.json`;
the brain's five runs are on the VM, `out/jobs/f1-reborn-m512-s*`).**

| side 8, shift 0.5; meals in the half of a life after the change | kept its memory | born again | the land not changed |
|---|---|---|---|
| `perfect` | 74.2 | 39.1 | 136.8 |
| `seeker` | 69.7 | 38.9 | 158.7 |
| `explorer` | 71.6 | 140.1 | 170.6 |
| born knowing: the old land / the land as it now is | 88.4 | 182.9 | 189.1 |

- Both met: `perfect` born again eats half of what it eats with its table kept; `explorer`
  born again eats twice as much (140 against 72).
- The land after the change is as rich as before: one born knowing it as it now is eats 183 in
  that half, against 189 where nothing changed. So every memory loses more than half of what
  is there to eat, and the one that seeks loses it to its own table: to a body that wants to
  know, what it knew of the old land is worth less than nothing (minus 68 meals).

**Outcome for the brain (2026-11-18; `hold-s*.world8.shift0.5.reborn.*.m512.json`).**

| 512 cells, side 8, shift 0.5, a road kept | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| meals in the half-life after the change: kept its wires (B14) | 54.7 | 60.8 | 48.6 | 46.5 | 48.7 |
| born again | 55.4 | 58.4 | 68.5 | 42.2 | 53.3 |
| born again over kept | 1.01 | 0.96 | 1.41 | 0.91 | 1.10 |
| the difference and its error over the 16 lives | +1 +- 10 | -2 +- 9 | +20 +- 13 | -4 +- 8 | +5 +- 7 |
| meals in the first 480 ticks after the change: kept / born again | 8.7 / 3.3 | 8.0 / 2.9 | 7.8 / 5.3 | 7.7 / 3.3 | 8.5 / 2.7 |
| meals per 160 ticks at the end of the life: kept / born again | 13.9 / 15.1 | 14.9 / 16.3 | 11.0 / 16.5 | 11.2 / 13.0 | 10.1 / 16.4 |
| lives with no meal in the last third: kept / born again | 2 / 0 | 0 / 0 | 2 / 2 | 2 / 0 | 1 / 0 |

- **Missed by the letter** (born again eats at least as much on three lineages, four asked),
  **true in what it says**: no lineage's difference stands clear of its error. All a brain
  knew of a land is worth nothing to it once half of that land has moved.
- It is worth something at first and costs later: with its wires kept the brain eats 8 meals
  in the first 480 ticks where the newborn eats 3 to 5, and by the end of the life the one born
  again eats more per tick on all five lineages (13-16.5 against 10-15) and 2 of its lives are
  locked against 7. The old map is a start and then a weight.

### Forgetting, step 2: how many walks write a way, how many undo one. Predictions before the run

`python -m tissue.unlearn` (a look, no exam): a life in a land that changes at its half; every
40 ticks the examiner asks a copy carried into every room what lies by each way, and the ways
are sorted by what the change did to them (closed, opened, a way still, a wall still) and by
the walks since (before the change: since birth). Side 8, shift 0.5, 256 cells, a road kept,
4096 rooms, the exam's lands, 8 lives, five lineages. Written after one pilot life on another
land (lineage 0), so these are not blind:

- Writing: a way gone once since birth is said right on more than half of the askings, gone
  twice on 0.9 or more, on four lineages.
- Undoing takes more walks than writing: a way that closed, gone once since the change, is
  said right (the room the body stays in) on less than half; 0.9 is reached at 3-4 walks or
  later, on four lineages.
- A wall that opened: the brain goes more than half of them by the end of the life, on four
  lineages (a table goes none: it holds them for walls).

**Outcome (2026-11-18; run on the VM, `hold-s*.unlearn8.shift0.5.json`). The share of askings
on which the brain says the room that lies by a way now, by the walks it has had:**

| 256 cells, side 8, shift 0.5, 8 lives | lineage 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| a way, since birth: gone once / twice / 3-4 times | 0.54 / 0.92 / 0.98 | 0.53 / 0.93 / 0.98 | 0.61 / 0.93 / 0.98 | 0.37 / 0.80 / 0.93 | 0.51 / 0.87 / 0.97 |
| a wall, since birth: once / twice | 0.71 / 0.94 | 0.77 / 0.95 | 0.80 / 0.96 | 0.72 / 0.91 | 0.79 / 0.95 |
| a way that closed, since the change: 1 / 2 / 3-4 / 5-8 | 0.23 / 0.47 / 0.91 / 1.00 | 0.00 / 0.35 / 0.92 / 0.98 | 0.01 / 0.61 / 0.93 / 0.99 | 0.43 / 0.68 / 0.84 / 0.90 | 0.00 / 0.52 / 0.73 / 0.80 |
| a wall that opened: 1 / 2 / 3-4 / 5-8 | 0.00 / 0.40 / 0.91 / 1.00 | 0.00 / 0.23 / 0.91 / 0.98 | 0.01 / 0.27 / 0.96 / 1.00 | 0.02 / 0.30 / 0.79 / 0.95 | 0.00 / 0.14 / 0.64 / 0.78 |
| gone once since the change, still said as before: closed / opened | 0.44 / 0.65 | 0.73 / 0.63 | 0.61 / 0.58 | 0.09 / 0.61 | 0.56 / 0.60 |
| not changed and not gone since the change, still right: a way / a wall | 0.96 / 0.89 | 0.97 / 0.92 | 0.91 / 0.88 | 0.83 / 0.84 | 0.90 / 0.87 |
| walls that opened, gone by the end of the life | 0.75 | 0.77 | 0.68 | 0.83 | 0.85 |

- **Writing: half met.** Gone once, a way is said right on more than half of the askings on
  four lineages (0.51-0.61; lineage 3: 0.37); gone twice, on 0.9 on three, not four (0.80-0.93).
  A wall is written faster than a way (0.71-0.80 at one bump).
- **Undoing: met** on all five. One walk of a way that closed leaves the brain saying neither
  (the new on 0.00-0.43, the old still on 0.09-0.73); 0.9 comes at 3-4 walks on three
  lineages, at 5-8 on lineage 3, never on lineage 4.
- **Walls that opened: met** on all five: the brain goes 0.68-0.85 of them, where a table goes
  none. Its dice and its false roads do try a wall.
- So undoing costs about one walk more than writing, not ten: the rule is not what makes the
  brain slow after a change (B7). And what did not change stays: 0.83-0.97 of it is still said
  right. What is slow is that every way, new or changed, wants two to four walks, and that
  nothing sends the body to look.

### Forgetting, step 3: a yardstick that doubts. Predictions before the run

`doubter` (`--yardsticks --doubt T`): an `explorer` whose table no longer holds a way it has
not gone for T ticks. Such a way is a way never gone again, and its will seeks it when it has
no road to the fruit. Nothing tells it that the land has changed. T at 320, 640 and 1280 ticks
(a life at side 8 is 2560), the exam's lands, 16 lives. Written after a look at four lives in
other lands (T 640, shift 0.5: a last third of 114 against the explorer's 73):

- shift 0.5, T 640: the last third at least 98 (0.85 of the explorer's last third in the
  land that does not change, 115; the explorer that never doubts: 60).
- the land that does not change, T 640: a life at least 281 (0.9 of the explorer's 312):
  doubt costs little where nothing changes.
- a guess, no bar: the shortest T is best after a change and worst without one.

**Outcome (2026-11-17 and 18; `runs/tissue/world8*.yardsticks.doubt*.json`; five of the nine
were run on the VM after the desktop ran short of memory: the old yardsticks, run again there
beside each `doubter`, come out meal for meal as on the desktop).**

| side 8, meals a life (by third) | `explorer` | `doubter`, T 320 | T 640 | T 1280 |
|---|---|---|---|---|
| the land not changed | 311.8 (83 / 114 / 115) | 251.0 (71 / 95 / 85) | 307.9 (83 / 113 / 112) | 311.5 (83 / 114 / 114) |
| shift 0.25 | 256.4 (83 / 89 / 85) | 229.0 (71 / 76 / 83) | 280.2 (83 / 95 / 103) | 275.9 (83 / 95 / 98) |
| shift 0.5 | 212.8 (83 / 70 / 60) | 235.6 (71 / 73 / 92) | 275.3 (83 / 88 / 105) | 269.3 (83 / 81 / 105) |

- **Shift 0.5, T 640: met**, a last third of 105 (98 asked) where the explorer that never
  doubts eats 60 and the same explorer in a land that never changed 115. Doubt and the wish to
  know together take a land half moved almost whole.
- **The land not changed, T 640: met**, 308 meals (281 asked) against 312: doubt at that span
  costs one meal in a hundred where nothing changes. At T 640 a life is worth 308, 280 and 275
  meals in the three lands: such a memory hardly feels that half its land moved.
- The guess was wrong in its first half: the shortest span is the worst after a change as
  well (92 in the last third against 105), and it costs a fifth of a life where nothing
  changes (251 against 312). A way has to be held for longer than it takes to come by it again.

### Forgetting, step 4: wires that fade. A pilot, and why no exam was run on it

`Plan.fade` (BY HAND, off unless asked; `--fade F` in a land): on a tick the brain sees, every
plastic wire first loses the share F of itself, the dendrites of the ways not gone as well. No
wire fades in thought. It is the plainest forgetting there is, and with seeking (6h) it was
meant to be the brain's `doubter`: what fades is spoken of quietly and sought again.

A pilot before any bar: ONE life of lineage 0, 256 cells, a land of side 8 that does not change
(other dice than the exam's), a road kept, seeking at 0.4. Every 80 ticks the examiner asks a
copy carried into every room what lies by each way; the ways sorted by the ticks since the
body last went them:

| fade a tick | meals (by third) | gone 0-80 ticks ago: right / loud | 160-320 | 320-640 | 640-1280 | never gone: loud |
|---|---|---|---|---|---|---|
| 0 | 234 (41 / 99 / 94) | 0.99 / 5.2 | 0.95 / 4.9 | 0.90 / 4.7 | 0.89 / 4.6 | 2.6 |
| 0.001 | 163 (31 / 70 / 62) | 0.99 / 4.6 | 0.92 / 4.0 | 0.77 / 3.6 | 0.44 / 3.2 | 2.5 |
| 0.003 | 57 (12 / 20 / 25) | 0.99 / 4.0 | 0.62 / 2.8 | 0.14 / 2.4 | 0.05 / 2.4 | 2.2 |

- A wire that fades makes the brain WRONG about a way long before it makes it quiet. At 0.001 a
  way not gone for 640 ticks is said right on 0.44 of the askings and still at 3.2, where what
  was never gone is said at 2.5. A table that doubts drops a way whole; a brain that fades goes
  on saying a room, and the wrong one. Its false roads cost a third of the meals where nothing
  changed (one life; the size of it is not measured).
- With no fading at all the brain already forgets a little: a way not gone for 1280 ticks is
  said right on 0.81 (0.99 when fresh).
- So fading alone is not the brain's doubt. Doubt needs the brain to speak QUIETLY of what it
  is no longer sure of before it speaks wrongly of it, and that is the cell's affair: what it
  says when its wires say little. The search's error does punish a wrong word said loudly;
  whether the walk arm of 6i, which loads a brain past what it can hold, buys some of that is
  looked at when 6i is in. No exam is run on fading as it is; the switch stays, off.

## Where the world line stands (2026-11-19), for whoever decides the next step

**What exists.** Five genomes (`runs/tissue/hold/hold-s0..4.pt`) found by gradient on abstract
exams of rung 1 (8-64 interneurons, chains and an unbroken stream), never on a world, and
since step 6i five more (`runs/tissue/walk/walk-s0..4.pt`), each searched on from one of those
in a world of forks where a link may be shown once (still 8-64 interneurons, still no land).
Grown to
64-1024 cells, each is born into a walled land it has never seen (exam 6, `tissue/world.py`),
told only what the room with the fruit looks like, and lives one life of 40 ticks a room with
no study. Its wires learn the land as it walks; it thinks by a hand-made procedure on those
wires; it acts on the thought.

**What it does.** With a wide thought (4096 rooms) and a road kept (step 6e) it eats about as
much as a perfect memory of every way it has gone, or more, in every land asked (sides 6-12,
mazes, open lands, 10 or 21 trees), and 0.4-0.7 of what a body born knowing the land eats. No
lineage, no land has the lock of 6b any more (except with slack in the match). All of it with
cells whose rule was never shown a world. THE YARDSTICK since step 6g is `explorer`, a perfect
memory that seeks the nearest way it has never gone: 0.81-0.89 of born-knowing in every land
that does not change. Against it the brain, given the same wish (6h), stands at 0.57-0.85 at
side 8, 0.34-0.69 at side 12 and 0.36-0.72 in the maze with the lineage genomes, and with the
genomes of step 6i at 0.85-1.00 at side 8 and 0.64-0.85 at side 12 (512 cells), and with the
control genomes of 6j (`runs/tissue/walk/twice-s0..4.pt`, the ones carried on) at 0.94-0.99 at
side 8.

**What was learned, in the order it was found** (each in the journal, 2026-11-12..17):
1. The depth of a thought was bound by my limit of rooms (6b's look), and a wider thought
   alone made the body pass between two rooms forever (6b): WHEN to think is as much a part of
   thinking as how. A road kept until it fails (6e) undid the lock and lifted the weak lineages
   most (36-95 meals in 6a, 160-241 now).
2. The brain beats a perfect memory because its mistakes explore: a false road through a way
   never gone is walked toward the fruit, and by the end of a life it has gone every open way
   of its land, where a perfect memory leaves 7-21% never gone. Worth most in a maze (x2.4-4.9),
   least in an open land. (True of memories that do not seek; see 7.)
3. What ends a thought carries weight: one bit of slack in the match and 70 lives of 80 lock.
4. GROWTH PAYS, up to a knee: doubling the cells pays x1.2-5.8 while the brain is small for its
   land and x0.9-1.2 once it is not. The knee sits where cells are about one to two times the
   land's ways (side 6: 256 cells for 144 ways; side 8: 256 for 256; side 10: 512-1024 for 400;
   side 12: 512 to 1024 cells for 576 ways still pays x1.07-1.37, B13: the knee is softer in a
   bigger land). Below the knee every lineage gains; the weakest gains most. More cells do not
   make a young brain know its land sooner: the gain is in the later thirds of a life. A maze of the same side does not move the knee as far as 16 lives can tell
   (B15: lives in a maze eat or starve, so a lineage's ratio there is known to 0.2-0.3).
5. The brain unlearns slowly: a quarter of the walls moved at half-life it takes about as well
   as a perfect memory; half moved, it falls to 0.37-0.55 of its unchanged last third (perfect
   0.65) though its map is again mostly right by the end. More cells do not mend it (B14: 0.35-
   0.49 at 512 cells). The curve of a life says what it costs: right after the change the brain
   eats an eighth to a fifth of what it did, and it climbs back no faster than a newborn learns
   a land. Nothing in the cell's rule was ever asked to forget. The looks of 2026-11-18 say
   what this is made of: a brain born again at the change eats as much in the half-life after
   it as one that kept its wires (the old map is a start and then a weight); undoing a way
   takes three or four walks where writing one takes two, so the rule is slower by a walk and
   not by ten; and no table does better (the explorer's last third: 60 of 115), because a table
   never doubts a wall. A memory that DOUBTS, no longer knowing a way it has not gone for 640
   ticks, and seeks, takes a land half moved almost whole (105 of 115) and pays one meal in a
   hundred where nothing changes.
6. Curiosity (with no road, the way the brain speaks most quietly of) helped when roads were
   rare (6c: x1.3) and adds 4-7% once a road is kept (B11). Imagery is mixed on every asking.
7. WANTING TO KNOW is where the meals are. A grown brain in a land that does not change eats
   0.85-0.91 of what one born knowing it eats; the gap is the first third of a life. A perfect
   memory that goes to the nearest way it has never gone eats 0.81-0.89 of born-knowing from
   birth (6g). The same wish in the brain (6h: a thought with no road leads to the nearest way
   a copy speaks quietly of) makes it get about its land nearly as fast as that explorer and
   is worth x1.09-1.16 at side 8, x1.07-1.17 at side 12 and x1.5-2.2 in a maze, where it also
   ends the last lock. It is worth so much less than in a table because a table knows a way
   it has gone once and the brain does not: gone once, a way is said right on 0.37-0.61 of the
   askings; gone twice, on 0.80-0.93. THE BINDING THING NOW IS HOW MANY WALKS WRITE A WAY.
8. ONE WALK CAN WRITE A WAY, once the rule is searched WITH THE BODY IT LIVES WITH (6i, 6j).
   The lineage came from the stream and was given its dendrite per way by hand, never having
   lived with it. Searched on for 4000 steps in a world of forks with that dendrite, all five
   lineages say a way gone once right on 0.80-0.94 of the askings, undo a way in one walk, and
   hold more (x1.1-1.3, four hops deep x1.3-1.7). I first put it down to the search showing a
   link once (the stream's seams had taught the rule not to believe that); the control of 6j,
   never shown a link once, does as well and better: 292-310 meals at side 8 on all five
   lineages (the explorer 312, born knowing 380), a last third of 125 on all five (born
   knowing 127), long trips straight 0.92-0.98 of the time. The lineages no longer differ.
   With the walk genomes: side 12 at 512 / 1024 cells 333-441 / 388-487 (explorer 517), the
   maze 152-205 (228); the knee stands where it stood. Seeking is worth less now (x1.05-1.12):
   when one walk writes, the dice find the ways. The price is in the old exam: in the stream,
   with no body, the new genomes lose the walk in the dark (and hold x2-3 the items in thought).
9. IN A LAND THAT CHANGES THE BRAIN WAS A TABLE THAT NEVER DOUBTS (6j): half the walls moved
   at half-life, it eats 158-215 a life where the explorer eats 213 and a memory that doubts
   275, and 72-91 against 135 in the half-life after the change. One walk undoes a way, so the
   rule is not what is slow; nothing sends the body to look.
10. DOUBT BY HAND WORKS ("Forgetting, step 5"): wires that fade slowly (0.0003 a tick), a line
   at 0.6 of the brain's own loudness under which a way is held for not known, and a thought
   that leaves out what is said under that line (`--sure`). Picked on other lands by a rule
   written beforehand, on the exam's lands it lifts the half-life after the change to 104-121
   meals on all five lineages (0.77-0.90 of the table that doubts), no life locks, each part
   is needed, and the brain is back at the doubter's rate 800 ticks after the change. It costs
   5-6% of a life where nothing changes (the table: 1%), and with a quarter of the walls moved
   it neither helps nor costs. All three parts are by hand.
11. THE SEARCH WITH THE BODY, pushed (6k, 6l): the control genomes are alike in every land (the
   maze 180-195 of the explorer's 228; side 12: 392-443 with 512 cells, 430-479 with 1024, of
   517), and need fewer cells (128 cells at side 8 eat 236-270). 8000 steps more add 0-4% at
   side 8 and 2-8% at side 12. Five random starts of five, no lineage behind them, give a
   brain that lives in a land (258-303 at side 8): the lineage is a head start worth 2-12%,
   not a need. In a big land what is left of the gap to born-knowing is the first third.
12. A THOUGHT WITH NO LIST OF ROOMS ("A thought as a wave", steps 1-3). One state of the tissue
   holds about 16 rooms at any size (a room's code is its look spread over the cells), and a
   life on four such waves fails (0.1-5 meals). What the tissue does well in the dark is WALK,
   and a thought of walks at random (dreams: K walks from the room the brain stands in, the
   soonest to come upon the wanted look gives its road) is enough to live on: 1024 dreams of 32
   steps eat 0.66-0.71 of the life on copies on all five lineages, no life locks, and the
   number of walks is a dial (64 / 256 / 1024 dreams: 0.39-0.45 / 0.54-0.58 / 0.66-0.71). A
   dream's reach is bought with walks, not with cells. Where it loses: its roads are 1.3-1.9
   times the shortest (1.05-1.10 with their loops cut), a young brain's dreams come upon the
   wanted look where it is not (0.39 of the ticks of the first third on false roads), and late
   in life every way "not known" a dream leads to is a walk's voice sunk in the dark.

**What is still by hand in the world**, with what it carries: a dendrite per way (the body's
sense; a search found a weak stand-in on one lineage of five, 4b); no learning in thought
(`Plan.still`); the will (the wanted look is told; with no road, dice, curiosity, or since 6h
the road to the nearest way a copy speaks quietly of); and the
whole of thinking: copies of the brain walked breadth-first on its wires, a limit of rooms
(4096), an exact match with the wanted look (load-bearing, 3 above), a road kept while it
comes true (load-bearing, 1 above). Found by search: the cell's rule, and on two lineages of
five in other exams the gate.

**Roads from here.** The order was chosen with the user on 2026-11-17 ("After the breadth"):
wanting to know, forgetting beside it, a thought as a wave later.
- Wanting to know: nearly done where nothing changes (6i, 6j). Left there: the first third in
  a big land, and the line of seeking set for the new genomes (0.6-0.7, not 0.4).
- Forgetting: doubt by hand works (10 above) and must be counted a lap, not a tick (step 7).
  The search with its parts in place failed once (step 6: an error that punishes a loud wrong
  word where anything may be wrong teaches a low voice, and a rule searched with fast fading
  leans on it) and is asked once more with the land's own kind of change and the land's own
  fading (step 6'). If that fails the three parts stay by hand.
- A thought as a wave, the third road of the order: a wave fails, dreams work (12 above).
  RUNNING (2026-11-22): dreams mended by hand, step 4 (loops cut, dreaming on with a road in
  hand, walks that end at what is not known), and dreams as they are in the bigger land. After
  it, what is still by hand in a dream: the dice of the walks, the walks all at once, the
  noting of the walk that met the look. And for the claim of the line: a code that gains
  dimensions with the cells, so that growth pays for a thought and not only for the map.
- Thinking into tissue: the procedure is now the largest hand-made part and the most
  load-bearing; what a brain without copies can do was asked once (5b-5d) on 32 rooms.
- The will into the brain: a fruit's look learned rather than told (hunger, reward).
- The genome searched in the world: no genome was ever selected on a life in a land.
- Larger lands past the knee: the knee says how many cells a land wants.

**Data for a reader.** `python -m tissue.ledger` writes `runs/tissue/world-ledger.csv`: one row
per record of exam 6 (genome, land, switches, size, meals, thirds, locked lives, straightness,
map, yardsticks of the same land). Records from 2026-11-17 hold the tick of every meal. Bars,
predictions and outcomes are in this note (Exam 6, steps 6a-6i, Breadth 1-5, "After the
breadth") and in the journal; wrong predictions are written as such. Two record files were lost to a name clash
and repeated; see the journal of 2026-11-16 and 17.

### Step 4b: the body's sense, asked of a search. Plan, bars, prediction, before the run

The first hand-made part on the world's list to go, if a search will have it. In 4a the way
gone on SENSORY cells of its own did nothing, and what worked was mine: a dendrite per way,
swapped by the examiner. Between the two lies the smallest thing a cell could be given:
`Plan.body`, BODY CELLS (one per way), each wired at birth to every interneuron by an
innate random wire, read by an interneuron on a dendrite of their own, one more number
beside its inputs. A cell then knows, by the luck of its wires, some ways better than
others, and what it does with that is its genome's. By hand: that the sense exists and
lands on a dendrite of its own. Not by hand any more: which wires speak on which way.

The search (`tissue.train --arm fork`): lives of exam 4 with two ways, 1/16 to 1/2 link per
interneuron, brains of 8 to 64 cells, closings of 1 to 8 ticks, started from the genomes
of 1d (whose new input begins at zero: a numb brain), 4000 steps, five lineages. It is
cheap (small brains, wired from every cell). Off-switch: the body cells silent
(`--body numb` on the found genome).

Bars, exam 4 with two ways at 64 to 512 cells (128 and up never seen by the search),
`--body felt`, against the same lineage with the hand-made dendrites (K90 at hop 1 at 256
cells 422 / 460 / 464 / 333 / 451, at hop 4 206 / 182 / 216 / 111 / 171):

- **B1, the cell uses its body.** K90 at hop 1 at 256 cells at least half the hand-made
  figure, on at least three lineages; with the body cells silent it holds no more than a
  numb brain does.
- **B2, and more of it buys more.** K90 at hop 1 at 512 cells at least 1.6 times that at
  256, on at least three.
- **B3, several hops.** K90 at hop 4 at 256 cells at least half the hand-made figure, on
  at least three.
- **B4, ways it was never searched on.** Grown with four body cells in a world of four
  ways: K90 at hop 1 at 256 cells at least what the same genome holds with two ways.

Prediction: B1 met (a hundred steps of a trial search already moved: accuracy 0.91 where a
numb brain starts near 0.8). (Corrected while the search was running and before any exam:
that reading was wrong. Measured, a numb brain scores 0.90 on these lives, 0.82 with the
eyes closed, so the trial said nothing and B1 is as open as the rest.) B2 met on three. B3 the doubt: a dendrite swapped by hand
keeps the ways wholly apart, a cell leaning by the luck of its wires keeps them apart
only roughly, and four hops multiply the leak. B4 missed: with four cells on one dendrite
a cell tells the ways apart worse.

If B1 and B3 are met the world of exam 6 is lived again with this brain in place of the
dendrite per way, and the list of hand-made parts is one shorter.

**Outcome (journal 2026-11-12, 18:10): all four bars missed.** One lineage of five (lineage
1) finds a use for its body: K90 at hop 1 26 / 57 / 108 / 216 links at 64 / 128 / 256 / 512
cells, twice as much at every doubling to sizes never searched, and nothing with the body
cells silent; but that is a quarter of the hand-made figure, 20 links at hop 2 and none at
hop 4, and 9 links with four ways. The other four stay numb. A hand-made guess at the
ceiling of a cell that leans to a way (`--body tribes`) holds less than the search found
(15-36 links at 256 cells), so it is no ceiling and that ceiling is not known. The
dendrite per way stays by hand. Not tried and cheap: many starts, longer searches, lives
that pay for depth, four ways in the search.

### Step 4b', asked again for depth. Bars written before the run

What 4b lacked, by its own lives: most of the error a numb brain makes there sits at hop 1
(closings of 1-8 ticks, half of them 1 or 2), brains of 8 cells live in worlds of two rooms,
and one start a lineage. Asked again: closings of 2 to 16 ticks (`--hops 16 --vary`), brains
of 16 to 64 cells, 8000 steps, and two arms:

- **(a) from the genome that found the first hop** (`runs/tissue/body/body-s1.pt`), four
  starts (seeds 510-513): can depth be asked for once the body is in use?
- **(b) from the genomes of 1d again**, one start each (seeds 520-524): do lives that pay
  for depth find the body more often?

Exam 4, two ways, `--body felt`, 64 to 512 cells. Bars:

- **D1.** Arm (a): at least two starts of four hold, at hop 4 at 256 cells, a quarter of what
  lineage 1 holds there with the hand-made dendrites (182, so 45 links). In 4b: none.
- **D2.** Arm (b): at least three lineages of five use their body (K90 at hop 1 at 256 cells
  at least 50 links, and nothing with the body cells silent). In 4b: one.

Prediction: both missed. If a cell with one set of wires keeps the ways apart only by which
cells speak, a second hop has to tell apart rooms entered by different ways as well, and
asking harder does not give it the wires for that. I would like to be wrong.

**Outcome (journal 2026-11-12, 19:45): both missed, as predicted.** (a) all four starts end
in one place: at 256 cells 12-13 links at hop 4 (none before), 26-27 at hop 2 (20 before)
and 71-78 at hop 1 (108 before), and it no longer doubles with the cells. Depth is bought
with the first hop. (b) one lineage of five uses its body, the same one (lineage 1), and
lands where (a) did. So it is the genome a search starts from and not the dice of the
search, and a cell with one set of wires trades the first hop against the later ones. The
dendrite per way stays by hand; a next asking would be of another kind (branches of a
dendrite as units with a rule of their own, as the sleep cell was for the gate).

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
| 2c cells in a space | done, 5 lineages. Wires from neighbours alone change nothing. A search given a sense of place found no use for it. With a gate BY HAND (cells silent away from their place) and 32 wires per cell: recall one step ahead in proportion to the cells, 0.4 items per cell to 4096; a thought held 24-30 items at 256 cells and 160-287 at 4096 (placeless: 18 at any size). Shuffle the wires: nothing. Sense off: a placeless brain. Open: the search finding the gate; the dials in the genome |
| 2c-6 a search living with the wider field | done: holds less, not more (T1, T2 missed); the genomes of 2c-3b stay the best with a gate |
| 3a growth within one life, the size turned by the examiner | done, 5 lineages, no search: a brain given new cells in mid-life (512, then 2048) is within 0.02 of one born with 2048 on old and new loops alike, and 0.16-0.20 above one left at 512 |
| 2d the search finds the gate | not found. First attempt: with the hand gate taken away the genomes go back to placeless brains (the lives did not make the gate pay). Second attempt (the cell taught its gate by imitation, lives where places pay): two lineages of five keep a gate of their own and one of them holds more as it grows, to 2048 cells; but the gate leaks (far cells 0.26-0.40 as loud) and the bars are missed. The gate stays by hand |
| 2e the brain chooses the place | by hand so far, nothing searched. Told only with the eyes open: the thought keeps its place for a lap, then the other cells join in. With a hand-made index of what lies where (`hand.Index`) the place need not be told when a thing is met again: recall 0.82-0.88 of the world-told figure, holding the same. Never told at all, new things in a row sharing a place and each new place opened far from the last: recall 0.6-0.8 of the world-told figure and holding x1.3-3.9 of it (63-86 items at 512 cells, 269-308 at 2048, 573-627 at 4096: in proportion to the cells) |
| 3b growth decided by the tissue | done by hand, 5 lineages, nothing searched and nothing told: places chosen by the brain, cells added when every stretch of the ring is in use. It grows x3 in a poor world and x8 in a rich one; in the rich world it is level with a brain born big for four fifths of the cost, in the poor one 0.06-0.07 under it for a third of the cost |

| 2f the gate as a cell (sleep cells) | by hand: a sleep cell set to the hand gate IS the hand gate at every size, provided it speaks within the tick (heard a tick late: 0.54-0.63 of the recall). Searched with the hand gate as a scaffold taken away: two lineages of five find a sharp gate of their own that holds to 4096 cells (K90 1221 and 1154, 0.7-0.8 of the hand gate's; 330 and 289 items held, more than the hand gate's); searched with no scaffold: none of five. W1, W2 missed on the count of lineages |
| 4a forks: the dark walk goes where the body goes | by hand, nothing searched: a dendrite per way. Lineage 0 holds 98 and 422 links at 64 and 256 cells with two ways (hop 4: 57 and 206); with the body on sensory cells of its own, or numb, all five lineages hold nothing. Five lineages to 512 cells: running |
| 5a a goal: thinking by hand on a learned map | done by hand, 5 lineages, nothing searched: a copy of the brain walks every string of four ways in the dark; wanted rooms 1 to 4 steps away are reached by a shortest road 1.00 / 0.94-0.97 / 0.92-0.98 / 0.92-0.96 of the time (not thinking: 0.53 / 0.28 / 0.17 / 0.15; thinking but numb: the same). The search through the map is by hand |
| 5a in a crowded world | 128 rooms (256 links): at 256 cells 0.21-0.48 at distance 4, at 512 cells 0.43-0.94. A thought is as good as the map and the map grows with the brain, but at least 0.9 at every distance is reached by one lineage of five (G4 missed) |
| 5b, 5c thinking with no copy | when the brain walks in thought itself and keeps what thought taught it the map wears out (0.40-0.59 at distance 4, worse in the second half of a life). The learning does all of the harm and the state none: with its wires still in thought the brain thinks in its own head as well as a copy (0.94-0.99 at distance 4), coming back by one look at the room |
| 5d wires still in the dark, in the kernel (`Plan.still`, by hand) | thinking in its own head: met on four lineages. On the loops of exam 2 recall is untouched and holding is 0.80-0.98 of what it was: there the learning in the dark was mending a little. Open: a rule that tells rehearsing from imagining; the search through the map as cells |
| 6a a world: a walled land, trees, one life with no study | done by hand, 5 lineages, nothing searched. Born into a land it has never seen and told only what the room with the fruit looks like, a brain of 256 cells learns 36 rooms by wandering and eats 86-130 times a life (a perfect memory with the same will 121, born knowing the land 259, a blind walker 5); numb: a blind walker. In the last third of a life trips of 3-4 steps are straight 0.98-1.00. At 100 rooms 256 cells eat 10-25 and 512 cells 17-73 (2.5-4.4 times as much in the last third). V2, V3 met; V1 missed by three lineages a few hundredths short; V4 met in its first half. What binds is the depth of a thought |
| 6b a wider thought (4096 rooms a thought in place of 256, side 8) | done, T1-T3 missed. Three lineages of five eat 1.4-1.7 times as much (163 / 69 / 123 / 51 / 62 against 95 / 55 / 88 / 53 / 36), long trips are no straighter (0.10-0.49), and 16 lives of 80 stop eating for good: a thought that always finds a road passes the body to and fro between two rooms. 1024 rooms is the better dial (139 / 87 / 103 / 68 / 62) |
| 6c curiosity in place of the dice (`--curious 0.7`, by hand: with no road, the way the brain speaks most quietly of) | done, C1, C2 met, C3 missed. 1.3-1.4 times the meals on four lineages of five (124 / 79 / 99 / 71 / 47), 0.58-0.62 of the ways gone by an eighth of a life in place of 0.47-0.51; the best is 0.59 of `seeker`. With 1024 rooms a thought as well: 173 / 97 / 123 / 71 / 80, and as many locked lives as before |
| 6d the land of side 10 with both dials (512 cells) | done, L1 missed, L2 and L3 met: with curiosity and 1024 rooms a thought 167 / 92 / 109 / 49 / 71 meals where 6a gave 73 / 39 / 42 / 19 / 17; 1024 rooms alone does as much (162 / 79 / 135 / 56 / 64), curiosity alone little (x1.2-1.8) |
| 6e a road kept (`--keep`, by hand: a road thought of is walked while every room is as a copy said) | done, K1-K4 met. Side 8, 4096 rooms: 241 / 212 / 214 / 193 / 160 meals (6b: 163 / 69 / 123 / 51 / 62), no life locked, every lineage at or above 0.88 of `perfect` and three at `seeker` or above |
| 6f growth in a world (side 10, a road kept; 256, 512, 1024 cells) | done, G1 and G4 met, G2 and G3 missed by little. 238 / 173 / 223 / 151 / 99, then 301 / 267 / 289 / 224 / 193, then 324 / 290 / 308 / 253 / 234 meals: every lineage gains at every doubling, x1.27-1.95 then x1.06-1.21 |
| breadth 1-4 (other lands, slack, a land that changes, the knee, old switches with a road kept) | the brain beats a perfect memory most in a maze (x2.4-4.9); one bit of slack locks 70 lives of 80; half the walls moved at half-life is unlearned slowly (0.37-0.55 of the last third); growth flattens where cells are one to two times the land's ways (sides 6, 8, 10, 12) |
| 6g, 6h wanting to know (yardstick `explorer`; `--explore 0.4`, by hand) | done. A perfect memory that seeks the nearest way it has never gone eats 0.81-0.89 of born-knowing: the yardstick from here on. The brain given the same wish gets about its land nearly as fast and eats x1.09-1.16 at side 8, x1.07-1.17 at side 12, x1.5-2.2 in a maze (H2, H3, H4, H6 met; H1, H5 missed), because a way gone once is said right on 0.37-0.61 of the askings |
| 6i one walk writes a way (search arm `walk`: forks with a dendrite per way, a link shown once) | done, W1-W5 met. Gone once a way is said right 0.80-0.93; side 8: 264-311 meals (the explorer 312), side 12 at 512 cells: 333-441 (517); all 160 lives eat more than with the lineage genome. The prediction asked first was wrong (the lineage genomes hold half with one lap, not a quarter). The control and the old exam: 6j |
| 6j the control of 6i, the stream, the new genomes in the old lands | done; J1, J4, J5, J7 met, J2, J3 missed, J6, J8, J9 in part. The control (never shown a link once) writes a way in one walk as well and eats 292-310 at side 8 on all five lineages with a last third at born-knowing: the gain of 6i came from searching the rule with the dendrites, not from the one lap. In the stream the new genomes lose the dark walk and hold x2-3 the items. In a land half moved the brain is a table that never doubts (158-204 against the doubter's 275) |
| forgetting 5: doubt by hand (`--fade`, the line `--explore 0.6`, `--sure`) | done; D1, D3, D4 met, D2 missed by a hair, D5 level. Half the walls moved: 104-121 meals in the half-life after the change on all five lineages (the base 72-91, the table that doubts 135), no life locked, each part needed; 5-6% of a life where nothing changes |
| 6k, 6l the control genomes everywhere; more search | done. The control genomes are alike in every land and lose the dark walk of the stream as the walk genomes do (it is the dendrite); the knee as before, with fewer cells needed. 8000 steps more: 0-8% more meals. Five random starts of five live in a land (258-303 at side 8): the lineage is a head start, not a need |
| forgetting 6: the rule searched with fading wires in forks whose ways change | failed, S2-S4 missed (S1, S5 met by the letter). The search learned a low voice for everything, not doubt of what is old, and three lineages of five eat a quarter to a half less in a land than the genomes they started from. Asked again (6') with the land's own kind of change and fading |
| forgetting 7: doubt by hand at slower fadings and in a bigger land | done, predictions missed. Slower fading buys back little of the cost. At side 12 the same fading a tick costs 12-74% of a life: doubt must be counted a lap (the table that doubts says the same: T 640 is worse than no doubt there, T 1440 right) |
| forgetting 8: doubt by hand counted a lap | done. Side 12 with 512 cells at 0.08 a lap: 0.90-1.00 of the base where nothing changes, x1.4-1.8 in the half-life after half the walls moved, no life locked. A maze that changes: x1.5-4.0, a quarter of the table that doubts. Step 6' (the search asked again) not run, decided by a count: the three parts stay by hand |
| wave 1: one state for many rooms (a look by hand) | done. A room has one code seen or thought of; one state holds about 16 rooms AT ANY SIZE (a room's code is its look spread over the cells); a wave kept clear of where it has been gives the first way 3 steps out on 0.8 of the askings, 4 steps 0.5-0.75, and little beyond. Growth does not pay for a wave in this code |
| wave 2: a life on four waves (`--wave`) | failed at the pilot: 0.1-5 meals a life (a blind walker 6). A wave reaches 3-4 steps, too short to live on, or the body is passed to and fro |
| wave 3: thinking as dreams (`--dreams`) | done, R1-R3 met on all five lineages. K walks in the dark by ways at random, the soonest to come upon the wanted look gives its road, no list of rooms: 1024 dreams of 32 steps eat 208 / 212 / 212 / 192 / 219 (0.66-0.71 of the life on copies), no life locked; 64 / 256 / 1024 dreams a dial (x1.3-1.4, x1.2). Reach is bought with walks, not with cells |
| wave 4: dreams mended (`--cut`, `--again`, `--sure` whole) | running (2026-11-22), bars M1-M6. A look says where dreams lose: roads 1.3-1.9 times the shortest, false roads in a young brain, false leads later |
| 4b the body's sense asked of a search (`Plan.body`) | one lineage of five finds a use for it (108 links at 256 cells, a quarter of the hand-made dendrites, twice as much at every doubling, one hop deep); B1-B4 missed. The dendrite per way stays by hand |

Open questions carried along: what separates the lineages that scale without a fade from
those that do not (lineage 3 is the weakest on every exam); why a little idleness seemed
to help hop 4; whether a thought survives a world with forks, where an item has more than
one successor.
