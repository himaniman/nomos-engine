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

| 2f the gate as a cell (sleep cells) | by hand: a sleep cell set to the hand gate IS the hand gate (191 against 193 items at 512 cells, lineage 0) provided it speaks within the tick; heard a tick late it loses nearly half the recall and four fifths of the holding. The searches (with a scaffold, and with none) are running |
| 4a forks: the dark walk goes where the body goes | by hand, nothing searched: a dendrite per way. Lineage 0 holds 98 and 422 links at 64 and 256 cells with two ways (hop 4: 57 and 206); with the body on sensory cells of its own, or numb, all five lineages hold nothing. Five lineages to 512 cells: running |
| 5a a goal: thinking by hand on a learned map | done by hand, 5 lineages, nothing searched: a copy of the brain walks every string of four ways in the dark; wanted rooms 1 to 4 steps away are reached by a shortest road 1.00 / 0.94-0.97 / 0.92-0.98 / 0.92-0.96 of the time (not thinking: 0.53 / 0.28 / 0.17 / 0.15; thinking but numb: the same). The search through the map is by hand |
| 5a in a crowded world | 128 rooms (256 links): at 256 cells 0.21-0.48 at distance 4, at 512 cells 0.43-0.94. A thought is as good as the map and the map grows with the brain, but at least 0.9 at every distance is reached by one lineage of five (G4 missed) |
| 5b, 5c thinking with no copy | when the brain walks in thought itself and keeps what thought taught it the map wears out (0.40-0.59 at distance 4, worse in the second half of a life). The learning does all of the harm and the state none: with its wires still in thought the brain thinks in its own head as well as a copy (0.94-0.99 at distance 4), coming back by one look at the room |
| 5d wires still in the dark, in the kernel (`Plan.still`, by hand) | thinking in its own head: met on four lineages. On the loops of exam 2 recall is untouched and holding is 0.80-0.98 of what it was: there the learning in the dark was mending a little. Open: a rule that tells rehearsing from imagining; the search through the map as cells |

Open questions carried along: what separates the lineages that scale without a fade from
those that do not (lineage 3 is the weakest on every exam); why a little idleness seemed
to help hop 4; whether a thought survives a world with forks, where an item has more than
one successor.
