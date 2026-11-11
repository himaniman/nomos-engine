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

**Outcome (journal 2026-11-11):** B2 and B3 met on all seeds, about one pair per
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

**Outcome (journal 2026-11-11):** one seed of three found the hop; two settled on a
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

**Outcome (journal 2026-11-11, night):** D1, D2, D3 met on five seeds of five, the
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

**Outcome (journal 2026-11-11, night):** E1 missed by one seed of five (0.913; the others
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

**Outcome (journal 2026-11-12):** three seeds of five found the hop; two settled on a code
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

**Outcome (journal 2026-11-12):** no lineage got stuck, which was the point. By the letter
both bars are missed: G1 by two lineages that sit 0.002 and 0.005 under hop 4 = 0.90 (hop
1 is 0.986-0.996 on all five), G2 by two lineages (x5.0 and x6.0 against x8.9-9.5 on the
other three). Growth without a fade is frequent, 5 of the 8 genomes that live in the
stream, not the rule. On loops of 16, never seen, with the eyes closed 12 ticks, hop 12 is
0.67-0.79 at 64 cells: the clean figure for a long closing.

**Kept as method:** a search for a new life starts from the genome of the life before.

## A thought that holds (chosen with the user, 2026-11-12)

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

**Outcome (journal 2026-11-12, night):** H1, H2, H3 met on all five lineages; my
prediction was wrong on every count. No new mechanism was needed: asked, the brain stops
whispering (loudness 0.57 through the closing, pattern 0.98 at every lap), loses no
capacity, and the best lineage walks its loop for 256 ticks, eight times the longest
closing it was searched on. Items held in thought grow x6-7 from 64 to 512 cells.

**Kept as method:** ask before adding a mechanism. The private state of a cell stays
unbuilt until an exam needs it.

## Where the ladder stands

| rung | state |
|---|---|
| 0 pairs | done, 3 seeds: about 1 pair per interneuron to 512 |
| 1a one hop through own wires | done, 5 seeds: 0.7 links per interneuron at 64, fading to 0.35 at 512 |
| 1b search on chains | done, 5 seeds: links walked 4 hops doubled (17-19 at 64 cells), depth carries to 8 hops; size still does not buy depth |
| 1c the stream: no resets, idleness | done: 8 genomes live an unbroken life (3 of 5 random starts, 5 of 5 lineages); on 5 of them capacity keeps pace with size to 512 (0.7 items per interneuron); idleness costs nothing; a walk holds 8-16 ticks, then drifts |
| 1d a thought that holds | done, 5 lineages, no new mechanism: a stable cycle through learned items for 64 ticks on all five and 256 on the best; items held in thought x6-7 from 64 to 512 cells; capacity kept |
| own pace; a world with forks (a choice, then a goal) | open, the user decides the order |
| 2 wiring from the genome | not started |
| 3 growth from a seed | not started |

Open questions carried along: what separates the lineages that scale without a fade from
those that do not (lineage 3 is the weakest on every exam); why a little idleness seemed
to help hop 4; whether a thought survives a world with forks, where an item has more than
one successor.
