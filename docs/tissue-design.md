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
| 1 | recurrence: the answer is fed back | chains A->B, B->C, asked A->C: ticks to answer vs chain length | more ticks do not buy longer chains |
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
