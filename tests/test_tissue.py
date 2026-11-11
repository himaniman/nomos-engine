"""tissue: the exam is honest and the kernel does what its docstring says."""

import pytest

torch = pytest.importorskip("torch")

from tissue import hand                                             # noqa: E402
from tissue.assoc import Task, accuracy, k90, k_grid, lives, run_life   # noqa: E402
from tissue.kernel import Gru, Plan, Tissue                         # noqa: E402


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_lives_are_distinct_cues_and_a_pure_function_of_the_seed():
    task = Task()
    cues, answers = lives(task, 40, 6, gen(3))
    again, _ = lives(task, 40, 6, gen(3))
    assert torch.equal(cues, again)
    assert cues.shape == (6, 40, task.d_in) and answers.shape == (6, 40, task.d_out)
    for life in cues:
        assert len({tuple(c.tolist()) for c in life}) == 40
    assert set(cues.unique().tolist()) == {-1.0, 1.0}


def test_k90_interpolates_the_first_crossing():
    assert k90([1, 2, 4, 8], [1.0, 1.0, 0.95, 0.85]) == pytest.approx(4 * 2 ** 0.5)
    assert k90([1, 2], [0.8, 0.7]) == 0.0
    assert k90([1, 2, 4], [1.0, 1.0, 0.99]) == 4.0


def test_k_grid_starts_at_one_pair_and_stops_at_eight_per_interneuron():
    ks = k_grid(64, Task())
    assert ks[0] == 1 and ks[-1] == 512 and ks == sorted(set(ks))
    assert k_grid(4096, Task())[-1] == 2 ** Task().d_in


def test_hand_memories_recall_a_light_load_and_the_ceiling_fills_its_rank():
    task = Task()
    cues, answers = lives(task, 1, 16, gen(1))
    assert accuracy(hand.direct(cues, answers, None), answers) == 1.0
    cues, answers = lives(task, 24, 8, gen(2))
    c = hand.codes(cues, 32, 0.5, gen(2))
    assert accuracy(hand.lstsq(c, answers), answers) > 0.99


def test_frozen_wires_know_nothing_about_the_answers():
    """The control arm: with learning off, what is said cannot depend on what was taught."""
    task = Task()
    cues, answers = lives(task, 6, 4, gen(5))
    brain = Tissue(Plan(learn=False), task).grow(16)
    a = run_life(brain, task, cues, answers, gen(7))
    b = run_life(brain, task, cues, -answers, gen(7))
    assert torch.equal(a, b)


def test_a_life_is_a_pure_function_of_genome_and_seed_and_teaching_reaches_the_answer():
    task = Task()
    torch.manual_seed(0)
    tissue = Tissue(Plan(), task)
    cues, answers = lives(task, 6, 4, gen(5))
    a = run_life(tissue.grow(16), task, cues, answers, gen(7))
    assert torch.equal(a, run_life(tissue.grow(16), task, cues, answers, gen(7)))
    assert not torch.equal(a, run_life(tissue.grow(16), task, cues, -answers, gen(7)))
    assert a.shape == answers.shape


def test_the_genome_is_the_same_size_at_every_brain_size():
    tissue = Tissue()
    n = sum(p.numel() for p in tissue.parameters())
    assert n < 200
    for m in (8, 512):
        _, wires = tissue.grow(m).born(2, gen())
        assert wires[0].shape == (2, m, Task().d_in) and wires[1].shape == (2, Task().d_out, m)
        assert not wires[1].any()
    assert sum(p.numel() for p in tissue.parameters()) == n


def test_the_ordinary_net_lives_the_same_exam():
    task = Task(dwell=1)
    cues, answers = lives(task, 5, 3, gen(1))
    assert run_life(Gru(task, 8), task, cues, answers, gen(2)).shape == answers.shape


# ---- rung 1: links and chains ------------------------------------------------------------

from tissue import chain                                            # noqa: E402
from tissue.kernel import RUNG1                                     # noqa: E402
from dataclasses import replace                                     # noqa: E402

ITEM = Task(d_in=12, d_out=12)


def test_chain_worlds_hold_distinct_items_and_fit_the_item_space():
    items = chain.worlds(ITEM, 5, 3, 4, gen(1))
    assert items.shape == (4, 5, 4, 12)
    for life in items.reshape(4, 20, 12):
        assert len({tuple(i.tolist()) for i in life}) == 20
    with pytest.raises(ValueError):
        chain.worlds(ITEM, 2049, 1, 1, gen(1))
    assert chain.k_grid(64, ITEM, 4)[0] == 4 and chain.k_grid(64, ITEM, 4)[-1] <= 256
    assert all(k % 4 == 0 for k in chain.k_grid(64, ITEM, 4))


def test_the_teacher_never_says_what_follows_what():
    """The control arm of exam 1. With the wires between interneurons frozen, once the
    senses fall silent no cell can tell which item the walk began from: every walk in a
    life says exactly the same thing. So a right answer can only come through those wires.
    (Motor wires are held still outside teaching here, or their drift during the test,
    which knows nothing of the links either, would blur the equality.)"""
    torch.manual_seed(0)
    brain = Tissue(replace(RUNG1, off=(1,), taught_only=True), ITEM).grow(16)
    items = chain.worlds(ITEM, 6, 2, 3, gen(2))
    said = chain.run_life(brain, ITEM, items, gen(3))
    assert said.shape == (3, 6, 3, 12)         # the echo, then two hops
    for c in range(1, 6):                           # hop 0 is the echo: it may differ
        assert torch.equal(said[:, c, 1:], said[:, 0, 1:])
    live = chain.run_life(Tissue(RUNG1, ITEM).grow(16), ITEM, items, gen(3))
    assert not torch.equal(live[:, 1, 1:], live[:, 0, 1:])


def test_a_walk_is_a_pure_function_and_reads_as_many_hops_as_asked():
    torch.manual_seed(0)
    tissue = Tissue(RUNG1, ITEM)
    items = chain.worlds(ITEM, 3, 4, 2, gen(4))
    a = chain.run_life(tissue.grow(16), ITEM, items, gen(5))
    assert a.shape == (2, 3, 5, 12)                 # the echo (hop 0), then hops 1..4
    assert torch.equal(a, chain.run_life(tissue.grow(16), ITEM, items, gen(5)))
    assert chain.run_life(tissue.grow(16), ITEM, items, gen(5), hops=2).shape == (2, 3, 3, 12)
    assert len(chain.by_hop(a, items)) == 5


def test_the_hand_made_hop_recalls_a_light_load():
    items = chain.worlds(ITEM, 3, 1, 8, gen(6))
    said = hand.hop(items, 256, 0.1, gen(7))
    assert min(chain.by_hop(said, items)) > 0.95            # the echo and the hop
    assert min(chain.by_hop(hand.hop(items, 256, 0.1, gen(7), exact=True), items)) > 0.95


def test_rest_silences_cells_and_keeps_wires():
    torch.manual_seed(0)
    brain = Tissue(RUNG1, ITEM).grow(8)
    state = brain.born(2, gen(1))
    cue = torch.ones(2, 12)
    state, _ = brain.tick(state, cue, torch.zeros(2, 12), 0.0, 1.0)
    state, _ = brain.tick(state, -cue, cue, 1.0, 1.0)
    act, wires = brain.rest(state)
    assert not any(a.any() for a in act.values())
    assert wires[1].any() and torch.equal(wires[1], state[1][1])
