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
        _, wires, _ = tissue.grow(m).born(2, gen())
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
    act, wires, _ = brain.rest(state)
    assert not any(a.any() for a in act.values())
    assert wires[1].any() and torch.equal(wires[1], state[1][1])


# ---- the stream: one unbroken life ---------------------------------------------------------

from tissue import stream                                           # noqa: E402


def test_a_life_knows_how_many_ticks_it_is():
    life = stream.Life(loops=3, length=4, laps=2, idle=5, blinks=2, open=3, closed=4)
    assert len(stream.script(life, 2, gen(0))) == life.ticks == 2 * 3 * 5 + 5 + 2 * 7


def test_the_script_walks_every_loop_round_and_is_the_same_for_any_brain():
    life = stream.Life(loops=3, length=4, laps=2, idle=5, blinks=4, open=3, closed=2)
    ticks = stream.script(life, 6, gen(1))
    again = stream.script(life, 6, gen(1))
    assert len(ticks) == 2 * 3 * 5 + 5 + 4 * (3 + 2)
    assert all(torch.equal(a[2], b[2]) and torch.equal(a[3], b[3]) for a, b in zip(ticks, again))
    kinds = [t[0] for t in ticks]
    assert kinds.count("study") == 30 and kinds.count("idle") == 5 and kinds.count("closed") == 8
    assert all(seen == (kind in ("study", "echo")) for kind, seen, _, _ in ticks)
    # a study visit starts somewhere and comes back to it after `length` steps
    assert torch.equal(ticks[0][3], ticks[4][3]) and torch.equal(ticks[0][2], ticks[4][2])
    # within a visit and through closed eyes the world only ever steps to the next place
    for a, b in zip(ticks[30:34], ticks[31:35]):
        assert torch.equal(b[3], (a[3] + 1) % 4) and torch.equal(a[2], b[2])
    # every loop is visited once per lap in every life
    first_lap = torch.stack([ticks[i * 5][2] for i in range(3)], 1)
    assert all(sorted(row.tolist()) == [0, 1, 2] for row in first_lap)


def test_a_perfect_memory_of_the_stream_is_right_at_every_hop_and_through_idleness():
    life = stream.Life(loops=4, length=4, laps=2, idle=12, blinks=6, open=2, closed=5)
    items = stream.world(ITEM, life, 8, gen(2))
    out = stream.ideal(items, life, gen(3))
    assert stream.by_hop(out, life) == [1.0] * 5
    assert stream.acc(out, "echo") == 1.0 and stream.idle_curve(out, 4) == [1.0] * 3


def test_a_stream_is_lived_without_a_reset_and_is_a_pure_function():
    torch.manual_seed(0)
    tissue = Tissue(RUNG1, ITEM)
    life = stream.Life(loops=2, length=4, laps=1, idle=3, blinks=3, open=2, closed=2)
    items = stream.world(ITEM, life, 3, gen(4))
    out = stream.run_life(tissue.grow(16), ITEM, items, life, gen(5))
    assert out["closed"][0].shape == (3, 6, 12) and out["echo"][0].shape == (3, 6, 12)
    assert out["idle"][0].shape == (3, 3, 12)
    same = stream.run_life(tissue.grow(16), ITEM, items, life, gen(5))
    assert torch.equal(out["closed"][0], same["closed"][0])
    # the targets are the world's own items, whoever lives the life
    told = stream.ideal(items, life, gen(5))
    assert torch.equal(out["closed"][1], told["closed"][1])
    assert torch.equal(out["echo"][1], told["echo"][1])
    assert len(stream.by_hop(out, life)) == 2


# ---- rung 2: a few wires per cell ---------------------------------------------------------


def test_sparse_wiring_keeps_a_few_wires_per_cell_and_only_those_learn():
    torch.manual_seed(0)
    plan = replace(RUNG1, fan=5)
    brain = Tissue(plan, ITEM).grow(32)
    act, wires, sources = brain.born(3, gen(1))
    assert wires[1].shape == (3, 32, 5) and sources[1].shape == (3, 32, 5)
    assert 0 <= int(sources[1].min()) and int(sources[1].max()) < 32
    assert wires[0].shape == (3, 32, 12) and wires[2].shape == (3, 12, 32)   # senses and motor: all
    cue = torch.ones(3, 12)
    state = (act, wires, sources)
    state, _ = brain.tick(state, cue, torch.zeros(3, 12), 0.0, 1.0)
    state, _ = brain.tick(state, -cue, cue, 1.0, 1.0)
    assert state[1][1].shape == (3, 32, 5) and state[1][1].any()
    assert torch.equal(state[2][1], sources[1])            # the wiring itself never changes
    # a brain smaller than the fan is simply wired from every cell
    assert Tissue(plan, ITEM).grow(4).born(2, gen(1))[1][1].shape == (2, 4, 4)


def test_a_sparse_brain_is_the_same_genome_and_lives_the_same_lives():
    torch.manual_seed(0)
    dense, few = Tissue(RUNG1, ITEM), Tissue(replace(RUNG1, fan=8), ITEM)
    few.load_state_dict(dense.state_dict())                 # a genome does not know its fan
    life = stream.Life(loops=2, length=4, laps=2, idle=4, blinks=2, open=2, closed=2)
    items = stream.world(ITEM, life, 3, gen(4))
    out = stream.run_life(few.grow(32), ITEM, items, life, gen(5))
    assert out["closed"][0].shape == (3, 4, 12)
    assert torch.equal(out["closed"][0], stream.run_life(few.grow(32), ITEM, items, life, gen(5))["closed"][0])
    walked = chain.run_life(few.grow(32), ITEM, chain.worlds(ITEM, 3, 2, 2, gen(6)), gen(7))
    assert walked.shape == (2, 3, 3, 12)


def test_a_thought_held_by_hand_follows_the_wires_and_falls_under_a_heavy_load():
    light = hand.think(64, 16, 0.5, 8, signed=True, steps=200)
    heavy = hand.think(64, 16, 0.5, 64, signed=True, steps=200)
    assert light > 0.95 > 0.9 > heavy
    assert light == hand.think(64, 16, 0.5, 8, signed=True, steps=200)      # a pure function
    assert 0.0 <= hand.think(64, 16, 0.125, 8, steps=50) <= 1.0             # few cells speaking


def test_wires_within_reach_come_from_neighbours_on_a_ring_and_never_from_the_cell_itself():
    brain = Tissue(replace(RUNG1, fan=6, reach=3), ITEM).grow(40)
    _, wires, sources = brain.born(4, gen(2))
    away = (sources[1] - torch.arange(40).view(1, -1, 1)) % 40
    away = torch.minimum(away, 40 - away)                   # distance round the ring
    assert wires[1].shape == (4, 40, 6)
    assert int(away.min()) >= 1 and int(away.max()) <= 3
    assert all(len(set(cell.tolist())) == 6 for cell in sources[1][0])   # six different neighbours
    # reach 0 is the shuffle: the same genome and seed give the wiring of a brain with no space
    plain = Tissue(replace(RUNG1, fan=6), ITEM).grow(40).born(4, gen(2))[2][1]
    assert torch.equal(plain, Tissue(replace(RUNG1, fan=6, reach=0), ITEM).grow(40).born(4, gen(2))[2][1])
    assert int((plain - torch.arange(40).view(1, -1, 1)).abs().max()) > 3


def test_a_sense_of_place_is_a_bell_round_the_place_and_silent_when_nothing_is_felt():
    brain = Tissue(replace(RUNG1, fan=6, reach=3, field=4), ITEM).grow(40)
    near = brain.near(40, torch.tensor([0.0, 0.5]))
    assert near.shape == (2, 40)
    assert near[0, 0] == 1.0 and near[1, 20] == 1.0
    assert float(near[0, 39]) == pytest.approx(float(near[0, 1]))       # the ring closes
    assert float(near[0, 20]) < 1e-5 and float(near[0, 4]) == pytest.approx(0.6065, abs=1e-3)


def test_a_genome_adopted_into_a_plan_with_a_new_sense_begins_as_the_old_one_lived():
    from tissue.train import adopt
    torch.manual_seed(0)
    old = Tissue(replace(RUNG1, fan=6), ITEM)
    new = Tissue(replace(RUNG1, fan=6, field=4), ITEM)
    assert new.cells["I"].a.weight.shape[1] == old.cells["I"].a.weight.shape[1] + 1
    adopt(new, old.state_dict())
    life = stream.Life(loops=2, length=4, laps=2, idle=4, blinks=2, open=2, closed=2)
    items = stream.world(ITEM, life, 3, gen(4))
    was = stream.run_life(old.grow(32), ITEM, items, life, gen(5))["closed"][0]
    now = stream.run_life(new.grow(32), ITEM, items, life, gen(5))["closed"][0]
    assert torch.allclose(was, now, atol=1e-5)              # the same up to the order of a sum
    felt =stream.run_life(new.grow(32), ITEM, items, life, gen(5), places=True)["closed"][0]
    assert felt.shape == was.shape
    assert torch.equal(felt, stream.run_life(new.grow(32), ITEM, items, life, gen(5), places=True)["closed"][0])
    with pytest.raises(ValueError):
        adopt(old, new.state_dict())                        # a sense cannot be taken away so
