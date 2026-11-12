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


def test_a_dendrite_scaled_by_what_was_said_does_not_care_how_many_sources_are_silent():
    torch.manual_seed(0)
    genome = Tissue(replace(RUNG1, spoken=True), ITEM)
    cue, teach = torch.ones(2, 12), -torch.ones(2, 12)

    def motor_reads(m, speaking):
        brain = genome.grow(m)
        act, wires, sources = brain.born(2, gen(1))
        act["I"][:, :speaking] = 1.0                        # the first cells speak, the rest are silent
        wires[2][:, :, :speaking] = 0.5                     # and the motor wires from them are set
        seen = []
        hook = genome.cells["O"].register_forward_hook(lambda mod, inp, out: seen.append(inp[0]))
        brain.tick((act, wires, sources), cue, teach, 1.0, 1.0)
        hook.remove()
        return seen[0][..., 0]                              # the motor cells' dendrite from interneurons

    small, big = motor_reads(16, 8), motor_reads(64, 8)
    assert torch.allclose(small, big)                       # eight speakers in 16 cells or in 64
    assert torch.allclose(small, torch.full_like(small, 8 * 0.5 / (8 + 1)))


def test_a_thought_held_by_hand_in_places_grows_with_the_ring():
    light = hand.think_in_places(128, 16, 16, reach=16, patch=32, steps=200)
    assert light > 0.95
    assert light == hand.think_in_places(128, 16, 16, reach=16, patch=32, steps=200)
    crowded = hand.think_in_places(128, 16, 64, reach=16, patch=32, steps=200)
    roomy = hand.think_in_places(1024, 16, 64, reach=16, patch=32, steps=200)
    assert crowded < 0.9 < roomy                            # the same load, eight times the ring


def test_the_gate_by_hand_silences_and_stills_the_cells_away_from_the_place():
    torch.manual_seed(0)
    brain = Tissue(replace(RUNG1, fan=6, reach=4, field=3, gate=0.3), ITEM).grow(40)
    state = brain.born(2, gen(1))
    cue, where = torch.ones(2, 12), torch.tensor([0.0, 0.5])
    for _ in range(3):
        state, _ = brain.tick(state, cue, cue, 1.0, 1.0, where=where)
    act, wires, _ = state
    awake = brain.near(40, where) >= 0.3
    assert 3 <= int(awake[0].sum()) <= 12
    assert not act["I"][~awake].any() and act["I"][awake].abs().min() > 0
    assert not wires[1][~awake].any() and wires[1][awake].any()
    # with nothing felt there is no gate
    state, _ = brain.tick(brain.born(2, gen(1)), cue, cue, 1.0, 1.0)
    assert state[0]["I"].abs().min() > 0


def test_a_gate_leaned_away_turns_the_far_cells_down_and_at_nothing_is_no_gate():
    torch.manual_seed(0)
    genome = Tissue(replace(RUNG1, fan=6, reach=4, field=3, gate=0.3), ITEM)
    cue, where = torch.ones(2, 12), torch.tensor([0.0, 0.5])

    def first_tick(lean):
        brain = Tissue(replace(genome.plan, lean=lean), ITEM)
        brain.load_state_dict(genome.state_dict())
        grown = brain.grow(40)
        state, _ = grown.tick(grown.born(2, gen(1)), cue, cue, 1.0, 1.0, where=where)
        return state[0]["I"], grown.near(40, where) >= 0.3

    full, awake = first_tick(1.0)
    half, _ = first_tick(0.5)
    none, _ = first_tick(0.0)
    assert not full[~awake].any()
    assert torch.allclose(half[~awake], none[~awake] / 2) and torch.equal(half[awake], none[awake])
    ungated = Tissue(replace(genome.plan, gate=0.0), ITEM)
    ungated.load_state_dict(genome.state_dict())
    grown = ungated.grow(40)
    state, _ = grown.tick(grown.born(2, gen(1)), cue, cue, 1.0, 1.0, where=where)
    assert torch.equal(none, state[0]["I"])


def test_a_brain_that_grows_in_mid_life_keeps_every_old_cell_and_says_the_same_at_first():
    torch.manual_seed(0)
    plan = replace(RUNG1, fan=6, reach=4, field=3, gate=0.3, spoken=True)
    small = Tissue(plan, ITEM).grow(40)
    state = small.born(2, gen(1))
    cue = torch.ones(2, 12)
    place = torch.tensor([10.0, 25.0])                      # in cells: it does not move as the ring grows
    for _ in range(4):
        state, _ = small.tick(state, cue, cue, 1.0, 1.0, where=place / 40)
    big, grown = small.more(state, 24, gen(2))
    assert big.n["I"] == 64 and grown[0]["I"].shape == (2, 64)
    assert grown[1][0].shape == (2, 64, 12) and grown[1][1].shape == (2, 64, 6) and grown[1][2].shape == (2, 12, 64)
    assert torch.equal(grown[1][1][:, :40], state[1][1]) and torch.equal(grown[2][1][:, :40], state[2][1])
    assert torch.equal(grown[1][2][:, :, :40], state[1][2])
    assert not grown[1][1][:, 40:].any() and not grown[1][2][:, :, 40:].any()      # new plastic wires: zero
    assert grown[1][0][:, 40:].any()                                               # new innate wires: their own
    away = (grown[2][1][:, 40:] - torch.arange(40, 64).view(1, -1, 1)) % 64
    assert int(torch.minimum(away, 64 - away).max()) <= 4                          # wired within the new ring
    # growing costs nothing by itself: the next tick says what the small brain would have said
    _, was = small.tick(state, -cue, cue, 1.0, 1.0, where=place / 40)
    _, now = big.tick(grown, -cue, cue, 1.0, 1.0, where=place / 64)
    assert torch.allclose(was, now, atol=1e-5)
    with pytest.raises(ValueError):
        Tissue(RUNG1, ITEM).grow(16).more(Tissue(RUNG1, ITEM).grow(16).born(1, gen(0)), 8, gen(1))


def test_a_life_in_two_halves_is_the_same_world_for_every_arm_and_a_pure_function():
    from tissue import grow
    h = grow.Halves(first=2, second=3, length=4, laps=2, open=2, closed=3)
    before, after = grow.script(h, 3, gen(0))
    assert len(before) == 2 * 2 * 5 and len(after) == 2 * 3 * 5 + 2 * 5 * (2 + 3)
    assert all(int(loop.max()) < 2 for _, _, loop, _ in before)             # only the first loops
    assert all(int(loop.min()) >= 2 for kind, _, loop, _ in after if kind == "study")
    assert sum(kind == "closed" for kind, *_ in after) == 2 * 5 * 3
    torch.manual_seed(0)
    genome = Tissue(replace(RUNG1, fan=6, reach=8, field=4, gate=0.3, spoken=True), ITEM)
    lived = {arm: grow.run_life(genome, ITEM, h, 48, 96, arm, lives=3, seed=7) for arm in grow.ARMS}
    for arm in grow.ARMS:
        assert set(lived[arm]) == {"first", "second"} and len(lived[arm]["first"]) == 3
        assert all(0.0 <= x <= 1.0 for half in lived[arm].values() for x in half)
    assert lived["grown"] == grow.run_life(genome, ITEM, h, 48, 96, "grown", lives=3, seed=7)


def test_a_place_can_be_told_always_or_only_with_the_eyes_open_or_only_in_study():
    torch.manual_seed(0)
    genome = Tissue(replace(RUNG1, fan=6, reach=4, field=3), ITEM)
    brain = genome.grow(32)
    life = stream.Life(loops=2, length=4, laps=1, idle=3, blinks=2, open=2, closed=2)
    items = stream.world(ITEM, life, 2, gen(4))
    felt = []
    tick = brain.tick
    brain.tick = lambda *a, where=None, **k: (felt.append(where is not None), tick(*a, where=where, **k))[1]
    kinds = [kind for kind, *_ in stream.script(life, 2, gen(5))] + ["end"]
    seen = [k in ("study", "echo") for k in kinds]
    for told, expect in (("always", [True] * len(kinds)), ("seen", seen),
                         ("study", [k == "study" for k in kinds])):
        felt.clear()
        stream.run_life(brain, ITEM, items, life, gen(5), places=True, told=told)
        assert felt[:-1] == expect[:-1], told               # the last tick has no world left
    with pytest.raises(ValueError):
        stream.run_life(brain, ITEM, items, life, gen(5), places=True, told="never")


def test_a_hand_made_index_says_where_a_thing_was_seen_and_is_wrong_when_overloaded():
    index = hand.Index(batch=2, cells=64, d=12, ring=256)
    assert index.bits == 4                                  # sixteen stretches of sixteen cells
    item = torch.ones(2, 12)
    item[1, :6] = -1
    place = torch.tensor([0.30, 0.71])
    index.learn(item, place)
    assert torch.allclose(index.say(item), torch.tensor([4.5 / 16, 11.5 / 16]))    # the stretch it lay in
    assert hand.index_right(256, 256, 16) > 0.95 > 0.6 > hand.index_right(256, 16, 256)
    assert hand.index_right(256, 256, 16) == hand.index_right(256, 256, 16)


def test_a_life_told_its_places_by_an_index_is_the_same_world_and_feels_what_the_index_says():
    torch.manual_seed(0)
    brain = Tissue(replace(RUNG1, fan=6, reach=4, field=3), ITEM).grow(32)
    life = stream.Life(loops=2, length=4, laps=2, idle=3, blinks=2, open=2, closed=2)
    items = stream.world(ITEM, life, 2, gen(4))
    felt = []
    tick = brain.tick
    brain.tick = lambda *a, where=None, **k: (felt.append(where), tick(*a, where=where, **k))[1]
    stream.run_life(brain, ITEM, items, life, gen(5), places=True, told="always")
    always = [None if w is None else w.clone() for w in felt]
    felt.clear()
    index = hand.Index(2, 64, 12, 32)
    stream.run_life(brain, ITEM, items, life, gen(5), places=True, told="index", index=index)
    kinds = [kind for kind, *_ in stream.script(life, 2, gen(5))]
    for k, (kind, a, b) in enumerate(zip(kinds, always, felt)):
        assert b is not None                                # a place is always felt
        if kind == "study":
            assert torch.equal(a, b)                        # in study it is the world's
    assert index.w.any()


def test_a_brain_never_told_a_place_writes_new_things_along_a_tape_and_finds_known_ones():
    index = hand.Index(batch=1, cells=256, d=12, ring=256, pace=8)
    a, b = torch.ones(1, 12), -torch.ones(1, 12)
    first = float(index.tape(a))
    second = float(index.tape(b))
    assert first == pytest.approx(8 / 256) and second == pytest.approx(16 / 256)    # new: further along
    again = float(index.tape(a))
    assert abs(again - first) <= 8 / 256                    # known: where it was put, to a stretch
    assert float(index.head) == pytest.approx(16 / 256)     # and the tape did not move
    torch.manual_seed(0)
    brain = Tissue(replace(RUNG1, fan=6, reach=4, field=3), ITEM).grow(32)
    life = stream.Life(loops=2, length=4, laps=2, idle=3, blinks=2, open=2, closed=2)
    items = stream.world(ITEM, life, 2, gen(4))
    felt = []
    tick = brain.tick
    brain.tick = lambda *x, where=None, **k: (felt.append(where), tick(*x, where=where, **k))[1]
    stream.run_life(brain, ITEM, items, life, gen(5), places=True, told="self",
                    index=hand.Index(2, 256, 12, 32, pace=2))
    assert all(w is not None for w in felt)
    assert len({round(float(w[0]), 4) for w in felt}) >= 4  # eight things seen, several places


def test_with_runs_new_things_in_a_row_share_a_place_and_a_new_one_after_a_known_opens_the_next():
    index = hand.Index(batch=1, cells=256, d=12, ring=256, pace=32, runs=True)
    things = [torch.ones(1, 12)]
    for k in range(3):
        t = torch.ones(1, 12)
        t[0, k] = -1
        things.append(t)
    a, b, c, d = things
    first = [float(index.tape(x)) for x in (a, b)]         # two new things in a row
    assert first[0] == first[1] == pytest.approx(32 / 256)
    back = float(index.tape(a))                             # a known one: where it was put
    assert abs(back - first[0]) <= 8 / 256
    then = [float(index.tape(x)) for x in (c, d)]           # new after known: the next place
    assert then[0] == then[1] == pytest.approx(64 / 256)


def test_places_by_hand_fill_the_emptiest_stretch_and_say_when_the_ring_is_crowded():
    from tissue import grow
    places = grow.Places(batch=1, cells=512, d=12, ring=256, most=512, full=1.0)
    new = lambda k: torch.tensor([[1.0 if (k >> j) & 1 else -1.0 for j in range(12)]])
    seen = []
    for k in range(6):                                      # six runs: a new thing, then a known one
        seen.append(float(places.tape(new(k))) * 512)
        places.tape(new(k))
    assert len({round(x) for x in seen}) == 6 and max(seen) < 256    # six stretches of the small ring
    assert bool(places.crowded())                           # every stretch has a place within reach
    places.ring = 512                                       # the ring grows: there is room again
    assert not bool(places.crowded())
    assert float(places.tape(new(9))) * 512 >= 256          # and the next new thing goes there


def test_a_life_told_nothing_runs_for_every_arm_and_the_one_that_grows_ends_between_the_others():
    from tissue import grow
    torch.manual_seed(0)
    genome = Tissue(replace(RUNG1, fan=6, reach=8, field=4, gate=0.3, spoken=True), ITEM)
    lived = {arm: grow.alone(genome, ITEM, loops=8, batches=4, small=64, big=128, arm=arm, lives=2,
                             step=32, full=0.4, closed=3) for arm in ("grows", "born big", "small")}
    assert lived["small"]["cells"] == 64 and lived["born big"]["cells"] == 128
    assert 64 < lived["grows"]["cells"] <= 128
    assert lived["small"]["cost"] < lived["grows"]["cost"] < lived["born big"]["cost"]
    assert all(len(r["hop"]) == 3 for r in lived.values())


def test_a_watched_life_is_the_same_life_and_a_picture_of_it_can_be_drawn():
    from tissue import grow, look
    torch.manual_seed(0)
    genome = Tissue(replace(RUNG1, fan=6, reach=8, field=4, gate=0.3, spoken=True), ITEM)
    rows = []
    kept = dict(loops=8, batches=4, small=64, big=128, arm="grows", lives=1, step=32, full=0.4, closed=3)
    watched = grow.alone(genome, ITEM, watch=lambda **tick: rows.append(
        {k: v[0].clone().numpy() if torch.is_tensor(v) else v for k, v in tick.items()}), **kept)
    assert watched == grow.alone(genome, ITEM, **kept)      # looking changes nothing
    assert len(rows) == watched["cost"] // 64 or len(rows[-1]["act"]) > 64
    assert len(rows[0]["act"]) == 64 and len(rows[-1]["act"]) == watched["cells"]
    img = look.draw(rows, "a test", loops=8, batches=4, closed=3, zoom=2)
    assert img.size[0] > 1000 and img.size[1] > 500


def test_a_forked_world_hangs_together_and_a_round_takes_every_way_out_of_every_room_once():
    from tissue import fork
    walk = fork.Walk(rooms=9, ways=3, laps=2, blinks=4, open=3, closed=4)
    items, nxt = fork.world(12, walk, 5, gen(1))
    assert items.shape == (5, 9, 12) and nxt.shape == (5, 9, 3)
    for b in range(5):
        assert len({tuple(x.tolist()) for x in items[b]}) == 9          # distinct rooms
        for w in range(3):
            assert sorted(nxt[b, :, w].tolist()) == list(range(9))      # a way is a shuffle
        room, seen = 0, set()                                           # way 0 is one round of all
        for _ in range(9):
            seen.add(room)
            room = int(nxt[b, room, 0])
        assert room == 0 and len(seen) == 9
        steps, room, taken = fork.tour(nxt[b], 4, gen(2)), 4, set()
        for to, way in steps:
            assert int(nxt[b, room, way]) == to
            taken.add((room, way))
            room = to
        assert len(steps) == len(taken) == 27 and room == 4            # every link once, and home
    ticks = fork.script(nxt, walk, gen(3))
    assert len(ticks) == 1 + 2 * 27 + 4 * 7
    assert [k for k, _, _, _ in ticks[-7:]] == ["echo"] * 3 + ["closed"] * 4
    assert all(seen == (kind != "closed") for kind, seen, _, _ in ticks)
    assert int(ticks[0][3].max()) == -1 and int(ticks[-7][3].max()) == -1       # carried: no way felt
    for (_, _, was, _), (_, _, room, way) in zip(ticks[-7:], ticks[-6:]):
        assert torch.equal(nxt[torch.arange(5), was, way], room)        # the body goes where the way leads
    same = fork.script(nxt, walk, gen(3))
    assert all(torch.equal(a[2], b[2]) and torch.equal(a[3], b[3]) for a, b in zip(ticks, same))


def test_a_memory_that_does_not_feel_its_body_is_right_about_one_hop_in_as_many_as_there_are_ways():
    from tissue import fork
    for ways, hop1 in ((1, 1.0), (2, 0.75), (4, 0.625)):
        walk = fork.Walk(rooms=64, ways=ways, blinks=64)
        items, nxt = fork.world(12, walk, 8, gen(4))
        hops = fork.by_hop(fork.numb(items, nxt, walk, gen(5)), walk)
        assert hops[0] == pytest.approx(hop1, abs=0.04)
        assert ways == 1 or hops[-1] < hops[0] - 0.05


@torch.no_grad()
def test_a_walk_is_lived_without_a_reset_and_a_dendrite_per_way_is_the_bodys_by_hand():
    from tissue import fork
    torch.manual_seed(0)
    genome = Tissue(RUNG1, ITEM)
    walk = fork.Walk(rooms=6, ways=2, blinks=3)
    items, nxt = fork.world(12, walk, 4, gen(6))
    out = fork.run_life(genome.grow(32), items, nxt, walk, gen(7), body="dendrites")
    assert out["closed"][0].shape == (4, 3 * 4, 12) and out["echo"][0].shape == (4, 3 * 3, 12)
    again = fork.run_life(genome.grow(32), items, nxt, walk, gen(7), body="dendrites")
    assert torch.equal(out["closed"][0], again["closed"][0])            # a pure function
    numb = fork.run_life(genome.grow(32), items, nxt, walk, gen(7), body="numb")
    assert torch.equal(out["closed"][1], numb["closed"][1])             # the same world for any body
    assert not torch.equal(out["closed"][0], numb["closed"][0])
    # with one way a dendrite per way is the brain as it always was
    one = fork.Walk(rooms=6, ways=1, blinks=3)
    items, nxt = fork.world(12, one, 4, gen(8))
    a = fork.run_life(genome.grow(32), items, nxt, one, gen(9), body="dendrites")["closed"][0]
    b = fork.run_life(genome.grow(32), items, nxt, one, gen(9), body="numb")["closed"][0]
    assert torch.allclose(a, b, atol=1e-2) and not torch.equal(a, b)    # but for what a jump teaches
    # the body on sensory cells of its own: the genome is the same, the brain has more of them
    felt = fork.embodied(Tissue(RUNG1, ITEM), 2)
    assert felt.grow(32).n["S"] == 14 and sum(p.numel() for p in felt.parameters()) == sum(
        p.numel() for p in genome.parameters())
    items, nxt = fork.world(12, walk, 4, gen(6))
    assert fork.run_life(felt.grow(32), items, nxt, walk, gen(7), body="cells")["closed"][0].shape == (4, 12, 12)
    with pytest.raises(ValueError):
        fork.run_life(genome.grow(32), items, nxt, walk, gen(7), body="legs")


@torch.no_grad()
def test_a_wanted_room_is_sought_by_walking_a_copy_in_the_dark_and_the_brain_is_left_as_it_was():
    from tissue import fork, seek
    torch.manual_seed(0)
    walk = fork.Walk(rooms=8, ways=2)
    items, nxt = fork.world(12, walk, 3, gen(1))
    far = seek.distances(nxt, 3)
    for b in range(3):                                      # against a plain search, room by room
        reach, at = {0: 0}, [0]
        for k in range(1, 4):
            at = [int(nxt[b, r, w]) for r in at for w in range(2)]
            for r in at:
                reach.setdefault(r, k)
        assert [int(x) for x in far[b, 0]] == [reach.get(r, 4) for r in range(8)]
    brain = Tissue(RUNG1, ITEM).grow(32)
    state = brain.born(3, gen(2))
    body = fork.Dendrites(brain, state, 2)
    for _, _, room, way in fork.script(nxt, fork.Walk(8, 2, blinks=0), gen(3)):
        state, _ = body.tick(brain, state, items[torch.arange(3), room], way, torch.zeros(3, 12), 0.0, 1.0)
    bank, wires = body.bank.clone(), {i: w.clone() for i, w in state[1].items()}
    _, said = seek.imagine(brain, body, state, torch.tensor([0, 1, 1]), 12)
    assert said.shape == (3, 3, 12)
    assert torch.equal(bank, body.bank) and all(torch.equal(w, state[1][i]) for i, w in wires.items())
    assert not torch.equal(said, seek.imagine(brain, body, state, torch.tensor([1, 1, 1]), 12)[1])
    kept = fork.Dendrites(brain, state, 2)                  # thinking with no copy: what it learns stays
    kept.bank = body.bank.clone()
    seek.imagine(brain, kept, state, torch.tensor([0, 1, 1]), 12, keep=True)
    assert not torch.equal(kept.bank, bank)
    # a life of wanted rooms: every arm lives the same worlds, and a life is a pure function
    lived = {arm: seek.seek(brain, items, nxt, walk, gen(4), depth=3, trials=4, arm=arm) for arm in seek.ARMS}
    assert all(r["far"].shape == (3, 4) and int(r["far"].min()) >= 1 and int(r["far"].max()) <= 3
               for r in lived.values())
    assert torch.equal(lived["thinks"]["far"][:, 0], lived["blind"]["far"][:, 0])
    assert not lived["blind"]["plan"].any()                 # a brain that does not think makes no plan
    assert int(lived["blind"]["steps"].min()) >= 1 and int(lived["blind"]["steps"].max()) <= 9
    again = seek.seek(brain, items, nxt, walk, gen(4), depth=3, trials=4, arm="thinks")
    assert all(torch.equal(lived["thinks"][k], again[k]) for k in again)
    table = seek.by_distance(lived["blind"], 3, 8)
    assert sum(row["n"] for row in table.values()) == 12


@torch.no_grad()
def test_with_still_no_wire_changes_while_the_eyes_are_closed_and_all_do_while_they_are_open():
    torch.manual_seed(0)
    genome = Tissue(replace(RUNG1, still=True), ITEM)
    brain = genome.grow(16)
    cue, silent = torch.ones(2, 12), torch.zeros(2, 12)
    state = brain.born(2, gen(1))
    for _ in range(3):
        state, _ = brain.tick(state, cue, cue, 1.0, 1.0)
    dark, _ = brain.tick(state, silent, cue, 1.0, 0.0)
    assert all(torch.equal(dark[1][i], state[1][i]) for i in (1, 2))       # eyes closed: still
    assert not torch.equal(dark[0]["I"], state[0]["I"])                   # though the cells go on
    seen, _ = brain.tick(state, cue, cue, 1.0, 1.0)
    assert not any(torch.equal(seen[1][i], state[1][i]) for i in (1, 2))  # eyes open: they learn
    plain = Tissue(RUNG1, ITEM)
    plain.load_state_dict(genome.state_dict())
    moved, _ = plain.grow(16).tick(state, silent, cue, 1.0, 0.0)
    assert not any(torch.equal(moved[1][i], state[1][i]) for i in (1, 2))  # the switch is what stills them


SLEEPY = replace(RUNG1, fan=6, reach=4, field=3, spoken=True, sleep=4)


@torch.no_grad()
def test_a_sleep_cell_is_born_saying_awake_and_by_hand_it_is_the_hand_gate():
    torch.manual_seed(0)
    genome = Tissue(SLEEPY, ITEM)
    assert set(genome.cells) == {"I", "O", "Z"} and genome.cells["Z"].a.weight.shape == (8, 2)
    brain = genome.grow(40)
    assert brain.n["Z"] == 10
    cue, here, there = torch.ones(2, 12), torch.tensor([0.0, 0.5]), torch.tensor([0.5, 0.0])
    state = brain.born(2, gen(1))
    for _ in range(3):
        state, _ = brain.tick(state, cue, cue, 1.0, 1.0, where=here)
    assert float(torch.sigmoid(state[0]["Z"]).min()) > 0.85         # born blank: nobody is put to sleep
    assert state[0]["I"].abs().min() > 0

    genome.sleep_by_hand(0.3)
    state = brain.born(2, gen(1))
    for _ in range(3):
        state, _ = brain.tick(state, cue, cue, 1.0, 1.0, where=here)
    act, wires, _ = state
    awake = (brain.near(40, here, 4) >= 0.3).repeat_interleave(4, dim=1)    # by stretches of four
    assert 4 <= int(awake[0].sum()) <= 12
    loud = float(act["I"][awake].abs().mean())
    assert float(act["I"][~awake].abs().max()) < 1e-3 * loud        # asleep: silent,
    assert float(wires[1][~awake].abs().max()) < 1e-2 * float(wires[1][awake].abs().max())    # and still
    # the place moves: the sleep cell speaks within the tick, so the new patch is awake at once
    moved, _ = brain.tick(state, cue, cue, 1.0, 1.0, where=there)
    assert float(moved[0]["I"][awake].abs().max()) < 1e-3 * loud and moved[0]["I"][~awake].any()
    # heard a tick late (the control), the old patch speaks on that tick and the new one a tick after
    slow = Tissue(replace(SLEEPY, late=True), ITEM)
    slow.load_state_dict(genome.state_dict())
    slow = slow.grow(40)
    late, _ = slow.tick(state, cue, cue, 1.0, 1.0, where=there)
    assert float(late[0]["I"][~awake].abs().max()) < 1e-3 * loud and late[0]["I"][awake].any()
    late, _ = slow.tick(late, cue, cue, 1.0, 1.0, where=there)
    assert float(late[0]["I"][awake].abs().max()) < 1e-3 * loud and late[0]["I"][~awake].any()
    # with nothing felt a sleep cell set by hand puts everyone to sleep
    state, _ = brain.tick(moved, cue, cue, 1.0, 1.0)
    assert float(state[0]["I"].abs().max()) < 1e-3 * loud


def test_sleep_cells_come_with_new_cells_in_mid_life_and_a_plan_without_them_drops_them():
    from tissue.train import adopt
    torch.manual_seed(0)
    genome = Tissue(SLEEPY, ITEM)
    genome.sleep_by_hand(0.3)
    small = genome.grow(40)
    state = small.born(2, gen(1))
    cue, place = torch.ones(2, 12), torch.tensor([10.0, 25.0])
    for _ in range(4):
        state, _ = small.tick(state, cue, cue, 1.0, 1.0, where=place / 40)
    big, grown = small.more(state, 24, gen(2))
    assert big.n["Z"] == 16 and grown[0]["Z"].shape == (2, 16)
    assert torch.equal(grown[0]["Z"][:, :10], state[0]["Z"])
    _, was = small.tick(state, -cue, cue, 1.0, 1.0, where=place / 40)
    grown, now = big.tick(grown, -cue, cue, 1.0, 1.0, where=place / 64)
    assert torch.allclose(was, now, atol=1e-5)              # growing costs nothing by itself
    far = (big.near(64, place / 64, 4) < 0.3).repeat_interleave(4, dim=1)
    assert float(grown[0]["I"][far].abs().max()) < 1e-3     # a new cell far from the place is born asleep
    with pytest.raises(ValueError):
        genome.grow(42).more(genome.grow(42).born(1, gen(0)), 8, gen(1))     # not whole stretches
    plain = Tissue(replace(SLEEPY, sleep=0), ITEM)
    adopt(plain, genome.state_dict())                       # the sleep cells are left out
    assert torch.equal(plain.cells["I"].a.weight, genome.cells["I"].a.weight)


@torch.no_grad()
def test_a_land_hangs_together_and_a_way_back_undoes_a_way_gone_unless_a_wall_stands():
    from tissue import world
    shape = world.Land(side=5)
    looks, nxt, trees, start = world.land(12, shape, gen(1))
    assert looks.shape == (25, 12) and len({world.key(x) for x in looks}) == 25      # no two rooms alike
    assert len(set(trees.tolist())) == shape.n_trees == 4 and 0 <= start < 25
    walls = 0
    for a in range(25):
        for w in range(4):
            b = int(nxt[a, w])
            row, col = a // 5 + world.WAYS[w][0], a % 5 + world.WAYS[w][1]
            inside = 0 <= row < 5 and 0 <= col < 5
            assert b in (a, row * 5 + col if inside else a)         # the room that way, or a wall
            if b != a:
                assert int(nxt[b, (w + 2) % 4]) == a                # and the way back leads back
            walls += inside and b == a
    assert walls == 2 * round(0.3 * 40)                             # a wall has two sides
    far = world.roads(nxt)
    assert int(far.max()) < 25 and torch.equal(far, far.T)          # every room is reached from every room
    again = world.land(12, shape, gen(1))
    assert torch.equal(nxt, again[1]) and torch.equal(looks, again[0])


@torch.no_grad()
def test_in_a_land_a_memory_born_knowing_every_way_goes_straight_and_one_that_never_plans_wanders():
    from tissue import world
    shape = world.Land(side=5, ticks=600)
    lives = {}
    for arm in world.YARDSTICKS:
        looks, nxt, trees, start = world.land(12, shape, gen(1))
        lives[arm] = world.live(world.body_for(arm, None, 0, looks, nxt, gen(2)), looks, nxt, trees, start,
                                600, gen(3), fruits=7)
    oracle, perfect, blind = lives["oracle"], lives["perfect"], lives["blind"]
    assert all(far == steps for far, steps, _ in oracle["trips"]) and oracle["planned"] == 600
    assert blind["planned"] == 0 and len(blind["meals"]) < len(perfect["meals"]) < len(oracle["meals"])
    # the fruit goes from tree to tree in the same order whoever lives there
    assert [far for far, _, _ in perfect["trips"]][:3] != [] and oracle["trips"][0][0] == blind["trips"][0][0]
    res = world.summary([oracle], 600)
    assert sum(res["thirds"]) == res["meals"] == len(oracle["meals"])
    assert all(row["straight"] == 1.0 for row in res["trips"].values())
    assert sum(row["n"] for row in res["late"].values()) == res["late_per_life"][0]


@torch.no_grad()
def test_a_brain_lives_in_a_land_and_a_thought_leaves_it_as_it_was():
    from tissue import world
    torch.manual_seed(0)
    shape = world.Land(side=3, ticks=40)
    looks, nxt, trees, start = world.land(12, shape, gen(1))
    with pytest.raises(ValueError):
        world.Mind(Tissue(RUNG1, ITEM).grow(16), gen(2))            # a thought needs its wires still
    genome = Tissue(replace(RUNG1, still=True), ITEM)
    mind = world.Mind(genome.grow(16), gen(2), budget=32)
    life = world.live(mind, looks, nxt, trees, start, 40, gen(3), fruits=5)
    assert int(life["walked"].sum()) == 40
    bank, act = mind.body.bank.clone(), {t: a.clone() for t, a in mind.state[0].items()}
    wires = {i: w.clone() for i, w in mind.state[1].items()}
    trace: list = []
    way = mind.think(looks[int(trees[0])], gen(4), trace=trace)
    assert way is None or 0 <= way < 4
    assert trace and all(hops >= 1 and 0 <= w < 4 for hops, _, w, _ in trace)
    says = mind.atlas(looks)
    assert says.shape == (9, 4, 12)
    assert torch.equal(bank, mind.body.bank) and all(torch.equal(a, mind.state[0][t]) for t, a in act.items())
    assert all(torch.equal(w, mind.state[1][i]) for i, w in wires.items())
    known = world.known_map(mind, looks, nxt, life["walked"])
    assert 0 < known["gone"] <= 1 and 0 <= known["right_gone"] <= 1
    again = world.Mind(genome.grow(16), gen(2), budget=32)          # a life is a pure function
    relived = world.live(again, looks, nxt, trees, start, 40, gen(3), fruits=5)
    assert relived["meals"] == life["meals"] and torch.equal(relived["walked"], life["walked"])
    numb = world.Mind(genome.grow(16), gen(2), numb=True, budget=32)
    world.live(numb, looks, nxt, trees, start, 40, gen(3), fruits=5)
    assert numb.body.bank.shape[1] == 1


@torch.no_grad()
def test_a_life_in_a_land_can_be_watched_and_drawn_and_a_thought_gives_back_its_road(tmp_path):
    from tissue import view, world
    from tissue.train import make
    torch.manual_seed(0)
    genome, _ = make("stream")
    path = str(tmp_path / "blank.pt")
    torch.save({"arm": "stream", "genome": genome.state_dict()}, path)
    shape = world.Land(side=3, ticks=60)
    rec = view.record(path, shape, 16, budget=16)
    assert len(rec["track"]) == 60 and sorted(rec["maps"]) == [7, 20, 60]
    assert set(rec["meals"]) == {"thinks", *world.YARDSTICKS}
    assert rec["maps"][60]["gone"].sum() >= rec["maps"][7]["gone"].sum()       # it has gone more ways by the end
    again = view.record(path, shape, 16, budget=16)                            # watching changes nothing
    assert again["track"] == rec["track"] and again["meals"] == rec["meals"]
    # rooms 0 -> 1 -> 4, found at the second hop among copies that led nowhere
    thought = {"here": 0, "want": 4, "tick": 50,
               "trace": [(1, 0, 1, 1), (1, 0, 2, 3), (1, 0, 0, None), (2, 1, 2, 4), (2, 3, 1, None)]}
    assert view.road(thought) == [0, 1, 4]
    assert view.road({**thought, "want": 8}) == []
    rec["thought"] = thought
    img = view.draw(rec, "a test", 16)
    assert img.size[0] > 1000 and img.size[1] > 500


@torch.no_grad()
def test_a_sense_of_the_body_is_one_more_dendrite_and_a_genome_without_it_begins_numb():
    from tissue import fork
    from tissue.train import adopt
    torch.manual_seed(0)
    old = Tissue(RUNG1, ITEM)
    new = Tissue(replace(RUNG1, body=2), ITEM)
    assert new.cells["I"].a.weight.shape[1] == old.cells["I"].a.weight.shape[1] + 1
    assert sum(p.numel() for p in new.parameters()) == sum(p.numel() for p in old.parameters()) + 8 + 1
    assert new.grow(16).born(3, gen(1))[1]["body"].shape == (3, 16, 2)     # from every body cell to every interneuron
    adopt(new, old.state_dict())
    felt = fork.feel(torch.tensor([1, -1, 0]), 2)
    assert felt.tolist() == [[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]]        # carried: the body cells are silent
    walk = fork.Walk(rooms=6, ways=2, blinks=3)
    items, nxt = fork.world(12, walk, 3, gen(2))
    numb = fork.run_life(old.grow(32), items, nxt, walk, gen(3), body="numb")["closed"][0]
    now = fork.run_life(new.grow(32), items, nxt, walk, gen(3), body="felt")["closed"][0]
    assert torch.allclose(numb, now, atol=1e-5)             # weights from the body at zero: the old brain
    new.cells["I"].a.weight[:, -1] = 1.0                    # a cell that listens to its body
    heard = fork.run_life(new.grow(32), items, nxt, walk, gen(3), body="felt")["closed"][0]
    assert not torch.allclose(heard, now, atol=1e-3)
    assert torch.equal(heard, fork.run_life(new.grow(32), items, nxt, walk, gen(3), body="felt")["closed"][0])
    three = fork.Walk(rooms=6, ways=3, blinks=3)
    with pytest.raises(ValueError):                         # as many body cells as there are ways
        fork.run_life(new.grow(32), *fork.world(12, three, 3, gen(2)), three, gen(3), body="felt")


def test_a_search_on_forks_runs_saves_and_the_genome_comes_back_with_its_body(tmp_path):
    from tissue import fork
    from tissue.train import load, train
    stream_genome = tmp_path / "stream.pt"
    train("stream", 0, 2, stream_genome, batch=2, hops=2, sizes=(8,))
    out = tmp_path / "fork.pt"
    train("fork", 0, 3, out, batch=2, hops=4, vary=True, sizes=(8, 16), init=str(stream_genome))
    model, task, saved = load(str(out))
    assert saved["body"] == 2 and model.plan.body == 2 and saved["arm"] == "fork"
    four, _, _ = load(str(out), body=4)                     # the genome does not say how many ways there are
    assert four.grow(8).born(1, gen(1))[1]["body"].shape == (1, 8, 4)
    res = fork.examine(str(out), [16], fork.Walk(rooms=1, ways=2, closed=2), lives=2, body="felt")
    assert len(res[16]["k90"]) == 3
