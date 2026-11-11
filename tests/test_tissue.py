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
