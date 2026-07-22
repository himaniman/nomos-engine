"""The LLM lab: a model writes a growth program, a body develops from it.

The claims that matter here are the ones that keep a run HONEST — that a candidate's
weights really are held fixed so a score difference is attributable to the program,
that a model cannot break the run by writing nonsense, and that the control arm is the
same loop with a different proposer rather than a different experiment."""

import json
from pathlib import Path

import numpy as np
import pytest

from cortex import brain as B
from cortex import lab
from cortex import program as PROG
from nomos.llm import MockClient

TICKS, PROBE = 260, 90
SEEDS = (0,)


def _reply(program: dict, hypothesis: str = "a guess", **extra) -> str:
    body = {"hypothesis": hypothesis, "program": program, **extra}
    return f"Reasoning about it.\n\n```json\n{json.dumps(body)}\n```\n"


# --- what the model may write ---------------------------------------------------------

def test_the_program_is_what_varies_and_the_weights_are_not():
    """The whole basis of attributing a score to a program: every candidate is grown
    from the same random weights."""
    a, _ = lab.build_genome({"program": {"grow_thresh": 0.1}}, brain_seed=3)
    b, _ = lab.build_genome({"program": {"grow_thresh": 0.9}}, brain_seed=3)
    assert a.program["grow_thresh"] != b.program["grow_thresh"]
    assert np.array_equal(a.P_in, b.P_in)
    assert np.array_equal(a.P_out, b.P_out)
    for ra, rb in zip(a.rules, b.rules):
        for La, Lb in zip(ra, rb):
            assert np.array_equal(La["W"], Lb["W"])


def test_omitted_fields_keep_their_current_value():
    g, _ = lab.build_genome({"program": {"grow_thresh": 0.11}}, brain_seed=0)
    assert g.program["grow_thresh"] == pytest.approx(0.11)
    assert g.program["chem_mix"] == PROG.DEFAULTS["chem_mix"]


def test_out_of_range_and_invented_fields_cannot_break_a_run():
    g, unknown = lab.build_genome(
        {"program": {"grow_thresh": 999.0, "elan_vital": 3.0}}, brain_seed=0)
    assert unknown == ["elan_vital"]
    assert g.program["grow_thresh"] == PROG.FIELDS["grow_thresh"][2]


def test_structural_asks_are_clamped():
    g, _ = lab.build_genome(
        {"program": {}, "subticks": 9999, "seed_types": 99, "organ_map": [0, 77, 3]},
        brain_seed=0)
    cfg = B.Config()
    assert g.subticks == cfg.subticks_max
    assert 1 <= g.n_types <= cfg.max_types
    assert all(1 <= int(v) <= cfg.organ_mods for v in g.organ_map)


# --- reading the model's reply --------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('```json\n{"program": {"grow_bias": 0.4}}\n```', {"grow_bias": 0.4}),
    ('prose\n```\n{"program": {"grow_bias": 0.4}}\n```\nmore', {"grow_bias": 0.4}),
    ('no fence at all {"program": {"grow_bias": 0.4}} trailing', {"grow_bias": 0.4}),
])
def test_a_program_is_found_however_it_is_fenced(text, expected):
    assert lab.extract_json(text)["program"] == expected


def test_the_last_block_wins_when_a_model_shows_its_working():
    text = ('```json\n{"program": {"grow_bias": 0.1}}\n```\n'
            'on reflection:\n```json\n{"program": {"grow_bias": 0.8}}\n```')
    assert lab.extract_json(text)["program"] == {"grow_bias": 0.8}


@pytest.mark.parametrize("text", ["nothing here", "```json\n{oops,}\n```", ""])
def test_junk_yields_no_program(text):
    assert lab.extract_json(text) is None or "program" not in lab.extract_json(text)


# --- the loop -------------------------------------------------------------------------

def test_a_run_leaves_a_journal_a_gallery_and_the_best_program(tmp_path):
    client = MockClient([_reply({"grow_thresh": 0.15}, "grow more"),
                         _reply({"sense_bias": 0.4}, "hear more")])
    out = lab.run_lab(client, iterations=2, seeds=SEEDS, ticks=TICKS,
                      probe_ticks=PROBE, out_dir=str(tmp_path))
    run = Path(out["dir"])
    assert (run / "journal.md").exists()
    assert (run / "gallery.png").exists()
    saved = json.loads((run / "best_program.json").read_text(encoding="utf-8"))
    assert set(saved["program"]) == set(PROG.FIELDS), \
        "the saved program must be complete, not just the fields that were changed"
    text = (run / "journal.md").read_text(encoding="utf-8")
    assert "grow more" in text and "hear more" in text
    for i in range(3):                       # generation 0 plus the two proposals
        assert (run / f"iter{i:02d}_probes.png").exists()
        assert (run / f"iter{i:02d}_brain.png").exists()
        assert f"iter{i:02d}_brain.png" in text, "the journal has to show the pictures"


def test_generation_zero_is_the_default_program(tmp_path):
    """Without it 'better' has no referent."""
    out = lab.run_lab(MockClient([_reply({"grow_thresh": 0.15})]), iterations=1,
                      seeds=SEEDS, ticks=TICKS, probe_ticks=PROBE,
                      out_dir=str(tmp_path), shots=False)
    gen0 = out["board"][0]
    assert gen0["iter"] == 0
    assert gen0["spec"]["program"] == PROG.Program().to_dict()


def test_a_reply_with_no_program_is_asked_again_not_counted(tmp_path):
    client = MockClient(["I would rather not.", _reply({"grow_thresh": 0.15})])
    out = lab.run_lab(client, iterations=1, seeds=SEEDS, ticks=TICKS,
                      probe_ticks=PROBE, out_dir=str(tmp_path), shots=False)
    assert len(out["board"]) == 2                     # gen 0 + one valid proposal
    assert "Resend" in client.calls[-1][-1]["content"]


def test_the_model_is_told_what_grew_and_where_it_stands(tmp_path):
    client = MockClient([_reply({"grow_thresh": 0.15}), _reply({"sense_bias": 0.4})])
    lab.run_lab(client, iterations=2, seeds=SEEDS, ticks=TICKS, probe_ticks=PROBE,
                out_dir=str(tmp_path), shots=False)
    last = client.calls[-1][-1]["content"]
    assert "LEADERBOARD" in last
    assert "cells" in last and "sensory" in last      # the body it cannot see
    assert "meals" in last


def test_the_model_is_never_shown_a_weight(tmp_path):
    """docs/cortex-design.md: handing a model raw matrices is the one way to waste a
    good one. Nothing in the conversation may carry them."""
    client = MockClient([_reply({"grow_thresh": 0.15})])
    lab.run_lab(client, iterations=1, seeds=SEEDS, ticks=TICKS, probe_ticks=PROBE,
                out_dir=str(tmp_path), shots=False)
    whole = "\n".join(m["content"] for call in client.calls for m in call)
    for banned in ("P_in", "P_out", "P_horm", "b_out", '"W"', "rules"):
        assert banned not in whole, f"the prompt leaked {banned}"


def test_the_spec_documents_every_field_the_model_may_write(tmp_path):
    """A knob the model is not told about is a knob it will never turn."""
    client = MockClient([_reply({"grow_thresh": 0.15})])
    lab.run_lab(client, iterations=1, seeds=SEEDS, ticks=TICKS, probe_ticks=PROBE,
                out_dir=str(tmp_path), shots=False)
    first = client.calls[0][-1]["content"]
    for k in PROG.FIELDS:
        assert k in first, f"{k} is writable but never described to the model"


def test_the_rng_control_arm_runs_the_same_loop(tmp_path):
    """The control has to differ ONLY in who proposes, or it answers a different
    question than 'is the LLM better than chance at this'."""
    out = lab.run_lab(lab.RngProposer(seed=0), iterations=2, seeds=SEEDS,
                      ticks=TICKS, probe_ticks=PROBE, out_dir=str(tmp_path),
                      shots=False)
    assert len(out["board"]) == 3
    assert out["board"][1]["spec"]["program"] != out["board"][2]["spec"]["program"]


def test_the_journal_records_the_program_that_actually_ran(tmp_path):
    """Values are clamped on the way in, so what was asked for and what ran can
    differ — and the journal must show the latter."""
    client = MockClient([_reply({"grow_thresh": 50.0})])
    out = lab.run_lab(client, iterations=1, seeds=SEEDS, ticks=TICKS,
                      probe_ticks=PROBE, out_dir=str(tmp_path), shots=False)
    assert out["board"][1]["spec"]["program"]["grow_thresh"] \
        == PROG.FIELDS["grow_thresh"][2]


def test_the_winning_program_is_saved_as_a_genome_you_can_watch(tmp_path):
    """A run whose answer can only be read as numbers is a run nobody looks at."""
    from cortex.agent import Mind
    out = lab.run_lab(MockClient([_reply({"grow_thresh": 0.15})]), iterations=1,
                      seeds=SEEDS, ticks=TICKS, probe_ticks=PROBE,
                      out_dir=str(tmp_path), shots=False)
    for path in (Path(out["dir"]) / "best_genome.json",
                 tmp_path / "best_genome.json"):
        g = B.Genome.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert g.program.to_dict() == out["best"]["spec"]["program"]
        Mind(g)                                  # it loads and runs, not just parses


def test_a_proposal_is_written_on_top_of_the_best_not_the_defaults(tmp_path):
    """The bug this pins down was found in a live run and is the kind that hides: the
    prompt promised omitted fields keep their value, and the code applied every
    proposal to the DEFAULTS, so a field the model chose not to mention silently
    reverted. The model believed it was refining one program; it was drawing
    independent samples around the default."""
    client = MockClient([_reply({"grow_thresh": 0.09}, "grow hard"),
                         _reply({"chem_decay": 0.21}, "fade faster")])
    out = lab.run_lab(client, iterations=2, seeds=SEEDS, ticks=TICKS,
                      probe_ticks=PROBE, out_dir=str(tmp_path), shots=False)
    first, second = out["board"][1], out["board"][2]
    if second["fitness"] >= first["fitness"] or first["fitness"] >= out["board"][0]["fitness"]:
        pass                                   # ordering is irrelevant to the claim
    base = max(out["board"][:2], key=lambda e: e["fitness"])["spec"]["program"]
    assert second["spec"]["program"]["chem_decay"] == pytest.approx(0.21)
    for k, v in base.items():
        if k != "chem_decay":
            assert second["spec"]["program"][k] == v, \
                f"{k} moved without being asked to"


def test_the_model_is_told_which_program_it_is_editing(tmp_path):
    client = MockClient([_reply({"grow_thresh": 0.09}), _reply({"chem_decay": 0.21})])
    lab.run_lab(client, iterations=2, seeds=SEEDS, ticks=TICKS, probe_ticks=PROBE,
                out_dir=str(tmp_path), shots=False)
    last = client.calls[-1][-1]["content"]
    assert "written on top of iteration" in last
    assert "What actually changed" in last


def test_the_journal_reports_what_moved_not_the_whole_vector(tmp_path):
    client = MockClient([_reply({"grow_thresh": 0.09}, "grow hard")])
    out = lab.run_lab(client, iterations=1, seeds=SEEDS, ticks=TICKS,
                      probe_ticks=PROBE, out_dir=str(tmp_path), shots=False)
    assert set(out["board"][1]["changed"]) == {"grow_thresh"}
