"""The law lab: a model (or rng) rewrites the atlas's rule as a nomos law; a broken law
goes back with its error, a working one is lived and boarded."""

from pathlib import Path

from cortex import atlas_law as LAW
from cortex import law_lab as LAB
from nomos.llm import make_client

GOOD = """\
channel source
channel solid
channel value
channel echo: init 0

rule keep:
  when 1
  set echo = 0.5 * echo + 0.5 * value

rule wave:
  when 1
  set value = max(source, 0.9 * max8(value) + 0.05 * echo) * (1 - solid)
"""

BAD = "channel source\nchannel value\nrule r:\n  when 1\n  set value = squish(source)\n"


def test_the_loop_with_a_scripted_model(tmp_path):
    client = make_client(mock=[f"Add an echo channel.\n```nomos\n{BAD}```",
                               f"Fixed.\n```nomos\n{GOOD}```",
                               "I forgot the fence entirely."])
    out = LAB.run_lab(client, iterations=1, seeds=(1,), ticks=300, jobs=1,
                      out_dir=str(tmp_path), holdout=(2,))
    board = out["board"]
    assert [e["iter"] for e in board] == [0, 1]
    assert board[1]["law"].strip() == GOOD.strip()
    assert board[1]["hypothesis"] == "Fixed."
    # the bad law was bounced with its error and never scored
    bounce = client.calls[1][-1]["content"]
    assert "did not load" in bounce and "squish" in bounce
    run = Path(out["dir"])
    assert (run / "law01.nomos").read_text(encoding="utf-8").strip() == GOOD.strip()
    assert (run / "best.nomos").exists()
    journal = (run / "journal.md").read_text(encoding="utf-8")
    assert "## Holdout" in journal and "reference" in journal
    # the model saw the reference law, the DSL and the bar — never a weight
    system = client.calls[0][0]["content"]
    assert "max8" in system and "channel source" in system and "W1" not in system


def test_rng_control_moves_constants_only():
    ref = Path(LAW.REFERENCE).read_text(encoding="utf-8")
    text, hyp = LAB.RngProposer(seed=2).propose(ref)
    LAW.check(text)
    assert text.count("channel") == ref.count("channel")


def test_files_proposer_walks_its_list(tmp_path):
    a = tmp_path / "a.nomos"
    a.write_text("# no wall leak at all\n" + GOOD, encoding="utf-8")
    out = LAB.run_lab(LAB.FileProposer([str(a)]), iterations=1, seeds=(1,), ticks=200,
                      jobs=1, out_dir=str(tmp_path / "run"))
    assert out["board"][1]["hypothesis"].endswith("no wall leak at all")
