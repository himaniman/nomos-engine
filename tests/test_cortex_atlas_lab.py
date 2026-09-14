"""The atlas lab: an LLM (or rng) writes the map's growth program, and every candidate
is lived on the same worlds so a score is a fact about the program."""

import json
from pathlib import Path

from cortex import atlas_lab as LAB
from cortex import atlas_lineage as L
from nomos.llm import make_client


def test_the_loop_with_a_scripted_model(tmp_path):
    client = make_client(mock=["Smaller map, cheaper.\n```json\n{\"radius\": 6}\n```",
                               "no fence here {\"decay\": 0.09, \"bogus\": 1}"])
    out = LAB.run_lab(client, iterations=2, seeds=(1,), ticks=300, jobs=1,
                      out_dir=str(tmp_path))
    board = out["board"]
    assert [e["iter"] for e in board] == [0, 1, 2]
    assert board[1]["program"]["radius"] == 6 and board[1]["changed"] == {"radius": 6}
    # the second reply had no fence and a made-up field: the field is dropped, the
    # program still carries the best-so-far's radius, and decay moved
    assert "bogus" not in board[2]["program"]
    assert board[2]["program"]["decay"] == 0.09
    run = Path(out["dir"])
    assert (run / "journal.md").exists() and (run / "best_program.json").exists()
    best = json.loads((run / "best_program.json").read_text())
    assert best == max(board, key=lambda e: e["fitness"])["program"]
    # the model was told the knobs and the reflex's bar, never a weight
    first = client.calls[0][1]["content"]
    assert "radius" in first and "reflex" in first and "W1" not in first


def test_rng_proposer_moves_something():
    prop = LAB.RngProposer(seed=3)
    spec, hyp = prop.propose(L.Program())
    assert spec and all(k in L.FIELDS for k in spec)
