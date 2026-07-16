from nomos import brain_lab, llm

GOOD = ("HYPOTHESIS: food into motor.\n```nomos\nchannel food\nchannel motor: init 0\n"
        "rule think:\n  when 1\n  set motor = 0.5*avg8(motor) + food\n```\n"
        "PREDICTION: eats a few.")
BAD = ("HYPOTHESIS: oops, no motor.\n```nomos\nchannel food\n"
       "rule r: when 1 set food = 1\n```\nPREDICTION: nothing.")


def test_evaluate_scores_a_brain():
    law = "channel food\nchannel motor: init 0\nrule t: when 1 set motor = food\n"
    res = brain_lab.evaluate(law, seeds=(0, 1), world_size=24, game_size=10,
                             moves=40, ticks=3)
    assert {"score", "scores", "survival", "immortal"} == set(res)
    assert len(res["scores"]) == 2


def test_brain_lab_runs_scores_and_journals(tmp_path):
    client = llm.MockClient([GOOD, GOOD])
    out = brain_lab.run_brain_lab(client, iterations=2, journal_path=tmp_path / "b.md",
                                  eval_seeds=(0, 1), world_size=24, game_size=10,
                                  moves=40, ticks=3)
    assert out["best"] is not None
    text = (tmp_path / "b.md").read_text(encoding="utf-8")
    assert "## Brain 1" in text and "## Brain 2" in text
    assert "Game:" in text
    # the second turn carried a leaderboard back to the model
    assert "LEADERBOARD" in client.calls[-1][-1]["content"]


def test_brain_lab_rejects_non_embodied_law(tmp_path):
    client = llm.MockClient([BAD, GOOD])
    out = brain_lab.run_brain_lab(client, iterations=1, journal_path=tmp_path / "b.md",
                                  eval_seeds=(0,), world_size=24, game_size=10,
                                  moves=30, ticks=2)
    text = (tmp_path / "b.md").read_text(encoding="utf-8")
    assert "(invalid)" in text          # the bad brain was rejected
    assert "## Brain 1" in text         # and it recovered to a real one
    assert out["best"] is not None
