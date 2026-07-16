import numpy as np

from nomos import archive, llm, scientist
from nomos.sim import digest, simulate


def test_extract_law_from_fenced_block():
    reply = ("HYPOTHESIS: things.\n```nomos\nchannel a\nrule r: when 1 add a = 1\n```\n"
             "PREDICTION: stuff.")
    law = scientist.extract_law(reply)
    assert law == "channel a\nrule r: when 1 add a = 1"


def test_extract_law_fallback_without_fence():
    reply = "some text channel a\nrule r: when 1 add a = 1"
    assert "channel a" in scientist.extract_law(reply)


def test_extract_law_none_when_absent():
    assert scientist.extract_law("just prose, no law here") is None


def test_simulate_and_digest_roundtrip():
    law = "channel alive: init bernoulli(0.2)\nrule d: when 1 set alive = sum8(alive) == 3\n"
    report = simulate(law, size=32, steps=30, seed=1)
    d = digest(report)
    assert "verdict:" in d
    assert "alive:" in d


def test_mock_client_scripts_replies():
    c = llm.MockClient(["one", "two"])
    assert c.chat([{"role": "user", "content": "x"}]) == "one"
    assert c.chat([{"role": "user", "content": "y"}]) == "two"
    assert len(c.calls) == 2


def test_loop_runs_and_writes_journal(tmp_path):
    reply = ("HYPOTHESIS: life persists.\n```nomos\n"
             "channel alive: init bernoulli(0.3)\n"
             "rule birth: when alive == 0 and sum8(alive) == 3 set alive = 1\n"
             "rule death: when alive == 1 and (sum8(alive) < 2 or sum8(alive) > 3) set alive = 0\n"
             "```\nPREDICTION: stays alive.")
    client = llm.MockClient([reply, reply, reply])
    jpath = tmp_path / "j.md"
    scientist.run_loop(client, iterations=2, size=32, steps=20, seed=1,
                       archive_path=str(tmp_path / "runs.db"), journal_path=jpath)
    text = jpath.read_text(encoding="utf-8")
    assert "## Experiment 1" in text and "## Experiment 2" in text
    assert "Observer:" in text
    assert "behavior-map:" in text
    # the loop fed observations back: system + 2*(user+assistant) at least
    assert len(client.calls[-1]) >= 4
    # the follow-up told the model its map cell
    assert "behavior-map:" in client.calls[-1][-1]["content"]


def test_extract_hypothesis():
    reply = ("HYPOTHESIS: diffusion smooths\nnoise into patches.\n\n"
             "```nomos\nchannel a\n```\nPREDICTION: flat.")
    assert scientist.extract_hypothesis(reply) == "diffusion smooths noise into patches."
    assert scientist.extract_hypothesis("no marker here") == ""


def test_memory_block_feeds_past_runs_into_first_turn(tmp_path):
    db = str(tmp_path / "runs.db")
    report = {"size": [32, 32], "seed": 1, "steps": 20, "verdict": "alive: waves",
              "activity": {"mean": 0.42, "final": 0.4, "trend": "steady"}}
    archive.record(db, report, "channel w\nrule r: when 1 add w = 1", "waves",
                   hypothesis="waves persist")
    reply = ("HYPOTHESIS: building on waves.\n```nomos\n"
             "channel a: init bernoulli(0.3)\nrule r: when 1 set a = sum8(a) >= 4\n"
             "```\nPREDICTION: patches.")
    client = llm.MockClient([reply])
    scientist.run_loop(client, iterations=1, size=32, steps=20, seed=1,
                       archive_path=db, journal_path=tmp_path / "j.md")
    first_user = client.calls[0][1]["content"]
    assert "MEMORY" in first_user and "waves persist" in first_user
    assert "channel w" in first_user and "alive: waves" in first_user
    assert "BEHAVIOR MAP" in first_user
    # and the new run was archived with its own hypothesis
    assert archive.memory(db)[0]["hypothesis"] == "building on waves."


def test_memory_block_empty_without_archive(tmp_path):
    assert scientist.memory_block(str(tmp_path / "absent.db"), 10) == ""


def test_loop_writes_gif_per_experiment(tmp_path):
    reply = ("HYPOTHESIS: patches.\n```nomos\nchannel a: init bernoulli(0.3)\n"
             "rule r: when 1 set a = sum8(a) >= 4\n```\nPREDICTION: blobs.")
    client = llm.MockClient([reply])
    gif_dir = tmp_path / "gifs"
    jpath = tmp_path / "j.md"
    scientist.run_loop(client, iterations=1, size=32, steps=20, seed=1,
                       archive_path=None, journal_path=jpath, gif_dir=gif_dir)
    gifs = list(gif_dir.glob("*.gif"))
    assert len(gifs) == 1
    assert gifs[0].stat().st_size > 0
    assert "![run 1]" in jpath.read_text(encoding="utf-8")


def test_simulate_frames_out_collects_render_frames():
    law = "channel a: init bernoulli(0.3)\nrule r: when 1 set a = sum8(a) >= 4\n"
    frames: dict = {}
    simulate(law, size=32, steps=20, seed=1, frames_out=frames, samples=10)
    assert "a" in frames
    assert len(frames["a"]) >= 2 and frames["a"][0].shape == (32, 32)


def test_loop_recovers_from_a_bad_law(tmp_path):
    bad = "HYPOTHESIS: oops.\n```nomos\nrule r: when ghost > 0 set ghost = 1\n```\nPREDICTION: x."
    good = ("HYPOTHESIS: fixed.\n```nomos\nchannel a: init bernoulli(0.3)\n"
            "rule r: when 1 set a = sum8(a) >= 4\n```\nPREDICTION: patches.")
    client = llm.MockClient([bad, good])
    jpath = tmp_path / "j.md"
    scientist.run_loop(client, iterations=1, size=32, steps=20, seed=1,
                       archive_path=None, journal_path=jpath)
    text = jpath.read_text(encoding="utf-8")
    assert "law error" in text            # the bad attempt was logged
    assert "## Experiment 1" in text      # and the loop recovered to a real run
