import numpy as np

from nomos import llm, scientist
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
    # the loop fed observations back: system + 2*(user+assistant) at least
    assert len(client.calls[-1]) >= 4


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
