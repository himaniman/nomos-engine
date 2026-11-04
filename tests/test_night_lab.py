"""The night lab: the model designs structure, the machine tunes constants, a control
arm tunes the ghost with the same budget — checked end to end with a mock model."""

import json

from cortex import night_lab as N

LAW = """\
channel source
channel solid
channel value

rule wave:
  when 1
  set value = max(source, 0.94 * max8(value)) * (1 - solid)  # the wave
"""


def test_structure_key_ignores_constants_and_comments():
    tweaked = LAW.replace("0.94", "0.9").replace("# the wave", "")
    assert N.structure_key(LAW) == N.structure_key(tweaked)
    other = LAW.replace("max8", "max4")
    assert N.structure_key(LAW) != N.structure_key(other)


def test_parse_design():
    hyp, pred = N.parse_design("HYPOTHESIS: remember the goal.\nIt helps.\n"
                               "PREDICTION: 23.5 meals\n```nomos\nchannel value\n```")
    assert hyp == "remember the goal. It helps." and pred == 23.5
    hyp, pred = N.parse_design("Just an idea.\n```nomos\nchannel value\n```")
    assert hyp == "Just an idea." and pred is None


def test_a_short_night_with_a_mock_model(tmp_path):
    lab = N.NightLab(N.NightMock(), tmp_path, hours=0.012, train=(0, 1), holdout=(2,),
                     ticks=300, jobs=1, tune=2, deep_every=3, theory_every=1,
                     observe_every=3, holdout_every=0.05, use_embed=False, grace=0.1,
                     final=(3,))
    lab.loop()
    s = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert s["done"]
    assert [i["id"] for i in s["islands"]][:2] == [0, 1] and s["islands"][1]["given"]
    assert not [t for t in s["turns"] if t.get("error") and "kind" in t
                and t["kind"] in ("observe", "theory")]      # side jobs never crash
    assert s["observations"] and s["theories"]
    turns = [t for t in s["turns"] if t.get("kind") in ("design", "deep")]
    assert any(t.get("repairs") for t in turns)       # a broken law went back
    # every valid law became an island or was caught as a twin of one
    assert len(s["islands"]) >= 3 or any(t.get("twin") for t in turns)
    assert s["evals"]["control"] > 0 and s["evals"]["llm"] > 0
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "CONTROL" in summary and "holdout" in summary and "FINAL" in summary


def test_track_image_draws_a_life():
    life = N.track_life(LAW, 0, 120)
    img = N.track_image(life, max_px=200)
    assert img.ndim == 3 and img.shape[2] == 3 and img.shape[0] >= 40
    assert life["stats"]["ticks"] == 120


def test_vocabulary_novelty_ignores_numbers():
    a = N.law_names(LAW)
    assert a == N.law_names(LAW.replace("0.94", "0.5"))
    assert "max8" in a and "value" in a and "when" not in a
    b = N.law_names(LAW.replace("max8(value)", "max8(value) * (1 - trail)"))
    assert N.jaccard(a, b) < 1.0


def test_mailbox_round_trip(tmp_path):
    import threading
    from cortex import mailbox as M
    box = tmp_path / "mailbox"
    (box).mkdir()
    (box / "0000.request.md").write_text("old", encoding="utf-8")   # a dead run's question
    client = M.MailboxClient(box, timeout=20, poll=0.05)
    assert (box / "0000.stale").exists() and M.pending(box) == []

    def answer():
        while not M.pending(box):
            pass
        n = M.pending(box)[0]
        (box / f"{n:04d}.reply.md").write_text("HYPOTHESIS: hi", encoding="utf-8")
    t = threading.Thread(target=answer, daemon=True)
    t.start()
    png = "data:image/png;base64,iVBORw0KGgo="
    reply = client.chat([{"role": "system", "content": "SYS"},
                         {"role": "user", "content": [{"type": "text", "text": "look"},
                                                      {"type": "image_url",
                                                       "image_url": {"url": png}}]}],
                        max_tokens=100)
    assert reply == "HYPOTHESIS: hi"
    req = (box / "0001.request.md").read_text(encoding="utf-8")
    assert "system-" in req and "0001.png" in req and (box / "0001.png").exists()
    assert M.pending(box) == []


def test_a_short_night_with_a_mock_model_and_extra_anchor(tmp_path):
    lab = N.NightLab(N.NightMock(), tmp_path, hours=0.004, train=(0,), holdout=(1,),
                     ticks=200, jobs=1, tune=1, use_embed=False, grace=0.1,
                     control_law="laws/atlas_confine.nomos",
                     anchors=["laws/atlas_ghost.nomos"])
    lab.loop()
    s = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    given = [i for i in s["islands"] if i.get("given")]
    assert len(given) == 3 and "Confinement" in given[1]["hypothesis"]
    assert s["control"]["evals"] >= 0 and s["done"]


def test_mailbox_names_the_kind_of_request():
    from cortex import mailbox as M
    assert M.kind_of(N.DESIGN_TURN).startswith("DESIGN")
    assert M.kind_of(N.THEORY_TURN).startswith("THEORY")
    assert M.kind_of(N.OBSERVE_TURN).startswith("OBSERVE")
    assert M.kind_of(N.DESIGN_TURN + "\nThat law was sent back: x").startswith("REPAIR")


def test_a_blind_night_has_no_observer(tmp_path):
    lab = N.NightLab(N.NightMock(), tmp_path, hours=0.004, train=(0,), holdout=(1,),
                     ticks=200, jobs=1, tune=1, use_embed=False, grace=0.1,
                     observe_every=0, theory_every=2)
    lab.loop()
    s = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert s["done"] and not s["observations"]
    assert not [t for t in s["turns"] if t.get("kind") == "observe"]
