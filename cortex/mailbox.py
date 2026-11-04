"""A mailbox model: the night lab's questions go to FILES and an agent in a shell answers
them — the coding agent itself (Claude in a cloud session) is the physicist, observer and
theorist, no API key, no external model.

    # the lab, in the background (a normal night with --agent):
    python -m cortex.night_lab --agent --hours 4 --dir out/cortex/night/cloud1 ...
    # the agent, in a loop:
    python -m cortex.mailbox next out/cortex/night/cloud1/mailbox --wait 600
    #   -> prints request 0007; the agent writes its answer to 0007.reply.md

The lab never waits on the agent for anything but the answer itself: tuning, the control
arm and holdout checks run while the agent thinks, exactly as they did while a slow local
model thought. One file per message so a crash loses nothing:

    system-<hash>.md      a system text, written once (the agent reads it once)
    NNNN.request.md       the conversation after the system text, newest message last
    NNNN.png              the picture of an OBSERVE request
    NNNN.reply.md         the agent's answer (plain text, the same shape the prompt asks)
    NNNN.stale            a request left unanswered by a previous run of the lab
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from nomos.llm import LLMError

_ID = re.compile(r"^(\d{4})\.request\.md$")


def _ids(box: Path, suffix: str) -> set[int]:
    return {int(p.name[:4]) for p in box.glob(f"[0-9][0-9][0-9][0-9].{suffix}")}


def _atomic(path: Path, data: bytes | str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, str):
        tmp.write_text(data, encoding="utf-8")
    else:
        tmp.write_bytes(data)
    tmp.replace(path)


_KINDS = (("That law was sent back", "REPAIR — fix the law and resend it whole"),
          ("Rewrite the THEORY", "THEORY — plain text, no law"),
          ("This is NOT a design turn", "OBSERVE — look at the picture, plain sentences"),
          ("Write a NEW STRUCTURE", "DESIGN — HYPOTHESIS, PREDICTION, ```nomos law```"))


def kind_of(text: str) -> str:
    """What the newest message asks for (the last marker in the request wins)."""
    best, at = "? — read the request", -1
    for marker, kind in _KINDS:
        i = text.rfind(marker)
        if i > at:
            best, at = kind, i
    return best


def pending(box: Path) -> list[int]:
    """Requests nobody has answered (and no earlier run abandoned), oldest first."""
    done = _ids(box, "reply.md") | _ids(box, "stale")
    return sorted(_ids(box, "request.md") - done)


class MailboxClient:
    """Speaks the `chat(messages, **kw)` contract of `nomos.llm` clients."""

    model = "agent-in-session (mailbox)"

    def __init__(self, box: Path, timeout: float = 3 * 3600, poll: float = 2.0):
        self.box, self.timeout, self.poll = Path(box), timeout, poll
        self.box.mkdir(parents=True, exist_ok=True)
        for n in pending(self.box):                     # a restarted lab asks afresh
            (self.box / f"{n:04d}.stale").write_text("abandoned by a restart\n",
                                                     encoding="utf-8")
        self.n = max(_ids(self.box, "request.md") | {-1}) + 1
        self.last: dict = {}

    def _system(self, text: str) -> str:
        h = hashlib.sha1(text.encode()).hexdigest()[:8]
        path = self.box / f"system-{h}.md"
        if not path.exists():
            _atomic(path, text)
        return path.name

    def chat(self, messages, temperature=None, max_tokens=None, thinking=False, **kw) -> str:
        n, self.n = self.n, self.n + 1
        parts = [f"# request {n:04d}\n"]
        hints = []
        if max_tokens:
            hints.append(f"keep the answer within ~{int(max_tokens * 0.75)} words")
        if thinking:
            hints.append("a DEEP turn: take your time")
        if hints:
            parts.append("_" + "; ".join(hints) + "_\n")
        for m in messages:
            content = m["content"]
            if m["role"] == "system":
                parts.append(f"\n## system: see `{self._system(content)}`\n")
                continue
            parts.append(f"\n## {m['role']}\n\n")
            if isinstance(content, str):
                parts.append(content + "\n")
                continue
            for c in content:                            # text + image_url blocks
                if c.get("type") == "text":
                    parts.append(c["text"] + "\n")
                elif c.get("type") == "image_url":
                    url = c["image_url"]["url"]
                    png = self.box / f"{n:04d}.png"
                    _atomic(png, base64.b64decode(url.split(",", 1)[1]))
                    parts.append(f"\n[picture: `{png.name}` — look at it]\n")
        _atomic(self.box / f"{n:04d}.request.md", "".join(parts))
        reply_path = self.box / f"{n:04d}.reply.md"
        t0 = time.time()
        while True:
            if reply_path.exists() and time.time() - reply_path.stat().st_mtime > 1.0:
                text = reply_path.read_text(encoding="utf-8").strip()
                if text:
                    self.last = {"seconds": time.time() - t0}
                    return text
            if time.time() - t0 > self.timeout:
                (self.box / f"{n:04d}.stale").write_text("timed out\n", encoding="utf-8")
                raise LLMError(f"mailbox request {n:04d}: no reply in {self.timeout:.0f} s")
            time.sleep(self.poll)


# --- the agent's side ------------------------------------------------------------------

def _status(box: Path) -> dict:
    p = box.parent / "status.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _brief(st: dict) -> str:
    if not st:
        return "status: (no heartbeat yet)"
    ch, co = st.get("champion") or {}, st.get("control") or {}
    return (f"status: {st.get('elapsed_h')} of {st.get('hours')} h, doing "
            f"'{st.get('doing')}', design turns {st.get('design_turns')}, islands "
            f"{st.get('islands')}, evals {st.get('evals')}, train leader (summary.md's "
            f"champion is the best by HOLDOUT) #{ch.get('id', '-')} train "
            f"{ch.get('train_meals', '-')} hold "
            f"{ch.get('holdout_meals', '-')} | control train {co.get('train_meals', '-')} "
            f"hold {co.get('holdout_meals', '-')}")


def cmd_next(box: Path, wait: float) -> int:
    t0 = time.time()
    while True:
        st = _status(box)
        if st.get("done"):
            print("NIGHT DONE — " + _brief(st))
            print(f"see {box.parent / 'summary.md'}")
            return 0
        todo = pending(box) if box.exists() else []
        if todo:
            n = todo[0]
            text = (box / f"{n:04d}.request.md").read_text(encoding="utf-8")
            print(f"REQUEST {n:04d}  ->  answer into {box / f'{n:04d}.reply.md'}")
            print(f"kind: {kind_of(text)}")
            if len(todo) > 1:
                print(f"(and {len(todo) - 1} more waiting)")
            print(_brief(_status(box)))
            print("-" * 72)
            print(text)
            return 0
        if time.time() - t0 > wait:
            print("no request yet — " + _brief(st))
            return 0
        time.sleep(2.0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="The agent's side of the night lab mailbox.")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("next", help="wait for the oldest unanswered request and print it")
    q.add_argument("box")
    q.add_argument("--wait", type=float, default=600.0, help="seconds to wait at most")
    s = sub.add_parser("status", help="the lab's heartbeat")
    s.add_argument("box")
    c = sub.add_parser("check", help="parse a law and show the value-field diagnosis the "
                                     "lab would give (no world is lived: free to use)")
    c.add_argument("law")
    args = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")     # a Windows console is not UTF-8
    except (AttributeError, ValueError):
        pass
    if args.cmd == "check":
        from cortex import atlas_law as LAW
        from cortex import night_lab as N
        from nomos.scientist import extract_law
        raw = Path(args.law).read_text(encoding="utf-8")
        text = extract_law(raw) or raw
        try:
            LAW.check(text)
        except Exception as e:
            print(f"REJECTED: {type(e).__name__}: {e}")
            return 1
        print(N.diagnose(text))
        for w in N.overwritten(text):
            print(f"warning: {w} — only the last set counts")
        return 0
    box = Path(args.box)
    if args.cmd == "next":
        return cmd_next(box, args.wait)
    print(_brief(_status(box)))
    print(f"pending: {pending(box) if box.exists() else []}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
