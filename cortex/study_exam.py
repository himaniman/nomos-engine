"""The cloud study's verdict: the laws of all six runs on worlds no run ever saw.

    python -m cortex.study_exam            # results exist = only the table is printed

The FINAL worlds of `docs/cloud-study.md` are 32 and were shared by all runs; a paired
difference on them has SE ~2.3, and finalists were picked by looking at them. Here every
law lives 128 fresh worlds through the lab's own `law_lab._life` (SE ~1.1).
Results go to runs/cloud-lab/exam/results.json; a law already in there is not re-lived.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cortex import atlas_lineage as L
from cortex import law_lab as LL

ROOT = Path("runs/cloud-lab")
SEEDS = list(range(1000, 1128))
ARMS = ["full-1", "full-2", "blind-1", "blind-2", "fresh-1", "fresh-2"]
EXTRA = {"full-1": [35], "full-2": [18], "fresh-1": [11]}     # beat their champion on FINAL


def laws() -> dict[str, str]:
    out = {"reference": Path("laws/atlas_wave.nomos").read_text(encoding="utf-8"),
           "confine": Path("laws/atlas_confine.nomos").read_text(encoding="utf-8")}
    for a in ARMS:
        s = json.loads((ROOT / a / "state.json").read_text(encoding="utf-8"))
        out[f"{a} control"] = s["control"]["best"]
        born = [i for i in s["islands"] if not i.get("given") and i.get("holdout") and i.get("train")]
        ch = max(born, key=lambda i: i["holdout"]["fitness"])  # summary.md's champion rule
        out[f"{a} champion #{ch['id']}"] = ch["best"]
        for iid in EXTRA.get(a, []):
            out[f"{a} #{iid}"] = next(i for i in s["islands"] if i["id"] == iid)["best"]
    s = json.loads((ROOT / "fresh-2" / "state.json").read_text(encoding="utf-8"))
    out["fresh-2 #9 as written"] = next(i for i in s["islands"] if i["id"] == 9)["law"]
    return out


def table(res: dict) -> str:
    m = {k: np.array(v["meals"], float) for k, v in res.items()}

    def vs(a: str, b: str) -> str:
        d = m[a] - m[b]
        return f"{d.mean():+5.1f} +- {d.std(ddof=1) / math.sqrt(len(d)):.1f}"

    rows = [f"{'law':<26} meals  immortal  vs confine     vs own control"]
    for k in m:
        arm = k.split()[0]
        ctrl = f"{arm} control"
        own = vs(k, ctrl) if ctrl in m and k != ctrl else ""
        conf = vs(k, "confine") if k != "confine" else ""
        rows.append(f"{k:<26} {m[k].mean():5.1f}  {sum(res[k]['immortal']):>3}/{len(SEEDS)}   "
                    f"{conf:<13}  {own}")
    return "\n".join(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jobs", type=int, default=18)
    p.add_argument("--ticks", type=int, default=8000)
    args = p.parse_args(argv)
    path = ROOT / "exam" / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = {k: v for k, v in laws().items() if k not in res}
    prog = L.Program({}).to_dict()
    if todo:
        with Pool(args.jobs) as pool:
            for name, text in todo.items():
                rows = pool.map(LL._life, [(text, s, args.ticks, "default", prog) for s in SEEDS],
                                chunksize=1)
                res[name] = {"meals": [r["meals"] for r in rows],
                             "immortal": [r["immortal"] for r in rows]}
                path.write_text(json.dumps(res), encoding="utf-8")
                print(f"{name}: {np.mean(res[name]['meals']):.1f}", flush=True)
    print(table(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
