"""Train one brain to predict its senses, and ask what it can do with its eyes closed.

    python -m cortex.pred.bench --arm sheet --side 16 --steps 3000 --threads 6

The exam (validation worlds; --final = the test worlds, once): watch 32 frames, then close
the eyes for 32 while still feeling the body and the actions, and predict what would be
seen. Rival predictions: MEAN (know nothing), HOLD (the last picture seen) and ROTATE (the
last picture turned by the known change of heading: geometry done by hand). Skill = 1 -
error / the best rival's error, so 0 = no better than the best simple trick, on the
GEOMETRY of the view (how near, how solid), per 32 rays.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cortex.pred.model import ARMS, PLAIN, RETINA, build

DATA = Path("out/cortex/pred/data")
RUNS = Path("out/cortex/pred/runs")
GEOM = np.array([[6 * r + 4, 6 * r + 5] for r in range(32)]).ravel()
SIN_H, COS_H = 212 + 3, 212 + 4              # the inner senses start at 212
HORIZONS = [1, 2, 4, 8, 16, 32]
FOV = 2.0943951023931953


class Data:
    def __init__(self, split, mean=None, std=None):
        d = np.load(DATA / f"{split}.npz")
        self.raw, self.act, self.lens = d["sens"], d["act"], d["lens"]
        self.mean = self.raw.mean(0) if mean is None else mean
        self.std = np.maximum(self.raw.std(0), 0.05) if std is None else std
        self.sens = (self.raw - self.mean) / self.std
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]])

    def batch(self, rng, n, t):
        s, a = [], []
        for _ in range(n):
            i = rng.integers(len(self.lens))
            o = self.starts[i] + rng.integers(self.lens[i] - t - 1)
            s.append(self.sens[o:o + t + 1])
            a.append(self.act[o:o + t + 1])
        return torch.tensor(np.array(s)), torch.tensor(np.array(a))


def blinks(rng, n, t, burn):
    """Eyes open for `burn`, then random stretches open and closed."""
    e = np.ones((n, t), np.float32)
    for b in range(n):
        i = burn
        while i < t:
            i += rng.geometric(1 / 8)
            k = rng.integers(1, 33)
            e[b, i:i + k] = 0.0
            i += k
    return torch.tensor(e)


def rotated(last, dh):
    """HOLD turned by the heading change: each ray takes the value of the ray that looked
    that way before; rays beyond the old fan know only the mean (0, normalised)."""
    n = 32
    out = np.zeros((len(dh), n * 2), np.float32)
    idx = np.floor(np.arange(n)[None] + dh[:, None] / (FOV / n)).astype(int)
    ok = (idx >= 0) & (idx < n)
    clipped = np.clip(idx, 0, n - 1)
    for c in range(2):
        val = np.take_along_axis(last[:, c::2], clipped, 1)
        out[:, c::2] = np.where(ok, val, 0.0)
    return out


@torch.no_grad()
def exam(brain, data, burn=32, closed=32, stride=150, batch=64):
    n = burn + closed
    starts = [s + o for s, l in zip(data.starts, data.lens) for o in range(0, l - n - 2, stride)]
    rows = {k: {"model": [], "mean": [], "hold": [], "rot": [], "turned": []} for k in HORIZONS}
    open_err, open_hold = [], []
    for at in range(0, len(starts), batch):
        ss = starts[at:at + batch]
        s = torch.tensor(np.array([data.sens[o:o + n + 1] for o in ss]))
        a = torch.tensor(np.array([data.act[o:o + n + 1] for o in ss]))
        e = torch.ones(len(ss), n)
        e[:, burn:] = 0.0
        pred = brain.run(s[:, :n], a[:, :n], e)[0].numpy()
        sn = s.numpy()
        raw = np.array([data.raw[o:o + n + 1] for o in ss])
        head = np.arctan2(raw[:, :, SIN_H], raw[:, :, COS_H])
        last = sn[:, burn - 1][:, GEOM]              # the last frame actually seen
        # eyes open: pred[t] predicts frame t+1
        open_err.append(((pred[:, burn - 2][:, GEOM] - last) ** 2).mean(1))
        open_hold.append(((sn[:, burn - 2][:, GEOM] - last) ** 2).mean(1))
        for h in HORIZONS:
            # eyes closed from frame `burn` on; pred[burn-2+h] predicts frame burn-1+h
            tgt = sn[:, burn - 1 + h][:, GEOM]
            p = pred[:, burn - 2 + h][:, GEOM]
            dh = (head[:, burn - 1 + h] - head[:, burn - 1] + math.pi) % (2 * math.pi) - math.pi
            r = rotated(last, dh)
            rows[h]["model"].append(((p - tgt) ** 2).mean(1))
            rows[h]["mean"].append((tgt ** 2).mean(1))
            rows[h]["hold"].append(((last - tgt) ** 2).mean(1))
            rows[h]["rot"].append(((r - tgt) ** 2).mean(1))
            rows[h]["turned"].append(np.abs(dh) > 1.0)
    out = {"open": {"model": float(np.concatenate(open_err).mean()),
                    "hold": float(np.concatenate(open_hold).mean())}}
    out["open"]["skill"] = 1 - out["open"]["model"] / out["open"]["hold"]
    for h, v in rows.items():
        m = {k: np.concatenate(x) for k, x in v.items()}

        def score(mask):
            base = {k: float(m[k][mask].mean()) for k in ("mean", "hold", "rot")}
            err = float(m["model"][mask].mean())
            return {"err": err, **base, "skill": 1 - err / min(base.values()),
                    "n": int(mask.sum())}
        out[f"h{h}"] = score(np.ones_like(m["turned"]))
        if m["turned"].sum() > 20:
            out[f"h{h}_turned"] = score(m["turned"])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", required=True, choices=sorted(ARMS) + sorted(RETINA))
    p.add_argument("--side", type=int, default=16)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--burn", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--every", type=int, default=500)
    p.add_argument("--final", action="store_true", help="also sit the TEST worlds (once!)")
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train = Data("train")
    del train.raw                  # training reads only the normalised copy
    val = Data("val", train.mean, train.std)
    brain = build(args.arm, args.seed, args.side)
    plain = args.arm in PLAIN
    name = f"{args.arm}-s{args.seed}" if plain else f"{args.arm}-{args.side}-s{args.seed}"
    run = RUNS / name
    run.mkdir(parents=True, exist_ok=True)
    ckpt = run / "ckpt.pt"
    log = (run / "log.jsonl").open("a" if ckpt.exists() else "w", encoding="utf-8")
    print(f"{name}: core weights {brain.core_weights()}, "
          f"all {sum(x.numel() for x in brain.parameters())}", flush=True)
    opt = torch.optim.Adam(brain.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t0 = time.time()
    losses = []
    first = 1
    if ckpt.exists():              # a stopped run goes on where it stopped
        c = torch.load(ckpt, weights_only=False)
        brain.load_state_dict(c["brain"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        rng.bit_generator.state = c["rng"]
        torch.set_rng_state(c["torch"])
        first = c["step"] + 1
        print(f"{name}: resumed at step {first}", flush=True)
    for step in range(first, args.steps + 1):
        s, a = train.batch(rng, args.batch, args.length)
        e = blinks(rng, args.batch, args.length, args.burn)
        pred = brain.run(s[:, :-1], a[:, :-1], e)[0]
        loss = ((pred[:, args.burn:] - s[:, args.burn + 1:]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % args.every == 0 or step == args.steps:
            r = exam(brain, val)
            row = {"step": step, "s": round(time.time() - t0),
                   "loss": round(float(np.mean(losses[-100:])), 4),
                   "open": round(r["open"]["skill"], 3),
                   **{f"h{h}": round(r[f"h{h}"]["skill"], 3) for h in HORIZONS},
                   "h8_turned": round(r.get("h8_turned", {}).get("skill", float("nan")), 3)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(name, row, flush=True)
            torch.save({"brain": brain.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                        "torch": torch.get_rng_state(), "step": step}, ckpt)
    torch.save(brain.state_dict(), run / "brain.pt")
    result = {"args": vars(args), "core_weights": brain.core_weights(), "val": exam(brain, val)}
    if args.final:
        result["test"] = exam(brain, Data("test", train.mean, train.std))
    (run / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
