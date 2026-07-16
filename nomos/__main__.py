"""CLI: run one universe from a law file.

    python -m nomos run laws/game_of_life.nomos --steps 300 --gif out/gol.gif

The runner is deliberately a pure function of its arguments: law file + size
+ seed + steps fully determine the result. That makes runs reproducible,
diffable and trivially parallel (locally or as k8s Jobs later).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from nomos.dsl import NomosError, parse_program
from nomos.engine import World
from nomos.metrics import summarize
from nomos.render import frames_to_images, save_gif, save_png


def cmd_run(args: argparse.Namespace) -> int:
    text = Path(args.law).read_text(encoding="utf-8")
    program = parse_program(text)
    world = World(program, size=args.size, seed=args.seed)

    channels = args.channels.split(",") if args.channels else [program.channels[0].name]
    for ch in channels:
        if ch not in world.state:
            raise NomosError(f"--channels: unknown channel {ch!r}")

    frames = {ch: [world.state[ch].copy()] for ch in channels}
    activity = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        activity.append(world.step())
        if step % args.every == 0:
            for ch in channels:
                frames[ch].append(world.state[ch].copy())
    elapsed = time.perf_counter() - t0

    report = summarize(world, activity)
    report["law"] = str(args.law)
    report["elapsed_sec"] = round(elapsed, 2)

    outputs = []
    if args.gif:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        images = frames_to_images(frames, scale=args.scale, cmap=args.cmap)
        save_gif(images, args.gif, fps=args.fps)
        outputs.append(args.gif)
    if args.png:
        Path(args.png).parent.mkdir(parents=True, exist_ok=True)
        images = frames_to_images({ch: [frames[ch][-1]] for ch in channels},
                                  scale=args.scale, cmap=args.cmap)
        save_png(images[-1], args.png)
        outputs.append(args.png)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        outputs.append(args.json)

    print(json.dumps(report, indent=2))
    if outputs:
        print("wrote:", ", ".join(outputs), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nomos", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one universe from a law file")
    run.add_argument("law", help="path to a .nomos law file")
    run.add_argument("--size", type=int, default=128, help="grid side (default 128)")
    run.add_argument("--steps", type=int, default=300, help="ticks to simulate")
    run.add_argument("--seed", type=int, default=0, help="RNG seed (fixes init and rand())")
    run.add_argument("--channels", default=None,
                     help="channels to render: one -> colormap, 2-3 -> RGB (default: first declared)")
    run.add_argument("--every", type=int, default=1, help="sample a frame every N steps")
    run.add_argument("--scale", type=int, default=4, help="upscale factor for output images")
    run.add_argument("--fps", type=int, default=25, help="GIF playback speed")
    run.add_argument("--cmap", choices=["fire", "gray"], default="fire")
    run.add_argument("--gif", default=None, help="write animated GIF here")
    run.add_argument("--png", default=None, help="write final-frame PNG here")
    run.add_argument("--json", default=None, help="write the observer report here")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except NomosError as e:
        print(f"nomos: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
