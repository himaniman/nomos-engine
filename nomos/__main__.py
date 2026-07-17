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

from nomos import archive
from nomos.dsl import NomosError, parse_program
from nomos.engine import World
from nomos.metrics import summarize
from nomos.render import frames_to_images, save_gif, save_png, split_images


def _observe_frame(world: World, cap: int) -> dict:
    """Snapshot every channel for the observer, downsampled to <= cap per axis."""
    h, w = world.shape
    sy = max(1, h // cap)
    sx = max(1, w // cap)
    return {name: arr[::sy, ::sx].copy() for name, arr in world.state.items()}


def cmd_run(args: argparse.Namespace) -> int:
    text = Path(args.law).read_text(encoding="utf-8")
    program = parse_program(text)
    world = World(program, size=args.size, seed=args.seed)

    channels = args.channels.split(",") if args.channels else [program.channels[0].name]
    for ch in channels:
        if ch not in world.state:
            raise NomosError(f"--channels: unknown channel {ch!r}")

    frames = {ch: [world.state[ch].copy()] for ch in channels}
    history = [_observe_frame(world, args.obs_cap)]   # all channels, for temporal metrics
    activity = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        activity.append(world.step())
        if step % args.every == 0:
            for ch in channels:
                frames[ch].append(world.state[ch].copy())
            history.append(_observe_frame(world, args.obs_cap))
    elapsed = time.perf_counter() - t0

    report = summarize(world, activity, history=history)
    report["law"] = str(args.law)
    report["law_name"] = Path(args.law).stem
    report["elapsed_sec"] = round(elapsed, 2)

    if args.archive:
        run_id = archive.record(args.archive, report, text, law_name=Path(args.law).stem)
        report["archive_id"] = run_id

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


def cmd_play(args: argparse.Namespace) -> int:
    from nomos.embodied import play

    text = Path(args.law).read_text(encoding="utf-8")
    want_frames = bool(args.gif)
    result = play(text, world_size=args.world_size, game_size=args.game_size,
                  seed=args.seed, moves=args.moves, ticks_per_move=args.ticks,
                  life_channel=args.life_channel, collect_frames=want_frames,
                  sensor_gain=args.sensor_gain, starve_every=args.starve_every)

    if args.gif:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        images = split_images(result["life_frames"], result["game_frames"],
                              scale=args.scale,
                              life_frames2=result["avoid_frames"] or None)
        save_gif(images, args.gif, fps=args.fps)

    summary = {k: result[k] for k in
               ("score", "steps_alive", "moves", "alive_at_end", "world_size",
                "game_size", "seed")}
    print(json.dumps(summary, indent=2))
    print(f"snake: score={result['score']} survived {result['steps_alive']} ticks"
          + (f", wrote {args.gif}" if args.gif else ""), file=sys.stderr)
    return 0


def cmd_forage(args: argparse.Namespace) -> int:
    from nomos.embodied import play_forage

    text = Path(args.law).read_text(encoding="utf-8")
    want_frames = bool(args.gif)
    result = play_forage(text, world_size=args.world_size, game_size=args.game_size,
                         seed=args.seed, moves=args.moves, ticks_per_move=args.ticks,
                         life_channel=args.life_channel, collect_frames=want_frames,
                         sensor_gain=args.sensor_gain, food_count=args.food_count,
                         starve_after=args.starve_after, wall_radius=args.wall_radius,
                         view_radius=args.view_radius)

    if args.gif:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        images = split_images(result["life_frames"], result["game_frames"],
                              scale=args.scale,
                              life_frames2=result["avoid_frames"] or None)
        save_gif(images, args.gif, fps=args.fps)

    summary = {k: result[k] for k in
               ("score", "steps_alive", "moves", "alive_at_end", "world_size",
                "game_size", "seed")}
    print(json.dumps(summary, indent=2))
    print(f"forage: score={result['score']} survived {result['steps_alive']} moves"
          + (f", wrote {args.gif}" if args.gif else ""), file=sys.stderr)
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    from nomos.embodied import play
    from nomos.evolve import evolve, law_from_genome

    gif_dir = Path(args.gif_dir) if args.gif_dir and args.gif_dir.lower() != "none" else None
    seeds = tuple(range(args.eval_seeds))

    def on_generation(gen, genome, fit):
        print(f"[gen {gen}] best score={fit['score']:.2f} survival={fit['survival']:.0f} "
              f"fitness={fit['fitness']:.1f} genome={ {k: round(v, 3) for k, v in genome.items()} }",
              file=sys.stderr)
        if gif_dir is not None:
            r = play(law_from_genome(genome), world_size=args.world_size,
                     game_size=args.game_size, seed=args.show_seed, moves=args.moves,
                     ticks_per_move=args.ticks, collect_frames=True)
            gif_dir.mkdir(parents=True, exist_ok=True)
            images = split_images(r["life_frames"], r["game_frames"], scale=args.scale,
                                  life_frames2=r["avoid_frames"] or None)
            path = gif_dir / f"gen{gen:02d}-score{r['score']}.gif"
            save_gif(images, str(path), fps=args.fps)
            print(f"         wrote {path} (show-seed score={r['score']})", file=sys.stderr)

    result = evolve(args.generations, lam=args.lam, seeds=seeds,
                    world_size=args.world_size, game_size=args.game_size,
                    moves=args.moves, ticks=args.ticks, rng_seed=args.seed,
                    on_generation=on_generation)

    best = {"best_genome": {k: round(v, 4) for k, v in result["best_genome"].items()},
            "best_score": result["best_fitness"]["score"],
            "best_survival": result["best_fitness"]["survival"],
            "best_fitness": result["best_fitness"]["fitness"]}
    print(json.dumps(best, indent=2))
    if args.save_law:
        Path(args.save_law).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_law).write_text(law_from_genome(result["best_genome"]),
                                       encoding="utf-8")
        print(f"best brain law -> {args.save_law}", file=sys.stderr)
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
    run.add_argument("--archive", default=None, help="append the run to this SQLite archive")
    run.add_argument("--obs-cap", type=int, default=64, dest="obs_cap",
                     help="downsample observer frames to <= this per axis (default 64)")
    run.set_defaults(func=cmd_run)

    play = sub.add_parser("play", help="let a cell-grid brain drive the Snake game")
    play.add_argument("law", help="path to an embodied .nomos law (declares food + motor)")
    play.add_argument("--world-size", type=int, default=48, dest="world_size",
                      help="brain grid side (default 48)")
    play.add_argument("--game-size", type=int, default=16, dest="game_size",
                      help="snake board side (default 16)")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--moves", type=int, default=1000,
                      help="max game ticks (safety cap; the run ends at death)")
    play.add_argument("--ticks", type=int, default=64,
                      help="physics steps the brain thinks per game tick (default 64)")
    play.add_argument("--sensor-gain", type=float, default=0.8, dest="sensor_gain",
                      help="how far the food bump sits from centre (default 0.8)")
    play.add_argument("--starve-every", type=int, default=50, dest="starve_every",
                      help="moves without food before the snake shrinks (default 50)")
    play.add_argument("--life-channel", default="motor", dest="life_channel",
                      help="which channel to render on the left (default motor)")
    play.add_argument("--scale", type=int, default=6, help="upscale factor for the GIF")
    play.add_argument("--fps", type=int, default=6, help="GIF playback speed")
    play.add_argument("--gif", default=None, help="write the split life|game GIF here")
    play.set_defaults(func=cmd_play)

    fo = sub.add_parser("forage", help="let a cell-grid brain play the walled forage world")
    fo.add_argument("law", help="path to an embodied .nomos law (declares food + motor; "
                                "optional wall, temperature, avoid)")
    fo.add_argument("--world-size", type=int, default=128, dest="world_size",
                    help="brain grid side (default 128; 512 for the dense showcase brain)")
    fo.add_argument("--game-size", type=int, default=96, dest="game_size",
                    help="forage map side (default 96 — a wide world)")
    fo.add_argument("--seed", type=int, default=0)
    fo.add_argument("--moves", type=int, default=400,
                    help="max game moves (safety cap; the run ends at starvation)")
    fo.add_argument("--ticks", type=int, default=32,
                    help="physics steps the brain thinks per game move (default 32)")
    fo.add_argument("--sensor-gain", type=float, default=0.9, dest="sensor_gain",
                    help="how far the ego view spreads from centre (default 0.9)")
    fo.add_argument("--food-count", type=int, default=None, dest="food_count",
                    help="food items alive on the map at once (default: area / 768)")
    fo.add_argument("--starve-after", type=int, default=120, dest="starve_after",
                    help="moves without food before the player starves (default 120)")
    fo.add_argument("--wall-radius", type=int, default=None, dest="wall_radius",
                    help="wall vision radius in map cells (default: min(8, map side / 4))")
    fo.add_argument("--view-radius", type=int, default=16, dest="view_radius",
                    help="egocentric game camera radius for the GIF (default 16)")
    fo.add_argument("--life-channel", default="motor", dest="life_channel",
                    help="which channel to render on the left (default motor)")
    fo.add_argument("--scale", type=int, default=6, help="upscale factor for the GIF")
    fo.add_argument("--fps", type=int, default=6, help="GIF playback speed")
    fo.add_argument("--gif", default=None, help="write the split life|game GIF here")
    fo.set_defaults(func=cmd_forage)

    ev = sub.add_parser("evolve", help="evolve a Snake brain across runs (Phase B)")
    ev.add_argument("--generations", type=int, default=8)
    ev.add_argument("--lam", type=int, default=6, help="offspring per generation")
    ev.add_argument("--eval-seeds", type=int, default=3, dest="eval_seeds",
                    help="boards each brain is scored on (seeds 0..N-1)")
    ev.add_argument("--seed", type=int, default=0, help="RNG seed for the evolution itself")
    ev.add_argument("--show-seed", type=int, default=0, dest="show_seed",
                    help="board seed used for the per-generation GIF")
    ev.add_argument("--world-size", type=int, default=48, dest="world_size")
    ev.add_argument("--game-size", type=int, default=16, dest="game_size")
    ev.add_argument("--moves", type=int, default=600, help="max game ticks (cap; ends at death)")
    ev.add_argument("--ticks", type=int, default=64)
    ev.add_argument("--scale", type=int, default=6)
    ev.add_argument("--fps", type=int, default=6)
    ev.add_argument("--gif-dir", default="out/evolve", dest="gif_dir",
                    help="write a best-of-generation GIF here ('none' = off)")
    ev.add_argument("--save-law", default=None, dest="save_law",
                    help="write the best brain law here at the end")
    ev.set_defaults(func=cmd_evolve)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except NomosError as e:
        print(f"nomos: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
