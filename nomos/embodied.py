"""Couple a cell-grid 'brain' to the Snake game (Phase A: coupling only).

The grid is treated as a controller: it SENSES the game through a read-only
`food` channel (a bump pointing toward the food, egocentric to the snake head)
and ACTS on it through a `motor` channel (the harness reads where the motor
activity concentrates and turns the snake that way). The game ticks once per
`ticks_per_move` physics steps, so the brain has time to organize between moves.

No learning here yet — this is the sense->organize->act loop. Whether any given
law produces goal-directed motion is the experiment. Fully deterministic: a run
is a pure function of (law, world size, game size, seed, cadence, moves).
"""

from __future__ import annotations

import numpy as np

from nomos.dsl import NomosError, parse_program
from nomos.engine import World
from nomos.game import DIRECTIONS, Snake

SENSOR_CHANNEL = "food"
MOTOR_CHANNEL = "motor"


def food_sensor(world_shape: tuple[int, int], snake: Snake, gain: float = 0.8,
                sigma: float | None = None) -> np.ndarray:
    """A Gaussian bump on the world grid pointing toward the food, egocentric to
    the head: food to the right -> bump right of centre, etc. Direction only;
    amplitude is constant. Zero field when there is no food."""
    H, W = world_shape
    if snake.food is None:
        return np.zeros((H, W), dtype=np.float64)
    hy, hx = snake.head()
    gh, gw = snake.h, snake.w
    # shortest toroidal offset head->food, as a fraction of the board in [-.5,.5]
    dy = (((snake.food[0] - hy + gh // 2) % gh) - gh // 2) / gh
    dx = (((snake.food[1] - hx + gw // 2) % gw) - gw // 2) / gw
    by = int(round(H / 2 + dy * H * gain))
    bx = int(round(W / 2 + dx * W * gain))
    by = min(max(by, 0), H - 1)
    bx = min(max(bx, 0), W - 1)
    yy, xx = np.ogrid[:H, :W]
    sig = sigma if sigma is not None else max(H, W) / 8.0
    return np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2.0 * sig ** 2)))


def read_move(world: World, channel: str = MOTOR_CHANNEL) -> str | None:
    """Turn the motor field into a move: the direction of its (non-negative)
    centre of mass from the grid centre. None when the motor is silent, so the
    snake keeps its heading."""
    m = world.state.get(channel)
    if m is None:
        return None
    w = np.clip(m, 0.0, None)
    total = float(w.sum())
    if total <= 1e-9:
        return None
    H, W = m.shape
    yy, xx = np.ogrid[:H, :W]
    cy = float((w * yy).sum()) / total - H / 2.0
    cx = float((w * xx).sum()) / total - W / 2.0
    if abs(cy) < 1e-9 and abs(cx) < 1e-9:
        return None
    if abs(cy) >= abs(cx):
        return "up" if cy < 0 else "down"
    return "left" if cx < 0 else "right"


def require_embodied(program) -> None:
    names = {c.name for c in program.channels}
    for req in (SENSOR_CHANNEL, MOTOR_CHANNEL):
        if req not in names:
            raise NomosError(f"an embodied law must declare a {req!r} channel "
                             f"(sensor {SENSOR_CHANNEL!r}, motor {MOTOR_CHANNEL!r})")


def play(law_text: str, world_size: int = 48, game_size: int = 16, seed: int = 0,
         moves: int = 200, ticks_per_move: int = 4, life_channel: str = MOTOR_CHANNEL,
         collect_frames: bool = False, sensor_gain: float = 0.8) -> dict:
    """Run the coupled brain+game loop. Returns score, survival, move sequence,
    and (optionally) paired life/game frames for a split render."""
    program = parse_program(law_text)
    require_embodied(program)
    if life_channel not in {c.name for c in program.channels}:
        raise NomosError(f"--life-channel {life_channel!r} is not a declared channel")

    world = World(program, world_size, seed)
    snake = Snake(game_size, seed)
    life_frames: list[np.ndarray] = []
    game_frames: list[np.ndarray] = []
    move_seq: list[str | None] = []

    for _ in range(moves):
        if not snake.alive:
            break
        sensor = food_sensor(world.shape, snake, gain=sensor_gain)
        for _ in range(ticks_per_move):
            world.state[SENSOR_CHANNEL] = sensor      # re-assert the read-only sense
            world.step()
        move = read_move(world, MOTOR_CHANNEL)
        snake.step(move)
        move_seq.append(move)
        if collect_frames:
            life_frames.append(world.state[life_channel].copy())
            game_frames.append(snake.field())

    return {
        "score": snake.score,
        "steps_alive": snake.steps,
        "moves": len(move_seq),
        "alive_at_end": snake.alive,
        "move_seq": move_seq,
        "world_size": list(world.shape),
        "game_size": [snake.h, snake.w],
        "seed": seed,
        "life_frames": life_frames,
        "game_frames": game_frames,
    }
