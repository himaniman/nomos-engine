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
DANGER_CHANNEL = "danger"        # optional sensor: the snake's own body
MOTOR_CHANNEL = "motor"


def _tor_frac(delta: int, n: int) -> float:
    """Shortest toroidal offset as a fraction of the axis length, in [-.5, .5)."""
    return (((delta + n // 2) % n) - n // 2) / n


def _ego_cell(dy: float, dx: float, shape: tuple[int, int], gain: float):
    """Map an egocentric board-fraction offset to a world-grid cell."""
    H, W = shape
    by = min(max(int(round(H / 2 + dy * H * gain)), 0), H - 1)
    bx = min(max(int(round(W / 2 + dx * W * gain)), 0), W - 1)
    return by, bx


def _bump(field: np.ndarray, by: int, bx: int, sigma: float, amp: float) -> None:
    H, W = field.shape
    yy, xx = np.ogrid[:H, :W]
    field += amp * np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2.0 * sigma ** 2)))


def _sharp_sigma(shape: tuple[int, int], sigma: float | None) -> float:
    """A CONCENTRATED bump width — a crisp point, not a blurred area. The brain
    can spread it itself (avg8) if it wants; the sensor stays sharp."""
    H, W = shape
    return sigma if sigma is not None else max(1.5, min(H, W) / 24.0)


def food_sensor(world_shape: tuple[int, int], snake: Snake, gain: float = 0.8,
                sigma: float | None = None) -> np.ndarray:
    """A sharp bump pointing toward the food, egocentric to the head — a
    concentrated 'smell'. Its amplitude encodes PROXIMITY (brighter when the food
    is near, dimmer when far), so the brain senses direction and distance at once.
    Zero if there is no food."""
    H, W = world_shape
    field = np.zeros((H, W), dtype=np.float64)
    if snake.food is None:
        return field
    hy, hx = snake.head()
    dy = _tor_frac(snake.food[0] - hy, snake.h)
    dx = _tor_frac(snake.food[1] - hx, snake.w)
    dist = (dy * dy + dx * dx) ** 0.5           # 0 (adjacent) .. ~0.707 (far)
    amp = 0.3 + 0.7 * max(0.0, 1.0 - dist / 0.707)
    by, bx = _ego_cell(dy, dx, world_shape, gain)
    _bump(field, by, bx, _sharp_sigma(world_shape, sigma), amp)
    return field


def body_sensor(world_shape: tuple[int, int], snake: Snake, gain: float = 0.8,
                sigma: float | None = None, radius_frac: float = 0.3) -> np.ndarray:
    """Danger = PROXIMITY to the snake's own body, as concentrated marks (not a
    diffuse whole-length cloud, which earlier drowned the food signal). Only body
    segments within `radius_frac` of the head are marked, each a sharp point whose
    amplitude grows as it nears the head — so the immediate collision risk is
    loud and the far tail is silent, in the same egocentric frame as the food."""
    H, W = world_shape
    field = np.zeros((H, W), dtype=np.float64)
    hy, hx = snake.head()
    sig = _sharp_sigma(world_shape, sigma)
    for sy, sx in snake.body[1:]:               # exclude the head itself
        dy = _tor_frac(sy - hy, snake.h)
        dx = _tor_frac(sx - hx, snake.w)
        dist = (dy * dy + dx * dx) ** 0.5
        if dist > radius_frac:
            continue                            # far body is not an imminent threat
        amp = 1.0 - dist / radius_frac          # nearer body = louder danger
        by, bx = _ego_cell(dy, dx, world_shape, gain)
        _bump(field, by, bx, sig, amp)
    return field


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
         moves: int = 1000, ticks_per_move: int = 16, life_channel: str = MOTOR_CHANNEL,
         collect_frames: bool = False, sensor_gain: float = 0.8,
         death_hold: int = 6) -> dict:
    """Run the coupled brain+game loop until the snake dies (or `moves` is hit —
    a safety cap; a brain still alive then is effectively immortal for this run).
    Returns score, survival, move sequence, and (optionally) paired life/game
    frames for a split render."""
    program = parse_program(law_text)
    require_embodied(program)
    if life_channel not in {c.name for c in program.channels}:
        raise NomosError(f"--life-channel {life_channel!r} is not a declared channel")

    has_danger = DANGER_CHANNEL in {c.name for c in program.channels}
    world = World(program, world_size, seed)
    snake = Snake(game_size, seed)
    life_frames: list[np.ndarray] = []
    game_frames: list[np.ndarray] = []
    move_seq: list[str | None] = []

    for _ in range(moves):
        if not snake.alive:
            break
        food = food_sensor(world.shape, snake, gain=sensor_gain)
        danger = body_sensor(world.shape, snake, gain=sensor_gain) if has_danger else None
        for _ in range(ticks_per_move):
            world.state[SENSOR_CHANNEL] = food        # re-assert the read-only senses
            if has_danger:
                world.state[DANGER_CHANNEL] = danger
            world.step()
        move = read_move(world, MOTOR_CHANNEL)
        snake.step(move)
        move_seq.append(move)
        if collect_frames:
            life_frames.append(world.state[life_channel].copy())
            game_frames.append(snake.field())

    # hold the death frame so the crash is actually watchable in the GIF
    if collect_frames and not snake.alive and game_frames:
        for _ in range(death_hold):
            life_frames.append(life_frames[-1])
            game_frames.append(game_frames[-1])

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
