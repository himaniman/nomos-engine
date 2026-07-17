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
from nomos.forage import Forage
from nomos.game import DIRECTIONS, _OPPOSITE, Snake

SENSOR_CHANNEL = "food"
DANGER_CHANNEL = "danger"           # optional sensor: the snake's own body
WALL_CHANNEL = "wall"               # optional sensor (forage): near-field wall vision
SIGHT_CHANNEL = "sight"             # optional sensor (forage): food in line of sight
TEMPERATURE_CHANNEL = "temperature"  # optional sensor: hunger, uniform over the grid
REWARD_CHANNEL = "reward"           # optional sensor: a global reward pulse (forage:
                                    # 1.0 the move food was eaten, decaying otherwise)
MOTOR_CHANNEL = "motor"             # 'go toward' output layer
AVOID_CHANNEL = "avoid"             # optional 'stay away' output layer (non-interfering)


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
    """Add a Gaussian bump, truncated at 3 sigma so painting stays local — with
    a dense brain (512^2) and hundreds of wall marks per move, whole-grid
    painting would dominate the run time."""
    H, W = field.shape
    r = int(3.0 * sigma) + 1
    y0, y1 = max(0, by - r), min(H, by + r + 1)
    x0, x1 = max(0, bx - r), min(W, bx + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    field[y0:y1, x0:x1] += amp * np.exp(
        -(((yy - by) ** 2 + (xx - bx) ** 2) / (2.0 * sigma ** 2)))


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
                sigma: float | None = None, radius_frac: float = 0.3,
                leverage: float = 0.7) -> np.ndarray:
    """Danger = PROXIMITY to the snake's own body, marked with LEVERAGE. Only body
    segments within `radius_frac` of the head count (near = an imminent threat),
    but each mark is placed in that segment's DIRECTION at a large fixed radius
    (`leverage` of the half-grid), not at its literal near-centre position. That
    matters: the neck sits one cell from the head, which maps almost onto the grid
    centre where it can barely shift the motor's centre of mass — so plain
    placement makes body-fear geometrically toothless against distant food.
    Pushing the mark outward in its direction gives 'body is that way, don't go'
    real steering authority. Amplitude still grows as the body nears the head."""
    H, W = world_shape
    field = np.zeros((H, W), dtype=np.float64)
    hy, hx = snake.head()
    sig = _sharp_sigma(world_shape, sigma)
    R = leverage * min(H, W) / 2.0
    for sy, sx in snake.body[1:]:               # exclude the head itself
        dy = _tor_frac(sy - hy, snake.h)
        dx = _tor_frac(sx - hx, snake.w)
        dist = (dy * dy + dx * dx) ** 0.5
        if dist > radius_frac or dist < 1e-9:
            continue                            # far body is not an imminent threat
        amp = 1.0 - dist / radius_frac          # nearer body = louder danger
        by = min(max(int(round(H / 2 + (dy / dist) * R)), 0), H - 1)
        bx = min(max(int(round(W / 2 + (dx / dist) * R)), 0), W - 1)
        _bump(field, by, bx, sig, amp)
    return field


def forage_food_sensor(world_shape: tuple[int, int], game: Forage,
                       gain: float = 0.9, sigma: float | None = None,
                       lam_frac: float = 0.3, min_rad: float = 0.2) -> np.ndarray:
    """Every food item as a SMELL that has DIFFUSED through the maze: amplitude
    decays exponentially with the GEODESIC (walking) distance, and the bump is
    painted in the direction of the shortest-path descent at the player's nose
    — the way to WALK, which around a wall is not the straight line to the
    food. Food with no open path to the player has no scent. The bump's radius
    from centre grows with distance (never closer than `min_rad` of the ego
    view, so a nearby meal keeps its moment arm — the leverage lesson from the
    attractor's side). Far food fades toward nothing: finding it is exploration,
    not perception."""
    H, W = world_shape
    field = np.zeros((H, W), dtype=np.float64)
    py, px = game.pos
    lam = lam_frac * max(game.h, game.w)
    sig = _sharp_sigma(world_shape, sigma)
    for food in sorted(game.foods):
        dmap = game.scent(food)
        d0 = int(dmap[py, px])
        if d0 <= 0:
            continue                               # unreachable (or standing on it)
        vy = vx = 0.0
        for dy, dx in DIRECTIONS.values():
            ny, nx = py + dy, px + dx
            if (0 <= ny < game.h and 0 <= nx < game.w and dmap[ny, nx] >= 0
                    and dmap[ny, nx] < d0):
                vy += dy
                vx += dx
        norm = (vy * vy + vx * vx) ** 0.5
        if norm < 1e-9:
            continue                               # local plateau: no gradient here
        amp = float(np.exp(-d0 / lam))
        rad = min(0.45, max(min_rad, d0 / (2.0 * max(game.h, game.w))))
        by, bx = _ego_cell(vy / norm * rad, vx / norm * rad, world_shape, gain)
        _bump(field, by, bx, sig, amp)
    return field


def wall_sensor(world_shape: tuple[int, int], game: Forage, gain: float = 0.9,
                sigma: float | None = None, radius: int | None = None) -> np.ndarray:
    """Sight down the four MOVEMENT AXES: for each direction, the first wall
    cell on that ray within `radius`, painted egocentric with the window
    magnified to fill the ego view (a wall one step away lands well off centre,
    where it has steering authority) and louder the closer (1/d^2). Only
    on-axis walls are sensed because only they BLOCK a step: point-repulsion
    from every nearby wall cell turned corridors into fear — the paired side
    walls repelled along the corridor axis and the player stalled at the mouth
    of the very passage that led to food. Walls are touch/blocking sense; the
    maze's global structure reaches the brain through the scent field, which
    has already diffused around it."""
    H, W = world_shape
    field = np.zeros((H, W), dtype=np.float64)
    py, px = game.pos
    r = radius if radius is not None else min(8, max(3, min(game.h, game.w) // 4))
    sig = _sharp_sigma(world_shape, sigma)
    for dy, dx in DIRECTIONS.values():
        for k in range(1, r + 1):
            ny, nx = py + dy * k, px + dx * k
            if not (0 <= ny < game.h and 0 <= nx < game.w) or game.walls[ny, nx]:
                by, bx = _ego_cell(dy * k / (2 * r), dx * k / (2 * r),
                                   world_shape, gain)
                _bump(field, by, bx, sig, 1.0 / (k * k))
                break                              # only the first hit blocks
    return field


def _pulls(field: np.ndarray) -> dict:
    """Non-negative 'pull' toward each cardinal direction: the field's moment
    away from the grid centre. A bump above centre gives a large 'up' pull, etc."""
    w = np.clip(field, 0.0, None)
    H, W = field.shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    return {"up": float((w * np.clip(cy - yy, 0, None)).sum()),
            "down": float((w * np.clip(yy - cy, 0, None)).sum()),
            "left": float((w * np.clip(cx - xx, 0, None)).sum()),
            "right": float((w * np.clip(xx - cx, 0, None)).sum())}


def read_move(world: World, channel: str = MOTOR_CHANNEL,
              avoid: str | None = None) -> str | None:
    """Pick a move from the motor layer(s). Each direction gets a score = the
    motor's pull that way; if an `avoid` layer is given, its pull that way is
    SUBTRACTED — so 'go toward' and 'stay away' are kept as separate layers that
    combine only here, at the decision, instead of cancelling in one field. The
    best positive direction wins; if none is positive the snake keeps heading."""
    net, _ = _net_pulls(world, channel, avoid)
    if net is None:
        return None
    best = max(net, key=net.get)
    return best if net[best] > 1e-9 else None


def _net_pulls(world: World, channel: str, avoid: str | None):
    """Net directional pulls (motor minus avoid) and the total signal mass."""
    m = world.state.get(channel)
    if m is None:
        return None, 0.0
    net = _pulls(m)
    total = sum(net.values())
    if avoid is not None and avoid in world.state:
        av = _pulls(world.state[avoid])
        total += sum(av.values())
        net = {d: net[d] - av[d] for d in net}
    return net, total


def sight_sensor(world_shape: tuple[int, int], game: Forage, gain: float = 0.9,
                 sigma: float | None = None, radius: int = 16,
                 min_rad: float = 0.2) -> np.ndarray:
    """Food the player can SEE: items within `radius` cells with an unbroken
    line of sight, painted at their TRUE egocentric direction (the straight
    line — sight, unlike smell, does not bend around corners) and much brighter
    than a smell of the same range: seeing is precise and certain. Amplitude
    grows as the food nears; anything behind a wall or beyond the range gives
    nothing. Complements the `food` scent: smell says 'somewhere that way,
    around the corner', sight says 'RIGHT THERE'."""
    H, W = world_shape
    field = np.zeros((H, W), dtype=np.float64)
    py, px = game.pos
    sig = _sharp_sigma(world_shape, sigma)
    for fy, fx in sorted(game.foods):
        dy, dx = fy - py, fx - px
        dist = (dy * dy + dx * dx) ** 0.5
        if dist > radius or dist < 1e-9 or not game.sees((fy, fx)):
            continue
        amp = 0.5 + 0.5 * (1.0 - dist / radius)
        rad = min(0.45, max(min_rad, dist / (2.0 * radius)))
        by, bx = _ego_cell(dy / dist * rad, dx / dist * rad, world_shape, gain)
        _bump(field, by, bx, sig, amp)
    return field


def _slide(ranked: list[str], game: Forage) -> str | None:
    """Contact dynamics for a blocked favourite: slide TANGENTIALLY along the
    wall (the next commanded direction that is not the exact reverse), falling
    back to the reverse only when it too was commanded and everything else is
    walled. Naive next-best sliding creates a two-cell limit cycle: pressed
    against a wall the runner-up is usually the reverse, the player steps back,
    the wall stops screaming, the favourite wins again — forever."""
    if not ranked:
        return None
    opposite = _OPPOSITE[ranked[0]]
    for d in ranked:
        if d != opposite and not game.blocked(d):
            return d
    return opposite if opposite in ranked and not game.blocked(opposite) else None


def read_ranked(world: World, channel: str = MOTOR_CHANNEL,
                avoid: str | None = None) -> list[str]:
    """The brain's move COMMANDS: directions whose net pull (motor minus avoid)
    is strictly POSITIVE, best first. Empty list = no command = the player
    STANDS STILL (hunger still grows — stillness is a choice, not safety). The
    game never moves by itself; far from any scent a brain must generate its
    own exploratory motor activity or starve where it stands. The forage
    harness walks this list to the first OPEN direction, so pressing into a
    wall slides along it (contact dynamics, not intelligence)."""
    net, _ = _net_pulls(world, channel, avoid)
    if net is None:
        return []
    return [d for d in sorted(net, key=net.get, reverse=True) if net[d] > 1e-9]


def require_embodied(program) -> None:
    names = {c.name for c in program.channels}
    for req in (SENSOR_CHANNEL, MOTOR_CHANNEL):
        if req not in names:
            raise NomosError(f"an embodied law must declare a {req!r} channel "
                             f"(sensor {SENSOR_CHANNEL!r}, motor {MOTOR_CHANNEL!r})")


def play(law_text: str, world_size: int = 48, game_size: int = 16, seed: int = 0,
         moves: int = 1000, ticks_per_move: int = 64, life_channel: str = MOTOR_CHANNEL,
         collect_frames: bool = False, sensor_gain: float = 0.8,
         death_hold: int = 6, starve_every: int = 50) -> dict:
    """Run the coupled brain+game loop until the snake dies (or `moves` is hit —
    a safety cap; a brain still alive then is effectively immortal for this run).
    Returns score, survival, move sequence, and (optionally) paired life/game
    frames for a split render."""
    program = parse_program(law_text)
    require_embodied(program)
    if life_channel not in {c.name for c in program.channels}:
        raise NomosError(f"--life-channel {life_channel!r} is not a declared channel")

    names = {c.name for c in program.channels}
    has_danger = DANGER_CHANNEL in names
    has_temp = TEMPERATURE_CHANNEL in names
    has_avoid = AVOID_CHANNEL in names
    world = World(program, world_size, seed)
    snake = Snake(game_size, seed, starve_every=starve_every)
    life_frames: list[np.ndarray] = []
    avoid_frames: list[np.ndarray] = []
    game_frames: list[np.ndarray] = []
    move_seq: list[str | None] = []

    for _ in range(moves):
        if not snake.alive:
            break
        food = food_sensor(world.shape, snake, gain=sensor_gain)
        danger = body_sensor(world.shape, snake, gain=sensor_gain) if has_danger else None
        temp = snake.hunger_fraction() if has_temp else None
        for _ in range(ticks_per_move):
            world.state[SENSOR_CHANNEL] = food        # re-assert the read-only senses
            if has_danger:
                world.state[DANGER_CHANNEL] = danger
            if has_temp:
                world.state[TEMPERATURE_CHANNEL][:] = temp   # uniform hunger field
            world.step()
        move = read_move(world, MOTOR_CHANNEL, avoid=AVOID_CHANNEL if has_avoid else None)
        snake.step(move)
        move_seq.append(move)
        if collect_frames:
            life_frames.append(world.state[life_channel].copy())
            if has_avoid:
                avoid_frames.append(world.state[AVOID_CHANNEL].copy())
            game_frames.append(snake.field())

    # hold the death frame so the crash is actually watchable in the GIF
    if collect_frames and not snake.alive and game_frames:
        for _ in range(death_hold):
            life_frames.append(life_frames[-1])
            game_frames.append(game_frames[-1])
            if has_avoid:
                avoid_frames.append(avoid_frames[-1])

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
        "avoid_frames": avoid_frames,
        "game_frames": game_frames,
    }


def _block_mean(a: np.ndarray, cap: int) -> np.ndarray:
    """Downsample by block averaging until each side is <= cap — how a dense
    brain (512^2) stays watchable as a GIF without storing every cell."""
    f = int(np.ceil(max(a.shape) / cap))
    if f <= 1:
        return a.copy()
    hc, wc = a.shape[0] - a.shape[0] % f, a.shape[1] - a.shape[1] % f
    return a[:hc, :wc].reshape(hc // f, f, wc // f, f).mean(axis=(1, 3))


def play_forage(law_text: str, world_size: int = 128, game_size: int = 96,
                seed: int = 0, moves: int = 400, ticks_per_move: int = 32,
                life_channel: str = MOTOR_CHANNEL, collect_frames: bool = False,
                sensor_gain: float = 0.9, death_hold: int = 6,
                food_count: int | None = None, starve_after: int = 120,
                wall_radius: int | None = None, view_radius: int = 32,
                view_fade: float | None = 0.7, view_fog: bool = True,
                render_cap: int = 128, reward_decay: float = 0.6) -> dict:
    """Run the coupled brain+forage loop until the player starves (or `moves` —
    the safety cap; a brain still alive then kept feeding itself the whole run).
    Same contract as `play`: senses re-asserted every physics tick, one game move
    per `ticks_per_move` ticks, deterministic in all arguments. Sensors: `food`
    (required, distance-attenuated smell of every food item), `wall` (optional,
    near-field sight of walls), `temperature` (optional, uniform hunger).
    Rendering follows the player: game frames are the egocentric `view_radius`
    crop (the camera, not the whole wide map) with a `view_fade` perception
    vignette, life frames are block-averaged down to `render_cap` so a dense
    brain stays a small GIF, and `temp_frames` carries the hunger fraction per
    frame so the renderer can paint hunger onto the brain (blue channel)."""
    program = parse_program(law_text)
    require_embodied(program)
    if life_channel not in {c.name for c in program.channels}:
        raise NomosError(f"--life-channel {life_channel!r} is not a declared channel")

    names = {c.name for c in program.channels}
    has_wall = WALL_CHANNEL in names
    has_sight = SIGHT_CHANNEL in names
    has_temp = TEMPERATURE_CHANNEL in names
    has_reward = REWARD_CHANNEL in names
    has_avoid = AVOID_CHANNEL in names
    world = World(program, world_size, seed)
    game = Forage(game_size, seed, food_count=food_count, starve_after=starve_after)
    reward_val = 0.0
    prev_score = 0
    life_frames: list[np.ndarray] = []
    avoid_frames: list[np.ndarray] = []
    game_frames: list[np.ndarray] = []
    temp_frames: list[float] = []
    move_seq: list[str | None] = []

    for _ in range(moves):
        if not game.alive:
            break
        food = forage_food_sensor(world.shape, game, gain=sensor_gain)
        wall = (wall_sensor(world.shape, game, gain=sensor_gain, radius=wall_radius)
                if has_wall else None)
        sight = sight_sensor(world.shape, game, gain=sensor_gain) if has_sight else None
        temp = game.hunger_fraction() if has_temp else None
        for _ in range(ticks_per_move):
            world.state[SENSOR_CHANNEL] = food        # re-assert the read-only senses
            if has_wall:
                world.state[WALL_CHANNEL] = wall
            if has_sight:
                world.state[SIGHT_CHANNEL] = sight
            if has_temp:
                world.state[TEMPERATURE_CHANNEL][:] = temp   # uniform hunger field
            if has_reward:
                world.state[REWARD_CHANNEL][:] = reward_val  # uniform reward pulse
            world.step()
        ranked = read_ranked(world, MOTOR_CHANNEL,
                             avoid=AVOID_CHANNEL if has_avoid else None)
        move = _slide(ranked, game)
        game.step(move)
        # reward pulses to 1.0 the move a meal is eaten, then decays — a global
        # 'that paid off' signal a learning rule can multiply into its weight
        # update (three-factor Hebbian). Set AFTER the move so the next think
        # cycle feels the reward for what it just did.
        reward_val = 1.0 if game.score > prev_score else reward_val * reward_decay
        prev_score = game.score
        move_seq.append(move)
        if collect_frames:
            life_frames.append(_block_mean(world.state[life_channel], render_cap))
            if has_avoid:
                avoid_frames.append(_block_mean(world.state[AVOID_CHANNEL], render_cap))
            game_frames.append(game.view(view_radius, fade=view_fade, fog=view_fog))
            temp_frames.append(game.hunger_fraction())

    if collect_frames and not game.alive and game_frames:
        for _ in range(death_hold):
            life_frames.append(life_frames[-1])
            game_frames.append(game_frames[-1])
            temp_frames.append(temp_frames[-1])
            if has_avoid:
                avoid_frames.append(avoid_frames[-1])

    return {
        "score": game.score,
        "steps_alive": game.steps,
        "moves": len(move_seq),
        "alive_at_end": game.alive,
        "move_seq": move_seq,
        "world_size": list(world.shape),
        "game_size": [game.h, game.w],
        "seed": seed,
        "life_frames": life_frames,
        "avoid_frames": avoid_frames,
        "game_frames": game_frames,
        "temp_frames": temp_frames,
    }
