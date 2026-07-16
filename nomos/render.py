"""Rendering: turn sampled channel frames into GIF/PNG.

One channel -> colormap ("fire" by default, "gray" available).
Two or three channels -> mapped to R, G, B, each normalized independently.
Normalization is global across all frames of a run so brightness is
comparable over time.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# black -> deep red -> orange -> white
_FIRE_STOPS = [
    (0.00, (0, 0, 0)),
    (0.35, (170, 20, 10)),
    (0.70, (255, 170, 30)),
    (1.00, (255, 255, 255)),
]


def _apply_fire(x: np.ndarray) -> np.ndarray:
    stops = np.array([p for p, _ in _FIRE_STOPS])
    colors = np.array([c for _, c in _FIRE_STOPS], dtype=np.float64)
    rgb = np.zeros((*x.shape, 3), dtype=np.float64)
    for i in range(3):
        rgb[..., i] = np.interp(x, stops, colors[:, i])
    return rgb


def _normalize(frames: list[np.ndarray]) -> list[np.ndarray]:
    lo = min(float(f.min()) for f in frames)
    hi = max(float(f.max()) for f in frames)
    if hi - lo < 1e-12:
        return [np.zeros_like(f) for f in frames]
    return [(f - lo) / (hi - lo) for f in frames]


def _add_progress_bar(img: Image.Image, frac: float) -> Image.Image:
    """Append a thin service strip at the bottom filled to `frac` of the width,
    so a viewer can see where in the run a frame sits."""
    w, h = img.width, img.height
    bar_h = max(4, h // 40)
    canvas = Image.new("RGB", (w, h + bar_h), (0, 0, 0))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, h, w - 1, h + bar_h - 1], fill=(30, 30, 30))
    fill_w = int(round(max(0.0, min(1.0, frac)) * (w - 1)))
    if fill_w > 0:
        draw.rectangle([0, h, fill_w, h + bar_h - 1], fill=(255, 170, 30))
    return canvas


def frames_to_images(
    channel_frames: dict[str, list[np.ndarray]],
    scale: int = 4,
    cmap: str = "fire",
    progress_bar: bool = False,
) -> list[Image.Image]:
    """channel_frames: {channel: [frame0, frame1, ...]}, all same length/shape."""
    names = list(channel_frames)
    n_frames = len(channel_frames[names[0]])
    norm = {name: _normalize(channel_frames[name]) for name in names}

    images = []
    for t in range(n_frames):
        if len(names) == 1:
            x = norm[names[0]][t]
            rgb = _apply_fire(x) if cmap == "fire" else np.repeat(x[..., None] * 255, 3, axis=2)
        else:
            h, w = norm[names[0]][t].shape
            rgb = np.zeros((h, w, 3), dtype=np.float64)
            for i, name in enumerate(names[:3]):
                rgb[..., i] = norm[name][t] * 255
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB")
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        if progress_bar:
            frac = t / (n_frames - 1) if n_frames > 1 else 1.0
            img = _add_progress_bar(img, frac)
        images.append(img)
    return images


def split_images(
    life_frames: list[np.ndarray],
    game_frames: list[np.ndarray],
    scale: int = 6,
    gap: int = 4,
    life_frames2: list[np.ndarray] | None = None,
) -> list[Image.Image]:
    """Side-by-side frames: the life field (left) and the game board (right,
    fire palette on its fixed 0..1 levels), matched in height, progress strip
    below. This is the watch-along view. If `life_frames2` is given (the 'avoid'
    layer), the two brain layers are merged into colour — green = go-toward
    (motor), red = stay-away (fear) — so you can see them without them fighting
    in one field; otherwise the single life field uses the fire colormap."""
    if life_frames2 is not None:
        go = _normalize(life_frames)
        fear = _normalize(life_frames2)
        life_imgs = []
        for t in range(len(life_frames)):
            rgb = np.zeros((*life_frames[t].shape, 3), dtype=np.float64)
            rgb[..., 0] = fear[t] * 255        # red   = fear / avoid
            rgb[..., 1] = go[t] * 255          # green = go / motor
            img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB")
            if scale != 1:
                img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
            life_imgs.append(img)
    else:
        life_imgs = frames_to_images({"life": life_frames}, scale=scale, cmap="fire")
    target_h = life_imgs[0].height
    out = []
    n = len(life_frames)
    for t in range(n):
        left = life_imgs[t]
        g = np.clip(game_frames[t], 0.0, 1.0)
        gimg = Image.fromarray(_apply_fire(g).clip(0, 255).astype(np.uint8), "RGB")
        gs = max(1, target_h // gimg.height)
        gimg = gimg.resize((gimg.width * gs, gimg.height * gs), Image.NEAREST)
        if gimg.height != target_h:
            gimg = gimg.resize((round(gimg.width * target_h / gimg.height), target_h),
                               Image.NEAREST)
        w = left.width + gap + gimg.width
        canvas = Image.new("RGB", (w, target_h), (0, 0, 0))
        canvas.paste(left, (0, 0))
        canvas.paste(gimg, (left.width + gap, 0))
        frac = t / (n - 1) if n > 1 else 1.0
        out.append(_add_progress_bar(canvas, frac))
    return out


def save_gif(images: list[Image.Image], path: str, fps: int = 25) -> None:
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(1000 // fps, 20),
        loop=0,
    )


def save_png(image: Image.Image, path: str) -> None:
    image.save(path)
