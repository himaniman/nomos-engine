import numpy as np

from nomos.render import frames_to_images


def _frames(n=5, size=16):
    return {"a": [np.random.default_rng(t).random((size, size)) for t in range(n)]}


def test_progress_bar_adds_a_bottom_strip():
    plain = frames_to_images(_frames(), scale=2, progress_bar=False)
    barred = frames_to_images(_frames(), scale=2, progress_bar=True)
    # same width, taller by the service strip
    assert barred[0].width == plain[0].width
    assert barred[0].height > plain[0].height


def test_progress_bar_grows_left_to_right():
    imgs = frames_to_images(_frames(n=5, size=16), scale=2, progress_bar=True)
    h_field = imgs[0].height - max(4, imgs[0].height // 41)   # roughly the field top

    def lit_width(img):
        row = img.height - 1                     # bottom row of the bar
        px = [img.getpixel((x, row)) for x in range(img.width)]
        return sum(1 for p in px if p[0] > 100)  # orange fill is bright red channel

    # first frame near-empty, last frame full
    assert lit_width(imgs[0]) < lit_width(imgs[-1])
    assert lit_width(imgs[-1]) >= imgs[-1].width - 2
