import numpy as np

from nomos.render import sense_split_images


def test_sense_split_images_builds_rgb_frames():
    senses = [np.random.rand(12, 12, 3) for _ in range(4)]
    game = [np.random.rand(10, 10) for _ in range(4)]
    imgs = sense_split_images(senses, game, scale=2)
    assert len(imgs) == 4
    assert imgs[0].mode == "RGB"
    assert imgs[0].width > 12 and imgs[0].height > 12   # scaled + game panel + bar

from nomos.render import frames_to_images


def _frames(n=5, size=16):
    return {"a": [np.random.default_rng(t).random((size, size)) for t in range(n)]}


def test_progress_bar_adds_a_bottom_strip():
    plain = frames_to_images(_frames(), scale=2, progress_bar=False)
    barred = frames_to_images(_frames(), scale=2, progress_bar=True)
    # same width, taller by the service strip
    assert barred[0].width == plain[0].width
    assert barred[0].height > plain[0].height


def test_split_images_blends_two_layers_into_colour():
    from nomos.render import split_images
    life = [np.zeros((8, 8)) for _ in range(3)]
    fear = [np.zeros((8, 8)) for _ in range(3)]
    for t in range(3):
        life[t][2, 2] = 1.0                # go layer -> green
        fear[t][5, 5] = 1.0                # fear layer -> red
    game = [np.zeros((6, 6)) for _ in range(3)]
    imgs = split_images(life, game, scale=2, life_frames2=fear)
    px = np.asarray(imgs[0])
    assert px[..., 1].max() > 100          # green present (go)
    assert px[..., 0].max() > 100          # red present (fear)


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
