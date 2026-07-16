import numpy as np

from nomos.game import Snake


def test_snake_moves_and_wraps_toroidally():
    s = Snake(8, seed=1)
    hy, hx = s.head()
    s.step("right")
    assert s.head() == (hy, (hx + 1) % 8)
    assert len(s.body) == 3            # length unchanged when not eating


def test_snake_cannot_reverse_into_itself():
    s = Snake(8, seed=1)               # heading right, body to the left
    s.step("left")                     # a 180 should be ignored
    assert s.direction == "right"
    assert s.alive


def test_snake_eats_grows_and_scores():
    s = Snake(8, seed=1)
    # drop food right in front of the head so one step eats it
    hy, hx = s.head()
    s.food = (hy, (hx + 1) % 8)
    s.step("right")
    assert s.score == 1
    assert len(s.body) == 4            # grew by one
    assert s.food is not None and s.food != (hy, (hx + 1) % 8)  # respawned


def test_snake_is_deterministic():
    a = Snake(12, seed=7)
    b = Snake(12, seed=7)
    moves = ["up", "left", "down", "right", None, "up"]
    for m in moves:
        a.step(m)
        b.step(m)
    assert a.body == b.body and a.food == b.food and a.score == b.score


def test_death_is_recorded_and_flashed():
    s = Snake(6, seed=2)
    # a 2x2 loop: head (3,3); (2,3) is a non-tail body cell (tail is (3,2))
    s.body = [(3, 3), (2, 3), (2, 2), (3, 2)]
    s.direction = "right"
    s.food = (0, 0)
    s.step("up")                       # head (3,3) -> (2,3), a body cell -> crash
    assert not s.alive
    assert s.death_cell == (2, 3)
    assert s.field()[2, 3] == 1.0      # the crash is flashed for the render


def test_field_marks_head_food_body():
    s = Snake(8, seed=1)
    f = s.field()
    hy, hx = s.head()
    assert f[hy, hx] == 0.85
    assert f[s.food] == 1.0
    assert (f == 0.5).sum() == len(s.body) - 1   # body cells minus the head
