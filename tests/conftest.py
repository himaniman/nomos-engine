"""Test-session setup.

SDL must be told to use dummy devices BEFORE pygame is imported anywhere, or every
`pygame.init()` in the suite spends ~8 s probing for a real video/audio device that a
test run has no use for. Setting it here (collection time) instead of inside each test
is what makes the render tests cheap.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
