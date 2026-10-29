import numpy as np

from classify import bright_override

# рамка кота, де 60% пікселів яскраві (білий лоток за чорним котом)
FRAME = np.zeros((100, 100, 3), np.uint8)
FRAME[:, :60] = 220
BBOX = (0, 0, 100, 100)


def test_dark_ratio_stays_black():
    assert bright_override("black", FRAME, BBOX, 0.23) == "black"


def test_mid_ratio_with_bright_pixels_becomes_white():
    assert bright_override("black", FRAME, BBOX, 0.50) == "white"


def test_no_ratio_keeps_old_behaviour():
    assert bright_override("black", FRAME, BBOX) == "white"
