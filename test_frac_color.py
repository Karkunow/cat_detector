"""Колір кота за часткою яскравих пікселів (03.11.2026)."""
import unittest

import numpy as np

from classify import BLACK_FRAC_ABS, BLACK_FRAC_REL, REL_K, WHITE_FRAC_ABS, crop_hist, frac_above, frac_color


def hist_from(values: list[int]) -> np.ndarray:
    return np.bincount(np.array(values, dtype=np.int64), minlength=256)


class FracAbove(unittest.TestCase):
    def test_strictly_above(self):
        h = hist_from([149, 150, 151, 200])
        self.assertAlmostEqual(frac_above(h, 150), 0.5)      # 151 і 200
        self.assertAlmostEqual(frac_above(h, 149), 0.75)

    def test_fractional_threshold(self):
        h = hist_from([113, 114, 115])
        self.assertAlmostEqual(frac_above(h, 113.05), 2 / 3)  # 114 і 115

    def test_empty(self):
        self.assertEqual(frac_above(np.zeros(256, dtype=np.int64), 150), 0.0)
        self.assertEqual(frac_above(hist_from([10]), 300), 0.0)


class FracColor(unittest.TestCase):
    def test_white_by_absolute(self):
        h = hist_from([200] * 60 + [40] * 40)                 # 60% пікселів > 150
        self.assertEqual(frac_color([h], 170)[0], "white")

    def test_white_by_relative_when_scene_is_dark(self):
        # сцена темна (опора 120): кіт світліший за 0.85*120=102, але майже нічого > 150
        h = hist_from([140] * 80 + [30] * 20)
        self.assertLess(frac_above(h, 150), WHITE_FRAC_ABS)
        self.assertEqual(frac_color([h], 120)[0], "white")

    def test_black(self):
        h = hist_from([40] * 90 + [200] * 10)
        self.assertEqual(frac_color([h], 170)[0], "black")

    def test_unknown_between(self):
        # 30% яскравих: вище чорного порога, нижче білого — чесне «не знаю»
        h = hist_from([200] * 30 + [60] * 70)
        c, f_abs, f_rel = frac_color([h], 170)
        self.assertEqual(c, "unknown")
        self.assertGreater(f_abs, BLACK_FRAC_ABS)
        self.assertLess(f_abs, WHITE_FRAC_ABS)

    def test_no_hists(self):
        self.assertEqual(frac_color([], 170)[0], "unknown")
        self.assertEqual(frac_color([np.zeros(256, dtype=np.int64)], 170)[0], "unknown")

    def test_median_over_frames(self):
        white = hist_from([200] * 60 + [40] * 40)
        black = hist_from([40] * 100)
        self.assertEqual(frac_color([white, white, black], 170)[0], "white")   # медіана — білий кадр

    def test_crop_hist_shape(self):
        crop = np.full((4, 5, 3), 200, dtype=np.uint8)
        h = crop_hist(crop)
        self.assertEqual(h.shape, (256,))
        self.assertEqual(int(h.sum()), 20)
        self.assertAlmostEqual(frac_above(h, 150), 1.0)

    def test_thresholds_leave_a_gap(self):
        self.assertLess(BLACK_FRAC_ABS, WHITE_FRAC_ABS)
        self.assertLess(BLACK_FRAC_REL, REL_K + 1)


if __name__ == "__main__":
    unittest.main()
