"""Тести чистої логіки drum_algo.inside_run (без cv2): python3 -m unittest test_drum_algo"""
import unittest

from drum_algo import inside_run, soften


class InsideRunTest(unittest.TestCase):
    def test_all_inside(self):
        self.assertEqual(inside_run(["inside"] * 3), 3)

    def test_two_of_three_with_last_inside(self):
        self.assertEqual(inside_run(["inside", "near", "inside"]), 1)
        self.assertEqual(inside_run(["near", "inside", "inside"]), 2)

    def test_last_not_inside(self):
        self.assertEqual(inside_run(["inside", "inside", "clear"]), 0)

    def test_single_inside_not_enough(self):
        self.assertEqual(inside_run(["clear", "clear", "inside"]), 0)
        self.assertEqual(inside_run(["inside"]), 0)

    def test_empty_or_other_classes(self):
        self.assertEqual(inside_run([]), 0)
        self.assertEqual(inside_run(["blocking", "blocking", "unknown"]), 0)

    def test_error_means_fallback(self):
        self.assertIsNone(inside_run(["inside", "error", "inside"]))
        self.assertIsNone(inside_run(["error", "error"]))


CELLS_OK = [0.46, 0.43, 0.46, 0.6, 0.79, 0.59, 0.78, 0.57]
RIM_OK = [0.37, 0.36, 0.29, 0.73]


class SoftenTest(unittest.TestCase):
    def test_cat_in_window_cells_slightly_low(self):   # 29.10 00:17:30 та 28.10 07:38
        self.assertEqual(soften("blocking", "occupied", CELLS_OK, RIM_OK), "inside")

    def test_cat_in_front_cells_low(self):
        self.assertEqual(soften("blocking", "occupied", [0.30] + CELLS_OK[1:], RIM_OK), "blocking")
        self.assertEqual(soften("blocking", "occupied", [0.12] + CELLS_OK[1:], RIM_OK), "blocking")

    def test_rim_covered(self):
        self.assertEqual(soften("blocking", "occupied", CELLS_OK, [0.37, 0.0, 0.29, 0.73]), "blocking")
        self.assertEqual(soften("blocking", "occupied", CELLS_OK, [0.37, 0.24, 0.29, 0.73]), "blocking")

    def test_other_classes_untouched(self):
        for c in ("clear", "near", "inside", "unknown"):
            self.assertEqual(soften(c, "occupied", CELLS_OK, RIM_OK), c)
        self.assertEqual(soften("blocking", "unknown", CELLS_OK, RIM_OK), "blocking")

    def test_missing_metrics(self):
        self.assertEqual(soften("blocking", "occupied", None, RIM_OK), "blocking")
        self.assertEqual(soften("blocking", "occupied", CELLS_OK, None), "blocking")


if __name__ == "__main__":
    unittest.main()
