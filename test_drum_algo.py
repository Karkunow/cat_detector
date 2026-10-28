"""Тести чистої логіки drum_algo.inside_run (без cv2): python3 -m unittest test_drum_algo"""
import unittest

from drum_algo import inside_run


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


if __name__ == "__main__":
    unittest.main()
