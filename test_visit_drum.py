"""Тести visit_drum: геометрія «кіт у вікні барабана чи перед лотком» і підсумок візиту. Без cv2/YOLO.
Запуск: .venv/bin/python -m unittest test_visit_drum"""
import unittest

from visit_drum import bbox_state, frame_state, summarize

WIN = (448, 180, 659, 403)   # вікно барабана з кадру 28.10 (1280x720)


class BBoxState(unittest.TestCase):
    def test_cat_inside_window(self):          # білий у барабані: рамка в межах вікна
        self.assertEqual(bbox_state((470, 230, 640, 400), WIN), "D")

    def test_cat_sitting_in_front(self):       # кіт перед лотком: рамка від вікна до підлоги (28.10 01:17)
        self.assertEqual(bbox_state((448, 182, 663, 642), WIN), "O")

    def test_cat_walking_past(self):           # повз, низько
        self.assertEqual(bbox_state((100, 450, 500, 700), WIN), "O")

    def test_head_slightly_out_of_window(self):  # вуха трохи вище рамки — ще в барабані
        self.assertEqual(bbox_state((480, 160, 630, 395), WIN), "D")


class FrameState(unittest.TestCase):
    def test_no_box_means_unknown(self):
        self.assertEqual(frame_state([(470, 230, 640, 400)], None), "?")

    def test_hidden_cat_by_algo(self):
        self.assertEqual(frame_state([], WIN, "inside", 0.45), "d")

    def test_weak_window_is_not_cat(self):     # хибний «inside» на порожньому денному лотку 26.10: f 0.10–0.16
        self.assertEqual(frame_state([], WIN, "inside", 0.14), ".")

    def test_yolo_outside_beats_algo(self):    # кіт перед лотком закрив вікно: YOLO бачить його ззовні
        self.assertEqual(frame_state([(448, 182, 663, 642)], WIN, "inside", 0.6), "O")


class Summary(unittest.TestCase):
    def test_visit_then_front(self):           # 01.11 03:22: у барабані ~10 с, потім сидить перед лотком
        st = [(t, ".") for t in range(0, 5)] + [(t, "D") for t in range(5, 16)] + [(t, "O") for t in range(16, 35)]
        s = summarize(st, 34)
        self.assertTrue(s["visit"])
        self.assertFalse(s["inside_at_end"])
        self.assertAlmostEqual(s["drum_seconds"], 11.0)

    def test_only_front_is_not_visit(self):    # сидів лише перед лотком
        s = summarize([(t, "O") for t in range(30)], 29)
        self.assertFalse(s["visit"])
        self.assertEqual(s["drum_seconds"], 0.0)

    def test_box_not_visible(self):            # 01.11 01:54: камера дивилась униз
        s = summarize([(t, "?") for t in range(20)], 19)
        self.assertFalse(s["visit"])
        self.assertFalse(s["seen_box"])

    def test_still_inside_at_end(self):
        s = summarize([(t, "O") for t in range(3)] + [(t, "d") for t in range(3, 20)], 20)
        self.assertTrue(s["visit"])
        self.assertTrue(s["inside_at_end"])

    def test_single_frame_is_not_visit(self):
        self.assertFalse(summarize([(0, "O"), (1, "D"), (2, "O")], 2)["visit"])

    def test_gap_splits_segments(self):        # дві частини з довгою паузою: пауза не рахується
        st = [(t, "D") for t in range(0, 6)] + [(t, "D") for t in range(40, 46)]
        self.assertAlmostEqual(summarize(st, 50)["drum_seconds"], 5 + 5 + 1.0)


if __name__ == "__main__":
    unittest.main()
