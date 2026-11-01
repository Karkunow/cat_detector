"""Регресія вердикту decide() на розмітці Слави (labels_2026-10-28.json: ознаки кадру + папка, у яку Слава його поклав).
Без cv2: лише cells/rim/вердикт вікна. Запуск: python3 -m unittest litter_algo.test_classify"""
import json
import os
import unittest

from litter_algo.classify import decide

LABELS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels_2026-10-28.json")))


def _got(name):
    v = LABELS[name]
    return decide(v["window"], v["cells"], v["rim"])


class TestDecideOnLabels(unittest.TestCase):
    def _check(self, label, expected):
        bad = [n for n, v in LABELS.items() if v["label"] == label and _got(n) != expected]
        self.assertEqual(bad, [], "%s: %d кадрів не %s" % (label, len(bad), expected))

    def test_clear_and_near_unchanged(self):
        self._check("clear", "clear")
        self._check("near", "near")

    def test_corner_cell_alone_is_not_near(self):
        """30.10.2026: порожній лоток, клітинка №2 = 0.43 (решта ≥0.5) — clear; та сама на 0.25 або інша клітинка на 0.43 — near."""
        ok = [0.91, 0.93, 0.43, 0.83, 0.88, 0.81, 0.87, 0.69]
        self.assertEqual(decide("empty", ok, None), "clear")
        self.assertEqual(decide("empty", ok[:2] + [0.225] + ok[3:], None), "near")  # чорний кіт збоку 26.10, поріг 0.24 (31.10)
        self.assertEqual(decide("empty", ok[:2] + [0.26] + ok[3:], None), "clear")  # порожній лоток 31.10 21:00
        self.assertEqual(decide("empty", [0.43] + ok[1:2] + [0.9] + ok[3:], None), "near")
        self.assertEqual(decide("empty", ok[:5] + [0.43] + ok[6:], None), "near")
        # 01.11.2026: той самий вектор із зайнятим вікном тепер «inside», а не «blocking». Клітинка №2 падає до 0.43
        # і на ПОРОЖНЬОМУ лотку (через фон за рамкою — див. SOFT_CELL вище), тож вона не є ознакою кота перед входом.
        # З її м'яким трактуванням розмітка Слави 01.11 дає «всередині» 85/85 замість 62/85, і в жодній із двох
        # розміток кіт ззовні не названий «всередині». На дозвіл чистки це не впливає: обидва вердикти її блокують.
        self.assertEqual(decide("occupied", ok, [0.5] * 4), "inside")
        self.assertEqual(decide("occupied", ok[:5] + [0.30] + ok[6:], [0.5] * 4), "blocking")   # справді закрита клітинка

    def test_lower_window_deviation_makes_near_inside(self):
        """Білий кіт у барабані, голова ззовні: вікно «порожнє», корпус закритий. Низ вікна не схожий на порожній → inside, інакше near."""
        import numpy as np
        from litter_algo import classify as C
        cells = [0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.9, 0.9]
        ex = C._library()[0]
        self.assertEqual(decide("empty", cells, None, ex), "near")
        bad = ex.copy()
        bad[32:] += 0.6
        self.assertEqual(decide("empty", cells, None, bad), "inside")
        self.assertEqual(decide("empty", [0.9] * 5 + [0.9] * 3, None, bad), "clear")
        self.assertEqual(decide("empty", [0.3] + [0.9] * 7, None, bad), "near")   # закрита верхня клітинка: низ вікна не вирішує

    def test_blocking_and_inside(self):
        self._check("blocking", "blocking")
        self._check("inside_blk", "blocking")   # Слава перекладав з inside: кіт ззовні, корпус закритий
        self._check("inside_ok", "inside")

    def test_empty_frame_in_inside_is_clear(self):
        """Єдиний кадр у inside, де Слава бачить порожньо (petbox-safe-clean-frames/20261026-070510-389): додано в бібліотеку еталонів, f 0.114→0.""" 
        v = LABELS["petbox-safe-clean-frames/20261026-070510-389.jpg"]
        self.assertEqual((v["label"], v["window"]), ("clear", "empty"))
        self.assertEqual(_got("petbox-safe-clean-frames/20261026-070510-389.jpg"), "clear")

    def test_nothing_occupied_becomes_clear(self):
        bad = [n for n, v in LABELS.items() if v["label"] in ("inside_ok", "inside_blk", "blocking", "inside_empty") and _got(n) == "clear"]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
