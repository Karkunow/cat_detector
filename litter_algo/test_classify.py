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
