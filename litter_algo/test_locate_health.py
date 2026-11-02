"""Телеметрія «чи не зсунули камеру» — без cv2 і без камери."""
import os
import tempfile
import time
import unittest

from litter_algo import locate_health as H


class Health(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._old = H.STATE
        H.STATE = os.path.join(self.dir, "locate_health.json")
        H._buf, H._flushed = [], 0.0

    def tearDown(self):
        H.STATE = self._old
        H._buf, H._flushed = [], 0.0

    def _fill(self, score, n, now):
        for i in range(n):
            H.record(score, "ref.jpg", now=now - n + i)
        H._flushed = 0.0
        H.record(score, "ref.jpg", now=now)

    def test_too_few_samples_is_unknown(self):
        now = time.time()
        self._fill(0.4, 2, now)
        st = H.status(now)
        self.assertIsNone(st["shifted"])          # замало даних — висновку не робимо
        self.assertIsNone(st["median"])

    def test_good_score_is_not_shifted(self):
        now = time.time()
        self._fill(0.95, 12, now)
        st = H.status(now)
        self.assertFalse(st["shifted"])
        self.assertAlmostEqual(st["median"], 0.95, places=3)

    def test_low_score_is_shifted(self):
        now = time.time()
        self._fill(0.47, 12, now)                 # так виглядала зсунута камера 02.11.2026
        self.assertTrue(H.status(now)["shifted"])

    def test_old_samples_fall_out_of_window(self):
        now = time.time()
        self._fill(0.47, 12, now - H.WINDOW_SEC - 60)
        self.assertIsNone(H.status(now)["shifted"])

    def test_single_bad_frame_does_not_trigger(self):
        now = time.time()
        for i in range(11):
            H.record(0.95, "ref.jpg", now=now - 20 + i)
        H._flushed = 0.0
        H.record(0.10, "ref.jpg", now=now)        # один поганий кадр медіану не зрушить
        self.assertFalse(H.status(now)["shifted"])

    def test_heal_cooldown(self):
        now = time.time()
        self._fill(0.47, 12, now)
        self.assertTrue(H.heal_allowed(now))
        H.mark_healed("ref_box_autoX.jpg", now=now)
        self.assertFalse(H.heal_allowed(now + 60))
        self.assertTrue(H.heal_allowed(now + H.HEAL_COOLDOWN + 1))

    def test_mark_healed_resets_window(self):
        now = time.time()
        self._fill(0.47, 12, now)
        H.mark_healed("ref_box_autoX.jpg", now=now)
        H._buf = []
        self.assertIsNone(H.status(now)["shifted"])   # старі низькі score вже не про поточний стан

    def test_record_never_raises(self):
        H.record(None)                                 # None ігнорується
        H.STATE = "/nonexistent/dir/x.json"            # недоступний шлях — теж не падаємо
        H._flushed = 0.0
        H.record(0.5, "ref.jpg")


if __name__ == "__main__":
    unittest.main()
