"""Тести розпізнавання руху очищення. Запуск: python3 -m unittest test_cleaning -v
Рядки журналу взято з реального ~/petbox-guard.log (28.10.2026)."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cleaning

LOG = """2026-10-28 19:38:30 стан: - → очищення (completion=1, error=1, вага=20000)
2026-10-28 19:40:11 стан: очищення → - (completion=1, error=1, вага=20000)
2026-10-28 20:28:23 стан: - → очищення (completion=1, error=1, вага=20000)
2026-10-28 20:30:24 стан: очищення → - (completion=1, error=1, вага=20000)
2026-10-28 20:31:00 помилка: 1 → 4096
2026-10-28 21:29:38 стан: - → очищення (completion=1, error=1, вага=20000)
"""


def ts(hms: str, day: str = "2026-10-28") -> float:
    return datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S").timestamp()


class Cleaning(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = Path(self.dir.name) / "guard.log"
        self.log.write_text(LOG, encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def check(self, a, b):
        return cleaning.is_cleaning_event(ts(a), ts(b), self.log)

    def test_event_357_is_cleaning(self):
        self.assertTrue(self.check("20:29:32", "20:29:38"))

    def test_event_outside_cycle_is_not(self):
        self.assertFalse(self.check("20:12:00", "20:12:03"))

    def test_event_only_touching_cycle_start_is_not(self):
        self.assertFalse(self.check("20:27:00", "20:28:40"))

    def test_event_straddling_cycle_end_is_not(self):
        self.assertFalse(self.check("20:30:00", "20:31:30"))

    def test_open_cycle_counts_as_cleaning(self):
        self.assertTrue(self.check("21:30:00", "21:30:05"))

    def test_missing_log_is_not_cleaning(self):
        self.assertFalse(cleaning.is_cleaning_event(ts("20:29:32"), ts("20:29:38"), Path(self.dir.name) / "none.log"))

    def test_phase_changes_stay_in_one_cycle(self):
        self.log.write_text(
            "2026-10-28 10:00:00 стан: - → очищення (x)\n2026-10-28 10:01:00 стан: очищення → пересипання (x)\n"
            "2026-10-28 10:02:00 стан: пересипання → - (x)\n", encoding="utf-8")
        self.assertEqual(len(cleaning.cycle_windows(self.log)), 1)
        self.assertTrue(self.check("10:01:10", "10:01:20"))


if __name__ == "__main__":
    unittest.main()
