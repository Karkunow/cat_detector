"""Тести тіньового режиму: збої й зависання не впливають на виклик, журнал пишеться, розбіжності класифікуються.
Без cv2: аналізатор підміняється. Запуск: python3 -m unittest litter_algo.test_shadow"""
import json
import os
import tempfile
import time
import unittest

from litter_algo import shadow


def _ok(verdict):
    return lambda p: {"algo": verdict, "window": "empty", "f": 0.01234567, "g": 0.1, "score": 0.5, "scale": 0.8, "cells": [0.6], "rim": [0.3]}


def _boom(p):
    raise RuntimeError("камера зсунулась")


def _hang(p):
    time.sleep(5)


class TestShadow(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.log = os.path.join(self.d, "sub", "shadow.jsonl")

    def rows(self):
        return [json.loads(l) for l in open(self.log, encoding="utf-8")]

    def test_returns_none_and_writes_row(self):
        self.assertIsNone(shadow.log_shadow("a.jpg", "allow", "x", "t", self.log, _ok("clear")))
        r = self.rows()[0]
        self.assertEqual((r["algo"], r["algo_allows"], r["agreement"], r["current"]["decision"]), ("clear", True, "agree", "allow"))
        self.assertEqual(r["f"], 0.0123)

    def test_agreement_kinds(self):
        for cur, algo, want in (("allow", "inside", "algo_stricter"), ("block", "clear", "algo_looser"),
                                ("block", "near", "agree"), (None, "clear", "no_current"), ("щось", "clear", "no_current")):
            shadow.log_shadow("a.jpg", cur, None, "", self.log, _ok(algo))
            self.assertEqual(self.rows()[-1]["agreement"], want, (cur, algo))

    def test_failure_is_swallowed_and_logged(self):
        self.assertIsNone(shadow.log_shadow("a.jpg", "allow", None, "", self.log, _boom))
        r = self.rows()[0]
        self.assertEqual((r["algo"], r["algo_allows"], r["agreement"]), ("error", None, "no_current"))
        self.assertIn("RuntimeError", r["error"])

    def test_hang_is_cut_by_time_limit(self):
        t = time.time()
        shadow.log_shadow("a.jpg", "allow", None, "", self.log, _hang, limit=1)
        self.assertLess(time.time() - t, 3)
        self.assertIn("timeout", self.rows()[0]["error"])

    def test_unwritable_log_does_not_raise(self):
        self.assertIsNone(shadow.log_shadow("a.jpg", "allow", None, "", "/proc/nope/x.jsonl", _ok("clear")))

    def test_default_analyzer_without_frame_does_not_raise(self):
        shadow.log_shadow(os.path.join(self.d, "нема.jpg"), "allow", None, "", self.log)
        self.assertEqual(self.rows()[0]["algo"], "error")

    def test_cli_always_exit_zero(self):
        self.assertEqual(shadow.main(["--frame", os.path.join(self.d, "нема.jpg"), "--current", "allow", "--log", self.log]), 0)
        self.assertEqual(shadow.main(["--bogus"]), 0)
        self.assertEqual(shadow.main([]), 0)

    def test_summary(self):
        shadow.log_shadow("a.jpg", "allow", None, "", self.log, _ok("inside"))
        shadow.log_shadow("b.jpg", "block", None, "", self.log, _ok("clear"))
        s = shadow.summarize(self.log)
        self.assertIn("записів: 2", s)
        self.assertIn("a.jpg", s)
        self.assertIn("b.jpg", s)


if __name__ == "__main__":
    unittest.main()
