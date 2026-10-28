"""Тести сторожа положення. Запуск: cd ~/cat_detector && .venv/bin/python -m unittest test_align_guard -v
Використовує реальні збережені кадри data/frames/events (без кота і з котом), тимчасові теки для стану/еталонів."""
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

import align_guard as ag

D = Path(__file__).parent / "data"
EV = D / "frames" / "events"
TPL = ag.Template(cv2.imread(str(D / "drum_base_irday.jpg")))
IMG = lambda rel: cv2.imread(str(EV / rel))
EMPTY = ["20261026-081637/004.jpg", "20261026-082005/014.jpg", "20261025-150118/002.jpg"]   # порожній лоток, поза 0
CAT = "20261026-030445/062.jpg"                                                            # кіт перед лотком
OUT = "20261025-134705/022.jpg"                                                            # камера дивиться вниз, корпус зрізано
NIGHT = "20261026-032026/006.jpg"


def chk(frames, busy, ts):
    return ag.check_frames(frames, TPL, busy, ts)


class Pose(unittest.TestCase):
    def test_empty_good(self):
        for r in EMPTY + [NIGHT]:
            c = chk([IMG(r)] * 3, False, 0)
            self.assertTrue(c["good"], (r, c))
            self.assertLessEqual(max(abs(c["pose"][0]), abs(c["pose"][1])), 6)

    def test_cat_not_trusted_as_good_unless_pose_ok(self):
        c = chk([IMG(CAT)] * 3, True, 0)
        self.assertFalse(c["good"])

    def test_out_of_frame_never_good(self):
        c = chk([IMG(OUT)] * 3, False, 0)
        self.assertFalse(c["good"])
        self.assertFalse(c["inb"])

    def test_degraded_frames_not_good(self):
        im = IMG(EMPTY[0])
        for name, f in {"black": np.zeros_like(im), "blur": cv2.GaussianBlur(im, (0, 0), 25), "flip": cv2.flip(im, 1),
                        "zoom": cv2.warpAffine(im, cv2.getRotationMatrix2D((640, 360), 0, 1.4), (1280, 720))}.items():
            self.assertFalse(chk([f] * 3, False, 0)["good"], name)

    def test_shift_tracked_with_pose(self):
        im = IMG(EMPTY[0])
        out = cv2.warpAffine(im, np.float32([[1, 0, -200], [0, 1, -30]]), (1280, 720), borderMode=cv2.BORDER_REPLICATE)
        c = chk([out] * 3, False, 0)
        self.assertTrue(c["good"])
        self.assertLessEqual(abs(c["pose"][0] + 200), 8)
        self.assertLessEqual(abs(c["pose"][1] + 30), 8)

    def test_no_frames(self):
        c = chk([], False, 0)
        self.assertFalse(c["good"])
        self.assertEqual(c["n"], 0)


def feed(g, frames, busy, start, n, step=60):
    for i in range(n):
        g.update(chk(frames, busy, start + i * step))
    return start + n * step


class Machine(unittest.TestCase):
    def setUp(self):
        ag.CFG["night_from"], ag.CFG["night_to"] = 23, 8
        self.g = ag.Guard(now=0.0)
        self.g.mode = "irday"
        self.g.ref_poses = {"irday": (0, 0), "ir": (-138, -25)}

    def test_start_uncertain_then_tracked_after_2_min(self):
        e = [IMG(EMPTY[0])] * 3
        self.g.update(chk(e, False, 60)); self.assertEqual(self.g.state, ag.UNCERTAIN)
        self.g.update(chk(e, False, 120)); self.assertEqual(self.g.state, ag.UNCERTAIN)
        self.g.update(chk(e, False, 180)); self.assertEqual(self.g.state, ag.TRACKED)
        self.assertTrue(self.g.refs_match())

    def test_cat_in_front_does_not_become_lost(self):
        t = feed(self.g, [IMG(EMPTY[0])] * 3, False, 60, 3)
        t = feed(self.g, [IMG(CAT)] * 3, True, t, 60)    # година кота перед лотком (рух/кіт у кадрі)
        self.assertEqual(self.g.state, ag.UNCERTAIN)
        self.assertIsNone(self.g.notice_due(t + 3600 * 24, 12))    # година кота — повідомлення нема

    def test_lost_when_box_out_of_frame_quiet(self):
        t = feed(self.g, [IMG(EMPTY[0])] * 3, False, 60, 3)
        t = feed(self.g, [IMG(OUT)] * 3, False, t, 2)
        self.assertEqual(self.g.state, ag.UNCERTAIN)
        t = feed(self.g, [IMG(OUT)] * 3, False, t, 3)    # ≥3 хв тихо
        self.assertEqual(self.g.state, ag.LOST)
        self.assertFalse(self.g.refs_match())

    def test_black_frame_lost_by_floor(self):
        t = feed(self.g, [IMG(EMPTY[0])] * 3, False, 60, 3)
        t = feed(self.g, [np.zeros((720, 1280, 3), np.uint8)] * 3, False, t, 4)
        self.assertEqual(self.g.state, ag.LOST)

    def test_no_frames_lost(self):
        t = feed(self.g, [IMG(EMPTY[0])] * 3, False, 60, 3)
        t = feed(self.g, [], False, t, 4)
        self.assertEqual(self.g.state, ag.LOST)

    def test_recovery_needs_hysteresis(self):
        t = feed(self.g, [IMG(OUT)] * 3, False, 60, 6)
        self.assertEqual(self.g.state, ag.LOST)
        t = feed(self.g, [IMG(EMPTY[0])] * 3, False, t, 2)
        self.assertEqual(self.g.state, ag.LOST)
        t = feed(self.g, [IMG(EMPTY[0])] * 3, False, t, 1)
        self.assertEqual(self.g.state, ag.TRACKED)

    def test_refs_mismatch_after_shift(self):
        im = IMG(EMPTY[0])
        out = cv2.warpAffine(im, np.float32([[1, 0, -100], [0, 1, 0]]), (1280, 720), borderMode=cv2.BORDER_REPLICATE)
        feed(self.g, [out] * 3, False, 60, 3)
        self.assertEqual(self.g.state, ag.TRACKED)
        self.assertFalse(self.g.refs_match())     # еталони зняті на позі 0 → чистка блокується

    def test_night_without_ref_pose_blocks(self):
        self.g.ref_poses = {"irday": (0, 0)}
        c = [chk([IMG(NIGHT)] * 3, False, 60 * i) for i in (1, 2, 3)]
        for x in c: self.g.update(x)
        self.assertEqual(self.g.mode, "ir")
        self.assertFalse(self.g.refs_match())

    def test_pose_jump_in_tracked(self):
        t = feed(self.g, [IMG(EMPTY[0])] * 3, False, 60, 3)
        im = cv2.warpAffine(IMG(EMPTY[0]), np.float32([[1, 0, -60], [0, 1, 0]]), (1280, 720), borderMode=cv2.BORDER_REPLICATE)
        self.g.update(chk([im] * 3, False, t))
        self.assertEqual(self.g.state, ag.UNCERTAIN)


class Notices(unittest.TestCase):
    def mk(self):
        g = ag.Guard(now=0.0)
        g.state, g.since, g.bad_since, g.reason = ag.LOST, 0.0, 0.0, "корпус частково поза кадром"
        return g

    def test_lost_notice_once_per_6h_and_max_3(self):
        g = self.mk()
        base = time.mktime((2026, 10, 26, 10, 0, 0, 0, 0, -1))
        g.bad_since = base
        self.assertIsNotNone(g.notice_due(base + 5, 10)); g.mark_notice(base + 5, "x")
        self.assertIsNone(g.notice_due(base + 3600, 11))
        self.assertIsNone(g.notice_due(base + 5 * 3600, 15))
        self.assertIsNotNone(g.notice_due(base + 6 * 3600 + 10, 16)); g.mark_notice(base + 6 * 3600 + 10, "x")
        self.assertIsNotNone(g.notice_due(base + 12 * 3600 + 20, 22)); g.mark_notice(base + 12 * 3600 + 20, "x")
        self.assertIsNone(g.notice_due(base + 18 * 3600 + 30, 9))      # 3 за добу вже було

    def test_night_deferred(self):
        g = self.mk()
        for h in (23, 0, 3, 7):
            self.assertIsNone(g.notice_due(1e9, h))
        self.assertIsNotNone(g.notice_due(1e9, 8))

    def test_uncertain_short_no_notice_long_yes(self):
        b = 1e9
        g = ag.Guard(now=b)
        g.bad_since, g.bad_quiet = b, 400.0
        self.assertIsNone(g.notice_due(b + 600, 12))
        self.assertIsNotNone(g.notice_due(b + 1900, 12))
        g.bad_quiet = 0.0      # весь час рух/кіт у кадрі — не просимо навести камеру
        self.assertIsNone(g.notice_due(b + 7200, 12))

    def test_tracked_no_notice(self):
        g = ag.Guard(now=0.0); g.state = ag.TRACKED; g.bad_since = None
        self.assertIsNone(g.notice_due(1e9, 12))

    def test_notice_file_atomic(self):
        with tempfile.TemporaryDirectory() as t:
            p = ag.write_notice(ag.notice_text({"reason": "тест"}, time.time()), Path(t))
            self.assertTrue(p.name.endswith("-cat-detector-camera-01.txt"))
            self.assertEqual([x.name for x in Path(t).iterdir()], [p.name])    # .part не лишився


class StateFile(unittest.TestCase):
    def test_roundtrip_and_blend(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "s.json"
            self.assertFalse(ag.blend_allowed(path=p))
            g = ag.Guard(now=0.0); g.state, g.mode, g.pose = ag.TRACKED, "irday", (0, 0)
            g.ref_poses = {"irday": (0, 0)}
            now = time.time()
            ag.write_state(g.to_dict(now), p)
            self.assertTrue(ag.blend_allowed(now, p))
            self.assertFalse(ag.blend_allowed(now + 1000, p))        # застарів
            g.enhanced_until = now + 100
            ag.write_state(g.to_dict(now), p)
            self.assertFalse(ag.blend_allowed(now, p))               # посилений режим
            p.write_text("{битий")
            self.assertFalse(ag.blend_allowed(now, p))


class Candidates(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        for n in ("drum_ref_irday.npy", "drum_ref_irday_side.npy", "drum_ref_irday_ratio.npy", "drum_ref_irday2.npy", "drum_ref_irday2_side.npy",
                  "drum_ref_irday2_ratio.npy", "drum_ref_ir.npy", "drum_ref_ir_side.npy", "drum_ref_ir_ratio.npy", "drum_base_irday.jpg", "drum_base_irday2.jpg", "drum_base_ir.jpg"):
            shutil.copy(D / n, self.t / n)
        (self.t / "align_refs.json").write_text(json.dumps({"irday": [0, 0], "irday2": [0, 0], "ir": [-138, -25]}))

    def tearDown(self):
        shutil.rmtree(self.t)

    def test_quiet_window_each_condition(self):
        g = ag.Guard(now=0.0); g.state = ag.TRACKED
        g.recent = [{"good": True, "busy": False, "pose": (0.0, 0.0)}] * 5
        q = lambda **k: ag.quiet_window(g, 1e6, **{**dict(capturing=False, processing=False, motion_active=False, last_motion_ts=0.0, last_event_ts=None, inside_unresolved=False), **k})
        self.assertIsNone(q())
        for kw in (dict(capturing=True), dict(processing=True), dict(motion_active=True), dict(last_motion_ts=1e6 - 600),
                   dict(last_event_ts=1e6 - 600), dict(inside_unresolved=True)):
            self.assertIsNotNone(q(**kw), kw)
        g.recent = g.recent[:4]; self.assertIsNotNone(q())
        g.recent = [{"good": True, "busy": True, "pose": (0.0, 0.0)}] * 5; self.assertIsNotNone(q())
        g.recent = [{"good": True, "busy": False, "pose": (float(i * 5), 0.0)} for i in range(5)]; self.assertIsNotNone(q())
        g.recent = [{"good": True, "busy": False, "pose": (0.0, 0.0)}] * 5; g.state = ag.UNCERTAIN; self.assertIsNotNone(q())

    def _mk(self, n, gap=7300, frame=EMPTY[0], pose=(0.0, 0.0), mode="irday"):
        im = IMG(frame)
        return [ag.capture_candidate(im, pose, mode, 1_000_000 + i * gap, self.t) for i in range(n)]

    def test_consensus_needs_3_with_gaps(self):
        self._mk(2)
        self.assertIsNone(ag.consensus(ag.list_candidates("irday", 1_020_000, self.t)))
        self._mk(3)
        self.assertIsNotNone(ag.consensus(ag.list_candidates("irday", 1_020_000, self.t)))

    def test_consensus_rejects_close_in_time(self):
        self._mk(5, gap=600)
        self.assertIsNone(ag.consensus(ag.list_candidates("irday", 1_020_000, self.t)))

    def test_consensus_rejects_cat_candidate(self):
        self._mk(2, frame=EMPTY[0])
        ag.capture_candidate(IMG(CAT), (0.0, 0.0), "irday", 1_000_000 + 2 * 7300, self.t)   # «кіт» серед кандидатів
        self.assertIsNone(ag.consensus(ag.list_candidates("irday", 1_020_000, self.t)))

    def test_old_ref_match_rejects_cat_and_accepts_empty(self):
        ok = ag.capture_candidate(IMG(EMPTY[0]), (0.0, 0.0), "irday", 1.0, self.t)
        bad = ag.capture_candidate(IMG(CAT), (0.0, 0.0), "irday", 2.0, self.t)
        rp = {"irday": (0, 0)}
        c_ok, c_bad = [ag._load_cand(d) for d in (ok, bad)]
        self.assertTrue(ag.matches_old(c_ok, "irday", rp, self.t)[0])
        self.assertFalse(ag.matches_old(c_bad, "irday", rp, self.t)[0])
        self.assertEqual(ag.matches_old(c_ok, "irdayX", rp, self.t), (None, None))

    def test_candidate_does_not_touch_working_refs(self):
        before = {p.name: p.read_bytes() for p in self.t.glob("drum_*")}
        self._mk(3)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.t.glob("drum_*")})

    def test_promote_and_rollback(self):
        group = ag.consensus(ag.list_candidates("irday", 0, self.t) or [], ) or [ag._load_cand(d) for d in self._mk(3)]
        before = {p.name: p.read_bytes() for p in self.t.glob("drum_*")}
        info = ag.promote(group, "irday", 1_100_000, self.t)
        self.assertGreater(info["enhanced_until"], 1_100_000)
        prev = Path(info["dir"]) / "prev"
        self.assertTrue((Path(info["dir"]) / "SHA256SUMS").exists())
        self.assertEqual(json.loads((self.t / "align_refs.json").read_text())["irday"], [0.0, 0.0])
        self.assertEqual(json.loads((self.t / "align_refs.json").read_text())["ir"], [-138, -25])   # інші режими лишаються застарілими
        for f in prev.iterdir():     # відкат = копія prev назад
            shutil.copy(f, self.t / f.name)
        after = {p.name: p.read_bytes() for p in self.t.glob("drum_*")}
        self.assertEqual(before, after)
        self.assertFalse(list(self.t.glob("*.tmp*")))


if __name__ == "__main__":
    unittest.main()
