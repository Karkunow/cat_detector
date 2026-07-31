import statistics

import cv2
import numpy as np

from detect import Detection, detect

DAY_SATURATION_THRESHOLD = 20  # mean saturation below this => treated as IR/night frame


def is_day_frame(frame, sat_threshold: float = DAY_SATURATION_THRESHOLD) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    return mean_sat >= sat_threshold


def crop_bbox(frame, bbox: tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return frame[y1:y2, x1:x2]


def mean_brightness(crop) -> float:
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def classify_cat_color(brightness: float, is_day: bool, thresholds: dict) -> str:
    mode = "day" if is_day else "night"
    t = thresholds[mode]
    if brightness >= t["white_min"]:
        return "white"
    if brightness <= t["black_max"]:
        return "black"
    return "unknown"


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_roi(point: tuple[float, float], roi: list[float], frame_shape: tuple[int, int]) -> bool:
    """roi is [x1, y1, x2, y2] as fractions (0-1) of frame width/height."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi[0] * w, roi[1] * h, roi[2] * w, roi[3] * h
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


class DwellTracker:
    """Accumulates time a cat spends inside the ROI across frames of one event,
    tolerating short gaps (e.g. the cat turning/digging and briefly losing detection)."""

    def __init__(self, gap_tolerance_seconds: float = 2.0):
        self.gap_tolerance = gap_tolerance_seconds
        self.total_inside = 0.0
        self._last_inside_t: float | None = None
        self._last_t: float | None = None

    def update(self, t: float, inside_roi: bool) -> None:
        if self._last_t is not None:
            dt = t - self._last_t
            if inside_roi and self._last_inside_t is not None and (t - self._last_inside_t) <= self.gap_tolerance:
                self.total_inside += dt
            elif inside_roi and self._last_inside_t is None:
                pass  # first time entering, nothing to add yet
        if inside_roi:
            self._last_inside_t = t
        self._last_t = t

    @property
    def dwell_seconds(self) -> float:
        return self.total_inside


def best_detection(detections: list[Detection], label: str) -> Detection | None:
    candidates = [d for d in detections if d.label == label]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence)


def classify_frames(frame_iter, config: dict) -> dict:
    """Runs the full detect -> color/ROI/dwell pipeline over an iterable of (t, frame)
    pairs (t = seconds since the event started) and returns a result dict:
    {cat, is_day, confidence, dwell_seconds}. Used identically by the offline
    (Phase A, frames from a saved clip) and live (Phase B, frames from RTSP during
    a motion event) drivers."""
    roi = config["roi"]
    thresholds = config["thresholds"]
    dwell = DwellTracker(gap_tolerance_seconds=config["dwell_gap_tolerance_seconds"])

    day_votes = []
    brightness_samples = []
    person_frames = 0
    cat_frames = 0
    total_frames = 0
    max_conf = 0.0

    for t, frame in frame_iter:
        total_frames += 1
        day_votes.append(is_day_frame(frame))
        dets = detect(frame, config["yolo_confidence"])
        cat_det = best_detection(dets, "cat")
        person_det = best_detection(dets, "person")

        inside = False
        if cat_det:
            cat_frames += 1
            max_conf = max(max_conf, cat_det.confidence)
            crop = crop_bbox(frame, cat_det.bbox)
            brightness_samples.append(mean_brightness(crop))
            center = bbox_center(cat_det.bbox)
            inside = point_in_roi(center, roi, frame.shape)
        elif person_det:
            person_frames += 1
            max_conf = max(max_conf, person_det.confidence)

        dwell.update(t, inside)

    is_day = (sum(day_votes) / len(day_votes)) >= 0.5 if day_votes else True

    if total_frames == 0:
        return {"cat": "unknown", "is_day": is_day, "confidence": 0.0, "dwell_seconds": 0.0}

    # A confirmed dwell in the ROI (sustained cat presence at the box) wins even if a
    # person also appears elsewhere in the same event (e.g. owner walks by earlier/later).
    if dwell.dwell_seconds >= config["min_dwell_seconds"]:
        brightness = statistics.median(brightness_samples)
        label = classify_cat_color(brightness, is_day, thresholds)
        confirmed_visit = True
    elif person_frames > cat_frames and person_frames > total_frames * 0.15:
        return {
            "cat": "human",
            "is_day": is_day,
            "confidence": max_conf,
            "dwell_seconds": 0.0,
            "confirmed_visit": False,
        }
    elif cat_frames == 0:
        return {
            "cat": "unknown",
            "is_day": is_day,
            "confidence": 0.0,
            "dwell_seconds": 0.0,
            "confirmed_visit": False,
        }
    else:
        # Cat was seen but didn't dwell in the ROI long enough for a real visit.
        # Still figure out which cat it was, if brightness allows, so a lighter
        # "passed by" notification can be sent.
        brightness = statistics.median(brightness_samples)
        color = classify_cat_color(brightness, is_day, thresholds)
        label = f"{color}_passby" if color != "unknown" else "pass_through"
        confirmed_visit = False

    return {
        "cat": label,
        "is_day": is_day,
        "confidence": max_conf,
        "dwell_seconds": dwell.dwell_seconds,
        "confirmed_visit": confirmed_visit,
    }
