import statistics

import cv2
import numpy as np

from detect import Detection, detect

DAY_SATURATION_THRESHOLD = 20  # mean saturation below this => treated as IR/night frame


def is_day_frame(frame, sat_threshold: float = DAY_SATURATION_THRESHOLD) -> bool:
    """Image-based day/night guess via color saturation. Unreliable for scenes
    dominated by white/gray objects (e.g. a white litter box against white walls
    has low saturation even in bright daylight) -- prefer is_day_time() when a
    real clock is available (i.e. live, not replaying an offline test clip)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    return mean_sat >= sat_threshold


def is_day_time(hour: int, day_start_hour: int = 7, day_end_hour: int = 21) -> bool:
    """Wall-clock day/night check. Coarse (fixed hours, no actual sunrise/sunset
    calculation) but far more reliable than image saturation for scenes that are
    mostly white/gray regardless of lighting."""
    return day_start_hour <= hour < day_end_hour


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


def best_detection(detections: list[Detection], label: str) -> Detection | None:
    candidates = [d for d in detections if d.label == label]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence)


def classify_frames(frame_iter, config: dict, current_hour: int | None = None) -> list[dict]:
    """Runs the full detect -> color/ROI/dwell pipeline over an iterable of (t, frame)
    pairs (t = seconds since the event started) and returns a list of result dicts:
    [{cat, is_day, confidence, dwell_seconds, confirmed_visit, best_frame}, ...]. Used
    identically by the offline (Phase A, frames from a saved clip) and live (Phase B,
    frames from RTSP during a motion event) drivers.

    A single motion event can contain more than one real litter-box visit (e.g. the
    cat sits, gets up, sits again, all without the camera's motion state fully
    stopping in between) -- each stretch of ROI presence separated from the next by
    more than dwell_gap_tolerance_seconds is treated as its own episode. A gap
    shorter than that is still credited to the same episode, since this litter box
    is an enclosed dome and a cat genuinely inside is mostly invisible to the
    camera. Reusing the same tolerance for both "still the same visit" and "still
    the same episode" is deliberate: a shorter, separate split threshold would risk
    slicing one real visit into fragments that individually never reach
    min_dwell_seconds, which is the exact bug this tolerance was raised to fix.

    Episodes that don't reach min_dwell_seconds are collapsed back into a single
    "passby" result (using every cat sighting in the event, not just ROI ones) --
    we don't want to spam multiple weak passby notifications for fragments of one
    non-visit event.

    current_hour: if given (live use -- pass datetime.now().hour), day/night is
    decided from the wall clock via is_day_time() instead of image saturation.
    Offline clip replay has no real "now", so it falls back to the per-frame
    saturation vote (see is_day_frame's caveat about white/gray-dominated scenes)."""
    roi = config["roi"]
    thresholds = config["thresholds"]
    gap_tolerance = config["dwell_gap_tolerance_seconds"]
    min_dwell = config["min_dwell_seconds"]

    day_votes = []
    all_brightness_samples = []
    cat_frames = 0
    total_frames = 0
    max_conf = 0.0
    best_cat_frame = None
    best_cat_conf = 0.0

    episodes: list[dict] = []
    current: dict | None = None

    def close_current() -> None:
        nonlocal current
        if current is not None:
            episodes.append(current)
            current = None

    for t, frame in frame_iter:
        total_frames += 1
        if current_hour is None:
            day_votes.append(is_day_frame(frame))
        dets = detect(frame, config["yolo_confidence"])
        cat_det = best_detection(dets, "cat")

        inside = False
        if cat_det:
            cat_frames += 1
            max_conf = max(max_conf, cat_det.confidence)
            crop = crop_bbox(frame, cat_det.bbox)
            brightness = mean_brightness(crop)
            all_brightness_samples.append(brightness)
            center = bbox_center(cat_det.bbox)
            inside = point_in_roi(center, roi, frame.shape)
            if cat_det.confidence > best_cat_conf:
                best_cat_conf = cat_det.confidence
                best_cat_frame = frame

        # Episode boundary: a gap since the cat was last confirmed inside the ROI
        # longer than gap_tolerance ends the current episode. Shorter gaps don't
        # start a new episode, but also don't themselves count as dwell time below
        # -- only frames actually confirmed inside do (see dwell accumulation).
        if current is not None and (t - current["last_inside_t"]) > gap_tolerance:
            close_current()

        if inside:
            if current is None:
                current = {
                    "dwell": 0.0,
                    "last_inside_t": None,
                    "last_t": None,
                    "brightness": [],
                    "best_conf": 0.0,
                    "best_frame": None,
                }
            current["brightness"].append(brightness)
            if cat_det.confidence > current["best_conf"]:
                current["best_conf"] = cat_det.confidence
                current["best_frame"] = frame

        if current is not None:
            # Same accumulation as the original single-episode DwellTracker:
            # each frame confirmed inside (with a recent-enough prior inside
            # sighting) adds its own sampling interval to dwell -- so time the
            # cat is simply not detected (e.g. genuinely inside the dome) is
            # bridged without being counted as dwell itself.
            if current["last_t"] is not None:
                dt = t - current["last_t"]
                if inside and current["last_inside_t"] is not None:
                    current["dwell"] += dt
            if inside:
                current["last_inside_t"] = t
            current["last_t"] = t

    close_current()

    if current_hour is not None:
        is_day = is_day_time(current_hour)
    else:
        is_day = (sum(day_votes) / len(day_votes)) >= 0.5 if day_votes else True

    if total_frames == 0:
        return [{"cat": "unknown", "is_day": is_day, "confidence": 0.0, "dwell_seconds": 0.0}]

    confirmed = [ep for ep in episodes if ep["dwell"] >= min_dwell]
    if confirmed:
        results = []
        for ep in confirmed:
            brightness = statistics.median(ep["brightness"])
            label = classify_cat_color(brightness, is_day, thresholds)
            results.append(
                {
                    "cat": label,
                    "is_day": is_day,
                    "confidence": ep["best_conf"],
                    "dwell_seconds": ep["dwell"],
                    "confirmed_visit": True,
                    "best_frame": ep["best_frame"],
                }
            )
        return results

    if cat_frames == 0:
        return [
            {
                "cat": "unknown",
                "is_day": is_day,
                "confidence": 0.0,
                "dwell_seconds": 0.0,
                "confirmed_visit": False,
            }
        ]

    # Cat was seen but no episode dwelled in the ROI long enough for a real visit.
    # Still figure out which cat it was, if brightness allows, so a lighter
    # "passed by" notification can be sent.
    best_dwell = max((ep["dwell"] for ep in episodes), default=0.0)
    brightness = statistics.median(all_brightness_samples)
    color = classify_cat_color(brightness, is_day, thresholds)
    label = f"{color}_passby" if color != "unknown" else "pass_through"
    return [
        {
            "cat": label,
            "is_day": is_day,
            "confidence": max_conf,
            "dwell_seconds": best_dwell,
            "confirmed_visit": False,
            "best_frame": best_cat_frame,
        }
    ]
