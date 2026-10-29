import logging
import statistics

import cv2
import numpy as np

import drum_algo
from detect import Detection, detect
from align_guard import blend_allowed
from drum_check import align_with_shift, NCC_CAT, REF_UPDATE_NCC, RIM_CAT, SIDE_CAT, blend_ref, load_refs, observe, ratio_color, save_ref

DAY_SATURATION_THRESHOLD = 20  # mean saturation below this => treated as IR/night frame


def is_day_frame(frame, sat_threshold: float = DAY_SATURATION_THRESHOLD) -> bool:
    """Image-based day/night guess via color saturation. Unreliable for scenes
    dominated by white/gray objects (e.g. a white litter box against white walls
    has low saturation even in bright daylight) -- prefer is_day_time() when a
    real clock is available (i.e. live, not replaying an offline test clip)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    return mean_sat >= sat_threshold


def color_spread(frame) -> float:
    """Mean |B-G|+|G-R|. IR night frames from the camera are near-gray (spread ~0),
    even when the wall clock still says day; a lit room has noticeably more."""
    f = frame.astype(np.int16)
    return float(np.mean(np.abs(f[:, :, 0] - f[:, :, 1]) + np.abs(f[:, :, 1] - f[:, :, 2])))


IR_SPREAD_THRESHOLD = 4.0
IR_DAY_WALL_MIN = 112  # у нічному ІЧ «стіна» ~98, у сірому денному кадрі >120
IR_DAY_THRESHOLDS = {"white_min": 0.55, "black_max": 0.50}  # 25-й перцентиль кота / фасад (медіана по подіях). Вночі: чорний 0.30–0.38, білий 0.54–0.66; вдень (відео 22.10): чорний 0.46–0.48, білий 0.62–0.85


def wall_and_front(frame, shift=(0.0, 0.0)) -> tuple[float, float]:
    """Яскравість стіни зліва і білого фасаду лотка (ділянки кадру 1280x720; shift — на скільки зсунулась камера від еталонного положення)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    sx, sy = int(round(shift[0])), int(round(shift[1]))

    def region(x1, y1, x2, y2):
        x1, x2 = max(0, int(x1 * w) + sx), min(w, int(x2 * w) + sx)
        y1, y2 = max(0, int(y1 * h) + sy), min(h, int(y2 * h) + sy)
        return g[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else g[0:1, 0:1]

    wall = float(np.mean(region(.016, .42, .125, .64)))
    front = float(np.mean(region(.18, .69, .47, .78)))
    return wall, front


BRIGHT_PIXEL, BRIGHT_WHITE_FRAC = 150, 0.35  # білий кіт близько до камери пересвічений: ≥35% пікселів кота яскравіші за 150 (чорні в ІЧ: ≤28%); було 0.40, 29.10 00:34 білий мав 39% і став «чорним»


BRIGHT_MIN_RATIO = 0.40  # кіт темніший за 0.40 фасаду — чорний напевно (чорний 0.23–0.38, білий ≥0.54): яскраві пікселі в рамці — це білий лоток за ним; 29.10 01:07 чорний (0.23) став «білим»


def bright_override(color: str, frame, bbox, ratio: float | None = None) -> str:
    """Чорний за відношенням до фасаду, але кадр кота майже весь яскравий — це білий кіт (медіана по кадрах події тягнулась униз фоном)."""
    if color == "white" or frame is None or bbox is None:
        return color
    if ratio is not None and ratio < BRIGHT_MIN_RATIO:
        return color
    v = cv2.cvtColor(crop_bbox(frame, bbox), cv2.COLOR_BGR2GRAY).ravel()
    if v.size and float((v > BRIGHT_PIXEL).mean()) >= BRIGHT_WHITE_FRAC:
        logging.getLogger(__name__).info(f"кіт на кадрі яскравий ({100 * float((v > BRIGHT_PIXEL).mean()):.0f}% пікселів >{BRIGHT_PIXEL}): {color} -> white")
        return "white"
    return color


def classify_ir_day(brightness: float, front: float) -> str:
    """Сірий денний режим: абсолютні нічні пороги тут дають «білий → чорний», тож міряємо відносно фасаду."""
    r = brightness / max(front, 1.0)
    if r >= IR_DAY_THRESHOLDS["white_min"]:
        return "white"
    if r <= IR_DAY_THRESHOLDS["black_max"]:
        return "black"
    return "unknown"


def sun_is_up(now=None, lat=38.72, lon=-9.14):
    """День/ніч за висотою сонця над Лісабоном (NOAA): ІЧ-режим камери вдень не означає ніч."""
    import math
    from datetime import datetime, timezone
    t = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    g = 2 * math.pi / 365 * (t.timetuple().tm_yday - 1 + (t.hour - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    d = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    ha = math.radians((t.hour * 60 + t.minute + eq + 4 * lon) / 4 - 180)
    la = math.radians(lat)
    elev = 90 - math.degrees(math.acos(math.sin(la) * math.sin(d) + math.cos(la) * math.cos(d) * math.cos(ha)))
    return elev > -0.833


def is_day_time(hour: int, day_start_hour: int = 7, day_end_hour: int = 21) -> bool:
    """Wall-clock day/night check. Coarse (fixed hours, no actual sunrise/sunset
    calculation) but far more reliable than image saturation for scenes that are
    mostly white/gray regardless of lighting."""
    return day_start_hour <= hour < day_end_hour


def is_day_time_color(spreads: list[float]) -> bool:
    """Кадри кольорові (камера не в ІЧ)."""
    return bool(spreads) and statistics.median(spreads) >= IR_SPREAD_THRESHOLD


def crop_bbox(frame, bbox: tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return frame[y1:y2, x1:x2]


def dark_brightness(crop) -> float:
    """25-й перцентиль сірого в рамці кота: чорний кіт лишається темним, навіть коли рамка захопила білий фасад лотка."""
    if crop.size == 0:
        return 0.0
    return float(np.percentile(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 25))


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


DRUM_WINDOW = (0.24, 0.22, 0.39, 0.50)  # вікно барабана, частки кадру 1280x720 (як у petbox-safe-clean)


def _overlap(bbox: tuple[int, int, int, int], frame_shape) -> float:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    wx1, wy1, wx2, wy2 = DRUM_WINDOW[0] * w, DRUM_WINDOW[1] * h, DRUM_WINDOW[2] * w, DRUM_WINDOW[3] * h
    ix, iy = min(x2, wx2) - max(x1, wx1), min(y2, wy2) - max(y1, wy1)
    return max(ix, 0) * max(iy, 0) / ((wx2 - wx1) * (wy2 - wy1))


def best_detection(detections: list[Detection], label: str, frame_shape=None) -> Detection | None:
    """Найвпевненіший кіт. Якщо в кадрі їх кілька, береться той, що перекриває вікно барабана (22.10 16:22: чорний
    стояв ближче до об'єктива, а у барабан заходив білий — найвпевненішим був чорний)."""
    candidates = [d for d in detections if d.label == label]
    if not candidates:
        return None
    if frame_shape is not None and len(candidates) > 1:
        at_drum = [d for d in candidates if _overlap(d.bbox, frame_shape) >= 0.25]
        if at_drum:
            return max(at_drum, key=lambda d: _overlap(d.bbox, frame_shape))
    return max(candidates, key=lambda d: d.confidence)


def drum_inside_confirmed(last3: list, config: dict) -> tuple[bool, int | None]:
    """Чи підтверджують останні спостереження барабана «кіт усередині»: (так/ні, довжина хвоста inside від litter_algo).
    Основне — litter_algo (config drum_algo, за замовчуванням true); якщо він збійнув або вимкнений, діють старі NCC-ознаки
    (друге значення тоді None). Кіт ззовні (cat_outside) блокує в обох режимах. Кожне спостереження має мати "frame"."""
    if len(last3) < 2 or any(o["cat_outside"] for o in last3):
        return False, None
    if config.get("drum_algo", True):
        run = drum_algo.inside_run(drum_algo.classes([o["frame"] for o in last3]))
        if run is not None:
            logging.getLogger(__name__).info("drum_algo: inside-хвіст %d з %d кадрів", run, len(last3))
            return run > 0, run
        logging.getLogger(__name__).warning("drum_algo збійнув, перевіряємо старими NCC-ознаками")
    return (
        last3[-1]["ncc"] < NCC_CAT
        and sum(o["ncc"] < NCC_CAT for o in last3) >= 2
        and sum(o["side"] < SIDE_CAT for o in last3) < 2
        and sum(o.get("rim") is not None and o["rim"] < RIM_CAT for o in last3) < 2
    ), None


def drum_inside_check(frames: list, config: dict) -> dict | None:
    """Кіт сидить у барабані, хоч руху вже нема (камера його не бачить): ті самі критерії, що в кінці classify_frames,
    але на кількох свіжих кадрах без події. Повертає {"color", "frame"} або None."""
    refs = load_refs()
    obs = []
    for f in frames:
        o = observe(f, refs)
        if o is None:
            continue
        cat_det = best_detection(detect(f, config["yolo_confidence"]), "cat", f.shape)
        o["cat_outside"] = bool(cat_det) and not point_in_roi(bbox_center(cat_det.bbox), config["roi"], f.shape)
        o["frame"] = f
        obs.append(o)
    last3 = obs[-3:]
    if drum_inside_confirmed(last3, config)[0]:
        return {"color": ratio_color([o["ratio"] for o in last3]), "frame": last3[-1]["frame"], "ncc": last3[-1]["ncc"]}
    return None


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
    max_visible_density = config.get("dwell_max_visible_density", 0.65)
    hidden_s = config.get("inside_hidden_seconds", 4.0)
    last_outside_t = -1.0
    last_t = 0.0
    last_frame = None
    refs = load_refs()
    refs_dirty = set()
    drum_obs = []
    sample_interval = 1.0 / config["sample_fps"]

    day_votes = []
    spreads = []
    walls = []
    fronts = []
    all_brightness_samples = []
    cat_frames = 0
    total_frames = 0
    max_conf = 0.0
    best_cat_frame = None
    best_cat_bbox = None
    best_cat_conf = 0.0
    cam_shift = None

    episodes: list[dict] = []
    current: dict | None = None

    def close_current() -> None:
        nonlocal current
        if current is not None:
            episodes.append(current)
            current = None

    for t, frame in frame_iter:
        total_frames += 1
        last_t, last_frame = t, frame
        if current_hour is None:
            day_votes.append(is_day_frame(frame))
        else:
            spreads.append(color_spread(frame))
            if cam_shift is None:
                bases = [r["base"] for r in refs.values() if r.get("base")]
                cam_shift = align_with_shift(frame, bases[0])[1] if bases else (0.0, 0.0)
            wl, fr = wall_and_front(frame, cam_shift)
            walls.append(wl)
            fronts.append(fr)
        dets = detect(frame, config["yolo_confidence"])
        cat_det = best_detection(dets, "cat", frame.shape)

        inside = False
        if cat_det:
            cat_frames += 1
            max_conf = max(max_conf, cat_det.confidence)
            crop = crop_bbox(frame, cat_det.bbox)
            ir_frame = current_hour is not None and bool(spreads) and spreads[-1] < IR_SPREAD_THRESHOLD
            brightness = dark_brightness(crop) if ir_frame else mean_brightness(crop)
            all_brightness_samples.append(brightness)
            center = bbox_center(cat_det.bbox)
            inside = point_in_roi(center, roi, frame.shape)
            if cat_det.confidence > best_cat_conf:
                best_cat_conf = cat_det.confidence
                best_cat_frame = frame
                best_cat_bbox = cat_det.bbox
            if not inside:
                last_outside_t = t

        obs = observe(frame, refs)
        if obs:
            obs.update(t=t, cat_outside=bool(cat_det) and not inside, frame=frame)
            drum_obs.append(obs)
            if len(drum_obs) > drum_algo.TAIL:
                drum_obs[-drum_algo.TAIL - 1].pop("frame", None)   # кадри потрібні лише для останніх TAIL спостережень
            if obs["ncc"] >= REF_UPDATE_NCC and obs["side"] >= REF_UPDATE_NCC and blend_allowed():
                blend_ref(refs[obs["mode"]], obs["cur"])
                refs_dirty.add(obs["mode"])

        # Episode boundary: a gap since the cat was last confirmed inside the ROI
        # longer than gap_tolerance ends the current episode. A shorter gap is
        # still credited as dwell time (see below) -- this litter box is an
        # enclosed dome, so a cat genuinely inside is invisible to the camera for
        # most of a real visit. Only counting frames actually confirmed inside
        # (as an earlier version of this function did) undercounts real visits
        # to almost nothing, since very few sampled frames catch the cat visible;
        # crediting the full entered->last-seen span is what actually lets a real
        # visit reach min_dwell_seconds.
        if current is not None and (t - current["last_inside_t"]) > gap_tolerance:
            close_current()

        if inside:
            if current is None:
                current = {
                    "entered_at": t,
                    "last_inside_t": t,
                    "brightness": [],
                    "best_conf": 0.0,
                    "best_frame": None,
                    "best_bbox": None,
                }
            current["last_inside_t"] = t
            current["brightness"].append(brightness)
            if cat_det.confidence > current["best_conf"]:
                current["best_conf"] = cat_det.confidence
                current["best_frame"] = frame
                current["best_bbox"] = cat_det.bbox

    close_current()

    if current_hour is not None:
        is_day = sun_is_up()
    else:
        is_day = (sum(day_votes) / len(day_votes)) >= 0.5 if day_votes else True

    # у будь-якому ІЧ-кадрі (день чи ніч) міряємо відносно фасаду: абсолютна яскравість кота плаває 140–160 і ловить «unknown»
    ir_day = bool(walls) and not is_day_time_color(spreads)
    front = statistics.median(fronts) if fronts else 0.0
    if ir_day:
        logging.getLogger(__name__).info(f"сірий денний ІЧ-режим: wall={statistics.median(walls):.0f}, фасад={front:.0f}")

    def label_color(brightness: float) -> str:
        color = classify_ir_day(brightness, front) if ir_day else classify_cat_color(brightness, is_day, thresholds)
        logging.getLogger(__name__).info(
            f"яскравість кота {brightness:.0f} (фасад {front:.0f}, відношення {brightness / max(front, 1.0):.2f}, ir_day={ir_day}) -> {color}"
        )
        return color

    if total_frames == 0:
        return [{"cat": "unknown", "is_day": is_day, "confidence": 0.0, "dwell_seconds": 0.0}]

    for ep in episodes:
        ep["dwell"] = ep["last_inside_t"] - ep["entered_at"]
        # Fraction of the episode's span actually spent visibly detected inside
        # the ROI. A real visit spends most of it hidden inside the enclosed
        # dome (observed density 0.35-0.57 across known real visits) -- a cat
        # that's just standing/rubbing right outside the box stays continuously
        # visible instead (observed 0.80 on a known false positive). Rejecting
        # high-density episodes here is what actually distinguishes "was inside
        # the box" from "spent a while near it", rather than dwell time alone.
        ep["visible_density"] = (
            len(ep["brightness"]) * sample_interval / ep["dwell"] if ep["dwell"] > 0 else 1.0
        )

    if config.get("drum_ref_update", True):
        for mode in refs_dirty:
            save_ref(mode, refs[mode])

    confirmed = [
        ep
        for ep in episodes
        if ep["dwell"] >= min_dwell and ep["visible_density"] <= max_visible_density
    ]
    if confirmed:
        results = []
        for ep in confirmed:
            brightness = statistics.median(ep["brightness"])
            label = bright_override(label_color(brightness), ep["best_frame"], ep.get("best_bbox"), brightness / max(front, 1.0))
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

    # «Кіт всередині» — без YOLO: контури вікна барабана наприкінці події не схожі на порожній референс режиму,
    # а бічна смуга фасаду не закрита (інакше кіт стоїть ззовні перед барабаном).
    last3 = drum_obs[-3:]
    inside_ok, algo_run = drum_inside_confirmed(last3, config)
    if inside_ok:
        if algo_run is not None:
            low_run = last3[::-1][:algo_run]   # хвіст кадрів, де litter_algo бачить кота всередині (від нового до старого)
        else:
            low_run = []
            for o in reversed(drum_obs):
                if o["ncc"] >= NCC_CAT:
                    break
                low_run.append(o)
        cat_seen = cat_frames > 0 or (
            drum_obs[0]["ncc"] >= NCC_CAT and any(o["ncc"] < NCC_CAT for o in drum_obs)
        )
        if algo_run is not None:
            confirmed = cat_seen or (len(last3) == drum_algo.TAIL and algo_run == len(last3))   # без YOLO — лише коли inside усі TAIL кадрів
        else:
            confirmed = cat_seen or (len(low_run) >= 3 and last3[-1]["mode"] in ("irday", "ir"))
        if confirmed:
            samples = episodes[-1]["brightness"] if episodes else all_brightness_samples
            color = label_color(statistics.median(samples)) if samples else "unknown"
            if color == "unknown":
                color = ratio_color([o["ratio"] for o in low_run])
            ep = episodes[-1] if episodes else None
            return [
                {
                    "cat": f"{color}_inside",
                    "color": color,
                    "is_day": is_day,
                    "confidence": ep["best_conf"] if ep else max_conf,
                    "dwell_seconds": last_t - low_run[-1]["t"],
                    "confirmed_visit": False,
                    "inside_now": True,
                    "best_frame": ep["best_frame"] if ep and ep["best_frame"] is not None else last_frame,
                }
            ]

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
    color = bright_override(label_color(brightness), best_cat_frame, best_cat_bbox, brightness / max(front, 1.0))
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
