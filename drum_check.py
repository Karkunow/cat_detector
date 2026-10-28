import json
import os
from pathlib import Path

import cv2
import numpy as np

CALIB = Path.home() / ".config" / "petbox-safe-clean.json"
ROI = (0.24, 0.22, 0.39, 0.50)
DEVIATION = 0.20
IRDAY_EMPTY = (50, 92)  # межі «порожньо» в сірому денному режимі, поставив Слава 26.10 (порожній о 15:00 26.10 = 80)
NCC_CAT = 0.55


def frame_mode(frame) -> str:
    f = frame.astype(np.int16)
    spread = float(np.mean(np.abs(f[:, :, 0] - f[:, :, 1]) + np.abs(f[:, :, 1] - f[:, :, 2])))
    if spread >= 4.0:
        return "color"
    h, w = frame.shape[:2]
    door = float(np.mean(cv2.cvtColor(frame[int(.083 * h):int(.61 * h), int(.828 * w):int(.992 * w)], cv2.COLOR_BGR2GRAY)))
    return "irday" if door >= 90 else "ir"


def roi_brightness(frame, mode: str) -> float:
    h, w = frame.shape[:2]
    if mode == "irday":
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        return 100 * g[int(.28 * h):int(.47 * h), int(.25 * w):int(.38 * w)].mean() / g[int(.69 * h):int(.78 * h), int(.18 * w):int(.47 * w)].mean()
    x1, y1, x2, y2 = int(ROI[0] * w), int(ROI[1] * h), int(ROI[2] * w), int(ROI[3] * h)
    return float(np.mean(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)))


def drum_differs_from_empty(frame) -> bool | None:
    """True — вікно барабана не схоже на порожнє; False — порожнє; None — еталона немає, перевірку пропускаємо."""
    mode = frame_mode(frame)
    b = roi_brightness(frame, mode)
    if mode == "irday":
        return b < IRDAY_EMPTY[0] or b > IRDAY_EMPTY[1]
    try:
        m = json.load(open(CALIB)).get(mode, {})
    except Exception:
        return None
    empties, occ = m.get("empty", []), m.get("occupied", [])
    if not empties:
        return None
    ref = min(empties, key=lambda e: abs(b - e))
    if occ and min(abs(b - o) for o in occ) < abs(b - ref):
        return True
    return abs(b - ref) > DEVIATION * ref


def grad_map(frame):
    """Карта модуля градієнта (контури) вікна барабана — не залежить від яскравості."""
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = g[int(.22 * h):int(.50 * h), int(.24 * w):int(.39 * w)]
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def ncc(a, b) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def drum_differs_from_baseline(base_frames, last_frame) -> bool | None:
    """True — контури вікна не схожі на базу (кіт); False — схожі; None — бази нема (irday)."""
    mode = frame_mode(last_frame)
    bases = [f for f in base_frames if frame_mode(f) == mode]
    if not bases:
        return drum_differs_from_empty(last_frame) if mode != "irday" else None
    base = np.median([grad_map(f) for f in bases], axis=0)
    return ncc(base, grad_map(last_frame)) < NCC_CAT


def drum_not_empty(base_frames, frames) -> bool:
    """Голосування за останніми кадрами без руху: «не порожній», якщо ≥2 з 3 (за наявними, якщо їх менше) True; бази нема — не заперечуємо."""
    votes = [drum_differs_from_baseline(base_frames, f) for f in frames]
    if all(v is None for v in votes):
        return True
    return sum(v is True for v in votes) >= min(2, len(votes))


DATA_DIR = Path(__file__).parent / "data"
SIDE_ROI = (0.06, 0.20, 0.23, 0.66)  # смуга фасаду ліворуч від барабана: кіт ззовні її закриває, кіт усередині — ні
REF_MODES = ("ir", "irday", "irday2", "color")   # irday2 — запасний денний еталон (інше освітлення), береться кращий за NCC
REF_PARTS = (("grad", ""), ("side", "_side"), ("ratio", "_ratio"))
REF_UPDATE_NCC = 0.8
REF_ALPHA = 0.02
SIDE_CAT = 0.45
RIM_CAT = 0.40            # NCC лівої/правої смуги рамки лотка нижче — рамку закрито (кіт стоїть перед лотком), «всередині» не кажемо
RIM_MODES = ("irday", "irday2")   # лише денні режими: на нічних кадрах метрику не перевіряли
RIM_SEGS = {"L": (.22, .12, .30, .80), "R": (.71, .12, .79, .80)}
WHITE_RATIO, BLACK_RATIO = 1.3, 0.75


def side_map(frame):
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g[int(SIDE_ROI[1] * h):int(SIDE_ROI[3] * h), int(SIDE_ROI[0] * w):int(SIDE_ROI[2] * w)], (5, 5), 0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def drum_ratio(frame) -> float:
    """Середня яскравість вікна барабана / яскравість фасаду (не залежить від загального світла)."""
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    drum = g[int(.22 * h):int(.50 * h), int(.24 * w):int(.39 * w)].mean()
    return float(drum / max(g[int(.69 * h):int(.78 * h), int(.18 * w):int(.47 * w)].mean(), 1.0))


def _is_daylight(frame) -> bool:
    h = frame.shape[0]
    return float(np.mean(cv2.cvtColor(frame[:int(.25 * h)], cv2.COLOR_BGR2GRAY))) >= 140 and frame_mode(frame) == "irday"


def _rim_maps(frame) -> dict:
    """Карти контурів лівої/правої смуги зовнішньої рамки лотка. Кіт, що стоїть перед лотком, закриває їх; кіт усередині — ні."""
    h, w = frame.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return {k: m[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] for k, (x1, y1, x2, y2) in RIM_SEGS.items()}


def frame_ref(frame) -> dict:
    return {"grad": grad_map(frame), "side": side_map(frame), "ratio": np.float32(drum_ratio(frame))}


ALIGN_MAX_SHIFT = 400     # px: більший зсув вважаємо збоєм оцінки й кадр не вирівнюємо (NCC тоді впаде = відмова, безпечно)
ALIGN_MIN_RESPONSE = 0.05 # мінімальна «чіткість» піка фазової кореляції


def _align_map(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(cv2.resize(g, None, fx=0.5, fy=0.5), (5, 5), 0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def estimate_shift(frame, base_map, window) -> tuple[float, float, float]:
    """Зсув кадру відносно еталонного кадру (px повного кадру) за контурами всього фасаду: камеру можуть зачепити."""
    (dx, dy), resp = cv2.phaseCorrelate(base_map, _align_map(frame), window)
    return float(dx) * 2, float(dy) * 2, float(resp)


def _refine_shift(frame, dx, dy, side_ref) -> tuple[int, int]:
    """Дошукати залишок зсуву ±8 px за бічною смугою фасаду (стіна/край коробки, кіт у барабані її не міняє)."""
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x1, y1, x2, y2 = int(SIDE_ROI[0] * w), int(SIDE_ROI[1] * h), int(SIDE_ROI[2] * w), int(SIDE_ROI[3] * h)
    best, best_d = -2.0, (0, 0)

    def score(rx, ry):
        ox, oy = int(round(dx)) - rx, int(round(dy)) - ry
        if y1 + oy < 0 or x1 + ox < 0 or y2 + oy > h or x2 + ox > w:
            return -2.0
        c = cv2.GaussianBlur(g[y1 + oy:y2 + oy, x1 + ox:x2 + ox], (5, 5), 0)
        m = np.hypot(cv2.Sobel(c, cv2.CV_32F, 1, 0), cv2.Sobel(c, cv2.CV_32F, 0, 1))
        return ncc(side_ref, m)

    for ry in range(-8, 9, 2):
        for rx in range(-8, 9, 2):
            v = score(rx, ry)
            if v > best:
                best, best_d = v, (rx, ry)
    cx, cy = best_d
    for ry in range(cy - 1, cy + 2):
        for rx in range(cx - 1, cx + 2):
            v = score(rx, ry)
            if v > best:
                best, best_d = v, (rx, ry)
    return best_d


def align_with_shift(frame, base):
    """(кадр, зсунутий на місце еталонного, (dx, dy) зсуву камери) за base = {'map','window','side'}; зсув завеликий чи нечіткий — (кадр, (0, 0))."""
    dx, dy, resp = estimate_shift(frame, base["map"], base["window"])
    if resp < ALIGN_MIN_RESPONSE or abs(dx) > ALIGN_MAX_SHIFT or abs(dy) > ALIGN_MAX_SHIFT:
        return frame, (0.0, 0.0)
    rx, ry = _refine_shift(frame, dx, dy, base["side"])
    h, w = frame.shape[:2]
    out = cv2.warpAffine(frame, np.float32([[1, 0, -dx + rx], [0, 1, -dy + ry]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, (dx - rx, dy - ry)


def align_to_base(frame, base):
    return align_with_shift(frame, base)[0]


def frame_mode_shifted(frame, shift) -> str | None:
    """Як frame_mode, але вікно дверей беремо там, де воно опинилось після зсуву камери; вийшло за кадр — None."""
    f = frame.astype(np.int16)
    spread = float(np.mean(np.abs(f[:, :, 0] - f[:, :, 1]) + np.abs(f[:, :, 1] - f[:, :, 2])))
    if spread >= 4.0:
        return "color"
    h, w = frame.shape[:2]
    x1, x2 = int(.828 * w + shift[0]), int(.992 * w + shift[0])
    y1, y2 = int(.083 * h + shift[1]), int(.61 * h + shift[1])
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:   # двері після зсуву камери за кадром: день/ніч за верхньою смугою (вночі ≤120, вдень ≥157)
        top = float(np.mean(cv2.cvtColor(frame[:int(.25 * h)], cv2.COLOR_BGR2GRAY)))
        return "irday" if top >= 140 else "ir"
    door = float(np.mean(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)))
    return "irday" if door >= 90 else "ir"


def load_refs() -> dict:
    refs = {}
    for mode in REF_MODES:
        try:
            refs[mode] = {k: np.load(DATA_DIR / f"drum_ref_{mode}{s}.npy") for k, s in REF_PARTS}
            refs[mode]["_mtime"] = (DATA_DIR / f"drum_ref_{mode}.npy").stat().st_mtime_ns
        except OSError:
            continue
        base = cv2.imread(str(DATA_DIR / f"drum_base_{mode}.jpg"))
        if base is not None:
            m = _align_map(base)
            refs[mode]["rim"] = _rim_maps(base)
            refs[mode]["base"] = {"map": m, "window": cv2.createHanningWindow((m.shape[1], m.shape[0]), cv2.CV_32F), "side": refs[mode]["side"]}
    return refs


def save_ref(mode: str, ref: dict) -> None:
    """Атомарно (.tmp → os.replace: petbox-safe-clean читає ті самі файли). Файл змінили після завантаження (промоція
    еталонів сторожем положення) — не перезаписуємо, інакше загубимо чуже оновлення."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if ref.get("_mtime") and (DATA_DIR / f"drum_ref_{mode}.npy").stat().st_mtime_ns != ref["_mtime"]:
            return
    except OSError:
        pass
    for k, s in REF_PARTS:
        tmp = DATA_DIR / f"drum_ref_{mode}{s}.tmp.npy"
        np.save(tmp, ref[k])
        os.replace(tmp, DATA_DIR / f"drum_ref_{mode}{s}.npy")


def observe(frame, refs: dict) -> dict | None:
    """NCC контурів барабана і бічної смуги до референса режиму кадру; None — референса режиму нема."""
    return _observe(frame, refs)


def _observe(frame, refs: dict) -> dict | None:
    bases = [r["base"] for r in refs.values() if r.get("base")]
    shift = (0.0, 0.0)
    if bases:
        aligned, shift = align_with_shift(frame, bases[0])
    else:
        aligned = frame
    mode = frame_mode_shifted(frame, shift) if bases else frame_mode(frame)
    if mode is None:
        cands = [m for m in ("ir", "irday") if m in refs]
        results = [r for r in (_observe_mode(aligned, refs, m) for m in cands) if r]
        return max(results, key=lambda r: r["ncc"]) if results else None
    results = [r for r in (_observe_mode(aligned, refs, m) for m in (mode, mode + "2") if m in refs) if r]
    return max(results, key=lambda r: r["ncc"]) if results else None


def _observe_mode(frame, refs: dict, mode: str) -> dict | None:
    ref = refs.get(mode)
    if ref is None:
        return None
    cur = frame_ref(frame)
    rim = None
    if mode in RIM_MODES and ref.get("rim") and _is_daylight(frame):   # справді день: на нічних кадрах рамку з денним еталоном не порівнюємо
        cm = _rim_maps(align_with_shift(frame, ref["base"])[0] if ref.get("base") else frame)   # еталон рамки — із власної бази режиму
        rim = min(ncc(ref["rim"][k], cm[k]) for k in RIM_SEGS)
    return {
        "rim": rim,
        "mode": mode,
        "ncc": ncc(ref["grad"], cur["grad"]),
        "side": ncc(ref["side"], cur["side"]),
        "ratio": float(cur["ratio"] / ref["ratio"]),
        "cur": cur,
    }


def blend_ref(ref: dict, cur: dict) -> None:
    for k, _ in REF_PARTS:
        ref[k] = ((1 - REF_ALPHA) * ref[k] + REF_ALPHA * cur[k]).astype(np.float32)


def ratio_color(ratios: list[float]) -> str:
    r = float(np.median(ratios))
    return "white" if r >= WHITE_RATIO else "black" if r <= BLACK_RATIO else "unknown"
