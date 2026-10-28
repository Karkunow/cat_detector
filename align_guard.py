"""Сторож положення камери: де лоток у кадрі і чи йому можна вірити (TRACKED / UNCERTAIN / LOST).

Положення береться лише з жорстких частин корпусу (без вікна барабана — кіт усередині їх не міняє), незалежно від режиму день/ніч.
Нема сигналу — «невідомо», ніколи «порожньо». Стан пишемо в data/align_state.json (його читає ~/bin/petbox-safe-clean)."""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
STATE_PATH = DATA_DIR / "align_state.json"
TEMPLATE_PATH = DATA_DIR / "align_template.jpg"

TRACKED, UNCERTAIN, LOST = "TRACKED", "UNCERTAIN", "LOST"

# Жорсткі частини корпусу в частках кадру (шаблон знятий з порожнього кадру): верхня дуга, права панель, ліве ребро, низ із наліпкою.
PARTS = {
    "top": (0.26, 0.12, 0.60, 0.22),
    "right": (0.61, 0.38, 0.75, 0.86),
    "left": (0.245, 0.22, 0.34, 0.84),
    "bottom": (0.34, 0.70, 0.60, 0.86),
}
SEARCH_PX = 400        # макс. зсув від положення шаблону (як ALIGN_MAX_SHIFT у drum_check)
AGREE_PX = 6.0         # частини «згодні», якщо їхні зсуви в межах стільки px
BODY_BOX = (0.245, 0.12, 0.75, 0.86)   # габарит корпусу в частках шаблону; після зсуву має лишатись у кадрі з полем MARGIN
MARGIN = 0.02

# Пороги підібрані за даними (див. звіт ~/cat_detector_autorecal_report_2026-10-26.md; `align_calibrate.py`).
CFG = {
    "s_hi": 0.60,
    "s_floor": 0.20,
    "min_parts": 2,
}


def edge_map(frame):
    """Нормована карта модуля градієнта половинного розміру (контури; не залежить від яскравості, ІЧ/день схожі)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), (5, 5), 0)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / max(float(m.mean()), 1e-3)


def _crop(m, box):
    h, w = m.shape
    return int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)


class Template:
    """Шаблони частин корпусу з еталонного кадру."""

    def __init__(self, frame):
        m = edge_map(frame)
        self.shape = m.shape
        self.parts = {}
        for name, box in PARTS.items():
            x1, y1, x2, y2 = _crop(m, box)
            self.parts[name] = (x1, y1, m[y1:y2, x1:x2].copy())


def measure(frame, tpl: Template) -> dict:
    """Положення кожної частини і зведена поза. pose = (dx, dy) у px повного кадру відносно шаблону (+ = корпус правіше/нижче)."""
    m = edge_map(frame)
    h, w = m.shape
    sr = SEARCH_PX // 2
    found = {}
    for name, (x1, y1, t) in tpl.parts.items():
        th, tw = t.shape
        X1, Y1 = max(0, x1 - sr), max(0, y1 - sr)
        X2, Y2 = min(w, x1 + tw + sr), min(h, y1 + th + sr)
        if X2 - X1 <= tw or Y2 - Y1 <= th:
            continue
        r = cv2.matchTemplate(m[Y1:Y2, X1:X2], t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        found[name] = (float((X1 + loc[0] - x1) * 2), float((Y1 + loc[1] - y1) * 2), float(mx))
    res = {"parts": found, "score": 0.0, "pose": None, "parts_ok": 0}
    if not found:
        return res
    # найбільша група частин, що згодні між собою
    best = []
    for a in found.values():
        grp = [b for b in found.values() if abs(b[0] - a[0]) <= AGREE_PX and abs(b[1] - a[1]) <= AGREE_PX]
        if len(grp) > len(best) or (len(grp) == len(best) and sum(g[2] for g in grp) > sum(g[2] for g in best)):
            best = grp
    res["parts_ok"] = len(best)
    res["pose"] = (float(np.median([g[0] for g in best])), float(np.median([g[1] for g in best])))
    # score = середнє двох найкращих частин усередині групи, але лише якщо група ≥ min_parts
    top = sorted((g[2] for g in best), reverse=True)[:2]
    res["score"] = float(np.mean(top)) if len(best) >= CFG["min_parts"] else float(max((g[2] for g in found.values()), default=0.0)) * 0.5
    return res


def body_in_frame(pose, shape) -> bool:
    h, w = shape[:2]
    x1, y1, x2, y2 = BODY_BOX
    dx, dy = pose
    return (x1 * w + dx >= MARGIN * w and x2 * w + dx <= (1 - MARGIN) * w
            and y1 * h + dy >= MARGIN * h and y2 * h + dy <= (1 - MARGIN) * h)


# ---------- автомат станів ----------
CFG.update({
    "interval_s": 60,       # як часто дивимось кадр
    "recover_s": 120,       # скільки безперервно «добрих» перевірок із сталою позою треба для TRACKED
    "recover_n": 3,
    "lost_after_s": 600,    # «не TRACKED без руху/кота» безперервно → LOST
    "floor_s": 180,         # score < s_floor без руху/кота → LOST
    "out_s": 180,           # корпус частково поза кадром / кадрів нема без руху/кота → LOST
    "jump_px": 12.0,        # стрибок пози у TRACKED більший за це = не довіряємо, доки не підтвердиться
    "stale_s": 180,         # читач (petbox-safe-clean) вважає стан застарілим
    "ref_tol_px": 12.0,     # поза далі за стільки від положення, де зняті еталони → еталони не відповідають, чистка блокується
    "notice_after_s": 1800, # не TRACKED так довго (будь-яка причина) → повідомлення Славі
    "notice_min_hours": 6.0,
    "notice_max_per_day": 3,
    "night_from": 23, "night_to": 8,
})


def _good_group(poses, tol=AGREE_PX):
    """Найбільша група поз, що згодні між собою (px)."""
    best = []
    for a in poses:
        grp = [b for b in poses if abs(b[0] - a[0]) <= tol and abs(b[1] - a[1]) <= tol]
        if len(grp) > len(best):
            best = grp
    return best


def make_check(measures: list[dict], shape, busy: bool, ts: float, cfg=CFG) -> dict:
    """Зводить кілька кадрів одної перевірки: «добра» = ≥2 кадри добрі й їхні пози згодні."""
    goods = [m for m in measures if m["pose"] and m["parts_ok"] >= cfg["min_parts"] and m["score"] >= cfg["s_hi"]
             and body_in_frame(m["pose"], shape)]
    grp = _good_group([m["pose"] for m in goods])
    good = len(grp) >= min(2, len(measures)) and len(measures) > 0
    best = max(measures, key=lambda m: m["score"], default=None)
    pose = None
    if good:
        pose = (float(np.median([p[0] for p in grp])), float(np.median([p[1] for p in grp])))
    elif best and best["pose"]:
        pose = best["pose"]
    return {
        "ts": ts, "busy": bool(busy), "n": len(measures), "good": good,
        "score": float(max((m["score"] for m in measures), default=0.0)),
        "parts_ok": int(max((m["parts_ok"] for m in measures), default=0)),
        "pose": pose,
        "inb": bool(pose) and body_in_frame(pose, shape),
    }


class Guard:
    """Стан положення. update(check) викликається раз на interval_s; час передається явно (тести)."""

    def __init__(self, cfg=None, now=0.0, prev: dict | None = None):
        self.cfg = dict(cfg or CFG)
        prev = prev or {}
        self.state, self.reason, self.since = UNCERTAIN, "start", now
        self.pose = None
        self.ref_poses = load_ref_poses()   # поза корпусу в момент зйомки еталонів, по режимах (data/align_refs.json)
        self.mode = None                     # режим (ir/irday/color) за останньою перевіркою
        self.recent: list[dict] = []         # останні перевірки (для умов тихого вікна кандидатів)
        self.score = 0.0
        self.parts_ok = 0
        self.bad_quiet = self.floor_quiet = self.out_quiet = 0.0
        self.streak: list[dict] = []
        self.last_ts = None
        self.bad_since = now          # коли почався безперервний «не TRACKED»
        self.notices = list(prev.get("notices", []))
        self.last_notice_reason = prev.get("last_notice_reason")
        self.last_notice_ts = prev.get("last_notice_ts", 0.0)
        self.was_lost_notified = bool(prev.get("lost_notified"))
        self.enhanced_until = float(prev.get("enhanced_until", 0.0))   # «посилений» режим чистки після промоції еталонів
        self.last_promote = dict(prev.get("last_promote", {}))        # {mode: ts} — не частіше за раз на добу на режим

    def _set(self, state, reason, ts):
        if state != self.state:
            self.state, self.since = state, ts
            if state == TRACKED:
                self.bad_since = None
            elif self.bad_since is None:
                self.bad_since = ts
        self.reason = reason

    def update(self, c: dict) -> str:
        cfg, ts = self.cfg, c["ts"]
        dt = min(ts - self.last_ts, 2 * cfg["interval_s"]) if self.last_ts is not None else 0.0
        self.last_ts = ts
        self.recent = (self.recent + [c])[-8:]
        self.mode = c.get("mode") or self.mode
        self.score, self.parts_ok = c["score"], c["parts_ok"]
        if c["good"]:
            self.bad_quiet = self.floor_quiet = self.out_quiet = 0.0
            if self.streak and (abs(self.streak[0]["pose"][0] - c["pose"][0]) > AGREE_PX or abs(self.streak[0]["pose"][1] - c["pose"][1]) > AGREE_PX):
                self.streak = []
            self.streak.append(c)
            span = self.streak[-1]["ts"] - self.streak[0]["ts"]
            if len(self.streak) >= cfg["recover_n"] and span >= cfg["recover_s"] - 5:
                self.pose = (float(np.median([s["pose"][0] for s in self.streak])), float(np.median([s["pose"][1] for s in self.streak])))
                self._set(TRACKED, "ok", ts)
            elif self.state == TRACKED:
                if self.pose and max(abs(c["pose"][0] - self.pose[0]), abs(c["pose"][1] - self.pose[1])) > cfg["jump_px"]:
                    self._set(UNCERTAIN, "стрибок положення, чекаємо підтвердження", ts)
            return self.state
        # недобра перевірка
        self.streak = []
        quiet = not c["busy"]
        if quiet:
            self.bad_quiet += dt
            self.floor_quiet = self.floor_quiet + dt if c["score"] < cfg["s_floor"] else 0.0
            self.out_quiet = self.out_quiet + dt if (c["n"] == 0 or not c["inb"]) else 0.0
        why = ("кадрів немає" if c["n"] == 0 else "корпус частково поза кадром" if not c["inb"]
               else f"впевненість {c['score']:.2f} < {cfg['s_hi']}")
        if c["busy"]:
            why += " (рух/кіт у кадрі)"
        if self.bad_quiet >= cfg["lost_after_s"] or self.floor_quiet >= cfg["floor_s"] or self.out_quiet >= cfg["out_s"]:
            self._set(LOST, why, ts)
        elif self.state == LOST:
            self.reason = why
        else:
            self._set(UNCERTAIN, why, ts)
        return self.state

    def refs_match(self) -> bool:
        """Еталони поточного режиму зняті там, де лоток зараз (поза в межах ref_tol_px). Режим чи еталона невідомо — False."""
        rp = self.ref_poses.get(self.mode) if self.mode else None
        if self.state != TRACKED or not self.pose or not rp:
            return False
        return max(abs(self.pose[0] - rp[0]), abs(self.pose[1] - rp[1])) <= self.cfg["ref_tol_px"]

    def notice_due(self, now_ts: float, hour: int) -> str | None:
        """Причина повідомлення (текст) або None. Антиспам: ≥notice_min_hours між повідомленнями, ≤notice_max_per_day на добу,
        вночі (night_from..night_to) відкладаємо до ранку."""
        cfg = self.cfg
        if self.state == TRACKED or self.bad_since is None:
            return None
        # довгий UNCERTAIN: не TRACKED ≥ notice_after_s І щонайменше 5 хв із цього — тихо (без руху/кота): кіт, що спить перед лотком, не привід просити Славу
        long_bad = now_ts - self.bad_since >= cfg["notice_after_s"] and self.bad_quiet >= 300
        if not (self.state == LOST or long_bad):
            return None
        if hour >= cfg["night_from"] or hour < cfg["night_to"]:
            return None
        self.notices = [t for t in self.notices if now_ts - t < 86400]
        if len(self.notices) >= cfg["notice_max_per_day"]:
            return None
        if now_ts - self.last_notice_ts < cfg["notice_min_hours"] * 3600:
            return None
        return self.reason

    def mark_notice(self, now_ts: float, reason: str) -> None:
        self.notices.append(now_ts)
        self.last_notice_ts, self.last_notice_reason = now_ts, reason

    def to_dict(self, now_ts: float, extra: dict | None = None) -> dict:
        d = {
            "state": self.state, "reason": self.reason, "since_ts": self.since,
            "pose": self.pose, "mode": self.mode, "ref_poses": self.ref_poses, "refs_match": self.refs_match(),
            "score": round(self.score, 3), "parts_ok": self.parts_ok, "updated_ts": now_ts,
            "last_notice_ts": self.last_notice_ts, "last_notice_reason": self.last_notice_reason,
            "notices": self.notices, "notices_today": len([t for t in self.notices if now_ts - t < 86400]),
            "lost_notified": self.was_lost_notified, "enhanced_until": self.enhanced_until, "last_promote": self.last_promote, "guard": "on",
        }
        d.update(extra or {})
        return d


REFS_PATH = DATA_DIR / "align_refs.json"


def load_ref_poses(path: Path = REFS_PATH) -> dict:
    try:
        return {k: tuple(v) for k, v in json.loads(path.read_text()).items()}
    except Exception:
        return {}


def write_state(d: dict, path: Path = STATE_PATH) -> None:
    """Атомарно: .tmp → os.replace (читач ~/bin/petbox-safe-clean не побачить половину файлу)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False))
    os.replace(tmp, path)


def read_state(path: Path = STATE_PATH) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def blend_allowed(now=None, path: Path = STATE_PATH) -> bool:
    """Живе оновлення еталона blend_ref дозволене лише в TRACKED зі свіжим станом, збігом пози з еталонами й поза «посиленим» режимом."""
    now = time.time() if now is None else now
    st = read_state(path)
    return bool(st) and st.get("state") == TRACKED and bool(st.get("refs_match")) \
        and now - st.get("updated_ts", 0) <= CFG["stale_s"] and now >= st.get("enhanced_until", 0)


def write_notice(text: str, out_dir: Path, prefix: str = "cat-detector-camera") -> Path:
    """Один файл = одне повідомлення, формат як у sd_analysis/daily: <unix>-<prefix>-01.txt, запис через .part → rename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{int(time.time())}-{prefix}-01.txt"
    tmp = out_dir / (name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out_dir / name)
    return out_dir / name


def notice_text(state: dict, since_ts: float) -> str:
    t = time.strftime("%H:%M", time.localtime(since_ts))
    return (
        f"Камера лотка: детектор не впевнений, де лоток у кадрі (з {t}): {state['reason']}.\n"
        "Чистку заблоковано, доки це не минеться.\n"
        "Будь ласка, наведіть камеру так, щоб увесь лоток був у кадрі (як на останньому знімку детектора). "
        "Після цього детектор перевірить сам, нічого натискати не треба."
    )


def check_frames(frames, tpl: Template, busy: bool, ts: float, cfg=CFG) -> dict:
    """Перевірка за кількома свіжими кадрами; frames порожні = «кадрів нема»."""
    import drum_check as dc
    ms = [measure(f, tpl) for f in frames]
    c = make_check(ms, frames[0].shape if frames else (720, 1280, 3), busy, ts, cfg)
    if frames:
        c["mode"] = dc.frame_mode_shifted(frames[-1], c["pose"] or (0.0, 0.0))
        if c["mode"] is None:
            c["mode"] = dc.frame_mode(frames[-1])
    return c


# ---------- кандидати еталонів і консенсус (рівень 2) ----------
CAND_CFG = {
    "min_cands": 3,
    "min_gap_s": 7200,        # кандидати з РІЗНИХ тихих періодів: проміжок ≥ 2 год
    "pair_ncc": 0.85,
    "pair_ratio": 0.10,
    "old_match_ncc": 0.45,    # кандидат має збігатися зі старим еталоном режиму (після вирівнювання за позою)
    "quiet_since_motion_s": 1800,
    "quiet_since_event_s": 1800,
    "quiet_checks": 5,        # ≥5 перевірок підряд (≈5 хв) без кота, поза стабільна
    "pose_std_px": 3.0,
    "cand_max_age_s": 3 * 86400,
    "cand_min_interval_s": 1800,
    "enhanced_s": 86400,
}


def quiet_window(guard: Guard, now: float, capturing: bool, processing: bool, motion_active: bool, last_motion_ts: float,
                 last_event_ts: float | None, inside_unresolved: bool, cc=CAND_CFG) -> str | None:
    """None, якщо можна знімати кандидата; інакше причина відмови. Усі умови з плану, жодної не послаблено."""
    if guard.state != TRACKED:
        return "не TRACKED"
    if capturing or processing:
        return "іде подія/обробка"
    if motion_active or now - last_motion_ts < cc["quiet_since_motion_s"]:
        return "був рух менш ніж 30 хв тому"
    if last_event_ts is not None and now - last_event_ts < cc["quiet_since_event_s"]:
        return "була подія менш ніж 30 хв тому"
    if inside_unresolved:
        return "кіт зайшов у лоток і вихід не зафіксовано (остання подія за 6 год — «усередині»)"
    r = guard.recent[-cc["quiet_checks"]:]
    if len(r) < cc["quiet_checks"] or not all(c["good"] and not c["busy"] for c in r):
        return "нема 5 тихих добрих перевірок підряд"
    xs, ys = [c["pose"][0] for c in r], [c["pose"][1] for c in r]
    if float(np.std(xs)) > cc["pose_std_px"] or float(np.std(ys)) > cc["pose_std_px"]:
        return "поза нестабільна"
    return None


def need_candidate(guard: Guard) -> str | None:
    """Режим, якому потрібна перезйомка еталонів, або None: еталона режиму нема чи він знятий не там, де лоток зараз."""
    m = guard.mode
    if guard.state != TRACKED or not m or m == "color":
        return None
    if m not in guard.ref_poses or not guard.refs_match():
        return m
    return None


def capture_candidate(frame, pose, mode: str, ts: float, data_dir: Path = DATA_DIR) -> Path:
    """Кандидат пишеться ЛИШЕ в refs_cand/<ts>_<mode>/ — робочі еталони не чіпає."""
    import drum_check as dc
    ref = dc.frame_ref(frame)
    d = data_dir / "refs_cand" / f"{int(ts)}_{mode}"
    d.mkdir(parents=True, exist_ok=True)
    for k in ("grad", "side", "ratio"):
        np.save(d / f"{k}.npy", ref[k])
    cv2.imwrite(str(d / "frame.jpg"), frame)
    (d / "meta.json").write_text(json.dumps({"ts": ts, "mode": mode, "pose": list(pose)}))
    return d


def _load_cand(d: Path) -> dict | None:
    try:
        m = json.loads((d / "meta.json").read_text())
        m.update(dir=d, grad=np.load(d / "grad.npy"), side=np.load(d / "side.npy"), ratio=float(np.load(d / "ratio.npy")))
        return m
    except Exception:
        return None


def list_candidates(mode: str, now: float, data_dir: Path = DATA_DIR, cc=CAND_CFG) -> list[dict]:
    out = []
    for d in sorted((data_dir / "refs_cand").glob(f"*_{mode}")):
        c = _load_cand(d)
        if c and now - c["ts"] <= cc["cand_max_age_s"]:
            out.append(c)
    return sorted(out, key=lambda c: c["ts"])


def consensus(cands: list[dict], cc=CAND_CFG) -> list[dict] | None:
    """≥min_cands кандидатів із різних тихих періодів (проміжок ≥min_gap_s), однакова поза (≤6 px), попарно NCC ≥ pair_ncc, ratio ±pair_ratio."""
    import drum_check as dc
    best: list[dict] = []
    for start in range(len(cands)):
        pick = [cands[start]]
        for c in cands[start + 1:]:
            if c["ts"] - pick[-1]["ts"] < cc["min_gap_s"]:
                continue
            if all(abs(c["pose"][0] - p["pose"][0]) <= AGREE_PX and abs(c["pose"][1] - p["pose"][1]) <= AGREE_PX
                   and dc.ncc(c["grad"], p["grad"]) >= cc["pair_ncc"] and abs(c["ratio"] - p["ratio"]) <= cc["pair_ratio"] for p in pick):
                pick.append(c)
        if len(pick) > len(best):
            best = pick
    return best if len(best) >= cc["min_cands"] else None


def matches_old(cand: dict, mode: str, ref_poses: dict, data_dir: Path = DATA_DIR, cc=CAND_CFG) -> tuple[bool | None, float | None]:
    """Збіг кандидата зі старим еталоном режиму після зсуву за позою. (None, None) — старого еталона нема (автопромоції немає)."""
    import drum_check as dc
    try:
        old = np.load(data_dir / f"drum_ref_{mode}.npy")
    except OSError:
        return None, None
    rp = ref_poses.get(mode)
    if not rp:
        return None, None
    frame = cv2.imread(str(cand["dir"] / "frame.jpg"))
    dx, dy = cand["pose"][0] - rp[0], cand["pose"][1] - rp[1]
    h, w = frame.shape[:2]
    warped = cv2.warpAffine(frame, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    v = dc.ncc(old, dc.grad_map(warped))
    return v >= cc["old_match_ncc"], float(v)


def _atomic_save_npy(path: Path, arr) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def promote(group: list[dict], mode: str, now: float, data_dir: Path = DATA_DIR, cc=CAND_CFG) -> dict:
    """Робочі еталони режиму = середнє консенсусної групи. Стара версія → refs_ver/<мітка>/prev (відкат: скопіювати назад),
    нова → refs_ver/<мітка>/new. Якір вирівнювання (drum_base_ir.jpg + drum_ref_ir_side.npy) беремо з останнього кандидата, як при ручній перезйомці.
    Викликати лише між подіями (не capturing, не під час обробки)."""
    import hashlib
    label = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"_{mode}"
    ver = data_dir / "refs_ver" / label
    (ver / "prev").mkdir(parents=True, exist_ok=True)
    (ver / "new").mkdir(parents=True, exist_ok=True)
    last = group[-1]
    grad = np.mean([c["grad"] for c in group], axis=0).astype(np.float32)
    side = np.mean([c["side"] for c in group], axis=0).astype(np.float32)
    ratio = np.float32(np.mean([c["ratio"] for c in group]))
    last_jpg = (last["dir"] / "frame.jpg").read_bytes()
    modes = [mode] + (["irday2"] if mode == "irday" else [])
    new_files: dict[str, bytes | np.ndarray] = {}
    for m in modes:
        new_files[f"drum_ref_{m}.npy"], new_files[f"drum_ref_{m}_side.npy"], new_files[f"drum_ref_{m}_ratio.npy"] = grad, side, ratio
        new_files[f"drum_base_{m}.jpg"] = last_jpg
    new_files["drum_base_ir.jpg"] = last_jpg
    if mode != "ir":
        new_files["drum_ref_ir_side.npy"] = side
    sums = []
    for name, val in new_files.items():
        tgt = data_dir / name
        if tgt.exists():
            (ver / "prev" / name).write_bytes(tgt.read_bytes())
        if name.endswith(".npy"):
            np.save(ver / "new" / name, val)
        else:
            (ver / "new" / name).write_bytes(val)
        sums.append(f"{hashlib.sha256((ver / 'new' / name).read_bytes()).hexdigest()}  new/{name}")
    (ver / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    for name, val in new_files.items():
        tgt = data_dir / name
        _atomic_write(tgt, (ver / "new" / name).read_bytes())
    poses = load_ref_poses(data_dir / "align_refs.json")
    for m in modes:
        poses[m] = tuple(last["pose"])
    (data_dir / "align_refs.json").write_text(json.dumps({k: list(v) for k, v in poses.items()}))
    for c in group:
        c_dir = c["dir"]
        done = data_dir / "refs_cand_used" / c_dir.name
        done.parent.mkdir(exist_ok=True)
        os.replace(c_dir, done)
    return {"label": label, "dir": str(ver), "enhanced_until": now + cc["enhanced_s"], "pose": list(last["pose"])}
