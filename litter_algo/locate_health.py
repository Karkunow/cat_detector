"""Чи не зсунули камеру: якість зіставлення шаблону, яку міряє КОЖНЕ розпізнавання.

Навіщо. `Locator` на кожному кадрі рахує score — наскільки добре шаблон корпусу лотка ліг на кадр.
Коли камеру зсувають, score падає з 0.87–0.99 до 0.42–0.50 (виміряно 02.11.2026), слідом валяться
NCC клітинок корпусу, і кожен кадр стає `near`/`blocking`: чистка стоїть, а детектор шле хибні
«кіт усередині». Раніше це помічали лише люди. Тепер score з кожного розпізнавання стікає сюди,
а `litter-ref-auto --if-shifted` раз на 10 хв дивиться на підсумок і, якщо камеру справді зсунули,
сам знімає шаблон нового положення (з усіма своїми запобіжниками й перевіркою на розмітці).

Запис дешевий: у пам'яті кільце, на диск скидаємо не частіше ніж раз на FLUSH_EVERY секунд, атомарно.
Будь-який збій ковтаємо — ця телеметрія не має права зламати розпізнавання.
"""
import json
import os
import statistics
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(os.path.dirname(HERE), "data", "locate_health.json")

WINDOW_SEC = 15 * 60     # за який час дивимось на score
MIN_SAMPLES = 8          # менше — висновків не робимо
SCORE_LOW = 0.70         # медіана нижче — шаблон не підходить (добре = 0.87–0.99, зсунута камера = 0.42–0.50)
FLUSH_EVERY = 20.0       # с: не частіше писати на диск
KEEP = 400               # скільки останніх вимірів тримаємо

_buf: list = []          # [(ts, score, ref)]
_flushed = 0.0


def _load():
    try:
        with open(STATE, encoding="utf-8") as f:
            d = json.load(f)
        return [tuple(x) for x in d.get("samples", [])], d
    except Exception:
        return [], {}


def _save(samples, extra):
    d = dict(extra or {})
    d["samples"] = [list(x) for x in samples]
    d["updated_ts"] = time.time()
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(STATE), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(tmp, STATE)          # атомарно: читач не побачить недописаного файлу
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def record(score, ref=None, now=None):
    """Викликається з Locator на кожному розпізнаванні. Нічого не повертає й ніколи не кидає."""
    global _buf, _flushed
    try:
        if score is None:
            return
        now = time.time() if now is None else now
        _buf.append((round(float(now), 1), round(float(score), 3), ref or ""))
        if now - _flushed < FLUSH_EVERY:
            return
        _flushed = now
        old, extra = _load()
        samples = [s for s in old + _buf if now - s[0] <= WINDOW_SEC][-KEEP:]
        _buf = []
        _save(samples, extra)
    except Exception:
        pass


def status(now=None):
    """Підсумок за вікном: {n, median, shifted, worst_ref, age}. shifted=None — даних замало."""
    now = time.time() if now is None else now
    samples, _ = _load()
    fresh = [s for s in samples + _buf if now - s[0] <= WINDOW_SEC]
    if len(fresh) < MIN_SAMPLES:
        return {"n": len(fresh), "median": None, "shifted": None, "ref": None, "age": None}
    med = statistics.median(s[1] for s in fresh)
    refs = [s[2] for s in fresh if s[2]]
    return {"n": len(fresh), "median": round(med, 3), "shifted": med < SCORE_LOW,
            "ref": max(set(refs), key=refs.count) if refs else None,
            "age": round(now - max(s[0] for s in fresh), 1)}


HEAL_COOLDOWN = 30 * 60   # с: не частіше одного самовиправлення


def heal_allowed(now=None):
    """Чи можна зараз запускати самовиправлення (не частіше HEAL_COOLDOWN)."""
    now = time.time() if now is None else now
    _, extra = _load()
    return now - float(extra.get("last_heal_ts", 0) or 0) >= HEAL_COOLDOWN


def mark_healed(note="", now=None):
    now = time.time() if now is None else now
    samples, extra = _load()
    extra = dict(extra)
    extra["last_heal_ts"] = now
    extra["last_heal_note"] = note[:200]
    # після зміни шаблону старі низькі score вже не про поточний стан — вікно починаємо заново
    _save([], extra)
