"""Тіньовий режим litter_algo: лише ПИШЕ вердикт у журнал поряд із рішенням чинної логіки.

Гарантії безпеки (кіт у лотку):
- нічого не повертає в чистку: `log_shadow()` завжди повертає None, CLI завжди виходить з кодом 0 і нічого не друкує
  (крім --summary), будь-який збій (немає cv2, битий кадр, повний диск, зависання) ковтається й лише потрапляє в журнал;
- жорсткий ліміт часу (SIGALRM, за замовчуванням 15 с), щоб зависання не затримало виклик;
- не імпортує й не змінює живу службу (main.py, drum_check.py, petbox-safe-clean); викликається з боку чинної логіки
  ПІСЛЯ її рішення, у фоні або з `|| true`, і рішення не читає назад.

Запис JSONL (один рядок на перевірку): ts, source, frame, current (рішення чинної логіки: "allow"/"block"/None
+ необов'язковий текст причини), algo (clear/near/inside/blocking/unknown/error), algo_allows (algo == clear),
window, f, g, score, scale, cells, rim, elapsed, agreement:
  agree         чинна й algo згодні;
  algo_stricter чинна дозволила чистку, а algo блокує (саме ці випадки і є «врятований кіт» або хибна тривога algo);
  algo_looser   чинна заблокувала, а algo дозволила (алго не можна пускати замість чинної, поки таких нема/пояснені);
  no_current    чинне рішення не передали.

Виклик з чинної логіки (приклад; ТІЛЬКИ після рішення, відповідь не використовується):
    timeout 20 python3 -m litter_algo.shadow --frame /tmp/frame.jpg --current allow --source petbox-safe-clean >/dev/null 2>&1 || true
Звіт: python3 -m litter_algo.shadow --summary [шлях до журналу]
"""
import argparse
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.environ.get("LITTER_SHADOW_LOG") or os.path.join(os.path.dirname(HERE), "data", "litter_shadow.jsonl")
TIME_LIMIT = 15
_ALLOW = ("allow", "allowed", "clear", "empty", "ok", "yes", "true", "1")
_BLOCK = ("block", "blocked", "deny", "refuse", "no", "false", "0")


def _norm_current(current):
    """'allow'/'block'/None з рядка чи bool (будь-що незрозуміле → None, а не вигадане рішення)."""
    if isinstance(current, bool):
        return "allow" if current else "block"
    if current is None:
        return None
    s = str(current).strip().lower()
    return "allow" if s in _ALLOW else "block" if s in _BLOCK else None


def agreement(current, algo_allows):
    cur = _norm_current(current)
    if cur is None or algo_allows is None:
        return "no_current"
    if (cur == "allow") == bool(algo_allows):
        return "agree"
    return "algo_stricter" if cur == "allow" else "algo_looser"


def _analyze_default(frame_path):
    import cv2   # імпорт тут: без cv2 (чи на Mac) модуль усе одно імпортується
    from litter_algo.classify import decide, verdict
    from litter_algo.locate import Locator
    global _LOC
    if _LOC is None:
        _LOC = Locator()
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError("кадр не читається")
    a = _LOC.analyze(frame)
    wv, f, g = verdict(a)
    return {"algo": decide(wv, a["cells"], a["rim"]), "window": wv, "f": f, "g": g, "score": a["score"],
            "scale": a["scale"], "cells": a["cells"], "rim": a["rim"]}


_LOC = None


def _round(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, (list, tuple)):
        return [_round(x) for x in o]
    return o


class _Timeout(Exception):
    pass


def _on_alarm(signum, frm):
    raise _Timeout()


def make_record(frame_path, current=None, reason=None, source="", analyze=None, limit=TIME_LIMIT):
    """Будує запис журналу. Не кидає виняток: збій → algo='error' + текст у поле error."""
    t0 = time.time()
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source": source, "frame": frame_path,
           "current": {"decision": _norm_current(current), "raw": None if current is None else str(current), "reason": reason}}
    old = None
    try:
        try:
            old = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(int(limit))
        except (ValueError, OSError, AttributeError):
            old = None   # не головний потік / нема SIGALRM: працюємо без ліміту
        res = (analyze or _analyze_default)(frame_path)
        rec.update({k: _round(res.get(k)) for k in ("window", "f", "g", "score", "scale", "cells", "rim")})
        rec["algo"] = res.get("algo", "error")
    except _Timeout:
        rec["algo"], rec["error"] = "error", "timeout %ss" % limit
    except BaseException as e:   # навмисно широко: тінь не має права впливати на виклик
        rec["algo"], rec["error"] = "error", "%s: %s" % (type(e).__name__, str(e)[:200])
    finally:
        try:
            signal.alarm(0)
            if old is not None:
                signal.signal(signal.SIGALRM, old)
        except Exception:
            pass
    rec["algo_allows"] = rec["algo"] == "clear" if rec["algo"] != "error" else None
    rec["agreement"] = agreement(current, rec["algo_allows"])
    rec["elapsed"] = round(time.time() - t0, 3)
    return rec


def write_record(rec, log_path=None):
    """Дописує один рядок JSONL. Не кидає виняток."""
    try:
        path = log_path or DEFAULT_LOG
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))   # один write з O_APPEND: рядки різних процесів не змішуються
        finally:
            os.close(fd)
    except BaseException:
        pass


def log_shadow(frame_path, current=None, reason=None, source="", log_path=None, analyze=None, limit=TIME_LIMIT):
    """Головна точка входу для інтеграції: порахувати вердикт і записати. ЗАВЖДИ повертає None."""
    try:
        write_record(make_record(frame_path, current, reason, source, analyze, limit), log_path)
    except BaseException:
        pass
    return None


def summarize(log_path=None):
    """Зведення журналу: скільки перевірок, згода/розбіжності, перелік algo_stricter і algo_looser."""
    path = log_path or DEFAULT_LOG
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
    by = {}
    for r in rows:
        by[r.get("agreement")] = by.get(r.get("agreement"), 0) + 1
    out = ["записів: %d" % len(rows), "algo: %s" % json.dumps(_count(rows, "algo"), ensure_ascii=False),
           "згода: %s" % json.dumps(by, ensure_ascii=False)]
    for kind, title in (("algo_stricter", "чинна дозволила, algo блокує (перевірити очима: кіт чи хибна тривога)"),
                        ("algo_looser", "чинна заблокувала, algo дозволила (algo не можна довіряти без пояснення)")):
        hits = [r for r in rows if r.get("agreement") == kind]
        if hits:
            out.append("%s: %d" % (title, len(hits)))
            out += ["  %s %s algo=%s cur=%s" % (r.get("ts"), r.get("frame"), r.get("algo"), (r.get("current") or {}).get("raw")) for r in hits[-20:]]
    err = [r for r in rows if r.get("algo") == "error"]
    if err:
        out.append("збоїв algo: %d, останній: %s" % (len(err), err[-1].get("error")))
    return "\n".join(out)


def _count(rows, key):
    c = {}
    for r in rows:
        c[r.get(key)] = c.get(r.get(key), 0) + 1
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame")
    ap.add_argument("--current", help="рішення чинної логіки: allow/block")
    ap.add_argument("--reason", help="причина чинної логіки (текст)")
    ap.add_argument("--source", default="")
    ap.add_argument("--log", default=None)
    ap.add_argument("--summary", nargs="?", const="", default=None, metavar="LOG")
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 0   # навіть невірні аргументи не мають давати ненульовий код
    try:
        if args.summary is not None:
            print(summarize(args.summary or args.log))
        elif args.frame:
            sys.path.insert(0, os.path.dirname(HERE))
            log_shadow(args.frame, args.current, args.reason, args.source, args.log)
    except BaseException as e:
        if args.summary is not None:
            print("помилка звіту: %s" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
