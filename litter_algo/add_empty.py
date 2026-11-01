"""Додати в бібліотеку еталонів (empty_exemplars.npz) вміст ПОРОЖНЬОГО вікна барабана.

Навіщо. `f` — частка пікселів вікна, що відрізняються від найближчого еталона порожнього вікна. Якщо в бібліотеці
немає порожнього вікна для поточного світла й положення камери, f на порожньому лотку перевищує F_MAX, вікно
читається як «зайняте» і кожен кадр стає `inside`: чистка блокується, а детектор шле хибне «кіт усередині».
Саме це сталось 01.11.2026 вдень після переїзду камери: нічні еталони нового положення збіглись (f 0.067),
а денних не було (f 0.17–0.21 при порозі 0.10). `add_ref.py` додає лише шаблон ПОШУКУ лотка, вмісту вікна — ні.

Кадр має бути з ЗАВІДОМО порожнім барабаном (без кота, без людини поруч). Спершу завжди перевірка очима:
без --yes файл не пишеться, лише готується смужка кропів у --check.

Запобіжник. Новий еталон робить алгоритм ДОЗВІЛЬНІШИМ, тож перед записом кандидати проганяються на розмітці
Слави: жоден кадр із котом (мітки I/O/U) не має стати `clear`, а порожніх (E) має розпізнаватись не менше,
ніж без них. Не пройшло — нічого не пишеться. Перевірку можна вимкнути лише явним --skip-validate.

  python3 -m litter_algo.add_empty кадр.jpg [ще.jpg ...] --check /tmp/empty.png
  python3 -m litter_algo.add_empty кадр.jpg [ще.jpg ...] --yes
"""
import argparse
import datetime
import os
import shutil

import cv2
import numpy as np

from .classify import DIFF, F_MAX, G_MAX, MIN_LOCATE, compare, decide
from .locate import HERE, Locator

LIB = os.path.join(HERE, "empty_exemplars.npz")
HOME = os.path.expanduser("~")
LABELS = os.path.join(HOME, "litter_labels_2026-11-01", "labels.tsv")
ARCHIVE = os.path.join(HOME, "litter_archive_2026-10-28", "frames")
MIN_LOCATE_ADD = 0.40   # суворіше за MIN_LOCATE=0.25 у classify: еталон беремо лише з кадру, де лоток знайдено впевнено
VALIDATE_N = 120        # скільки архівних кадрів кожного роду брати на перевірку


def crop_of(frame, loc):
    """Нормований вміст вікна (48x64, медіана = 1) або (None, причина)."""
    a = loc.analyze(frame)
    if a["crop"] is None:
        return None, "лоток не знайдено", a
    if a["score"] < MIN_LOCATE_ADD:
        return None, "лоток знайдено невпевнено (score %.2f < %.2f)" % (a["score"], MIN_LOCATE_ADD), a
    c = a["crop"]
    return c / max(float(np.median(c)), 1e-3), None, a


def _library():
    d = np.load(LIB, allow_pickle=True)
    return d["ex"].astype(np.float32), [str(s) for s in d["src"]]


def verdict_with(ex, crop):
    """(f, g) кропа відносно бібліотеки ex."""
    best = (9.0, 9.0)
    for e in ex:
        df = np.abs(crop - e)
        f = float((df > DIFF).mean())
        if f < best[0]:
            best = (f, float(df.reshape(3, 16, 4, 16).mean(axis=(1, 3)).max()))
    return best


def labelled(limit):
    """{мітка: [шлях, ...]} з розмітки Слави."""
    out = {}
    if not os.path.exists(LABELS):
        return out
    for line in open(LABELS, encoding="utf-8").read().splitlines()[1:]:
        p = line.split("\t")
        if len(p) > 2:
            out.setdefault(p[2], []).append(os.path.join(ARCHIVE, p[1]))
    for k, v in out.items():
        step = max(len(v) // limit, 1)
        out[k] = v[::step][:limit]
    return out


def validate(ex_old, ex_new, loc, log=print):
    """Кандидати не мають дозволяти чистку там, де її не дозволяла бібліотека без них."""
    sets = labelled(VALIDATE_N)
    if not sets:
        log("  УВАГА: розмітки %s нема — перевірити кандидата нема на чому" % LABELS)
        return False
    cats = sets.get("I", []) + sets.get("O", []) + sets.get("U", [])
    bad = 0
    for p in cats:
        fr = cv2.imread(p)
        if fr is None:
            continue
        c, why, a = crop_of(fr, loc)
        if c is None:
            continue
        f, g = verdict_with(ex_new, c)
        wv = "empty" if f <= F_MAX and g <= G_MAX else "occupied"
        if decide(wv, a["cells"], a["rim"], a["crop"]) == "clear":
            bad += 1
            log("  УВАГА: з кандидатом кадр із котом став clear: %s (f=%.3f)" % (p, f))
    empties = sets.get("E", [])

    def n_clear(ex):
        n = 0
        for p in empties:
            fr = cv2.imread(p)
            if fr is None:
                continue
            c, why, a = crop_of(fr, loc)
            if c is None:
                continue
            f, g = verdict_with(ex, c)
            wv = "empty" if f <= F_MAX and g <= G_MAX else "occupied"
            n += decide(wv, a["cells"], a["rim"], a["crop"]) == "clear"
        return n

    before, after = n_clear(ex_old), n_clear(ex_new)
    log("  перевірка на розмітці: кадрів із котом дозволено %d (має бути 0); порожніх розпізнано %d → %d"
        % (bad, before, after))
    return bad == 0 and after >= before


def append(crops, names, log=print, do_validate=True):
    """Програмний шлях (litter-ref-auto): додати готові нормовані кропи в бібліотеку.
    Повертає True, якщо записано. Перевірку на розмітці вимикати лише там, де її вже зроблено."""
    loc = Locator()
    ex_old, src_old = _library()
    ex_new = np.concatenate([ex_old, np.array(list(crops), np.float32)])
    if do_validate and not validate(ex_old, ex_new, loc, log=log):
        log("  кандидати в еталони НЕ пройшли перевірку — бібліотеку не чіпаю")
        return False
    bak = LIB + ".bak-" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copy(LIB, bak)
    tmp = LIB + ".part.npz"
    np.savez_compressed(tmp, ex=ex_new, src=np.array(src_old + list(names)))
    os.replace(tmp, LIB)   # одним рухом: служба не має прочитати недописаний файл
    log("  додано еталонів порожнього вікна: %d → %d (бекап %s)" % (len(ex_old), len(ex_new), os.path.basename(bak)))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+", help="кадри ЗАВІДОМО порожнього лотка")
    ap.add_argument("--check", default="/tmp/add_empty_check.png", help="куди покласти смужку кропів для перевірки очима")
    ap.add_argument("--yes", action="store_true", help="записати в бібліотеку (без цього — лише картинка)")
    ap.add_argument("--skip-validate", action="store_true", help="не ганяти перевірку на розмітці (НЕ рекомендовано)")
    a = ap.parse_args(argv)

    loc = Locator()
    ex_old, src_old = _library()
    new, names, tiles = [], [], []
    for p in a.frames:
        fr = cv2.imread(p)
        if fr is None:
            raise SystemExit("кадр не читається: %s" % p)
        c, why, an = crop_of(fr, loc)
        if c is None:
            raise SystemExit("%s: %s" % (p, why))
        f, g = verdict_with(ex_old, c)
        cls = decide("empty" if f <= F_MAX and g <= G_MAX else "occupied", an["cells"], an["rim"], an["crop"])
        print("%s: score=%.2f f=%.3f g=%.3f клас зараз=%s" % (os.path.basename(p), an["score"], f, g, cls))
        new.append(c)
        names.append(os.path.abspath(p))
        t = cv2.resize(np.clip(c * 110, 0, 255).astype(np.uint8), (192, 144), interpolation=cv2.INTER_NEAREST)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        t = np.vstack([t, np.zeros((26, 192, 3), np.uint8)])
        cv2.putText(t, "f=%.3f %s" % (f, os.path.basename(os.path.dirname(p))[:13]), (3, 164), 0, 0.38, (0, 255, 255), 1)
        tiles.append(t)
    cv2.imwrite(a.check, np.hstack(tiles))
    print("смужка для перевірки очима:", a.check)

    if not a.yes:
        print("файл НЕ записано (додай --yes, коли на смужці всі вікна порожні):", LIB)
        return 0
    if not append(new, names, do_validate=not a.skip_validate):
        raise SystemExit("перевірка не пройдена — нічого не записано")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
