"""Додати в банк шаблон нового положення камери.

Кадр має бути ЗАВІДОМО порожній лоток (без кота, без людини поруч) у новому положенні камери.
Скрипт знаходить вікно барабана (або беремо його з --win), переводить кадр у систему координат ref_box.jpg
(зсув + масштаб, щоб вікно лягло на ті самі частки WIN) і зберігає ref_box_<ім'я>.jpg.
Спершу завжди роби перевірку очима: без --yes файл не пишеться, лише готується картинка з рамкою.

  python3 -m litter_algo.add_ref кадр.jpg --name nov --check /tmp/check.png
  python3 -m litter_algo.add_ref кадр.jpg --name nov --yes
"""
import argparse
import os
import cv2
import numpy as np

from .locate import HERE, WIN, Locator


def build(frame, win):
    """Кадр → опорний кадр: вікно (x1,y1,x2,y2) стає на місце WIN у 1280x720."""
    x1, y1, x2, y2 = win
    tx1, ty1, tx2, ty2 = WIN[0] * 1280, WIN[1] * 720, WIN[2] * 1280, WIN[3] * 720
    s = ((tx2 - tx1) / (x2 - x1) + (ty2 - ty1) / (y2 - y1)) / 2
    M = np.float32([[s, 0, tx1 - s * x1], [0, s, ty1 - s * y1]])
    return cv2.warpAffine(frame, M, (1280, 720), borderMode=cv2.BORDER_REPLICATE), s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("--name", required=True, help="коротка мітка положення, напр. 'oct24' → ref_box_oct24.jpg")
    ap.add_argument("--win", help="вікно барабана вручну: x1,y1,x2,y2 у кадрі 1280x720")
    ap.add_argument("--check", default="/tmp/add_ref_check.png", help="куди покласти картинку для перевірки очима")
    ap.add_argument("--yes", action="store_true", help="записати шаблон (без цього — лише картинка)")
    a = ap.parse_args(argv)
    frame = cv2.imread(a.frame)
    if frame is None:
        raise SystemExit("кадр не читається: %s" % a.frame)
    if frame.shape[1] != 1280:
        frame = cv2.resize(frame, (1280, 720))
    if a.win:
        win = tuple(int(v) for v in a.win.split(","))
        score = None
    else:
        score, _, win = Locator().locate(frame)
        if win is None:
            raise SystemExit("лоток не знайдено жодним шаблоном — вкажи вікно вручну через --win")
    vis = frame.copy()
    cv2.rectangle(vis, (win[0], win[1]), (win[2], win[3]), (0, 0, 255), 3)
    cv2.putText(vis, "вікно %s score=%s" % (str(win), "-" if score is None else round(score, 3)), (10, 40), 0, 1.0, (0, 255, 255), 2)
    ref, s = build(frame, win)
    both = np.vstack([cv2.resize(vis, (640, 360)), cv2.resize(ref, (640, 360))])
    cv2.imwrite(a.check, both)
    out = os.path.join(HERE, "ref_box_%s.jpg" % a.name)
    print("вікно:", win, "масштаб:", round(s, 3), "картинка для перевірки:", a.check)
    if not a.yes:
        print("файл НЕ записано (додай --yes, коли на картинці лоток порожній і рамка на вікні барабана):", out)
        return
    if os.path.exists(out):
        raise SystemExit("такий шаблон уже є: %s" % out)
    cv2.imwrite(out, ref, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("записано:", out)


if __name__ == "__main__":
    main()
