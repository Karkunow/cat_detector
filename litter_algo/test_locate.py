"""Перевірка локатора на випадкових кадрах: малює знайдене вікно й оцінку. Запуск на Ubuntu: .venv/bin/python test_locate.py"""
import glob
import os
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locate import Locator

random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
H = os.path.expanduser("~")
files = []
for d in sorted(glob.glob(H + "/cat_detector/data/frames/events/*")):
    fs = sorted(glob.glob(d + "/*.jpg"))
    files += fs[::6]
random.shuffle(files)
loc = Locator()
tiles = []
t0 = time.time()
for f in files[:48]:
    im = cv2.imread(f)
    sc, s, rect = loc.locate(im)
    c = im.copy()
    if rect:
        cv2.rectangle(c, rect[:2], rect[2:], (0, 255, 255) if sc > 0.3 else (0, 0, 255), 4)
    c = cv2.resize(c, (320, 180))
    cv2.putText(c, "%s s=%.2f sc=%.2f" % (os.path.basename(os.path.dirname(f))[-6:], s or 0, sc), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    tiles.append(c)
print("сек на кадр: %.2f" % ((time.time() - t0) / len(tiles)))
while len(tiles) % 6:
    tiles.append(np.zeros((180, 320, 3), np.uint8))
cv2.imwrite("/tmp/test_locate.jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
