"""Проходить по кадрах подій за останні дні, зберігає розбір (вікно, клітинки, нормований вміст) у /tmp/litter_crops.npz.
Запуск на Ubuntu у фоні: nohup .venv/bin/python build_crops.py [крок] &"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locate import Locator

step = int(sys.argv[1]) if len(sys.argv) > 1 else 3
H = os.path.expanduser("~")
files = []
for d in sorted(glob.glob(H + "/cat_detector/data/frames/events/*")):
    files += sorted(glob.glob(d + "/*.jpg"))[::step]
files += sorted(glob.glob(H + "/petbox-safe-clean-frames/*.jpg"))
files += sorted(glob.glob(H + "/reshoot_2026-10-26/*.jpg"))
print(len(files), flush=True)
loc = Locator()
meta, crops = [], []
for i, f in enumerate(files):
    im = cv2.imread(f)
    if im is None:
        continue
    r = loc.analyze(im)
    meta.append({"f": f, "score": r["score"], "scale": r["scale"], "rect": r["rect"], "cells": r["cells"], "rim": r["rim"], "bezel": r["bezel"], "ok": r["crop"] is not None})
    crops.append(r["crop"] if r["crop"] is not None else np.zeros((48, 64), np.float32))
    if i % 200 == 0:
        print(i, flush=True)
np.savez_compressed("/tmp/litter_crops.npz", crops=np.array(crops, np.float32))
json.dump(meta, open("/tmp/litter_meta.json", "w"))
print("done", len(meta), flush=True)
