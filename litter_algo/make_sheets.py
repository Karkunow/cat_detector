"""Контактні аркуші нормованих вікон для ручної розмітки. Вибірка кадрів фіксована (seed), індекси = індекси в litter_meta.json.
Запуск на Ubuntu: .venv/bin/python make_sheets.py [N] -> /tmp/sheet_k.jpg, /tmp/sheet_idx.json"""
import json
import os
import random
import sys

import cv2
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
meta = json.load(open("/tmp/litter_meta.json"))
crops = np.load("/tmp/litter_crops.npz")["crops"]
good = [i for i, m in enumerate(meta) if m["ok"] and m["score"] >= 0.25]
random.seed(11)
random.shuffle(good)
pick = good[:N]
json.dump(pick, open("/tmp/sheet_idx.json", "w"))
print("кадрів з нормальним пошуком:", len(good), "з", len(meta), "; у вибірці", len(pick))
COLS, ROWS = 10, 8
per = COLS * ROWS
for k in range(0, len(pick), per):
    tiles = []
    for j, i in enumerate(pick[k:k + per]):
        c = np.clip(crops[i] * 150, 0, 255).astype(np.uint8)
        c = cv2.cvtColor(cv2.resize(c, (128, 96), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        cv2.putText(c, str(k + j), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(c)
    while len(tiles) % COLS:
        tiles.append(np.zeros((96, 128, 3), np.uint8))
    cv2.imwrite("/tmp/sheet_%d.jpg" % (k // per), np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]))
