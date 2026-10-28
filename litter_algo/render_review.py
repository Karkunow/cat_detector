"""Папка для перевірки очима: кадри з намальованим вікном і вердиктом, розкладені по теках empty / occupied / unknown.
Вхід: /tmp/litter_meta.json (від build_crops.py) і /tmp/litter_verdicts.json ([[verdict, f, g], ...] у тому самому порядку).
Вихід: /tmp/litter_review/<вердикт>/<подія>_<кадр>_f.._g...jpg (640x360) і /tmp/litter_review.tar"""
import json
import os
import random
import shutil
import tarfile

import cv2

meta = json.load(open("/tmp/litter_meta.json"))
ver = json.load(open("/tmp/litter_verdicts.json"))
OUT = "/tmp/litter_review"
shutil.rmtree(OUT, ignore_errors=True)
random.seed(4)
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import decide

idx = {"clear": [], "near": [], "inside": [], "blocking": [], "unknown": []}
for i, v in enumerate(ver):
    v[0] = decide(v[0], meta[i]["cells"], meta[i].get("rim"))
    idx[v[0]].append(i)
for k in ("inside", "blocking"):
    random.shuffle(idx[k])
chosen = idx["clear"] + idx["near"] + idx["inside"][:150] + idx["blocking"][:150] + idx["unknown"]
COL = {"clear": (0, 200, 0), "near": (255, 160, 0), "inside": (0, 0, 255), "blocking": (200, 0, 200), "unknown": (0, 200, 255)}
for i in chosen:
    m, (v, f, g) = meta[i], ver[i]
    im = cv2.imread(m["f"])
    if im is None:
        continue
    if m["rect"]:
        cv2.rectangle(im, tuple(m["rect"][:2]), tuple(m["rect"][2:]), COL[v], 6)
    im = cv2.resize(im, (640, 360))
    ev = os.path.basename(os.path.dirname(m["f"]))
    rv = [r for r in (m.get("rim") or []) if r is not None]
    tag = ("f%.2f_g%.2f_r%.2f" % (f, g, min(rv) if rv else -1)) if f is not None else "noloc"
    cv2.putText(im, "%s %s" % (v, tag), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL[v], 2)
    os.makedirs("%s/%s" % (OUT, v), exist_ok=True)
    cv2.imwrite("%s/%s/%s_%s_%s.jpg" % (OUT, v, ev, os.path.basename(m["f"])[:-4], tag), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
with tarfile.open("/tmp/litter_review.tar", "w") as t:
    t.add(OUT, arcname="litter_review")
print({k: len(v) for k, v in idx.items()}, "у папці:", len(chosen))
