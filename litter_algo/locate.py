"""Знаходить лоток і вікно його барабана на кадрі камери незалежно від положення/масштабу камери.

Шаблон: контури корпусу лотка (рамка, панель, кнопка) з опорного кадру ref_box.jpg, вікно барабана в шаблоні
замасковане (кіт у вікні не заважає пошуку). Шукаємо за масштабами й зсувом (matchTemplate по високочастотній карті
контурів у півроздільності). Повертає прямокутник вікна в координатах кадру й оцінку впевненості;
низька оцінка = «не знайшов лоток», тоді рішення «порожньо» не приймається.
"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# частки опорного кадру 1280x720: корпус лотка (шаблон) і вікно барабана всередині
BOX = (0.24, 0.10, 0.77, 0.87)
WIN = (0.345, 0.25, 0.51, 0.56)
SCALES = [round(0.30 + 0.05 * i, 2) for i in range(18)]   # 0.30 .. 1.15 від розміру в опорному кадрі
HALF = 0.5


def edge_map(gray):
    g = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = cv2.resize(m, None, fx=HALF, fy=HALF, interpolation=cv2.INTER_AREA)
    # прибираємо локальне середнє (щоб однорідні світлі/темні ділянки не давали кореляції)
    return m - cv2.blur(m, (41, 41))


class Locator:
    def __init__(self, ref_path=os.path.join(HERE, "ref_box.jpg")):
        ref = cv2.imread(ref_path)
        h, w = ref.shape[:2]
        e = edge_map(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
        eh, ew = e.shape
        x1, y1, x2, y2 = (int(BOX[0] * ew), int(BOX[1] * eh), int(BOX[2] * ew), int(BOX[3] * eh))
        self.tmpl = e[y1:y2, x1:x2].copy()
        mask = np.ones_like(self.tmpl)
        wx1, wy1, wx2, wy2 = (int((WIN[0] - 0.012) * ew) - x1, int((WIN[1] - 0.02) * eh) - y1,
                              int((WIN[2] + 0.012) * ew) - x1, int((WIN[3] + 0.02) * eh) - y1)
        mask[max(wy1, 0):wy2, max(wx1, 0):wx2] = 0
        self.mask = mask
        self.tmpl = self.tmpl * mask
        # смуги рамки одразу довкола дірки-вікна (в координатах шаблону): left, right, top, bottom (x1, y1, x2, y2)
        hx1, hy1, hx2, hy2 = max(wx1, 0), max(wy1, 0), wx2, wy2
        R = 20
        self.rim_strips = [(max(hx1 - R, 0), hy1, hx1, hy2), (hx2, hy1, hx2 + R, hy2), (hx1, max(hy1 - R, 0), hx2, hy1), (hx1, hy2, hx2, hy2 + R)]
        self.box_origin = (x1, y1)
        self.win_in_tmpl = (WIN[0] * ew - x1, WIN[1] * eh - y1, WIN[2] * ew - x1, WIN[3] * eh - y1)

    def locate(self, frame):
        """(score, scale, win_rect_px(x1,y1,x2,y2) у координатах повного кадру) або (score, None, None)."""
        H, W = frame.shape[:2]
        e = edge_map(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        best = (-1.0, None, None)
        self._last = None
        for s in SCALES:
            t = cv2.resize(self.tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            m = cv2.resize(self.mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            if t.shape[0] >= e.shape[0] or t.shape[1] >= e.shape[1]:
                continue
            r = cv2.matchTemplate(e, t, cv2.TM_CCORR_NORMED, mask=m)
            r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                wx1, wy1, wx2, wy2 = self.win_in_tmpl
                rect = ((loc[0] + wx1 * s) / HALF, (loc[1] + wy1 * s) / HALF, (loc[0] + wx2 * s) / HALF, (loc[1] + wy2 * s) / HALF)
                best = (float(mx), s, tuple(int(round(v)) for v in rect))
                self._last = (e, loc, s)
        return best

    def analyze(self, frame):
        """Повний розбір кадру: оцінка пошуку, прямокутник вікна, NCC 8 клітинок корпусу (низький = закрито котом/людиною),
        нормований вміст вікна (64x48, яскравість / яскравість корпусу навколо вікна)."""
        score, s, rect = self.locate(frame)
        out = {"score": score, "scale": s, "rect": rect, "cells": None, "rim": None, "crop": None, "bezel": None}
        if rect is None:
            return out
        e, loc, s = self._last
        t = cv2.resize(self.tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(self.mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        th, tw = t.shape
        patch = e[loc[1]:loc[1] + th, loc[0]:loc[0] + tw]
        cells = []
        for gy in range(3):
            for gx in range(3):
                if gx == 1 and gy == 1:
                    continue
                ys, ye, xs, xe = gy * th // 3, (gy + 1) * th // 3, gx * tw // 3, (gx + 1) * tw // 3
                a, b, mm = patch[ys:ye, xs:xe], t[ys:ye, xs:xe], m[ys:ye, xs:xe]
                sel = mm > 0.5
                if sel.sum() < 50:
                    cells.append(None)
                    continue
                a, b = a[sel] - a[sel].mean(), b[sel] - b[sel].mean()
                d = float(np.sqrt((a * a).sum() * (b * b).sum()))
                cells.append(float((a * b).sum() / d) if d > 0 else 0.0)
        out["cells"] = cells
        rim = []
        for (sx1, sy1, sx2, sy2) in self.rim_strips:
            ys, ye, xs, xe = int(sy1 * s), int(sy2 * s), int(sx1 * s), int(sx2 * s)
            a, b = patch[ys:ye, xs:xe], t[ys:ye, xs:xe]
            if a.size < 50 or a.shape != b.shape:
                rim.append(None)
                continue
            a, b = a - a.mean(), b - b.mean()
            d = float(np.sqrt((a * a).sum() * (b * b).sum()))
            rim.append(float((a * b).sum() / d) if d > 0 else 0.0)
        out["rim"] = rim
        H, W = frame.shape[:2]
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        x1, y1, x2, y2 = rect
        w_, h_ = x2 - x1, y2 - y1
        ox1, oy1, ox2, oy2 = max(int(x1 - .35 * w_), 0), max(int(y1 - .35 * h_), 0), min(int(x2 + .35 * w_), W), min(int(y2 + .35 * h_), H)
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            return out
        ring = g[oy1:oy2, ox1:ox2].copy()
        ring[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = np.nan
        bez = float(np.nanmedian(ring))
        out["bezel"] = bez
        out["crop"] = cv2.resize(g[y1:y2, x1:x2] / max(bez, 1.0), (64, 48), interpolation=cv2.INTER_AREA)
        return out
