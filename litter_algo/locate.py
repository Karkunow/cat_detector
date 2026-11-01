"""Знаходить лоток і вікно його барабана на кадрі камери незалежно від положення/масштабу камери.

Шаблон: контури корпусу лотка (рамка, панель, кнопка) з опорного кадру ref_box*.jpg, вікно барабана в шаблоні
замасковане (кіт у вікні не заважає пошуку). Шукаємо за масштабами й зсувом (matchTemplate по високочастотній карті
контурів у півроздільності). Повертає прямокутник вікна в координатах кадру й оцінку впевненості;
низька оцінка = «не знайшов лоток», тоді рішення «порожньо» не приймається.

Шаблонів кілька (банк): по одному на кожне перевірене положення камери (ref_box.jpg — поточне, ref_box_*.jpg —
попередні/інші). Зсув і масштаб matchTemplate покриває сам, а от ракурс (камеру повернули, лоток видно під іншим
кутом) — ні: контури корпусу перестають збігатися, NCC клітинок падає до ~0.05 і кожен кадр стає «near» (хибний блок).
Тому беремо шаблон із найвищою оцінкою, і клітинки/рамку рахуємо саме проти нього. Якщо жоден шаблон не впізнав
лоток (score < MIN_LOCATE у classify) — «unknown», чистка блокується: нове, не перевірене положення камери
безпечніше пропустити, ніж вгадувати.

Новий ref робиться скриптом add_ref.py з кадру ЗАВІДОМО порожнього лотка в новому положенні.
"""
import glob
import os
import time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# частки опорного кадру 1280x720: корпус лотка (шаблон) і вікно барабана всередині
BOX = (0.24, 0.10, 0.77, 0.87)
WIN = (0.345, 0.25, 0.51, 0.56)
SCALES = [round(0.30 + 0.05 * i, 2) for i in range(18)]   # 0.30 .. 1.15 від розміру в опорному кадрі
ANGLES = [0, -2, 2, -4, 4, -6, 6, -8, 8]   # камеру могли ще й нахилити; 0° пробуємо першим
FAST_OK = 0.45
COARSE = 0.5     # у скільки разів дрібніша карта для грубого пошуку кута   # якщо без повороту лоток упізнано впевнено — кути не перебираємо (швидкий шлях, як раніше)
HALF = 0.5
REF_W = 1280     # шаблони зняті на кадрі 1280x720; кадр іншої ширини спершу зводимо до неї
                 # (інакше потрібного масштабу немає серед SCALES: лоток не знаходиться, а пошук кутів з'їдає секунди)


def edge_map(gray):
    g = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = cv2.resize(m, None, fx=HALF, fy=HALF, interpolation=cv2.INTER_AREA)
    # прибираємо локальне середнє (щоб однорідні світлі/темні ділянки не давали кореляції)
    return m - cv2.blur(m, (41, 41))


def ref_paths():
    """Усі опорні кадри: ref_box.jpg першим (поточне положення), далі ref_box_*.jpg за абеткою."""
    main = os.path.join(HERE, "ref_box.jpg")
    rest = sorted(p for p in glob.glob(os.path.join(HERE, "ref_box_*.jpg")) if p != main)
    return ([main] if os.path.exists(main) else []) + rest


class _Ref:
    """Один шаблон: карта контурів корпусу з діркою на місці вікна + геометрія вікна й рамки."""

    def __init__(self, path):
        self.name = os.path.basename(path)
        ref = cv2.imread(path)
        if ref is None:
            raise ValueError("не читається опорний кадр: %s" % path)
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
        self.hole = (hx1, hy1, hx2, hy2)
        R = 20
        self.rim_strips = [(max(hx1 - R, 0), hy1, hx1, hy2), (hx2, hy1, hx2 + R, hy2),
                           (hx1, max(hy1 - R, 0), hx2, hy1), (hx1, hy2, hx2, hy2 + R)]
        self.box_origin = (x1, y1)
        self.win_in_tmpl = (WIN[0] * ew - x1, WIN[1] * eh - y1, WIN[2] * ew - x1, WIN[3] * eh - y1)
        self._cache = {}

    def scaled(self, s, ang=0):
        """Шаблон і маска, зменшені в s разів і повернуті на ang градусів (кеш)."""
        key = (s, ang)
        if key not in self._cache:
            t = cv2.resize(self.tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            m = cv2.resize(self.mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            if ang:
                h, w = t.shape
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
                t = cv2.warpAffine(t, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
                m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
                t = t * m
            self._cache[key] = (t, m)
        return self._cache[key]

    def win_rect(self, s, ang):
        """Вікно барабана в координатах (зменшеного й повернутого) шаблону: (x1, y1, x2, y2)."""
        return self.rect_at(self.win_in_tmpl, s, ang)

    def hole_rect(self, s, ang):
        """Дірка в масці (вікно з запасом) — від неї відмірюються смуги рамки."""
        return self.rect_at(self.hole, s, ang)

    def rect_at(self, rect, s, ang):
        wx1, wy1, wx2, wy2 = [v * s for v in rect]
        if not ang:
            return wx1, wy1, wx2, wy2
        th, tw = self.scaled(s, ang)[0].shape
        M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), ang, 1.0)
        cx, cy = (wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0
        nx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
        ny = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
        hw, hh = (wx2 - wx1) / 2.0, (wy2 - wy1) / 2.0
        return nx - hw, ny - hh, nx + hw, ny + hh


RELOAD_EVERY = 60   # раз на стільки секунд перевіряємо, чи не з'явився в теці новий ref_box*.jpg


class Locator:
    def __init__(self, ref_path=None):
        """ref_path: нічого (увесь банк ref_box*.jpg, з автоперечитуванням), один шлях або список шляхів."""
        self._watch = ref_path is None   # стежимо за текою лише коли банк наш, а не заданий явно
        paths = ref_paths() if ref_path is None else ([ref_path] if isinstance(ref_path, str) else list(ref_path))
        self._load(paths)

    def _load(self, paths):
        if not paths:
            raise ValueError("немає жодного опорного кадру ref_box*.jpg у %s" % HERE)
        self.refs = [_Ref(p) for p in paths]
        self._paths = list(paths)
        self._checked = time.time()
        self._hint_ang = 0   # кут, яким лоток знайшовся минулого разу: пробуємо його одразу (камеру нахилили — далі кадри теж нахилені)
        # сумісність зі старим кодом/тестами, які читали ці поля напряму
        r0 = self.refs[0]
        self.tmpl, self.mask, self.rim_strips = r0.tmpl, r0.mask, r0.rim_strips
        self.box_origin, self.win_in_tmpl = r0.box_origin, r0.win_in_tmpl

    def _maybe_reload(self):
        """Шаблон нового положення/освітлення, доданий збоку (add_ref, litter-ref-auto), підхоплюється сам —
        інакше служби, що тримають Locator у пам'яті, не бачили б його до перезапуску."""
        if not self._watch or time.time() - self._checked < RELOAD_EVERY:
            return
        self._checked = time.time()
        paths = ref_paths()
        if paths != self._paths:
            try:
                self._load(paths)
            except Exception:
                pass   # битий чи недописаний файл не має ламати перевірку: працюємо зі старим банком

    def _sweep(self, e, angles, scales=None):
        best = (-1.0, None, None)
        last = None
        for ref in self.refs:
            for ang in angles:
                for s in (scales or SCALES):
                    t, m = ref.scaled(s, ang)
                    if t.shape[0] >= e.shape[0] or t.shape[1] >= e.shape[1]:
                        continue
                    r = cv2.matchTemplate(e, t, cv2.TM_CCORR_NORMED, mask=m)
                    r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
                    _, mx, _, loc = cv2.minMaxLoc(r)
                    if mx > best[0]:
                        wx1, wy1, wx2, wy2 = ref.win_rect(s, ang)
                        rect = ((loc[0] + wx1) / HALF, (loc[1] + wy1) / HALF,
                                (loc[0] + wx2) / HALF, (loc[1] + wy2) / HALF)
                        best = (float(mx), s, tuple(int(round(v)) for v in rect))
                        last = (e, loc, s, ref, ang)
        return best, last

    @staticmethod
    def _prep(frame):
        """Кадр до ширини REF_W і коефіцієнт повернення координат у вихідний кадр."""
        w = frame.shape[1]
        if w == REF_W:
            return frame, 1.0
        k = w / float(REF_W)
        return cv2.resize(frame, (REF_W, int(round(frame.shape[0] / k))), interpolation=cv2.INTER_AREA), k

    def locate(self, frame):
        """(score, scale, win_rect_px(x1,y1,x2,y2) у координатах переданого кадру) або (score, None, None).
        Спершу швидкий прохід без повороту (всі шаблони × масштаби). Якщо лоток упізнано невпевнено
        (камеру зсунули/нахилили) — другий прохід із перебором кутів ANGLES."""
        self._maybe_reload()
        frame, k = self._prep(frame)
        e = edge_map(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        first = [0] + ([self._hint_ang] if self._hint_ang else [])
        best, last = self._sweep(e, first)
        if best[0] < FAST_OK:
            # 2-й крок: камеру, схоже, нахилили. Кут шукаємо грубо на вдвічі дрібнішій карті (дешево),
            # а на повній роздільності перевіряємо лише два найкращі кути.
            small = cv2.resize(e, None, fx=COARSE, fy=COARSE, interpolation=cv2.INTER_AREA)
            cand = []
            for ang in ANGLES:
                if ang in first:
                    continue
                b, _ = self._sweep(small, [ang], [round(x * COARSE, 3) for x in SCALES])
                cand.append((b[0], ang))
            cand.sort(reverse=True)
            for _, ang in cand[:2]:
                b2, l2 = self._sweep(e, [ang])
                if b2[0] > best[0]:
                    best, last = b2, l2
        self._last = last
        self._hint_ang = last[4] if last else 0
        if k != 1.0 and best[2] is not None:
            best = (best[0], best[1], tuple(int(round(v * k)) for v in best[2]))
        return best

    def analyze(self, frame):
        """Повний розбір кадру: оцінка пошуку, прямокутник вікна, NCC 8 клітинок корпусу (низький = закрито котом/людиною),
        нормований вміст вікна (64x48, яскравість / яскравість корпусу навколо вікна), ім'я шаблону («положення камери»)."""
        frame, k = self._prep(frame)
        score, s, rect = self.locate(frame)
        out = {"score": score, "scale": s, "rect": rect, "cells": None, "rim": None, "crop": None, "bezel": None,
               "ref": None, "angle": None}
        if rect is None:
            return out
        e, loc, s, ref, ang = self._last
        out["ref"] = ref.name
        out["angle"] = ang
        t, m = ref.scaled(s, ang)
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
        rx1, ry1, rx2, ry2 = ref.hole_rect(s, ang)
        R = 20 * s
        strips = [(rx1 - R, ry1, rx1, ry2), (rx2, ry1, rx2 + R, ry2), (rx1, ry1 - R, rx2, ry1), (rx1, ry2, rx2, ry2 + R)]
        for (sx1, sy1, sx2, sy2) in strips:
            ys, ye, xs, xe = int(max(sy1, 0)), int(max(sy2, 0)), int(max(sx1, 0)), int(max(sx2, 0))
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
        if ang:
            # кадр нахилений: повертаємо його навколо центру вікна назад у систему шаблону, інакше в кроп
            # потрапляє край рамки й порожнє вікно читається як зайняте (хибний блок)
            M = cv2.getRotationMatrix2D(((x1 + x2) / 2.0, (y1 + y2) / 2.0), -ang, 1.0)
            g = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
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
