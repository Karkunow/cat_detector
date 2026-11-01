"""Чи заходив кіт у барабан лотка і скільки там просидів (01.11.2026).

Навіщо. Раніше «візит» = YOLO бачив кота в широкій зоні кадру config["roi"] (зашиті частки кадру) щонайменше
min_dwell_seconds. Кіт, що сидить ПЕРЕД лотком, теж у цій зоні — і рахувався як «сходив у туалет» (03:23 01.11:
білий посидів у барабані ~10 с, вийшов і сів перед входом — записалось «18 с у туалеті»; 01:54 — камера дивилась
униз, вікна барабана не видно, кіт пройшов повз — записалось «чорний сходив, 15 с»). А «тривалість» була часом,
коли кота видно біля входу, а не часом у барабані.

Як тепер. Вікно барабана на кадрі знаходить litter_algo (банк шаблонів, не залежить від положення камери). Для
кожного кадру події стан:
  D — YOLO бачить кота, і його рамка майже вся (>= CONTAIN_MIN) у вікні барабана (розширеному на EXPAND),
      а низ рамки не нижче низу вікна більше ніж на BELOW_MAX висоти вікна: кіт у барабані, видно крізь вікно;
  O — YOLO бачить кота, але рамка виходить за вікно / сягає підлоги: кіт ззовні (сидить перед лотком, іде повз);
  d — YOLO кота не бачить, а litter_algo каже «inside» і вікно явно зайняте (f >= F_STRONG): кіт у барабані,
      але детектор його не впізнав (кіт сидить глибоко, видно лише шерсть);
  ? — лоток (вікно барабана) на кадрі не знайдено: судити не можна;
  . — інше.
Перевірено на розмітці 823 кадрів (~/litter_labels_2026-11-01): жоден із 215 кадрів «кіт ззовні» і жоден із 504
порожніх не отримав D чи d; на рівні подій — 0 хибних візитів.

Візит у барабан — щонайменше MIN_DRUM_FRAMES кадрів D/d і разом щонайменше MIN_DRUM_SECONDS. Час у барабані —
сума відрізків між першим і останнім D/d, розірваних паузами > DRUM_GAP. Це лише для розпізнавання й сповіщень:
на чистку лотка модуль не впливає (її вирішує petbox-safe-clean своїми перевірками).
"""
import logging

logger = logging.getLogger(__name__)

CONTAIN_MIN = 0.75      # частка площі рамки кота всередині (розширеного) вікна; кіт ззовні на розмітці <= 0.49, у барабані 0.85–1.0
BELOW_MAX = 0.35        # низ рамки нижче низу вікна, у висотах вікна; кіт ззовні >= 1.04, у барабані ~0
EXPAND = 0.15           # вікно розширюємо на стільки з кожного боку (голова/вуха виходять за рамку вікна)
F_STRONG = 0.30         # «вікно явно зайняте» для стану d: порожній лоток дає f 0.02–0.16, кіт у вікні >= 0.32
MIN_LOCATE = 0.40       # довіра до знайденого вікна для ВІЗИТІВ (суворіше за 0.25 у litter_algo): 01.11 01:54, коли камера
                        # дивилась униз, локатор «знаходив» крихітне вікно на кнопці з оцінкою 0.21–0.26; справжнє — 0.65–0.98
ALGO_EVERY = 1.0        # litter_algo (0.7 с CPU) рахуємо не частіше разу на стільки секунд, решта кадрів бере останнє вікно
MIN_DRUM_FRAMES = 2
MIN_DRUM_SECONDS = 2.0
DRUM_GAP = 20.0         # кіт у барабані часто зовсім не видно; довша пауза — це вже окремий відрізок
END_SLACK = 3.0         # «кіт досі в барабані» — останній D/d не раніше ніж за стільки секунд до кінця події


def bbox_state(bbox, rect) -> str:
    """'D' — рамка кота у вікні барабана, 'O' — кіт ззовні. bbox, rect = (x1, y1, x2, y2) у пікселях одного кадру."""
    a, b, c, d = bbox
    x1, y1, x2, y2 = rect
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    ex1, ey1, ex2, ey2 = x1 - EXPAND * w, y1 - EXPAND * h, x2 + EXPAND * w, y2 + EXPAND * h
    area = max((c - a) * (d - b), 1)
    inter = max(0.0, min(c, ex2) - max(a, ex1)) * max(0.0, min(d, ey2) - max(b, ey1))
    return "D" if inter / area >= CONTAIN_MIN and (d - y2) / h <= BELOW_MAX else "O"


def frame_state(cat_bboxes, rect, algo_cls=None, f=None) -> str:
    if rect is None:
        return "?"
    if cat_bboxes:
        return "D" if any(bbox_state(bb, rect) == "D" for bb in cat_bboxes) else "O"
    if algo_cls == "inside" and f is not None and f >= F_STRONG:
        return "d"
    return "."


def summarize(states: list, last_t: float) -> dict:
    """states — [(t, стан)] події. Повертає ознаки візиту в барабан."""
    drum_t = [t for t, s in states if s in ("D", "d")]
    seen_box = any(s != "?" for _, s in states)
    out = {"seen_box": seen_box, "n_drum": len(drum_t), "drum_seconds": 0.0, "drum_start_t": None,
           "drum_end_t": None, "inside_at_end": False, "visit": False}
    if not drum_t:
        return out
    seconds, start, prev = 0.0, drum_t[0], drum_t[0]
    for t in drum_t[1:]:
        if t - prev > DRUM_GAP:
            seconds += prev - start
            start = t
        prev = t
    seconds += prev - start
    out.update(drum_seconds=round(seconds + ALGO_EVERY, 1), drum_start_t=drum_t[0], drum_end_t=drum_t[-1])
    out_after = any(s == "O" and t > drum_t[-1] for t, s in states)
    out["inside_at_end"] = (last_t - drum_t[-1]) <= END_SLACK and not out_after
    out["visit"] = len(drum_t) >= MIN_DRUM_FRAMES and out["drum_seconds"] >= MIN_DRUM_SECONDS
    return out


class DrumTrack:
    """Збирає стан кожного кадру події. litter_algo — через drum_algo.analyze (спільний Locator під замком)."""

    def __init__(self):
        self.states = []
        self._rect = None
        self._algo_t = None
        self._cls, self._f = None, None
        self.failed = False

    def add(self, t: float, frame, cat_bboxes: list) -> str:
        if not self.failed and (self._algo_t is None or t - self._algo_t >= ALGO_EVERY):
            import drum_algo
            res = drum_algo.analyze(frame)
            if res is None:
                self.failed = True
                logger.warning("visit_drum: litter_algo недоступний, візити рахуються по-старому")
            else:
                cls, f, rect, score = res
                self._algo_t = t
                self._rect = rect if (rect is not None and score >= MIN_LOCATE) else None
                self._cls, self._f = cls, f
        fresh = self._algo_t == t
        st = frame_state(cat_bboxes, self._rect, self._cls if fresh else None, self._f if fresh else None)
        self.states.append((t, st))
        return st

    def summary(self, last_t: float) -> dict | None:
        if self.failed or not self.states:
            return None
        s = summarize(self.states, last_t)
        s["trace"] = "".join(st for _, st in self.states)
        return s
