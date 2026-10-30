"""Вердикт «вікно барабана порожнє / зайняте / лоток не знайдено» незалежно від положення камери.

1) Locator знаходить лоток (масштаб і зсув) і вікно барабана; оцінка пошуку < MIN_LOCATE → «unknown».
2) Вміст вікна нормується яскравістю корпусу навколо вікна й власною медіаною, порівнюється з бібліотекою
   порожніх вікон (empty_exemplars.npz, 93 перевірені очима кадри різного світла/положення камери).
   f = частка пікселів, що відрізняються від найближчого еталона більше ніж на DIFF; g = найбільше відхилення в одному
   з 12 блоків (кіт, що закрив лише куток вікна, дає малий f, але великий g).
3) «empty» лише коли f <= F_MAX і g <= G_MAX; усе інше «occupied»; не знайдено лоток або поганий розріз «unknown».
Неоднозначність завжди в бік відмови: чистка дозволяється тільки за «empty» (+ окремі перевірки руху/запису/workstatus).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_LOCATE = 0.25
DIFF, F_MAX, G_MAX = 0.12, 0.10, 0.22


def _library():
    return np.load(os.path.join(HERE, "empty_exemplars.npz"))["ex"].astype(np.float32)


_EX = None


def compare(crop):
    """(f, g) відносно найближчого порожнього еталона; crop = 48x64, яскравість/яскравість корпусу."""
    global _EX
    if _EX is None:
        _EX = _library()
    c = crop / max(float(np.median(crop)), 1e-3)
    best = (9.0, 9.0)
    for e in _EX:
        df = np.abs(c - e)
        f = float((df > DIFF).mean())
        if f < best[0]:
            best = (f, float(df.reshape(3, 16, 4, 16).mean(axis=(1, 3)).max()))
    return best


CELL_MIN = 0.45   # найгірша з 8 клітинок корпусу лотка (NCC контурів із шаблоном): нижче — корпус закритий (кіт/ноги перед лотком)


SOFT_CELL = 2     # клітинка №2 (верхній правий кут ref_box.jpg; індекс з 0) на порожньому лотку балансує на 0.43–0.46 (хибні near 28–30.10.2026):
SOFT_CELL_MIN = 0.35   # для near вона вважається закритою лише нижче цього порога; решта клітинок — за CELL_MIN. На мітках 28.10 жоден near не тримався на ній самій.
                       # Індекс прив'язаний до шаблону ref_box.jpg: після його перезйомки перевірити заново.

RIM_MIN = 0.25    # найгірша з 4 смуг рамки входу (NCC контурів із шаблоном): вище — рамка видна, кіт за нею = всередині


def decide(window_verdict, cells, rim=None):
    """Підсумок: clear (вікно порожнє й корпус не закритий) / near (вікно порожнє, але корпус чимось закритий) /
    inside (вікно зайняте, рамка входу ціла й корпус відкритий: кіт всередині) / blocking (вікно зайняте, а рамка
    входу або корпус закриті: кіт перекриває вхід ззовні) / unknown (лоток не знайдено). Кіт, що стоїть далі за межі корпусу, цією ознакою не ловиться:
    для цього лишається YOLO/рух детектора. Усе, крім clear, чистку блокує."""
    if window_verdict == "occupied":
        vals = [r for r in (rim or []) if r is not None]
        body = [c for c in (cells or []) if c is not None]
        if body and min(body) < CELL_MIN:
            return "blocking"   # корпус закритий тілом (кіт перед лотком, ззовні), навіть коли рамка вікна ціла
        return "inside" if vals and min(vals) >= RIM_MIN else "blocking"
    if window_verdict != "empty":
        return window_verdict
    vals = [c for c in (cells or []) if c is not None]
    if not vals:
        return "near"
    closed = any(c is not None and c < (SOFT_CELL_MIN if i == SOFT_CELL else CELL_MIN) for i, c in enumerate(cells))
    return "near" if closed else "clear"


def verdict(analysis):
    """analysis — результат Locator.analyze(frame). Повертає (verdict, f, g)."""
    if analysis["crop"] is None or analysis["score"] < MIN_LOCATE:
        return "unknown", None, None
    f, g = compare(analysis["crop"])
    return ("empty" if f <= F_MAX and g <= G_MAX else "occupied"), f, g
