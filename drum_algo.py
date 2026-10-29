"""Жива «кіт у барабані» перевірка детектора на litter_algo (замість NCC-ознак drum_check у classify.py).

classes(frames) -> клас кожного кадру (clear / near / inside / blocking / unknown / error), та сама логіка, що в petbox-safe-clean.
inside_run(classes) -> скільки кадрів наприкінці підряд «inside», якщо це підтверджує «кіт усередині», інакше 0;
None — алгоритм збійнув (error), і викликач лишається на старій перевірці.
cv2/litter_algo імпортуються ліниво, щоб чиста логіка inside_run тестувалась без cv2.
"""
import logging
import threading

logger = logging.getLogger(__name__)

SOFT_CELL_MIN = 0.40   # litter_algo.classify.CELL_MIN = 0.45: кіт у вікні з цілою рамкою й клітинками 0.40–0.45 — це «усередині» (порожній лоток теж дає 0.43–0.46)
TAIL = 3         # скільки останніх кадрів дивимось
MIN_INSIDE = 2   # скільки з них (разом з останнім) мають бути inside

_lock = threading.Lock()    # Locator тримає стан між викликами (_last), а classify і recheck ідуть з різних потоків
_locator = None


def soften(cls: str, window_verdict: str, cells, rim) -> str:
    """Для сповіщення (не для чистки) «blocking» через клітинки корпусу трохи нижче порога CELL_MIN, коли вікно зайняте, а рамка
    входу ціла (min >= RIM_MIN), вважаємо «inside»: кіт сидить у вікні (29.10 00:17 білий, клітинки 0.43; 28.10 07:38, 0.43).
    Закриту рамку чи клітинки сильно нижче порога (кіт ззовні перед лотком: 0.0–0.35) це не стосується."""
    if cls != "blocking" or window_verdict != "occupied":
        return cls
    from litter_algo.classify import RIM_MIN
    rims = [r for r in (rim or []) if r is not None]
    body = [c for c in (cells or []) if c is not None]
    if rims and min(rims) >= RIM_MIN and body and min(body) >= SOFT_CELL_MIN:
        return "inside"
    return cls


def classes(frames: list) -> list[str]:
    """Клас litter_algo для кожного кадру; збій імпорту чи кадру дає 'error', а не виняток."""
    global _locator
    with _lock:
        try:
            from litter_algo.classify import decide, verdict
            from litter_algo.locate import Locator
            if _locator is None:
                _locator = Locator()
        except Exception as e:
            logger.error("drum_algo: litter_algo недоступний (%s: %s)", type(e).__name__, e)
            return ["error"] * len(frames)
        out = []
        for f in frames:
            try:
                a = _locator.analyze(f)
                wv, _, _ = verdict(a)
                out.append(soften(decide(wv, a["cells"], a["rim"]), wv, a["cells"], a["rim"]))
            except Exception as e:
                logger.error("drum_algo: збій на кадрі (%s: %s)", type(e).__name__, e)
                out.append("error")
        return out


def inside_run(cls: list[str]) -> int | None:
    """cls — класи останніх кадрів (не більше TAIL), від старого до нового. Повертає довжину хвоста підряд «inside»
    (0 = кота всередині нема), якщо щонайменше MIN_INSIDE з кадрів inside і останній теж; None, якщо є 'error'."""
    if "error" in cls:
        return None
    if len(cls) < MIN_INSIDE or cls[-1] != "inside" or sum(c == "inside" for c in cls) < MIN_INSIDE:
        return 0
    run = 0
    for c in reversed(cls):
        if c != "inside":
            break
        run += 1
    return run
