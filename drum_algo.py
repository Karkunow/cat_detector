"""Жива «кіт у барабані» перевірка детектора на litter_algo (замість NCC-ознак drum_check у classify.py).

classes(frames) -> клас кожного кадру (clear / near / inside / blocking / unknown / error), та сама логіка, що в petbox-safe-clean.
inside_run(classes) -> скільки кадрів наприкінці підряд «inside», якщо це підтверджує «кіт усередині», інакше 0;
None — алгоритм збійнув (error), і викликач лишається на старій перевірці.
cv2/litter_algo імпортуються ліниво, щоб чиста логіка inside_run тестувалась без cv2.
"""
import logging
import threading

logger = logging.getLogger(__name__)

TAIL = 3         # скільки останніх кадрів дивимось
MIN_INSIDE = 2   # скільки з них (разом з останнім) мають бути inside

_lock = threading.Lock()    # Locator тримає стан між викликами (_last), а classify і recheck ідуть з різних потоків
_locator = None


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
                out.append(decide(wv, a["cells"], a["rim"]))
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
