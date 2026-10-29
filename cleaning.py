"""Рух від циклу очищення лотка (кіт тут ні до чого): визначаємо за журналом сторожа петбокса ~/petbox-guard.log,
рядки «<дата час> стан: <було> → <стало> (...)». Цикл = стани очищення/пересипання/пакування/вирівнювання піску."""
from datetime import datetime
from pathlib import Path

GUARD_LOG = Path.home() / "petbox-guard.log"
CYCLE_STATES = {"очищення", "пересипання", "пакування", "вирівнювання піску"}
TAIL_BYTES = 64 * 1024
START_SLACK_S = 30.0    # сторож опитує пристрій з паузою: рух міг початися трохи раніше за запис про старт
END_SLACK_S = 20.0      # подію завершуємо через coalesce + обробку вже після запису про кінець циклу


def _parse(line: str) -> tuple[float, str, str] | None:
    try:
        stamp, rest = line.split(" стан: ", 1)
        prev, cur = rest.split(" (", 1)[0].split(" → ")
        return datetime.strptime(stamp.strip(), "%Y-%m-%d %H:%M:%S").timestamp(), prev.strip(), cur.strip()
    except ValueError:
        return None


def cycle_windows(path: Path = GUARD_LOG) -> list[tuple[float, float | None]]:
    """[(початок, кінець або None, якщо цикл іде)] за хвостом журналу; нечитаний чи відсутній журнал → []."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - TAIL_BYTES))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    windows: list[tuple[float, float | None]] = []
    start = None
    for line in text.splitlines():
        tr = _parse(line)
        if not tr:
            continue
        ts, prev, cur = tr
        if cur in CYCLE_STATES and prev not in CYCLE_STATES and start is None:
            start = ts
        elif cur not in CYCLE_STATES and start is not None:
            windows.append((start, ts))
            start = None
    if start is not None:
        windows.append((start, None))
    return windows


def is_cleaning_event(event_start: float, event_end: float, path: Path = GUARD_LOG) -> bool:
    """Подія цілком у межах циклу очищення (з невеликим запасом). Подія, що лише зачепила цикл, лишається звичайною:
    кіт біля лотка перед стартом циклу не має стати «очищенням»."""
    for s, e in cycle_windows(path):
        if event_start >= s - START_SLACK_S and (e is None or event_end <= e + END_SLACK_S):
            return True
    return False
