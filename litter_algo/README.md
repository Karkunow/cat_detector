# litter_algo

Вердикт «вікно барабана порожнє / зайняте» і клас кадру (clear / near / inside / blocking / unknown) незалежно від положення камери. Експериментальний код, **у живу службу (`main.py`, `drum_check.py`) не підключено**; лише тіньовий режим (нижче).

- `locate.py`: пошук лотка й вікна барабана; `classify.py`: вердикт і `decide()`; `empty_exemplars.npz`: 93 порожніх вікна (еталони).
- Тест без cv2: `python3 -m unittest litter_algo.test_classify` (мітки Слави в `labels_2026-10-28.json`); `test_locate.py` потребує cv2 (venv на Ubuntu).
- Скрипти `build_crops.py`, `make_sheets.py`, `render_review.py` збирають кропи й папку перевірки очима.
- Історія раундів і пороги: `docs/litter.md` у репозиторії ai-agents-dispatcher (гілка `worktree-litter-bounds-r2`, PR #1).

## Тіньовий режим (`shadow.py`, 28.10.2026)
Лише пише вердикт у `data/litter_shadow.jsonl` (або `$LITTER_SHADOW_LOG`) поряд із рішенням чинної логіки; чистку не запускає й не блокує. Збої/зависання (ліміт 15 с) ковтаються й потрапляють у журнал, CLI завжди з кодом 0.
- Запис: `python3 -m litter_algo.shadow --frame КАДР.jpg --current allow|block [--reason ТЕКСТ] [--source ІМʼЯ]` (з venv, де є cv2). Викликати з чинної логіки ПІСЛЯ її рішення, у фоні або `|| true`.
- Звіт: `python3 -m litter_algo.shadow --summary [ЖУРНАЛ]`: згода/розбіжності. `algo_stricter` (чинна дозволила, algo блокує) перевіряти очима; `algo_looser` (чинна заблокувала, algo дозволила): algo не можна пускати замість чинної.
- Тести: `python3 -m unittest litter_algo.test_shadow`. Підключення в `petbox-safe-clean`/`drum_check.py` на Ubuntu НЕ зроблено (чекає слова Слави).
