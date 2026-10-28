# litter_algo

Вердикт «вікно барабана порожнє / зайняте» і клас кадру (clear / near / inside / blocking / unknown) незалежно від положення камери. Експериментальний код, **у живу службу (`main.py`, `drum_check.py`) не підключено**.

- `locate.py`: пошук лотка й вікна барабана; `classify.py`: вердикт і `decide()`; `empty_exemplars.npz`: 93 порожніх вікна (еталони).
- Тест без cv2: `python3 -m unittest litter_algo.test_classify` (мітки Слави в `labels_2026-10-28.json`); `test_locate.py` потребує cv2 (venv на Ubuntu).
- Скрипти `build_crops.py`, `make_sheets.py`, `render_review.py` збирають кропи й папку перевірки очима.
- Історія раундів і пороги: `docs/litter.md` у репозиторії ai-agents-dispatcher (гілка `worktree-litter-bounds-r2`, PR #1).
