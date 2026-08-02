import logging
import os
import time

import cv2
import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
SEND_RETRIES = 2  # a live Wi-Fi hiccup has already caused a confirmed-visit photo
# to silently fail to send (read timeout) -- retrying the exact same request
# handles a transient blip; falling back to a text-only message (see _notify)
# handles the case where photo upload specifically keeps failing.
RETRY_BACKOFF_SECONDS = 3

CAT_EMOJI = {"white": "⬜", "black": "⬛"}
CAT_LABEL_UK = {"white": "Білий", "black": "Чорний"}


def _post_with_retry(*args, **kwargs) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1 + SEND_RETRIES):
        try:
            resp = requests.post(*args, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < SEND_RETRIES:
                logger.warning("Telegram request failed (attempt %d), retrying: %s", attempt + 1, exc)
                time.sleep(RETRY_BACKOFF_SECONDS)
    raise last_exc


def send_message(text: str, token: str | None = None, chat_id: str | None = None) -> None:
    token = token or os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = chat_id or os.environ["TELEGRAM_CHAT_ID"]
    _post_with_retry(
        f"{API_BASE}/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )


def send_photo(frame, caption: str, token: str | None = None, chat_id: str | None = None) -> None:
    token = token or os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = chat_id or os.environ["TELEGRAM_CHAT_ID"]
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    _post_with_retry(
        f"{API_BASE}/bot{token}/sendPhoto",
        data={"chat_id": chat_id, "caption": caption},
        files={"photo": ("cat.jpg", buf.tobytes(), "image/jpeg")},
        timeout=15,
    )


def _notify(text: str, frame) -> None:
    if frame is not None:
        try:
            send_photo(frame, text)
            return
        except requests.RequestException:
            logger.exception("Failed to send Telegram photo, falling back to text-only")
    send_message(text)


def notify_visit(cat: str, is_day: bool, dwell_seconds: float, frame=None) -> None:
    label = CAT_LABEL_UK.get(cat, cat)
    emoji = CAT_EMOJI.get(cat, "🐾")
    time_of_day = "вдень" if is_day else "вночі"
    text = f"{emoji} {label} кіт сходив в туалет ({time_of_day}, {dwell_seconds:.0f}с)"
    _notify(text, frame)


def notify_passby(cat: str, is_day: bool, frame=None) -> None:
    label = CAT_LABEL_UK.get(cat, cat)
    emoji = CAT_EMOJI.get(cat, "🐾")
    time_of_day = "вдень" if is_day else "вночі"
    text = f"{emoji} {label} кіт пройшов повз ({time_of_day}, до лотка не заходив)"
    _notify(text, frame)
