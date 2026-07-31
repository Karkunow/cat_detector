import os

import cv2
import requests

API_BASE = "https://api.telegram.org"

CAT_EMOJI = {"white": "⬜", "black": "⬛"}
CAT_LABEL_UK = {"white": "Білий", "black": "Чорний"}


def send_message(text: str, token: str | None = None, chat_id: str | None = None) -> None:
    token = token or os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = chat_id or os.environ["TELEGRAM_CHAT_ID"]
    resp = requests.post(
        f"{API_BASE}/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    resp.raise_for_status()


def send_photo(frame, caption: str, token: str | None = None, chat_id: str | None = None) -> None:
    token = token or os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = chat_id or os.environ["TELEGRAM_CHAT_ID"]
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    resp = requests.post(
        f"{API_BASE}/bot{token}/sendPhoto",
        data={"chat_id": chat_id, "caption": caption},
        files={"photo": ("cat.jpg", buf.tobytes(), "image/jpeg")},
        timeout=15,
    )
    resp.raise_for_status()


def _notify(text: str, frame) -> None:
    if frame is not None:
        send_photo(frame, text)
    else:
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
