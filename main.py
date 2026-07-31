"""Phase B live entrypoint: ONVIF motion events -> RTSP capture -> classify -> storage -> Telegram."""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from urllib.parse import quote

import storage
import telegram_notify
from capture import MotionCapture, watch_motion_events
from classify import classify_frames

CONFIG_PATH = Path(__file__).parent / "config.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


async def main() -> None:
    load_dotenv()
    config = load_config()

    host = os.environ["CAMERA_HOST"]
    onvif_port = int(os.environ.get("ONVIF_PORT", "2020"))
    user = os.environ["CAMERA_USER"]
    password = os.environ["CAMERA_PASS"]
    rtsp_url = os.environ.get("RTSP_URL") or (
        f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}@{host}:554/stream2"
    )

    conn = storage.init_db()
    grabber = MotionCapture(rtsp_url, config["sample_fps"])
    capturing = False

    async for motion in watch_motion_events(host, onvif_port, user, password):
        if motion and not capturing:
            logger.info("Motion started")
            grabber.start()
            capturing = True
        elif not motion and capturing:
            frames = grabber.stop()
            capturing = False
            logger.info("Motion stopped, %d frames captured", len(frames))
            if not frames:
                continue

            result = await asyncio.to_thread(classify_frames, frames, config)
            timestamp = datetime.now().isoformat()
            storage.insert_event(
                conn,
                timestamp=timestamp,
                cat=result["cat"],
                is_day=result["is_day"],
                confidence=result["confidence"],
                dwell_seconds=result["dwell_seconds"],
                source_clip="live",
            )
            logger.info("Event classified: %s", result)

            try:
                if result["cat"] in ("white", "black"):
                    telegram_notify.notify_visit(result["cat"], result["is_day"], result["dwell_seconds"])
                elif result["cat"] in ("white_passby", "black_passby"):
                    telegram_notify.notify_passby(result["cat"].removesuffix("_passby"), result["is_day"])
            except Exception:
                logger.exception("Failed to send Telegram notification")


if __name__ == "__main__":
    asyncio.run(main())
