"""Phase B live entrypoint: ONVIF motion events -> RTSP capture -> classify -> storage -> Telegram."""

import asyncio
import logging
import os
import signal
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
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
    grabber.start_reader()
    capturing = False
    finalize_task: asyncio.Task | None = None
    coalesce_gap = config.get("motion_coalesce_gap_seconds", 3.0)

    async def finalize_event() -> None:
        nonlocal capturing, finalize_task
        try:
            await asyncio.sleep(coalesce_gap)
        except asyncio.CancelledError:
            return

        frames = grabber.stop_event()
        capturing = False
        finalize_task = None
        logger.info("Motion event finalized, %d frames captured", len(frames))
        if not frames:
            return

        try:
            now = datetime.now()
            results = await asyncio.to_thread(classify_frames, frames, config, now.hour)
            timestamp = now.isoformat()
            for result in results:
                storage.insert_event(
                    conn,
                    timestamp=timestamp,
                    cat=result["cat"],
                    is_day=result["is_day"],
                    confidence=result["confidence"],
                    dwell_seconds=result["dwell_seconds"],
                    source_clip="live",
                )
                frame = result.get("best_frame")
                loggable = {k: v for k, v in result.items() if k != "best_frame"}
                logger.info("Event classified: %s (photo: %s)", loggable, frame is not None)

                if result["cat"] in ("white", "black"):
                    telegram_notify.notify_visit(
                        result["cat"], result["is_day"], result["dwell_seconds"], frame=frame
                    )
                elif result["cat"] in ("white_passby", "black_passby"):
                    telegram_notify.notify_passby(
                        result["cat"].removesuffix("_passby"), result["is_day"], frame=frame
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to process/notify motion event, continuing")

    try:
        async for motion in watch_motion_events(host, onvif_port, user, password):
            if motion:
                if not capturing:
                    logger.info("Motion started")
                    grabber.start_event()
                    capturing = True
                if finalize_task is not None:
                    finalize_task.cancel()
                    finalize_task = None
                    logger.info("Motion resumed within coalesce window, continuing same event")
            elif capturing and finalize_task is None:
                # Don't finalize immediately -- the camera's motion state can flicker
                # off/on in short bursts during one continuous cat presence (observed
                # live: several sub-threshold fragments of what was clearly one visit).
                # Wait a short grace period; a motion-start during it cancels this and
                # the event just keeps accumulating frames instead of being split.
                finalize_task = asyncio.create_task(finalize_event())
    finally:
        if finalize_task is not None:
            finalize_task.cancel()
        grabber.stop_reader()


async def run_with_graceful_shutdown() -> None:
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(main())

    def _cancel():
        logger.info("Shutdown signal received, closing ONVIF subscription cleanly...")
        task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _cancel)

    try:
        await task
    except asyncio.CancelledError:
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(run_with_graceful_shutdown())
