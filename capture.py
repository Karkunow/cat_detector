"""Phase B: ONVIF motion-event subscription + on-demand RTSP frame grabbing.

The camera already does its own motion detection (confirmed: it only records when
something moves). Instead of re-implementing motion detection ourselves, we subscribe
to the camera's ONVIF PullPoint event source and only open the RTSP stream while a
motion event is active.
"""

import asyncio
import datetime as dt
import logging
import threading
import time

import cv2
from onvif import ONVIFCamera

logger = logging.getLogger(__name__)

MOTION_TOPIC_HINTS = ("motion", "cellmotiondetector")
PULL_TIMEOUT = "PT20S"  # ONVIF duration: block up to 20s per PullMessages call
MAX_EVENT_SECONDS = 120  # safety cap so a stuck "motion=true" can't capture forever


class MotionCapture:
    """Opens the RTSP stream in a background thread and buffers (t, frame) pairs
    from the moment start() is called until stop() is called."""

    def __init__(self, rtsp_url: str, sample_fps: float):
        self.rtsp_url = rtsp_url
        self.sample_fps = sample_fps
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frames: list[tuple[float, "cv2.Mat"]] = []

    def start(self) -> None:
        self._frames = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error("Failed to open RTSP stream at %s", self.rtsp_url)
            return
        start_t = time.monotonic()
        last_sample_t = 0.0
        min_interval = 1.0 / self.sample_fps
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.monotonic() - start_t
                if now - last_sample_t >= min_interval:
                    self._frames.append((now, frame))
                    last_sample_t = now
                if now > MAX_EVENT_SECONDS:
                    logger.warning("Motion event exceeded %ss, cutting it off", MAX_EVENT_SECONDS)
                    break
        finally:
            cap.release()

    def stop(self) -> list[tuple[float, "cv2.Mat"]]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self._frames


def _extract_motion_state(message) -> bool | None:
    """Returns True/False if `message` is a motion event, None if it's something else
    or we can't parse it (ONVIF event schemas vary a lot between camera vendors)."""
    topic = getattr(message, "Topic", None)
    topic_str = getattr(topic, "_value_1", "") if topic else ""
    if not any(hint in topic_str.lower() for hint in MOTION_TOPIC_HINTS):
        return None

    msg = getattr(message, "Message", None)
    data = getattr(getattr(msg, "_value_1", None), "Data", None)
    items = getattr(data, "SimpleItem", None) if data else None
    if not items:
        return None
    for item in items:
        name = getattr(item, "Name", "").lower()
        if name in ("state", "ismotion", "motion"):
            value = getattr(item, "Value", "")
            return str(value).lower() in ("true", "1")
    return None


MAX_CONSECUTIVE_FAILURES = 3


async def _subscribe(host: str, port: int, user: str, password: str):
    # no_cache: avoid the WSDL sqlite cache (seen implicated in long-session
    # memory issues with some ONVIF cameras). adjust_time: Tapo cameras can
    # reject WS-Security-authenticated requests (generic SOAP faults, no detail)
    # if the camera's clock has drifted from ours -- this compensates for it.
    cam = ONVIFCamera(host, port, user, password, no_cache=True, adjust_time=True)
    await cam.update_xaddrs()

    def on_subscription_lost():
        logger.warning("ONVIF PullPoint subscription lost")

    manager = await cam.create_pullpoint_manager(dt.timedelta(minutes=10), on_subscription_lost)
    await manager.start()
    return cam, manager


async def _close(cam: ONVIFCamera, manager) -> None:
    # Best-effort: if the connection is already broken, these will raise too --
    # that's fine, we're tearing down regardless.
    try:
        await manager.stop()
    except Exception:
        logger.debug("manager.stop() failed during teardown, ignoring", exc_info=True)
    try:
        await cam.close()
    except Exception:
        logger.debug("cam.close() failed during teardown, ignoring", exc_info=True)


SUBSCRIBE_BACKOFF_BASE = 5
SUBSCRIBE_BACKOFF_MAX = 300  # cap at 5 min -- hammering a camera that's stuck
# subscribing (e.g. it hit an internal subscription-count limit) just makes
# things worse and never recovers on its own.


async def watch_motion_events(host: str, port: int, user: str, password: str):
    """Async generator yielding True on motion-start and False on motion-stop.

    Re-subscribes from scratch after repeated PullMessages failures, since a
    stuck/stale subscription (e.g. left over from an unclean previous shutdown)
    doesn't recover by just retrying the same call."""
    subscribe_failures = 0
    while True:
        try:
            cam, manager = await _subscribe(host, port, user, password)
            subscribe_failures = 0
        except asyncio.CancelledError:
            raise
        except Exception:
            backoff = min(SUBSCRIBE_BACKOFF_BASE * (2**subscribe_failures), SUBSCRIBE_BACKOFF_MAX)
            subscribe_failures += 1
            logger.warning("Failed to (re)subscribe, retrying in %ds", backoff, exc_info=True)
            await asyncio.sleep(backoff)
            continue

        service = manager.get_service()
        logger.info("Subscribed to ONVIF motion events on %s", host)

        consecutive_failures = 0
        try:
            while True:
                try:
                    result = await service.PullMessages({"Timeout": PULL_TIMEOUT, "MessageLimit": 20})
                    consecutive_failures = 0
                except Exception:
                    consecutive_failures += 1
                    logger.warning(
                        "PullMessages failed (%d/%d), retrying in 5s",
                        consecutive_failures,
                        MAX_CONSECUTIVE_FAILURES,
                        exc_info=True,
                    )
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.warning("Too many consecutive failures, re-subscribing from scratch")
                        break
                    await asyncio.sleep(5)
                    continue

                for message in getattr(result, "NotificationMessage", []) or []:
                    state = _extract_motion_state(message)
                    if state is not None:
                        yield state
        finally:
            await _close(cam, manager)
