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
from collections import deque

import cv2
from onvif import ONVIFCamera

logger = logging.getLogger(__name__)

MOTION_TOPIC_HINTS = ("motion", "cellmotiondetector")
PULL_TIMEOUT = "PT5S"  # ONVIF duration: block up to 5s per PullMessages call.
# Kept short since Tapo's lightweight network stack appears to unilaterally drop
# long-held long-poll connections (ServerDisconnectedError) -- shorter polls mean
# more requests, but each is less likely to outlast whatever the camera's own
# connection timeout is. 5s matches what other ONVIF client examples use
# successfully against similar embedded cameras.
MAX_EVENT_SECONDS = 120  # safety cap so a stuck "motion=true" can't capture forever


PRE_ROLL_SECONDS = 2.0  # keep this much recent video buffered at all times, so a
# motion event that's already brief by the time ONVIF tells us doesn't also lose
# its first couple of seconds to RTSP connection setup latency.
RECONNECT_BACKOFF = 2.0


class MotionCapture:
    """Keeps the RTSP stream open continuously in a background thread (avoids
    paying ~2-3s of RTSP connection/handshake latency on every single motion
    event -- measured directly against this camera, and confirmed to cause real
    misses on brief events). Always maintains a short rolling pre-roll buffer;
    start_event()/stop_event() mark which frames belong to a motion event, seeded
    with whatever pre-roll was already buffered so fast entrances aren't cut off."""

    def __init__(self, rtsp_url: str, sample_fps: float):
        self.rtsp_url = rtsp_url
        self.sample_fps = sample_fps
        self._thread: threading.Thread | None = None
        self._stop_reader = threading.Event()
        self._lock = threading.Lock()
        self._ring_buffer: deque[tuple[float, "cv2.Mat"]] = deque()
        self._event_frames: list[tuple[float, "cv2.Mat"]] | None = None

    def start_reader(self) -> None:
        self._stop_reader.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_reader(self) -> None:
        self._stop_reader.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        min_interval = 1.0 / self.sample_fps
        while not self._stop_reader.is_set():
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                logger.error("Failed to open RTSP stream at %s, retrying in %ss", self.rtsp_url, RECONNECT_BACKOFF)
                cap.release()
                time.sleep(RECONNECT_BACKOFF)
                continue

            last_sample_t = 0.0
            try:
                while not self._stop_reader.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        logger.warning("RTSP read failed, reconnecting")
                        break
                    now = time.monotonic()
                    if now - last_sample_t < min_interval:
                        continue
                    last_sample_t = now
                    with self._lock:
                        if self._event_frames is not None:
                            self._event_frames.append((now, frame))
                            if now - self._event_frames[0][0] > MAX_EVENT_SECONDS:
                                logger.warning("Motion event exceeded %ss, cutting it off", MAX_EVENT_SECONDS)
                                self._event_frames = self._event_frames[-1:]
                        else:
                            self._ring_buffer.append((now, frame))
                            while self._ring_buffer and now - self._ring_buffer[0][0] > PRE_ROLL_SECONDS:
                                self._ring_buffer.popleft()
            finally:
                cap.release()
            if not self._stop_reader.is_set():
                time.sleep(RECONNECT_BACKOFF)

    def start_event(self) -> None:
        with self._lock:
            self._event_frames = list(self._ring_buffer)

    def stop_event(self) -> list[tuple[float, "cv2.Mat"]]:
        with self._lock:
            frames = self._event_frames or []
            self._event_frames = None
        if not frames:
            return []
        t0 = frames[0][0]
        return [(t - t0, frame) for t, frame in frames]


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
    # nat_override: this camera's CreatePullPointSubscription response advertises
    # a bogus internal port (e.g. :1028) for its own SubscriptionReference address
    # instead of the port we actually connected on (:2020) -- observed directly via
    # "TimeoutError: Request to http://<host>:1028/event-... timed out". With
    # nat_override, the library ignores that self-reported address and always
    # talks back to the host:port we originally connected to.
    cam = ONVIFCamera(host, port, user, password, no_cache=True, adjust_time=True, nat_override=True)
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
