"""Phase B live entrypoint: ONVIF motion events -> RTSP capture -> classify -> storage -> Telegram."""

import asyncio
import json
import logging
import os
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from dotenv import load_dotenv
from urllib.parse import quote

import align_guard as ag
import storage
import telegram_notify
from capture import MotionCapture, watch_motion_events
from classify import classify_frames, drum_inside_check
from detect import detect, person_present

CONFIG_PATH = Path(__file__).parent / "config.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


OUTBOX_DIR = Path.home() / "outbox"
EVENTS_DIR = Path(__file__).parent / "data" / "frames" / "events"
MOTION_STATE = Path(__file__).parent / "data" / "motion_state.json"
last_cleanup = 0.0


def cleanup_event_frames(now_ts: float, keep_days: int = 3) -> None:
    global last_cleanup
    last_cleanup = now_ts
    for old in EVENTS_DIR.glob("*"):
        if old.stat().st_mtime < now_ts - keep_days * 86400:
            shutil.rmtree(old) if old.is_dir() else old.unlink()


def save_event_frames(frames, now, step: int = 2, pre=()) -> Path:
    """Кожна подія камери лишає кожен step-й кадр (і останній) у data/frames/events/<мітка>/<індекс>.jpg (навіть без кота
    в кадрі) — для калібрування й тестів: кіт у барабані YOLO не бачить. t кадру = індекс / sample_fps. Кадри за ~10 с ДО події
    (pre, не класифікуються) лежать поряд у pre/<індекс>.jpg. Старе чистимо раз на добу. Повертає теку події."""
    out = EVENTS_DIR / f"{now:%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    for i in sorted(set(range(0, n, step)) | {n - 1}):
        cv2.imwrite(str(out / f"{i:03d}.jpg"), frames[i][1])
    save_context_frames(out / "pre", pre, step)
    if now.timestamp() - last_cleanup >= 86400:
        cleanup_event_frames(now.timestamp())
    return out


def save_context_frames(out: Path, frames, step: int = 2) -> None:
    """Кадри поза подією (pre/ або post/ у теці події): кожен step-й, від старого до нового."""
    if not frames:
        return
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(frames), step):
        cv2.imwrite(str(out / f"{i:03d}.jpg"), frames[i])


async def main() -> None:
    load_dotenv()
    config = load_config()
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_event_frames(datetime.now().timestamp())

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
    inside_since: tuple[float, str] | None = None
    last_visit: tuple[float, str] | None = None
    motion_active = False
    last_motion_ts = 0.0

    def write_motion_state() -> None:
        tmp = MOTION_STATE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({
                "active": motion_active,
                "last_motion_ts": last_motion_ts,
                "updated_ts": time.time(),
            }))
            os.replace(tmp, MOTION_STATE)
        except OSError:
            logger.exception("Failed to write motion_state.json")

    async def motion_heartbeat() -> None:
        nonlocal last_motion_ts
        while True:
            if motion_active:
                last_motion_ts = time.time()
            write_motion_state()
            await asyncio.sleep(5)

    heartbeat_task = asyncio.create_task(motion_heartbeat())
    processing = False   # іде classify_frames/сповіщення після події: еталони не підміняємо

    def recent_events() -> tuple[float | None, bool]:
        """(час останньої події, чи остання подія за 6 год — «усередині» без наступного виходу/проходу)."""
        try:
            row = conn.execute("select timestamp, cat from events order by id desc limit 1").fetchone()
        except Exception:
            logger.exception("align_guard: events.db недоступна")
            return time.time(), True    # не знаємо — вважаємо «щойно була подія»
        if not row:
            return None, False
        ts = datetime.fromisoformat(row[0]).timestamp()
        return ts, row[1].endswith("_inside") and time.time() - ts < 6 * 3600

    async def align_loop() -> None:
        """Сторож положення: раз на interval_s дивиться свіжий кадр, веде TRACKED/UNCERTAIN/LOST, пише data/align_state.json,
        шле рідкісне повідомлення в ~/outbox і збирає кандидатів еталонів (див. align_guard.py)."""
        if not ag.TEMPLATE_PATH.exists():
            shutil.copy(ag.DATA_DIR / "drum_base_irday.jpg", ag.TEMPLATE_PATH)
        tpl = ag.Template(cv2.imread(str(ag.TEMPLATE_PATH)))
        guard = ag.Guard(now=time.time(), prev=ag.read_state())
        ag.write_state(guard.to_dict(time.time()))
        last_cand = 0.0
        logger.info("align_guard: запущено (notify=%s, promote=%s)", config.get("align_notify", True), config.get("align_promote", "auto"))
        while True:
            await asyncio.sleep(ag.CFG["interval_s"])
            try:
                now = time.time()
                frames = [] if capturing else grabber.latest_frames(3)
                cat = False
                if frames:
                    cat = bool(await asyncio.to_thread(detect, frames[-1], 0.15))
                busy = capturing or motion_active or now - last_motion_ts < 180 or cat
                chk = await asyncio.to_thread(ag.check_frames, frames, tpl, busy, now)
                if capturing:     # кадрів події не беремо, це не втрата лотка
                    chk["busy"] = True
                prev_state = guard.state
                guard.update(chk)
                if guard.state != prev_state:
                    logger.info("align_guard: %s → %s (%s; score %.2f, поза %s, режим %s)", prev_state, guard.state,
                                guard.reason, chk["score"], chk["pose"] and tuple(round(x) for x in chk["pose"]), chk.get("mode"))
                extra = {}
                # повідомлення Славі
                why = guard.notice_due(now, datetime.now().hour)
                if why and config.get("align_notify", True):
                    path = ag.write_notice(ag.notice_text({"reason": why}, guard.bad_since or now), OUTBOX_DIR)
                    guard.mark_notice(now, why)
                    guard.was_lost_notified = guard.state == ag.LOST
                    logger.info("align_guard: повідомлення Славі → %s (%s)", path.name, why)
                # кандидати еталонів і промоція
                mode = ag.need_candidate(guard)
                if mode and config.get("align_promote", "auto") != "off" and frames and now - last_cand >= ag.CAND_CFG["cand_min_interval_s"]:
                    ev_ts, unresolved = await asyncio.to_thread(recent_events)
                    block = ag.quiet_window(guard, now, capturing, processing, motion_active, last_motion_ts, ev_ts, unresolved)
                    if block is None:
                        d = await asyncio.to_thread(ag.capture_candidate, frames[-1], chk["pose"], mode, now)
                        last_cand = now
                        logger.info("align_guard: кандидат еталона %s → %s", mode, d.name)
                        group = ag.consensus(ag.list_candidates(mode, now))
                        if group and config.get("align_promote", "auto") != "auto":
                            logger.info("align_guard: консенсус еталонів %s (%d кандидатів); promote=manual — вручну: align_promote_cli.py --apply %s", mode, len(group), mode)
                            extra["cand_pending"] = {"mode": mode, "n": len(group)}
                        elif group and not capturing and not processing and now - guard.last_promote.get(mode, 0) >= 86400:
                            res = [ag.matches_old(c, mode, guard.ref_poses) for c in group]
                            if all(r[0] for r in res):
                                info = await asyncio.to_thread(ag.promote, group, mode, now)
                                guard.ref_poses = ag.load_ref_poses()
                                guard.last_promote[mode] = now
                                guard.enhanced_until = info["enhanced_until"]
                                logger.info("align_guard: еталони %s промотовано (%s), відкат: %s/prev", mode, info["label"], info["dir"])
                            else:
                                logger.info("align_guard: консенсус %s є, але кандидат не збігається зі старим еталоном чи старого нема (%s) — вручну", mode, res)
                                extra["cand_pending"] = {"mode": mode, "n": len(group), "match": [r[1] for r in res]}
                    else:
                        extra["cand_block"] = block
                ag.write_state(guard.to_dict(now, extra))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("align_guard: збій перевірки (стан не оновлено → чистка вважатиме його застарілим)")

    align_task = asyncio.create_task(align_loop()) if config.get("align_guard", True) in (True, "on") else None

    async def recheck_inside(color: str, event_ts: float) -> None:
        """Кіт зайшов у лоток і сидить без руху: ONVIF мовчить, події нема. Через inside_recheck_seconds після входу дивимось
        свіжі кадри барабана; не порожній — шлемо «сидить усередині» (ті самі dedupe й last_visit, що в classify)."""
        nonlocal inside_since, last_visit
        try:
            await asyncio.sleep(config.get("inside_recheck_seconds", 20))
            if capturing:
                return
            frames = grabber.latest_frames(3)
            if frames and await asyncio.to_thread(person_present, frames, 0.25, 1):
                logger.info("Inside recheck: у кадрі людина, мовчимо")
                return
            res = await asyncio.to_thread(drum_inside_check, frames, config) if frames else None
            if res is None:
                logger.info("Inside recheck: барабан порожній або кадрів нема")
                return
            if capturing:
                return
            if color == "unknown":
                color = res["color"]
            ts = time.time()
            last_visit = (ts, color)
            if inside_since and inside_since[1] == color and ts - inside_since[0] < config.get("inside_dedup_minutes", 10) * 60:
                return
            inside_since = (ts, color)
            logger.info("Inside recheck: кіт %s усередині (ncc %.2f)", color, res["ncc"])
            telegram_notify.notify_inside(color, frame=res["frame"])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Inside recheck failed")

    async def save_post_context(event_dir: Path, event_end: float) -> None:
        """Кадри за ~context_seconds ПІСЛЯ події: чекаємо, поки вони накопичаться в буфері, і пишемо в <подія>/post/."""
        try:
            await asyncio.sleep(grabber.context_seconds + 0.5)
            post = grabber.post_context(event_end)
            await asyncio.to_thread(save_context_frames, event_dir / "post", post)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to save post-event context frames")

    async def finalize_event() -> None:
        nonlocal capturing, finalize_task, inside_since, last_visit, processing
        try:
            await asyncio.sleep(coalesce_gap)
        except asyncio.CancelledError:
            return

        frames = grabber.stop_event()
        pre_frames = grabber.pre_context()
        event_end = grabber.last_event_end()
        capturing = False
        finalize_task = None
        logger.info("Motion event finalized, %d frames captured", len(frames))
        if not frames:
            return

        processing = True
        try:
            now = datetime.now()
            event_dir = save_event_frames(frames, now, pre=pre_frames)
            if event_end is not None:
                asyncio.create_task(save_post_context(event_dir, event_end))
            results = await asyncio.to_thread(classify_frames, frames, config, now.hour)
            human = await asyncio.to_thread(person_present, [f for _, f in frames])
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
                if frame is not None:  # кадри лишаємо для калібрування порогів білий/чорний
                    frames_dir = Path(__file__).parent / "data" / "frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(frames_dir / f"{now:%Y%m%d-%H%M%S}-{result['cat']}.jpg"), frame)
                loggable = {k: v for k, v in result.items() if k != "best_frame"}
                logger.info("Event classified: %s (photo: %s)", loggable, frame is not None)
                if human:
                    logger.info("У кадрі події людина, сповіщення про кота не шлемо")
                    continue

                ts = now.timestamp()
                if last_visit and ts - last_visit[0] > config.get("exit_window_minutes", 4) * 60:
                    last_visit = None
                if result.get("inside_now"):
                    last_visit = (ts, result["color"])
                    if not (
                        inside_since
                        and inside_since[1] == result["color"]
                        and ts - inside_since[0] < config.get("inside_dedup_minutes", 10) * 60
                    ):
                        inside_since = (ts, result["color"])
                        telegram_notify.notify_inside(result["color"], frame=frame)
                    continue
                if result["cat"] in ("white", "black", "white_passby", "black_passby"):
                    inside_since = None
                if result["cat"] in ("white", "black"):
                    last_visit = (ts, result["cat"])
                    telegram_notify.notify_visit(
                        result["cat"], result["is_day"], result["dwell_seconds"], frame=frame
                    )
                elif result["cat"] in ("white_passby", "black_passby"):
                    color = result["cat"].removesuffix("_passby")
                    if last_visit and last_visit[1] == color and ts >= last_visit[0]:
                        last_visit = None
                        telegram_notify.notify_exit(color, frame=frame)
                    else:
                        telegram_notify.notify_passby(color, result["is_day"], frame=frame)
                        asyncio.create_task(recheck_inside(color, ts))
                elif result["cat"] == "pass_through":
                    asyncio.create_task(recheck_inside("unknown", ts))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to process/notify motion event, continuing")
        finally:
            processing = False

    try:
        async for motion in watch_motion_events(host, onvif_port, user, password):
            motion_active = bool(motion)
            if motion:
                last_motion_ts = time.time()
            write_motion_state()
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
        heartbeat_task.cancel()
        if align_task is not None:
            align_task.cancel()
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
