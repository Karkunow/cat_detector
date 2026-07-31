# Cat litter-box detector — MVP

## Context

Two cats (one white, one black) use an automatic litter box that's in view of a TP-Link Tapo C200 camera. The camera also sees other activity in the room (owner, cleaner), so a naive "camera saw motion" trigger would produce false alerts. The goal is a locally-run (on the user's Mac, always-on) MVP that watches the camera feed, figures out *which* cat actually used the box (vs. a person just passing through), logs each visit, sends a Telegram message when it happens, and shows a small local dashboard with daily per-cat stats. This is explicitly scoped as a quick MVP to validate the idea, not a production service — no Docker/systemd, minimal moving parts, tune-as-you-go.

Project directory (`/Users/slava.karkunov/Projects/cat_detector`) is currently empty — this is a from-scratch build.

## Architecture

The build is split into two phases so the "brain" (detection/classification) gets proven on real footage before any live camera plumbing is written.

### Phase A — offline, on existing recorded clips (build this first)

```
A handful of clips manually exported from the Tapo app
(white cat day/night, black cat day/night, human day/night)
        │  data/sample_clips/*.mp4
        ▼
detect.py + classify.py, driven by classify_offline.py (CLI)
   - decode each clip with OpenCV, sample frames
   - YOLOv8n (ultralytics, COCO classes: 0=person, 15=cat) per sampled frame
   - person-dominant frame → event = "human"
   - cat bbox → crop → mean brightness → white / black / unknown
     (brightness works in both color-day and IR-night frames)
   - track cat bbox center vs. configured ROI (litter box area);
     needs ≥ MIN_DWELL_SECONDS inside ROI to count as a real "visit"
     vs. "pass_through"
        ▼
storage.py → SQLite (data/events.db)   [same schema Phase B will use]
```
`classify_offline.py` just points the same `detect.py`/`classify.py`/`storage.py` modules at a folder of pre-existing clips instead of a live stream — this is the calibration tool, and its code is not thrown away, it's reused as-is by Phase B.

### Phase B — live triggering (build after Phase A classification looks right)

```
Tapo C200 — ONVIF motion event subscription (local LAN)
        │  push: motion started / motion stopped
        ▼
capture.py — on motion-started event, opens RTSP
   (rtsp://camuser:campass@<ip>:554/stream2) just long enough to grab
   frames until motion-stopped, then closes it
        ▼
same detect.py + classify.py + storage.py + telegram_notify.py pipeline as Phase A
        ▼
dashboard.py (Streamlit, separate process) reads events.db, shows
daily charts per cat.
```

**Key design calls (researched, not assumed):**
- **Build and validate on existing exported clips before touching the live camera.** Decouples "does the classifier work" from "does the camera plumbing work" — fewer moving parts to debug at once, and the offline tool becomes the calibration tool too.
- **ONVIF motion-event subscription + on-demand RTSP for Phase B**, not a continuously-running frame-differencing loop and not `pytapo` SD-card polling. The camera already does motion detection (confirmed by the user — it only records when something moves); duplicating that with our own always-on frame-diff loop wastes CPU and is redundant. ONVIF is a documented, first-party local push mechanism (no cloud password needed, unlike `pytapo`'s SD-card download path which requires the TP-Link *cloud* account password and only supports date-level polling, not real-time events).
- **Color, not shape/pattern**, distinguishes the cats: mean brightness of the YOLO-cropped cat region. This holds in both day (color) and night (IR grayscale) frames, which is why the user's own observation — "white looks white, black looks black even under IR" — is the right signal to use.
- **ROI + dwell time**, not just "a cat was detected", distinguishes an actual box visit from a cat/person walking through frame.
- **Telegram via plain `requests`** against the Bot API (not `python-telegram-bot`) since we only ever send, never receive — avoids unneeded async machinery.
- **Streamlit** for the dashboard — least code for a local Python-only charts UI, reads `events.db` directly with pandas.

## File layout

```
cat_detector/
├── config.yaml            # ROI rect, day/night brightness thresholds, MIN_DWELL_SECONDS,
│                           # YOLO confidence threshold, sample fps
│                           # (RTSP URL/creds + telegram token added in Phase B)
├── .env                    # secrets: telegram token, camera account password (gitignored)
├── requirements.txt         # opencv-python, ultralytics, requests, pyyaml, streamlit, pandas, python-dotenv
├── detect.py                # YOLOv8n load + inference wrapper                         [Phase A]
├── classify.py              # brightness→cat-color heuristic, day/night check,
│                           # ROI/dwell tracker                                        [Phase A]
├── storage.py               # sqlite3 wrapper: init_db(), insert_event(), query helpers  [Phase A]
├── classify_offline.py       # CLI: runs detect+classify+storage over a folder of
│                           # pre-existing clips (data/sample_clips/) — this IS the
│                           # calibration tool, reused unchanged in Phase B             [Phase A]
├── dashboard.py               # Streamlit app: charts from events.db — can already be
│                           # used on Phase A data                                     [Phase A]
├── capture.py                 # ONVIF motion-event subscription + on-demand RTSP grab   [Phase B]
├── telegram_notify.py          # send_message()/send_photo() via Bot API                [Phase B]
├── main.py                     # live entrypoint: capture → detect → classify → storage
│                           # → notify                                                [Phase B]
├── data/
│   ├── sample_clips/            # manually exported clips for Phase A (white/black cat
│   │                           # × day/night, plus a couple of human/false-positive ones)
│   ├── events.db
│   ├── clips/                   # optional saved clips (skip retention policy for v1)
│   └── thumbnails/
└── README.md
```

`events` table: `id, timestamp, cat (white|black|unknown|human|pass_through), is_day, confidence, dwell_seconds, clip_path, thumbnail_path`.

## One-time setup (outside code)

**For Phase A (do this now):**
1. **Export sample clips** from the Tapo app: a handful covering white cat / day, white cat / night (IR), black cat / day, black cat / night, and a couple of "human passing through" clips for both day and night. Drop them in `data/sample_clips/`.
2. **Python env:** Python 3.11+ recommended (best wheel support for `ultralytics`/`opencv-python`). `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. First `detect.py` run auto-downloads `yolov8n.pt` (~6MB).

**For Phase B (later, once Phase A classification looks right):**
3. **Tapo Camera Account / RTSP + ONVIF:** Tapo app → C200 → gear icon → Advanced Settings → Camera Account → set a camera username/password → note the camera's LAN IP (reserve it via DHCP so it doesn't change). Sanity-check RTSP with `ffplay rtsp://<camuser>:<campass>@<ip>:554/stream2`, and confirm ONVIF motion events are reachable (e.g. via `onvif-zeep`'s event subscription) before wiring `capture.py`.
4. **Telegram bot:** message @BotFather → `/newbot` → get bot token. Message the new bot once yourself. Fetch `https://api.telegram.org/bot<token>/getUpdates` to read your `chat_id`. Put both in `.env`.
5. **ffmpeg:** `brew install ffmpeg` as a fallback if OpenCV's RTSP/H264 support is flaky.

## Calibration (Phase A, before trusting any classification)

1. Run `classify_offline.py` over `data/sample_clips/` in a "dump only" mode: for each clip, save sampled frames/crops and print mean-brightness values per detected cat box, plus a reference frame with a pixel grid for reading off ROI coordinates.
2. Read ROI rectangle coordinates off the reference frame, write into `config.yaml`.
3. From the printed brightness values, set `thresholds: {day: {...}, night: {...}}` in `config.yaml` around the midpoint between the two cats' observed brightness (day and night thresholds may differ).
4. Re-run `classify_offline.py` for real over all sample clips, manually check the resulting `events.db` rows against what you know each clip actually shows (which cat, visit vs. pass-through, human or not).

## Verification plan

**Phase A (offline, on exported clips):**
1. `detect.py` standalone on a few saved frames (person / white cat / black cat / empty) — confirm correct COCO classes and boxes.
2. `classify.py` against calibration crops — confirm correct white/black labels with chosen thresholds.
3. ROI/dwell filter — test with a "walk-through" clip vs. a "sits in box" clip, confirm `pass_through` vs `visit`.
4. Human filtering — run the human-passing-through clips, confirm they log as `human` with no visit recorded.
5. Full `classify_offline.py` run over all sample clips — check `events.db` rows match expectations for each clip.
6. Dashboard — `streamlit run dashboard.py` against this Phase A data, confirm charts render sensibly.

**Phase B (live, once Phase A passes):**
7. RTSP smoke test (`ffplay`) — confirm stream connects, note actual fps/resolution.
8. ONVIF event subscription smoke test — trigger motion, confirm `capture.py` receives the start/stop events and opens/closes RTSP accordingly.
9. End-to-end: let a real cat use the box, confirm an `events.db` row appears and a Telegram message arrives within a few seconds of the dwell threshold being met.
10. Human filtering live: walk through frame yourself, confirm it logs as `human` with no Telegram message.
11. Dashboard: confirm it updates as live events accumulate alongside the Phase A data.

## Explicitly out of scope for v1

- Telling the cats apart when both are in frame simultaneously (log `multiple_cats_detected`, pick the higher-confidence box).
- Clip retention/cleanup policy.
- Process supervision beyond a simple `launchd` plist or a left-open terminal/tmux session.
- ROI auto-detection (manual config only).
- Using `pytapo` as a CPU-saving pre-filter — revisit later if YOLO-on-every-motion-event proves too heavy.
