# Cat litter-box detector

MVP for detecting which cat (white/black) used the litter box, from camera footage.
Full design rationale: see [PLAN.md](PLAN.md).

## Status: Phase A and Phase B both working

- **Phase A** (offline classification, `classify_offline.py`): calibrated against 12 real
  exported clips in `data/sample_clips/` (gitignored — private footage, not pushed).
- **Phase B** (live, `main.py`): ONVIF motion-event subscription + on-demand RTSP capture +
  Telegram notifications, confirmed working live against the real camera (RTSP connects,
  ONVIF PullPoint delivers motion start/stop, human filtering confirmed with no false
  alerts). Currently running unattended overnight under `caffeinate` to catch a real cat
  visit end-to-end.

### Phase A results (`python classify_offline.py run --clear`)

| clip | ground truth | predicted | correct? |
|---|---|---|---|
| day_black_inside | black visit | black, dwell 24.5s | ✅ |
| day_black_inside_2 | brief pop-in, cat sat ~1-2s and left — not a real visit | black_passby | ✅ |
| day_black_passby | no visit | unknown (no detections) | ✅ (no false alert) |
| day_human | human | human | ✅ |
| day_white_inside | white visit | white, dwell 57.8s | ✅ |
| night_black_inside | brief pop-in, cat sat ~2s and left — not a real visit | pass_through | ✅ |
| night_black_inside_3 | black visit (long, cat close to entrance) | black, dwell 37.4s | ✅ |
| night_black_passby | no visit | black_passby | ✅ |
| night_human | human | unknown (no detections) | ✅ (no false alert) |
| night_white_inside | white visit | white, dwell 44.6s | ✅ |
| night_white_passby | no visit (cat brushes lens) | white_passby | ✅ |
| night_white_passby_2 | no visit (dim/far white cat) | black_passby | ⚠️ color wrong, but no false "visit" alert either way |

**12/12 correct on confirmed visits vs. non-visits (the thing that drives Telegram alerts),
1 wrong color label on a non-visit passby (cosmetic, no alert sent).**

`_passby` labels are cats that were seen but didn't dwell in the ROI long enough to count
as a real visit — `main.py` sends a lighter "passed by" Telegram message for these, added
specifically to get faster live signal on whether the pipeline works end-to-end without
waiting for a full ≥12s box visit.

### Important recalibration: IR brightness ≠ visible-light color

Original night thresholds (`black_max: 70, white_min: 90`) were calibrated only from
`night_black_passby`, where the black cat was far from the camera/IR illuminator (median
brightness 54). When `night_black_inside_3` came in — a real, long visit with the black cat
close to the entrance — its IR brightness came out at median **120**, comfortably inside
the old "white" bucket, which would have produced a **wrong alert** ("white cat" instead of
black) rather than a silent miss. That's the failure mode we most want to avoid.

Root cause: IR reflectance depends on fur material and distance to the illuminator, not on
visible-light color — a black cat close to the IR source can reflect more IR than a white
cat farther away. Brightness is still a usable signal (black cat close-up: ~94-201,
median 120; white cat: consistently ~172-173 regardless of distance in our samples), but
the gap is narrower than in daylight. Current night thresholds were widened to
`black_max: 135, white_min: 150` (buffer 135-150 = unknown) to cover this. This was found
and fixed *before* going live with real alerts — a good example of why Phase A validates
on real footage before Phase B ships.

**Practical implication:** night color thresholds are only as good as the range of
distances/angles they were calibrated on. If a future real visit gets misclassified,
re-run `classify_offline.py dump` on the new clip, check its median brightness, and widen
the buffer rather than shifting a single threshold.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in camera + Telegram credentials, see below
```

## Usage — Phase A (offline calibration/testing)

```
# Inspect brightness/detections per clip, dump sample crops + a reference frame
# for reading off ROI coordinates (used to (re)calibrate config.yaml):
python classify_offline.py dump

# Classify every clip in data/sample_clips/ and write results to data/events.db:
python classify_offline.py run --clear

# View the dashboard:
streamlit run dashboard.py
```

Calibrated values currently in `config.yaml`:
- `roi: [0.30, 0.05, 1.00, 0.65]` — litter box entrance area (camera-specific, see `data/calibration/reference_frame.jpg`)
- `min_dwell_seconds: 12`
- day thresholds: white ≥100, black ≤90 (brightness 0-255)
- night thresholds: white ≥150, black ≤135 (see recalibration note above)

## Usage — Phase B (live)

Requires `.env` filled in (see `.env.example`):
- `CAMERA_HOST`, `CAMERA_USER`, `CAMERA_PASS`, `ONVIF_PORT` — from Tapo app → C200 → gear
  icon → Advanced Settings → Camera Account.
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — from @BotFather and
  `https://api.telegram.org/bot<token>/getUpdates` after messaging the bot once.

```
python main.py
```

Runs forever: subscribes to ONVIF motion events, opens RTSP only while motion is active,
classifies each event, logs it to `data/events.db`, and sends a Telegram message for
confirmed visits (`🐾 ... сходив в туалет ...`) and passbys (`🐾 ... пройшов повз ...`).

To survive the Mac going to sleep, run under `caffeinate`:
```
caffeinate -is python3 main.py
```

Confirmed live: RTSP connects (with URL-encoded credentials — passwords with `#`/special
chars broke the URL before this was added), ONVIF PullPoint delivers repeated `True` motion
states followed by `False` on motion end (harmless duplicates, guarded by a `capturing`
flag in `main.py`), and a real human-in-frame event was correctly classified as `human`
with no Telegram alert sent.

## Known limitations (v1, not fixed)

- Multiple cats in frame simultaneously: picks the higher-confidence detection only.
- No clip/thumbnail retention policy.
- No process supervision beyond `caffeinate` + manual restart.
- ROI and brightness thresholds are specific to this camera's exact mounting position —
  moving the camera requires recalibration (`classify_offline.py dump` + edit `config.yaml`).
