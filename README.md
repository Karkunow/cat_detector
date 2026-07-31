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
| day_black_passby_3 | black cat passes behind the box | black_passby | ✅ |
| day_human | owner in frame, but a black cat also cameos in the background ~t=79-83s | black_passby | ✅ (see note below) |
| day_white_inside | white visit | white, dwell 57.8s | ✅ |
| day_white_passby_2 | white cat visible for ~0.3s only, backlit | unknown | ⚠️ missed — see sampling-rate note below |
| night_black_inside | brief pop-in, cat sat ~2s and left — not a real visit | pass_through | ✅ |
| night_black_inside_3 | black visit (long, cat close to entrance) | black, dwell 37.4s | ✅ |
| night_black_passby | no visit | black_passby | ✅ |
| night_human | no cat, only a person | unknown (no detections) | ✅ (no false alert) |
| night_white_inside | white visit | white, dwell 44.6s | ✅ |
| night_white_passby | no visit (cat brushes lens) | white_passby | ✅ |
| night_white_passby_2 | no visit (dim/far white cat) | black_passby | ⚠️ color wrong, but no false "visit" alert either way |

**No false "visit" alerts across the whole set.** Two known-and-accepted misses, both
low-stakes (see below) — no case has ever produced a wrong-cat *visit* notification, only
wrong/missed passby labels.

`_passby` labels are cats that were seen but didn't dwell in the ROI long enough to count
as a real visit — `main.py` sends a lighter "passed by" Telegram message for these, added
specifically to get faster live signal on whether the pipeline works end-to-end without
waiting for a full ≥12s box visit.

### Design change: cat presence always wins, we don't classify "human" at all

Originally, a "person"-dominant event was labeled `human` and suppressed any cat passby
notification. Dropped this after `day_black_passby_3` (a clip with a real, clearly visible
black cat for 15+ seconds) got classified as `human` — a piece of dark furniture at the
frame's left edge was intermittently misdetected as "person" at ~0.4-0.46 confidence
(right at our `yolo_confidence` threshold) and, summed over a long clip, outnumbered the
genuine cat detections.

Tried two spatial/confidence fixes (an ignore-zone over the furniture, a higher
person-only confidence threshold) — both rejected: tested against all sample clips, the
ignore-zone would have dropped real cat detections in 3 of them, and real human confidence
(`day_human.MP4`) spans 0.25-0.92, fully overlapping the furniture's false-positive range.
No clean separation existed.

The actual fix: **we don't care about "human" as a category at all** — we only care whether
a cat was there. `detect.py` no longer even asks YOLO for the `person` class; `classify.py`
just checks `cat_frames == 0` → `unknown`. This is simpler and more correct: it also fixed
`day_human.MP4`, which turned out to have a real black cat cameo in the background behind
the person (missed entirely under the old "human wins" logic) — confirmed visually, not a
false positive.

### Known miss: sub-second appearances at low confidence

`day_white_passby_2`: a white cat crosses behind the box, but is only unoccluded for ~0.3s,
backlit, at 0.28-0.46 confidence — below our `sample_fps: 2` sampling grid (samples landed
on either side of the ~0.3s window) and marginal even when checked frame-by-frame. Decided
not to chase this: the cat wasn't heading to the box anyway (entered right, exited left), so
it wasn't a real "visit" candidate regardless of detection. Revisit only if this pattern
starts affecting real box *visits*, not just passersby.

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

To survive the Mac going to sleep, run under `caffeinate`, and use `watchdog.sh` to
auto-restart if the process ever dies for a reason we haven't hardened against yet:
```
./watchdog.sh &   # polls every 30s, restarts `caffeinate -is python3 main.py` if it's gone
```

### Live hardening notes (found running this overnight against a real Tapo C210)

- **RTSP credentials must be URL-encoded** — a `#` in the password breaks the URL otherwise.
- **ONVIF PullPoint subscriptions need `nat_override=True`** — this camera's own
  `CreatePullPointSubscription` response advertises a wrong/internal port (observed: `:1028`)
  for its own subscription reference, inconsistent with `GetCapabilities` (which reports
  every service, Events included, at `:2020`). `nat_override` forces the client to keep
  talking to the host:port we actually connected on. This was the single biggest reliability
  fix — before it, the camera would get stuck refusing new subscriptions almost entirely.
- **`no_cache=True` and `adjust_time=True`** are cheap, recommended defensive settings for
  Tapo cameras (WSDL cache issues / clock-drift-triggered auth failures).
- **Keep PullMessages' poll timeout short** (`PT5S`) — this camera's lightweight network
  stack appears to unilaterally drop long-held long-poll HTTP connections.
- **Never let an exception escape the subscribe loop or the per-event handler** — an
  uncaught SOAP Fault during resubscribe once killed the whole process silently (and
  `caffeinate`, which was just wrapping it, exited too). Both are now wrapped so a single
  failure logs and retries instead of taking the whole thing down. `watchdog.sh` is the
  last line of defense for anything still missed.
- **RTSP connection setup takes ~1.8-3.2s** (measured directly). If the camera's ONVIF
  motion state flickers on/off in short bursts during one continuous cat presence (observed
  live — several "Motion started" → "Motion stopped, 0 frames captured" in a row during a
  single real visit), a fresh RTSP connection per event misses it entirely. `MotionCapture`
  now keeps RTSP open continuously in a background thread with a 2s rolling pre-roll buffer,
  so `start_event()`/`stop_event()` just mark frames already flowing — no per-event
  connection latency. Confirmed against a real missed black-cat clip that this was the cause.
- **Graceful shutdown matters**: `kill -9` leaves the ONVIF subscription dangling on the
  camera (it doesn't get `Unsubscribe()`d), which can contribute to subscription-limit
  issues. `main.py` handles `SIGTERM`/`SIGINT` to unsubscribe cleanly; prefer that over `-9`.

## Known limitations (v1, not fixed)

- Multiple cats in frame simultaneously: picks the higher-confidence detection only.
- No clip/thumbnail retention policy.
- No process supervision beyond `watchdog.sh` (simple poll-and-restart, no systemd/launchd).
- ROI and brightness thresholds are specific to this camera's exact mounting position —
  moving the camera requires recalibration (`classify_offline.py dump` + edit `config.yaml`).
- If the camera's ONVIF motion state fragments one real visit into several short on/off
  bursts, dwell time accumulates per logical event as currently coded — not yet observed to
  cause a missed real visit, but worth watching. Would need coalescing motion events
  separated by only a short gap into one logical event if it does.
