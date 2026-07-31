# Cat litter-box detector

MVP for detecting which cat (white/black) used the litter box, from camera footage.
Full design rationale: see [PLAN.md](PLAN.md).

## Status: Phase A (offline classification) working

Built and calibrated against 11 real exported clips in `data/sample_clips/`.
Result of `python classify_offline.py run --clear`:

| clip | ground truth | predicted | correct? |
|---|---|---|---|
| day_black_inside | black visit | black, dwell 24.5s | ✅ |
| day_black_inside_2 | brief pop-in, cat sat ~1-2s and left — not a real visit | pass_through | ✅ |
| day_black_passby | no visit | unknown (no detections) | ✅ (no false alert) |
| day_human | human | human | ✅ |
| day_white_inside | white visit | white, dwell 57.8s | ✅ |
| night_black_inside | brief pop-in, cat sat ~2s and left — not a real visit | pass_through | ✅ |
| night_black_passby | no visit | pass_through | ✅ |
| night_human | human | unknown (no detections) | ✅ (no false alert) |
| night_white_inside | white visit | white, dwell 44.6s | ✅ |
| night_white_passby | no visit (cat brushes lens) | pass_through | ✅ |
| night_white_passby_2 | no visit | pass_through | ✅ |

**11/11 correct, 0 false alerts.**

(Initial read of `day_black_inside_2` / `night_black_inside` as "missed detections due to
poor visibility of black fur in the dark box" was wrong — confirmed with the user that
the cat was only there for 1-2s in both clips, i.e. `pass_through` is the right call, not
a detection failure. The apparent low YOLO recall on those two clips just reflects how
little time the cat was actually in frame.)

### Open question: is a black cat detectable at all inside the dark box interior?

We still don't have a confirmed example of the black cat sitting in the box long enough
(≥`min_dwell_seconds`) to know whether YOLO can track it through a full real visit — every
black-cat clip we have is either near the entrance (well detected, `day_black_inside`) or
a sub-2s pop-in. Contrast/shadow enhancement (gamma correction, CLAHE) was suggested as a
mitigation but not yet tested, since it's not clear it's needed. Worth revisiting once a
longer real black-cat visit clip is available — if recall turns out to be poor even with
good lighting, gamma/CLAHE preprocessing before YOLO is the next thing to try.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (Phase A)

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
- night thresholds: white ≥90, black ≤70

## Phase B (not started)

Live triggering via ONVIF motion events + on-demand RTSP, Telegram notifications.
See PLAN.md for the design. Requires enabling the Tapo Camera Account (RTSP/ONVIF)
and creating a Telegram bot before starting.
