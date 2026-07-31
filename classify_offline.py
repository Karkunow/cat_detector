"""Phase A driver: runs detect+classify+storage over a folder of pre-existing clips.

Two modes:
  dump  - decode clips, print per-frame brightness/detection info and save sample
          crops + a reference frame, to help pick ROI coords and thresholds.
  run   - full pipeline: classify each clip as one event and write it to events.db.

Usage:
  python classify_offline.py dump [--clips-dir data/sample_clips]
  python classify_offline.py run  [--clips-dir data/sample_clips] [--clear]
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import yaml

import storage
from classify import best_detection, classify_frames, crop_bbox, is_day_frame, mean_brightness
from detect import detect

CLIPS_DIR = Path(__file__).parent / "data" / "sample_clips"
CALIBRATION_DIR = Path(__file__).parent / "data" / "calibration"
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def iter_clip_frames(path: Path, sample_fps: float):
    cap = cv2.VideoCapture(str(path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = max(1, round(orig_fps / sample_fps))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % skip == 0:
            yield frame_idx / orig_fps, frame
        frame_idx += 1
    cap.release()


def dump(clips_dir: Path, config: dict) -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    clips = sorted(clips_dir.glob("*.MP4")) + sorted(clips_dir.glob("*.mp4"))
    if not clips:
        print(f"No clips found in {clips_dir}")
        return

    saved_reference = False
    for clip_path in clips:
        print(f"\n=== {clip_path.name} ===")
        for t, frame in iter_clip_frames(clip_path, config["sample_fps"]):
            day = is_day_frame(frame)
            dets = detect(frame, config["yolo_confidence"])
            cat_det = best_detection(dets, "cat")

            if not saved_reference:
                out = CALIBRATION_DIR / "reference_frame.jpg"
                cv2.imwrite(str(out), frame)
                h, w = frame.shape[:2]
                print(f"Saved reference frame {out} ({w}x{h}) for reading off ROI coords.")
                saved_reference = True

            if cat_det:
                crop = crop_bbox(frame, cat_det.bbox)
                brightness = mean_brightness(crop)
                crop_path = CALIBRATION_DIR / f"{clip_path.stem}_t{t:.1f}_cat.jpg"
                cv2.imwrite(str(crop_path), crop)
                print(
                    f"  t={t:5.1f}s day={day!s:5} cat conf={cat_det.confidence:.2f} "
                    f"brightness={brightness:6.1f} bbox={cat_det.bbox} -> {crop_path.name}"
                )
            else:
                print(f"  t={t:5.1f}s day={day!s:5} (no cat detected)")


def classify_clip(clip_path: Path, config: dict) -> dict:
    return classify_frames(iter_clip_frames(clip_path, config["sample_fps"]), config)


def run(clips_dir: Path, config: dict, clear: bool) -> None:
    conn = storage.init_db()
    if clear:
        storage.clear_events(conn)

    clips = sorted(clips_dir.glob("*.MP4")) + sorted(clips_dir.glob("*.mp4"))
    if not clips:
        print(f"No clips found in {clips_dir}")
        return

    for clip_path in clips:
        result = classify_clip(clip_path, config)
        timestamp = datetime.fromtimestamp(clip_path.stat().st_mtime).isoformat()
        storage.insert_event(
            conn,
            timestamp=timestamp,
            cat=result["cat"],
            is_day=result["is_day"],
            confidence=result["confidence"],
            dwell_seconds=result["dwell_seconds"],
            source_clip=clip_path.name,
        )
        print(
            f"{clip_path.name:30} -> cat={result['cat']:12} day={result['is_day']!s:5} "
            f"dwell={result['dwell_seconds']:5.1f}s conf={result['confidence']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    dump_p = sub.add_parser("dump")
    dump_p.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)

    run_p = sub.add_parser("run")
    run_p.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    run_p.add_argument("--clear", action="store_true")

    args = parser.parse_args()
    config = load_config()

    if args.mode == "dump":
        dump(args.clips_dir, config)
    elif args.mode == "run":
        run(args.clips_dir, config, args.clear)


if __name__ == "__main__":
    main()
