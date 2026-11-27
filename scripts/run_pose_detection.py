#!/usr/bin/env python3
"""Run a YOLO pose model and use the ankles as the foot point.

For every person box, the foot is the midpoint of the two ankle keypoints when
both are visible (confidence >= --ankle-conf), and the box bottom-center
otherwise. Writes the usual detection columns plus foot_u, foot_v and
foot_source ("ankles" or "box") to results/detections_pose/, so the main
box-based results in results/detections/ are never touched.

Then track and score the pose detections in their own folders:

    python scripts/run_pose_detection.py --cam C1
    python scripts/run_tracking.py --cam C1 \\
        --det-dir results/detections_pose --out-dir results/tracks_pose
    python scripts/check_tracks.py --cam C1 --tracks-dir results/tracks_pose
    python scripts/run_events_on_tracks.py --cam C1 --tracks-dir results/tracks_pose \\
        --out outputs/events_on_tracks_C1_pose.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, normalize_cam_id, project_root
from src.tracking import (
    ANKLE_MIN_CONF,
    DETECTION_COLUMNS,
    FOOT_COLUMNS,
    PERSON_CLASS_ID,
    annotated_frames,
    detection_rows,
    pose_foot_points,
    save_csv,
    validate_boxes,
)

# The pose versions of the two detectors in run_detection.py.
DEFAULT_MODELS = ("yolo11n-pose.pt", "yolo11x-pose.pt")
# Same detection settings as run_detection.py, so only the foot point changes.
DEFAULT_CONF = 0.1
DEFAULT_IMGSZ = 1280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cam", default="C1")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument(
        "--ankle-conf",
        type=float,
        default=ANKLE_MIN_CONF,
        help="both ankles need at least this keypoint confidence (default 0.5)",
    )
    parser.add_argument(
        "--ankle-shift",
        type=float,
        default=0.0,
        help="move the ankle midpoint down by this fraction of the box height, "
        "because ankles sit above the floor (default 0: plain midpoint)",
    )
    parser.add_argument(
        "--device", default=None, help="for example 0, cpu, or mps; default lets ultralytics choose"
    )
    parser.add_argument(
        "--out-dir", default=str(project_root() / "results" / "detections_pose")
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N frames, for a quick test"
    )
    return parser.parse_args()


def detect_pose(
    model_path: str, frames: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, float]:
    """Run one pose model on every frame. Returns the detections and ms per frame."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    # Warm-up so model loading does not count in the timing.
    model.predict(
        frames["image_path"].iloc[0],
        classes=[PERSON_CLASS_ID],
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    rows = []
    start = time.perf_counter()
    for frame in tqdm(frames.to_dict("records"), desc=Path(model_path).stem):
        result = model.predict(
            frame["image_path"],
            classes=[PERSON_CLASS_ID],
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        # boxes.xyxy and keypoints.xy are already in original image pixels.
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints
        points = None if keypoints is None else keypoints.xy.cpu().numpy()
        conf = None if keypoints is None or keypoints.conf is None else keypoints.conf.cpu().numpy()
        feet, source = pose_foot_points(
            boxes, points, conf, min_conf=args.ankle_conf, ankle_shift=args.ankle_shift
        )
        for row, (u, v), src in zip(detection_rows(frame, boxes, scores), feet, source):
            row.update({"foot_u": float(u), "foot_v": float(v), "foot_source": str(src)})
            rows.append(row)
    ms_per_frame = 1000.0 * (time.perf_counter() - start) / len(frames)
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS + FOOT_COLUMNS), ms_per_frame


def main() -> None:
    args = parse_args()
    cam_id = normalize_cam_id(args.cam)
    frames = annotated_frames(args.data_root, cam_id)
    if args.limit:
        frames = frames.head(args.limit)
    print(f"{cam_id}: {len(frames)} annotated frames")
    out_dir = Path(args.out_dir)

    summary = []
    for model_path in args.models:
        name = Path(model_path).stem
        detections, ms_per_frame = detect_pose(model_path, frames, args)
        validate_boxes(detections)
        path = save_csv(
            detections, DETECTION_COLUMNS + FOOT_COLUMNS, out_dir / f"det_{cam_id}_{name}.csv"
        )
        ankles = detections["foot_source"] == "ankles"
        ankle_share = float(ankles.mean()) if len(detections) else 0.0
        print(
            f"{name}: {len(detections)} boxes, {100.0 * ankle_share:.1f}% with both ankles, "
            f"{ms_per_frame:.1f} ms per frame -> {path.name}"
        )
        summary.append(
            {
                "model": name,
                "frames": len(frames),
                "boxes": len(detections),
                "ankle_share": ankle_share,
                "ms_per_frame": ms_per_frame,
                "conf": args.conf,
                "imgsz": args.imgsz,
                "ankle_conf": args.ankle_conf,
                "ankle_shift": args.ankle_shift,
            }
        )
    # Ends in _summary, so run_tracking.py skips it.
    summary_path = out_dir / f"det_{cam_id}_pose_summary.csv"
    table = pd.DataFrame(summary)
    if summary_path.is_file():
        previous = pd.read_csv(summary_path)
        table = pd.concat(
            [previous.loc[~previous["model"].isin(table["model"])], table], ignore_index=True
        )
    table.to_csv(summary_path, index=False, float_format="%.3f")
    print(f"Saved {summary_path.name}")


if __name__ == "__main__":
    main()
