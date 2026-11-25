#!/usr/bin/env python
"""Reproduce the full Calibration-Aware Safety Events pipeline in order.

Cross-platform version of run_all.sh: works on Windows, macOS and Linux,
including from the VS Code terminal. Uses the Python interpreter that runs
this script, so activate the project environment first.

Logs each step to outputs/logs/<step>.log, prints PASS/FAIL with timing,
and stops on the first failure.

Usage:
    python run_all.py --skip-perception          # reuse results/ detections and tracks
    python run_all.py                            # full run; ultralytics picks the device
    python run_all.py --device 0                 # full run on the first NVIDIA GPU
    python run_all.py --device mps               # full run on an Apple GPU
    python run_all.py --skip-perception --resume # keep the predicted-track sweep checkpoint
    python run_all.py --dry-run                  # print the steps without running them
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "outputs" / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-perception",
        action="store_true",
        help="reuse existing detection/track CSVs in results/ instead of rerunning YOLO and the trackers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep sweep checkpoints and only run missing trials (default recomputes with --force)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="passed to run_detection.py, e.g. cpu, mps or 0 (default: let ultralytics choose)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the steps without running them"
    )
    return parser.parse_args()


def build_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Return (step name, command) pairs in the same order as run_all.sh."""
    py = sys.executable
    steps: list[tuple[str, list[str]]] = [
        # 1. Tests
        ("pytest", [py, "-m", "pytest"]),
        # 2. Dataset / calibration checks
        ("inspect_dataset", [py, "scripts/inspect_dataset.py"]),
        ("check_annotations", [py, "scripts/check_annotations.py"]),
        ("check_calibration", [py, "scripts/check_calibration.py"]),
        ("check_homography", [py, "scripts/check_homography.py"]),
        # 3. Zones and answer keys (C1 + C6)
        ("show_zones_C1", [py, "scripts/show_zones.py", "--cam", "C1"]),
        ("make_gt_events_C1", [py, "scripts/make_gt_events.py", "--cam", "C1"]),
        ("show_zones_C6", [py, "scripts/show_zones.py", "--cam", "C6"]),
        ("make_gt_events_C6", [py, "scripts/make_gt_events.py", "--cam", "C6"]),
        ("check_scoring", [py, "scripts/check_scoring.py"]),
        # 4. Zero-noise sanity
        ("check_zero_noise", [py, "scripts/check_zero_noise.py"]),
        # 5. Calibration sweeps
        ("run_sweep", [py, "scripts/run_sweep.py"]),
        ("run_sweep_npoints_C1", [py, "scripts/run_sweep_npoints.py", "--cam", "C1"]),
        ("run_sweep_npoints_C6", [py, "scripts/run_sweep_npoints.py", "--cam", "C6"]),
        ("run_sweep_outlier", [py, "scripts/run_sweep_outlier.py"]),
        ("run_sweep_layouts", [py, "scripts/run_sweep_layouts.py"]),
        ("run_sweep_bump_C1", [py, "scripts/run_sweep_bump.py", "--cam", "C1"]),
        ("run_sweep_bump_C6", [py, "scripts/run_sweep_bump.py", "--cam", "C6"]),
        ("save_calib_points_C6", [py, "scripts/save_calib_points_C6.py"]),
        ("compare_C1_C6", [py, "scripts/compare_C1_C6.py"]),
    ]

    # 6. Perception (optional)
    if not args.skip_perception:
        detection = [py, "scripts/run_detection.py"]
        if args.device:
            detection += ["--device", args.device]
        steps += [
            ("run_detection", detection),
            ("run_tracking", [py, "scripts/run_tracking.py"]),
        ]

    # 7. Tracks -> events -> calibration on predicted tracks
    sweep_tracks = [py, "scripts/run_sweep_tracks.py"]
    if not args.resume:
        sweep_tracks.append("--force")
    steps += [
        ("check_tracks", [py, "scripts/check_tracks.py"]),
        ("diagnose_foot_points", [py, "scripts/diagnose_foot_points.py"]),
        ("run_events_on_tracks", [py, "scripts/run_events_on_tracks.py"]),
        ("run_sweep_tracks", sweep_tracks),
        # 8. Report figures
        ("make_report_figures", [py, "scripts/make_report_figures.py"]),
    ]
    return steps


def format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m {s % 60}s"


def print_summary(results: list[tuple[str, str, float]], total: float, failed: bool) -> None:
    print()
    print("================ summary ================")
    print(f"{'step':<28}  {'status':<6}  time")
    for name, status, seconds in results:
        print(f"{name:<28}  {status:<6}  {format_duration(seconds)}")
    print("----------------------------------------")
    print(f"{'TOTAL':<28}  {'FAIL' if failed else 'PASS':<6}  {format_duration(total)}")
    print(f"logs: {LOG_DIR.relative_to(ROOT)}")


def tail(path: Path, n: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def main() -> int:
    args = parse_args()
    steps = build_steps(args)

    print(f"python: {sys.executable}")
    print(f"device: {args.device or 'auto'}")
    print(f"skip-perception: {int(args.skip_perception)}")
    print(f"sweep-tracks: {'resume' if args.resume else 'force'}")
    print()

    if args.dry_run:
        for name, cmd in steps:
            print(f"{name:<28}  {' '.join(cmd)}")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Matplotlib needs a writable cache folder; use the system temp folder on any OS.
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl"))
    # Windows consoles default to a non-UTF-8 code page; force UTF-8 so scripts that
    # print symbols such as epsilon or arrows do not crash when output goes to a log.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str, float]] = []
    total_start = time.monotonic()

    for name, cmd in steps:
        log_path = LOG_DIR / f"{name}.log"
        print(f"-> {name} ...", flush=True)
        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            rc = subprocess.run(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        elapsed = time.monotonic() - start
        status = "PASS" if rc == 0 else "FAIL"
        results.append((name, status, elapsed))
        rel_log = log_path.relative_to(ROOT)
        print(f"  {status:<6}  {name}  ({format_duration(elapsed)})  log={rel_log}", flush=True)

        if rc != 0:
            print(f"  last lines of {rel_log}:", file=sys.stderr)
            print(tail(log_path), file=sys.stderr)
            print_summary(results, time.monotonic() - total_start, failed=True)
            return rc

    print_summary(results, time.monotonic() - total_start, failed=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
