#!/usr/bin/env python3
"""Print intrinsics, extrinsics, and ground-plane homographies for one camera.

Inputs:  calibrations under data/Wildtrack_dataset for --cam (default C1).
Outputs: printed K, R, t, H*, H*_img_to_world, and camera center (stdout).

Example:
    python scripts/check_calibration.py --cam C1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, normalize_cam_id
from src.geometry import ground_homography, invert_homography, load_camera_calibration


def _print_array(name: str, values: np.ndarray) -> None:
    print(f"{name}  shape {tuple(values.shape)}")
    print(np.array2string(values, precision=6, suppress_small=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam", default="C1", help="Camera id (default C1)")
    return parser.parse_args()


def main() -> None:
    """Load one camera and print K, R, t, both homographies, and the camera center."""
    args = _parse_args()
    cam_id = normalize_cam_id(args.cam)
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    K, R, t = load_camera_calibration(DATA_ROOT, cam_id)
    H_world_to_img = ground_homography(K, R, t)
    H_img_to_world = invert_homography(H_world_to_img)
    # Camera center in the world frame: X_camera = R X_world + t = 0.
    camera_position = -R.T @ t
    height_m = float(camera_position[2, 0])

    print(f"Camera: {cam_id}")
    _print_array("K (pixels)", K)
    _print_array("R", R)
    _print_array("t (meters)", t)
    _print_array("H_world_to_img", H_world_to_img)
    _print_array("H_img_to_world", H_img_to_world)
    _print_array("Camera position C = -R^T t (meters)", camera_position)
    print(f"Camera height (Z of C): {height_m:.3f} m")


if __name__ == "__main__":
    main()
