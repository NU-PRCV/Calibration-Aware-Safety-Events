"""Tests for the pose-based foot points (ankle midpoint, box fallback).

These use small synthetic tables, so they need neither the dataset nor the
pose model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.tracking import (
    DETECTION_COLUMNS,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    TRACK_COLUMNS,
    attach_foot_points,
    foot_pixels,
    pose_foot_points,
    run_tracker,
)

BOX = [100.0, 100.0, 150.0, 300.0]


def _keypoints(left: tuple[float, float], right: tuple[float, float]) -> np.ndarray:
    points = np.zeros((1, 17, 2))
    points[0, LEFT_ANKLE] = left
    points[0, RIGHT_ANKLE] = right
    return points


def _conf(left: float, right: float) -> np.ndarray:
    conf = np.full((1, 17), 0.9)
    conf[0, LEFT_ANKLE] = left
    conf[0, RIGHT_ANKLE] = right
    return conf


def test_both_ankles_visible_gives_their_midpoint() -> None:
    feet, source = pose_foot_points([BOX], _keypoints((110, 280), (130, 290)), _conf(0.8, 0.7))
    np.testing.assert_allclose(feet, [[120.0, 285.0]])
    assert list(source) == ["ankles"]


def test_one_hidden_ankle_falls_back_to_box_bottom_center() -> None:
    feet, source = pose_foot_points([BOX], _keypoints((110, 280), (0, 0)), _conf(0.8, 0.1))
    np.testing.assert_allclose(feet, [[125.0, 300.0]])
    assert list(source) == ["box"]


def test_missing_keypoints_fall_back_to_box() -> None:
    feet, source = pose_foot_points([BOX], None, None)
    np.testing.assert_allclose(feet, [[125.0, 300.0]])
    assert list(source) == ["box"]


def test_no_boxes_gives_empty_output() -> None:
    feet, source = pose_foot_points(np.empty((0, 4)), np.empty((0, 17, 2)), np.empty((0, 17)))
    assert feet.shape == (0, 2)
    assert len(source) == 0


def test_ankle_shift_moves_midpoint_down_by_box_height_fraction() -> None:
    feet, _ = pose_foot_points(
        [BOX], _keypoints((110, 280), (130, 290)), _conf(0.8, 0.8), ankle_shift=0.05
    )
    # Box height 200 px, so 5% is 10 px lower.
    np.testing.assert_allclose(feet, [[120.0, 295.0]])


def test_foot_pixels_without_foot_columns_is_bottom_center() -> None:
    table = pd.DataFrame([dict(zip(["xmin", "ymin", "xmax", "ymax"], BOX))])
    np.testing.assert_allclose(foot_pixels(table), [[125.0, 300.0]])


def test_foot_pixels_uses_foot_columns_and_falls_back_on_nan() -> None:
    table = pd.DataFrame(
        [
            {"xmin": 100, "ymin": 100, "xmax": 150, "ymax": 300, "foot_u": 120, "foot_v": 285},
            {"xmin": 200, "ymin": 100, "xmax": 260, "ymax": 320, "foot_u": np.nan,
             "foot_v": np.nan},
        ]
    )
    np.testing.assert_allclose(foot_pixels(table), [[120.0, 285.0], [230.0, 320.0]])


def _detections() -> pd.DataFrame:
    rows = [
        (0, 100, 100, 150, 300, 0.9, 120.0, 285.0, "ankles"),
        (0, 400, 120, 450, 330, 0.8, 425.0, 330.0, "box"),
        (1, 105, 102, 155, 302, 0.9, 126.0, 288.0, "ankles"),
    ]
    table = pd.DataFrame(
        [
            {
                "frame_idx": frame,
                "frame_number": 5 * frame,
                "time_sec": frame / 2.0,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "score": score,
                "foot_u": u,
                "foot_v": v,
                "foot_source": src,
            }
            for frame, xmin, ymin, xmax, ymax, score, u, v, src in rows
        ]
    )
    return table


class _EchoTracker:
    """Returns every detection box as its own track, like ByteTrack's output boxes."""

    def update(self, boxes, scores, image_path):
        pairs = enumerate(zip(boxes, scores))
        return [(index + 1, np.asarray(box), float(score)) for index, (box, score) in pairs]


def test_attach_foot_points_links_track_rows_to_their_detection() -> None:
    detections = _detections()
    frames = pd.DataFrame(
        {
            "frame_idx": [0, 1],
            "frame_number": [0, 5],
            "time_sec": [0.0, 0.5],
            "image_path": ["", ""],
        }
    )
    tracks = run_tracker(detections[DETECTION_COLUMNS], frames, _EchoTracker())
    assert list(tracks.columns) == TRACK_COLUMNS
    linked = attach_foot_points(tracks, detections)
    np.testing.assert_allclose(
        linked[["foot_u", "foot_v"]].to_numpy(), [[120.0, 285.0], [425.0, 330.0], [126.0, 288.0]]
    )
    assert list(linked["foot_source"]) == ["ankles", "box", "ankles"]


def test_attach_foot_points_leaves_unlinked_rows_on_the_box() -> None:
    detections = _detections()
    track = pd.DataFrame(
        [
            {
                "frame_idx": 0,
                "frame_number": 0,
                "time_sec": 0.0,
                "track_id": 7,
                "xmin": 700.0,
                "ymin": 100.0,
                "xmax": 760.0,
                "ymax": 320.0,
                "score": 0.5,
            }
        ]
    )
    linked = attach_foot_points(track, detections)
    assert linked["foot_u"].isna().all()
    assert list(linked["foot_source"]) == ["box"]
    np.testing.assert_allclose(foot_pixels(linked), [[730.0, 320.0]])


def test_attach_foot_points_without_pose_columns_is_a_no_op() -> None:
    detections = _detections()[DETECTION_COLUMNS]
    tracks = detections.assign(track_id=[1, 2, 1])[TRACK_COLUMNS]
    linked = attach_foot_points(tracks, detections)
    assert linked["foot_u"].isna().all()
    np.testing.assert_allclose(foot_pixels(linked), foot_pixels(tracks))
