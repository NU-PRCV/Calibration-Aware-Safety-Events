#!/usr/bin/env python3
"""Rebuild every final report figure and table from saved CSVs only.

Sized for a NeurIPS single-column paper (full-width 5.5 in, half-width 2.7 in).
Overall figure titles/suptitles are omitted; multi-panel figures keep short panel
titles (Zone / Dwell / Proximity). Missing inputs are reported and skipped.

Clears outputs/report/ at the start (that folder only), then regenerates everything.

Inputs:  CSVs under outputs/ (and results/tracks/tracking_summary_C1.csv).
Outputs: PDF/PNG figures and CSV/LaTeX tables under outputs/report/.

Example:
    python scripts/make_report_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import project_root

# --- shared style -----------------------------------------------------------------

EVENT_TYPES = ("zone", "dwell", "proximity")
EVENT_TITLES = {"zone": "Zone", "dwell": "Dwell", "proximity": "Proximity"}
EVENT_COLORS = {
    "zone": "#e07a2f",
    "dwell": "#2e9b4f",
    "proximity": "#3d6fd8",
}
ZONE_IDS = ("near", "medium", "far")
ZONE_LABELS = {"near": "Near", "medium": "Medium", "far": "Far"}
ZONE_COLORS = {
    "near": "#f2731a",
    "medium": "#26994d",
    "far": "#3366d9",
}
LAYOUTS_NPOINTS = ("n4", "n6", "n8", "n12")
LAYOUT_COLORS = {
    "n4": "#e07a2f",
    "n6": "#2e9b4f",
    "n8": "#3d6fd8",
    "n12": "#c43c6a",
}
LAYOUT_MARKERS = {"n4": "o", "n6": "s", "n8": "^", "n12": "D"}
LAYOUT_LABELS = {"n4": "4 pts", "n6": "6 pts", "n8": "8 pts", "n12": "12 pts"}
METERS_LAYOUTS = ("n4", "n12")
METERS_STYLES = {"n4": "-", "n12": "--"}
METERS_MARKERS = {"n4": "o", "n12": "s"}
GUIDELINE_M = {"zone": 0.25, "dwell": 0.06, "proximity": 0.4}

PLACEMENT_FAMILIES = ("wide", "narrow", "near_only", "far_only")
PLACEMENT_FAMILY_LABELS = {
    "wide": "wide",
    "narrow": "narrow",
    "near_only": "near-only",
    "far_only": "far-only",
}
PLACEMENT_LAYOUTS = (
    "wide",
    "wide_n12",
    "narrow",
    "narrow_n12",
    "near_only",
    "near_only_n12",
    "far_only",
    "far_only_n12",
)
PLACEMENT_COLORS = {
    "wide": "#e07a2f",
    "wide_n12": "#e07a2f",
    "narrow": "#2e9b4f",
    "narrow_n12": "#2e9b4f",
    "near_only": "#3d6fd8",
    "near_only_n12": "#3d6fd8",
    "far_only": "#c43c6a",
    "far_only_n12": "#c43c6a",
}
PLACEMENT_STYLES = {
    "wide": "-",
    "wide_n12": "--",
    "narrow": "-",
    "narrow_n12": "--",
    "near_only": "-",
    "near_only_n12": "--",
    "far_only": "-",
    "far_only_n12": "--",
}
PLACEMENT_LABELS = {
    "wide": "wide (4)",
    "wide_n12": "wide (12)",
    "narrow": "narrow (4)",
    "narrow_n12": "narrow (12)",
    "near_only": "near-only (4)",
    "near_only_n12": "near-only (12)",
    "far_only": "far-only (4)",
    "far_only_n12": "far-only (12)",
}

BUMP_AXES = ("pitch", "yaw", "roll")
BUMP_COLORS = {"pitch": "#e07a2f", "yaw": "#2e9b4f", "roll": "#3d6fd8"}
BUMP_TABLE_ANGLES = (0.25, 1.0)

ESTIMATORS = ("dlt", "ransac", "cv2_ransac")
ESTIMATOR_COLORS = {"dlt": "#e07a2f", "ransac": "#2e9b4f", "cv2_ransac": "#3d6fd8"}
ESTIMATOR_LABELS = {
    "dlt": "Least-squares DLT",
    "ransac": "Exhaustive RANSAC",
    "cv2_ransac": "OpenCV RANSAC",
}
OUTLIER_TABLE_MAG = 50.0

MODEL_ORDER = (
    "oracle_gt_boxes",
    "yolo11n_bytetrack",
    "yolo11n_deepsort",
    "yolo11x_bytetrack",
    "yolo11x_deepsort",
)
MODEL_LABELS = {
    "oracle_gt_boxes": "Oracle (GT boxes)",
    "yolo11n_bytetrack": "YOLO11n + ByteTrack",
    "yolo11n_deepsort": "YOLO11n + DeepSORT",
    "yolo11x_bytetrack": "YOLO11x + ByteTrack",
    "yolo11x_deepsort": "YOLO11x + DeepSORT",
}
MODEL_COLORS = {
    "oracle_gt_boxes": "#222222",
    "yolo11n_bytetrack": "#e07a2f",
    "yolo11n_deepsort": "#2e9b4f",
    "yolo11x_bytetrack": "#3d6fd8",
    "yolo11x_deepsort": "#c43c6a",
}
GT_REF_COLOR = "0.45"
BEST_TRACKER = "yolo11x_deepsort"
PERFECT_TRACKS_LABEL = "Perfect tracks"
CAM_COLORS = {"C1": "#e07a2f", "C6": "#3d6fd8"}

# NeurIPS single-column widths (inches).
FIG_HALF = (2.7, 2.15)
FIG_FULL = (5.5, 2.15)
FIG_FULL_TALL = (5.5, 2.45)
FONT_SIZE = 8
LABEL_SIZE = 9
LEGEND_SIZE = 7
TICK_SIZE = 8
DPI = 300


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": FONT_SIZE,
            "axes.titlesize": LABEL_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _paths() -> dict[str, Path]:
    root = project_root()
    out = root / "outputs" / "report"
    return {
        "out": out,
        "gt_sweep": root / "outputs" / "sweep_C1_gt_tracks.csv",
        "npoints": root / "outputs" / "sweep_C1_npoints.csv",
        "npoints_c6": root / "outputs" / "sweep_C6_npoints.csv",
        "pred": root / "outputs" / "sweep_C1_pred_tracks.csv",
        "tolerance": root / "outputs" / "tolerance_npoints_C1.csv",
        "meters": root / "outputs" / "tolerance_meters_C1_C6.csv",
        "outlier": root / "outputs" / "sweep_C1_outlier.csv",
        "layouts": root / "outputs" / "sweep_C1_layouts.csv",
        "bump": root / "outputs" / "sweep_C1_bump.csv",
        "events": root / "outputs" / "events_on_tracks_C1.csv",
        "tracking": root / "results" / "tracks" / "tracking_summary_C1.csv",
    }


def _clear_report_dir(out: Path) -> int:
    """Delete every file in outputs/report/ (files only; never touch other outputs/)."""
    out.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in sorted(out.iterdir()):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def _load(path: Path, missing: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        rel = str(path.relative_to(project_root()))
        missing.append(rel)
        print(f"MISSING  {rel}  — skipping dependents")
        return None
    return pd.read_csv(path)


def _save_fig(fig: plt.Figure, stem: Path, created: list[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
        created.append(str(path.relative_to(project_root())))
        print(f"Saved {path.relative_to(project_root())}")
    plt.close(fig)


def _eps_levels(frame: pd.DataFrame) -> list[float]:
    return [float(x) for x in sorted(frame["eps_px"].unique())]


def _curve(
    frame: pd.DataFrame,
    column: str,
    *,
    mask: pd.Series | None = None,
    x_col: str = "eps_px",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, mean, p5, p95 for one column."""
    data = frame if mask is None else frame.loc[mask]
    xs = [float(x) for x in sorted(data[x_col].unique())]
    means, lows, highs = [], [], []
    for value in xs:
        vals = data.loc[np.isclose(data[x_col], value), column].to_numpy(dtype=float)
        means.append(float(np.mean(vals)))
        lows.append(float(np.percentile(vals, 5)))
        highs.append(float(np.percentile(vals, 95)))
    return (
        np.asarray(xs, dtype=float),
        np.asarray(means, dtype=float),
        np.asarray(lows, dtype=float),
        np.asarray(highs, dtype=float),
    )


def _f1_ylim(*low_arrays: np.ndarray) -> tuple[float, float]:
    """Lower bound that keeps 5th-percentile bands visible (~0.35 when possible)."""
    lows = [float(np.min(arr)) for arr in low_arrays if len(arr)]
    if not lows:
        return 0.35, 1.02
    ymin = min(0.35, max(0.0, min(lows) - 0.05))
    return ymin, 1.02


def _style_eps_axis(ax: plt.Axes, eps: np.ndarray, ylabel: str) -> None:
    ax.set_xlabel("calibration noise (px)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(eps))
    ax.set_xlim(float(eps.min()) - 0.4, float(eps.max()) + 0.4)


def _panel_title(ax: plt.Axes, event_type: str) -> None:
    ax.set_title(EVENT_TITLES[event_type], pad=3)


def _legend_above(fig: plt.Figure, axes: Any, ncol: int) -> None:
    """Compact legend just above the panels (no overall title / large gap)."""
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model))


def _write_csv(frame: pd.DataFrame, path: Path, created: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    created.append(str(path.relative_to(project_root())))
    print(f"Saved {path.relative_to(project_root())}")


def _write_tex(lines: list[str], path: Path, created: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    created.append(str(path.relative_to(project_root())))
    print(f"Saved {path.relative_to(project_root())}")


def _layout_npoints_name(family: str, n_points: int) -> str:
    return family if n_points == 4 else f"{family}_n12"


def _mean_at(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    return float(frame.loc[mask, column].mean())


def _print_layouts_mean_median(layouts: pd.DataFrame) -> None:
    """Print mean and median position error at eps 5 and 10 for 4- and 12-point layouts."""
    print("\nLayouts sweep — mean vs median position error (m)")
    print(
        f"{'layout':>10} {'n':>3} {'eps':>4}  {'mean':>8}  {'median':>8}"
    )
    for eps in (5.0, 10.0):
        for family in PLACEMENT_FAMILIES:
            for n_points in (4, 12):
                name = _layout_npoints_name(family, n_points)
                mask = (layouts["layout"] == name) & np.isclose(layouts["eps_px"], eps)
                if not mask.any():
                    continue
                mean_err = float(layouts.loc[mask, "pos_mean_m"].mean())
                if "pos_median_m" in layouts.columns:
                    median_err = float(layouts.loc[mask, "pos_median_m"].mean())
                else:
                    median_err = float(layouts.loc[mask, "pos_mean_m"].median())
                print(
                    f"{PLACEMENT_FAMILY_LABELS[family]:>10} {n_points:>3} {eps:4.0f}  "
                    f"{mean_err:8.3f}  {median_err:8.3f}"
                )


# --- figures ----------------------------------------------------------------------


def fig_f1_vs_eps(gt: pd.DataFrame, out: Path, created: list[str]) -> None:
    fig, ax = plt.subplots(figsize=FIG_HALF)
    lows: list[np.ndarray] = []
    eps = None
    for event_type in EVENT_TYPES:
        x, mean, low, high = _curve(gt, f"f1_{event_type}")
        eps = x
        lows.append(low)
        color = EVENT_COLORS[event_type]
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=color, marker="o", markersize=3.5, label=event_type, linewidth=1.4)
    assert eps is not None
    _style_eps_axis(ax, eps, "F1")
    ax.set_ylim(*_f1_ylim(*lows))
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    _save_fig(fig, out / "fig_f1_vs_eps", created)


def fig_f1_by_zone(gt: pd.DataFrame, out: Path, created: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIG_FULL, sharey=True)
    lows: list[np.ndarray] = []
    eps = None
    for ax, event_type in zip(axes, ("zone", "dwell")):
        for zone_id in ZONE_IDS:
            x, mean, low, high = _curve(gt, f"f1_{event_type}_{zone_id}")
            eps = x
            lows.append(low)
            color = ZONE_COLORS[zone_id]
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
            ax.plot(
                x,
                mean,
                color=color,
                marker="o",
                markersize=3.5,
                label=ZONE_LABELS[zone_id],
                linewidth=1.4,
            )
        assert eps is not None
        _panel_title(ax, event_type)
        _style_eps_axis(ax, eps, "F1" if event_type == "zone" else "")
        ax.legend(frameon=False, loc="lower left", fontsize=LEGEND_SIZE)
    ymin, ymax = _f1_ylim(*lows)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    _save_fig(fig, out / "fig_f1_by_zone", created)


def fig_position_error(gt: pd.DataFrame, out: Path, created: list[str]) -> None:
    fig, ax = plt.subplots(figsize=FIG_HALF)
    scopes = [("overall", "pos_mean_m", "0.25")] + [
        (ZONE_LABELS[z], f"pos_mean_m_{z}", ZONE_COLORS[z]) for z in ZONE_IDS
    ]
    eps = None
    for label, column, color in scopes:
        x, mean, _low, _high = _curve(gt, column)
        eps = x
        ax.plot(x, mean, color=color, marker="o", markersize=3.5, label=label, linewidth=1.4)
    assert eps is not None
    _style_eps_axis(ax, eps, "mean position error (m)")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE - 0.5)
    fig.tight_layout()
    _save_fig(fig, out / "fig_position_error", created)


def fig_npoints(npoints: pd.DataFrame, out: Path, created: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIG_FULL, sharey=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for layout in LAYOUTS_NPOINTS:
            x, mean, _low, _high = _curve(
                npoints, f"f1_{event_type}", mask=npoints["layout"] == layout
            )
            ax.plot(
                x,
                mean,
                color=LAYOUT_COLORS[layout],
                marker=LAYOUT_MARKERS[layout],
                markersize=3.5,
                label=LAYOUT_LABELS[layout],
                linewidth=1.4,
            )
        _panel_title(ax, event_type)
        _style_eps_axis(ax, np.asarray(_eps_levels(npoints)), "F1" if event_type == "zone" else "")
        ax.set_ylim(0.55, 1.02)
    _legend_above(fig, axes, ncol=4)
    _save_fig(fig, out / "fig_npoints", created)


def fig_pred_tracks(
    pred: pd.DataFrame, gt: pd.DataFrame | None, out: Path, created: list[str]
) -> None:
    """Zone/dwell: anonymous F1; proximity: identity F1. Perfect-tracks dashed ref."""
    fig, axes = plt.subplots(1, 3, figsize=FIG_FULL_TALL, sharey=True)
    models = [m for m in MODEL_ORDER if m in set(pred["model"])]
    scoring_by_event = {
        "zone": "anonymous",
        "dwell": "anonymous",
        "proximity": "identity",
    }
    ylabel_by_event = {
        "zone": "anonymous F1",
        "dwell": "anonymous F1",
        "proximity": "identity F1",
    }
    for ax, event_type in zip(axes, EVENT_TYPES):
        scoring = scoring_by_event[event_type]
        for model in models:
            x, mean, _low, _high = _curve(
                pred, f"f1_{event_type}_{scoring}", mask=pred["model"] == model
            )
            ax.plot(
                x,
                mean,
                color=MODEL_COLORS.get(model, "0.3"),
                marker="o",
                markersize=3.0,
                label=_model_label(model),
                linewidth=1.3,
            )
        if gt is not None:
            x_gt, mean_gt, _l, _h = _curve(gt, f"f1_{event_type}")
            ax.plot(
                x_gt,
                mean_gt,
                color=GT_REF_COLOR,
                linestyle="--",
                linewidth=1.3,
                label=PERFECT_TRACKS_LABEL,
            )
        _panel_title(ax, event_type)
        _style_eps_axis(ax, np.asarray(_eps_levels(pred)), ylabel_by_event[event_type])
        ax.set_ylim(0.0, 1.05)
    _legend_above(fig, axes, ncol=3)
    _save_fig(fig, out / "fig_pred_tracks", created)


def fig_perception_vs_calibration(
    gt: pd.DataFrame,
    events: pd.DataFrame,
    out: Path,
    created: list[str],
) -> None:
    """Perfect tracks @ 10 px vs YOLO11x+DeepSORT @ perfect calibration."""
    calib = []
    for event_type in EVENT_TYPES:
        vals = gt.loc[np.isclose(gt["eps_px"], 10.0), f"f1_{event_type}"].to_numpy(dtype=float)
        calib.append(float(np.mean(vals)))
    row = events.loc[(events["model"] == BEST_TRACKER) & (events["scoring"] == "identity")]
    if row.empty:
        print(
            f"MISSING  identity row for {BEST_TRACKER} in events_on_tracks — "
            "skipping fig_perception_vs_calibration"
        )
        return
    row = row.iloc[0]
    track = [float(row[f"f1_{event_type}"]) for event_type in EVENT_TYPES]

    fig, ax = plt.subplots(figsize=FIG_HALF)
    x = np.arange(len(EVENT_TYPES))
    width = 0.36
    bars1 = ax.bar(
        x - width / 2,
        calib,
        width,
        label="Perfect tracks, 10 px calibration noise",
        color="#7a7a7a",
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        track,
        width,
        label="YOLO11x + DeepSORT, perfect calibration",
        color=MODEL_COLORS[BEST_TRACKER],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(EVENT_TYPES))
    ax.set_ylabel("identity F1")
    ax.set_ylim(0.0, 1.18)
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        fontsize=LEGEND_SIZE - 0.5,
    )
    fig.tight_layout()
    _save_fig(fig, out / "fig_perception_vs_calibration", created)


def fig_outlier(outlier: pd.DataFrame, out: Path, created: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIG_FULL, sharey=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for estimator in ESTIMATORS:
            x, mean, _low, _high = _curve(
                outlier,
                f"f1_{event_type}",
                mask=outlier["estimator"] == estimator,
                x_col="outlier_mag_px",
            )
            ax.plot(
                x,
                mean,
                color=ESTIMATOR_COLORS[estimator],
                marker="o",
                markersize=3.5,
                label=ESTIMATOR_LABELS[estimator],
                linewidth=1.4,
            )
        _panel_title(ax, event_type)
        ax.set_xlabel("outlier magnitude (px)")
        ax.set_ylabel("F1" if event_type == "zone" else "")
        ax.set_ylim(0.55, 1.02)
        ax.set_xticks(sorted(outlier["outlier_mag_px"].unique()))
    _legend_above(fig, axes, ncol=3)
    _save_fig(fig, out / "fig_outlier", created)


def fig_layouts(layouts: pd.DataFrame, out: Path, created: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIG_FULL, sharey=True)
    present = [name for name in PLACEMENT_LAYOUTS if name in set(layouts["layout"])]
    for ax, event_type in zip(axes, EVENT_TYPES):
        for layout in present:
            x, mean, _low, _high = _curve(
                layouts, f"f1_{event_type}", mask=layouts["layout"] == layout
            )
            ax.plot(
                x,
                mean,
                color=PLACEMENT_COLORS[layout],
                linestyle=PLACEMENT_STYLES[layout],
                marker="o",
                markersize=2.8,
                label=PLACEMENT_LABELS[layout],
                linewidth=1.3,
            )
        _panel_title(ax, event_type)
        _style_eps_axis(ax, np.asarray(_eps_levels(layouts)), "F1" if event_type == "zone" else "")
        ax.set_ylim(0.0, 1.05)
    _legend_above(fig, axes, ncol=4)
    _save_fig(fig, out / "fig_layouts", created)


def fig_bump(bump: pd.DataFrame, out: Path, created: list[str]) -> None:
    """Plot bump angles on evenly spaced ticks so small angles remain readable."""
    angles = [float(a) for a in sorted(bump["angle_deg"].unique())]
    x_pos = np.arange(len(angles), dtype=float)
    angle_to_x = {a: i for i, a in enumerate(angles)}
    tick_labels = [f"{a:g}" for a in angles]

    fig, axes = plt.subplots(1, 3, figsize=FIG_FULL, sharey=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for axis in BUMP_AXES:
            xs_raw, mean, _low, _high = _curve(
                bump,
                f"f1_{event_type}",
                mask=bump["axis"] == axis,
                x_col="angle_deg",
            )
            xs = np.asarray([angle_to_x[float(a)] for a in xs_raw], dtype=float)
            ax.plot(
                xs,
                mean,
                color=BUMP_COLORS[axis],
                marker="o",
                markersize=3.5,
                label=axis,
                linewidth=1.4,
            )
        _panel_title(ax, event_type)
        ax.set_xlabel("bump angle (deg)")
        ax.set_ylabel("F1" if event_type == "zone" else "")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-0.3, len(angles) - 0.7)
    _legend_above(fig, axes, ncol=3)
    _save_fig(fig, out / "fig_bump", created)


def fig_meters(
    npoints_c1: pd.DataFrame,
    npoints_c6: pd.DataFrame,
    out: Path,
    created: list[str],
) -> None:
    """Mean F1 vs mean ground position error for C1/C6, n4 solid and n12 dashed."""
    fig, axes = plt.subplots(1, 3, figsize=FIG_FULL, sharey=True)
    frames = {"C1": npoints_c1, "C6": npoints_c6}
    for ax, event_type in zip(axes, EVENT_TYPES):
        for cam, frame in frames.items():
            for layout in METERS_LAYOUTS:
                mask = frame["layout"] == layout
                if not mask.any():
                    continue
                # Aggregate mean F1 and mean position error at each eps, then sort by pos.
                rows = []
                for eps in sorted(frame.loc[mask, "eps_px"].unique()):
                    sub = frame.loc[mask & np.isclose(frame["eps_px"], float(eps))]
                    rows.append(
                        (
                            float(sub["pos_mean_m"].mean()),
                            float(sub[f"f1_{event_type}"].mean()),
                        )
                    )
                rows.sort(key=lambda t: t[0])
                xs = [r[0] for r in rows]
                ys = [r[1] for r in rows]
                ax.plot(
                    xs,
                    ys,
                    color=CAM_COLORS[cam],
                    linestyle=METERS_STYLES[layout],
                    marker=METERS_MARKERS[layout],
                    markersize=3.0,
                    linewidth=1.4,
                    label=f"{cam} {layout}",
                )
        guide = GUIDELINE_M[event_type]
        ax.axvline(guide, color="0.45", linestyle=":", linewidth=1.0)
        _panel_title(ax, event_type)
        ax.set_xlabel("mean position error (m)")
        ax.set_ylabel("F1" if event_type == "zone" else "")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(left=0.0)
    _legend_above(fig, axes, ncol=4)
    _save_fig(fig, out / "fig_meters", created)


# --- tables -----------------------------------------------------------------------


def tab_sweep_gt(gt: pd.DataFrame, out: Path, created: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for eps in _eps_levels(gt):
        row: dict[str, Any] = {"eps_px": eps}
        for event_type in EVENT_TYPES:
            vals = gt.loc[np.isclose(gt["eps_px"], eps), f"f1_{event_type}"].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            p5 = float(np.percentile(vals, 5))
            p95 = float(np.percentile(vals, 95))
            row[f"f1_{event_type}"] = mean
            row[f"f1_{event_type}_p5"] = p5
            row[f"f1_{event_type}_p95"] = p95
            row[f"f1_{event_type}_cell"] = f"{mean:.3f} ({p5:.3f}–{p95:.3f})"
        pos = gt.loc[np.isclose(gt["eps_px"], eps), "pos_mean_m"].to_numpy(dtype=float)
        row["pos_mean_m"] = float(np.mean(pos))
        rows.append(row)
    table = pd.DataFrame(rows)
    _write_csv(
        table[
            ["eps_px"]
            + [f"f1_{et}" for et in EVENT_TYPES]
            + [f"f1_{et}_p5" for et in EVENT_TYPES]
            + [f"f1_{et}_p95" for et in EVENT_TYPES]
            + ["pos_mean_m"]
        ],
        out / "tab_sweep_gt.csv",
        created,
    )
    lines = [
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"$\varepsilon$ (px) & Zone F1 & Dwell F1 & Proximity F1 & Pos. err. (m) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['eps_px']:g} & "
            f"{row['f1_zone_cell']} & "
            f"{row['f1_dwell_cell']} & "
            f"{row['f1_proximity_cell']} & "
            f"{row['pos_mean_m']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_sweep_gt.tex", created)


def _fmt_eps_tol(value: float) -> str:
    """Format a tolerance eps: <$1 if 1 px fails, $\\geq$10 if 10 px still passes."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if float(value) < 1.0:
        return r"$<$1"
    if float(value) >= 10.0:
        return r"$\geq$10"
    return f"{float(value):.0f}"


def tab_tolerance(tol: pd.DataFrame, out: Path, created: list[str]) -> None:
    """Compact: rows = events, columns = 4/6/8/12 pts, cells = mean / p5."""
    sub = tol.loc[tol["table"] == "tolerance"].copy()
    if sub.empty:
        print("MISSING  tolerance rows in tolerance_npoints_C1.csv — skipping tab_tolerance")
        return
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for _, row in sub.iterrows():
        lookup[(str(row["event_type"]), str(row["layout"]))] = (
            float(row["max_eps_mean_f1_ge_0.95"]),
            float(row["max_eps_p5_f1_ge_0.95"]),
        )

    rows: list[dict[str, Any]] = []
    for event_type in EVENT_TYPES:
        row: dict[str, Any] = {"event_type": event_type}
        for layout in LAYOUTS_NPOINTS:
            mean_v, p5_v = lookup.get((event_type, layout), (float("nan"), float("nan")))
            row[f"eps_mean_{layout}"] = mean_v
            row[f"eps_p5_{layout}"] = p5_v
            row[f"cell_{layout}"] = f"{_fmt_eps_tol(mean_v)} / {_fmt_eps_tol(p5_v)}"
        rows.append(row)
    table = pd.DataFrame(rows)
    _write_csv(
        table[
            ["event_type"]
            + [f"eps_mean_{layout}" for layout in LAYOUTS_NPOINTS]
            + [f"eps_p5_{layout}" for layout in LAYOUTS_NPOINTS]
        ],
        out / "tab_tolerance.csv",
        created,
    )
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Calibration points} \\",
        r"\cmidrule(lr){2-5}",
        r"Event & 4 & 6 & 8 & 12 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{EVENT_TITLES[str(row['event_type'])]} & "
            f"{row['cell_n4']} & {row['cell_n6']} & "
            f"{row['cell_n8']} & {row['cell_n12']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_tolerance.tex", created)


def tab_meters(meters: pd.DataFrame, out: Path, created: list[str]) -> None:
    table = meters.copy()
    table["event_type"] = pd.Categorical(table["event_type"], list(EVENT_TYPES), ordered=True)
    table = table.sort_values("event_type").reset_index(drop=True)
    table["guideline_m"] = table["event_type"].map(GUIDELINE_M).astype(float)
    _write_csv(table, out / "tab_meters.csv", created)
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Event & C1 (m) & C6 (m) & Guideline (m) \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['event_type']} & "
            f"{row['C1_pos_m_f1_ge_0.95']:.3f} & "
            f"{row['C6_pos_m_f1_ge_0.95']:.3f} & "
            f"{row['guideline_m']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_meters.tex", created)


def tab_tracking(tracking: pd.DataFrame, out: Path, created: list[str]) -> None:
    cols = {
        "model": "model",
        "ground_det_precision": "precision",
        "ground_det_recall": "recall",
        "ground_id_switches": "id_switches",
        "ground_mota": "mota",
    }
    table = tracking[list(cols.keys())].rename(columns=cols).copy()
    order = [m for m in MODEL_ORDER if m != "oracle_gt_boxes"]
    table["model"] = pd.Categorical(table["model"], order, ordered=True)
    table = table.sort_values("model").reset_index(drop=True)
    display = table.copy()
    display["model"] = display["model"].map(_model_label)
    _write_csv(display, out / "tab_tracking.csv", created)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Precision & Recall & ID switches & MOTA \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{_model_label(str(row['model']))} & "
            f"{row['precision']:.3f} & {row['recall']:.3f} & "
            f"{int(row['id_switches'])} & {row['mota']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_tracking.tex", created)


def tab_events_tracks(events: pd.DataFrame, out: Path, created: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        row: dict[str, Any] = {"model": model}
        for scoring in ("identity", "anonymous"):
            match = events.loc[(events["model"] == model) & (events["scoring"] == scoring)]
            if match.empty:
                for et in EVENT_TYPES:
                    row[f"f1_{et}_{scoring}"] = np.nan
            else:
                r = match.iloc[0]
                for et in EVENT_TYPES:
                    row[f"f1_{et}_{scoring}"] = float(r[f"f1_{et}"])
        rows.append(row)
    table = pd.DataFrame(rows)
    display = table.copy()
    display["model"] = display["model"].map(_model_label)
    _write_csv(display, out / "tab_events_tracks.csv", created)
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Identity F1} & \multicolumn{3}{c}{Anonymous F1} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Model & Zone & Dwell & Prox. & Zone & Dwell & Prox. \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{_model_label(str(row['model']))} & "
            f"{_fmt(row['f1_zone_identity'])} & {_fmt(row['f1_dwell_identity'])} & "
            f"{_fmt(row['f1_proximity_identity'])} & "
            f"{_fmt(row['f1_zone_anonymous'])} & {_fmt(row['f1_dwell_anonymous'])} & "
            f"{_fmt(row['f1_proximity_anonymous'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_events_tracks.tex", created)


def tab_relative_f1(pred: pd.DataFrame, out: Path, created: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        if model not in set(pred["model"]):
            continue
        m = pred.loc[pred["model"] == model]
        row: dict[str, Any] = {"model": model}
        for event_type, scoring in (
            ("zone", "anonymous"),
            ("dwell", "anonymous"),
            ("proximity", "identity"),
        ):
            col = f"f1_{event_type}_{scoring}"
            base = float(m.loc[np.isclose(m["eps_px"], 0.0), col].mean())
            at10 = float(m.loc[np.isclose(m["eps_px"], 10.0), col].mean())
            rel = float("nan") if not np.isfinite(base) or base == 0.0 else at10 / base
            row[f"rel_f1_{event_type}"] = rel
            row[f"f1_{event_type}_eps0"] = base
            row[f"f1_{event_type}_eps10"] = at10
        rows.append(row)
    table = pd.DataFrame(rows)
    display = table.copy()
    display["model"] = display["model"].map(_model_label)
    _write_csv(display, out / "tab_relative_f1.csv", created)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Zone (anon.) & Dwell (anon.) & Prox. (identity) \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{_model_label(str(row['model']))} & "
            f"{_fmt(row['rel_f1_zone'])} & {_fmt(row['rel_f1_dwell'])} & "
            f"{_fmt(row['rel_f1_proximity'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_relative_f1.tex", created)


def tab_outlier(outlier: pd.DataFrame, out: Path, created: list[str]) -> None:
    """One row per estimator at 50 px outlier magnitude, including detection rate."""
    rows: list[dict[str, Any]] = []
    for estimator in ESTIMATORS:
        mask = (outlier["estimator"] == estimator) & np.isclose(
            outlier["outlier_mag_px"], OUTLIER_TABLE_MAG
        )
        if not mask.any():
            continue
        row: dict[str, Any] = {
            "estimator": estimator,
            "outlier_mag_px": OUTLIER_TABLE_MAG,
        }
        for et in EVENT_TYPES:
            row[f"f1_{et}"] = float(outlier.loc[mask, f"f1_{et}"].mean())
        row["pos_mean_m"] = float(outlier.loc[mask, "pos_mean_m"].mean())
        if "outlier_flagged" in outlier.columns:
            row["detect_rate"] = float(outlier.loc[mask, "outlier_flagged"].mean())
        else:
            row["detect_rate"] = np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table["estimator_label"] = table["estimator"].map(ESTIMATOR_LABELS)
    _write_csv(table, out / "tab_outlier.csv", created)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Estimator & Zone F1 & Dwell F1 & Prox. F1 & Pos. err. (m) & Detect rate \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        # Least-squares DLT never flags outliers — show -- rather than 0.00.
        if str(row["estimator"]) == "dlt":
            detect_cell = "--"
        else:
            detect_cell = _fmt(float(row["detect_rate"]), 2)
        lines.append(
            f"{ESTIMATOR_LABELS[str(row['estimator'])]} & "
            f"{row['f1_zone']:.3f} & {row['f1_dwell']:.3f} & {row['f1_proximity']:.3f} & "
            f"{row['pos_mean_m']:.3f} & {detect_cell} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_outlier.tex", created)


def tab_layouts(layouts: pd.DataFrame, out: Path, created: list[str]) -> None:
    """eps=5 only; one row per family with 4-pt and 12-pt F1 / position error."""
    rows: list[dict[str, Any]] = []
    for family in PLACEMENT_FAMILIES:
        row: dict[str, Any] = {"layout": family, "eps_px": 5.0}
        for n_points, suffix in ((4, "n4"), (12, "n12")):
            name = _layout_npoints_name(family, n_points)
            mask = (layouts["layout"] == name) & np.isclose(layouts["eps_px"], 5.0)
            if not mask.any():
                for et in EVENT_TYPES:
                    row[f"f1_{et}_{suffix}"] = np.nan
                row[f"pos_mean_m_{suffix}"] = np.nan
                continue
            for et in EVENT_TYPES:
                row[f"f1_{et}_{suffix}"] = _mean_at(layouts, mask, f"f1_{et}")
            row[f"pos_mean_m_{suffix}"] = _mean_at(layouts, mask, "pos_mean_m")
        rows.append(row)
    table = pd.DataFrame(rows)
    table["layout_label"] = table["layout"].map(PLACEMENT_FAMILY_LABELS)
    _write_csv(table, out / "tab_layouts.csv", created)
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{4 pts} & \multicolumn{4}{c}{12 pts} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"Layout & Zone & Dwell & Prox. & Pos. & Zone & Dwell & Prox. & Pos. \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{PLACEMENT_FAMILY_LABELS[str(row['layout'])]} & "
            f"{_fmt(row['f1_zone_n4'])} & {_fmt(row['f1_dwell_n4'])} & "
            f"{_fmt(row['f1_proximity_n4'])} & {_fmt(row['pos_mean_m_n4'])} & "
            f"{_fmt(row['f1_zone_n12'])} & {_fmt(row['f1_dwell_n12'])} & "
            f"{_fmt(row['f1_proximity_n12'])} & {_fmt(row['pos_mean_m_n12'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_layouts.tex", created)


def tab_bump(bump: pd.DataFrame, out: Path, created: list[str]) -> None:
    """One row per axis; F1 and position error at 0.25° and 1°."""
    rows: list[dict[str, Any]] = []
    for axis in BUMP_AXES:
        row: dict[str, Any] = {"axis": axis}
        for angle in BUMP_TABLE_ANGLES:
            tag = f"{angle:g}".replace(".", "p")
            mask = (bump["axis"] == axis) & np.isclose(bump["angle_deg"], float(angle))
            for et in EVENT_TYPES:
                row[f"f1_{et}_{tag}"] = _mean_at(bump, mask, f"f1_{et}")
            row[f"pos_mean_m_{tag}"] = _mean_at(bump, mask, "pos_mean_m")
        rows.append(row)
    table = pd.DataFrame(rows)
    _write_csv(table, out / "tab_bump.csv", created)
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{$0.25^\circ$} & \multicolumn{4}{c}{$1^\circ$} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"Axis & Zone & Dwell & Prox. & Pos. & Zone & Dwell & Prox. & Pos. \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['axis']} & "
            f"{_fmt(row['f1_zone_0p25'])} & {_fmt(row['f1_dwell_0p25'])} & "
            f"{_fmt(row['f1_proximity_0p25'])} & {_fmt(row['pos_mean_m_0p25'])} & "
            f"{_fmt(row['f1_zone_1'])} & {_fmt(row['f1_dwell_1'])} & "
            f"{_fmt(row['f1_proximity_1'])} & {_fmt(row['pos_mean_m_1'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_bump.tex", created)


# --- main -------------------------------------------------------------------------


def main() -> None:
    _apply_style()
    paths = _paths()
    out = paths["out"]
    removed = _clear_report_dir(out)
    print(f"Removed {removed} old file(s) from outputs/report/")

    created: list[str] = []
    missing: list[str] = []

    gt = _load(paths["gt_sweep"], missing)
    npoints = _load(paths["npoints"], missing)
    npoints_c6 = _load(paths["npoints_c6"], missing)
    pred = _load(paths["pred"], missing)
    tol = _load(paths["tolerance"], missing)
    meters = _load(paths["meters"], missing)
    outlier = _load(paths["outlier"], missing)
    layouts = _load(paths["layouts"], missing)
    bump = _load(paths["bump"], missing)
    events = _load(paths["events"], missing)
    tracking = _load(paths["tracking"], missing)

    if gt is not None:
        fig_f1_vs_eps(gt, out, created)
        fig_f1_by_zone(gt, out, created)
        fig_position_error(gt, out, created)
        tab_sweep_gt(gt, out, created)
    if npoints is not None:
        fig_npoints(npoints, out, created)
    if pred is not None:
        fig_pred_tracks(pred, gt, out, created)
        tab_relative_f1(pred, out, created)
    if gt is not None and events is not None:
        fig_perception_vs_calibration(gt, events, out, created)
    if outlier is not None:
        fig_outlier(outlier, out, created)
        tab_outlier(outlier, out, created)
    if layouts is not None:
        fig_layouts(layouts, out, created)
        tab_layouts(layouts, out, created)
        _print_layouts_mean_median(layouts)
    if bump is not None:
        fig_bump(bump, out, created)
        tab_bump(bump, out, created)
    if npoints is not None and npoints_c6 is not None:
        fig_meters(npoints, npoints_c6, out, created)
    elif npoints is None or npoints_c6 is None:
        print("SKIP  fig_meters — need both sweep_C1_npoints.csv and sweep_C6_npoints.csv")
    if meters is not None:
        tab_meters(meters, out, created)
    if tol is not None:
        tab_tolerance(tol, out, created)
    if tracking is not None:
        tab_tracking(tracking, out, created)
    if events is not None:
        tab_events_tracks(events, out, created)

    print("\nCreated files:")
    for path in created:
        print(f"  {path}")
    print(f"Total created: {len(created)}")
    print("\nMissing input CSVs:")
    if missing:
        for path in missing:
            print(f"  {path}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
