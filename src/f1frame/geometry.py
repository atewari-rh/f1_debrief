"""Corners from the curvature of a reference polyline.

The public position stream is a timing map, sampled about 15 m apart and
normalized onto one path. Radius here is in those meters. It is coarser than
a surveyed corner, and a short chicane can merge or split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1frame.types import F1FrameError, ReferenceLine


@dataclass(frozen=True)
class Corner:
    """One bend on the reference line, numbered from the start of the lap."""

    index: int
    entry_s: float
    apex_s: float
    exit_s: float
    direction: str
    min_radius_m: float


def extract_corners(
    reference: ReferenceLine,
    *,
    min_radius_m: float = 180.0,
    min_length_m: float = 30.0,
    merge_gap_m: float = 25.0,
    smooth: int = 5,
) -> tuple[Corner, ...]:
    """Group the reference line into corners.

    A vertex is in a corner when the smoothed radius is tighter than
    ``min_radius_m``. ``direction`` is ``left`` when signed curvature is
    positive, which is a left turn along the direction of the polyline.
    """

    if min_radius_m <= 0 or min_length_m <= 0 or merge_gap_m < 0:
        raise F1FrameError("corner thresholds must be positive")
    if smooth < 1:
        raise F1FrameError("smooth must be at least 1")
    kappa = _signed_curvature(reference.x, reference.y, smooth)
    labels = np.zeros(kappa.shape, dtype=np.int8)
    labels[kappa >= 1.0 / min_radius_m] = 1
    labels[kappa <= -1.0 / min_radius_m] = -1
    labels[:2] = 0
    labels[-2:] = 0

    runs = _merge_runs(_runs(labels), reference.s_m, merge_gap_m)
    corners: list[Corner] = []
    for start, stop, sign in runs:
        length = float(reference.s_m[stop - 1] - reference.s_m[start])
        if length < min_length_m:
            continue
        local = np.abs(kappa[start:stop])
        apex_offset = int(np.argmax(local))
        apex = start + apex_offset
        peak = float(local[apex_offset])
        if peak <= 0:
            continue
        corners.append(
            Corner(
                index=len(corners) + 1,
                entry_s=float(reference.s_m[start]),
                apex_s=float(reference.s_m[apex]),
                exit_s=float(reference.s_m[stop - 1]),
                direction="left" if sign > 0 else "right",
                min_radius_m=float(1.0 / peak),
            )
        )
    return tuple(corners)


def corner_at(
    corners: tuple[Corner, ...] | list[Corner],
    s_m: float,
    *,
    approach_m: float = 180.0,
) -> Corner | None:
    """Corner the car is in, or the next one if ``s_m`` is within ``approach_m`` of it."""

    if not corners:
        return None
    if not np.isfinite(s_m):
        raise F1FrameError("s_m must be finite")
    containing = [
        corner
        for corner in corners
        if corner.entry_s - approach_m <= s_m <= corner.exit_s + 40.0
    ]
    if containing:
        return min(containing, key=lambda corner: abs(corner.apex_s - s_m))
    ahead = [corner for corner in corners if corner.entry_s >= s_m]
    if ahead:
        return ahead[0]
    return min(corners, key=lambda corner: abs(corner.apex_s - s_m))


def corner_window(
    corner: Corner,
    s_m: float,
    length_m: float,
    *,
    before_m: float = 80.0,
    after_m: float = 40.0,
) -> tuple[float, float]:
    """Meters covering the pass and the corner, padded so the braking zone is inside."""

    if length_m <= 0 or before_m < 0 or after_m < 0:
        raise F1FrameError("window lengths must be positive")
    start = min(s_m, corner.entry_s) - before_m
    end = max(s_m, corner.exit_s) + after_m
    start = max(0.0, start)
    end = min(float(length_m), end)
    if end <= start:
        end = min(float(length_m), start + 10.0)
    return float(start), float(end)


def _signed_curvature(x: np.ndarray, y: np.ndarray, smooth: int) -> np.ndarray:
    xs = _smooth(np.asarray(x, dtype=np.float64), smooth)
    ys = _smooth(np.asarray(y, dtype=np.float64), smooth)
    segment = np.hypot(np.diff(xs), np.diff(ys))
    s_m = np.concatenate([[0.0], np.cumsum(segment)])
    # A doubled vertex makes the derivative undefined. Nudge those steps.
    flat = np.diff(s_m) <= 1e-6
    if np.any(flat):
        s_m = s_m.copy()
        s_m[1:][flat] += 1e-3
        s_m = np.maximum.accumulate(s_m)
    dx = np.gradient(xs, s_m)
    dy = np.gradient(ys, s_m)
    ddx = np.gradient(dx, s_m)
    ddy = np.gradient(dy, s_m)
    speed2 = dx * dx + dy * dy
    kappa = (dx * ddy - dy * ddx) / np.maximum(speed2, 1e-12) ** 1.5
    kappa[~np.isfinite(kappa)] = 0.0
    return kappa


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] < window:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    index = 0
    n = labels.shape[0]
    while index < n:
        sign = int(labels[index])
        if sign == 0:
            index += 1
            continue
        stop = index + 1
        while stop < n and int(labels[stop]) == sign:
            stop += 1
        runs.append((index, stop, sign))
        index = stop
    return runs


def _merge_runs(
    runs: list[tuple[int, int, int]],
    s_m: np.ndarray,
    merge_gap_m: float,
) -> list[tuple[int, int, int]]:
    merged: list[tuple[int, int, int]] = []
    for start, stop, sign in runs:
        if merged and merged[-1][2] == sign:
            gap = float(s_m[start] - s_m[merged[-1][1] - 1])
            if gap <= merge_gap_m:
                merged[-1] = (merged[-1][0], stop, sign)
                continue
        merged.append((start, stop, sign))
    return merged
