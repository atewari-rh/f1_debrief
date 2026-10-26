"""Build the polyline that defines meters from the start of the lap."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from f1frame.types import F1FrameError, PositionStream, ReferenceLine

_DEDUP_M = 0.25
_MAX_VERTICES = 4000


def reference_from_xy(x: np.ndarray, y: np.ndarray) -> ReferenceLine:
    """Reference line from one polyline, in meters, starting at the first point."""

    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.ndim != 1 or ys.ndim != 1 or xs.shape != ys.shape:
        raise F1FrameError("x and y must be 1-D arrays of equal length")
    if xs.shape[0] < 2:
        raise F1FrameError("a reference line needs at least 2 points")
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise F1FrameError("reference coordinates contain NaN or infinity")
    xs, ys = _dedup(xs, ys, _DEDUP_M)
    if xs.shape[0] < 2:
        raise F1FrameError("reference coordinates collapse to a single point")
    segment = np.hypot(np.diff(xs), np.diff(ys))
    s_m = np.concatenate([[0.0], np.cumsum(segment)])
    return ReferenceLine(xs, ys, s_m)


def build_reference(positions: PositionStream | Sequence[PositionStream]) -> ReferenceLine:
    """Build one reference line from a lap, or the median shape of several laps.

    Every lap must run the same direction and start at the same part of the
    circuit. With one lap, the line is that lap's polyline. With several, each
    lap is resampled by fraction of its own length and the median vertex is kept.
    """

    laps = _as_position_list(positions)
    if len(laps) == 1:
        return reference_from_xy(laps[0].x, laps[0].y)

    n = min(_MAX_VERTICES, max(lap.x.shape[0] for lap in laps))
    stacked_x = np.empty((len(laps), n), dtype=np.float64)
    stacked_y = np.empty((len(laps), n), dtype=np.float64)
    for i, lap in enumerate(laps):
        stacked_x[i], stacked_y[i] = _resample_fraction(lap.x, lap.y, n)
    return reference_from_xy(np.median(stacked_x, axis=0), np.median(stacked_y, axis=0))


def _as_position_list(
    positions: PositionStream | Sequence[PositionStream],
) -> list[PositionStream]:
    if isinstance(positions, PositionStream):
        return [positions]
    laps = list(positions)
    if not laps:
        raise F1FrameError("build_reference requires at least one position stream")
    if not all(isinstance(lap, PositionStream) for lap in laps):
        raise F1FrameError("build_reference expected PositionStream objects")
    return laps


def _dedup(x: np.ndarray, y: np.ndarray, tol_m: float) -> tuple[np.ndarray, np.ndarray]:
    keep = [0]
    for i in range(1, x.shape[0]):
        if np.hypot(x[i] - x[keep[-1]], y[i] - y[keep[-1]]) >= tol_m:
            keep.append(i)
    index = np.asarray(keep, dtype=int)
    return x[index], y[index]


def _resample_fraction(x: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = _dedup(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), _DEDUP_M)
    if xs.shape[0] < 2:
        raise F1FrameError("a lap polyline collapses to a single point")
    chord = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    if chord[-1] <= 0:
        raise F1FrameError("a lap polyline has zero length")
    fraction = chord / chord[-1]
    # Repeated fractions make np.interp undefined. Keep the first vertex.
    _, unique = np.unique(fraction, return_index=True)
    unique.sort()
    target = np.linspace(0.0, 1.0, n)
    return np.interp(target, fraction[unique], xs[unique]), np.interp(target, fraction[unique], ys[unique])
