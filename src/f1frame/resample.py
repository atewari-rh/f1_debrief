"""Place car channels onto a uniform meter grid."""

from __future__ import annotations

import numpy as np

from f1frame.types import F1FrameError


def grid_s(length_m: float, step_m: float) -> np.ndarray:
    if step_m <= 0:
        raise F1FrameError("step_m must be positive")
    if length_m <= step_m:
        raise F1FrameError("reference line is shorter than step_m")
    count = int(np.floor(length_m / step_m)) + 1
    return np.arange(count, dtype=np.float64) * step_m


def interp_s(car_time_s: np.ndarray, pos_time_s: np.ndarray, s_pos: np.ndarray) -> np.ndarray:
    """Geometric s at each car-sample time. Times outside the position clock are NaN."""

    finite = np.isfinite(s_pos)
    if finite.sum() < 2:
        return np.full(car_time_s.shape[0], np.nan)
    time_s = pos_time_s[finite]
    s_m = s_pos[finite]
    advance = np.diff(s_m, prepend=s_m[0] - 1.0) > 1e-6
    time_s = time_s[advance]
    s_m = s_m[advance]
    if time_s.shape[0] < 2 or np.any(np.diff(time_s) <= 0):
        return np.full(car_time_s.shape[0], np.nan)
    return np.interp(car_time_s, time_s, s_m, left=np.nan, right=np.nan)


def monotonic_samples(
    s_m: np.ndarray,
    time_s: np.ndarray,
    columns: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Drop car samples whose projected s is missing or not moving forward."""

    finite = np.isfinite(s_m) & np.isfinite(time_s)
    order = np.argsort(time_s[finite], kind="mergesort")
    s_sorted = s_m[finite][order]
    advance = np.diff(s_sorted, prepend=s_sorted[0] - 1.0) > 1e-3 if s_sorted.size else np.array([], dtype=bool)
    chosen = np.flatnonzero(finite)[order][advance]
    out = {"s_m": s_m[chosen], "time_s": time_s[chosen]}
    for name, values in columns.items():
        out[name] = values[chosen]
    return out


def interp_with_gaps(
    s_samples: np.ndarray,
    values: np.ndarray,
    s_grid: np.ndarray,
    max_gap_m: float,
) -> np.ndarray:
    """Linear interpolation that refuses to cross a hole larger than ``max_gap_m``."""

    out = np.full(s_grid.shape[0], np.nan)
    if s_samples.shape[0] < 2:
        return out
    inside = (s_grid >= s_samples[0]) & (s_grid <= s_samples[-1])
    out[inside] = np.interp(s_grid[inside], s_samples, values)
    left = np.searchsorted(s_samples, s_grid, side="right") - 1
    bracket = (left >= 0) & (left + 1 < s_samples.shape[0])
    span = np.full(s_grid.shape[0], np.inf)
    span[bracket] = s_samples[left[bracket] + 1] - s_samples[left[bracket]]
    # A sample that lands on a grid meter is a measurement. The meters between
    # two samples are not, once those samples are farther apart than max_gap_m.
    on_sample = np.full(s_grid.shape[0], np.inf)
    has_left = left >= 0
    on_sample[has_left] = np.abs(s_grid[has_left] - s_samples[left[has_left]])
    out[(span > max_gap_m) & (on_sample > 1e-6)] = np.nan
    return out


def hold_last(
    s_samples: np.ndarray,
    values: np.ndarray,
    s_grid: np.ndarray,
    max_gap_m: float,
) -> np.ndarray:
    """Zero-order hold: the value of the last sample at or before each grid meter."""

    out = np.full(s_grid.shape[0], np.nan)
    if s_samples.shape[0] == 0:
        return out
    index = np.searchsorted(s_samples, s_grid, side="right") - 1
    valid = index >= 0
    gap = np.full(s_grid.shape[0], np.inf)
    gap[valid] = s_grid[valid] - s_samples[index[valid]]
    ok = valid & (gap <= max_gap_m) & (gap >= 0)
    out[ok] = values[index[ok]]
    return out
