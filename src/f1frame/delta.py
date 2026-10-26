"""Time gap along the shared meter grid.

The integral uses recorded speed on the geometric grid. It is not the ruler.
``closure_error_ms`` is how far that integral misses the official lap-time
gap. On a timing map the miss includes the scale of the map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1frame.types import F1FrameError, LapFrame

_MIN_MPS = 5.0 / 3.6
_CLOSURE_COVERAGE = 0.95


@dataclass(frozen=True)
class SpatialTimeDelta:
    """Cumulative time gap of ``other`` relative to ``reference`` along ``s_m``.

    Positive ``delta_s`` means ``other`` is behind. Holes in either speed
    channel are NaN. ``integrated_gap_s`` sums only the segments where both
    speeds are usable, so it is a full-lap gap only when ``covered_fraction``
    is high.

    ``closure_error_ms`` is ``integrated_gap_s`` minus the official lap-time
    difference, in milliseconds. It is ``None`` when either lap has no official
    time or the paired integral covers less than 95% of the reference.
    """

    s_m: np.ndarray
    delta_s: np.ndarray
    integrated_gap_s: float
    closure_error_ms: float | None
    covered_fraction: float


def spatial_time_delta(reference: LapFrame, other: LapFrame) -> SpatialTimeDelta:
    """Integrate ``1/v_other - 1/v_reference`` along the shared grid."""

    if reference.s_m.shape != other.s_m.shape or not np.array_equal(reference.s_m, other.s_m):
        raise F1FrameError("spatial_time_delta requires frames from the same align_laps call")
    s_m = reference.s_m
    ds = np.diff(s_m)
    v_reference = reference.speed_kph[:-1] / 3.6
    v_other = other.speed_kph[:-1] / 3.6
    usable = (
        np.isfinite(v_reference)
        & np.isfinite(v_other)
        & (v_reference > _MIN_MPS)
        & (v_other > _MIN_MPS)
    )
    delta_s = np.full(s_m.shape, np.nan)
    delta_s[0] = 0.0
    running = 0.0
    for index, ok in enumerate(usable):
        if not ok:
            continue
        running += float(ds[index] * (1.0 / v_other[index] - 1.0 / v_reference[index]))
        delta_s[index + 1] = running
    length = float(s_m[-1]) if s_m.size else 0.0
    covered = float(ds[usable].sum() / length) if length > 0 else 0.0
    official_reference = reference.report.official_lap_time_s
    official_other = other.report.official_lap_time_s
    closure = None
    if (
        official_reference is not None
        and official_other is not None
        and covered >= _CLOSURE_COVERAGE
    ):
        closure = (running - (official_other - official_reference)) * 1000.0
    return SpatialTimeDelta(
        s_m=s_m,
        delta_s=delta_s,
        integrated_gap_s=float(running),
        closure_error_ms=None if closure is None else float(closure),
        covered_fraction=covered,
    )


def gap_between(delta: SpatialTimeDelta, start_m: float, end_m: float) -> float | None:
    """Time gained by ``other`` between two meters. Positive means ``other`` lost time."""

    if not np.isfinite(start_m) or not np.isfinite(end_m) or start_m >= end_m:
        raise F1FrameError("the window must be a finite interval of meters")
    finite = np.isfinite(delta.delta_s)
    in_window = (delta.s_m >= start_m) & (delta.s_m <= end_m)
    if int(in_window.sum()) < 2 or float(finite[in_window].mean()) < 0.8:
        return None
    s_m = delta.s_m[finite]
    gap = delta.delta_s[finite]
    step = float(delta.s_m[1] - delta.s_m[0]) if delta.s_m.size > 1 else 10.0
    if start_m < float(s_m[0]) - step or end_m > float(s_m[-1]) + step:
        return None
    return float(np.interp(end_m, s_m, gap) - np.interp(start_m, s_m, gap))
