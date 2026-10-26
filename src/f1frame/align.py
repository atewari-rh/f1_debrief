"""Align raw lap streams onto one geometric meter grid."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from f1frame.project import project_s
from f1frame.reference import build_reference
from f1frame.resample import grid_s, hold_last, interp_s, interp_with_gaps, monotonic_samples
from f1frame.types import (
    SCHEMA_VERSION,
    CarStream,
    F1FrameError,
    FrameReport,
    Lap,
    LapFrame,
    PositionStream,
    ReferenceLine,
)

_SPEED_INTEGRAL_MIN_MPS = 5.0 / 3.6
_INTEGRAL_COVERAGE = 0.95


def align(
    car: CarStream,
    position: PositionStream,
    *,
    lap_time_s: float | None = None,
    reference: ReferenceLine | None = None,
    step_m: float = 10.0,
    lap_id: str | None = None,
    lookahead_m: float = 250.0,
    lookback_m: float = 20.0,
    max_dist_m: float = 40.0,
    max_advance_m: float = 80.0,
    max_channel_gap_m: float | None = None,
) -> LapFrame:
    """Resample one lap onto ``reference``.

    Pass the same ``reference`` for every lap you want to compare. When it is
    omitted, the reference is this lap's own polyline, which is the right input
    for a single-lap check and the wrong input for comparing two laps.
    """

    if not isinstance(car, CarStream) or not isinstance(position, PositionStream):
        raise F1FrameError("align expects a CarStream and a PositionStream")
    if lap_time_s is not None and (not np.isfinite(lap_time_s) or lap_time_s <= 0):
        raise F1FrameError("lap_time_s must be a positive number of seconds")
    line = reference if reference is not None else build_reference(position)
    if max_channel_gap_m is None:
        # One late sample at the end of a straight is about 80 m at race speed.
        max_channel_gap_m = max(100.0, 4.0 * step_m)
    if max_channel_gap_m <= 0:
        raise F1FrameError("max_channel_gap_m must be positive")

    s_pos, dist = project_s(
        position.x,
        position.y,
        line,
        lookahead_m=lookahead_m,
        lookback_m=lookback_m,
        max_dist_m=max_dist_m,
        max_advance_m=max_advance_m,
    )
    n_projected = int(np.isfinite(s_pos).sum())
    if n_projected < 2:
        raise F1FrameError("fewer than 2 position samples landed on the reference line")

    s_car = interp_s(car.time_s, position.time_s, s_pos)
    columns = {
        "speed_kph": car.speed_kph,
        "throttle_pct": car.throttle_pct,
        "brake": car.brake,
        "gear": car.gear,
        "rpm": car.rpm if car.rpm is not None else np.full(car.time_s.shape[0], np.nan),
    }
    samples = monotonic_samples(s_car, car.time_s, columns)
    s_grid = grid_s(line.length_m, step_m)
    gap = max_channel_gap_m

    speed = interp_with_gaps(samples["s_m"], samples["speed_kph"], s_grid, gap)
    frame = LapFrame(
        s_m=s_grid,
        time_s=interp_with_gaps(samples["s_m"], samples["time_s"], s_grid, gap),
        speed_kph=speed,
        throttle_pct=interp_with_gaps(samples["s_m"], samples["throttle_pct"], s_grid, gap),
        brake=hold_last(samples["s_m"], samples["brake"], s_grid, gap),
        gear=hold_last(samples["s_m"], samples["gear"], s_grid, gap),
        rpm=interp_with_gaps(samples["s_m"], samples["rpm"], s_grid, gap),
        reference=line,
        report=_report(
            position=position,
            reference=line,
            s_grid=s_grid,
            speed_kph=speed,
            dist=dist,
            step_m=step_m,
            lap_time_s=None if lap_time_s is None else float(lap_time_s),
            n_projected=n_projected,
        ),
        lap_id=lap_id,
    )
    return frame


def align_laps(
    laps: Sequence[Lap],
    *,
    step_m: float = 10.0,
    reference: ReferenceLine | None = None,
    lookahead_m: float = 250.0,
    lookback_m: float = 20.0,
    max_dist_m: float = 40.0,
    max_advance_m: float = 80.0,
    max_channel_gap_m: float | None = None,
) -> list[LapFrame]:
    """Align several laps to one reference and one meter grid.

    The reference is the median polyline of the laps unless you pass one.
    Every returned frame has an identical ``s_m``.
    """

    if len(laps) == 0:
        raise F1FrameError("align_laps requires at least one lap")
    line = reference if reference is not None else build_reference([lap.position for lap in laps])
    frames = [
        align(
            lap.car,
            lap.position,
            lap_time_s=lap.lap_time_s,
            reference=line,
            step_m=step_m,
            lap_id=lap.lap_id,
            lookahead_m=lookahead_m,
            lookback_m=lookback_m,
            max_dist_m=max_dist_m,
            max_advance_m=max_advance_m,
            max_channel_gap_m=max_channel_gap_m,
        )
        for lap in laps
    ]
    s0 = frames[0].s_m
    for frame in frames[1:]:
        if frame.s_m.shape != s0.shape or not np.array_equal(frame.s_m, s0):
            raise F1FrameError("aligned laps did not share a meter grid")
    return frames


def brake_onsets(frame: LapFrame) -> np.ndarray:
    """Meters where brake changes from off to on along the grid."""

    brake = frame.brake
    previous = np.roll(brake, 1)
    previous[0] = 0.0
    edges = (brake >= 0.5) & (np.nan_to_num(previous, nan=1.0) < 0.5) & np.isfinite(brake)
    return frame.s_m[edges]


def speed_delta(left: LapFrame, right: LapFrame) -> np.ndarray:
    """``left.speed_kph - right.speed_kph`` on a shared grid."""

    if left.s_m.shape != right.s_m.shape or not np.array_equal(left.s_m, right.s_m):
        raise F1FrameError("speed_delta requires frames from the same align_laps call")
    return left.speed_kph - right.speed_kph


def _report(
    *,
    position: PositionStream,
    reference: ReferenceLine,
    s_grid: np.ndarray,
    speed_kph: np.ndarray,
    dist: np.ndarray,
    step_m: float,
    lap_time_s: float | None,
    n_projected: int,
) -> FrameReport:
    finite_dist = dist[np.isfinite(dist)]
    covered = np.isfinite(speed_kph)
    integrated_time, integrated_distance = _integrate_speed(s_grid, speed_kph)
    coverage = integrated_distance / reference.length_m if reference.length_m else 0.0
    timestamp_elapsed = float(position.time_s[-1] - position.time_s[0])
    timestamp_error = None
    integrated_error = None
    if lap_time_s is not None:
        timestamp_error = (timestamp_elapsed - lap_time_s) * 1000.0
        if integrated_time is not None and coverage >= _INTEGRAL_COVERAGE:
            integrated_error = (integrated_time - lap_time_s) * 1000.0
    return FrameReport(
        schema_version=SCHEMA_VERSION,
        step_m=step_m,
        reference_length_m=reference.length_m,
        official_lap_time_s=lap_time_s,
        timestamp_elapsed_s=timestamp_elapsed,
        timestamp_error_ms=timestamp_error,
        integrated_speed_time_s=integrated_time,
        integrated_speed_distance_m=integrated_distance,
        integrated_speed_error_ms=integrated_error,
        loop_closure_m=float(np.hypot(position.x[-1] - position.x[0], position.y[-1] - position.y[0])),
        start_offset_m=float(np.hypot(position.x[0] - reference.x[0], position.y[0] - reference.y[0])),
        projection_median_m=float(np.median(finite_dist)) if finite_dist.size else float("nan"),
        projection_max_m=float(np.max(finite_dist)) if finite_dist.size else float("nan"),
        n_position_samples=int(position.x.shape[0]),
        n_projected=n_projected,
        n_rejected=int(position.x.shape[0] - n_projected),
        grid_points=int(s_grid.shape[0]),
        covered_fraction=float(covered.mean()) if covered.size else 0.0,
    )


def _integrate_speed(s_grid: np.ndarray, speed_kph: np.ndarray) -> tuple[float | None, float]:
    """Time to cover the grid at the recorded speed. This is a diagnostic, not the ruler."""

    ds = np.diff(s_grid)
    v0 = speed_kph[:-1] / 3.6
    v1 = speed_kph[1:] / 3.6
    usable = np.isfinite(v0) & np.isfinite(v1) & (v0 > _SPEED_INTEGRAL_MIN_MPS) & (v1 > _SPEED_INTEGRAL_MIN_MPS)
    if not np.any(usable):
        return None, 0.0
    v_mid = 0.5 * (v0[usable] + v1[usable])
    distance = float(ds[usable].sum())
    return float(np.sum(ds[usable] / v_mid)), distance
