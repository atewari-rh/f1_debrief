"""Shared types for lap streams, reference lines, and aligned frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = "1"


class F1FrameError(ValueError):
    """Invalid input or a lap that cannot be projected onto the reference."""


def _as_float(name: str, values: np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise F1FrameError(f"{name} must be a 1-D array")
    if n is not None and array.shape[0] != n:
        raise F1FrameError(f"{name} length {array.shape[0]} does not match {n}")
    if not np.all(np.isfinite(array)):
        raise F1FrameError(f"{name} contains NaN or infinity")
    return array.copy()


def _as_brake(values: np.ndarray, n: int) -> np.ndarray:
    """Accept FastF1's 0/1 brake and OpenF1's 0/100 brake."""

    raw = np.asarray(values)
    if raw.dtype == np.bool_:
        if raw.ndim != 1 or raw.shape[0] != n:
            raise F1FrameError(f"car.brake length does not match {n}")
        brake = raw.astype(np.float64)
    else:
        brake = _as_float("car.brake", raw, n)
    levels = set(np.unique(brake).tolist())
    if levels <= {0.0, 1.0}:
        return brake
    if levels <= {0.0, 100.0}:
        return brake / 100.0
    raise F1FrameError("car.brake must be 0/1 (FastF1) or 0/100 (OpenF1)")


def _as_time(name: str, values: np.ndarray) -> np.ndarray:
    time_s = _as_float(name, values)
    if time_s.shape[0] < 2:
        raise F1FrameError(f"{name} needs at least 2 samples")
    if np.any(np.diff(time_s) <= 0):
        raise F1FrameError(f"{name} must be strictly increasing")
    return time_s


@dataclass(frozen=True)
class CarStream:
    """Car channels on their own clock.

    ``time_s`` shares an epoch with the position stream for the same lap.
    ``brake`` is off/on. FastF1 sends 0 and 1. OpenF1 sends 0 and 100.
    Both are stored as 0 and 1.
    ``throttle_pct`` is the pedal reading, usually 0 to 100. The value 104,
    which the timing feed uses for a missing sample, is left unchanged.
    """

    time_s: np.ndarray
    speed_kph: np.ndarray
    throttle_pct: np.ndarray
    brake: np.ndarray
    gear: np.ndarray
    rpm: np.ndarray | None = None

    def __post_init__(self) -> None:
        time_s = _as_time("car.time_s", self.time_s)
        n = time_s.shape[0]
        speed = _as_float("car.speed_kph", self.speed_kph, n)
        if np.any(speed < 0):
            raise F1FrameError("car.speed_kph must be >= 0")
        throttle = _as_float("car.throttle_pct", self.throttle_pct, n)
        brake = _as_brake(self.brake, n)
        gear = _as_float("car.gear", self.gear, n)
        if not np.all(gear == np.round(gear)):
            raise F1FrameError("car.gear must be whole numbers")
        rpm = None
        if self.rpm is not None:
            rpm = _as_float("car.rpm", self.rpm, n)
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "speed_kph", speed)
        object.__setattr__(self, "throttle_pct", throttle)
        object.__setattr__(self, "brake", brake)
        object.__setattr__(self, "gear", gear)
        object.__setattr__(self, "rpm", rpm)


@dataclass(frozen=True)
class PositionStream:
    """Position samples on their own clock, in meters.

    FastF1 publishes X and Y in tenths of a meter. Convert before constructing
    this stream, or use :func:`f1frame.adapters.fastf1.streams_from_fastf1_lap`.
    """

    time_s: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        time_s = _as_time("position.time_s", self.time_s)
        n = time_s.shape[0]
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "x", _as_float("position.x", self.x, n))
        object.__setattr__(self, "y", _as_float("position.y", self.y, n))


@dataclass(frozen=True)
class ReferenceLine:
    """Polyline that defines s = 0 at the first vertex.

    ``s_m[i]`` is the arc length in meters from the first vertex to vertex i.
    The line is open: a closed lap ends near the first vertex with s near
    ``length_m``, and the closing chord is not part of the line.
    """

    x: np.ndarray
    y: np.ndarray
    s_m: np.ndarray

    def __post_init__(self) -> None:
        x = _as_float("reference.x", self.x)
        n = x.shape[0]
        if n < 2:
            raise F1FrameError("reference line needs at least 2 vertices")
        y = _as_float("reference.y", self.y, n)
        s_m = _as_float("reference.s_m", self.s_m, n)
        if s_m[0] != 0.0 or np.any(np.diff(s_m) <= 0):
            raise F1FrameError("reference.s_m must start at 0 and increase")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "s_m", s_m)

    @property
    def length_m(self) -> float:
        return float(self.s_m[-1])


@dataclass(frozen=True)
class Lap:
    """One lap's raw streams, ready to align with others."""

    car: CarStream
    position: PositionStream
    lap_time_s: float | None = None
    lap_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.car, CarStream) or not isinstance(self.position, PositionStream):
            raise F1FrameError("Lap.car and Lap.position must be streams")
        if self.lap_time_s is not None:
            lap_time = float(self.lap_time_s)
            if not np.isfinite(lap_time) or lap_time <= 0:
                raise F1FrameError("lap_time_s must be a positive number of seconds")
            object.__setattr__(self, "lap_time_s", lap_time)


@dataclass(frozen=True)
class FrameReport:
    """How this frame relates to the official lap and to the reference line.

    ``timestamp_error_ms`` compares the position-stream duration with the
    official lap time. A large value means the streams do not cover the lap
    that the timing system measured.

    ``integrated_speed_error_ms`` is the lap time you would get by integrating
    the recorded speed along the grid, minus the official lap time. Alignment
    does not use that integral. The field exists so a caller can see the
    disagreement, which includes both speed-integration drift and the scale of
    the timing map. It is ``None`` when the integral covers less than 95% of
    the reference, because a partial integral is not a lap time.
    """

    schema_version: str
    step_m: float
    reference_length_m: float
    official_lap_time_s: float | None
    timestamp_elapsed_s: float
    timestamp_error_ms: float | None
    integrated_speed_time_s: float | None
    integrated_speed_distance_m: float
    integrated_speed_error_ms: float | None
    loop_closure_m: float
    start_offset_m: float
    projection_median_m: float
    projection_max_m: float
    n_position_samples: int
    n_projected: int
    n_rejected: int
    grid_points: int
    covered_fraction: float


@dataclass(frozen=True)
class LapFrame:
    """One lap resampled onto the shared meter grid.

    Discrete channels (``brake``, ``gear``) hold the last sample at or before
    that meter. Continuous channels are linear between samples. Missing
    stretches are NaN rather than filled across a gap.
    """

    s_m: np.ndarray
    time_s: np.ndarray
    speed_kph: np.ndarray
    throttle_pct: np.ndarray
    brake: np.ndarray
    gear: np.ndarray
    rpm: np.ndarray
    reference: ReferenceLine
    report: FrameReport
    lap_id: str | None = None
