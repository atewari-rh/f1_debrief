"""Synthetic laps for tests and the example.

These are generated polylines, not Formula 1 telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1frame.types import CarStream, PositionStream


@dataclass(frozen=True)
class SampledLap:
    car: CarStream
    position: PositionStream
    lap_time_s: float
    brake_s_m: float
    length_m: float


def stadium_polyline(
    straight_m: float = 800.0,
    radius_m: float = 80.0,
    spacing_m: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A stadium loop. The last point is one ``spacing_m`` short of the start."""

    if min(straight_m, radius_m, spacing_m) <= 0:
        raise ValueError("stadium dimensions must be positive")
    bottom_x = np.arange(0.0, straight_m, spacing_m)
    bottom = np.column_stack([bottom_x, np.zeros_like(bottom_x)])
    arc_n = max(8, int(np.ceil(np.pi * radius_m / spacing_m)))
    theta_right = np.linspace(-np.pi / 2, np.pi / 2, arc_n, endpoint=False)
    right = np.column_stack(
        [straight_m + radius_m * np.cos(theta_right), radius_m + radius_m * np.sin(theta_right)]
    )
    top_x = np.arange(straight_m, 0.0, -spacing_m)
    top = np.column_stack([top_x, np.full_like(top_x, 2 * radius_m)])
    theta_left = np.linspace(np.pi / 2, 3 * np.pi / 2, arc_n, endpoint=False)
    left = np.column_stack([radius_m * np.cos(theta_left), radius_m + radius_m * np.sin(theta_left)])
    points = np.vstack([bottom, right, top, left])
    return points[:, 0].copy(), points[:, 1].copy()


def sample_constant_lap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    speed_kph: float,
    hz: float = 20.0,
    brake_at_s: float,
    speed_scale: float = 1.0,
    gear: int = 7,
    pos_time_offset_s: float = 0.0,
) -> SampledLap:
    """Walk ``(x, y)`` at a constant true speed.

    ``speed_scale`` multiplies only the recorded speed channel. The geometry
    and the clock stay true, which is the failure mode of integrating speed
    to recover distance.
    """

    if speed_kph <= 0 or hz <= 0:
        raise ValueError("speed_kph and hz must be positive")
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    chord = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    length = float(chord[-1])
    if not 0 < brake_at_s < length:
        raise ValueError("brake_at_s must lie on the polyline")
    v = speed_kph / 3.6
    dt = 1.0 / hz
    sample_s = np.arange(0.0, length, v * dt)
    if not np.any(np.isclose(sample_s, brake_at_s)):
        sample_s = np.sort(np.append(sample_s, brake_at_s))
    sample_s = sample_s[sample_s < length]
    time_s = sample_s / v
    sx = np.interp(sample_s, chord, xs)
    sy = np.interp(sample_s, chord, ys)
    brake = (sample_s >= brake_at_s - 1e-9).astype(np.float64)
    throttle = np.where(brake > 0.5, 0.0, 100.0)
    recorded_speed = np.full(sample_s.shape, speed_kph * speed_scale)
    car = CarStream(
        time_s=time_s,
        speed_kph=recorded_speed,
        throttle_pct=throttle,
        brake=brake,
        gear=np.full(sample_s.shape, gear, dtype=np.float64),
        rpm=recorded_speed * 40.0,
    )
    position = PositionStream(time_s=time_s + pos_time_offset_s, x=sx, y=sy)
    return SampledLap(
        car=car,
        position=position,
        lap_time_s=float(time_s[-1] - time_s[0]),
        brake_s_m=float(brake_at_s),
        length_m=length,
    )
