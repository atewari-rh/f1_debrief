"""A synthetic pass used by the CLI demo and the tests.

NOR's speed-integral brake point is computed with an 8% high speed channel.
The approach card uses the geometric grid and the unscaled speed samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from f1frame import Lap, LapFrame, align_laps
from f1frame.synthetic import stadium_polyline
from f1frame.types import CarStream, PositionStream

from f1debrief.audit import ApproachCard, approach_card

WINDOW = (1700.0, 2100.0)


@dataclass(frozen=True)
class DemoPass:
    frames: tuple[LapFrame, ...]
    card: ApproachCard
    integrated_brake_m: dict[str, float]


def demo_pass() -> DemoPass:
    x, y = stadium_polyline()
    ver = _braking_lap(x, y, lap_id="VER", brake_at_s=1800.0, apex_kph=112.0)
    nor = _braking_lap(x, y, lap_id="NOR", brake_at_s=1830.0, apex_kph=104.0)
    frames = tuple(align_laps([ver, nor], step_m=10.0))
    return DemoPass(
        frames=frames,
        card=approach_card(frames, WINDOW[0], WINDOW[1], title="Turn approach"),
        integrated_brake_m={
            "VER": _integrated_brake(ver.car, speed_scale=1.0),
            "NOR": _integrated_brake(nor.car, speed_scale=1.08),
        },
    )


def _braking_lap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lap_id: str,
    brake_at_s: float,
    apex_kph: float,
    cruise_kph: float = 280.0,
    hz: float = 20.0,
) -> Lap:
    chord = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    length = float(chord[-1])
    dt = 1.0 / hz
    s_m = 0.0
    time_s = 0.0
    samples: list[tuple[float, float, float, float]] = []
    while s_m < length:
        true_kph = _true_speed(s_m, brake_at_s, cruise_kph, apex_kph)
        samples.append((time_s, s_m, true_kph, 1.0 if s_m >= brake_at_s else 0.0))
        s_m += (true_kph / 3.6) * dt
        time_s += dt
    if not any(abs(sample[1] - brake_at_s) < 1e-6 for sample in samples):
        true_kph = _true_speed(brake_at_s, brake_at_s, cruise_kph, apex_kph)
        time_at_brake = _time_at(brake_at_s, cruise_kph)
        samples.append((time_at_brake, brake_at_s, true_kph, 1.0))
    samples.sort(key=lambda sample: sample[0])
    # The inserted brake sample can share a timestamp with the stepped sample.
    cleaned: list[tuple[float, float, float, float]] = []
    for sample in samples:
        if cleaned and sample[0] <= cleaned[-1][0]:
            continue
        cleaned.append(sample)
    time = np.array([sample[0] for sample in cleaned])
    distance = np.array([sample[1] for sample in cleaned])
    true_speed = np.array([sample[2] for sample in cleaned])
    brake = np.array([sample[3] for sample in cleaned])
    car = CarStream(
        time_s=time,
        speed_kph=true_speed,
        throttle_pct=np.where(brake > 0.5, 20.0, 100.0),
        brake=brake,
        gear=np.full(time.shape, 7.0),
        rpm=true_speed * 40.0,
    )
    position = PositionStream(
        time_s=time,
        x=np.interp(distance, chord, x),
        y=np.interp(distance, chord, y),
    )
    return Lap(car, position, lap_time_s=float(time[-1] - time[0]), lap_id=lap_id)


def _true_speed(s_m: float, brake_at_s: float, cruise_kph: float, apex_kph: float) -> float:
    if s_m < brake_at_s:
        return cruise_kph
    progress = min(1.0, (s_m - brake_at_s) / 120.0)
    return (1.0 - progress) * cruise_kph + progress * apex_kph


def _time_at(target_s: float, cruise_kph: float) -> float:
    return target_s / (cruise_kph / 3.6)


def _integrated_brake(car: CarStream, *, speed_scale: float) -> float:
    """Distance at the brake sample if speed is integrated, with an optional bias."""

    index = int(np.flatnonzero(car.brake >= 0.5)[0])
    dt = np.diff(car.time_s[: index + 1])
    speed = car.speed_kph[: index + 1] / 3.6 * speed_scale
    mid = 0.5 * (speed[:-1] + speed[1:])
    return float(np.sum(mid * dt))
