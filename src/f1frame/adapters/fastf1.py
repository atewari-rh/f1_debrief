"""Optional conversion from a FastF1 lap.

The core library never imports FastF1. This adapter reads the raw car clock
and the raw position clock separately. Do not pass ``lap.get_telemetry()``:
that object has already interpolated both clocks onto one time base.
"""

from __future__ import annotations

import numpy as np

from f1frame.types import F1FrameError, CarStream, PositionStream


def streams_from_fastf1_lap(
    lap,
    *,
    xy_scale: float = 0.1,
) -> tuple[CarStream, PositionStream, float | None]:
    """Return ``(car, position, official lap time)`` for one FastF1 ``Lap``.

    FastF1 position channels are in tenths of a meter, so ``xy_scale`` defaults
    to 0.1. Both streams use ``SessionTime`` seconds when that column exists.
    """

    if xy_scale <= 0:
        raise F1FrameError("xy_scale must be positive")
    try:
        car_df = lap.get_car_data()
        pos_df = lap.get_pos_data()
    except AttributeError as exc:
        raise F1FrameError("expected a FastF1 lap with get_car_data() and get_pos_data()") from exc
    car = CarStream(
        time_s=_seconds(car_df),
        speed_kph=_column(car_df, "Speed"),
        throttle_pct=_column(car_df, "Throttle"),
        brake=_column(car_df, "Brake").astype(np.float64),
        gear=_column(car_df, "nGear"),
        rpm=_column(car_df, "RPM"),
    )
    position = PositionStream(
        time_s=_seconds(pos_df),
        x=_column(pos_df, "X") * xy_scale,
        y=_column(pos_df, "Y") * xy_scale,
    )
    return car, position, _lap_time_seconds(lap)


def _seconds(frame) -> np.ndarray:
    if "SessionTime" in frame.columns:
        values = frame["SessionTime"]
        if hasattr(values.dt, "total_seconds"):
            return values.dt.total_seconds().to_numpy(dtype=np.float64)
    if "Date" in frame.columns:
        values = frame["Date"]
        delta = values - values.iloc[0]
        if hasattr(delta.dt, "total_seconds"):
            return delta.dt.total_seconds().to_numpy(dtype=np.float64)
    raise F1FrameError("FastF1 frame has neither SessionTime nor Date")


def _column(frame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise F1FrameError(f"FastF1 frame is missing {name}")
    return np.asarray(frame[name].to_numpy(), dtype=np.float64)


def _lap_time_seconds(lap) -> float | None:
    try:
        value = lap["LapTime"]
    except (KeyError, TypeError):
        return None
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        seconds = float(value.total_seconds())
    else:
        seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0:
        return None
    return seconds
