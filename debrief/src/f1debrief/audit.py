"""Brake point and apex speed for several laps on one meter grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from f1frame import F1FrameError, LapFrame, brake_onsets, gap_between, spatial_time_delta
from f1frame.geometry import Corner

from f1debrief.confidence import Confidence, judge_frames


@dataclass(frozen=True)
class DriverApproach:
    lap_id: str
    brake_m: float | None
    apex_speed_kph: float | None
    apex_s_m: float | None


@dataclass(frozen=True)
class ApproachCard:
    """What each car did between two meters.

    ``brake_m`` is the first brake onset inside the window. Apex speed is the
    minimum speed inside the window. The card has no lateral offset: the public
    position stream does not say who was on the inside.
    """

    title: str
    start_m: float
    end_m: float
    step_m: float
    drivers: tuple[DriverApproach, ...]
    confidence: Confidence
    closure_error_ms: float | None = None
    window_delta_s: float | None = None
    delta_reference_id: str | None = None
    delta_other_id: str | None = None
    corner_index: int | None = None
    corner_direction: str | None = None
    corner_radius_m: float | None = None
    pass_s_m: float | None = None
    note: str | None = None

    @property
    def later_brake(self) -> tuple[DriverApproach, DriverApproach, float] | None:
        """``(earlier, later, meters)`` when two drivers both have a brake point."""

        known = [driver for driver in self.drivers if driver.brake_m is not None]
        if len(known) < 2:
            return None
        earlier = min(known, key=lambda driver: driver.brake_m or 0.0)
        later = max(known, key=lambda driver: driver.brake_m or 0.0)
        assert earlier.brake_m is not None and later.brake_m is not None
        return earlier, later, later.brake_m - earlier.brake_m


def approach_card(
    frames: list[LapFrame] | tuple[LapFrame, ...],
    start_m: float,
    end_m: float,
    *,
    title: str = "Approach",
    corner: Corner | None = None,
    note: str | None = None,
    pass_s_m: float | None = None,
) -> ApproachCard:
    """Read brake onset and minimum speed on a window of a shared grid.

    Every frame must come from the same ``align_laps`` call. Calling ``align``
    once per driver builds a different ruler for each car, and the meter
    numbers then cannot be compared.
    """

    if len(frames) < 2:
        raise F1FrameError("an approach card needs at least two laps")
    if not np.isfinite(start_m) or not np.isfinite(end_m) or start_m >= end_m:
        raise F1FrameError("the window must be a finite interval of meters")
    _require_shared_grid(frames)
    drivers = tuple(_driver(frame, start_m, end_m) for frame in frames)
    delta = spatial_time_delta(frames[0], frames[1])
    return ApproachCard(
        title=title,
        start_m=float(start_m),
        end_m=float(end_m),
        step_m=frames[0].report.step_m,
        drivers=drivers,
        confidence=judge_frames(frames),
        closure_error_ms=delta.closure_error_ms,
        window_delta_s=gap_between(delta, start_m, end_m),
        delta_reference_id=frames[0].lap_id or "lap",
        delta_other_id=frames[1].lap_id or "lap",
        corner_index=None if corner is None else corner.index,
        corner_direction=None if corner is None else corner.direction,
        corner_radius_m=None if corner is None else corner.min_radius_m,
        pass_s_m=None if pass_s_m is None else float(pass_s_m),
        note=note,
    )


def _require_shared_grid(frames: list[LapFrame] | tuple[LapFrame, ...]) -> None:
    ruler = frames[0].s_m
    for frame in frames[1:]:
        if frame.s_m.shape != ruler.shape or not np.array_equal(frame.s_m, ruler):
            raise F1FrameError("approach cards require frames from the same align_laps call")


def _driver(frame: LapFrame, start_m: float, end_m: float) -> DriverApproach:
    window = (frame.s_m >= start_m) & (frame.s_m <= end_m)
    onsets = brake_onsets(frame)
    inside = onsets[(onsets >= start_m) & (onsets <= end_m)]
    brake_m: float | None
    if inside.size:
        brake_m = float(inside[0])
    else:
        indexes = np.flatnonzero(window)
        already_braking = indexes.size and np.isfinite(frame.brake[indexes[0]]) and frame.brake[indexes[0]] >= 0.5
        prior = onsets[onsets <= start_m]
        brake_m = float(prior[-1]) if already_braking and prior.size else None

    apex_speed = None
    apex_s = None
    if np.any(window):
        speed = frame.speed_kph[window]
        meters = frame.s_m[window]
        finite = np.isfinite(speed)
        if np.any(finite):
            choice = int(np.argmin(np.where(finite, speed, np.inf)))
            apex_speed = float(speed[choice])
            apex_s = float(meters[choice])
    return DriverApproach(
        lap_id=frame.lap_id or "lap",
        brake_m=brake_m,
        apex_speed_kph=apex_speed,
        apex_s_m=apex_s,
    )
