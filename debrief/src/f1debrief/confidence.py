"""Turn an f1frame report into a confidence badge.

The thresholds come from the public timing feed, not from a survey-grade GPS
clock. On 2024 Monza qualifying the aligned frames sat 0.1–0.25 s off the
official lap time while every position sample still landed on the reference.
A gate of 0.05 s would have rejected those laps.
"""

from __future__ import annotations

from dataclasses import dataclass

from f1frame import FrameReport, LapFrame

# Absolute timestamp error against the official lap time.
_LIMITED_TIMESTAMP_MS = 400.0
_LOW_TIMESTAMP_MS = 1000.0
_LIMITED_PROJECTION_M = 15.0
_LOW_PROJECTION_M = 30.0
_LIMITED_COVERAGE = 0.97
_LOW_COVERAGE = 0.90
_REJECT_FRACTION_LIMITED = 0.0
_REJECT_FRACTION_LOW = 0.05


@dataclass(frozen=True)
class Confidence:
    """``high``, ``limited``, or ``low``, plus the reasons a reader can check."""

    level: str
    reasons: tuple[str, ...]

    @property
    def brake_points_verified(self) -> bool:
        return self.level != "low"


def judge_frames(frames: tuple[LapFrame, ...] | list[LapFrame]) -> Confidence:
    """Badge for a set of frames that were aligned together."""

    return judge_reports([(frame.lap_id, frame.report) for frame in frames])


def judge_reports(items: list[tuple[str | None, FrameReport]]) -> Confidence:
    level = "high"
    reasons: list[str] = []

    def downgrade(new_level: str, reason: str) -> None:
        nonlocal level
        rank = {"high": 0, "limited": 1, "low": 2}
        if rank[new_level] > rank[level]:
            level = new_level
        reasons.append(reason)

    for lap_id, report in items:
        name = lap_id or "lap"
        if report.n_position_samples:
            rejected = report.n_rejected / report.n_position_samples
            if rejected > _REJECT_FRACTION_LOW:
                downgrade("low", f"{name}: {report.n_rejected} position samples missed the reference.")
            elif rejected > _REJECT_FRACTION_LIMITED:
                downgrade("limited", f"{name}: {report.n_rejected} position samples missed the reference.")
        if report.projection_max_m > _LOW_PROJECTION_M:
            downgrade("low", f"{name}: a matched sample sits {report.projection_max_m:.0f} m off the reference.")
        elif report.projection_max_m > _LIMITED_PROJECTION_M:
            downgrade(
                "limited",
                f"{name}: the worst matched sample is {report.projection_max_m:.0f} m off the reference.",
            )
        if report.timestamp_error_ms is not None and abs(report.timestamp_error_ms) > _LOW_TIMESTAMP_MS:
            downgrade(
                "low",
                f"{name}: the frame clock disagrees with the official lap time by {report.timestamp_error_ms:.0f} ms.",
            )
        elif report.timestamp_error_ms is not None and abs(report.timestamp_error_ms) > _LIMITED_TIMESTAMP_MS:
            downgrade(
                "limited",
                f"{name}: the frame clock disagrees with the official lap time by {report.timestamp_error_ms:.0f} ms.",
            )
        if report.covered_fraction < _LOW_COVERAGE:
            downgrade("low", f"{name}: only {report.covered_fraction:.0%} of the grid has a speed value.")
        elif report.covered_fraction < _LIMITED_COVERAGE:
            downgrade("limited", f"{name}: {report.covered_fraction:.0%} of the grid has a speed value.")

    if level == "low":
        reasons.append("Brake points on this card are unverified.")
    elif not reasons:
        reasons.append("Position samples landed on the shared reference, and the clocks are close to the official lap times.")
    return Confidence(level, tuple(reasons))
