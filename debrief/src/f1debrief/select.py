"""On-track passes, using the published pit and penalty times."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class PassEvent:
    """One position swap from the timing feed."""

    date: datetime
    overtaking: str
    overtaken: str
    position: int | None = None


@dataclass(frozen=True)
class PitStop:
    date: datetime
    driver: str
    duration_s: float


@dataclass(frozen=True)
class Penalty:
    date: datetime
    driver: str


def on_track_passes(
    overtakes: list[PassEvent],
    pits: list[PitStop],
    penalties: list[Penalty],
    *,
    pit_margin_s: float = 5.0,
    penalty_margin_s: float = 30.0,
) -> list[PassEvent]:
    """Drop swaps that happen while a car is in the pit window, or next to a penalty.

    OpenF1 ``/overtakes`` includes pit-lane position changes. The pit timestamp is
    the stop itself, so the window starts ``duration`` before that timestamp,
    where the car is already in the pit entry, and ends ``duration`` after it.
    A penalty message drops swaps for that driver within ``penalty_margin_s``.
    """

    kept: list[PassEvent] = []
    for event in overtakes:
        if event.overtaking == event.overtaken:
            continue
        if _near_pit(event, pits, pit_margin_s) or _near_penalty(event, penalties, penalty_margin_s):
            continue
        kept.append(event)
    return kept


def passes_between(events: list[PassEvent], first: str, second: str) -> list[PassEvent]:
    """Passes in which these two drivers swapped places, either way, in time order."""

    pair = {first.upper(), second.upper()}
    return sorted(
        (event for event in events if {event.overtaking, event.overtaken} == pair),
        key=lambda event: event.date,
    )


def _near_pit(event: PassEvent, pits: list[PitStop], margin_s: float) -> bool:
    for pit in pits:
        if pit.driver not in (event.overtaking, event.overtaken):
            continue
        start = pit.date - timedelta(seconds=max(pit.duration_s, 20.0) + margin_s)
        end = pit.date + timedelta(seconds=pit.duration_s + margin_s)
        if start <= event.date <= end:
            return True
    return False


def _near_penalty(event: PassEvent, penalties: list[Penalty], margin_s: float) -> bool:
    margin = timedelta(seconds=margin_s)
    for penalty in penalties:
        if penalty.driver not in (event.overtaking, event.overtaken):
            continue
        if abs(event.date - penalty.date) <= margin:
            return True
    return False
