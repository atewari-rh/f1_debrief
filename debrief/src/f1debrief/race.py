"""One race pass, aligned once, with the corner chosen from the shared line."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from f1frame import (
    Lap,
    LapFrame,
    align_laps,
    corner_at,
    corner_window,
    extract_corners,
)
from f1frame.geometry import Corner
from f1frame.types import CarStream, PositionStream

from f1debrief.audit import ApproachCard, approach_card
from f1debrief.openf1 import (
    DebriefError,
    find_race,
    laps_for,
    session_index,
    telemetry,
)
from f1debrief.select import PassEvent, on_track_passes, passes_between

_XY_SCALE = 0.1
_NOTE = "Pit stops and penalty messages next to this pass were left out."


def recorded_fixture_path() -> Path:
    return Path(__file__).with_name("fixtures") / "monza_2024_race_pass.json"


def card_from_recorded(path: Path | None = None) -> ApproachCard:
    """The committed 2024 Monza race pass. No network."""

    fixture = path or recorded_fixture_path()
    if not fixture.is_file():
        raise DebriefError(f"recorded session not found at {fixture}")
    return card_from_payload(json.loads(fixture.read_text(encoding="utf-8")))


def card_from_race(year: int, race: str, drivers: tuple[str, str], *, index: int = 0) -> ApproachCard:
    """Load one on-track pass from OpenF1 and align both laps together."""

    return card_from_payload(payload_from_race(year, race, drivers, index=index))


def payload_from_race(year: int, race: str, drivers: tuple[str, str], *, index: int = 0) -> dict:
    """The recorded-session document for one on-track pass. Coordinates are meters."""

    if index < 0:
        raise DebriefError("index starts at 0")
    session = find_race(year, race)
    session_key = int(session["session_key"])
    index_data = session_index(session_key)
    kept = on_track_passes(index_data["overtakes"], index_data["pits"], index_data["penalties"])
    pair = passes_between(kept, drivers[0], drivers[1])
    if not pair:
        raise DebriefError(
            f"No on-track pass between {drivers[0].upper()} and {drivers[1].upper()} in the {year} {race} race."
        )
    if index >= len(pair):
        raise DebriefError(f"That pair has {len(pair)} on-track pass(es); index {index} is past the end.")
    event = pair[index]
    numbers = {code: number for number, code in index_data["drivers"].items()}
    laps = {}
    for code in (event.overtaken, event.overtaking):
        if code not in numbers:
            raise DebriefError(f"{code} is not in this session")
        laps[code] = _lap_payload(session_key, numbers[code], code, event.date)
    return {
        "source": "openf1",
        "year": year,
        "race": str(session.get("location") or race),
        "session_name": "Race",
        "session_key": session_key,
        "pass": _event_payload(event, laps[event.overtaking]["lap_number"]),
        "overtakes": [_event_dict(item) for item in index_data["overtakes"]],
        "pits": [
            {"date": pit.date.isoformat(), "driver": pit.driver, "duration_s": pit.duration_s}
            for pit in index_data["pits"]
        ],
        "penalties": [
            {"date": penalty.date.isoformat(), "driver": penalty.driver} for penalty in index_data["penalties"]
        ],
        "on_track": len(kept),
        "laps": laps,
    }


def card_from_payload(payload: dict) -> ApproachCard:
    event = payload["pass"]
    order = (event["overtaken"], event["overtaking"])
    frames = align_laps(
        [_lap_from_json(code, payload["laps"][code]) for code in order],
        step_m=10.0,
    )
    by_id = {frame.lap_id: frame for frame in frames}
    overtaker = by_id[event["overtaking"]]
    pass_s = _s_at(overtaker, float(payload["laps"][event["overtaking"]]["pass_offset_s"]))
    corners = extract_corners(frames[0].reference)
    corner = corner_at(corners, pass_s)
    start, end = _window(corner, pass_s, frames[0].reference.length_m)
    title = (
        f"{payload['year']} {payload['race']} race, lap {event['lap_number']}: "
        f"{event['overtaking']} passes {event['overtaken']}"
    )
    return approach_card(
        frames,
        start,
        end,
        title=title,
        corner=corner,
        note=_NOTE,
        pass_s_m=pass_s,
    )


def _window(corner: Corner | None, pass_s: float, length_m: float) -> tuple[float, float]:
    if corner is None:
        return max(0.0, pass_s - 100.0), min(length_m, pass_s + 150.0)
    return corner_window(corner, pass_s, length_m)


def _lap_payload(session_key: int, driver_number: int, code: str, when: datetime) -> dict:
    rows = laps_for(session_key, driver_number)
    chosen, start, end = _lap_containing(rows, when)
    if chosen.get("is_pit_out_lap"):
        raise DebriefError(f"{code} lap {chosen.get('lap_number')} leaves the pits")
    duration = chosen.get("lap_duration")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise DebriefError(f"{code} lap {chosen.get('lap_number')} has no official lap time")
    car_rows, location_rows = telemetry(session_key, driver_number, start, end)
    car = _car_arrays(car_rows, start)
    location = _location_arrays(location_rows, start)
    if car is None or location is None:
        raise DebriefError(f"{code} has no car or location samples on lap {chosen.get('lap_number')}")
    return {
        "lap_number": int(chosen["lap_number"]),
        "lap_time_s": float(duration),
        "date_start": start.isoformat(),
        "pass_offset_s": (when - start).total_seconds(),
        **car,
        **location,
    }


def _lap_containing(rows: list[dict], when: datetime) -> tuple[dict, datetime, datetime]:
    for index, row in enumerate(rows):
        start = datetime.fromisoformat(row["date_start"].replace("Z", "+00:00"))
        if index + 1 < len(rows):
            end = datetime.fromisoformat(rows[index + 1]["date_start"].replace("Z", "+00:00"))
        else:
            duration = row.get("lap_duration") or 120
            end = start + timedelta(seconds=float(duration))
        if start <= when < end:
            return row, start, end
    raise DebriefError("the pass does not fall inside a published lap")


def _car_arrays(rows: list[dict], start: datetime) -> dict | None:
    samples = []
    for row in rows:
        moment = datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
        samples.append(
            (
                (moment - start).total_seconds(),
                row.get("speed"),
                row.get("throttle"),
                row.get("brake"),
                row.get("n_gear"),
                row.get("rpm"),
            )
        )
    samples.sort()
    kept = _strict(samples)
    if len(kept) < 2:
        return None
    columns = list(zip(*kept))
    return {
        "t_car": [round(value, 3) for value in columns[0]],
        "speed": [None if value is None else int(value) for value in columns[1]],
        "throttle": [None if value is None else int(value) for value in columns[2]],
        "brake": [None if value is None else int(value) for value in columns[3]],
        "gear": [None if value is None else int(value) for value in columns[4]],
        "rpm": [None if value is None else int(value) for value in columns[5]],
    }


def _location_arrays(rows: list[dict], start: datetime) -> dict | None:
    samples = []
    for row in rows:
        x = row.get("x")
        y = row.get("y")
        if x is None or y is None:
            continue
        if x == 0 and y == 0:
            continue
        moment = datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
        samples.append(((moment - start).total_seconds(), x * _XY_SCALE, y * _XY_SCALE))
    samples.sort()
    kept = _strict(samples)
    if len(kept) < 2:
        return None
    columns = list(zip(*kept))
    return {
        "t_loc": [round(value, 3) for value in columns[0]],
        "x": [round(value, 2) for value in columns[1]],
        "y": [round(value, 2) for value in columns[2]],
    }


def _strict(samples: list[tuple]) -> list[tuple]:
    kept: list[tuple] = []
    for sample in samples:
        if sample[0] is None or any(value is None for value in sample):
            continue
        if kept and sample[0] <= kept[-1][0]:
            continue
        kept.append(sample)
    return kept


def _lap_from_json(code: str, lap: dict) -> Lap:
    car = CarStream(
        time_s=np.asarray(lap["t_car"], dtype=np.float64),
        speed_kph=np.asarray(lap["speed"], dtype=np.float64),
        throttle_pct=np.asarray(lap["throttle"], dtype=np.float64),
        brake=np.asarray(lap["brake"], dtype=np.float64),
        gear=np.asarray(lap["gear"], dtype=np.float64),
        rpm=np.asarray(lap["rpm"], dtype=np.float64),
    )
    position = PositionStream(
        time_s=np.asarray(lap["t_loc"], dtype=np.float64),
        x=np.asarray(lap["x"], dtype=np.float64),
        y=np.asarray(lap["y"], dtype=np.float64),
    )
    return Lap(car, position, lap_time_s=float(lap["lap_time_s"]), lap_id=code)


def _s_at(frame: LapFrame, time_s: float) -> float:
    finite = np.isfinite(frame.time_s) & np.isfinite(frame.s_m)
    if int(finite.sum()) < 2:
        raise DebriefError("the aligned lap has no time on the grid")
    clock = frame.time_s[finite]
    meters = frame.s_m[finite]
    order = np.argsort(clock)
    clock = clock[order]
    meters = meters[order]
    unique = np.concatenate([[True], np.diff(clock) > 1e-6])
    clock = clock[unique]
    meters = meters[unique]
    if time_s < float(clock[0]) or time_s > float(clock[-1]):
        raise DebriefError("the pass time is outside the aligned lap")
    return float(np.interp(time_s, clock, meters))


def _event_payload(event: PassEvent, lap_number: int) -> dict:
    payload = _event_dict(event)
    payload["lap_number"] = lap_number
    return payload


def _event_dict(event: PassEvent) -> dict:
    return {
        "date": event.date.isoformat(),
        "overtaking": event.overtaking,
        "overtaken": event.overtaken,
        "position": event.position,
    }
