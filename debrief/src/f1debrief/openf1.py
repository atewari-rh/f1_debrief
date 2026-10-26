"""OpenF1 reads. Historical sessions do not need an F1 account."""

from __future__ import annotations

import json
import re
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from f1debrief.select import PassEvent, Penalty, PitStop

_BASE = "https://api.openf1.org/v1"
_CAR = re.compile(r"CAR\s+(\d+)")


class DebriefError(Exception):
    """A session, pass, or stream the dossier cannot build."""


def fetch_json(path: str, params: list[tuple[str, str]]) -> list[dict]:
    url = f"{_BASE}/{path}"
    if params:
        url = f"{url}?{_query(params)}"
    request = Request(url, headers={"User-Agent": "f1-debrief", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise DebriefError(f"OpenF1 {path} returned {exc.code}") from exc
    except URLError as exc:
        raise DebriefError(f"OpenF1 {path} could not be reached ({exc.reason})") from exc
    if isinstance(payload, dict):
        raise DebriefError(f"OpenF1 {path} returned an error: {payload}")
    return payload


def _query(params: list[tuple[str, str]]) -> str:
    parts = []
    for key, value in params:
        encoded = quote(str(value), safe="")
        if key.endswith((">=", "<=", ">", "<")):
            parts.append(f"{key}{encoded}")
        else:
            parts.append(f"{key}={encoded}")
    return "&".join(parts)


def find_race(year: int, race: str) -> dict:
    sessions = fetch_json("sessions", [("year", str(year)), ("session_name", "Race")])
    needle = race.casefold().strip()
    hits = [session for session in sessions if needle in _haystack(session)]
    if len(hits) == 1:
        return hits[0]
    names = sorted({str(session.get("location") or session.get("country_name")) for session in sessions})
    if not hits:
        raise DebriefError(f"No {year} race matches {race!r}. Races that year: {', '.join(names)}.")
    which = ", ".join(str(session.get("location")) for session in hits)
    raise DebriefError(f"{race!r} matches more than one race ({which}). Use the circuit name.")


def session_index(session_key: int) -> dict:
    """Drivers, overtakes, pit stops, and penalty messages for one session."""

    drivers = fetch_json("drivers", [("session_key", str(session_key))])
    code_by_number = {
        int(driver["driver_number"]): str(driver.get("name_acronym") or driver["driver_number"])
        for driver in drivers
    }
    overtakes = []
    for row in fetch_json("overtakes", [("session_key", str(session_key))]):
        overtakes.append(
            PassEvent(
                date=_parse_time(row["date"]),
                overtaking=code_by_number.get(int(row["overtaking_driver_number"]), str(row["overtaking_driver_number"])),
                overtaken=code_by_number.get(int(row["overtaken_driver_number"]), str(row["overtaken_driver_number"])),
                position=None if row.get("position") is None else int(row["position"]),
            )
        )
    pits = []
    for row in fetch_json("pit", [("session_key", str(session_key))]):
        duration = _duration(row)
        pits.append(
            PitStop(
                date=_parse_time(row["date"]),
                driver=code_by_number.get(int(row["driver_number"]), str(row["driver_number"])),
                duration_s=duration,
            )
        )
    penalties = []
    for row in fetch_json("race_control", [("session_key", str(session_key))]):
        message = str(row.get("message") or "")
        if "PENALTY" not in message.upper() and "PENALTY" not in str(row.get("category") or "").upper():
            continue
        when = _parse_time(row["date"])
        named = _penalty_drivers(row, message, code_by_number)
        penalties.extend(Penalty(date=when, driver=driver) for driver in named)
    return {
        "drivers": code_by_number,
        "overtakes": overtakes,
        "pits": pits,
        "penalties": penalties,
    }


def laps_for(session_key: int, driver_number: int) -> list[dict]:
    rows = fetch_json(
        "laps",
        [("session_key", str(session_key)), ("driver_number", str(driver_number))],
    )
    return sorted(rows, key=lambda row: row["date_start"])


def telemetry(session_key: int, driver_number: int, start: datetime, end: datetime) -> tuple[list[dict], list[dict]]:
    params = [
        ("session_key", str(session_key)),
        ("driver_number", str(driver_number)),
        ("date>=", _iso(start)),
        ("date<", _iso(end)),
    ]
    return fetch_json("car_data", params), fetch_json("location", params)


def _haystack(session: dict) -> str:
    fields = ("location", "country_name", "circuit_short_name", "country_code")
    return " ".join(str(session.get(field) or "") for field in fields).casefold()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _iso(value: datetime) -> str:
    return value.isoformat()


def _duration(row: dict) -> float:
    for key in ("pit_duration", "lane_duration"):
        value = row.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return 25.0


def _penalty_drivers(row: dict, message: str, code_by_number: dict[int, str]) -> list[str]:
    number = row.get("driver_number")
    if isinstance(number, int):
        return [code_by_number.get(number, str(number))]
    found = []
    for match in _CAR.findall(message.upper()):
        driver_number = int(match)
        found.append(code_by_number.get(driver_number, str(driver_number)))
    return found
