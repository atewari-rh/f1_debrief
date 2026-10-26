"""A recorded 2024 Monza race pass. The test does not download anything."""

import json
from datetime import datetime

from f1frame import align_laps
from f1debrief.cli import main
from f1debrief.dossier import render_text
from f1debrief.openf1 import _query
from f1debrief.race import _lap_from_json, card_from_recorded, recorded_fixture_path
from f1debrief.select import PassEvent, Penalty, PitStop, on_track_passes, passes_between


def test_recorded_monza_pass_locks_brakes_and_the_official_gap():
    payload = json.loads(recorded_fixture_path().read_text(encoding="utf-8"))
    assert payload["source"] == "openf1"
    assert payload["session_key"] == 9590
    assert payload["laps"]["NOR"]["lap_time_s"] == 82.234
    assert payload["laps"]["VER"]["lap_time_s"] == 88.738
    assert round(payload["laps"]["VER"]["lap_time_s"] - payload["laps"]["NOR"]["lap_time_s"], 3) == 6.504

    when = datetime.fromisoformat(payload["pass"]["date"])
    overtakes = [
        PassEvent(datetime.fromisoformat(row["date"]), row["overtaking"], row["overtaken"], row.get("position"))
        for row in payload["overtakes"]
    ]
    pits = [
        PitStop(datetime.fromisoformat(row["date"]), row["driver"], float(row["duration_s"]))
        for row in payload["pits"]
    ]
    penalties = [Penalty(datetime.fromisoformat(row["date"]), row["driver"]) for row in payload["penalties"]]
    kept = on_track_passes(overtakes, pits, penalties)
    assert len(kept) < len(overtakes)
    pair = passes_between(kept, "NOR", "VER")
    assert any(event.overtaking == "NOR" and event.date == when for event in pair)

    frames = align_laps(
        [_lap_from_json(code, payload["laps"][code]) for code in ("VER", "NOR")],
        step_m=10.0,
    )
    assert (frames[0].s_m == frames[1].s_m).all()
    for frame in frames:
        assert frame.report.n_rejected == 0
        assert frame.report.integrated_speed_error_ms is not None
        assert frame.report.integrated_speed_error_ms < -1000
        assert frame.report.timestamp_error_ms is not None
        assert abs(frame.report.timestamp_error_ms) < 400

    card = card_from_recorded()
    by_id = {driver.lap_id: driver for driver in card.drivers}
    assert by_id["NOR"].brake_m == 730
    assert by_id["VER"].brake_m == 750
    assert card.corner_index == 1
    assert card.pass_s_m is not None and card.start_m <= card.pass_s_m <= card.end_m
    assert card.closure_error_ms is not None and abs(card.closure_error_ms - 224) < 5
    assert card.confidence.level == "high"

    text = render_text(card)
    assert "NOR passes VER" in text
    assert "Corner 1" in text
    assert "Lap-time closure:" in text
    assert "on the inside" in text


def test_cli_recorded_pass(capsys):
    assert main(["pass", "--recorded"]) == 0
    out = capsys.readouterr().out
    assert "730" in out
    assert "750" in out
    assert "Lap-time closure:" in out


def test_pass_command_requires_a_session(capsys):
    assert main(["pass"]) == 2
    assert "recorded" in capsys.readouterr().err


def test_date_filter_is_a_single_operator():
    query = _query(
        [
            ("session_key", "9590"),
            ("date>=", "2024-09-01T14:00:30+00:00"),
            ("date<", "2024-09-01T14:01:58+00:00"),
        ]
    )
    assert "date>==" not in query
    assert "date>=2024-09-01" in query
    assert "date<2024-09-01" in query
    assert query.startswith("session_key=9590")
