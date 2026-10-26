"""The approach card uses the shared grid, and a bad clock does not hide it."""

from f1frame import F1FrameError
from f1debrief.audit import approach_card
from f1debrief.cli import main
from f1debrief.confidence import judge_reports
from f1debrief.demo import demo_pass
from f1debrief.dossier import render_html, render_text
from f1frame.types import FrameReport


def test_shared_grid_keeps_brake_points_and_shows_integral_drift():
    demo = demo_pass()
    by_id = {driver.lap_id: driver for driver in demo.card.drivers}

    assert abs((by_id["VER"].brake_m or 0) - 1800) <= 10
    assert abs((by_id["NOR"].brake_m or 0) - 1830) <= 10
    assert abs((by_id["VER"].apex_speed_kph or 0) - 112) <= 2
    assert abs((by_id["NOR"].apex_speed_kph or 0) - 104) <= 2
    assert abs(demo.integrated_brake_m["NOR"] - 1830) > 80
    assert demo.card.confidence.level == "high"
    assert demo.card.confidence.brake_points_verified

    text = render_text(demo.card)
    assert "later than" in text
    assert "on the inside" in text
    assert "down the inside" not in text
    assert "Lap-time closure:" in text
    assert demo.card.closure_error_ms is not None
    assert abs(demo.card.closure_error_ms) < 500
    html = render_html(demo.card)
    assert "Confidence: high" in html
    assert "on the inside" in html
    assert "Lap-time closure:" in html


def test_mismatched_grids_are_rejected():
    demo = demo_pass()
    shortened = _shorten(demo.frames[1])
    try:
        approach_card([demo.frames[0], shortened], 1700, 2100)
    except F1FrameError:
        return
    raise AssertionError("expected F1FrameError")


def test_large_clock_error_marks_the_card_unverified():
    report = FrameReport(
        schema_version="1",
        step_m=10.0,
        reference_length_m=5000.0,
        official_lap_time_s=80.0,
        timestamp_elapsed_s=81.2,
        timestamp_error_ms=1200.0,
        integrated_speed_time_s=None,
        integrated_speed_distance_m=0.0,
        integrated_speed_error_ms=None,
        loop_closure_m=5.0,
        start_offset_m=1.0,
        projection_median_m=0.2,
        projection_max_m=2.0,
        n_position_samples=200,
        n_projected=200,
        n_rejected=0,
        grid_points=500,
        covered_fraction=0.99,
    )
    badge = judge_reports([("VER", report), ("NOR", report)])
    assert badge.level == "low"
    assert not badge.brake_points_verified
    assert any("unverified" in reason for reason in badge.reasons)


def test_cli_demo(capsys):
    assert main(["approach", "--demo"]) == 0
    out = capsys.readouterr().out
    assert "Confidence: high" in out
    assert "Speed-integral brake point" in out


def _shorten(frame):
    from dataclasses import replace

    return replace(frame, s_m=frame.s_m[:-1])
