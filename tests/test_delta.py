"""The spatial gap reports its miss against the official lap-time gap."""

from dataclasses import replace

from f1frame import F1FrameError, Lap, align_laps, spatial_time_delta
from f1frame.delta import gap_between
from f1frame.synthetic import sample_constant_lap, stadium_polyline


def _frames(left_time: float | None, right_time: float | None):
    x, y = stadium_polyline(straight_m=400.0, radius_m=40.0, spacing_m=5.0)
    left = sample_constant_lap(x, y, speed_kph=200.0, brake_at_s=200.0, hz=20.0)
    right = sample_constant_lap(x, y, speed_kph=180.0, brake_at_s=200.0, hz=20.0)
    return align_laps(
        [
            Lap(left.car, left.position, lap_time_s=left_time if left_time is not None else left.lap_time_s, lap_id="L"),
            Lap(right.car, right.position, lap_time_s=right_time if right_time is not None else right.lap_time_s, lap_id="R"),
        ],
        step_m=10.0,
    )


def test_slower_car_is_behind_and_closure_is_small_when_times_match():
    x, y = stadium_polyline(straight_m=400.0, radius_m=40.0, spacing_m=5.0)
    left = sample_constant_lap(x, y, speed_kph=200.0, brake_at_s=200.0, hz=20.0)
    right = sample_constant_lap(x, y, speed_kph=180.0, brake_at_s=200.0, hz=20.0)
    frames = align_laps(
        [
            Lap(left.car, left.position, lap_time_s=left.lap_time_s, lap_id="L"),
            Lap(right.car, right.position, lap_time_s=right.lap_time_s, lap_id="R"),
        ],
        step_m=10.0,
    )
    delta = spatial_time_delta(frames[0], frames[1])

    assert delta.integrated_gap_s > 1.0
    assert delta.covered_fraction > 0.95
    assert delta.closure_error_ms is not None
    assert abs(delta.closure_error_ms) < 400.0
    window = gap_between(delta, 150.0, 400.0)
    assert window is not None and window > 0.0


def test_closure_is_the_miss_against_the_official_gap():
    x, y = stadium_polyline(straight_m=400.0, radius_m=40.0, spacing_m=5.0)
    left = sample_constant_lap(x, y, speed_kph=200.0, brake_at_s=200.0, hz=20.0)
    right = sample_constant_lap(x, y, speed_kph=180.0, brake_at_s=200.0, hz=20.0)
    frames = align_laps(
        [
            Lap(left.car, left.position, lap_time_s=left.lap_time_s, lap_id="L"),
            Lap(right.car, right.position, lap_time_s=left.lap_time_s, lap_id="R"),
        ],
        step_m=10.0,
    )
    delta = spatial_time_delta(frames[0], frames[1])

    assert delta.closure_error_ms is not None
    assert delta.closure_error_ms > 1000.0


def test_closure_is_absent_without_an_official_time():
    x, y = stadium_polyline(straight_m=400.0, radius_m=40.0, spacing_m=5.0)
    left = sample_constant_lap(x, y, speed_kph=200.0, brake_at_s=200.0, hz=20.0)
    right = sample_constant_lap(x, y, speed_kph=180.0, brake_at_s=200.0, hz=20.0)
    frames = align_laps(
        [
            Lap(left.car, left.position, lap_time_s=None, lap_id="L"),
            Lap(right.car, right.position, lap_time_s=right.lap_time_s, lap_id="R"),
        ],
        step_m=10.0,
    )
    delta = spatial_time_delta(frames[0], frames[1])
    assert delta.closure_error_ms is None
    assert delta.integrated_gap_s > 0.0


def test_delta_rejects_a_different_grid():
    frames = _frames(None, None)
    shortened = replace(frames[1], s_m=frames[1].s_m[:-1])
    try:
        spatial_time_delta(frames[0], shortened)
    except F1FrameError:
        return
    raise AssertionError("expected F1FrameError")
