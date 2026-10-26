"""The meter grid lines brake points up where integrated speed does not."""

import numpy as np

from f1frame import (
    F1FrameError,
    Lap,
    align,
    align_laps,
    brake_onsets,
    build_reference,
    speed_delta,
)
from f1frame.synthetic import sample_constant_lap, stadium_polyline
from f1frame.types import CarStream, PositionStream


def _straight_streams():
    s = np.arange(0.0, 1001.0, 20.0)
    time_s = s / 50.0
    gear = np.where(s >= 500.0, 4.0, 2.0)
    brake = np.where(s >= 700.0, 1.0, 0.0)
    car = CarStream(
        time_s=time_s,
        speed_kph=np.full_like(s, 180.0),
        throttle_pct=np.where(brake > 0.5, 0.0, 100.0),
        brake=brake,
        gear=gear,
        rpm=np.full_like(s, 11000.0),
    )
    position = PositionStream(time_s=time_s, x=s, y=np.zeros_like(s))
    return car, position


def test_discrete_channels_do_not_invent_intermediate_values():
    car, position = _straight_streams()
    frame = align(car, position, lap_time_s=float(car.time_s[-1]), step_m=10.0)

    gear_490 = frame.gear[np.where(frame.s_m == 490.0)[0][0]]
    gear_500 = frame.gear[np.where(frame.s_m == 500.0)[0][0]]
    brake_690 = frame.brake[np.where(frame.s_m == 690.0)[0][0]]
    brake_700 = frame.brake[np.where(frame.s_m == 700.0)[0][0]]

    assert gear_490 == 2.0
    assert gear_500 == 4.0
    assert 3.0 not in frame.gear
    assert brake_690 == 0.0
    assert brake_700 == 1.0
    assert brake_onsets(frame).tolist() == [700.0]
    assert abs(frame.report.timestamp_error_ms) < 1e-6
    assert abs(frame.report.integrated_speed_error_ms) < 200.0


def test_shared_grid_keeps_brake_points_together_when_speed_is_biased():
    x, y = stadium_polyline()
    clean = sample_constant_lap(x, y, speed_kph=250.0, brake_at_s=1800.0, speed_scale=1.0, gear=8)
    biased = sample_constant_lap(x, y, speed_kph=180.0, brake_at_s=1800.0, speed_scale=1.08, gear=6)
    frames = align_laps(
        [
            Lap(clean.car, clean.position, lap_time_s=clean.lap_time_s, lap_id="clean"),
            Lap(biased.car, biased.position, lap_time_s=biased.lap_time_s, lap_id="biased"),
        ],
        step_m=10.0,
    )

    assert np.array_equal(frames[0].s_m, frames[1].s_m)
    onset_clean = brake_onsets(frames[0])[0]
    onset_biased = brake_onsets(frames[1])[0]
    assert abs(onset_clean - onset_biased) <= 10.0

    brake_index = int(np.flatnonzero(biased.car.brake > 0.5)[0])
    integrated = _integrated_distance(biased.car.time_s, biased.car.speed_kph)
    assert abs(integrated[brake_index] - onset_biased) > 80.0

    assert abs(frames[1].report.timestamp_error_ms) < 1.0
    assert frames[1].report.integrated_speed_error_ms < -1000.0
    assert frames[0].report.covered_fraction > 0.95
    delta = speed_delta(frames[0], frames[1])
    assert np.nanmax(delta) > 50.0


def test_offset_position_clock_still_aligns():
    x, y = stadium_polyline()
    lap = sample_constant_lap(x, y, speed_kph=220.0, brake_at_s=1800.0, pos_time_offset_s=0.01)
    frame = align(lap.car, lap.position, lap_time_s=lap.lap_time_s, step_m=10.0)
    onset = brake_onsets(frame)[0]
    assert abs(onset - 1800.0) <= 15.0
    assert frame.report.n_rejected == 0


def test_sample_far_from_the_line_is_rejected_and_the_lap_continues():
    car, position = _straight_streams()
    x = position.x.copy()
    y = position.y.copy()
    x[5] = 500.0
    y[5] = 100.0
    moved = PositionStream(time_s=position.time_s, x=x, y=y)
    reference = build_reference(position)
    frame = align(car, moved, reference=reference, lap_time_s=float(car.time_s[-1]), step_m=10.0)
    assert frame.report.n_rejected == 1
    assert frame.report.projection_max_m < 1.0
    assert frame.report.covered_fraction > 0.9


def test_openf1_brake_scale_is_stored_as_off_on():
    time_s = np.array([0.0, 1.0, 2.0])
    car = CarStream(
        time_s=time_s,
        speed_kph=np.array([100.0, 100.0, 100.0]),
        throttle_pct=np.array([100.0, 0.0, 100.0]),
        brake=np.array([0.0, 100.0, 0.0]),
        gear=np.array([3.0, 3.0, 3.0]),
    )
    assert car.brake.tolist() == [0.0, 1.0, 0.0]


def test_empty_lap_list_is_rejected():
    try:
        align_laps([])
    except F1FrameError:
        return
    raise AssertionError("expected F1FrameError")


def _integrated_distance(time_s: np.ndarray, speed_kph: np.ndarray) -> np.ndarray:
    dt = np.diff(time_s)
    v = speed_kph / 3.6
    step = 0.5 * (v[:-1] + v[1:]) * dt
    return np.concatenate([[0.0], np.cumsum(step)])
