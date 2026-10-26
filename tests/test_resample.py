"""Gaps in the car stream stay empty instead of becoming a straight line."""

import numpy as np

from f1frame.resample import hold_last, interp_with_gaps


def test_wide_gap_keeps_the_samples_and_drops_the_span():
    s_m = np.array([0.0, 100.0])
    speed = np.array([10.0, 20.0])
    grid = np.array([0.0, 10.0, 50.0, 100.0])

    out = interp_with_gaps(s_m, speed, grid, max_gap_m=40.0)

    assert out[0] == 10.0
    assert out[3] == 20.0
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_brake_hold_stops_after_the_gap_limit():
    s_m = np.array([0.0, 100.0])
    brake = np.array([1.0, 0.0])
    grid = np.arange(0.0, 101.0, 10.0)

    out = hold_last(s_m, brake, grid, max_gap_m=40.0)

    assert out[0] == 1.0
    assert out[4] == 1.0
    assert np.isnan(out[5])
    assert out[-1] == 0.0
