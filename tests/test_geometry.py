"""Corners from curvature, on a polyline whose bends are known."""

import numpy as np
from f1frame import corner_at, corner_window, extract_corners, reference_from_xy
from f1frame.synthetic import stadium_polyline


def test_stadium_has_two_left_bends():
    x, y = stadium_polyline()
    corners = extract_corners(reference_from_xy(x, y))

    assert len(corners) == 2
    assert [corner.direction for corner in corners] == ["left", "left"]
    assert all(50.0 < corner.min_radius_m < 130.0 for corner in corners)
    assert 850 < corners[0].apex_s < 1050
    assert 1850 < corners[1].apex_s < 2150


def test_a_straight_has_no_corners():
    reference = reference_from_xy(np.linspace(0.0, 500.0, 40), np.zeros(40))
    assert extract_corners(reference) == ()


def test_clockwise_quarter_circle_is_a_right_turn():
    theta = np.linspace(np.pi / 2, -np.pi / 2, 80)
    radius = 60.0
    x = np.concatenate([np.linspace(-40.0, 0.0, 15, endpoint=False), radius * np.cos(theta)])
    y = np.concatenate([np.full(15, radius), radius * np.sin(theta)])
    corners = extract_corners(reference_from_xy(x, y), min_radius_m=120.0, min_length_m=20.0)

    assert len(corners) == 1
    assert corners[0].direction == "right"
    assert 40.0 < corners[0].min_radius_m < 90.0


def test_corner_at_prefers_the_bend_ahead_of_the_pass():
    x, y = stadium_polyline()
    reference = reference_from_xy(x, y)
    corners = extract_corners(reference)
    chosen = corner_at(corners, corners[0].entry_s - 100.0)

    assert chosen is not None
    assert chosen.index == 1
    start, end = corner_window(chosen, corners[0].entry_s - 100.0, reference.length_m)
    assert start <= corners[0].entry_s - 100.0
    assert end >= corners[0].exit_s
