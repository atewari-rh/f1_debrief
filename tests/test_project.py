"""Projection stays on the lap instead of jumping across the circuit."""

import numpy as np

from f1frame import project_s, reference_from_xy


def test_closed_loop_shorter_than_the_search_window_starts_at_zero():
    theta = np.linspace(0.0, 2.0 * np.pi, 90, endpoint=False)
    radius = 20.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    reference = reference_from_xy(x, y)

    s_m, dist = project_s(x, y, reference, lookahead_m=250.0)

    assert reference.length_m < 250.0
    assert s_m[0] < 2.0
    assert s_m[-1] > reference.length_m - (reference.length_m / x.shape[0]) - 1.0
    finite = s_m[np.isfinite(s_m)]
    assert np.all(np.diff(finite) >= -1e-9)
    assert np.nanmax(dist) < 1e-6


def test_parallel_straight_does_not_steal_a_nearby_sample():
    outbound_y = np.linspace(0.0, 200.0, 41)
    outbound = np.column_stack([np.zeros_like(outbound_y), outbound_y])
    theta = np.linspace(np.pi, 0.0, 12, endpoint=False)
    turn = np.column_stack([12.5 + 12.5 * np.cos(theta), 200.0 + 12.5 * np.sin(theta)])
    return_y = np.linspace(200.0, 0.0, 41)
    back = np.column_stack([np.full_like(return_y, 25.0), return_y])
    points = np.vstack([outbound, turn, back])
    reference = reference_from_xy(points[:, 0], points[:, 1])

    # Step along the outbound straight, then one sample 2 m from the return
    # straight. That return segment is inside the long search window and is
    # the closer piece of geometry. The match should stay on the outbound.
    x = np.array([0.0, 0.0, 0.0, 23.0])
    y = np.array([0.0, 50.0, 100.0, 100.0])
    s_m, dist = project_s(x, y, reference, max_advance_m=80.0, max_dist_m=40.0)

    assert abs(s_m[2] - 100.0) < 1.0
    assert s_m[3] < 180.0
    assert dist[3] > 20.0


def test_vertex_arc_length_is_recovered():
    x = np.array([0.0, 10.0, 20.0, 30.0])
    y = np.array([0.0, 0.0, 5.0, 5.0])
    reference = reference_from_xy(x, y)
    s_m, dist = project_s(x, y, reference)
    assert np.allclose(s_m, reference.s_m)
    assert np.nanmax(dist) < 1e-6
