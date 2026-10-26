"""Parquet round trip keeps the grid, the report, and the reference."""

from pathlib import Path

import numpy as np

from f1frame import Lap, align_laps, read_frame, write_frame
from f1frame.synthetic import sample_constant_lap, stadium_polyline


def test_round_trip(tmp_path: Path):
    x, y = stadium_polyline()
    lap = sample_constant_lap(x, y, speed_kph=240.0, brake_at_s=1800.0)
    frame = align_laps([Lap(lap.car, lap.position, lap_time_s=lap.lap_time_s, lap_id="A")])[0]
    path = tmp_path / "A.parquet"
    write_frame(path, frame)
    loaded = read_frame(path)

    assert loaded.lap_id == "A"
    assert loaded.report.schema_version == "1"
    assert np.allclose(loaded.s_m, frame.s_m)
    assert np.allclose(loaded.brake, frame.brake, equal_nan=True)
    assert np.allclose(loaded.gear, frame.gear, equal_nan=True)
    assert np.allclose(loaded.reference.s_m, frame.reference.s_m)
    assert loaded.report.timestamp_error_ms == frame.report.timestamp_error_ms
    assert loaded.report.integrated_speed_error_ms == frame.report.integrated_speed_error_ms
