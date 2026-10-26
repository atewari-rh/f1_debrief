"""The adapter reads the two FastF1 clocks without merging them."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from f1frame.adapters.fastf1 import streams_from_fastf1_lap


class _Lap:
    def __init__(self, car, position, lap_time):
        self._car = car
        self._position = position
        self._lap_time = lap_time

    def get_car_data(self):
        return self._car

    def get_pos_data(self):
        return self._position

    def __getitem__(self, key):
        if key == "LapTime":
            return self._lap_time
        raise KeyError(key)


def test_fastf1_tenths_of_a_meter_become_meters():
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    car_time = [start, start + timedelta(seconds=0.2)]
    pos_time = [start, start + timedelta(milliseconds=120), start + timedelta(seconds=0.25)]
    car = pd.DataFrame(
        {
            "SessionTime": pd.to_timedelta([0.0, 0.2], unit="s"),
            "Date": car_time,
            "Speed": [300, 301],
            "Throttle": [100, 100],
            "Brake": [False, False],
            "nGear": [8, 8],
            "RPM": [11000, 11100],
        }
    )
    position = pd.DataFrame(
        {
            "SessionTime": pd.to_timedelta([0.0, 0.12, 0.25], unit="s"),
            "Date": pos_time,
            "X": [0.0, 1000.0, 2000.0],
            "Y": [0.0, 500.0, 500.0],
        }
    )
    streams_car, streams_pos, lap_time = streams_from_fastf1_lap(
        _Lap(car, position, timedelta(seconds=80.0))
    )

    assert lap_time == 80.0
    assert streams_car.time_s.shape != streams_pos.time_s.shape
    assert np.allclose(streams_pos.x, [0.0, 100.0, 200.0])
    assert np.allclose(streams_pos.y, [0.0, 50.0, 50.0])
    assert streams_car.brake.tolist() == [0.0, 0.0]
    assert streams_car.gear.tolist() == [8.0, 8.0]
