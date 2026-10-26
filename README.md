# f1frame

f1frame puts every lap on one geometric ruler.

Formula 1 car channels and position channels arrive on two clocks. The distance column most plots use is the integral of speed, and that integral drifts. Corner markers are a separate layer meant for pictures. Projects that compare drivers each build their own meter grid, and the braking points do not land on the same piece of track.

This library takes the raw streams and returns one shared grid. Speed, throttle, brake, and gear are resampled by meters from the start of a reference line. Gear and brake keep their last real value. The frame also reports how far its clock sits from the official lap time, and how far a speed integral would have drifted.

Public position samples are a single normalized path, so the frame has no lateral-offset column. It does not estimate energy, active aero, or whether a driver pressed Overtake Mode.

## Why this exists

FastF1, OpenF1, and tif1 already deliver the car stream and the position stream. Projects that compare two drivers usually integrate speed onto a private distance axis, so two of them can disagree about which meter a brake point was on. f1frame is that missing grid, plus the report of how far the frame sits from the official lap time. The Parquet file carries the meters, so another tool can use the result without importing this package. `f1-debrief` is one consumer.

## Install

```bash
pip install -e ".[dev]"
```

Run that from this repository, not from `debrief/`. One install provides the library and the `f1-debrief` command. These packages are not on PyPI. Installing `debrief/` by itself asks PyPI for `f1frame` and stops.

Use the `fastf1` extra when you want the adapter:

```bash
pip install -e ".[fastf1]"
```

Python 3.11 or newer.

## Repository

`src/f1frame` is the grid. `debrief/` is the approach card. The recorded Monza pass is `debrief/src/f1debrief/fixtures/monza_2024_race_pass.json`. Tests for the grid are in `tests/`. Tests for the card are in `debrief/tests/`. `pytest` from this directory runs both.

`build/`, `*.egg-info/`, `__pycache__/`, and `.venv/` are produced by install and test runs. They repeat the sources under `src/` and `debrief/src/`. `.gitignore` leaves them out of git. CI installs with `pip install -e ".[dev]"` and runs `pytest` on Python 3.11 and 3.12.

## Compare two laps

```python
from f1frame import Lap, align_laps, brake_onsets, extract_corners, spatial_time_delta, speed_delta

frames = align_laps(
    [
        Lap(ver_car, ver_position, lap_time_s=ver_lap_time, lap_id="VER"),
        Lap(lec_car, lec_position, lap_time_s=lec_lap_time, lap_id="LEC"),
    ],
    step_m=10,
)

frames[0].s_m          # 0, 10, 20, ... shared by both laps
frames[0].speed_kph
brake_onsets(frames[0])
speed_delta(frames[0], frames[1])
frames[0].report.timestamp_error_ms
frames[0].report.integrated_speed_error_ms

corners = extract_corners(frames[0].reference)
delta = spatial_time_delta(frames[0], frames[1])
delta.closure_error_ms
```

`extract_corners` groups the reference line by curvature. The radius is in the meters of the timing map, and those samples are coarse, so this is a way to choose a window rather than a survey of the corner. `spatial_time_delta` is the time gap along the shared grid. `closure_error_ms` is how far that integral misses the official lap-time gap. On a timing map the miss includes the scale of the map.

`car` and `position` are `CarStream` and `PositionStream`. Their `time_s` arrays must share an epoch. They do not need to share sample times. Coordinates are meters. Brake may be 0/1 or 0/100.

A synthetic run of that comparison is in `examples/compare_two_laps.py`.

## FastF1

Call `get_car_data()` and `get_pos_data()`, not `get_telemetry()`. The merged telemetry object has already interpolated both clocks.

```python
import fastf1

from f1frame import Lap, align_laps
from f1frame.adapters.fastf1 import streams_from_fastf1_lap

session = fastf1.get_session(2024, "Monza", "Q")
session.load(telemetry=True, weather=False, messages=False)

laps = []
for code in ("VER", "LEC"):
    lap = session.laps.pick_driver(code).pick_fastest()
    car, position, lap_time = streams_from_fastf1_lap(lap)
    laps.append(Lap(car, position, lap_time_s=lap_time, lap_id=code))

frames = align_laps(laps, step_m=10)
```

The adapter multiplies FastF1 X and Y by 0.1 because those channels are published in tenths of a meter. OpenF1 `location` uses the same units, so divide those by 10 as well. FastF1 3.9 asks for `F1_USERNAME` and `F1_PASSWORD` before it will download the timing feed.

## The report

| Field | Meaning |
|---|---|
| `timestamp_error_ms` | Position-stream duration minus the official lap time. A large value means the streams are not the lap the timing system measured. |
| `integrated_speed_error_ms` | Lap time from integrating recorded speed along the grid, minus the official lap time. Alignment does not use this number. It is `None` when the integral covers less than 95% of the reference. On a real lap this also absorbs the scale of the timing map. |
| `loop_closure_m` | Distance between the first and last position sample. |
| `start_offset_m` | Distance from the first sample to the reference origin. |
| `projection_max_m` | Farthest raw sample from the reference. A pit lane or an off-track moment shows up here. |
| `n_rejected` | Samples that were not matched, because they sat outside the search window. |
| `covered_fraction` | Share of grid rows with a speed value. |

## Parquet

```python
from f1frame import read_frame, write_frame

write_frame("ver.parquet", frames[0])
frame = read_frame("ver.parquet")
```

Schema version is `"1"`. The table columns are `s_m`, `time_s`, `speed_kph`, `throttle_pct`, `brake`, `gear`, and `rpm`. The report and the reference polyline are stored in the file metadata.

## How a sample is placed

1. Build one reference polyline. One lap uses that lap. Several laps use the median shape. Every lap must run the same direction and start at the same part of the circuit.
2. Walk each position sample forward along that line. The start/finish samples of a closed lap sit next to both ends of the line, so the first sample is forced toward s = 0, and later samples cannot jump to a nearer part of the circuit that is farther along the lap.
3. Attach that arc length to each car sample by time.
4. Resample onto `0, step, 2·step, ...`. Speed, throttle, and rpm are linear between samples. Brake and gear hold the last value. Where two car samples are more than `max(100 m, 4·step)` apart, the meters between them stay empty.

The public position stream is a timing map. FastF1 documents the coordinates as tenths of a meter, and they are consistent from car to car, but the arc length is not the published circuit length. On the 2024 Monza qualifying laps the map came out at about 4.84 km, against a circuit length of 5.793 km. The grid is still the comparison axis. It is not a survey of the asphalt.

`max_advance_m` (default 80) is the farthest a sample may jump ahead while a closer continuation still exists. Raise it when the position stream is coarser than about 4 Hz at race speed.

## A recorded race pass

```bash
f1-debrief pass --recorded
```

This reads a committed slice of the 2024 Monza race, OpenF1 session 9590, lap 41. Norris passes Verstappen. Both laps go through one `align_laps` call. Curvature of that shared line picks the corner, and pit stops and penalty messages beside the pass are left out.

On this recording Norris brakes at 730 m and Verstappen at 750 m, into the first corner on the map. The shared line for these two laps is about 5.72 km. Each speed integral is about 2 s short of that driver's official lap time. The closure of the gap between them is 224 ms, and the card prints it. The official lap times are 82.234 s and 88.738 s. The card stays on the one published path.

The first on-track pass between this pair is the lap above. OpenF1 historical sessions do not need an F1 login:

```bash
f1-debrief pass --year 2024 --race Monza --drivers NOR VER
```

`f1-debrief approach --demo` is still the synthetic check, with an 8% fast speed channel underneath so the integral's brake point moves and the geometric one stays put.

## Debrief

The approach card is documented in `debrief/README.md`. It is part of this repository and is installed with the library.

## Tests

```bash
pytest
```
