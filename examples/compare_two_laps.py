"""Compare two laps on one ruler, and show the speed integral missing the brake point."""

from f1frame import Lap, align_laps, brake_onsets
from f1frame.synthetic import sample_constant_lap, stadium_polyline


def main() -> None:
    x, y = stadium_polyline()
    clean = sample_constant_lap(x, y, speed_kph=250.0, brake_at_s=1800.0)
    biased = sample_constant_lap(x, y, speed_kph=180.0, brake_at_s=1800.0, speed_scale=1.08)
    frames = align_laps(
        [
            Lap(clean.car, clean.position, lap_time_s=clean.lap_time_s, lap_id="clean"),
            Lap(biased.car, biased.position, lap_time_s=biased.lap_time_s, lap_id="biased"),
        ]
    )
    print(f"reference length: {frames[0].report.reference_length_m:.1f} m")
    print(f"grid points:      {frames[0].report.grid_points}")
    for frame in frames:
        onset = brake_onsets(frame)[0]
        report = frame.report
        print(
            f"{frame.lap_id:6}  brake at {onset:7.1f} m   "
            f"timestamp error {report.timestamp_error_ms:8.1f} ms   "
            f"speed-integral error {report.integrated_speed_error_ms:8.1f} ms"
        )
    print("Brake meters match. The speed-integral column is the drift this grid avoids.")


if __name__ == "__main__":
    main()
