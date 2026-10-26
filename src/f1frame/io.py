"""Read and write a lap frame as a Parquet file.

The grid is the table. The error report and the reference polyline are stored
in the file metadata under the schema version, so a downstream project can
reload a frame without the original telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from f1frame.types import SCHEMA_VERSION, F1FrameError, FrameReport, LapFrame, ReferenceLine

_META_KEY = b"f1frame"
_COLUMNS = ("s_m", "time_s", "speed_kph", "throttle_pct", "brake", "gear", "rpm")


def write_frame(path: str | Path, frame: LapFrame) -> None:
    """Write one aligned lap. The parent directory must already exist."""

    path = Path(path)
    columns = {name: getattr(frame, name) for name in _COLUMNS}
    table = pa.table(columns)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "lap_id": frame.lap_id,
        "report": _report_dict(frame.report),
        "reference": {
            "x": frame.reference.x.tolist(),
            "y": frame.reference.y.tolist(),
            "s_m": frame.reference.s_m.tolist(),
        },
    }
    table = table.replace_schema_metadata({_META_KEY: json.dumps(payload).encode("utf-8")})
    pq.write_table(table, path)


def read_frame(path: str | Path) -> LapFrame:
    """Load a frame written by :func:`write_frame`."""

    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    raw = meta.get(_META_KEY)
    if raw is None:
        raise F1FrameError("parquet file has no f1frame metadata")
    payload = json.loads(raw.decode("utf-8"))
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise F1FrameError(f"unsupported f1frame schema {version!r}; this package reads {SCHEMA_VERSION!r}")
    data = {name: table.column(name).to_numpy(zero_copy_only=False).astype(np.float64, copy=False) for name in _COLUMNS}
    ref = payload["reference"]
    return LapFrame(
        reference=ReferenceLine(
            np.asarray(ref["x"], dtype=np.float64),
            np.asarray(ref["y"], dtype=np.float64),
            np.asarray(ref["s_m"], dtype=np.float64),
        ),
        report=_report_from_dict(payload["report"]),
        lap_id=payload.get("lap_id"),
        **data,
    )


def _report_dict(report: FrameReport) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in report.__dataclass_fields__:
        value = getattr(report, name)
        if isinstance(value, float):
            out[name] = None if not np.isfinite(value) else value
        else:
            out[name] = value
    return out


def _report_from_dict(payload: dict[str, object]) -> FrameReport:
    numbers = {
        "step_m",
        "reference_length_m",
        "official_lap_time_s",
        "timestamp_elapsed_s",
        "timestamp_error_ms",
        "integrated_speed_time_s",
        "integrated_speed_distance_m",
        "integrated_speed_error_ms",
        "loop_closure_m",
        "start_offset_m",
        "projection_median_m",
        "projection_max_m",
        "covered_fraction",
    }
    kwargs: dict[str, object] = {}
    for name in FrameReport.__dataclass_fields__:
        value = payload[name]
        if name in numbers and value is not None:
            value = float(value)
        kwargs[name] = value
    return FrameReport(**kwargs)  # type: ignore[arg-type]
