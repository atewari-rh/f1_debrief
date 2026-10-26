"""Align Formula 1 laps onto one geometric meter grid."""

from f1frame.align import align, align_laps, brake_onsets, speed_delta
from f1frame.delta import SpatialTimeDelta, gap_between, spatial_time_delta
from f1frame.geometry import Corner, corner_at, corner_window, extract_corners
from f1frame.io import read_frame, write_frame
from f1frame.project import project_s
from f1frame.reference import build_reference, reference_from_xy
from f1frame.types import (
    SCHEMA_VERSION,
    CarStream,
    F1FrameError,
    FrameReport,
    Lap,
    LapFrame,
    PositionStream,
    ReferenceLine,
)

__all__ = [
    "SCHEMA_VERSION",
    "CarStream",
    "Corner",
    "F1FrameError",
    "FrameReport",
    "Lap",
    "LapFrame",
    "PositionStream",
    "ReferenceLine",
    "SpatialTimeDelta",
    "align",
    "align_laps",
    "brake_onsets",
    "build_reference",
    "corner_at",
    "corner_window",
    "extract_corners",
    "gap_between",
    "project_s",
    "read_frame",
    "reference_from_xy",
    "spatial_time_delta",
    "speed_delta",
    "write_frame",
]

__version__ = "0.1.0"
