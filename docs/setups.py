"""The synthetic cameras the error tables and figures are computed from.

    uv run python docs/make_error_tables.py
    uv run --group docs python docs/figures/make_figures.py

They live here rather than in the library because nothing but this documentation
needs them, and a camera nobody has installed has no business shipping in a
wheel. For a camera on real ground, see scripts/project_points.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from cam2geo import (
    SURFACES,
    CameraPose,
    CameraStation,
    Lens,
    MetricPlane,
    Surface,
)

__all__ = ["SETUPS", "Setup", "coastal_mast", "moorland_mast", "pole_camera"]


@dataclass(frozen=True)
class Setup:
    name: str
    description: str
    station: CameraStation
    surface: Surface


def _setup(name: str, description: str, height_m: float, tilt_deg: float,
           focal_px: float, surface: Surface, width: int = 1920,
           height: int = 1080) -> Setup:
    pose = CameraPose(height_m=height_m, x=0.0, y=0.0, tilt_deg=tilt_deg, yaw_deg=0.0)
    lens = Lens(focal_px, focal_px, width / 2.0, height / 2.0)
    station = CameraStation.from_pose(pose, lens, width, height, MetricPlane(), name)
    return Setup(name, description, station, surface)


def coastal_mast() -> Setup:
    return _setup("coastal_mast", "30 m mast over the sea, 8 deg down-tilt",
                  30.0, 8.0, 1400.0, SURFACES["sea"])


def moorland_mast() -> Setup:
    return _setup("moorland_mast", "12 m mast over moorland, 3 deg down-tilt, long lens",
                  12.0, 3.0, 2800.0, SURFACES["moorland"])


def pole_camera() -> Setup:
    return _setup("pole_camera", "8 m pole over a yard, 20 deg down-tilt",
                  8.0, 20.0, 1000.0, SURFACES["runway"])


SETUPS = (coastal_mast, moorland_mast, pole_camera)
