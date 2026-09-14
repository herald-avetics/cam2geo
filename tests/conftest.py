"""Shared fixtures: one synthetic camera whose ground truth we know exactly.

The camera is an ideal pinhole 10 m above a metric plane, looking north, tilted
20 degrees below horizontal, on a 1920x1080 frame at 1000 px focal length. Plane
coordinates are metres, x east and y north, with the camera above the origin.

Those numbers put the horizon at image row 176.03, which is what the NaN and
conditioning tests need. Working in metres rather than degrees means a one-pixel
finite difference keeps full double precision, so tolerances stay tight.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cam2geo import CameraPose, CameraStation, Lens, MetricPlane, PlaneHomography

WIDTH, HEIGHT = 1920, 1080
FOCAL_PX = 1000.0
TILT_DEG = 20.0
CAMERA_HEIGHT_M = 10.0

#: Image row the horizon falls on, from the pinhole geometry alone.
HORIZON_ROW = HEIGHT / 2.0 - FOCAL_PX * math.tan(math.radians(TILT_DEG))

#: The intrinsics every fixture camera is built with.
LENS = Lens(FOCAL_PX, FOCAL_PX, WIDTH / 2.0, HEIGHT / 2.0)


def build_station(tilt_deg: float = TILT_DEG, roll_deg: float = 0.0,
                  yaw_deg: float = 0.0, height_m: float = CAMERA_HEIGHT_M,
                  x: float = 0.0, y: float = 0.0) -> CameraStation:
    """The reference installation, with anything worth varying exposed."""
    pose = CameraPose(height_m=height_m, x=x, y=y, tilt_deg=tilt_deg,
                      yaw_deg=yaw_deg, roll_deg=roll_deg)
    return CameraStation.from_pose(pose, LENS, WIDTH, HEIGHT, MetricPlane(), "reference")


def build_camera(tilt_deg: float = TILT_DEG, roll_deg: float = 0.0,
                 height_m: float = CAMERA_HEIGHT_M) -> PlaneHomography:
    """Just the view, for tests that need no camera position."""
    return build_station(tilt_deg=tilt_deg, roll_deg=roll_deg, height_m=height_m).view


@pytest.fixture
def station() -> CameraStation:
    """The reference installation: 10 m up, 20 degrees down, looking north."""
    return build_station()


@pytest.fixture
def camera() -> PlaneHomography:
    """The reference view. Horizon at row 176.03, ground below it."""
    return build_camera()


@pytest.fixture
def centre_column() -> np.ndarray:
    """Rows down the middle of the frame, from the bottom edge to the horizon."""
    return np.array([1079.0, 900.0, 700.0, 500.0, 300.0, 200.0, 180.0])
