"""Synthetic camera setups, so every example in the docs is runnable."""

from __future__ import annotations

from dataclasses import dataclass

from .frames import MetricPlane
from .homography import CameraPose, CameraStation
from .lens import Lens
from .surface import SURFACES, Surface

__all__ = ["Setup", "SETUPS", "coastal_mast", "moorland_mast", "pole_camera",
           "wide_quay"]


@dataclass(frozen=True)
class Setup:
    """A named camera and the surface it looks at.

    Attributes:
        name: Short identifier.
        description: What it stands for.
        station: An installed camera, pose-derived so it can be re-aimed.
        surface: A starting-point ``Surface`` for that site.
    """

    name: str
    description: str
    station: CameraStation
    surface: Surface

    @property
    def view(self):  # noqa: ANN201
        """The station's ``PlaneHomography``, for examples that need only the map."""
        return self.station.view


def _setup(name: str, description: str, height_m: float, tilt_deg: float,
           focal_px: float, surface: Surface, width: int = 1920,
           height: int = 1080, k1: float = 0.0) -> Setup:
    pose = CameraPose(height_m=height_m, x=0.0, y=0.0, tilt_deg=tilt_deg, yaw_deg=0.0)
    lens = Lens(focal_px, focal_px, width / 2.0, height / 2.0, k1=k1)
    station = CameraStation.from_pose(pose, lens, width, height, MetricPlane(), name)
    return Setup(name, description, station, surface)


def coastal_mast() -> Setup:
    """A headland camera watching a bay: 30 m up, grazing, wide."""
    return _setup("coastal_mast", "30 m mast over the sea, 8 deg down-tilt",
                  30.0, 8.0, 1400.0, SURFACES["sea"])


def moorland_mast() -> Setup:
    """A telephoto view across open moor, where the horizon sits near the centre."""
    return _setup("moorland_mast", "12 m mast over moorland, 3 deg down-tilt, long lens",
                  12.0, 3.0, 2800.0, SURFACES["moorland"])


def pole_camera() -> Setup:
    """A steeply tilted pole camera over a hard surface: the easy case."""
    return _setup("pole_camera", "8 m pole over a yard, 20 deg down-tilt",
                  8.0, 20.0, 1000.0, SURFACES["runway"])


def wide_quay() -> Setup:
    """A wide-angle camera with real barrel distortion, for the lens path.

    Deliberately outside ``SETUPS``: projecting through it needs OpenCV, and the
    other three keep the documentation runnable on a numpy-only install.
    """
    return _setup("wide_quay", "6 m wide-angle camera over a quay, k1 = -0.15",
                  6.0, 25.0, 900.0, SURFACES["runway"], k1=-0.15)


#: The undistorted setups, which need nothing but numpy.
SETUPS = (coastal_mast, moorland_mast, pole_camera)
