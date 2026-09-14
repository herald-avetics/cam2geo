"""Put a fixed camera's pixels on the ground, through a plane homography."""

from __future__ import annotations

from .anchors import BOX_ANCHORS, anchor_pixel
from .demo import (
    SETUPS,
    Setup,
    coastal_mast,
    moorland_mast,
    pole_camera,
    wide_quay,
)
from .fit import (
    ALIGN_MODELS,
    FIT_METHODS,
    Alignment,
    HomographyFit,
    align_pixels,
    fit_homography,
)
from .frames import GeodeticPlane, MetricPlane, PlaneFrame
from .homography import (
    EARTH_RADIUS_M,
    CameraPose,
    CameraStation,
    GroundPoints,
    PlaneHomography,
    homography_from_pose,
)
from .lens import Lens
from .surface import SURFACES, ErrorBudget, Surface, position_error

__version__ = "0.1.0"

__all__ = [
    "ALIGN_MODELS",
    "BOX_ANCHORS",
    "EARTH_RADIUS_M",
    "FIT_METHODS",
    "SETUPS",
    "SURFACES",
    "Alignment",
    "CameraPose",
    "CameraStation",
    "ErrorBudget",
    "GeodeticPlane",
    "GroundPoints",
    "HomographyFit",
    "Lens",
    "MetricPlane",
    "PlaneFrame",
    "PlaneHomography",
    "Setup",
    "Surface",
    "__version__",
    "align_pixels",
    "anchor_pixel",
    "coastal_mast",
    "fit_homography",
    "homography_from_pose",
    "moorland_mast",
    "pole_camera",
    "position_error",
    "wide_quay",
]
