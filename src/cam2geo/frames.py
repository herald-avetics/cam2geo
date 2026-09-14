"""Plane coordinates to metres. See docs/INTERFACE.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["PlaneFrame", "MetricPlane", "GeodeticPlane", "DEFAULT_FRAME"]


@runtime_checkable
class PlaneFrame(Protocol):
    """How a homography's output units relate to metres on the ground."""

    def metres_per_unit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """(N, 2, 2) converting a small displacement in plane units to metres."""

    def distance_m(self, x0: np.ndarray, y0: np.ndarray,
                   x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
        """Ground distance in metres between two plane coordinates."""


@dataclass(frozen=True)
class MetricPlane:
    """The homography already maps to a linear ground unit.

    Attributes:
        metres_per_plane_unit: 1.0 when the unit is the metre.
    """

    metres_per_plane_unit: float = 1.0

    def __post_init__(self) -> None:
        if not self.metres_per_plane_unit > 0.0:
            raise ValueError(
                "metres_per_plane_unit must be positive, got "
                + repr(self.metres_per_plane_unit)
            )

    def metres_per_unit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = np.eye(2) * self.metres_per_plane_unit
        return np.broadcast_to(m, (*np.shape(x), 2, 2)).copy()

    def distance_m(self, x0: np.ndarray, y0: np.ndarray,
                   x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
        return np.hypot(np.asarray(x1) - np.asarray(x0),
                        np.asarray(y1) - np.asarray(y0)) * self.metres_per_plane_unit


@dataclass(frozen=True)
class GeodeticPlane:
    """The homography maps to WGS-84 degrees, x latitude and y longitude.

    Attributes:
        height_m: Ellipsoidal height the plane sits at.
    """

    height_m: float = 0.0

    def _pm(self):  # noqa: ANN202
        try:
            import pymap3d
        except ImportError as exc:
            raise ImportError(
                "GeodeticPlane needs pymap3d: pip install 'cam2geo[geodetic]'. "
                "A homography that maps to metres needs MetricPlane instead."
            ) from exc
        return pymap3d

    def metres_per_unit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pm = self._pm()
        lat, lon = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        # One degree is far too large to difference over; 1e-7 deg is ~1 cm, which
        # is inside the linear regime and still well clear of double precision.
        d = 1e-7
        north = pm.geodetic2ned(lat + d, lon, self.height_m, lat, lon, self.height_m)[0]
        east = pm.geodetic2ned(lat, lon + d, self.height_m, lat, lon, self.height_m)[1]
        out = np.zeros((*lat.shape, 2, 2))
        out[..., 0, 0] = np.asarray(north) / d
        out[..., 1, 1] = np.asarray(east) / d
        return out

    def distance_m(self, x0: np.ndarray, y0: np.ndarray,
                   x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
        pm = self._pm()
        n, e, _ = pm.geodetic2ned(np.asarray(x1, dtype=float), np.asarray(y1, dtype=float),
                                  self.height_m, np.asarray(x0, dtype=float),
                                  np.asarray(y0, dtype=float), self.height_m)
        return np.hypot(np.asarray(n), np.asarray(e))


#: The default frame: the homography maps straight to metres. A module-level
#: singleton because a frozen dataclass in a signature default is shared anyway.
DEFAULT_FRAME = MetricPlane()
