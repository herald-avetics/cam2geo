"""What the ground does to a plane model, priced. See docs/LIMITATIONS.md."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .homography import EARTH_RADIUS_M, CameraStation

__all__ = ["Surface", "ErrorBudget", "position_error", "SURFACES"]


@dataclass(frozen=True)
class Surface:
    """How far a real surface departs from the plane fitted to it.

    Every field is a property of your site and conditions, not of the terrain
    type. The presets in ``SURFACES`` are starting points to be replaced, and
    each names the source its figure should come from.

    Attributes:
        name: For reports.
        offset_m: Typical height of a target's base above the plane -- swell,
            crop, cargo. A bias, correctable via ``project(target_height_m=)``.
        offset_sigma_m: Variability of that height. Not correctable.
        roughness_m: Residual relief of the surface itself after fitting.
        curvature: Whether the earth's curvature is worth modelling. True over
            open water, false almost everywhere else.
        refraction_k: Coefficient of refraction, reducing curvature by ``1 - k``.
            Zero assumes none, which overstates curvature.
    """

    name: str = "plane"
    offset_m: float = 0.0
    offset_sigma_m: float = 0.0
    roughness_m: float = 0.0
    curvature: bool = False
    refraction_k: float = 0.0

    def __post_init__(self) -> None:
        for field in ("offset_sigma_m", "roughness_m"):
            value = float(getattr(self, field))
            if value < 0.0:
                raise ValueError(field + " must not be negative, got " + repr(value))
        if not 0.0 <= self.refraction_k < 1.0:
            raise ValueError("refraction_k must be in [0, 1), got "
                             + repr(self.refraction_k))

    def but(self, **changes) -> Surface:
        """A copy with fields replaced -- how a preset is adapted to a site."""
        return replace(self, **changes)


# Starting points, not constants. Replace every figure with one measured or read
# off the standard named in docs/LIMITATIONS.md before quoting a result.
SURFACES: dict[str, Surface] = {
    "sea": Surface("sea", offset_sigma_m=0.5, roughness_m=0.5, curvature=True),
    "tidal_flat": Surface("tidal_flat", offset_sigma_m=0.1, roughness_m=0.1),
    "salt_flat": Surface("salt_flat", roughness_m=0.05),
    "moorland": Surface("moorland", roughness_m=1.0),
    "runway": Surface("runway", roughness_m=0.03),
    # offset_m stays 0: what is stacked on a yard is a property of the job, not
    # of the surface. Set it per site with .but(offset_m=...).
    "container_yard": Surface("container_yard", roughness_m=0.03),
    "pitch": Surface("pitch", roughness_m=0.02),
    "warehouse_floor": Surface("warehouse_floor", roughness_m=0.005),
}


@dataclass(frozen=True)
class ErrorBudget:
    """Per-pixel position error, itemised so the dominant term is visible.

    Every field is **metres of horizontal position error**, one entry per input
    pixel, named for the cause it comes from -- not to be confused with the
    ``Surface`` inputs of similar name, which are heights. ``total_m`` adds the
    terms in quadrature, which assumes they are independent.

    Biases and random errors are kept apart on purpose. Quadrature -- the GUM
    law of propagation of uncertainty -- applies to independent, zero-mean
    random terms. A known mean swell height and the earth's curvature are
    neither: they are deterministic, they point the same way every time, and
    combining them in quadrature would understate them. So biases add linearly,
    random terms add in quadrature, and ``total_m`` is the sum of the two.
    See docs/LIMITATIONS.md, "How the terms combine".

    Attributes:
        range_m: Ground distance from the camera. Every term but ``pixel_m``
            and ``control_m`` scales with it.
        pixel_m: Random, from detection or annotation error.
        height_bias_m: Bias, from ``Surface.offset_m`` -- a target whose base
            is known to sit off the plane. Correctable.
        height_sigma_m: Random, from ``Surface.offset_sigma_m``, the part of
            that height you do not know. Not correctable.
        roughness_m: Random, from the surface departing from the fitted plane.
        curvature_m: Bias, from the earth falling below the tangent plane.
            Correctable.
        pointing_m: Random, from the camera not pointing where the fit assumed.
        control_m: Random, from the fit's own reprojection error.
        bias_m: The bias terms, added linearly.
        random_m: The random terms, added in quadrature.
        total_m: ``bias_m + random_m``.
    """

    range_m: np.ndarray
    pixel_m: np.ndarray
    height_bias_m: np.ndarray
    height_sigma_m: np.ndarray
    roughness_m: np.ndarray
    curvature_m: np.ndarray
    pointing_m: np.ndarray
    control_m: np.ndarray
    bias_m: np.ndarray
    random_m: np.ndarray
    total_m: np.ndarray


def position_error(
    station: CameraStation,
    u: np.ndarray,
    v: np.ndarray,
    surface: Surface,
    *,
    pixel_sigma_px: float = 1.0,
    pointing_sigma_deg: float = 0.0,
    control_rms_px: float = 0.0,
    corrected_height: bool = False,
    corrected_curvature: bool = False,
) -> ErrorBudget:
    """Price every error term at each pixel, for one surface.

    Args:
        station: The installed camera. Every term but the pixel ones scales with
            range, which is why a bare homography is not enough.
        u, v: Pixels to price.
        surface: The site's departures from the plane.
        pixel_sigma_px: Detection or annotation error, in pixels. One pixel is a
            reasonable starting point for a hand-drawn box.
        pointing_sigma_deg: How far the camera's true orientation may differ from
            the one the fit assumed -- mast sway, thermal drift, a knocked
            bracket. Also called boresight error. Measure it by tracking a fixed
            landmark across frames: ``degrees(atan(pixels_moved / focal_px))``.
        control_rms_px: The fit's own reprojection error. Take it straight from
            ``HomographyFit.rms_px``.
        corrected_height: True if ``CameraStation.project(target_height_m=)``
            already removed the ``Surface.offset_m`` bias, leaving only its
            spread.
        corrected_curvature: True if ``project(curvature=True)`` already removed
            the curvature bias.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    placed = station.view.project(u, v)
    rng = station.range_m(u, v)
    gsd = placed.ground_m_per_px
    # A height error h displaces the point by h * range / camera height: the ray
    # is a straight line and this is its similar triangle.
    lever = rng / station.height_m

    drop = ((1.0 - surface.refraction_k) * rng**2 / (2.0 * EARTH_RADIUS_M)
            if surface.curvature else np.zeros(np.shape(rng)))

    biases = {
        "height_bias_m": (0.0 if corrected_height else abs(surface.offset_m)) * lever,
        "curvature_m": (np.zeros(np.shape(rng)) if corrected_curvature
                        else drop * lever),
    }
    randoms = {
        "pixel_m": pixel_sigma_px * gsd,
        "height_sigma_m": surface.offset_sigma_m * lever,
        "roughness_m": surface.roughness_m * lever,
        # d(range)/d(depression) = (range^2 + height^2) / height.
        "pointing_m": (math.radians(pointing_sigma_deg)
                       * (rng**2 + station.height_m**2) / station.height_m),
        "control_m": control_rms_px * gsd,
    }
    terms = {k: np.broadcast_to(np.asarray(t, dtype=float), np.shape(rng)).copy()
             for k, t in {**biases, **randoms}.items()}

    zero = np.zeros(np.shape(rng))
    bias = sum((terms[k] for k in biases), start=zero)
    random = np.sqrt(sum((terms[k] ** 2 for k in randoms), start=zero))
    return ErrorBudget(range_m=rng, bias_m=bias, random_m=random,
                       total_m=bias + random, **terms)
