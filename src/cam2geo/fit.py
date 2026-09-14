"""Fit a homography from ground control points. See docs/INTERFACE.md."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import DEFAULT_FRAME, PlaneFrame
from .homography import PlaneHomography
from .lens import Lens, require_cv2

__all__ = ["HomographyFit", "FIT_METHODS", "fit_homography"]

FIT_METHODS = ("exact", "ransac", "lmeds")


@dataclass(frozen=True)
class HomographyFit:
    """A fitted homography and how well it fits.

    Attributes:
        homography: The result.
        residuals_px: Reprojection error per control point, in pixels.
        rms_px: Root mean square of the residuals over the inliers.
        max_px: Worst inlier residual.
        inliers: Which control points the fit kept.
    """

    homography: PlaneHomography
    residuals_px: np.ndarray
    rms_px: float
    max_px: float
    inliers: np.ndarray


def fit_homography(
    plane_points: np.ndarray,
    pixel_points: np.ndarray,
    width: int,
    height: int,
    *,
    lens: Lens | None = None,
    frame: PlaneFrame = DEFAULT_FRAME,
    ground_pixel: tuple[float, float] | None = None,
    method: str = "ransac",
    ransac_threshold_px: float = 3.0,
) -> HomographyFit:
    """Fit a plane-to-image homography to ground control points.

    Args:
        plane_points: ``(N, 2)`` known ground positions, in plane units.
        pixel_points: ``(N, 2)`` where each appears in the image, raw pixels.
        width, height: Frame size.
        lens: Undistorts ``pixel_points`` first, so the matrix lives in the same
            pixel space the rest of the library assumes.
        frame, ground_pixel: Passed to the resulting homography.
        method: One of ``FIT_METHODS``. ``exact`` is plain least squares.
        ransac_threshold_px: Inlier threshold, ``ransac`` only.

    Returns:
        The fit, including per-point residuals. A uniform tilt of the surface --
        runway crossfall, pitch camber -- is absorbed here for free; do not
        correct for it again.

    Raises:
        ValueError: If there are fewer than four points, the shapes disagree, the
            method is unknown, or the points are degenerate.
    """
    if method not in FIT_METHODS:
        raise ValueError("method must be one of " + ", ".join(FIT_METHODS)
                         + ", got " + repr(method))
    plane = np.asarray(plane_points, dtype=float).reshape(-1, 2)
    pixel = np.asarray(pixel_points, dtype=float).reshape(-1, 2)
    if len(plane) != len(pixel):
        raise ValueError("got " + str(len(plane)) + " plane points but "
                         + str(len(pixel)) + " pixel points")
    if len(plane) < 4:
        raise ValueError("a homography needs at least 4 control points, got "
                         + str(len(plane)))
    if not (np.isfinite(plane).all() and np.isfinite(pixel).all()):
        raise ValueError("control points must all be finite")

    cv2 = require_cv2()
    if lens is not None:
        iu, iv = lens.undistort(pixel[:, 0], pixel[:, 1])
        if not np.isfinite(iu).all():
            raise ValueError("the lens model has no inverse for "
                             + str(int(np.isnan(iu).sum())) + " control point(s)")
        pixel = np.column_stack([iu, iv])

    flags = {"exact": 0, "ransac": cv2.RANSAC, "lmeds": cv2.LMEDS}[method]
    matrix, mask = cv2.findHomography(plane, pixel, flags, ransac_threshold_px)
    if matrix is None:
        raise ValueError(
            "no homography fits these control points; they are probably "
            "collinear or otherwise degenerate"
        )
    # A plain least-squares fit has no inlier concept and OpenCV hands back an
    # all-zero mask for it, which would read as "every point rejected".
    inliers = np.ones(len(plane), dtype=bool)
    if method != "exact" and mask is not None:
        flagged = np.asarray(mask, dtype=bool).ravel()
        if flagged.any():
            inliers = flagged

    homography = PlaneHomography(matrix, width, height, lens=lens, frame=frame,
                                 ground_pixel=ground_pixel)
    # Residuals are measured where the matrix lives, which is undistorted pixels
    # when there is a lens -- and `pixel` was undistorted above to match.
    back = np.column_stack(_undistorted(homography, plane))
    residuals = np.hypot(*(back - pixel).T)
    kept = residuals[inliers]
    return HomographyFit(homography, residuals,
                         float(np.sqrt(np.mean(kept**2))), float(kept.max()), inliers)


def _undistorted(homography: PlaneHomography, plane: np.ndarray) -> tuple:
    """Reproject without re-applying distortion, to compare in undistorted pixels."""
    p = homography.matrix @ np.vstack([plane[:, 0], plane[:, 1], np.ones(len(plane))])
    return p[0] / p[2], p[1] / p[2]
