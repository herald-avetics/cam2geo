"""Fit a homography, from control points or from drift. See docs/INTERFACE.md."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .frames import DEFAULT_FRAME, PlaneFrame
from .homography import PlaneHomography
from .lens import Lens, require_cv2

__all__ = ["HomographyFit", "FIT_METHODS", "fit_homography", "Alignment",
           "ALIGN_MODELS", "align_pixels"]

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


def _solve(source: np.ndarray, target: np.ndarray, method: str, threshold: float
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the 3x3 taking ``source`` to ``target``, with its residuals."""
    if method not in FIT_METHODS:
        raise ValueError("method must be one of " + ", ".join(FIT_METHODS)
                         + ", got " + repr(method))
    if len(source) != len(target):
        raise ValueError("got " + str(len(source)) + " source points but "
                         + str(len(target)) + " target points")
    if len(source) < 4:
        raise ValueError("a homography needs at least 4 point pairs, got "
                         + str(len(source)))
    if not (np.isfinite(source).all() and np.isfinite(target).all()):
        raise ValueError("point pairs must all be finite")

    cv2 = require_cv2()
    flags = {"exact": 0, "ransac": cv2.RANSAC, "lmeds": cv2.LMEDS}[method]
    matrix, mask = cv2.findHomography(source, target, flags, threshold)
    if matrix is None:
        raise ValueError(
            "no homography fits these points; they are probably collinear or "
            "otherwise degenerate"
        )
    # A plain least-squares fit has no inlier concept and OpenCV hands back an
    # all-zero mask for it, which would read as "every point rejected".
    inliers = np.ones(len(source), dtype=bool)
    if method != "exact" and mask is not None:
        flagged = np.asarray(mask, dtype=bool).ravel()
        if flagged.any():
            inliers = flagged

    p = matrix @ np.vstack([source[:, 0], source[:, 1], np.ones(len(source))])
    residuals = np.hypot(*(np.column_stack([p[0] / p[2], p[1] / p[2]]) - target).T)
    return matrix, inliers, residuals


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
    plane = np.asarray(plane_points, dtype=float).reshape(-1, 2)
    pixel = np.asarray(pixel_points, dtype=float).reshape(-1, 2)
    if lens is not None:
        iu, iv = lens.undistort(pixel[:, 0], pixel[:, 1])
        if not np.isfinite(iu).all():
            raise ValueError("the lens model has no inverse for "
                             + str(int(np.isnan(iu).sum())) + " control point(s)")
        # Residuals are then measured where the matrix lives: undistorted pixels.
        pixel = np.column_stack([iu, iv])

    # Solve about the centroid, then fold the shift back into the matrix. Plane
    # coordinates are often degrees -- magnitude 50, spread 0.001 -- and OpenCV's
    # own normalisation does not survive that: fitting four marks in raw lon/lat
    # costs 0.66 px of pure arithmetic on a fit that is exact by construction,
    # against 5e-5 px centred. Residuals are unchanged, being the same map.
    centre = plane.mean(axis=0)
    matrix, inliers, residuals = _solve(plane - centre, pixel, method,
                                        ransac_threshold_px)
    shift = np.array([[1.0, 0.0, -centre[0]], [0.0, 1.0, -centre[1]],
                      [0.0, 0.0, 1.0]])
    homography = PlaneHomography(matrix @ shift, width, height, lens=lens,
                                 frame=frame, ground_pixel=ground_pixel)
    kept = residuals[inliers]
    return HomographyFit(homography, residuals,
                         float(np.sqrt(np.mean(kept**2))), float(kept.max()), inliers)


@dataclass(frozen=True)
class Alignment:
    """How far a fixed camera has drifted, and the map that undoes it.

    Attributes:
        transform: ``(3, 3)`` taking a reference-image pixel to where that same
            fixed feature sits now. Hand it to ``PlaneHomography.realigned``.
        residuals_px: Landmark error left after alignment, per landmark.
        rms_px: Root mean square of those residuals, over the inliers.
        max_px: Worst inlier residual.
        motion_px: RMS apparent motion of the landmarks *before* alignment --
            how far the image moved. Larger than ``drift_deg`` implies, because
            a rotation moves off-axis pixels further than on-axis ones.
        drift_deg: How far the camera turned, as a single angle about a single
            axis. Exact for a camera that rotated about its optical centre; for
            one that also shifted, the nearest rotation to what it did.
        residual_deg: ``rms_px`` read as an angle. What alignment could not
            explain, and so a defensible ``pointing_sigma_deg`` afterwards.
        inliers: Which landmarks the fit kept.
    """

    transform: np.ndarray
    residuals_px: np.ndarray
    rms_px: float
    max_px: float
    motion_px: float
    drift_deg: float
    residual_deg: float
    inliers: np.ndarray


ALIGN_MODELS = ("rotation", "homography")


def _rays(pixels: np.ndarray, lens: Lens) -> np.ndarray:
    """Unit viewing directions for pixels, i.e. the image with its lens taken off."""
    rays = np.linalg.inv(lens.matrix) @ np.vstack(
        [pixels[:, 0], pixels[:, 1], np.ones(len(pixels))])
    return rays / np.linalg.norm(rays, axis=0)


def _solve_rotation(reference: np.ndarray, current: np.ndarray, lens: Lens
                    ) -> np.ndarray:
    """The rotation best taking one set of viewing directions to the other.

    Wahba's problem, solved by Kabsch's SVD construction. A camera on a mast has
    three rotational freedoms, so estimating a general eight-parameter homography
    for it lets clicking noise flow into five parameters the camera does not
    have -- which reads back as drift that never happened. This is the
    constrained fit, and it is what makes a small drift measurable at all.
    """
    covariance = _rays(reference, lens) @ _rays(current, lens).T
    u, _, vt = np.linalg.svd(covariance)
    # Guard against the reflection SVD is equally happy to return.
    flip = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
    return vt.T @ flip @ u.T


def _rotation_deg(transform: np.ndarray, lens: Lens) -> float:
    """The angle of the camera rotation a pixel-to-pixel transform implies.

    A camera rotating about its optical centre induces exactly ``K R K^-1`` on
    the image, whatever the scene's depth, so undoing the intrinsics recovers the
    rotation. A transform that is not purely rotational -- the camera shifted, or
    the landmarks are noisy -- gives a matrix that is merely close to one, and
    the orthogonal factor of its polar decomposition is the nearest rotation to
    it. Scale drops out there too, which matters: a homography is only ever
    defined up to one.
    """
    k = lens.matrix
    m = np.linalg.inv(k) @ transform @ k
    u, _, vt = np.linalg.svd(m)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        rotation = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(rotation) - 1.0) / 2.0))))


def _reproject(transform: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    p = transform @ np.vstack([pixels[:, 0], pixels[:, 1], np.ones(len(pixels))])
    return np.column_stack([p[0] / p[2], p[1] / p[2]])


def _align_rotation(reference: np.ndarray, current: np.ndarray, lens: Lens,
                    method: str, threshold: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a rotation, then once more without the landmarks that disagree."""
    if method not in FIT_METHODS:
        raise ValueError("method must be one of " + ", ".join(FIT_METHODS)
                         + ", got " + repr(method))
    if len(reference) != len(current):
        raise ValueError("got " + str(len(reference)) + " source points but "
                         + str(len(current)) + " target points")
    if len(reference) < 2:
        raise ValueError("a rotation needs at least 2 point pairs, got "
                         + str(len(reference)))
    if not (np.isfinite(reference).all() and np.isfinite(current).all()):
        raise ValueError("point pairs must all be finite")

    k = lens.matrix
    inliers = np.ones(len(reference), dtype=bool)
    for _ in range(2 if method != "exact" else 1):
        rotation = _solve_rotation(reference[inliers], current[inliers], lens)
        transform = k @ rotation @ np.linalg.inv(k)
        residuals = np.hypot(*(_reproject(transform, reference) - current).T)
        flagged = residuals <= threshold
        if flagged.sum() < 2:
            break
        inliers = flagged
    return transform, inliers, residuals


def align_pixels(
    reference_px: np.ndarray,
    current_px: np.ndarray,
    *,
    lens: Lens,
    model: str = "rotation",
    method: str = "ransac",
    ransac_threshold_px: float = 3.0,
) -> Alignment:
    """Measure a fixed camera's drift from landmarks that have not moved.

    Boresighting without a survey. The landmarks need no known coordinates and
    need never have been control points -- a chimney, a mooring, the corner of a
    sea wall -- because only their motion *within the image* is measured. This is
    the same estimation ``fit_homography`` performs, with the reference image
    standing in for the ground plane.

    Args:
        reference_px: ``(N, 2)`` landmark pixels in the image the homography was
            fitted against.
        current_px: ``(N, 2)`` the same landmarks, same order, in today's image.
        lens: The camera's intrinsics. Distortion is removed from both sets
            first, because raw pixels of a rotated camera are *not* related by a
            homography but undistorted ones are; and the focal length and centre
            turn the transform into an angle. With no calibration to hand,
            ``Lens(f, f, width / 2, height / 2)`` is enough for the angle.
        model: What the camera is allowed to have done. ``rotation`` gives it the
            three freedoms a camera on a mount actually has and needs only two
            landmarks; ``homography`` gives it eight, which also covers a camera
            that shifted, but lets clicking noise into five parameters the camera
            does not have. **Prefer the default.** With six landmarks and half a
            pixel of clicking error, the eight-parameter fit reports around a
            degree of drift for a camera that has not moved, which is larger
            than most of the drift worth correcting.
        method: One of ``FIT_METHODS``. ``exact`` keeps every landmark; the other
            two refit once without those beyond ``ransac_threshold_px``, which is
            what catches the mooring that dragged.
        ransac_threshold_px: Inlier threshold. Ignored by ``exact``.

    Returns:
        The alignment. Read ``rms_px`` before using it: a large residual means a
        landmark moved, a pair was mismatched, or the camera did something a
        homography cannot describe.

    Raises:
        ValueError: If the model or method is unknown, there are too few
            landmarks for the model, the shapes disagree, the lens has no inverse
            at a landmark, or the points are degenerate.
    """
    if model not in ALIGN_MODELS:
        raise ValueError("model must be one of " + ", ".join(ALIGN_MODELS)
                         + ", got " + repr(model))
    reference = np.asarray(reference_px, dtype=float).reshape(-1, 2)
    current = np.asarray(current_px, dtype=float).reshape(-1, 2)
    if not lens.is_identity:
        pairs = [np.column_stack(lens.undistort(p[:, 0], p[:, 1]))
                 for p in (reference, current)]
        if not all(np.isfinite(p).all() for p in pairs):
            raise ValueError("the lens model has no inverse at some landmark(s)")
        reference, current = pairs

    if model == "homography":
        matrix, inliers, residuals = _solve(reference, current, method,
                                            ransac_threshold_px)
    else:
        matrix, inliers, residuals = _align_rotation(
            reference, current, lens, method, ransac_threshold_px)
    kept = residuals[inliers]
    rms = float(np.sqrt(np.mean(kept**2)))
    motion = float(np.sqrt(np.mean(
        np.sum((current - reference)[inliers] ** 2, axis=1))))
    return Alignment(
        transform=matrix,
        residuals_px=residuals,
        rms_px=rms,
        max_px=float(kept.max()),
        motion_px=motion,
        drift_deg=_rotation_deg(matrix, lens),
        residual_deg=math.degrees(math.atan(rms / lens.fx)),
        inliers=inliers,
    )
