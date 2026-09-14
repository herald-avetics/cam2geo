"""Brown-Conrady undistortion, via OpenCV. See docs/INTERFACE.md."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Lens", "require_cv2"]

# OpenCV's inverse is iterative and its default iteration count is low enough to
# return silently short of converged, so criteria are always passed explicitly.
_ITERATIONS = 100
_TOLERANCE = 1e-12
# OpenCV reports nothing when the inverse diverges, so a forward reprojection is
# what actually backs the NaN contract. See docs/LIMITATIONS.md, "The lens".
_ROUND_TRIP_PX = 1e-3


def require_cv2():  # noqa: ANN201
    """Import OpenCV on demand, with an actionable message when it is absent."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "this needs OpenCV: pip install 'cam2geo[lens]'. The projection path "
            "itself does not -- a PlaneHomography with no Lens stays on numpy."
        ) from exc
    return cv2


@dataclass(frozen=True)
class Lens:
    """The intrinsics an image was undistorted with before a homography was fitted.

    Coefficient names and order follow OpenCV: k1, k2, p1, p2, k3.

    Attributes:
        fx, fy: Focal length in pixels.
        cx, cy: Principal point in absolute pixels.
        k1, k2, k3: Radial terms.
        p1, p2: Tangential terms.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def __post_init__(self) -> None:
        for name in ("fx", "fy"):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise ValueError(name + " must be positive, got " + repr(value))

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 intrinsic matrix, linear part only."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def coefficients(self) -> np.ndarray:
        """The five distortion terms in OpenCV's order."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=float)

    @property
    def is_identity(self) -> bool:
        """True when no distortion is modelled, so OpenCV is never needed."""
        return not self.coefficients.any()

    def undistort(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map real pixels to where an ideal pinhole would put them.

        Returns:
            Ideal pixels, NaN wherever the inverse did not round-trip. One output
            per input, always.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if self.is_identity or u.size == 0:
            return u.copy(), v.copy()

        cv2 = require_cv2()
        k, d = self.matrix, self.coefficients
        src = np.stack([u.ravel(), v.ravel()], axis=-1).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS,
                    _ITERATIONS, _TOLERANCE)
        if hasattr(cv2, "undistortPointsIter"):
            ideal = cv2.undistortPointsIter(src, k, d, None, None, criteria)
        else:
            ideal = cv2.undistortPoints(src, k, d, None, None, None, criteria)
        ideal = np.asarray(ideal, dtype=float).reshape(-1, 2)

        ok = np.isfinite(ideal).all(axis=1)
        back = np.full_like(ideal, np.nan)
        if ok.any():
            obj = np.column_stack([ideal[ok], np.ones(int(ok.sum()))]).reshape(-1, 1, 3)
            zero = np.zeros(3)
            back[ok] = np.asarray(
                cv2.projectPoints(obj, zero, zero, k, d)[0], dtype=float
            ).reshape(-1, 2)
        good = ok & (np.abs(back - src.reshape(-1, 2)) <= _ROUND_TRIP_PX).all(axis=1)

        out = np.where(good[:, None], ideal, np.nan)
        out[:, 0] = out[:, 0] * self.fx + self.cx
        out[:, 1] = out[:, 1] * self.fy + self.cy
        return out[:, 0].reshape(u.shape), out[:, 1].reshape(v.shape)
