"""Invert a plane-to-image homography. See docs/LIMITATIONS.md for why each rule."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from .frames import DEFAULT_FRAME, MetricPlane, PlaneFrame
from .lens import Lens, require_cv2

__all__ = ["PlaneHomography", "GroundPoints", "CameraPose", "CameraStation",
           "homography_from_pose", "EARTH_RADIUS_M"]

# IUGG mean radius. Only the curvature correction uses it.
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class GroundPoints:
    """Pixels placed on the plane. One entry per input pixel, always.

    Attributes:
        x, y: Plane coordinates. NaN where ``valid`` is False.
        ground_m_per_px: Metres of ground one pixel moves the point, worse axis.
            NaN off the plane; inf within a pixel of where the plane or the lens
            model ends.
        valid: On the plane and no worse conditioned than the cutoff.
        beyond_horizon: How many pixels were on the sky side.
        outside_lens_model: How many had no undistorted position.
        ill_conditioned: How many were on the plane but over the cutoff.
    """

    x: np.ndarray
    y: np.ndarray
    ground_m_per_px: np.ndarray
    valid: np.ndarray
    beyond_horizon: int
    outside_lens_model: int
    ill_conditioned: int


@dataclass(frozen=True, eq=False)
class PlaneHomography:
    """A fixed camera's plane-to-image homography, ready to invert.

    Attributes:
        matrix: 3x3 mapping plane coordinates ``(x, y, 1)`` to pixels -- to
            undistorted pixels when ``lens`` is given.
        width, height: Frame size in pixels.
        lens: The intrinsics the image was undistorted with before the matrix was
            fitted, or None when it applies to raw pixels.
        frame: How plane units relate to metres.
        ground_pixel: A pixel known to see the ground, which fixes which side of
            the horizon is ground. Defaults to the bottom centre of the frame.
    """

    matrix: np.ndarray
    width: int
    height: int
    lens: Lens | None = None
    frame: PlaneFrame = DEFAULT_FRAME
    ground_pixel: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float)
        if m.shape != (3, 3) or not np.isfinite(m).all():
            raise ValueError("matrix must be a finite 3x3, got " + repr(m.tolist()))
        if np.linalg.matrix_rank(m) < 3:
            raise ValueError("matrix is singular: " + repr(m.tolist()))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame size must be positive, got "
                             + str(self.width) + "x" + str(self.height))
        object.__setattr__(self, "matrix", m)

    # -- the matrix itself ------------------------------------------------

    @cached_property
    def inverse(self) -> np.ndarray:
        """Pixels to plane coordinates, up to scale."""
        return np.linalg.inv(self.matrix)

    @property
    def anchor_pixel(self) -> tuple[float, float]:
        """The pixel that defines the ground side of the horizon."""
        if self.ground_pixel is not None:
            return float(self.ground_pixel[0]), float(self.ground_pixel[1])
        return self.width / 2.0, self.height - 1.0

    @cached_property
    def ground_sign(self) -> float:
        """Sign of the third homogeneous coordinate on the ground side.

        A homography is defined up to scale, sign included, so nothing in the
        matrix says which side of the horizon is ground until a pixel known to
        see it does. Nothing checks that the anchor really does -- an anchor
        above the horizon inverts every answer silently. See
        docs/LIMITATIONS.md, "What fixes the ground side".

        Raises:
            ValueError: If the anchor pixel has no undistorted position, or sits
                within a pixel of the horizon, where its side is not decidable.
        """
        u, v = self._ideal(*(np.array([c]) for c in self.anchor_pixel))
        line = self.inverse[2]
        w = float(line @ np.array([u[0], v[0], 1.0]))
        scale = math.hypot(line[0], line[1])
        if not np.isfinite(w):
            raise ValueError(
                "the lens model has no inverse at " + repr(self.anchor_pixel)
                + ", so the ground side of the horizon cannot be fixed; pass "
                "ground_pixel=(u, v) for a pixel it can place"
            )
        if scale > 0.0 and abs(w) / scale < 1.0:
            raise ValueError(
                "the pixel " + repr(self.anchor_pixel) + " is within a pixel of the "
                "horizon, so which side of it is ground is not decidable; pass "
                "ground_pixel=(u, v) for one clearly on the ground"
            )
        return float(np.sign(w))

    @cached_property
    def horizon_line(self) -> np.ndarray:
        """``(a, b, c)`` with ``a*u + b*v + c`` positive on the ground side.

        Normalised so the value is a signed distance in pixels. This is the
        vanishing line of the plane.
        """
        line = self.inverse[2] * self.ground_sign
        return line / math.hypot(line[0], line[1])

    @cached_property
    def reference_point(self) -> tuple[float, float]:
        """Plane coordinates of the anchor pixel's ground point."""
        x, y = self._to_plane(*(np.array([c]) for c in self.anchor_pixel))
        return float(x[0]), float(y[0])

    def fingerprint(self) -> str:
        """A stable hash of everything that decides where a pixel lands."""
        lens = None if self.lens is None else [
            self.lens.fx, self.lens.fy, self.lens.cx, self.lens.cy,
            *self.lens.coefficients.tolist(),
        ]
        tree = {"matrix": self.matrix.tolist(), "width": int(self.width),
                "height": int(self.height), "lens": lens,
                "frame": repr(self.frame), "ground_pixel": self.ground_pixel}
        text = json.dumps(tree, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(text.encode("ascii")).hexdigest()

    # -- pixels to ground -------------------------------------------------

    def _ideal(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.lens is None:
            return u, v
        return self.lens.undistort(u, v)

    def _to_plane(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        iu, iv = self._ideal(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        g = self.inverse @ np.vstack([iu.ravel(), iv.ravel(), np.ones(iu.size)])
        # A NaN from a failed undistortion has no sign, so it compares False here
        # too and falls through to the same invalid marker.
        on = np.sign(g[2]) == self.ground_sign
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(on, g[0] / g[2], np.nan).reshape(iu.shape)
            y = np.where(on, g[1] / g[2], np.nan).reshape(iu.shape)
        return x, y

    def jacobian(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Metres of ground per pixel, as a ``(N, 2, 2)`` derivative.

        The exact local behaviour of the map, for propagating a covariance. The
        perspective-divide term is what makes it diverge toward the horizon --
        see docs/LIMITATIONS.md, "The Jacobian".

        Returns:
            NaN rows where the pixel does not see the plane.
        """
        u = np.asarray(u, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()
        iu, iv = self._ideal(u, v)
        g = self.inverse @ np.vstack([iu, iv, np.ones(iu.size)])
        w = g[2]
        on = np.sign(w) == self.ground_sign

        inv = self.inverse
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = g[0] / w, g[1] / w
            plane = np.empty((u.size, 2, 2))
            for k in range(2):
                for j in range(2):
                    plane[:, k, j] = (inv[k, j] - (g[k] / w) * inv[2, j]) / w
        metric = self.frame.metres_per_unit(x, y)
        out = np.einsum("nij,njk->nik", metric, plane)
        return np.where(on[:, None, None], out, np.nan)

    def ground_m_per_px(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Metres of ground a one-pixel step moves the point, along the worse axis.

        **Local, never a constant.** Deliberately not called a ground sample
        distance: nothing here is orthorectified, so the scale varies down the
        frame by a factor of tens, varies along a single row, and is anisotropic
        even at one pixel -- near the horizon a pixel can stretch eighteen times
        further one way than the other. The worse axis is a worst-case summary;
        ``jacobian`` returns the whole 2x2.

        A secant rather than a derivative, because a one-pixel annotation error is
        what it is meant to price.

        Returns:
            NaN off the plane, inf when a neighbouring pixel falls off it.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        x, y = self._to_plane(u, v)
        steps = []
        for du, dv in ((1.0, 0.0), (0.0, 1.0)):
            sx, sy = self._to_plane(u + du, v + dv)
            steps.append(self.frame.distance_m(x, y, sx, sy))
        missing = np.isnan(steps[0]) | np.isnan(steps[1])
        on = np.isfinite(x)
        return np.where(on, np.where(missing, np.inf, np.fmax(*steps)), np.nan)

    def project(self, u: np.ndarray, v: np.ndarray,
                max_m_per_px: float = math.inf) -> GroundPoints:
        """Place pixels on the plane. Pure; one output per input, always.

        Args:
            u: Pixel columns.
            v: Pixel rows.
            max_m_per_px: Points worse conditioned than this are invalid.

        Returns:
            Positions on the fitted plane. A target that does not sit on that
            plane needs ``CameraStation.project``.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        x, y = self._to_plane(u, v)
        on_plane = np.isfinite(x)

        outside_lens = 0
        if self.lens is not None and (~on_plane).any():
            outside_lens = int(np.isnan(self._ideal(u[~on_plane], v[~on_plane])[0]).sum())

        m_per_px = self.ground_m_per_px(u, v)
        valid = on_plane & (m_per_px <= max_m_per_px)
        return GroundPoints(
            x=np.where(valid, x, np.nan),
            y=np.where(valid, y, np.nan),
            ground_m_per_px=m_per_px,
            valid=valid,
            beyond_horizon=int((~on_plane).sum()) - outside_lens,
            outside_lens_model=outside_lens,
            ill_conditioned=int((on_plane & ~valid).sum()),
        )

    def translated(self, dx: float, dy: float) -> PlaneHomography:
        """The same view, re-expressed about a plane origin moved by ``(dx, dy)``.

        Exact and composable -- it is a change of coordinates, not a change of
        camera. Use it to put several cameras of one site on a shared origin.
        Physically moving or re-aiming a camera is not this: that changes the
        matrix and needs a new fit.
        """
        shift = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)],
                          [0.0, 0.0, 1.0]])
        return PlaneHomography(self.matrix @ shift, self.width, self.height,
                               lens=self.lens, frame=self.frame,
                               ground_pixel=self.ground_pixel)

    # -- ground to pixels -------------------------------------------------

    def unproject(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Plane coordinates back to pixels, distortion included."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        p = self.matrix @ np.vstack([x.ravel(), y.ravel(), np.ones(x.size)])
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = p[0] / p[2], p[1] / p[2]
        if self.lens is not None:
            cv2 = require_cv2()
            norm = np.column_stack([(u - self.lens.cx) / self.lens.fx,
                                    (v - self.lens.cy) / self.lens.fy,
                                    np.ones(u.size)]).reshape(-1, 1, 3)
            zero = np.zeros(3)
            out = np.asarray(cv2.projectPoints(norm, zero, zero, self.lens.matrix,
                                               self.lens.coefficients)[0]).reshape(-1, 2)
            u, v = out[:, 0], out[:, 1]
        return u.reshape(x.shape), v.reshape(y.shape)

    def footprint(self, max_m_per_px: float = math.inf, columns: int = 64) -> np.ndarray:
        """The ground polygon this camera can usefully see.

        Walks each of ``columns`` image columns for the highest row still valid,
        then closes the ring along the bottom of the frame.

        Returns:
            ``(M, 2)`` plane coordinates, first point repeated last. Empty when
            nothing in the frame is valid.
        """
        cols = np.linspace(0.0, self.width - 1.0, columns)
        rows = np.arange(self.height, dtype=float)
        uu, vv = np.meshgrid(cols, rows, indexing="ij")
        gsd = self.ground_m_per_px(uu, vv)
        ok = np.isfinite(self._to_plane(uu, vv)[0]) & (gsd <= max_m_per_px)

        top = []
        for i in range(columns):
            hit = np.flatnonzero(ok[i])
            if hit.size:
                top.append((cols[i], rows[hit[0]]))
        if not top:
            return np.zeros((0, 2))
        ring = top + [(cols[i], self.height - 1.0) for i in reversed(range(columns))
                      if ok[i].any()]
        pix = np.asarray(ring, dtype=float)
        x, y = self._to_plane(pix[:, 0], pix[:, 1])
        pts = np.column_stack([x, y])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if not len(pts):
            return np.zeros((0, 2))
        return np.vstack([pts, pts[:1]])


@dataclass(frozen=True)
class CameraPose:
    """Where a camera stands and, if known, where it points.

    Angles are degrees. The orientation fields are None for a station whose view
    was fitted from control points, because a hand-fitted homography cannot be
    decomposed back into a trustworthy pose -- see docs/LIMITATIONS.md.

    Attributes:
        height_m: Height above the plane.
        x, y: Position in plane coordinates.
        tilt_deg: Depression of the optical axis below horizontal.
        yaw_deg: Bearing of the optical axis, measured from +y toward +x.
        roll_deg: Rotation about the optical axis.
    """

    height_m: float
    x: float = 0.0
    y: float = 0.0
    tilt_deg: float | None = None
    yaw_deg: float | None = None
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        if not self.height_m > 0.0:
            raise ValueError("height_m must be positive, got " + repr(self.height_m))
        tilt = self.tilt_deg
        if tilt is not None and not 0.0 < tilt < 90.0:
            raise ValueError("tilt_deg must be in (0, 90) so the axis meets the "
                             "plane in front of the camera, got " + repr(tilt))

    @property
    def oriented(self) -> bool:
        """True when the pose is complete enough to rebuild the view from."""
        return self.tilt_deg is not None and self.yaw_deg is not None


def homography_from_pose(pose: CameraPose, lens: Lens, width: int, height: int
                         ) -> np.ndarray:
    """The plane-to-image matrix an ideal camera at ``pose`` would have.

    Plane coordinates are **metres**, x east and y north, on the plane the camera
    is ``height_m`` above -- the pose mixes its position with its height in one
    rotation, so the two must share a unit. Distortion is not baked in: it stays
    on the ``Lens``.

    Raises:
        ValueError: If the pose has no orientation.
    """
    if pose.tilt_deg is None or pose.yaw_deg is None:
        raise ValueError("a pose needs tilt_deg and yaw_deg to build a view from")
    t, yaw = math.radians(pose.tilt_deg), math.radians(pose.yaw_deg)
    forward = np.array([math.sin(yaw) * math.cos(t), math.cos(yaw) * math.cos(t),
                        -math.sin(t)])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    down = np.cross(forward, right)
    c, s = math.cos(math.radians(pose.roll_deg)), math.sin(math.radians(pose.roll_deg))
    rotation = np.vstack([c * right + s * down, -s * right + c * down, forward])
    centre = np.array([pose.x, pose.y, pose.height_m])
    return lens.matrix @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], -(rotation @ centre)])


@dataclass(frozen=True)
class CameraStation:
    """An installed camera: its view of the plane and its pose.

    Two ways in, and they differ in what can afterwards be changed.
    ``from_pose`` **derives** the view, so ``updated`` can re-aim the camera and
    the matrix follows. A station built around a **fitted** homography keeps that
    matrix as the truth; its position can still be recorded for corrections, but
    re-aiming it would need a new fit, and ``updated`` says so rather than
    silently returning a view that no longer matches.

    Attributes:
        view: The plane-to-image map, with its lens and frame.
        pose: Where the camera is, and where it points if that is known.
        name: Free-form label, for reports and multi-camera sites.
        lens: The intrinsics the view was derived from, kept so the view can be
            rebuilt when the pose changes. None for a fitted view.
    """

    view: PlaneHomography
    pose: CameraPose
    name: str = ""
    lens: Lens | None = None

    @classmethod
    def from_pose(cls, pose: CameraPose, lens: Lens, width: int, height: int,
                  frame: PlaneFrame = DEFAULT_FRAME, name: str = "") -> CameraStation:
        """Build a station by deriving its view from a known pose and intrinsics.

        Raises:
            ValueError: If ``frame`` is not metric. A pose is expressed in the
                same units as the camera height, so deriving a view in degrees
                would mix degrees with metres in the translation.
        """
        if not isinstance(frame, MetricPlane):
            raise ValueError(
                "from_pose needs a MetricPlane: a pose mixes position with "
                "height, so both must be metres. For a view in degrees, fit one "
                "with fit_homography(..., frame=GeodeticPlane())"
            )
        matrix = homography_from_pose(pose, lens, width, height)
        view = PlaneHomography(matrix, width, height,
                               lens=None if lens.is_identity else lens, frame=frame)
        return cls(view, pose, name, lens)

    @property
    def height_m(self) -> float:
        """Camera height above the plane."""
        return self.pose.height_m

    @property
    def x(self) -> float:
        """Camera position along the plane's first axis."""
        return self.pose.x

    @property
    def y(self) -> float:
        """Camera position along the plane's second axis."""
        return self.pose.y

    @property
    def derived(self) -> bool:
        """True when the view came from the pose, so the pose can be changed."""
        return self.pose.oriented and self.lens is not None

    def updated(self, **changes) -> CameraStation:
        """A copy with pose fields changed, e.g. ``updated(tilt_deg=12.0)``.

        On a derived station the view is rebuilt, so moving or re-aiming the
        camera is a one-liner. On a station wrapping a fitted homography only
        position may change, and it affects the corrections alone.

        Raises:
            ValueError: If an orientation change is asked of a fitted view, or a
                field is not part of ``CameraPose``.
        """
        unknown = sorted(set(changes) - set(CameraPose.__dataclass_fields__))
        if unknown:
            raise ValueError("CameraPose has no field(s) " + ", ".join(map(repr, unknown)))
        pose = replace(self.pose, **changes)
        if self.derived:
            assert self.lens is not None
            return CameraStation.from_pose(pose, self.lens, self.view.width,
                                           self.view.height, self.view.frame, self.name)
        if any(changes[f] != getattr(self.pose, f)
               for f in ("tilt_deg", "yaw_deg", "roll_deg") if f in changes):
            raise ValueError(
                "this view was fitted from control points, not derived from a "
                "pose, so re-aiming the camera cannot update it; fit a new "
                "homography instead. Position may still be changed."
            )
        return CameraStation(self.view, pose, self.name, self.lens)

    def range_m(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Ground distance from the camera to each pixel's point, in metres."""
        placed = self.view.project(u, v)
        return self.view.frame.distance_m(np.full(np.shape(placed.x), self.x),
                                          np.full(np.shape(placed.y), self.y),
                                          placed.x, placed.y)

    def project(
        self,
        u: np.ndarray,
        v: np.ndarray,
        max_m_per_px: float = math.inf,
        target_height_m: float = 0.0,
        curvature: bool = False,
        refraction_k: float = 0.0,
    ) -> GroundPoints:
        """Place pixels, correcting for targets that do not sit on the plane.

        Args:
            u: Pixel columns.
            v: Pixel rows.
            max_m_per_px: Points worse conditioned than this are invalid.
            target_height_m: Height of the target's base above the plane -- swell,
                cargo, crop. The dominant error term at range if left uncorrected.
            curvature: Correct for the surface falling below the tangent plane.
                Worth having over open water and essentially nowhere else.
            refraction_k: Coefficient of refraction, reducing that correction by
                ``1 - k``. Zero assumes none, which overstates curvature.
        """
        placed = self.view.project(u, v, max_m_per_px)
        if not (target_height_m or curvature):
            return placed

        x, y = placed.x, placed.y
        h = np.full(np.shape(x), float(target_height_m))
        if curvature:
            rng = self.view.frame.distance_m(np.full(np.shape(x), self.x),
                                             np.full(np.shape(y), self.y), x, y)
            h = h - (1.0 - refraction_k) * rng**2 / (2.0 * EARTH_RADIUS_M)
        with np.errstate(invalid="ignore"):
            # At or above the camera the ray never reaches that height.
            factor = np.where(1.0 - h / self.height_m > 0.0, 1.0 - h / self.height_m,
                              np.nan)
        moved_x = self.x + (x - self.x) * factor
        moved_y = self.y + (y - self.y) * factor
        return GroundPoints(
            x=moved_x, y=moved_y,
            ground_m_per_px=placed.ground_m_per_px,
            valid=placed.valid & np.isfinite(moved_x),
            beyond_horizon=placed.beyond_horizon,
            outside_lens_model=placed.outside_lens_model,
            ill_conditioned=placed.ill_conditioned,
        )

    def translated(self, dx: float, dy: float) -> CameraStation:
        """The whole station re-expressed about a plane origin moved by ``(dx, dy)``.

        A change of coordinates, not of camera: view and pose move together, so a
        site's cameras can be put on one shared origin without any of them
        disagreeing about where they are.
        """
        pose = replace(self.pose, x=self.pose.x - dx, y=self.pose.y - dy)
        return CameraStation(self.view.translated(dx, dy), pose, self.name, self.lens)
