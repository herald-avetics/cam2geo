"""fit_homography: recovering a matrix from ground control points."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import HEIGHT, WIDTH, build_camera

from cam2geo import Lens, MetricPlane, fit_homography

pytest.importorskip("cv2", reason="fitting is OpenCV's job")

#: Six well-spread points on the ground, in metres east and north.
CONTROL = np.array([[-50.0, 60.0], [50.0, 60.0], [-30.0, 200.0],
                    [30.0, 200.0], [0.0, 120.0], [80.0, 300.0]])


def _pixels_of(camera, points):
    """Where a known camera sees each control point."""
    u, v = camera.unproject(points[:, 0], points[:, 1])
    return np.column_stack([u, v])


class TestValidation:
    """Bad control points fail at the call, naming what is wrong."""

    def test_fewer_than_four_points_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            fit_homography(CONTROL[:3], CONTROL[:3] * 2.0, WIDTH, HEIGHT)

    def test_mismatched_counts_raise(self):
        with pytest.raises(ValueError, match="but"):
            fit_homography(CONTROL, CONTROL[:4], WIDTH, HEIGHT)

    def test_a_non_finite_point_raises(self):
        bad = CONTROL.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            fit_homography(bad, CONTROL * 2.0, WIDTH, HEIGHT)

    def test_an_unknown_method_raises_with_the_value(self):
        with pytest.raises(ValueError, match="'spline'"):
            fit_homography(CONTROL, CONTROL * 2.0, WIDTH, HEIGHT, method="spline")

    def test_collinear_points_are_rejected(self):
        """Four points on a line define no plane, so no homography exists."""
        line = np.array([[0.0, 10.0], [0.0, 20.0], [0.0, 30.0], [0.0, 40.0]])
        pixels = np.array([[100.0, 100.0], [100.0, 200.0], [100.0, 300.0],
                           [100.0, 400.0]])

        with pytest.raises(ValueError, match="degenerate|collinear"):
            fit_homography(line, pixels, WIDTH, HEIGHT, method="exact")


class TestRecovery:
    """A fit of a known camera's own projections must return that camera."""

    def test_it_recovers_the_matrix_it_was_generated_from(self):
        camera = build_camera()
        pixels = _pixels_of(camera, CONTROL)

        fit = fit_homography(CONTROL, pixels, WIDTH, HEIGHT, method="exact")

        assert fit.rms_px < 1e-3
        assert fit.inliers.all()

    def test_the_recovered_view_places_points_where_the_original_does(self):
        """The matrix is only defined up to scale, so compare what it does."""
        camera = build_camera()
        pixels = _pixels_of(camera, CONTROL)
        fit = fit_homography(CONTROL, pixels, WIDTH, HEIGHT, method="exact")

        probe_u = np.array([300.0, 960.0, 1600.0])
        probe_v = np.array([800.0, 700.0, 1000.0])
        original = camera.project(probe_u, probe_v)
        recovered = fit.homography.project(probe_u, probe_v)

        # A tenth of a millimetre. The fit is a least-squares solve, not an
        # algebraic inverse, so exact equality is not on offer -- but this is
        # nine orders of magnitude below anything the error budget cares about.
        np.testing.assert_allclose(recovered.x, original.x, atol=1e-4)
        np.testing.assert_allclose(recovered.y, original.y, atol=1e-4)

    def test_exactly_four_points_are_enough(self):
        camera = build_camera()
        fit = fit_homography(CONTROL[:4], _pixels_of(camera, CONTROL[:4]),
                             WIDTH, HEIGHT, method="exact")
        assert fit.rms_px < 1e-3

    def test_an_exact_fit_keeps_every_point(self):
        """Least squares has no inlier concept, and OpenCV's all-zero mask for it
        must not be read as "every point rejected" -- which once crashed the
        RMS on an empty array."""
        camera = build_camera()

        fit = fit_homography(CONTROL, _pixels_of(camera, CONTROL), WIDTH, HEIGHT,
                             method="exact")

        assert fit.inliers.all()
        assert np.isfinite(fit.rms_px)
        assert np.isfinite(fit.max_px)

    def test_residuals_are_reported_per_point(self):
        camera = build_camera()
        fit = fit_homography(CONTROL, _pixels_of(camera, CONTROL), WIDTH, HEIGHT,
                             method="exact")

        assert fit.residuals_px.shape == (len(CONTROL),)
        assert fit.max_px >= fit.rms_px

    def test_a_disturbed_point_shows_up_in_the_residuals(self):
        """The fit is only as good as the survey, and it should say so."""
        camera = build_camera()
        pixels = _pixels_of(camera, CONTROL)
        pixels[2] += 25.0

        fit = fit_homography(CONTROL, pixels, WIDTH, HEIGHT, method="exact")

        assert fit.rms_px > 1.0
        assert fit.residuals_px.argmax() == 2


class TestRansac:
    """With an outlier present, RANSAC should exclude it rather than absorb it."""

    def test_an_outlier_is_excluded(self):
        camera = build_camera()
        points = np.vstack([CONTROL, [[10.0, 150.0]]])
        pixels = _pixels_of(camera, points)
        pixels[-1] += 300.0

        fit = fit_homography(points, pixels, WIDTH, HEIGHT, method="ransac",
                             ransac_threshold_px=3.0)

        assert not fit.inliers[-1]
        assert fit.inliers[:-1].all()
        assert fit.rms_px < 1.0, "the kept points still fit tightly"


class TestSurfaceTilt:
    """A uniform tilt is absorbed by the fit, which is why it needs no correction."""

    def test_a_crossfall_costs_nothing(self):
        """Runway crossfall and pitch camber are a tilt of the plane. Fitting to
        points on the tilted surface produces a matrix for that surface, with no
        residual left over -- so correcting for it again would be double-counting.
        """
        camera = build_camera()
        crossfall = 0.015
        tilted = CONTROL.copy()
        # The control points now sit on a plane sloping 1.5% across the x axis.
        heights = tilted[:, 0] * crossfall
        pixels = np.column_stack(_project_with_height(camera, tilted, heights))

        fit = fit_homography(tilted, pixels, WIDTH, HEIGHT, method="exact")

        assert fit.rms_px < 0.5, "a plane is a plane, whatever its tilt"


def _project_with_height(camera, points, heights):
    """Where points at a small height above the plane appear, by ray geometry."""
    # A point at height z sits on the ray from the camera, so its ground
    # intersection is scaled outward about the nadir by 1 / (1 - z/H).
    station_height = 10.0
    factor = 1.0 / (1.0 - heights / station_height)
    return camera.unproject(points[:, 0] * factor, points[:, 1] * factor)


class TestALensInTheFit:
    def test_control_pixels_are_undistorted_before_fitting(self):
        """The matrix must live in the pixel space the rest of the library uses."""
        camera = build_camera()
        lens = Lens(1000.0, 1000.0, WIDTH / 2.0, HEIGHT / 2.0, k1=-0.05)
        ideal = _pixels_of(camera, CONTROL)
        # Re-distort the ideal pixels, so the fit has to undo it.
        import cv2
        normalised = np.column_stack([(ideal[:, 0] - lens.cx) / lens.fx,
                                      (ideal[:, 1] - lens.cy) / lens.fy,
                                      np.ones(len(ideal))]).reshape(-1, 1, 3)
        raw = cv2.projectPoints(normalised, np.zeros(3), np.zeros(3),
                                lens.matrix, lens.coefficients)[0].reshape(-1, 2)

        fit = fit_homography(CONTROL, raw, WIDTH, HEIGHT, lens=lens,
                             frame=MetricPlane(), method="exact")

        assert fit.rms_px < 1e-3
        assert fit.homography.lens is lens
