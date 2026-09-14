"""Check the bespoke maths against independent, established implementations.

The conditioning code is the one part of this library with no off-the-shelf
equivalent to call, so instead it answers to other people's code: SciPy's
numerical differentiation for the Jacobian, and OpenCV's own projective
transform for the forward map. None of these is a runtime dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import WIDTH, build_camera

#: Rows spanning the usable image, from the bottom edge toward the horizon.
PROBE_ROWS = np.array([1000.0, 800.0, 600.0, 400.0, 300.0, 200.0])


class TestTheJacobianAgainstScipy:
    """scipy.differentiate.jacobian is an adaptive, Richardson-extrapolated
    numerical derivative -- an implementation with nothing in common with ours."""

    @staticmethod
    def _numerical(camera):
        """SciPy's Jacobian of the pixel-to-plane map, at each probe row."""
        from scipy.differentiate import jacobian as scipy_jacobian

        inverse = camera.inverse

        def to_plane(uv):
            u, v = uv[0], uv[1]
            g = np.einsum("ij,j...->i...", inverse,
                          np.stack([u, v, np.ones_like(u)]))
            return np.stack([g[0] / g[2], g[1] / g[2]])

        out = []
        for row in PROBE_ROWS:
            result = scipy_jacobian(to_plane, np.array([WIDTH / 2.0, row]))
            out.append(np.asarray(result.df).reshape(2, 2))
        return np.array(out)

    def test_the_analytic_jacobian_matches_scipy(self):
        pytest.importorskip("scipy")
        camera = build_camera()
        cols = np.full(PROBE_ROWS.size, WIDTH / 2.0)

        ours = camera.jacobian(cols, PROBE_ROWS)
        theirs = self._numerical(camera)

        # Compared row by row with an atol scaled to that row's magnitude. Not
        # rtol: for an unrolled camera the off-diagonals are structural zeros,
        # and no relative tolerance can describe 0 against 1e-14.
        for i, row in enumerate(PROBE_ROWS):
            tolerance = 1e-9 * max(float(np.abs(theirs[i]).max()), 1.0)
            np.testing.assert_allclose(ours[i], theirs[i], atol=tolerance,
                                       err_msg="image row " + str(row))

    def test_the_singular_values_match_scipy_relatively(self):
        """The quantity we actually publish is the largest singular value."""
        pytest.importorskip("scipy")
        camera = build_camera()
        cols = np.full(PROBE_ROWS.size, WIDTH / 2.0)

        ours = np.linalg.svd(camera.jacobian(cols, PROBE_ROWS), compute_uv=False)
        theirs = np.linalg.svd(self._numerical(camera), compute_uv=False)

        np.testing.assert_allclose(ours, theirs, rtol=1e-7)

    def test_it_matches_for_a_rolled_camera_too(self):
        """With roll the off-diagonals are real numbers, so this is the case that
        would catch a transposed or dropped cross term."""
        pytest.importorskip("scipy")
        from scipy.differentiate import jacobian as scipy_jacobian

        camera = build_camera(roll_deg=25.0)
        inverse = camera.inverse

        def to_plane(uv):
            g = np.einsum("ij,j...->i...", inverse,
                          np.stack([uv[0], uv[1], np.ones_like(uv[0])]))
            return np.stack([g[0] / g[2], g[1] / g[2]])

        point = np.array([700.0, 900.0])
        theirs = np.asarray(scipy_jacobian(to_plane, point).df).reshape(2, 2)
        ours = camera.jacobian(np.array([point[0]]), np.array([point[1]]))[0]

        assert abs(theirs[0, 1]) > 1e-6, "this camera must have real cross terms"
        np.testing.assert_allclose(ours, theirs, rtol=1e-7)


class TestTheForwardMapAgainstOpenCV:
    """cv2.perspectiveTransform is the same projective map -- and is exactly why
    it cannot be used for the implementation."""

    def test_it_agrees_with_opencv_on_the_ground_side(self):
        cv2 = pytest.importorskip("cv2")
        camera = build_camera()
        cols = np.full(PROBE_ROWS.size, WIDTH / 2.0)

        ours = camera.project(cols, PROBE_ROWS)
        theirs = cv2.perspectiveTransform(
            np.stack([cols, PROBE_ROWS], axis=-1).reshape(-1, 1, 2).astype(np.float64),
            camera.inverse).reshape(-1, 2)

        np.testing.assert_allclose(ours.x, theirs[:, 0], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(ours.y, theirs[:, 1], rtol=1e-12, atol=1e-9)

    def test_opencv_returns_the_mirrored_point_beyond_the_horizon(self):
        """The reason the implementation keeps its own homogeneous divide.

        perspectiveTransform normalises by w internally, so the sign that says
        which side of the horizon a pixel is on is gone by the time it returns.
        What comes back for a sky pixel is finite, plausible, and behind the
        camera -- which is the failure this library exists to refuse.
        """
        cv2 = pytest.importorskip("cv2")
        camera = build_camera()
        sky = np.array([[[WIDTH / 2.0, 100.0]]], dtype=np.float64)

        theirs = cv2.perspectiveTransform(sky, camera.inverse).reshape(2)
        ours = camera.project(np.array([sky[0, 0, 0]]), np.array([sky[0, 0, 1]]))

        assert np.isfinite(theirs).all(), "OpenCV gives an answer, with no warning"
        assert theirs[1] < 0.0, "and it is behind the camera"
        assert np.isnan(ours.x[0]), "we refuse it"
        assert ours.beyond_horizon == 1

    def test_unproject_agrees_with_opencv_in_the_other_direction(self):
        cv2 = pytest.importorskip("cv2")
        camera = build_camera()
        east = np.array([-30.0, 0.0, 45.0])
        north = np.array([40.0, 80.0, 150.0])

        u, v = camera.unproject(east, north)
        theirs = cv2.perspectiveTransform(
            np.stack([east, north], axis=-1).reshape(-1, 1, 2).astype(np.float64),
            camera.matrix).reshape(-1, 2)

        np.testing.assert_allclose(u, theirs[:, 0], rtol=1e-12)
        np.testing.assert_allclose(v, theirs[:, 1], rtol=1e-12)


class TestTheFitAgainstOpenCV:
    def test_a_matrix_survives_a_fit_of_its_own_projections(self):
        """cv2.findHomography is the fit; this checks we wired it up correctly."""
        cv2 = pytest.importorskip("cv2")
        from cam2geo import fit_homography

        camera = build_camera()
        plane = np.array([[-50.0, 60.0], [50.0, 60.0], [-30.0, 200.0],
                          [30.0, 200.0], [0.0, 120.0]])
        u, v = camera.unproject(plane[:, 0], plane[:, 1])
        pixels = np.column_stack([u, v])

        ours = fit_homography(plane, pixels, camera.width, camera.height,
                              method="exact").homography.matrix
        theirs, _ = cv2.findHomography(plane, pixels, 0)

        # Both are defined up to scale, so normalise before comparing.
        np.testing.assert_allclose(ours / ours[2, 2], theirs / theirs[2, 2],
                                   rtol=1e-9)
