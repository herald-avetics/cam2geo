"""Lens: undistortion, and the NaN contract when the model has no inverse."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import HEIGHT, WIDTH, build_camera

from cam2geo import Lens, MetricPlane, PlaneHomography

cv2 = pytest.importorskip("cv2", reason="undistortion is OpenCV's job")

#: Strong barrel distortion. At fx = width the model has no inverse beyond a
#: normalised radius of about 0.50, and the frame corner sits at about 0.57.
STRONG = Lens(WIDTH, WIDTH, WIDTH / 2.0, HEIGHT / 2.0, k1=-0.6)
MILD = Lens(1000.0, 1000.0, WIDTH / 2.0, HEIGHT / 2.0, k1=-0.05)


class TestConstruction:
    def test_a_non_positive_focal_length_raises_with_the_value(self):
        with pytest.raises(ValueError, match="fx"):
            Lens(0.0, 1000.0, 960.0, 540.0)
        with pytest.raises(ValueError, match="fy"):
            Lens(1000.0, -2.0, 960.0, 540.0)

    def test_the_matrix_is_the_usual_pinhole_layout(self):
        k = MILD.matrix
        assert k[0, 0] == MILD.fx
        assert k[1, 1] == MILD.fy
        assert k[0, 2] == MILD.cx
        assert k[1, 2] == MILD.cy
        assert k[0, 1] == 0.0, "no skew"
        assert k[2, 2] == 1.0

    def test_coefficients_come_out_in_opencv_order(self):
        lens = Lens(1.0, 1.0, 0.0, 0.0, k1=1.0, k2=2.0, p1=3.0, p2=4.0, k3=5.0)
        np.testing.assert_array_equal(lens.coefficients, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_a_lens_with_no_coefficients_is_the_identity(self):
        assert Lens(1000.0, 1000.0, 960.0, 540.0).is_identity
        assert not MILD.is_identity


class TestUndistortion:
    """One output per input, and NaN rather than a diverged answer."""

    def test_an_identity_lens_returns_its_input_untouched(self):
        lens = Lens(1000.0, 1000.0, 960.0, 540.0)
        u = np.array([0.0, 500.0, 1919.0])
        v = np.array([0.0, 500.0, 1079.0])

        out_u, out_v = lens.undistort(u, v)

        np.testing.assert_array_equal(out_u, u)
        np.testing.assert_array_equal(out_v, v)

    def test_the_principal_point_does_not_move(self):
        """Distortion is radial about the principal point, so its centre is fixed."""
        out_u, out_v = MILD.undistort(np.array([MILD.cx]), np.array([MILD.cy]))

        assert out_u[0] == pytest.approx(MILD.cx, abs=1e-9)
        assert out_v[0] == pytest.approx(MILD.cy, abs=1e-9)

    def test_a_pixel_off_axis_actually_moves(self):
        out_u, out_v = MILD.undistort(np.array([1800.0]), np.array([1000.0]))

        assert abs(out_u[0] - 1800.0) > 1.0

    def test_it_inverts_opencv_own_forward_distortion(self):
        """The check that matters: undistort then re-distort must be the identity.

        cv2.projectPoints is the forward model, so this measures our inverse
        against OpenCV's own forward implementation rather than against itself.
        """
        u = np.array([200.0, 700.0, 960.0, 1400.0, 1700.0])
        v = np.array([200.0, 400.0, 540.0, 800.0, 1000.0])

        ideal_u, ideal_v = MILD.undistort(u, v)
        normalised = np.column_stack([(ideal_u - MILD.cx) / MILD.fx,
                                      (ideal_v - MILD.cy) / MILD.fy,
                                      np.ones(u.size)]).reshape(-1, 1, 3)
        back = cv2.projectPoints(normalised, np.zeros(3), np.zeros(3),
                                 MILD.matrix, MILD.coefficients)[0].reshape(-1, 2)

        np.testing.assert_allclose(back[:, 0], u, atol=1e-6)
        np.testing.assert_allclose(back[:, 1], v, atol=1e-6)

    def test_a_pixel_the_model_cannot_invert_is_nan(self):
        """Beyond where the distortion model folds, there is no answer to give."""
        out_u, out_v = STRONG.undistort(np.array([0.0]), np.array([HEIGHT - 1.0]))

        assert np.isnan(out_u[0])
        assert np.isnan(out_v[0])

    def test_the_centre_of_the_same_lens_is_still_fine(self):
        """The failure is local to the corners, not a broken lens."""
        out_u, _ = STRONG.undistort(np.array([WIDTH / 2.0]), np.array([HEIGHT / 2.0]))
        assert np.isfinite(out_u[0])

    def test_one_output_per_input_when_some_fail(self):
        u = np.array([0.0, WIDTH / 2.0, 0.0])
        v = np.array([HEIGHT - 1.0, HEIGHT / 2.0, 0.0])

        out_u, out_v = STRONG.undistort(u, v)

        assert out_u.shape == (3,)
        assert np.isfinite(out_u).sum() == 1

    def test_an_empty_input_is_allowed(self):
        out_u, out_v = MILD.undistort(np.zeros(0), np.zeros(0))
        assert out_u.shape == (0,)

    def test_the_input_shape_is_preserved(self):
        u, v = np.meshgrid(np.linspace(400, 1500, 4), np.linspace(300, 800, 3))

        out_u, _ = MILD.undistort(u, v)

        assert out_u.shape == u.shape


class TestALensOnAHomography:
    """A view with a lens undistorts before it inverts, and counts the failures."""

    def test_the_pixel_is_undistorted_before_the_matrix_is_applied(self):
        """Projecting a raw pixel with a lens equals projecting its ideal pixel
        without one. That is the whole contract, in one assertion."""
        camera = build_camera()
        with_lens = PlaneHomography(camera.matrix, WIDTH, HEIGHT, lens=MILD,
                                    frame=MetricPlane())
        raw_u, raw_v = np.array([1500.0]), np.array([900.0])
        ideal_u, ideal_v = MILD.undistort(raw_u, raw_v)

        through_lens = with_lens.project(raw_u, raw_v)
        directly = camera.project(ideal_u, ideal_v)

        assert abs(ideal_u[0] - raw_u[0]) > 1.0, "the lens must actually do something"
        np.testing.assert_allclose(through_lens.x, directly.x, atol=1e-9)
        np.testing.assert_allclose(through_lens.y, directly.y, atol=1e-9)

    def test_a_pixel_with_no_inverse_is_counted_separately_from_the_horizon(self):
        """Two different failures, reported as two different numbers."""
        camera = build_camera()
        with_lens = PlaneHomography(camera.matrix, WIDTH, HEIGHT, lens=STRONG,
                                    frame=MetricPlane(),
                                    ground_pixel=(WIDTH / 2.0, HEIGHT / 2.0 + 200.0))

        placed = with_lens.project(np.array([0.0]), np.array([HEIGHT - 1.0]))

        assert np.isnan(placed.x[0])
        assert (placed.outside_lens_model, placed.beyond_horizon) == (1, 0)
