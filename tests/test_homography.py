"""PlaneHomography: the projection contract, the horizon rule, and conditioning."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import HEIGHT, HORIZON_ROW, WIDTH, build_camera, build_station

from cam2geo import CameraPose, Lens, MetricPlane, PlaneHomography


class TestConstruction:
    """The matrix and frame size are validated up front, naming the bad value."""

    def test_a_non_square_matrix_raises(self):
        with pytest.raises(ValueError, match="finite 3x3"):
            PlaneHomography(np.eye(2), WIDTH, HEIGHT)

    def test_a_non_finite_matrix_raises(self):
        bad = np.eye(3)
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite 3x3"):
            PlaneHomography(bad, WIDTH, HEIGHT)

    def test_a_singular_matrix_raises(self):
        with pytest.raises(ValueError, match="singular"):
            PlaneHomography(np.ones((3, 3)), WIDTH, HEIGHT)

    def test_a_non_positive_frame_size_raises_with_the_value(self):
        with pytest.raises(ValueError, match="0x1080"):
            PlaneHomography(np.eye(3), 0, HEIGHT)

    def test_a_list_of_lists_is_accepted_and_stored_as_an_array(self, camera):
        rebuilt = PlaneHomography(camera.matrix.tolist(), WIDTH, HEIGHT)
        assert isinstance(rebuilt.matrix, np.ndarray)
        np.testing.assert_allclose(rebuilt.matrix, camera.matrix)


class TestTheGroundSideOfTheHorizon:
    """A homography is defined up to scale and sign, so a known pixel fixes ground."""

    def test_the_default_anchor_is_the_bottom_centre_of_the_frame(self, camera):
        assert camera.anchor_pixel == (WIDTH / 2.0, HEIGHT - 1.0)

    @pytest.mark.parametrize("scale", [-1.0, 2.0, -0.001, 1000.0],
                             ids=["negated", "doubled", "tiny", "large"])
    def test_any_scaling_of_the_matrix_gives_the_same_answer(self, camera, scale):
        # Off-centre columns, so east is a real number rather than a numerical
        # zero that no relative tolerance can describe.
        cols = np.array([400.0, 960.0, 1500.0])
        rows = np.array([1079.0, 700.0, 300.0])
        scaled = PlaneHomography(camera.matrix * scale, WIDTH, HEIGHT,
                                 frame=MetricPlane())

        out = scaled.project(cols, rows)
        ref = camera.project(cols, rows)

        # atol because the centre column's east coordinate is an exact zero,
        # which no relative tolerance can describe. A nanometre is not a result.
        np.testing.assert_allclose(out.x, ref.x, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(out.y, ref.y, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(out.ground_m_per_px, ref.ground_m_per_px, rtol=1e-12)
        np.testing.assert_array_equal(out.valid, ref.valid)

    def test_a_ground_pixel_on_the_horizon_raises(self, camera):
        """Within a pixel of the line, which side you are on is not decidable."""
        line = camera.horizon_line
        col = WIDTH / 2.0
        on_the_line = -(line[0] * col + line[2]) / line[1]
        ambiguous = PlaneHomography(camera.matrix, WIDTH, HEIGHT,
                                    ground_pixel=(col, on_the_line))

        with pytest.raises(ValueError, match="within a pixel of the horizon"):
            _ = ambiguous.ground_sign

    def test_a_ground_pixel_above_the_horizon_silently_inverts_everything(self, camera):
        """A limitation, pinned deliberately rather than papered over.

        Nothing in the matrix says which side is ground, so an anchor chosen in
        the sky is self-consistent and wrong: the sky becomes valid and the road
        becomes NaN. Only the caller knows, which is why the anchor is an input.
        """
        inverted = PlaneHomography(camera.matrix, WIDTH, HEIGHT,
                                   frame=MetricPlane(), ground_pixel=(WIDTH / 2.0, 0.0))

        assert inverted.ground_sign == -camera.ground_sign
        assert inverted.project(np.array([960.0]), np.array([100.0])).valid[0]
        assert not inverted.project(np.array([960.0]), np.array([1079.0])).valid[0]

    def test_choosing_a_different_ground_pixel_still_works(self, camera):
        """Any pixel below the horizon should fix the same side."""
        other = PlaneHomography(camera.matrix, WIDTH, HEIGHT, frame=MetricPlane(),
                                ground_pixel=(10.0, HEIGHT - 5.0))
        assert other.ground_sign == camera.ground_sign


class TestBeyondTheHorizon:
    """The far side of the horizon is NaN, never the finite point behind the camera."""

    def test_a_sky_pixel_is_nan_even_with_no_conditioning_cutoff(self, camera):
        placed = camera.project(np.array([WIDTH / 2.0]), np.array([100.0]),
                                max_m_per_px=math.inf)

        assert np.isnan(placed.x).all()
        assert np.isnan(placed.y).all()
        assert not placed.valid.any()
        assert (placed.beyond_horizon, placed.ill_conditioned) == (1, 0)

    def test_the_mirrored_point_is_what_we_are_refusing_to_return(self, camera):
        """Inverting a sky pixel gives a finite, plausible, wrong point.

        This test exists to show the trap is real: the naive answer is a number
        you would not notice was wrong, on the wrong side of the camera.
        """
        sky = np.array([WIDTH / 2.0, 100.0, 1.0])
        g = camera.inverse @ sky
        mirrored = g[:2] / g[2]

        assert np.isfinite(mirrored).all(), "the trap only exists because it is finite"
        assert mirrored[1] < 0.0, "and it lands behind the camera"
        assert np.isnan(camera.project(np.array([sky[0]]), np.array([sky[1]])).x).all()

    def test_one_output_per_input_even_when_none_is_valid(self, camera):
        cols = np.full(3, WIDTH / 2.0)
        rows = np.zeros(3)

        placed = camera.project(cols, rows)

        assert placed.x.shape == (3,)
        assert placed.y.shape == (3,)
        assert placed.ground_m_per_px.shape == (3,)
        assert not placed.valid.any()

    def test_an_empty_input_gives_empty_output(self, camera):
        placed = camera.project(np.zeros(0), np.zeros(0))

        assert placed.x.shape == (0,)
        assert (placed.beyond_horizon, placed.ill_conditioned) == (0, 0)

    def test_the_input_shape_is_preserved(self, camera):
        """A 2-D grid of pixels comes back as a 2-D grid of positions."""
        cols, rows = np.meshgrid(np.linspace(0, WIDTH - 1, 5),
                                 np.linspace(600, 1000, 4))

        placed = camera.project(cols, rows)

        assert placed.x.shape == cols.shape


class TestTheHorizonLine:
    """The vanishing line is available directly, and agrees with the projection."""

    def test_the_line_is_normalised_to_a_signed_pixel_distance(self, camera):
        a, b, _ = camera.horizon_line
        assert math.hypot(a, b) == pytest.approx(1.0)

    def test_it_sits_where_the_geometry_says_it_should(self, camera):
        """For an unrolled pinhole the horizon is a level row we can predict."""
        line = camera.horizon_line
        row_at_centre = -(line[0] * (WIDTH / 2.0) + line[2]) / line[1]

        assert row_at_centre == pytest.approx(HORIZON_ROW, abs=1e-6)

    def test_it_marks_exactly_where_projection_starts_returning_nan(self, camera):
        """The line and the NaN boundary must be the same boundary."""
        rows = np.arange(HEIGHT, dtype=float)
        cols = np.full(rows.size, WIDTH / 2.0)

        placed = camera.project(cols, rows, max_m_per_px=math.inf)
        line = camera.horizon_line
        on_ground_side = (line[0] * cols + line[1] * rows + line[2]) > 0.0

        np.testing.assert_array_equal(np.isfinite(placed.x), on_ground_side)

    def test_a_rolled_camera_has_a_tilted_horizon(self):
        rolled = build_camera(roll_deg=30.0)
        a, b, _ = rolled.horizon_line

        assert abs(a) > 0.4, "a 30 degree roll must tilt the line substantially"
        assert abs(b) > 0.0


class TestConditioning:
    """ground_m_per_px is stored on every row, including rows that were cut."""

    def test_it_worsens_monotonically_toward_the_horizon(self, camera, centre_column):
        cols = np.full(centre_column.size, WIDTH / 2.0)

        gsd = camera.ground_m_per_px(cols, centre_column)

        assert np.all(np.diff(gsd) > 0.0)
        assert 0.005 < gsd[0] < 0.05, "a pixel at the bottom edge is centimetres"
        assert gsd[-1] > 100.0, "and metres near the horizon"

    def test_a_point_over_the_cutoff_is_invalid_but_keeps_its_conditioning(self, camera):
        """A cut point still reports how bad it was -- that is the useful part."""
        row = np.array([HORIZON_ROW + 4.0])
        col = np.array([WIDTH / 2.0])

        placed = camera.project(col, row, max_m_per_px=1.0)

        assert not placed.valid[0]
        assert np.isnan(placed.x[0])
        assert np.isfinite(placed.ground_m_per_px[0])
        assert placed.ground_m_per_px[0] > 1.0
        assert (placed.beyond_horizon, placed.ill_conditioned) == (0, 1)

    def test_raising_the_cutoff_admits_the_same_point(self, camera):
        row = np.array([HORIZON_ROW + 4.0])
        col = np.array([WIDTH / 2.0])

        assert camera.project(col, row, max_m_per_px=math.inf).valid[0]

    def test_a_pixel_whose_neighbour_is_sky_has_unbounded_conditioning(self):
        """Within a pixel of the horizon the scale is unbounded, not unknown.

        A rolled horizon is not level, so stepping one column crosses it. The
        roll sign is chosen so the step goes toward the sky, because the
        conditioning secant only ever looks at u + 1 and v + 1.
        """
        rolled = build_camera(roll_deg=-30.0)
        line = rolled.horizon_line
        assert line[0] < 0.0, "stepping one column must approach the horizon"
        col = WIDTH / 2.0
        row_on_line = -(line[0] * col + line[2]) / line[1]
        # A quarter pixel onto the ground side; one column takes half a pixel
        # back off it.
        row = row_on_line + 0.25 * np.sign(line[1])

        placed = rolled.project(np.array([col]), np.array([row]), max_m_per_px=math.inf)

        assert not np.isnan(placed.ground_m_per_px[0])
        assert placed.ground_m_per_px[0] == math.inf

    def test_it_is_nan_where_the_point_is_nan(self, camera):
        placed = camera.project(np.array([WIDTH / 2.0]), np.array([50.0]))
        assert np.isnan(placed.ground_m_per_px[0])

    def test_the_scale_varies_along_a_single_row_too(self, camera):
        """Not only with range: one image row is a curve on the ground."""
        cols = np.array([50.0, WIDTH / 2.0, WIDTH - 50.0])
        rows = np.full(cols.size, 700.0)

        gsd = camera.ground_m_per_px(cols, rows)

        assert gsd[0] > gsd[1] and gsd[2] > gsd[1], "the centre is the best case"

    def test_a_single_pixel_is_anisotropic(self, camera):
        """Even at one pixel the scale is two numbers, not one.

        A pixel maps to a parallelogram, not a square, and the further out the
        more elongated it gets. That is what the reported worse axis summarises.
        """
        near = np.linalg.svd(camera.jacobian(np.array([WIDTH / 2.0]),
                                             np.array([1079.0])), compute_uv=False)[0]
        far = np.linalg.svd(camera.jacobian(np.array([WIDTH / 2.0]),
                                            np.array([400.0])), compute_uv=False)[0]

        assert near[0] / near[1] > 1.0, "not square even at the bottom edge"
        assert far[0] / far[1] > near[0] / near[1], "and it worsens with range"

    def test_it_reports_the_worse_axis_not_the_average(self, camera):
        """A worst case, deliberately: an error budget wants the bad direction."""
        cols, rows = np.array([WIDTH / 2.0]), np.array([700.0])

        singular = np.linalg.svd(camera.jacobian(cols, rows), compute_uv=False)[0]
        reported = camera.ground_m_per_px(cols, rows)[0]

        assert reported == pytest.approx(singular[0], rel=0.01)
        assert reported > singular[1], "neither the better axis nor the mean"


class TestTheJacobian:
    """The analytic derivative, for propagating a covariance."""

    def test_it_tracks_the_secant_away_from_the_horizon(self, camera):
        """Two different questions, but they must agree where the map is smooth."""
        rows = np.array([1000.0, 800.0, 600.0, 400.0])
        cols = np.full(rows.size, WIDTH / 2.0)

        largest = np.linalg.svd(camera.jacobian(cols, rows), compute_uv=False)[:, 0]
        secant = camera.ground_m_per_px(cols, rows)

        np.testing.assert_allclose(largest, secant, rtol=0.01)

    def test_it_diverges_faster_than_the_secant_at_the_horizon(self, camera):
        """The secant averages over a pixel; the derivative does not. Near the
        horizon that difference is real and the docs must not claim otherwise."""
        row, col = np.array([180.0]), np.array([WIDTH / 2.0])

        largest = np.linalg.svd(camera.jacobian(col, row), compute_uv=False)[0, 0]
        secant = camera.ground_m_per_px(col, row)[0]

        assert largest > secant * 1.2

    def test_it_is_nan_beyond_the_horizon(self, camera):
        j = camera.jacobian(np.array([WIDTH / 2.0]), np.array([100.0]))
        assert np.isnan(j).all()

    def test_it_is_constant_for_an_affine_matrix(self):
        """With no perspective row there is no divergence: scale is uniform."""
        affine = np.array([[2.0, 0.0, 100.0], [0.0, 3.0, 50.0], [0.0, 0.0, 1.0]])
        flat = PlaneHomography(affine, WIDTH, HEIGHT, frame=MetricPlane(),
                               ground_pixel=(WIDTH / 2.0, HEIGHT - 1.0))
        cols = np.array([10.0, 900.0, 1900.0])
        rows = np.array([10.0, 500.0, 1000.0])

        j = flat.jacobian(cols, rows)

        np.testing.assert_allclose(j, np.broadcast_to(j[0], j.shape), rtol=1e-12)
        np.testing.assert_allclose(j[0], np.diag([0.5, 1.0 / 3.0]), atol=1e-15)

    def test_a_rolled_camera_has_non_zero_off_diagonals(self):
        """Without roll the off-diagonals are structurally zero, which hides bugs."""
        rolled = build_camera(roll_deg=30.0)

        j = rolled.jacobian(np.array([WIDTH / 2.0]), np.array([900.0]))

        assert abs(j[0, 0, 1]) > 1e-6
        assert abs(j[0, 1, 0]) > 1e-6


class TestRoundTrips:
    """Ground to pixel and back, which is the cheapest check that any of this works."""

    def test_a_ground_point_survives_a_round_trip(self, camera):
        east = np.array([-20.0, 0.0, 35.0])
        north = np.array([30.0, 60.0, 120.0])

        u, v = camera.unproject(east, north)
        placed = camera.project(u, v)

        assert placed.valid.all()
        np.testing.assert_allclose(placed.x, east, atol=1e-9)
        np.testing.assert_allclose(placed.y, north, atol=1e-9)

    def test_a_pixel_survives_a_round_trip(self, camera):
        cols = np.array([100.0, 960.0, 1800.0])
        rows = np.array([700.0, 900.0, 1050.0])

        placed = camera.project(cols, rows)
        u, v = camera.unproject(placed.x, placed.y)

        np.testing.assert_allclose(u, cols, atol=1e-9)
        np.testing.assert_allclose(v, rows, atol=1e-9)


class TestHeightCorrections:
    """A target off the plane, which is the biggest error term at range.

    These live on CameraStation, not PlaneHomography: walking a ray back to a
    different height needs to know where the camera stands, and the bare map
    does not.
    """

    def test_the_bare_view_offers_no_height_correction(self, camera):
        """The map alone cannot do this, and says so by not having the argument."""
        with pytest.raises(TypeError):
            camera.project(np.array([960.0]), np.array([900.0]), target_height_m=1.0)

    def test_zero_height_changes_nothing(self, station):
        cols, rows = np.array([960.0]), np.array([700.0])

        plain = station.project(cols, rows)
        zeroed = station.project(cols, rows, target_height_m=0.0)

        np.testing.assert_array_equal(plain.x, zeroed.x)
        np.testing.assert_array_equal(plain.y, zeroed.y)

    def test_a_raised_target_moves_toward_the_camera(self, station):
        """Relief displacement: the radius from the nadir becomes r * (1 - h/H).

        Exact for any orientation, because it depends only on the ray through the
        camera centre. See docs/LIMITATIONS.md, "A target above the plane".
        """
        cols, rows = np.array([960.0]), np.array([700.0])
        height_m = 2.0

        ground = station.project(cols, rows)
        raised = station.project(cols, rows, target_height_m=height_m)

        factor = 1.0 - height_m / station.height_m
        np.testing.assert_allclose(raised.x, ground.x * factor, rtol=1e-12)
        np.testing.assert_allclose(raised.y, ground.y * factor, rtol=1e-12)

    def test_it_holds_for_an_off_centre_camera_too(self):
        """The correction is radial about the nadir, not about the plane origin."""
        offset = build_station(x=17.0, y=-9.0)
        cols, rows = np.array([700.0]), np.array([800.0])
        height_m = 1.5

        ground = offset.project(cols, rows)
        raised = offset.project(cols, rows, target_height_m=height_m)

        factor = 1.0 - height_m / offset.height_m
        np.testing.assert_allclose(raised.x - offset.x,
                                   (ground.x - offset.x) * factor, rtol=1e-12)
        np.testing.assert_allclose(raised.y - offset.y,
                                   (ground.y - offset.y) * factor, rtol=1e-12)

    def test_the_displacement_grows_with_range(self, station):
        rows = np.array([1000.0, 800.0, 600.0])
        cols = np.full(rows.size, 960.0)

        ground = station.project(cols, rows)
        raised = station.project(cols, rows, target_height_m=1.0)
        moved = np.hypot(raised.x - ground.x, raised.y - ground.y)

        assert np.all(np.diff(moved) > 0.0)

    def test_a_target_at_camera_height_has_no_intersection(self, station):
        placed = station.project(np.array([960.0]), np.array([700.0]),
                                 target_height_m=station.height_m)
        assert np.isnan(placed.x).all()
        assert not placed.valid.any()

    def test_conditioning_is_untouched_by_a_height_correction(self, station):
        """The correction moves the point; it does not change the pixel scale."""
        cols, rows = np.array([960.0]), np.array([700.0])

        plain = station.project(cols, rows)
        raised = station.project(cols, rows, target_height_m=1.0)

        np.testing.assert_array_equal(plain.ground_m_per_px, raised.ground_m_per_px)

    def test_curvature_pushes_the_point_away_from_the_camera(self, station):
        """The surface falls below the tangent plane, so the true point is further."""
        cols, rows = np.array([960.0]), np.array([400.0])

        flat = station.project(cols, rows)
        curved = station.project(cols, rows, curvature=True)

        assert abs(curved.y[0]) > abs(flat.y[0])

    def test_refraction_reduces_the_curvature_correction(self, station):
        """0.14 is the value the surveying texts' constants imply. Over water the
        measured coefficient ranges far wider, which is its own limitation."""
        cols, rows = np.array([960.0]), np.array([400.0])

        none = station.project(cols, rows, curvature=True, refraction_k=0.0)
        some = station.project(cols, rows, curvature=True, refraction_k=0.14)
        flat = station.project(cols, rows)

        assert abs(flat.y[0]) < abs(some.y[0]) < abs(none.y[0])

    def test_curvature_is_negligible_close_in(self, station):
        """It is a term for open water at range, not for a yard."""
        cols, rows = np.array([960.0]), np.array([1079.0])

        flat = station.project(cols, rows)
        curved = station.project(cols, rows, curvature=True)

        assert abs(curved.y[0] - flat.y[0]) < 1e-3


class TestTheFootprint:
    """The ground polygon a camera can usefully see."""

    def test_it_is_a_closed_ring(self, camera):
        ring = camera.footprint(max_m_per_px=1.0)

        assert len(ring) > 3
        np.testing.assert_allclose(ring[0], ring[-1])

    def test_every_vertex_is_finite(self, camera):
        assert np.isfinite(camera.footprint(max_m_per_px=1.0)).all()

    def test_a_tighter_cutoff_gives_a_smaller_reach(self, camera):
        far = camera.footprint(max_m_per_px=10.0)
        near = camera.footprint(max_m_per_px=0.1)

        assert near[:, 1].max() < far[:, 1].max()

    def test_an_impossible_cutoff_gives_nothing(self, camera):
        assert len(camera.footprint(max_m_per_px=1e-9)) == 0


class TestTheFingerprint:
    """A stable hash of everything that decides where a pixel lands."""

    def test_the_same_calibration_hashes_the_same(self, camera):
        copy = PlaneHomography(camera.matrix.copy(), WIDTH, HEIGHT,
                               frame=MetricPlane())
        assert copy.fingerprint() == camera.fingerprint()

    def test_a_different_matrix_hashes_differently(self, camera):
        assert build_camera(tilt_deg=21.0).fingerprint() != camera.fingerprint()

    def test_a_different_ground_pixel_hashes_differently(self, camera):
        moved = PlaneHomography(camera.matrix, WIDTH, HEIGHT, frame=MetricPlane(),
                                ground_pixel=(10.0, 1070.0))
        assert moved.fingerprint() != camera.fingerprint()

    def test_it_covers_the_lens_too(self, camera):
        """Two views differing only in distortion must not share a hash."""
        distorted = PlaneHomography(camera.matrix, WIDTH, HEIGHT, frame=MetricPlane(),
                                    lens=Lens(1000.0, 1000.0, 960.0, 540.0, k1=-0.3))
        assert distorted.fingerprint() != camera.fingerprint()


class TestCameraPose:
    """The pose is validated so an impossible installation fails at construction."""

    def test_a_non_positive_height_raises_with_the_value(self):
        with pytest.raises(ValueError, match="-3.0"):
            CameraPose(height_m=-3.0)

    def test_a_horizontal_camera_raises(self):
        """At zero tilt the optical axis never meets the plane in front."""
        with pytest.raises(ValueError, match="tilt_deg"):
            CameraPose(height_m=10.0, tilt_deg=0.0, yaw_deg=0.0)

    def test_a_pose_with_no_angles_is_allowed_but_not_oriented(self):
        """Position alone is enough for corrections, not for rebuilding a view."""
        pose = CameraPose(height_m=10.0, x=1.0, y=2.0)
        assert not pose.oriented
