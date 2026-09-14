"""CameraStation: an installed camera, and what can be changed about it."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import HEIGHT, LENS, WIDTH, build_station

from cam2geo import CameraPose, CameraStation, PlaneHomography


class TestBuildingFromAPose:
    """A derived station: the pose is the truth and the view follows from it."""

    def test_the_view_matches_the_pose_it_was_built_from(self, station):
        """A camera 10 m up, 20 degrees down, sees its own nadir-ward ground."""
        placed = station.view.project(np.array([WIDTH / 2.0]), np.array([HEIGHT - 1.0]))

        assert placed.valid[0]
        assert placed.x[0] == pytest.approx(0.0, abs=1e-9), "looking due north"
        assert placed.y[0] > 0.0, "and the ground is in front"

    def test_the_station_exposes_its_position_directly(self, station):
        assert station.height_m == 10.0
        assert (station.x, station.y) == (0.0, 0.0)

    def test_a_pose_derived_station_knows_it_is_derived(self, station):
        assert station.derived

    def test_an_unoriented_pose_cannot_build_a_view(self):
        from cam2geo import homography_from_pose

        with pytest.raises(ValueError, match="tilt_deg and yaw_deg"):
            homography_from_pose(CameraPose(height_m=10.0), LENS, WIDTH, HEIGHT)

    def test_a_geodetic_frame_is_refused(self):
        """A pose mixes position with height, so both must be in metres.

        Deriving one in degrees would put degrees and metres in the same
        translation column and produce a silently wrong camera.
        """
        from cam2geo import GeodeticPlane

        pose = CameraPose(height_m=30.0, x=-4.14, y=50.35, tilt_deg=8.0, yaw_deg=0.0)

        with pytest.raises(ValueError, match="MetricPlane"):
            CameraStation.from_pose(pose, LENS, WIDTH, HEIGHT, GeodeticPlane())

    def test_yaw_turns_the_camera(self):
        """Facing east should put the ground east of the camera, not north."""
        north = build_station(yaw_deg=0.0)
        east = build_station(yaw_deg=90.0)
        col, row = np.array([WIDTH / 2.0]), np.array([HEIGHT - 1.0])

        looking_north = north.view.project(col, row)
        looking_east = east.view.project(col, row)

        assert looking_north.y[0] == pytest.approx(looking_east.x[0], abs=1e-9)
        assert looking_east.y[0] == pytest.approx(0.0, abs=1e-9)

    def test_a_taller_camera_sees_further(self, station):
        taller = build_station(height_m=40.0)
        col, row = np.array([WIDTH / 2.0]), np.array([HEIGHT - 1.0])

        assert (taller.view.project(col, row).y[0]
                > station.view.project(col, row).y[0])

    def test_a_steeper_tilt_brings_the_ground_closer(self, station):
        steep = build_station(tilt_deg=45.0)
        col, row = np.array([WIDTH / 2.0]), np.array([HEIGHT - 1.0])

        assert (steep.view.project(col, row).y[0]
                < station.view.project(col, row).y[0])


class TestUpdating:
    """Re-aiming a derived station rebuilds its view; a fitted one refuses."""

    def test_changing_the_tilt_rebuilds_the_view(self, station):
        steeper = station.updated(tilt_deg=30.0)

        assert steeper.pose.tilt_deg == 30.0
        assert not np.allclose(steeper.view.matrix, station.view.matrix)

    def test_the_rebuilt_view_equals_one_built_from_scratch(self, station):
        """Updating must be exactly equivalent to constructing afresh."""
        updated = station.updated(tilt_deg=30.0, height_m=25.0)
        fresh = build_station(tilt_deg=30.0, height_m=25.0)

        np.testing.assert_allclose(updated.view.matrix, fresh.view.matrix, rtol=1e-12)

    def test_moving_the_camera_moves_where_it_sees(self, station):
        moved = station.updated(x=100.0)
        col, row = np.array([WIDTH / 2.0]), np.array([HEIGHT - 1.0])

        before = station.view.project(col, row)
        after = moved.view.project(col, row)

        assert after.x[0] == pytest.approx(before.x[0] + 100.0, abs=1e-6)

    def test_the_original_is_untouched(self, station):
        """Frozen dataclasses: updating returns a new station, it does not mutate."""
        before = station.view.matrix.copy()

        station.updated(tilt_deg=35.0)

        np.testing.assert_array_equal(station.view.matrix, before)

    def test_an_unknown_field_raises_naming_it(self, station):
        with pytest.raises(ValueError, match="'zoom'"):
            station.updated(zoom=2.0)

    def test_updating_is_chainable(self, station):
        out = station.updated(tilt_deg=25.0).updated(yaw_deg=10.0).updated(x=5.0)

        assert (out.pose.tilt_deg, out.pose.yaw_deg, out.pose.x) == (25.0, 10.0, 5.0)


class TestAFittedStation:
    """A station wrapping a fitted homography: position yes, re-aiming no."""

    def _fitted(self, station):
        """The same view, but treated as fitted -- no lens, no orientation."""
        return CameraStation(station.view, CameraPose(height_m=10.0, x=0.0, y=0.0),
                             "fitted")

    def test_it_knows_it_is_not_derived(self, station):
        assert not self._fitted(station).derived

    def test_its_position_may_still_be_changed(self, station):
        """Position feeds the corrections, which do not need the matrix rebuilt."""
        fitted = self._fitted(station)

        moved = fitted.updated(x=12.0, height_m=15.0)

        assert (moved.x, moved.height_m) == (12.0, 15.0)
        np.testing.assert_array_equal(moved.view.matrix, fitted.view.matrix)

    def test_re_aiming_it_raises_and_says_why(self, station):
        fitted = self._fitted(station)

        with pytest.raises(ValueError, match="fitted or realigned"):
            fitted.updated(tilt_deg=30.0)

    def test_the_corrections_still_work(self, station):
        """Everything that needs only a position is available to a fitted view."""
        fitted = self._fitted(station)
        col, row = np.array([960.0]), np.array([700.0])

        raised = fitted.project(col, row, target_height_m=1.0)

        assert np.isfinite(raised.x[0])


class TestRange:
    def test_range_grows_toward_the_horizon(self, station):
        rows = np.array([1079.0, 900.0, 700.0, 500.0])
        cols = np.full(rows.size, WIDTH / 2.0)

        rng = station.range_m(cols, rows)

        assert np.all(np.diff(rng) > 0.0)

    def test_range_is_nan_beyond_the_horizon(self, station):
        assert np.isnan(station.range_m(np.array([960.0]), np.array([100.0]))[0])

    def test_range_is_measured_from_the_camera_not_the_origin(self):
        """An off-centre camera measures range from where it stands."""
        offset = build_station(x=500.0)
        col, row = np.array([WIDTH / 2.0]), np.array([HEIGHT - 1.0])

        placed = offset.view.project(col, row)

        assert placed.x[0] == pytest.approx(500.0, abs=1e-6)
        assert offset.range_m(col, row)[0] == pytest.approx(placed.y[0], abs=1e-6)


class TestTranslating:
    """Re-expressing a site about a shared plane origin, exactly."""

    def test_the_view_and_the_pose_move_together(self, station):
        shifted = station.translated(100.0, -50.0)
        col, row = np.array([WIDTH / 2.0]), np.array([HEIGHT - 1.0])

        before = station.view.project(col, row)
        after = shifted.view.project(col, row)

        assert after.x[0] == pytest.approx(before.x[0] - 100.0, abs=1e-6)
        assert after.y[0] == pytest.approx(before.y[0] + 50.0, abs=1e-6)
        assert shifted.x == pytest.approx(station.x - 100.0)
        assert shifted.y == pytest.approx(station.y + 50.0)

    def test_range_is_unchanged_by_a_change_of_origin(self, station):
        """The camera did not move, so nothing physical may change."""
        cols = np.array([400.0, 960.0, 1500.0])
        rows = np.array([1000.0, 800.0, 900.0])

        before = station.range_m(cols, rows)
        after = station.translated(250.0, 700.0).range_m(cols, rows)

        np.testing.assert_allclose(after, before, rtol=1e-9)

    def test_translating_back_returns_the_original(self, station):
        there_and_back = station.translated(30.0, 40.0).translated(-30.0, -40.0)

        np.testing.assert_allclose(there_and_back.view.matrix, station.view.matrix,
                                   rtol=1e-12)

    def test_two_cameras_can_be_put_on_one_origin(self):
        """The multi-camera use: both must agree about a point they both see."""
        left = build_station(x=-40.0)
        right = build_station(x=40.0)
        col, row = np.array([WIDTH / 2.0]), np.array([HEIGHT - 1.0])

        shared_left = left.translated(-40.0, 0.0)
        shared_right = right.translated(-40.0, 0.0)

        assert shared_left.x == pytest.approx(0.0)
        assert shared_right.x == pytest.approx(80.0)
        assert (shared_right.view.project(col, row).x[0]
                - shared_left.view.project(col, row).x[0]) == pytest.approx(80.0, abs=1e-6)

    def test_a_bare_view_can_be_translated_too(self, camera):
        shifted = camera.translated(10.0, 20.0)

        assert isinstance(shifted, PlaneHomography)
        assert shifted.frame == camera.frame
        assert shifted.width == camera.width


class TestPlaneHomographyTranslation:
    def test_it_is_a_change_of_coordinates_not_of_scale(self, camera):
        """Conditioning is a physical property and must survive a relabelling."""
        cols = np.array([400.0, 960.0, 1500.0])
        rows = np.array([1000.0, 800.0, 600.0])

        before = camera.ground_m_per_px(cols, rows)
        after = camera.translated(123.0, -456.0).ground_m_per_px(cols, rows)

        np.testing.assert_allclose(after, before, rtol=1e-9)
