"""Boresighting: measuring a fixed camera's drift and undoing it.

Every test here builds two cameras from the same conftest pinhole -- a reference
and a nudged one -- so the truth is known exactly and the question is only
whether alignment recovers it.

The tolerances are looser than the rest of the suite and the reason is not the
geometry: ``cv2.findHomography`` solves a DLT and lands about 2e-5 px from the
exact answer even on noiseless input, which is 3e-6 degrees of angle and about a
micrometre on the ground. Everything below is pinned just outside that floor, so
a real regression still fails while OpenCV's own arithmetic does not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import CAMERA_HEIGHT_M, FOCAL_PX, HEIGHT, LENS, TILT_DEG, WIDTH

from cam2geo import CameraPose, CameraStation, Lens, MetricPlane, align_pixels

pytest.importorskip("cv2", reason="align_pixels fits with cv2.findHomography")

#: Ground features that have not moved, well spread below the horizon.
LANDMARKS_M = np.array([[-30.0, 22.0], [28.0, 25.0], [-14.0, 48.0], [16.0, 52.0],
                        [0.0, 33.0], [40.0, 90.0], [-38.0, 85.0]])


def landmark_pixels(station):
    """Where a station sees the fixed landmarks."""
    u, v = station.view.unproject(LANDMARKS_M[:, 0], LANDMARKS_M[:, 1])
    return np.column_stack([u, v])


class TestMeasuringDrift:
    """What align_pixels reports about a camera that has been knocked."""

    def test_a_camera_that_has_not_moved_reports_no_drift(self, station):
        pixels = landmark_pixels(station)

        alignment = align_pixels(pixels, pixels, lens=LENS)

        assert alignment.motion_px == pytest.approx(0.0, abs=1e-9)
        assert alignment.drift_deg == pytest.approx(0.0, abs=1e-4)
        np.testing.assert_allclose(alignment.transform / alignment.transform[2, 2],
                                   np.eye(3), atol=1e-6)

    @pytest.mark.parametrize("change, expected_deg", [
        ({"yaw_deg": 0.4}, 0.4),
        ({"tilt_deg": 20.5}, 0.5),
        ({"roll_deg": 0.7}, 0.7),
    ], ids=["yaw", "tilt", "roll"])
    def test_the_reported_drift_is_the_real_rotation(self, station, change,
                                                     expected_deg):
        """Not an estimate: a rotation about the optical centre is exactly a
        homography, so undoing the intrinsics recovers the angle itself."""
        drifted = station.updated(**change)

        alignment = align_pixels(landmark_pixels(station), landmark_pixels(drifted),
                                 lens=LENS, method="exact")

        assert alignment.drift_deg == pytest.approx(expected_deg, abs=1e-4)

    def test_motion_in_pixels_overstates_the_angle(self, station):
        """Why drift_deg is not just RMS motion over focal length: a rotation
        moves off-axis pixels much further than on-axis ones."""
        drifted = station.updated(yaw_deg=0.4)

        alignment = align_pixels(landmark_pixels(station), landmark_pixels(drifted),
                                 lens=LENS, method="exact")

        naive_deg = math.degrees(math.atan(alignment.motion_px / FOCAL_PX))
        assert naive_deg > 1.4 * alignment.drift_deg

    def test_a_rigid_rotation_leaves_no_residual(self, station):
        """A camera rotating about its centre is exactly a homography. Nothing left."""
        drifted = station.updated(yaw_deg=0.3, tilt_deg=20.2, roll_deg=0.5)

        alignment = align_pixels(landmark_pixels(station), landmark_pixels(drifted),
                                 lens=LENS, method="exact")

        assert alignment.rms_px == pytest.approx(0.0, abs=1e-3)
        assert alignment.residual_deg == pytest.approx(0.0, abs=1e-4)

    def test_the_residual_angle_is_the_rms_read_through_the_focal_length(self, station):
        noisy = landmark_pixels(station.updated(yaw_deg=0.2))
        noisy[:, 0] += np.array([0.8, -0.6, 0.5, -0.9, 0.3, -0.4, 0.7])

        alignment = align_pixels(landmark_pixels(station), noisy,
                                 lens=LENS, method="exact")

        expected = math.degrees(math.atan(alignment.rms_px / FOCAL_PX))
        assert alignment.residual_deg == pytest.approx(expected)


class TestUndoingDrift:
    """The point of the exercise: does the position error actually go away?"""

    def _cost_m(self, view, truth_station, rows):
        """How far ``view`` puts a pixel from where the camera really sees it."""
        cols = np.full(rows.size, WIDTH / 2.0 + 200.0)
        placed = view.project(cols, rows)
        actual = truth_station.view.project(cols, rows)
        return np.hypot(placed.x - actual.x, placed.y - actual.y)

    def test_a_stale_homography_is_wrong_and_worsens_with_range(self, station):
        """Establish the cost first, or the correction proves nothing. Three
        tenths of a degree costs 0.13 m near the camera and 0.57 m at 500."""
        drifted = station.updated(tilt_deg=20.3)
        rows = np.array([900.0, 700.0, 500.0])

        cost = self._cost_m(station.view, drifted, rows)

        assert (cost > 0.1).all()
        assert (np.diff(cost) > 0.0).all(), "and it is worst furthest out"

    def test_realigning_cancels_it(self, station):
        drifted = station.updated(tilt_deg=20.3)
        rows = np.array([900.0, 700.0, 500.0])
        alignment = align_pixels(landmark_pixels(station), landmark_pixels(drifted),
                                 lens=LENS, method="exact")

        corrected = station.view.realigned(alignment.transform)

        np.testing.assert_allclose(self._cost_m(corrected, drifted, rows), 0.0,
                                   atol=1e-4)

    def test_a_camera_that_shifted_as_well_as_turned_needs_the_homography_model(
            self, station):
        """A mast that leaned as well as twisted is no longer a pure rotation.
        Landmarks on the plane make it a homography, so the wider model holds --
        and this is the one case worth paying its extra freedoms for."""
        drifted = station.updated(yaw_deg=0.5, x=0.4, y=-0.3, height_m=10.05)
        rows = np.array([900.0, 700.0, 500.0])
        alignment = align_pixels(landmark_pixels(station), landmark_pixels(drifted),
                                 lens=LENS, model="homography", method="exact")

        corrected = station.view.realigned(alignment.transform)

        np.testing.assert_allclose(self._cost_m(corrected, drifted, rows), 0.0,
                                   atol=1e-4)

    def test_a_station_keeps_its_position_but_loses_its_stale_orientation(self, station):
        """The mast did not move; where it points is now only the matrix's business."""
        drifted = station.updated(tilt_deg=20.3)
        alignment = align_pixels(landmark_pixels(station), landmark_pixels(drifted),
                                 lens=LENS)

        corrected = station.realigned(alignment.transform)

        assert (corrected.height_m, corrected.x, corrected.y) == (10.0, 0.0, 0.0)
        assert corrected.pose.tilt_deg is None
        assert not corrected.derived

    def test_realigning_twice_composes(self, station):
        """Drift is measured against the last alignment, not the original survey."""
        once = station.updated(tilt_deg=20.2)
        twice = once.updated(tilt_deg=20.2, yaw_deg=0.3)
        first = align_pixels(landmark_pixels(station), landmark_pixels(once),
                             lens=LENS, method="exact")
        second = align_pixels(landmark_pixels(once), landmark_pixels(twice),
                              lens=LENS, method="exact")

        corrected = station.view.realigned(first.transform).realigned(second.transform)

        rows = np.array([900.0, 700.0, 500.0])
        np.testing.assert_allclose(self._cost_m(corrected, twice, rows), 0.0, atol=1e-4)


class TestWhatItRefuses:
    """Bad input fails at the call, naming what is wrong."""

    def test_one_landmark_is_not_enough_for_a_rotation(self, station):
        pixels = landmark_pixels(station)[:1]

        with pytest.raises(ValueError, match="at least 2"):
            align_pixels(pixels, pixels, lens=LENS)

    def test_the_homography_model_still_wants_four(self, station):
        pixels = landmark_pixels(station)[:3]

        with pytest.raises(ValueError, match="at least 4"):
            align_pixels(pixels, pixels, lens=LENS, model="homography")

    def test_an_unknown_model_raises(self, station):
        pixels = landmark_pixels(station)

        with pytest.raises(ValueError, match="model must be one of"):
            align_pixels(pixels, pixels, lens=LENS, model="affine")

    def test_mismatched_counts_raise(self, station):
        pixels = landmark_pixels(station)

        with pytest.raises(ValueError, match="7 source points but 5"):
            align_pixels(pixels, pixels[:5], lens=LENS)

    def test_landmarks_on_one_line_break_the_homography_model_only(self, station):
        """Six marks along a sea wall are degenerate for an eight-parameter fit.
        The rotation model survives them: collinear pixels are still viewing
        directions spanning a plane, which is enough to orient a camera."""
        line = np.column_stack([np.linspace(200.0, 1700.0, 6), np.full(6, 800.0)])
        moved = line + np.array([3.0, 1.0])

        with pytest.raises(ValueError, match="degenerate|collinear"):
            align_pixels(line, moved, lens=LENS, model="homography", method="exact")

        assert align_pixels(line, moved, lens=LENS, method="exact").drift_deg > 0.0

    def test_an_unknown_method_raises(self, station):
        pixels = landmark_pixels(station)

        with pytest.raises(ValueError, match="method must be one of"):
            align_pixels(pixels, pixels, lens=LENS, method="icp")

    def test_a_transform_that_is_not_a_3x3_raises(self, camera):
        with pytest.raises(ValueError, match="finite 3x3"):
            camera.realigned(np.eye(2))

    def test_a_landmark_that_moved_shows_up_as_a_residual(self, station):
        """The check that stops a drifting buoy being mistaken for a drifting camera."""
        current = landmark_pixels(station.updated(yaw_deg=0.2))
        current[3] += np.array([40.0, 25.0])

        alignment = align_pixels(landmark_pixels(station), current,
                                 lens=LENS, method="exact")

        assert alignment.rms_px > 5.0


class TestThroughALens:
    """Distortion has to come off both images before the transform means anything."""

    def _barrelled(self, yaw_deg):
        """A station whose view carries real distortion, so unproject applies it."""
        lens = Lens(FOCAL_PX, FOCAL_PX, WIDTH / 2.0, HEIGHT / 2.0, k1=-0.12)
        pose = CameraPose(height_m=CAMERA_HEIGHT_M, tilt_deg=TILT_DEG, yaw_deg=yaw_deg)
        return CameraStation.from_pose(pose, lens, WIDTH, HEIGHT, MetricPlane())

    def test_drift_is_recovered_through_a_distorting_lens(self, station):
        """Raw pixels are not related by a homography; undistorted ones are."""
        lens = Lens(FOCAL_PX, FOCAL_PX, WIDTH / 2.0, HEIGHT / 2.0, k1=-0.12)
        raw = [landmark_pixels(self._barrelled(yaw)) for yaw in (0.0, 0.35)]

        corrected = align_pixels(raw[0], raw[1], lens=lens,
                                 method="exact")
        naive = align_pixels(raw[0], raw[1], lens=LENS, method="exact")

        assert corrected.rms_px == pytest.approx(0.0, abs=1e-3)
        assert naive.rms_px > corrected.rms_px

    def test_it_matches_the_undistorted_camera_it_came_from(self, station):
        """Take the lens off and the same rotation reads the same angle."""
        lens = Lens(FOCAL_PX, FOCAL_PX, WIDTH / 2.0, HEIGHT / 2.0, k1=-0.12)
        raw = [landmark_pixels(self._barrelled(yaw)) for yaw in (0.0, 0.35)]

        through = align_pixels(raw[0], raw[1], lens=lens,
                               method="exact")
        ideal = align_pixels(landmark_pixels(station),
                             landmark_pixels(station.updated(yaw_deg=0.35)),
                             lens=LENS, method="exact")

        assert through.drift_deg == pytest.approx(ideal.drift_deg, abs=1e-4)

    def test_a_landmark_the_lens_cannot_invert_raises(self, station):
        # k1 this strong has no inverse far off axis, which is a refusal, not a NaN:
        # a boresight measurement is not a per-pixel bulk operation.
        lens = Lens(FOCAL_PX, FOCAL_PX, WIDTH / 2.0, HEIGHT / 2.0, k1=-0.9)
        far = np.array([[-4000.0, -3000.0], [4000.0, -3000.0],
                        [-4000.0, HEIGHT + 3000.0], [4000.0, HEIGHT + 3000.0]])

        with pytest.raises(ValueError, match="no inverse"):
            align_pixels(far, far, lens=lens)


class TestWhyRotationIsTheDefault:
    """The eight-parameter fit has five freedoms the camera does not have, and
    clicking noise flows straight into them.

    The effect is statistical, so these tests are too: a single noise draw can
    flatter either model, and how badly the wider one misreads depends on the
    landmarks' spread and range. What holds across draws is the variance.
    """

    TRUE_DRIFT_DEG = 0.01

    def _estimates(self, station, model, draws=30):
        """Drift measured over many independent sets of clicks, truth held fixed."""
        drifted = station.updated(tilt_deg=20.0 + self.TRUE_DRIFT_DEG)
        rng = np.random.default_rng(4)
        out = []
        for _ in range(draws):
            noise = rng.normal(0.0, 0.5, (2, len(LANDMARKS_M), 2))
            out.append(align_pixels(landmark_pixels(station) + noise[0],
                                    landmark_pixels(drifted) + noise[1],
                                    lens=LENS, model=model, method="exact").drift_deg)
        return np.array(out)

    def test_the_wider_model_is_the_noisier_measurement(self, station):
        """Same landmarks, same clicks, same truth -- and several times the spread."""
        rotation = self._estimates(station, "rotation")
        homography = self._estimates(station, "homography")

        assert homography.std() > 3.0 * rotation.std()

    def test_the_rotation_model_stays_near_the_truth(self, station):
        """A hundredth of a degree is measurable with it, and that is the point:
        it is the threshold a monitoring rule has to sit above."""
        estimates = self._estimates(station, "rotation")

        assert estimates.mean() == pytest.approx(self.TRUE_DRIFT_DEG, abs=0.05)
        assert estimates.max() < 0.2

    def test_the_wider_model_hides_that_noise_in_its_residual(self, station):
        """And so it cannot be caught by checking rms_px: the extra freedoms
        absorb the very error that is corrupting the estimate."""
        rng = np.random.default_rng(1)
        noise = rng.normal(0.0, 0.5, (2, len(LANDMARKS_M), 2))
        reference = landmark_pixels(station) + noise[0]
        current = landmark_pixels(station.updated(tilt_deg=20.01)) + noise[1]

        rotation = align_pixels(reference, current, lens=LENS, method="exact")
        homography = align_pixels(reference, current, lens=LENS,
                                  model="homography", method="exact")

        assert homography.rms_px < rotation.rms_px
        assert rotation.rms_px == pytest.approx(0.5, abs=0.4), "about the click error"
