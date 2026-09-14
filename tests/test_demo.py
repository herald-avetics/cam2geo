"""The demo setups, which every example in the documentation is built on."""

from __future__ import annotations

import numpy as np
import pytest

from cam2geo import SETUPS, position_error


@pytest.fixture(params=SETUPS, ids=lambda f: f.__name__)
def setup(request):
    """Each named setup in turn, so the contract holds for all three."""
    return request.param()


class TestEverySetup:
    """Whatever the camera, the same guarantees must hold."""

    def test_it_is_named_and_described(self, setup):
        assert setup.name
        assert setup.description
        assert setup.station.name == setup.name

    def test_it_can_be_re_aimed(self, setup):
        """Derived from a pose, so the docs can show a camera being adjusted."""
        assert setup.station.derived
        assert setup.station.updated(tilt_deg=15.0).pose.tilt_deg == 15.0

    def test_the_bottom_of_the_frame_sees_the_ground(self, setup):
        placed = setup.view.project(np.array([setup.view.width / 2.0]),
                                    np.array([setup.view.height - 1.0]))
        assert placed.valid[0]

    def test_the_horizon_is_inside_the_frame(self, setup):
        """All three are oblique views, which is the interesting case."""
        line = setup.view.horizon_line
        row = -(line[0] * setup.view.width / 2.0 + line[2]) / line[1]

        assert 0.0 < row < setup.view.height

    def test_conditioning_worsens_upward(self, setup):
        rows = np.linspace(setup.view.height - 1.0,
                           setup.view.height * 0.75, 5)
        cols = np.full(rows.size, setup.view.width / 2.0)

        gsd = setup.view.ground_m_per_px(cols, rows)

        assert np.all(np.diff(gsd) > 0.0)

    def test_it_has_a_usable_footprint(self, setup):
        assert len(setup.view.footprint(max_m_per_px=1.0)) > 3

    def test_its_surface_prices_an_error_budget(self, setup):
        rows = np.array([setup.view.height - 1.0, setup.view.height * 0.8])
        cols = np.full(rows.size, setup.view.width / 2.0)

        budget = position_error(setup.station, cols, rows, setup.surface,
                                pixel_sigma_px=1.0, pointing_sigma_deg=0.05)

        assert np.isfinite(budget.total_m).all()
        assert budget.total_m[1] > budget.total_m[0]


class TestTheDistortedSetup:
    """wide_quay is the one with a real lens, so it exercises the OpenCV path."""

    def test_it_carries_distortion(self):
        from cam2geo import wide_quay

        assert not wide_quay().view.lens.is_identity

    def test_it_is_kept_out_of_setups(self):
        """Projecting it needs OpenCV; the other three keep the docs numpy-only."""
        from cam2geo import wide_quay

        assert wide_quay not in SETUPS

    def test_it_projects_through_the_lens(self):
        pytest.importorskip("cv2", reason="a distorted view needs OpenCV")
        from cam2geo import wide_quay

        placed = wide_quay().view.project(np.array([960.0]), np.array([1000.0]))

        assert placed.valid[0]

    def test_undistortion_actually_moves_a_corner_pixel(self):
        pytest.importorskip("cv2", reason="a distorted view needs OpenCV")
        from cam2geo import wide_quay

        lens = wide_quay().view.lens
        u, v = np.array([1500.0]), np.array([900.0])

        ideal_u, ideal_v = lens.undistort(u, v)

        assert np.hypot(ideal_u[0] - u[0], ideal_v[0] - v[0]) > 5.0

    def test_ignoring_the_lens_moves_the_ground_point(self):
        """If modelling distortion cost nothing, there would be no reason to."""
        pytest.importorskip("cv2", reason="a distorted view needs OpenCV")
        from cam2geo import PlaneHomography, wide_quay

        setup = wide_quay()
        bare = PlaneHomography(setup.view.matrix, setup.view.width,
                               setup.view.height, frame=setup.view.frame)
        u, v = np.array([1500.0]), np.array([900.0])

        through_lens = setup.view.project(u, v)
        ignoring_it = bare.project(u, v)

        moved = np.hypot(through_lens.x[0] - ignoring_it.x[0],
                         through_lens.y[0] - ignoring_it.y[0])
        range_m = setup.station.range_m(u, v)[0]

        assert moved / range_m > 0.05, "ignoring the lens costs several per cent"


class TestTheSetupsDiffer:
    """They exist to span the range of cases, so they must not be near-copies."""

    def test_they_have_distinct_names(self):
        names = [make().name for make in SETUPS]
        assert len(set(names)) == len(names)

    def test_the_coastal_camera_reaches_much_further_than_the_pole(self):
        from cam2geo import coastal_mast, pole_camera

        coastal, pole = coastal_mast(), pole_camera()
        col = np.array([960.0])
        row = np.array([1079.0])

        assert coastal.station.range_m(col, row)[0] > 5.0 * pole.station.range_m(
            col, row)[0]

    def test_only_the_coastal_setup_models_curvature(self):
        from cam2geo import coastal_mast, moorland_mast, pole_camera

        assert coastal_mast().surface.curvature
        assert not moorland_mast().surface.curvature
        assert not pole_camera().surface.curvature

    def test_the_moorland_camera_is_the_most_grazing(self):
        """A long lens on a low mast is the hardest case for conditioning."""
        from cam2geo import moorland_mast, pole_camera

        col, row = np.array([960.0]), np.array([1000.0])
        moor = moorland_mast().view.ground_m_per_px(col, row)[0]
        pole = pole_camera().view.ground_m_per_px(col, row)[0]

        assert moor > pole
