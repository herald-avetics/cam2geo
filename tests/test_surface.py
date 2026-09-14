"""Surface and position_error: what the ground does to a plane model, priced."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import WIDTH, build_station

from cam2geo import SURFACES, Surface, position_error

FLAT = Surface("ideal")


class TestSurface:
    """A profile is a description of a site, validated but never invented."""

    def test_the_default_surface_departs_from_the_plane_in_no_way(self):
        assert (FLAT.offset_m, FLAT.offset_sigma_m, FLAT.roughness_m) == (0.0, 0.0, 0.0)
        assert not FLAT.curvature

    def test_a_negative_roughness_raises_with_the_value(self):
        with pytest.raises(ValueError, match="-1.0"):
            Surface("bad", roughness_m=-1.0)

    def test_a_negative_offset_sigma_raises(self):
        with pytest.raises(ValueError, match="offset_sigma_m"):
            Surface("bad", offset_sigma_m=-0.5)

    @pytest.mark.parametrize("k", [-0.1, 1.0, 2.0], ids=["negative", "one", "over"])
    def test_an_out_of_range_refraction_coefficient_raises(self, k):
        with pytest.raises(ValueError, match="refraction_k"):
            Surface("bad", refraction_k=k)

    def test_but_returns_an_adapted_copy(self):
        """Presets are starting points, so adapting one must be a one-liner."""
        rough_day = SURFACES["sea"].but(offset_sigma_m=1.5)

        assert rough_day.offset_sigma_m == 1.5
        assert rough_day.curvature is SURFACES["sea"].curvature
        assert SURFACES["sea"].offset_sigma_m != 1.5, "the preset is untouched"

    def test_every_preset_is_valid_and_named(self):
        for key, surface in SURFACES.items():
            assert surface.name == key

    def test_only_the_sea_preset_bothers_with_curvature(self):
        """It matters over open water and essentially nowhere else."""
        curved = {k for k, s in SURFACES.items() if s.curvature}
        assert curved == {"sea"}

    def test_the_presets_are_ordered_by_how_flat_they_really_are(self):
        """A sanity check on the numbers, not on the physics."""
        assert (SURFACES["warehouse_floor"].roughness_m
                < SURFACES["runway"].roughness_m
                < SURFACES["sea"].roughness_m
                <= SURFACES["moorland"].roughness_m)


class TestPositionError:
    """Every term but the pixel ones scales with range, and the docs say so."""

    def test_a_perfectly_flat_surface_leaves_only_the_pixel_term(self, station):
        cols, rows = np.array([960.0]), np.array([900.0])

        budget = position_error(station, cols, rows, FLAT, pixel_sigma_px=1.0)

        assert budget.height_bias_m[0] == 0.0
        assert budget.height_sigma_m[0] == 0.0
        assert budget.roughness_m[0] == 0.0
        assert budget.curvature_m[0] == 0.0
        assert budget.pointing_m[0] == 0.0
        assert budget.bias_m[0] == 0.0
        assert budget.total_m[0] == pytest.approx(budget.pixel_m[0])

    def test_the_pixel_term_is_the_conditioning_times_the_error(self, station):
        cols, rows = np.array([960.0]), np.array([900.0])
        gsd = station.view.ground_m_per_px(cols, rows)

        budget = position_error(station, cols, rows, FLAT, pixel_sigma_px=2.5)

        np.testing.assert_allclose(budget.pixel_m, gsd * 2.5)

    def test_random_terms_combine_in_quadrature(self, station):
        """The GUM law of propagation, which needs independence to be valid."""
        cols, rows = np.array([960.0]), np.array([900.0])
        surface = Surface("mixed", offset_sigma_m=0.5, roughness_m=0.3)

        b = position_error(station, cols, rows, surface, pixel_sigma_px=1.0,
                           pointing_sigma_deg=0.05, control_rms_px=2.0)
        parts = np.array([b.pixel_m[0], b.height_sigma_m[0], b.roughness_m[0],
                          b.pointing_m[0], b.control_m[0]])

        assert b.random_m[0] == pytest.approx(float(np.sqrt((parts**2).sum())))

    def test_biases_combine_linearly_not_in_quadrature(self, station):
        """A bias points the same way every time, so quadrature would flatter it."""
        cols, rows = np.array([960.0]), np.array([700.0])
        surface = Surface("biased", offset_m=1.0, curvature=True)

        b = position_error(station, cols, rows, surface)

        assert b.bias_m[0] == pytest.approx(b.height_bias_m[0] + b.curvature_m[0])
        assert b.bias_m[0] > np.hypot(b.height_bias_m[0], b.curvature_m[0])

    def test_the_total_is_the_bias_plus_the_random_part(self, station):
        cols, rows = np.array([960.0]), np.array([700.0])
        surface = Surface("mixed", offset_m=1.0, offset_sigma_m=0.4, roughness_m=0.2)

        b = position_error(station, cols, rows, surface, pixel_sigma_px=1.0)

        assert b.total_m[0] == pytest.approx(b.bias_m[0] + b.random_m[0])

    def test_every_term_grows_with_range(self, station):
        rows = np.array([1000.0, 800.0, 600.0, 500.0])
        cols = np.full(rows.size, 960.0)
        surface = Surface("mixed", offset_m=0.4, offset_sigma_m=0.5,
                          roughness_m=0.3, curvature=True)

        b = position_error(station, cols, rows, surface, pixel_sigma_px=1.0,
                           pointing_sigma_deg=0.05, control_rms_px=1.0)

        for term in (b.range_m, b.pixel_m, b.height_bias_m, b.height_sigma_m,
                     b.roughness_m, b.curvature_m, b.pointing_m,
                     b.bias_m, b.random_m, b.total_m):
            assert np.all(np.diff(term) > 0.0), "every term worsens with distance"

    def test_half_a_metre_of_swell_beats_a_pixel_everywhere_usable(self, station):
        """The thesis of the error budget: pixel precision is not the limit.

        Everywhere the conditioning cutoff admits, half a metre of uncertainty in
        the target's height costs more than a whole pixel of detection error.
        """
        rows = np.linspace(300.0, station.view.height - 1.0, 40)
        cols = np.full(rows.size, station.view.width / 2.0)
        surface = Surface("swell", offset_sigma_m=0.5)

        usable = station.view.project(cols, rows, max_m_per_px=1.0).valid
        budget = position_error(station, cols, rows, surface, pixel_sigma_px=1.0)

        assert usable.sum() > 10, "there must be a usable region to speak about"
        assert np.all(budget.height_sigma_m[usable] > budget.pixel_m[usable])

    def test_but_the_pixel_term_catches_up_at_the_horizon(self, station):
        """Conditioning grows faster than range, so past the cutoff the pixel
        term overtakes everything. It is also where no answer is worth having."""
        col = np.array([station.view.width / 2.0])
        surface = Surface("swell", offset_sigma_m=0.5)

        usable = position_error(station, col, np.array([800.0]), surface)
        beyond = position_error(station, col, np.array([185.0]), surface)

        assert usable.height_sigma_m[0] > usable.pixel_m[0]
        assert beyond.pixel_m[0] > beyond.height_sigma_m[0]

    def test_correcting_the_height_removes_the_bias_and_leaves_the_spread(self, station):
        cols, rows = np.array([960.0]), np.array([800.0])
        surface = Surface("swell", offset_m=1.0, offset_sigma_m=0.3)

        raw = position_error(station, cols, rows, surface)
        fixed = position_error(station, cols, rows, surface, corrected_height=True)

        assert raw.height_bias_m[0] > 0.0
        assert fixed.height_bias_m[0] == 0.0
        assert fixed.height_sigma_m[0] == pytest.approx(raw.height_sigma_m[0])

    def test_correcting_the_curvature_removes_that_bias_too(self, station):
        cols, rows = np.array([960.0]), np.array([400.0])
        surface = Surface("sea", curvature=True)

        raw = position_error(station, cols, rows, surface)
        fixed = position_error(station, cols, rows, surface, corrected_curvature=True)

        assert raw.curvature_m[0] > 0.0
        assert fixed.curvature_m[0] == 0.0

    def test_the_height_bias_is_the_relief_displacement(self, station):
        """It must equal what the correction itself would move the point by."""
        cols, rows = np.array([960.0]), np.array([800.0])
        height_m = 1.0

        ground = station.project(cols, rows)
        raised = station.project(cols, rows, target_height_m=height_m)
        moved = np.hypot(raised.x - ground.x, raised.y - ground.y)
        budget = position_error(station, cols, rows, Surface("s", offset_m=height_m))

        np.testing.assert_allclose(budget.height_bias_m, moved, rtol=1e-9)

    def test_range_is_nan_beyond_the_horizon_and_so_is_the_total(self, station):
        budget = position_error(station, np.array([960.0]), np.array([100.0]), FLAT)

        assert np.isnan(budget.range_m[0])
        assert np.isnan(budget.total_m[0])

    def test_one_entry_per_pixel(self, station):
        cols = np.full(5, 960.0)
        rows = np.linspace(600.0, 1000.0, 5)

        budget = position_error(station, cols, rows, FLAT)

        assert budget.total_m.shape == (5,)
        assert budget.bias_m.shape == (5,)

    def test_a_lower_camera_is_hurt_more_by_the_same_swell(self):
        """The lever is range over camera height, so height buys accuracy."""
        low = build_station(height_m=5.0)
        high = build_station(height_m=50.0)
        col, row = np.array([WIDTH / 2.0]), np.array([1000.0])
        surface = Surface("swell", offset_sigma_m=1.0)

        # Compare at a matched range rather than a matched pixel.
        assert (position_error(low, col, row, surface).height_sigma_m[0]
                / low.range_m(col, row)[0]) > (
            position_error(high, col, row, surface).height_sigma_m[0]
            / high.range_m(col, row)[0])

    def test_pointing_error_is_an_angle_not_a_distance(self, station):
        """A degree of sway is worth more metres the further out you look."""
        rows = np.array([1000.0, 700.0])
        cols = np.full(rows.size, 960.0)

        b = position_error(station, cols, rows, FLAT, pointing_sigma_deg=0.1)
        half = position_error(station, cols, rows, FLAT, pointing_sigma_deg=0.05)

        np.testing.assert_allclose(half.pointing_m, b.pointing_m / 2.0, rtol=1e-12)
        assert b.pointing_m[1] > b.pointing_m[0]
