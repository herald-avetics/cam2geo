"""Plane frames: how the homography's output units become metres."""

from __future__ import annotations

import numpy as np
import pytest

from cam2geo import GeodeticPlane, MetricPlane


class TestMetricPlane:
    """The simple case: the homography already maps to a linear ground unit."""

    def test_a_non_positive_scale_raises_with_the_value(self):
        with pytest.raises(ValueError, match="-1.0"):
            MetricPlane(-1.0)

    def test_metres_are_the_identity(self):
        frame = MetricPlane()

        m = frame.metres_per_unit(np.array([10.0, 20.0]), np.array([0.0, -5.0]))

        assert m.shape == (2, 2, 2)
        np.testing.assert_allclose(m[0], np.eye(2))

    def test_a_non_metre_unit_scales_everything(self):
        """A homography fitted in feet is still usable, once it says so."""
        feet = MetricPlane(metres_per_plane_unit=0.3048)

        m = feet.metres_per_unit(np.array([0.0]), np.array([0.0]))
        d = feet.distance_m(np.array([0.0]), np.array([0.0]),
                            np.array([3.0]), np.array([4.0]))

        np.testing.assert_allclose(m[0], np.eye(2) * 0.3048)
        assert d[0] == pytest.approx(5.0 * 0.3048)

    def test_distance_is_euclidean(self):
        frame = MetricPlane()

        d = frame.distance_m(np.array([0.0]), np.array([0.0]),
                             np.array([3.0]), np.array([4.0]))

        assert d[0] == pytest.approx(5.0)

    def test_the_scale_does_not_depend_on_position(self):
        frame = MetricPlane()

        near = frame.metres_per_unit(np.array([0.0]), np.array([0.0]))
        far = frame.metres_per_unit(np.array([1e5]), np.array([-1e5]))

        np.testing.assert_array_equal(near, far)


class TestGeodeticPlane:
    """Degrees of latitude and longitude, checked against pymap3d itself."""

    def test_it_asks_for_pymap3d_by_name_when_absent(self, monkeypatch):
        """The error must name the extra to install, not just fail to import."""
        monkeypatch.setattr("builtins.__import__", _refuse_pymap3d)
        with pytest.raises(ImportError, match="pymap3d"):
            GeodeticPlane().metres_per_unit(np.array([10.0]), np.array([20.0]))

    def test_a_degree_of_latitude_is_about_111_km(self):
        pytest.importorskip("pymap3d")

        m = GeodeticPlane().metres_per_unit(np.array([10.0]), np.array([20.0]))

        assert 110_000 < m[0, 0, 0] < 112_000

    def test_a_degree_of_longitude_shrinks_with_latitude(self):
        """The property that makes degrees the wrong unit to reason in."""
        pytest.importorskip("pymap3d")
        frame = GeodeticPlane()

        equator = frame.metres_per_unit(np.array([0.0]), np.array([0.0]))
        high = frame.metres_per_unit(np.array([60.0]), np.array([0.0]))

        assert high[0, 1, 1] == pytest.approx(equator[0, 1, 1] * 0.5, rel=0.01)

    def test_the_scale_matrix_is_diagonal(self):
        """Latitude moves north and longitude moves east, and they do not mix."""
        pytest.importorskip("pymap3d")

        m = GeodeticPlane().metres_per_unit(np.array([45.0]), np.array([7.0]))

        assert m[0, 0, 1] == 0.0
        assert m[0, 1, 0] == 0.0

    def test_distance_matches_pymap3d_directly(self):
        """The oracle: our distance is pymap3d's, or it is wrong."""
        pm = pytest.importorskip("pymap3d")
        lat0, lon0 = np.array([10.0]), np.array([20.0])
        lat1, lon1 = np.array([10.01]), np.array([20.02])

        ours = GeodeticPlane().distance_m(lat0, lon0, lat1, lon1)
        n, e, _ = pm.geodetic2ned(lat1, lon1, 0.0, lat0, lon0, 0.0)

        np.testing.assert_allclose(ours, np.hypot(n, e), rtol=1e-12)

    def test_the_linearised_scale_agrees_with_a_real_geodesic_step(self):
        """A metre-scale step should be described by the local scale matrix."""
        pytest.importorskip("pymap3d")
        frame = GeodeticPlane()
        lat, lon = np.array([10.0]), np.array([20.0])
        step_deg = 1e-4

        m = frame.metres_per_unit(lat, lon)
        predicted = m[0, 0, 0] * step_deg
        actual = frame.distance_m(lat, lon, lat + step_deg, lon)

        assert predicted == pytest.approx(actual[0], rel=1e-6)


def _refuse_pymap3d(name, *args, **kwargs):
    """An __import__ that pretends pymap3d is not installed."""
    if name == "pymap3d":
        raise ImportError("no pymap3d")
    return _REAL_IMPORT(name, *args, **kwargs)


_REAL_IMPORT = __import__
