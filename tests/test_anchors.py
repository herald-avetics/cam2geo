"""Box anchors: which pixel of a detection is taken as its ground contact."""

from __future__ import annotations

import numpy as np
import pytest

from cam2geo import BOX_ANCHORS, anchor_pixel


class TestTheAnchors:
    def test_the_two_supported_anchors(self):
        assert BOX_ANCHORS == ("centre", "bottom_centre")

    def test_an_unknown_anchor_raises_with_the_value(self):
        with pytest.raises(ValueError, match="'corner'"):
            anchor_pixel(0.0, 0.0, 10.0, 10.0, anchor="corner")

    def test_the_centre_anchor_is_the_middle_of_the_box(self):
        out = anchor_pixel(100.0, 200.0, 40.0, 30.0, anchor="centre")
        np.testing.assert_allclose(out, [120.0, 215.0])

    def test_the_bottom_centre_anchor_is_the_middle_of_the_bottom_edge(self):
        out = anchor_pixel(100.0, 200.0, 40.0, 30.0, anchor="bottom_centre")
        np.testing.assert_allclose(out, [120.0, 230.0])

    def test_bottom_centre_is_the_default(self):
        """A target standing on the plane meets it at its feet, not its middle."""
        np.testing.assert_allclose(anchor_pixel(100.0, 200.0, 40.0, 30.0),
                                   anchor_pixel(100.0, 200.0, 40.0, 30.0,
                                                anchor="bottom_centre"))

    def test_it_is_vectorised_over_many_boxes(self):
        x = np.array([0.0, 100.0, 500.0])
        y = np.array([0.0, 200.0, 600.0])
        w = np.array([10.0, 40.0, 80.0])
        h = np.array([20.0, 30.0, 60.0])

        out = anchor_pixel(x, y, w, h)

        assert out.shape == (3, 2)
        np.testing.assert_allclose(out[:, 0], [5.0, 120.0, 540.0])
        np.testing.assert_allclose(out[:, 1], [20.0, 230.0, 660.0])

    def test_an_empty_batch_gives_an_empty_result(self):
        out = anchor_pixel(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        assert out.shape == (0, 2)


class TestRotatedBoxes:
    """angle_deg is carried because oriented boxes exist, not because they are common."""

    def test_zero_rotation_changes_nothing(self):
        plain = anchor_pixel(100.0, 200.0, 40.0, 30.0)
        rotated = anchor_pixel(100.0, 200.0, 40.0, 30.0, angle_deg=0.0)
        np.testing.assert_allclose(plain, rotated)

    def test_the_centre_anchor_is_unmoved_by_any_rotation(self):
        """Rotation is about the centre, so the centre is its fixed point."""
        for angle in (0.0, 37.0, 90.0, 180.0, -15.0):
            out = anchor_pixel(100.0, 200.0, 40.0, 30.0, angle_deg=angle,
                               anchor="centre")
            np.testing.assert_allclose(out, [120.0, 215.0], atol=1e-12)

    def test_a_quarter_turn_puts_the_bottom_edge_to_the_side(self):
        out = anchor_pixel(100.0, 200.0, 40.0, 30.0, angle_deg=90.0)

        # The offset (0, +h/2) rotates to (-h/2, 0).
        np.testing.assert_allclose(out, [120.0 - 15.0, 215.0], atol=1e-12)

    def test_a_half_turn_puts_it_on_top(self):
        out = anchor_pixel(100.0, 200.0, 40.0, 30.0, angle_deg=180.0)
        np.testing.assert_allclose(out, [120.0, 200.0], atol=1e-12)

    def test_the_anchor_stays_half_a_height_from_the_centre(self):
        """Whatever the angle, the anchor is on the box's circumscribed offset."""
        angles = np.array([0.0, 23.0, 90.0, 200.0, 355.0])

        out = anchor_pixel(np.full(5, 100.0), np.full(5, 200.0), np.full(5, 40.0),
                           np.full(5, 30.0), angle_deg=angles)
        distance = np.hypot(out[:, 0] - 120.0, out[:, 1] - 215.0)

        np.testing.assert_allclose(distance, 15.0, atol=1e-12)
