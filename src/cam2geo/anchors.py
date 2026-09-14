"""Which pixel of a detection box is its ground position."""

from __future__ import annotations

import numpy as np

__all__ = ["BOX_ANCHORS", "anchor_pixel"]

BOX_ANCHORS = ("centre", "bottom_centre")


def anchor_pixel(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    angle_deg: np.ndarray | float = 0.0,
    anchor: str = "bottom_centre",
) -> np.ndarray:
    """The anchor pixel of each box, from its top-left corner and size.

    ``bottom_centre`` is the default because a target standing on the plane meets
    it at its feet; its centre floats. See docs/LIMITATIONS.md, "The anchor".

    Args:
        x, y: Top-left corner in pixels.
        w, h: Box width and height in pixels.
        angle_deg: In-plane rotation about the box centre, anticlockwise.
        anchor: One of ``BOX_ANCHORS``.

    Returns:
        ``(N, 2)`` pixels.

    Raises:
        ValueError: If ``anchor`` is not one of ``BOX_ANCHORS``.
    """
    if anchor not in BOX_ANCHORS:
        raise ValueError("anchor must be one of " + ", ".join(BOX_ANCHORS)
                         + ", got " + repr(anchor))
    x, y, w, h = (np.asarray(a, dtype=float) for a in (x, y, w, h))
    offset = np.zeros((*np.shape(x), 2)) if anchor == "centre" else np.stack(
        [np.zeros_like(h), h / 2.0], axis=-1)
    t = np.radians(np.asarray(angle_deg, dtype=float))
    cos, sin = np.cos(t), np.sin(t)
    du = cos * offset[..., 0] - sin * offset[..., 1]
    dv = sin * offset[..., 0] + cos * offset[..., 1]
    return np.stack([x + w / 2.0 + du, y + h / 2.0 + dv], axis=-1)
