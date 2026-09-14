"""The smallest useful program: four known points in, world coordinates out.

No calibration, no telemetry, no camera model. Click four things in the image
whose coordinates you know, and every pixel below the horizon becomes a position.

    uv run python examples/four_points.py

Needs the lens and geodetic extras: pip install 'cam2geo[lens,geodetic]'
"""

from __future__ import annotations

import numpy as np

from cam2geo import GeodeticPlane, fit_homography

WIDTH, HEIGHT = 1920, 1080

# Four marks on the water off a headland: where each one sits, and where it
# appears. In your case the first column comes off a chart or a satellite image
# and the second from clicking the frame. These were generated once by the
# synthetic camera in demo.py, with 0.4 px of clicking error added.
CONTROL = [
    # (longitude, latitude),      (column, row)
    ((-4.140562, 50.350719), (288.5, 852.4)),
    ((-4.139438, 50.350719), (1632.0, 851.6)),
    ((-4.140773, 50.351528), (513.6, 588.9)),
    ((-4.139227, 50.351528), (1406.6, 589.1)),
]

# Two more, to make the residual mean something. See the note at the bottom.
EXTRA = [
    ((-4.140000, 50.351079), (960.3, 687.3)),
    ((-4.138665, 50.352517), (1433.2, 493.9)),
]


def place(control, label: str) -> None:
    """Fit to these control points, then put three detections on the ground."""
    plane = np.array([c[0] for c in control])
    pixels = np.array([c[1] for c in control])

    view = fit_homography(plane, pixels, WIDTH, HEIGHT, frame=GeodeticPlane(),
                          method="exact")

    detections = np.array([[700.0, 900.0], [1200.0, 640.0], [980.0, 300.0]])
    placed = view.homography.project(detections[:, 0], detections[:, 1])

    print(f"\n{label}: {len(control)} marks, "
          f"rms {view.rms_px:.2f} px, worst {view.max_px:.2f} px")
    for i, (u, v) in enumerate(detections):
        if not placed.valid[i]:
            print(f"  ({u:6.0f},{v:5.0f})  beyond the horizon")
            continue
        print(f"  ({u:6.0f},{v:5.0f})  {placed.x[i]:11.6f} {placed.y[i]:10.6f}"
              f"   {placed.ground_m_per_px[i]:6.3f} m/px")


def main() -> None:
    place(CONTROL, "four marks")
    place(CONTROL + EXTRA, "six marks")

    print("\nFour points is the minimum a homography admits, and it fits them")
    print("exactly -- an rms of 0.00 px however wrong the survey is. The first")
    print("honest signal about the fit arrives with the fifth mark.")


if __name__ == "__main__":
    main()
