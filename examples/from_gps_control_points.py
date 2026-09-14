"""From four GPS-tagged marks to a latitude and longitude for every detection.

The common real case: a fixed camera, no calibration, but a handful of points on
the ground whose coordinates you know -- charted marks, a GPS visit, corners
picked off a satellite image. Fit a homography to those, and every pixel below
the horizon becomes a position.

    uv run python examples/from_gps_control_points.py

Needs the lens and geodetic extras: pip install 'cam2geo[lens,geodetic]'
"""

from __future__ import annotations

import numpy as np

from cam2geo import (
    SURFACES,
    CameraPose,
    CameraStation,
    GeodeticPlane,
    Lens,
    anchor_pixel,
    fit_homography,
    position_error,
)

WIDTH, HEIGHT = 1920, 1080
# Somewhere off a headland. Any origin does; the fit works in degrees.
LAT0, LON0 = 50.35, -4.14


def _pretend_survey() -> tuple[np.ndarray, np.ndarray]:
    """Stand in for the real work: six marks, their coordinates and their pixels.

    In your case these two arrays come from a chart and from clicking the image.
    Here a metric camera generates them and pymap3d turns its metres into
    degrees, which is the same trip a real survey makes in reverse.
    """
    import pymap3d

    truth = CameraStation.from_pose(
        CameraPose(height_m=30.0, x=0.0, y=0.0, tilt_deg=8.0, yaw_deg=0.0),
        Lens(1400.0, 1400.0, WIDTH / 2.0, HEIGHT / 2.0), WIDTH, HEIGHT,
    )
    # Marks spread across the water, in metres east and north of the mast.
    east = np.array([-60.0, 60.0, -40.0, 40.0, 0.0, 90.0])
    north = np.array([70.0, 70.0, 150.0, 150.0, 110.0, 260.0])
    u, v = truth.view.unproject(east, north)

    lat, lon, _ = pymap3d.ned2geodetic(north, east, 0.0, LAT0, LON0, 0.0)
    # A survey is never exact. Half a pixel of clicking error.
    rng = np.random.default_rng(7)
    pixels = np.column_stack([u, v]) + rng.normal(0.0, 0.5, (east.size, 2))
    return np.column_stack([np.asarray(lon), np.asarray(lat)]), pixels


def main() -> None:
    control_deg, control_px = _pretend_survey()

    # 1. Fit. Plane coordinates here are (longitude, latitude) in degrees, so the
    #    frame must say so -- that is the only place units are known.
    fit = fit_homography(control_deg, control_px, WIDTH, HEIGHT,
                         frame=GeodeticPlane(), method="exact")
    print(f"fit over {len(control_deg)} marks: "
          f"rms {fit.rms_px:.2f} px, worst {fit.max_px:.2f} px")

    # 2. Say where the camera stands. Not recovered from the matrix -- you know
    #    your own mast height, and a fitted homography does not.
    station = CameraStation(fit.homography,
                            CameraPose(height_m=30.0, x=LON0, y=LAT0),
                            name="headland")

    # 3. Detections: boxes as (x, y, w, h) from a top-left corner.
    boxes = np.array([[900.0, 940.0, 60.0, 40.0],
                      [1310.0, 760.0, 38.0, 26.0],
                      [640.0, 560.0, 22.0, 16.0],
                      [980.0, 300.0, 14.0, 10.0]])
    anchors = anchor_pixel(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])

    # 4. Project. A vessel's waterline sits on the plane, so no height offset;
    #    curvature is worth having over open water.
    placed = station.project(anchors[:, 0], anchors[:, 1], max_m_per_px=1.0,
                            curvature=True)

    # 5. Price it. A sea state with about half a metre of swell uncertainty.
    # The preset, adapted to the day: calmer than its default half-metre swell.
    surface = SURFACES["sea"].but(roughness_m=0.3)
    budget = position_error(station, anchors[:, 0], anchors[:, 1], surface,
                            pixel_sigma_px=1.0, pointing_sigma_deg=0.05,
                            control_rms_px=fit.rms_px, corrected_curvature=True)

    print(f"\n{'box':>4} {'longitude':>12} {'latitude':>11} {'range m':>9} "
          f"{'m/px':>8} {'+/- m':>8}")
    for i in range(len(boxes)):
        if not placed.valid[i]:
            why = ("beyond the horizon" if np.isnan(placed.ground_m_per_px[i])
                   else f"too coarse: {placed.ground_m_per_px[i]:.1f} m/px")
            print(f"{i:>4} {'--':>12} {'--':>11}   {why}")
            continue
        print(f"{i:>4} {placed.x[i]:12.6f} {placed.y[i]:11.6f} "
              f"{budget.range_m[i]:9.1f} {placed.ground_m_per_px[i]:8.3f} "
              f"{budget.total_m[i]:8.1f}")

    print(f"\n{placed.beyond_horizon} beyond the horizon, "
          f"{placed.ill_conditioned} too ill-conditioned to keep")

    # 6. What this camera can usefully see, as a ring of lon/lat you can plot.
    ring = station.view.footprint(max_m_per_px=1.0)
    reach = station.view.frame.distance_m(
        np.array([LON0]), np.array([LAT0]), ring[:, 0], ring[:, 1]).max()
    print(f"usable footprint: {len(ring)} vertices, reaching {reach:.0f} m")


if __name__ == "__main__":
    main()
