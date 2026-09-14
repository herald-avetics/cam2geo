"""Eight named features read off one camera frame, then boresighted.

The whole workflow on one real scene, with nothing computed that a person would
not have typed in: eight things you can identify in *both* an oblique camera
frame and a Sentinel-2 scene, their pixel positions in the frame, and their
coordinates read off the satellite image. Then the same eight features found
again in a frame taken months later, which is all boresighting needs.

Run it with no arguments -- the two point tables below are the input, and they
are literals on purpose. To use your own camera, replace them:

    uv run python examples/from_a_real_image.py
    uv run python examples/from_a_real_image.py --image plymouth_2026-03.jpg

Getting the coordinates, which is the only part needing anything downloaded:

    1. Open the Copernicus Browser (dataspace.copernicus.eu) and find your
       scene. It is free and open, commercial use included; the credit line is
       "Copernicus Sentinel data [Year]". See examples/images/MANIFEST.yaml.
    2. Identify features you can also see in the camera frame. Hard edges and
       isolated objects survive 10 m per pixel; a shoreline or a field boundary
       does not. Breakwater ends, fort corners, beacons and slipways do.
    3. Read off longitude and latitude for each, and click the same feature in
       the camera frame. Spread them across the frame, not along one edge.

Every coordinate below is **estimated and then deliberately corrupted by 5 m**,
half a Sentinel-2 pixel, because that is what reading marks off a 10 m image
actually costs. Treat them the way the workflow treats them -- as ground truth --
while knowing they are not. Site centre is approximate.

Needs the lens and geodetic extras: pip install 'cam2geo[lens,geodetic]'
"""

from __future__ import annotations

import argparse

import numpy as np

from cam2geo import (
    SURFACES,
    CameraPose,
    CameraStation,
    GeodeticPlane,
    Lens,
    align_pixels,
    anchor_pixel,
    fit_homography,
    position_error,
)

WIDTH, HEIGHT = 1920, 1080
#: The camera: a 30 m mast on the headland, 8 deg down, 1400 px focal length.
MAST_HEIGHT_M, LENS = 30.0, Lens(1400.0, 1400.0, WIDTH / 2.0, HEIGHT / 2.0)
#: Where the mast stands. Approximate.
LAT0, LON0 = 50.3320, -4.1600

NAMES = ["breakwater west end", "breakwater east end", "fort NW corner",
         "fort SE corner", "navigation beacon", "mooring buoy A",
         "slipway toe", "quay corner"]

#: March frame: (column, row) clicked in the image, (longitude, latitude) read
#: off Sentinel-2. Eight marks, well spread, all below the horizon at row 343.
MARKS = [
    (65.9, 434.0, -4.164265, 50.336202),     # breakwater west end
    (1883.7, 436.9, -4.155808, 50.336102),   # breakwater east end
    (658.8, 404.2, -4.162128, 50.338277),    # fort NW corner
    (1152.3, 404.7, -4.158709, 50.338207),   # fort SE corner
    (987.2, 424.8, -4.159954, 50.336677),    # navigation beacon
    (557.1, 472.1, -4.161242, 50.334932),    # mooring buoy A
    (1657.5, 483.7, -4.157896, 50.334824),   # slipway toe
    (1755.8, 399.2, -4.154030, 50.338759),   # quay corner
]

#: September frame, same eight features, same order. No coordinates needed --
#: the camera has moved and only the features' motion in the image matters.
LATER = [
    (53.8, 434.5), (1873.3, 424.8), (650.9, 400.7), (1144.3, 398.4),
    (978.9, 420.3), (549.4, 469.3), (1647.1, 473.5), (1743.9, 388.8),
]

#: Detections to place: boxes as (x, y, w, h) from a top-left corner.
BOXES = np.array([[880.0, 900.0, 70.0, 48.0],
                  [1240.0, 640.0, 44.0, 30.0],
                  [700.0, 500.0, 26.0, 18.0],
                  [1010.0, 300.0, 16.0, 11.0]])


def load_frame(path: str | None) -> None:
    """Report the frame's size if one was supplied, and check it matches."""
    if path is None:
        print("no --image given: the point tables are the input, so the frame is\n"
              "only needed to click new marks or to draw an overlay.\n")
        return
    from cam2geo.lens import require_cv2
    cv2 = require_cv2()
    image = cv2.imread(path)
    if image is None:
        raise SystemExit("could not read " + path)
    height, width = image.shape[:2]
    print(f"{path}: {width}x{height}")
    if (width, height) != (WIDTH, HEIGHT):
        raise SystemExit(
            f"the marks below were clicked on a {WIDTH}x{HEIGHT} frame, so a "
            f"{width}x{height} one needs its own. Pixel coordinates are not "
            "resolution-independent."
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image", help="the camera frame the marks were clicked on")
    args = parser.parse_args()

    print("Copernicus Sentinel data 2026 -- see examples/images/MANIFEST.yaml\n")
    load_frame(args.image)

    marks = np.array(MARKS)
    pixels, degrees = marks[:, :2], marks[:, 2:]

    # 1. The eight marks, as you would check them against the frame.
    print(f"{'#':>2}  {'feature':<21} {'column':>8} {'row':>7} "
          f"{'longitude':>11} {'latitude':>10}")
    for i, name in enumerate(NAMES):
        print(f"{i:>2}  {name:<21} {pixels[i, 0]:8.1f} {pixels[i, 1]:7.1f} "
              f"{degrees[i, 0]:11.6f} {degrees[i, 1]:10.6f}")

    # 2. Fit. Least squares over every mark, NOT ransac -- see the comparison
    #    printed below, which is the reason.
    fit = fit_homography(degrees, pixels, WIDTH, HEIGHT, frame=GeodeticPlane(),
                         method="exact")
    print(f"\nfit over {len(marks)} marks: rms {fit.rms_px:.2f} px, "
          f"worst {fit.max_px:.2f} px, kept {int(fit.inliers.sum())}")
    print("  that residual is the 10 m imagery, not the clicking. It is what 5 m")
    print("  of coordinate error looks like once projected into pixels.")

    # Why not a robust method, when eight marks could carry a misidentified one?
    print("\n  method   threshold   rms px   kept   discarded")
    for method, threshold in [("exact", None), ("ransac", 8.0),
                              ("ransac", 20.0), ("lmeds", None)]:
        trial = fit_homography(degrees, pixels, WIDTH, HEIGHT,
                               frame=GeodeticPlane(), method=method,
                               ransac_threshold_px=threshold or 3.0)
        dropped = [NAMES[i] for i in np.flatnonzero(~trial.inliers)]
        shown = "--" if threshold is None else f"{threshold:.0f} px"
        print(f"  {method:<8} {shown:>9} {trial.rms_px:8.2f} "
              f"{int(trial.inliers.sum()):>6}   {', '.join(dropped) or '--'}")
    print("\n  Nothing in this data is an outlier: every mark carries the same 5 m.")
    print("  Yet each robust method discards some, and *which* ones depends on the")
    print("  threshold. Worse, the reported rms spans a factor of two hundred")
    print("  while the real accuracy barely moves -- a robust fit to a lucky")
    print("  subset reports a beautiful residual and is no more correct.")
    print("  RANSAC assumes a few marks are badly wrong. Marks read off coarse")
    print("  imagery are instead *all* mildly wrong, which is the case it cannot")
    print("  handle. Use exact, and distrust a suspiciously small rms.")

    station = CameraStation(fit.homography,
                            CameraPose(height_m=MAST_HEIGHT_M, x=LON0, y=LAT0),
                            name="headland")

    # 3. Place detections. Waterline at the bottom of the box, swell corrected.
    sea = SURFACES["sea"].but(offset_m=0.8, roughness_m=0.4)
    anchors = anchor_pixel(BOXES[:, 0], BOXES[:, 1], BOXES[:, 2], BOXES[:, 3])
    placed = station.project(anchors[:, 0], anchors[:, 1], max_m_per_px=2.0,
                             target_height_m=sea.offset_m, curvature=True)
    budget = position_error(station, anchors[:, 0], anchors[:, 1], sea,
                            pixel_sigma_px=1.0, pointing_sigma_deg=0.02,
                            control_rms_px=fit.rms_px, corrected_height=True,
                            corrected_curvature=True)

    print(f"\n{'box':>4} {'longitude':>12} {'latitude':>11} {'range m':>9} "
          f"{'m/px':>8} {'+/- m':>8}")
    for i in range(len(BOXES)):
        if not placed.valid[i]:
            why = ("beyond the horizon" if np.isnan(placed.ground_m_per_px[i])
                   else f"too coarse at {placed.ground_m_per_px[i]:.1f} m/px")
            print(f"{i:>4} {'--':>12} {'--':>11}   {why}")
            continue
        print(f"{i:>4} {placed.x[i]:12.6f} {placed.y[i]:11.6f} "
              f"{budget.range_m[i]:9.0f} {placed.ground_m_per_px[i]:8.2f} "
              f"{budget.total_m[i]:8.1f}")

    # 4. Boresight against the September frame. The same eight features, and
    #    this time their coordinates are irrelevant -- only their motion is.
    alignment = align_pixels(pixels, np.array(LATER), lens=LENS)
    print(f"\nSeptember frame: the camera has turned "
          f"{alignment.drift_deg:.3f} deg "
          f"(features moved {alignment.motion_px:.1f} px)")
    print(f"  residual {alignment.rms_px:.2f} px = "
          f"{alignment.residual_deg:.4f} deg unexplained, "
          f"kept {int(alignment.inliers.sum())} of {len(LATER)}")

    stale = station.project(anchors[:, 0], anchors[:, 1],
                            target_height_m=sea.offset_m, curvature=True)
    fixed = station.realigned(alignment.transform).project(
        anchors[:, 0], anchors[:, 1], target_height_m=sea.offset_m,
        curvature=True)
    print(f"\n{'box':>4} {'range m':>9} {'moved by':>10}   using the March matrix "
          f"on September pixels")
    for i in range(len(BOXES)):
        if not (np.isfinite(stale.x[i]) and np.isfinite(fixed.x[i])):
            print(f"{i:>4} {'--':>9} {'--':>10}   not placeable")
            continue
        moved = station.view.frame.distance_m(
            np.array([stale.x[i]]), np.array([stale.y[i]]),
            np.array([fixed.x[i]]), np.array([fixed.y[i]]))[0]
        print(f"{i:>4} {budget.range_m[i]:9.0f} {moved:9.1f} m")

    print("\nThat last column is the error you would have carried silently all")
    print("summer. It needed no survey to remove -- only the same eight features,")
    print("found twice. The fit residual above is the part that does need one.")


if __name__ == "__main__":
    main()
