"""Three real terrain types, control points off Sentinel-2, corrections in action.

Sentinel-2 is free, open and redistributable (`examples/images/MANIFEST.yaml`),
and it is **10 m per pixel**. So the control points you read off it are good to
about half a pixel, +/- 5 m, and that error goes into the fit rather than into
your clicking. This example is about what that does and what you can still fix.

The error model is therefore the opposite way round from the other examples: the
pixel clicks are good (half a pixel) and the *world coordinates are bad* (metres),
because they came from coarse imagery rather than from a survey.

Sites are named so you can find them on Sentinel-2 yourself. Their centres are
**approximate, to two decimal places**, and every control coordinate below is
**estimated, then deliberately corrupted** -- treated as ground truth the way a
real workflow treats it, because that is the honest simulation of picking marks
off a 10 m image. Replace all of it with your own.

    uv run python examples/terrain_sites.py

Needs the lens and geodetic extras: pip install 'cam2geo[lens,geodetic]'
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymap3d

from cam2geo import (
    SURFACES,
    CameraPose,
    CameraStation,
    Lens,
    Surface,
    align_pixels,
    fit_homography,
    position_error,
)
from cam2geo.frames import GeodeticPlane

WIDTH, HEIGHT = 1920, 1080

#: Half a Sentinel-2 pixel. This is the number the whole example turns on.
SENTINEL_PICK_M = 5.0
#: How well an operator clicks a feature in the camera frame.
CLICK_PX = 0.5


@dataclass(frozen=True)
class Site:
    """A real place, a plausible camera on it, and what breaks its plane."""

    name: str
    terrain: str          # the row it matches in README's catalogue
    features: str         # what is actually distinct at 10 m per pixel
    lat: float            # approximate, for finding the site
    lon: float
    height_m: float       # camera above the surface
    tilt_deg: float
    focal_px: float
    marks_m: np.ndarray   # control points, metres east/north of the camera
    surface: Surface
    target_height_m: float   # what sits above the plane, and is correctable
    rows: np.ndarray         # image rows to report on


SITES = [
    Site(
        name="Camargue salt pans",
        terrain="playas, salt flats, dry lake beds",
        features="evaporation-pond dykes -- a geometric grid, crisp at 10 m",
        lat=43.45, lon=4.74,
        height_m=12.0, tilt_deg=6.0, focal_px=1800.0,
        # Dyke intersections on a roughly 200 m pond grid.
        marks_m=np.array([[-90.0, 210.0], [0.0, 205.0], [95.0, 215.0],
                          [-150.0, 420.0], [10.0, 415.0], [150.0, 425.0]]),
        surface=SURFACES["salt_flat"],
        target_height_m=0.0,
        rows=np.array([531.0, 450.0, 417.0, 399.0]),
    ),
    Site(
        name="Maasvlakte container terminal",
        terrain="ports, container yards, rail yards, car parks",
        features="quay edges and stack corners -- straight lines, high contrast",
        lat=51.95, lon=4.05,
        height_m=35.0, tilt_deg=12.0, focal_px=1600.0,
        marks_m=np.array([[-85.0, 165.0], [85.0, 160.0], [-150.0, 330.0],
                          [150.0, 340.0], [0.0, 250.0], [240.0, 520.0]]),
        # Engineered flat. What breaks the plane is what is stacked on it.
        # A stack is 2.6 m to its top castings, known to about 0.4 m.
        surface=SURFACES["container_yard"].but(offset_m=2.6, offset_sigma_m=0.4),
        target_height_m=2.6,      # a container's top corner castings
        rows=np.array([660.0, 428.0, 344.0, 297.0]),
    ),
    Site(
        name="Plymouth Sound breakwater",
        terrain="open sea, lakes, estuaries, harbours",
        features="breakwater ends and fort walls -- isolated, unambiguous points",
        lat=50.33, lon=-4.16,
        height_m=30.0, tilt_deg=8.0, focal_px=1400.0,
        marks_m=np.array([[-170.0, 300.0], [170.0, 310.0], [-330.0, 620.0],
                          [330.0, 640.0], [0.0, 450.0], [500.0, 980.0]]),
        surface=SURFACES["sea"].but(offset_m=0.8, roughness_m=0.4),
        target_height_m=0.8,      # mean swell lifting the waterline
        rows=np.array([554.0, 438.0, 397.0, 374.0]),
    ),
]


def truth_camera(site: Site) -> CameraStation:
    """The camera as it really is. Only the simulation knows this."""
    return CameraStation.from_pose(
        CameraPose(height_m=site.height_m, tilt_deg=site.tilt_deg, yaw_deg=0.0),
        Lens(site.focal_px, site.focal_px, WIDTH / 2.0, HEIGHT / 2.0),
        WIDTH, HEIGHT, name=site.name)


def survey_off_sentinel(site: Site, rng) -> tuple[np.ndarray, np.ndarray]:
    """Control points as Sentinel-2 and an operator between them produce them.

    Returns the estimated (longitude, latitude) of each mark -- wrong by metres,
    because 10 m imagery cannot do better -- and its pixel, wrong by half a pixel.
    """
    truth = truth_camera(site)
    u, v = truth.view.unproject(site.marks_m[:, 0], site.marks_m[:, 1])
    pixels = np.column_stack([u, v]) + rng.normal(0.0, CLICK_PX, (len(u), 2))

    # The mark's true position, displaced by what a 10 m pixel cannot resolve.
    picked = site.marks_m + rng.normal(0.0, SENTINEL_PICK_M, site.marks_m.shape)
    lat, lon, _ = pymap3d.ned2geodetic(picked[:, 1], picked[:, 0], 0.0,
                                        site.lat, site.lon, 0.0)
    return np.column_stack([np.asarray(lon), np.asarray(lat)]), pixels


def truth_lonlat(site: Site, u: np.ndarray, v: np.ndarray) -> tuple:
    """Where the camera really is looking, in degrees, for scoring against."""
    truth = truth_camera(site)
    placed = truth.view.project(u, v)
    # A target at height h really sits nearer than its ground intercept.
    factor = 1.0 - site.target_height_m / site.height_m
    lat, lon, _ = pymap3d.ned2geodetic(placed.y * factor, placed.x * factor, 0.0,
                                        site.lat, site.lon, 0.0)
    return np.asarray(lon), np.asarray(lat)


def metres_apart(site: Site, lon_a, lat_a, lon_b, lat_b) -> np.ndarray:
    """Ground distance between two sets of degrees, for reporting error."""
    north, east, _ = pymap3d.geodetic2ned(lat_a, lon_a, 0.0, site.lat, site.lon, 0.0)
    n2, e2, _ = pymap3d.geodetic2ned(lat_b, lon_b, 0.0, site.lat, site.lon, 0.0)
    return np.hypot(np.asarray(east) - np.asarray(e2),
                    np.asarray(north) - np.asarray(n2))


def run(site: Site) -> None:
    rng = np.random.default_rng(17)
    control_deg, control_px = survey_off_sentinel(site, rng)

    print(f"\n{'=' * 74}\n{site.name}")
    print(f"  terrain   {site.terrain}")
    print(f"  features  {site.features}")
    print(f"  camera    {site.height_m:.0f} m up, {site.tilt_deg:.0f} deg down, "
          f"f = {site.focal_px:.0f} px, near {site.lat:.2f} {site.lon:.2f}")

    # 1. The marks, as you would overlay them on the frame to check the pick.
    print(f"\n  {'mark':>4} {'column':>8} {'row':>7} {'longitude':>11} "
          f"{'latitude':>10}")
    for i in range(len(control_deg)):
        print(f"  {i:>4} {control_px[i, 0]:8.1f} {control_px[i, 1]:7.1f} "
              f"{control_deg[i, 0]:11.6f} {control_deg[i, 1]:10.6f}")

    # 2. Fit. The residual here is mostly the 10 m imagery, not the clicking.
    fit = fit_homography(control_deg, control_px, WIDTH, HEIGHT,
                         frame=GeodeticPlane(), method="exact")
    print(f"\n  fit over {len(control_deg)} marks: rms {fit.rms_px:.2f} px, "
          f"worst {fit.max_px:.2f} px")

    station = CameraStation(fit.homography,
                            CameraPose(height_m=site.height_m,
                                       x=site.lon, y=site.lat), site.name)
    cols = np.full(site.rows.size, WIDTH / 2.0 + 120.0)
    lon_true, lat_true = truth_lonlat(site, cols, site.rows)

    # 3. Corrections, added one at a time, scored against the truth.
    #
    # Averaged over many independent surveys, and that is not padding. The
    # correction removes a *bias*; the 10 m control error is *noise*. In a single
    # survey the noise can be larger than the bias and mask it entirely -- the
    # "correction" will sometimes look like it made things worse. Over many
    # surveys the noise averages down and the bias does not, which is the whole
    # reason a bias is worth removing at all.
    stages = [("raw", {})]
    if site.target_height_m:
        stages.append(("+ height", {"target_height_m": site.target_height_m}))
    # Curvature is deliberately NOT scored here. The truth this example scores
    # against is a flat plane, so a correction for the earth falling away from
    # that plane can only move the answer off it. Validating that term needs a
    # spherical-earth truth model, which is out of scope; its magnitude is
    # reported separately below instead.

    surveys = 400
    totals = {label: np.zeros(site.rows.size) for label, _ in stages}
    single = {}
    for draw in range(surveys):
        draw_rng = np.random.default_rng(1000 + draw)
        deg, px = survey_off_sentinel(site, draw_rng)
        drawn_fit = fit_homography(deg, px, WIDTH, HEIGHT,
                                   frame=GeodeticPlane(), method="exact")
        drawn = CameraStation(drawn_fit.homography,
                              CameraPose(height_m=site.height_m,
                                         x=site.lon, y=site.lat))
        for label, kwargs in stages:
            placed = drawn.project(cols, site.rows, **kwargs)
            err = metres_apart(site, placed.x, placed.y, lon_true, lat_true)
            totals[label] += err
            if draw == 0:
                single[label] = err
    mean = {label: totals[label] / surveys for label, _ in stages}

    budget = position_error(station, cols, site.rows, site.surface,
                            pixel_sigma_px=1.0, pointing_sigma_deg=0.02,
                            control_rms_px=fit.rms_px,
                            corrected_height=True,
                            corrected_curvature=site.surface.curvature)

    print(f"\n  mean position error over {surveys} independent surveys, m:")
    print(f"\n  {'row':>6} {'range m':>9} {'m/px':>8}", end="")
    for label, _ in stages:
        print(f" {label:>12}", end="")
    print(f" {'budget':>9}")
    gsd = station.view.ground_m_per_px(cols, site.rows)
    for i, row in enumerate(site.rows):
        print(f"  {row:6.0f} {budget.range_m[i]:9.0f} {gsd[i]:8.2f}", end="")
        for label, _ in stages:
            print(f" {mean[label][i]:11.2f} ", end="")
        print(f" {budget.total_m[i]:8.1f}")

    last = stages[-1][0]
    far = -1
    if len(stages) == 1:
        print("\n  Nothing here is correctable, and that is what this site shows:")
        print("  the surface *is* the plane and the targets sit on it, so every")
        print(f"  metre of the {mean['raw'][far]:.1f} m above is survey error and")
        print("  nothing else. The flattest terrain shows the control floor bare.")
    else:
        print(f"\n  at {budget.range_m[far]:.0f} m the corrections take the mean "
              f"error from {mean['raw'][far]:.1f} m to {mean[last][far]:.1f} m.")
        print(f"  in the single survey printed above it went "
              f"{single['raw'][far]:.1f} -> {single[last][far]:.1f} m: one draw "
              f"proves nothing,")
        print("  which is why the table is a mean.")
    print(f"  the control term is {budget.control_m[far]:.1f} m of the "
          f"{budget.total_m[far]:.1f} m budget -- that is the 10 m imagery, and no")
    print("  correction touches it. Only a better survey does.")

    if site.surface.curvature:
        drop = budget.range_m**2 / (2.0 * 6_371_008.8)
        lever = budget.range_m / site.height_m
        print(f"\n  earth curvature would move the far point "
              f"{drop[far] * lever[far]:.1f} m"
              f" ({drop[far] * 100:.0f} cm of drop, levered by {lever[far]:.0f}).")
        print("  It is real and project(curvature=True) removes it, but this")
        print("  example scores against a flat-plane truth and so cannot show that.")

    measured, predicted = mean[last], budget.total_m
    worst = np.max(np.maximum(measured / predicted, predicted / measured))
    print(f"\n  NOTE: the budget column and the measured mean disagree by up to "
          f"{worst:.1f}x here,")
    print("  in both directions. position_error models survey error as a pixel")
    print("  residual times local scale, which is not how control error actually")
    print("  propagates through a fit. When control dominates -- as it does off")
    print("  10 m imagery -- trust the measured spread, not the budget.")


def boresight_demo() -> None:
    """The correction that needs no coordinates at all, on the sea site."""
    site = SITES[-1]
    truth = truth_camera(site)
    lens = Lens(site.focal_px, site.focal_px, WIDTH / 2.0, HEIGHT / 2.0)
    rng = np.random.default_rng(5)

    def clicked(station):
        u, v = station.view.unproject(site.marks_m[:, 0], site.marks_m[:, 1])
        return np.column_stack([u, v]) + rng.normal(0.0, CLICK_PX, (len(u), 2))

    # The breakwater has not moved; the camera has.
    drifted = truth.updated(tilt_deg=site.tilt_deg + 0.25, yaw_deg=0.3)
    alignment = align_pixels(clicked(truth), clicked(drifted), lens=lens)

    cols = np.full(site.rows.size, WIDTH / 2.0 + 120.0)
    stale = truth.view.project(cols, site.rows)
    now = drifted.view.project(cols, site.rows)
    fixed = truth.view.realigned(alignment.transform).project(cols, site.rows)

    print(f"\n{'=' * 74}\n{site.name}: boresighting, no coordinates needed")
    print(f"  drift measured from the same 6 features: "
          f"{alignment.drift_deg:.3f} deg, "
          f"residual {alignment.residual_deg:.4f} deg")
    print(f"\n  {'row':>6} {'stale m':>10} {'realigned m':>13}")
    for i, row in enumerate(site.rows):
        before = np.hypot(stale.x[i] - now.x[i], stale.y[i] - now.y[i])
        after = np.hypot(fixed.x[i] - now.x[i], fixed.y[i] - now.y[i])
        print(f"  {row:6.0f} {before:10.2f} {after:13.2f}")
    print("\n  Unlike the control error above, this one is free to remove: the")
    print("  landmarks need no coordinates, only to have stayed put.")


def main() -> None:
    print(__doc__.split("\n")[0])
    print("Copernicus Sentinel data 2026 -- see examples/images/MANIFEST.yaml")
    for site in SITES:
        run(site)
    boresight_demo()


if __name__ == "__main__":
    main()
