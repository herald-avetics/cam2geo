"""Boresighting: the camera has been knocked, and you cannot re-survey it.

A gale moves the mast. A contractor leans a ladder on the bracket. The housing
warms through the afternoon. The homography you fitted in spring still describes
where the camera *was* pointing, and every position it produces is now
systematically wrong -- silently, because nothing about a stale matrix looks
wrong from the inside.

The fix needs no survey team and no known coordinates. Any features that have
not moved will do -- a mooring, the corner of a sea wall, a chimney on the far
shore. Find them in the reference image and in today's image, and the transform
between the two *is* the camera's drift. Compose it onto the old homography and
the error is gone.

    uv run python examples/boresight_from_landmarks.py

Needs the lens extra: pip install 'cam2geo[lens]'
"""

from __future__ import annotations

import numpy as np

from cam2geo import (
    CameraPose,
    CameraStation,
    Lens,
    MetricPlane,
    Surface,
    align_pixels,
    position_error,
)

WIDTH, HEIGHT = 1920, 1080
LENS = Lens(1400.0, 1400.0, WIDTH / 2.0, HEIGHT / 2.0)

#: Things on the shore that do not move, in metres east and north of the mast.
#: Their coordinates are needed only to *simulate* the two images; the real
#: procedure never knows them, which is the whole point.
LANDMARKS_M = np.array([[-180.0, 240.0], [160.0, 260.0], [-90.0, 520.0],
                        [110.0, 560.0], [-20.0, 380.0], [230.0, 900.0]])

#: Where detections land, from the near field out toward the horizon at row 343.
DETECTIONS = np.array([[1100.0, 950.0], [1050.0, 620.0], [1010.0, 450.0],
                       [1000.0, 390.0]])

#: Half a pixel, which is about as well as anyone finds the same feature twice.
CLICK_SIGMA_PX = 0.5


def installed(tilt_deg: float = 8.0, yaw_deg: float = 0.0,
              roll_deg: float = 0.0) -> CameraStation:
    """The mast on the headland, 30 m above the water, aimed out to sea."""
    pose = CameraPose(height_m=30.0, x=0.0, y=0.0, tilt_deg=tilt_deg,
                      yaw_deg=yaw_deg, roll_deg=roll_deg)
    return CameraStation.from_pose(pose, LENS, WIDTH, HEIGHT, MetricPlane(),
                                   "headland")


def seen(station: CameraStation, points: np.ndarray, seed: int = 0) -> np.ndarray:
    """Pixel positions of fixed ground features, i.e. what you click in an image.

    With clicking error, because the residual is only meaningful if the input is
    as imprecise as a real operator picking the same corner twice.
    """
    u, v = station.view.unproject(points[:, 0], points[:, 1])
    noise = np.random.default_rng(seed).normal(0.0, CLICK_SIGMA_PX, (len(points), 2))
    return np.column_stack([u, v]) + noise


def offsets_m(view, truth: CameraStation) -> np.ndarray:
    """How far this view puts the detections from where they actually are."""
    placed = view.project(DETECTIONS[:, 0], DETECTIONS[:, 1])
    actual = truth.view.project(DETECTIONS[:, 0], DETECTIONS[:, 1])
    return np.hypot(placed.x - actual.x, placed.y - actual.y)


def main() -> None:
    # -- In spring -------------------------------------------------------
    # The camera is surveyed and the landmarks are photographed. Those pixels
    # are the reference: store them beside the homography, they cost nothing.
    spring = installed()
    reference_px = seen(spring, LANDMARKS_M, seed=1)

    # -- After the gale --------------------------------------------------
    # The bracket has shifted: a third of a degree of yaw, a fifth of tilt, and
    # a little roll. Nobody saw it happen.
    autumn = installed(tilt_deg=8.2, yaw_deg=0.35, roll_deg=0.4)
    current_px = seen(autumn, LANDMARKS_M, seed=2)

    stale = offsets_m(spring.view, autumn)
    print("Using the spring homography on autumn's images:")
    for (u, v), off in zip(DETECTIONS, stale, strict=True):
        print(f"  detection at ({u:6.0f},{v:5.0f})   out by {off:7.1f} m")

    # -- Boresight -------------------------------------------------------
    # Six landmarks, no coordinates, no survey. Just the same features found
    # twice. RANSAC because a "fixed" landmark occasionally is not.
    alignment = align_pixels(reference_px, current_px, lens=LENS)

    print(f"\nthe camera has turned {alignment.drift_deg:.3f} deg"
          f"  (landmarks moved {alignment.motion_px:.1f} px)")
    print(f"alignment residual   {alignment.rms_px:.4f} px"
          f"  = {alignment.residual_deg:.5f} deg unexplained")
    print(f"landmarks kept       {int(alignment.inliers.sum())} of "
          f"{len(alignment.inliers)}")

    corrected = spring.realigned(alignment.transform)
    fixed = offsets_m(corrected.view, autumn)
    print("\nafter realigning:")
    for (u, v), before, after in zip(DETECTIONS, stale, fixed, strict=True):
        print(f"  detection at ({u:6.0f},{v:5.0f})   {before:7.1f} m ->"
              f" {after:8.4f} m")

    # -- What it is worth now --------------------------------------------
    # The payoff is not only the metres removed. Drift you cannot measure has to
    # be *budgeted* for, as pointing_sigma_deg; drift you can measure is removed
    # and only the alignment residual is left to budget.
    sea = Surface("sea", offset_sigma_m=0.5, roughness_m=0.3, curvature=True)
    u, v = DETECTIONS[:, 0], DETECTIONS[:, 1]
    print(f"\n{'detection':>12} {'range m':>9} {'unmeasured':>11} {'boresighted':>12}")
    guessed = position_error(autumn, u, v, sea, pointing_sigma_deg=0.5,
                             corrected_curvature=True)
    measured = position_error(corrected, u, v, sea,
                              pointing_sigma_deg=alignment.residual_deg,
                              corrected_curvature=True)
    for i in range(len(DETECTIONS)):
        print(f"{i:>12} {guessed.range_m[i]:9.0f} {guessed.total_m[i]:10.1f} m"
              f" {measured.total_m[i]:11.1f} m")

    print("\nThe budget improves because a measured error is no longer an")
    print("uncertainty. That is the general move, and it is the only one there")
    print("is: measuring converts a random term into a bias, and a bias can be")
    print("subtracted. See docs/LIMITATIONS.md, 'Negating an error'.")

    # -- Keeping it aligned ----------------------------------------------
    # In service this runs on a schedule rather than after a storm: re-measure
    # daily, realign when the drift is worth more than the noise.
    print("\nas a monitoring rule:")
    for label, station in [("calm day", installed(tilt_deg=8.01)),
                           ("after the gale", autumn)]:
        check = align_pixels(reference_px, seen(station, LANDMARKS_M, seed=3),
                             lens=LENS)
        verdict = "realign" if check.drift_deg > 0.05 else "leave it"
        print(f"  {label:>15}: drift {check.drift_deg:6.3f} deg -> {verdict}")


if __name__ == "__main__":
    main()
