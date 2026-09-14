# Interface

Every public name, what it promises, and how to fill in the numbers it asks for.

Two invariants hold everywhere:

1. **One output per input, always.** Projection never drops a row and never
   raises per pixel.
2. **NaN is the only failure marker.** A pixel that cannot be placed comes back
   as NaN with `valid=False`, never as a plausible wrong number. Check it.

Install:

```
pip install cam2geo                 # numpy only
pip install 'cam2geo[lens]'         # + OpenCV: fitting and undistortion
pip install 'cam2geo[geodetic]'     # + pymap3d: lat/lon planes
```

![The interfaces and what flows between them](figures/pipeline.svg)

---

## PlaneHomography

The map itself, and nothing else. It needs no camera position: the matrix
already absorbs pose, intrinsics and the plane.

```python
from cam2geo import PlaneHomography, MetricPlane

view = PlaneHomography(matrix, width=1920, height=1080,
                       lens=None, frame=MetricPlane(), ground_pixel=None)
```

| field | meaning |
|---|---|
| `matrix` | 3×3 mapping plane coordinates `(x, y, 1)` to pixels. Any scale, either sign. |
| `width`, `height` | frame size in pixels |
| `lens` | the `Lens` the image was undistorted with *before* the matrix was fitted, or `None` |
| `frame` | how plane units become metres — `MetricPlane()` or `GeodeticPlane()` |
| `ground_pixel` | a pixel you know sees the ground. Default: bottom centre. |

Raises on a non-finite, non-3×3 or singular matrix, or a non-positive frame size.

**Methods**

- `project(u, v, max_m_per_px=inf) -> GroundPoints` — pixels to the plane.
- `unproject(x, y) -> (u, v)` — the other direction, distortion re-applied.
- `ground_m_per_px(u, v)` — metres of ground a one-pixel step moves the point,
  along the worse axis. Local, never a constant; see LIMITATIONS §2.
- `jacobian(u, v) -> (N, 2, 2)` — the exact local derivative in metres per pixel,
  for propagating a covariance: `C_ground = J · C_pixel · Jᵀ`.
- `horizon_line -> (a, b, c)` — the vanishing line, normalised so
  `a·u + b·v + c` is a signed distance in pixels, positive on the ground side.
- `footprint(max_m_per_px, columns=64) -> (M, 2)` — the usable ground polygon,
  closed. Empty if nothing qualifies.
- `reference_point`, `ground_sign`, `inverse`, `fingerprint()`
- `translated(dx, dy)` — the same view about a shifted plane origin. Exact.

## GroundPoints

What `project` returns. Arrays are one entry per input pixel, shaped like it.

| field | meaning |
|---|---|
| `x`, `y` | plane coordinates, NaN where `valid` is False |
| `ground_m_per_px` | conditioning at that pixel — kept even for rejected points |
| `valid` | on the plane and within the cutoff |
| `beyond_horizon` | count of pixels on the sky side |
| `outside_lens_model` | count with no undistorted position |
| `ill_conditioned` | count on the plane but over the cutoff |

## CameraPose and CameraStation

A station is the **installation**: a view plus where the camera stands. Only the
corrections need this; plain projection does not.

```python
from cam2geo import CameraPose, CameraStation, Lens

pose = CameraPose(height_m=30.0, x=0.0, y=0.0, tilt_deg=8.0, yaw_deg=0.0)
station = CameraStation.from_pose(pose, Lens(1400, 1400, 960, 540), 1920, 1080)
```

`tilt_deg` is depression below horizontal, `yaw_deg` the bearing of the optical
axis from `+y` toward `+x`, `roll_deg` about the axis. The orientation fields are
`None` for a station built around a *fitted* matrix, because a hand-fitted
homography cannot be decomposed into a trustworthy pose (LIMITATIONS §5).

**Two ways in, and they differ in what can change afterwards.**

| built by | `derived` | `updated(tilt_deg=…)` | `updated(x=…)` |
|---|---|---|---|
| `from_pose(...)` | True | rebuilds the view | rebuilds the view |
| `CameraStation(view, pose)` around a fitted matrix | False | **raises** | updates corrections only |

```python
station.updated(tilt_deg=12.0)      # re-aim; the matrix follows
station.updated(x=5.0, height_m=32.0)
station.translated(dx, dy)          # put a site's cameras on one shared origin
station.range_m(u, v)               # ground distance to each pixel's point
```

**`project` with corrections:**

```python
placed = station.project(u, v, max_m_per_px=1.0,
                         target_height_m=0.0,   # target base above the plane
                         curvature=False,       # open water at km ranges
                         refraction_k=0.0)
```

## Lens

OpenCV's Brown-Conrady model. Coefficient names and order follow OpenCV.

```python
Lens(fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)
```

`undistort(u, v)` returns ideal pixels, NaN where the model has no inverse —
verified by projecting forward again, because OpenCV signals nothing when its
iterative inverse diverges. `is_identity` is True when no distortion is modelled,
and then OpenCV is never imported at all.

## PlaneFrame: MetricPlane and GeodeticPlane

How the homography's output units become metres. `MetricPlane()` when the matrix
maps to metres (pass `metres_per_plane_unit` for feet or similar);
`GeodeticPlane()` when it maps to WGS-84 degrees, x latitude and y longitude.

The library is otherwise unit-agnostic — this is the only place units are known.

## fit_homography

```python
from cam2geo import fit_homography

fit = fit_homography(plane_points, pixel_points, width, height,
                     lens=None, method="ransac", ransac_threshold_px=3.0)
fit.homography, fit.rms_px, fit.max_px, fit.residuals_px, fit.inliers
```

Needs at least four non-collinear control points. `method` is `"exact"` (plain
least squares), `"ransac"` or `"lmeds"`. If a `lens` is given the pixel points
are undistorted first, so the matrix lands in the pixel space everything else
assumes. **A uniform tilt of the surface is absorbed here for free** — do not
correct for crossfall or camber again afterwards.

Four points is the minimum, and it fits them *exactly* — `rms_px` is 0.00
however wrong the survey is. The first honest signal about the fit arrives with
the fifth mark. See [`examples/four_points.py`](../examples/four_points.py).

> Plane coordinates are often degrees: magnitude 50, spread 0.001. The fit solves
> about their centroid because OpenCV's own normalisation does not survive that
> ratio — fitting four marks in raw longitude and latitude costs **0.66 px of
> pure arithmetic** on a fit that is exact by construction, against 5e-5 px
> centred. Nothing to configure; noted because the symptom is a plausible
> residual that is not measuring your survey.

## align_pixels and realigned

```python
from cam2geo import align_pixels

alignment = align_pixels(reference_px, current_px, lens=lens,
                         model="rotation", method="ransac")
alignment.drift_deg, alignment.residual_deg, alignment.rms_px
alignment.motion_px, alignment.inliers, alignment.transform

view = view.realigned(alignment.transform)          # PlaneHomography
station = station.realigned(alignment.transform)    # keeps position, drops orientation
```

Boresighting: measure how far a fixed camera has drifted since its homography was
fitted, and undo it. The landmarks need **no known coordinates** — only to be
things that have not moved, found in both images. Two or more for
`model="rotation"`, four or more for `model="homography"`.

`model` is the parameter that matters. `"rotation"` gives the camera the three
freedoms a mount actually has; `"homography"` gives it eight, which also covers a
camera that *shifted* but lets clicking error into five parameters the camera does
not have — and hides that noise in a smaller residual, so `rms_px` will not warn
you. Prefer the default unless the mast itself moved.

`residual_deg` is what alignment could not explain, and is the number to pass back
as `pointing_sigma_deg`: drift you measured is gone, and only this is left to
budget for. Full treatment in [LIMITATIONS.md §7.1](LIMITATIONS.md), worked
example in
[`examples/boresight_from_landmarks.py`](../examples/boresight_from_landmarks.py).

`station.realigned` keeps the camera's position — the mast has not moved — and
drops `tilt_deg`/`yaw_deg` rather than leaving them stale, so the matrix becomes
the only record of where it points, exactly as for a fitted view.

## anchor_pixel

```python
anchor_pixel(x, y, w, h, angle_deg=0.0, anchor="bottom_centre")
```

`(x, y)` is the box's top-left corner. `bottom_centre` is the default because a
target standing on the plane meets it at its feet; its centre floats, and at
range that costs metres.

---

## Surface and position_error: filling in the numbers

This is the part with no right answer in the library, because every input
describes *your* site. Each preset in `SURFACES` is a starting point to be
replaced, not a constant to be trusted.

```python
from cam2geo import SURFACES, Surface, position_error

surface = SURFACES["sea"].but(offset_m=0.8, offset_sigma_m=0.4)
budget = position_error(station, u, v, surface,
                        pixel_sigma_px=1.0,
                        pointing_sigma_deg=0.05,
                        control_rms_px=fit.rms_px)
```

### The Surface inputs — all heights, in metres

| input | what it is | how to get it |
|---|---|---|
| `offset_m` | the **typical** height of a target's base above the fitted plane | mean swell for the sea state; deck height; crop height; the height of a container. A *bias* — correct it with `project(target_height_m=)` rather than carrying it. |
| `offset_sigma_m` | how much that height varies, one sigma | half the peak-to-trough of the swell; the spread of vehicle types. This is the part you can never remove. |
| `roughness_m` | how far the **surface itself** departs from the plane you fitted, RMS, vertically | for the sea, residual chop; for a moor, the undulation left after fitting; for a runway or floor, the published construction tolerance. Measure it by fitting to control points and looking at their vertical residuals. |
| `curvature` | whether the earth's curvature is worth modelling | `True` over open water at kilometre ranges. `False` everywhere else — at 300 m it is centimetres. |
| `refraction_k` | reduces curvature by `1 − k` | leave at `0.0` unless you have a measured value. Over water, published coefficients range from +1 to +18 against a nominal 0.14. See LIMITATIONS §4. |

Curvature is not a number you supply — the drop is computed from range.

### The measurement inputs

| input | what it is | how to get it |
|---|---|---|
| `pixel_sigma_px` | detection or annotation error, in pixels | 1 px for a hand-drawn box is a fair start; for a detector, the RMS of its box-centre error on your validation set. |
| `pointing_sigma_deg` | how far the camera's true orientation may differ from the one the fit assumed. **An angle, not a distance.** Also called boresight error. | Track a fixed landmark — a post, a building corner — across frames or across weather. If it wanders `p` pixels, that is `degrees(atan(p / focal_px))`. A mast in wind might give 0.02–0.1°; a rigid indoor mount, essentially zero. |
| `control_rms_px` | the fit's own reprojection error | straight from `HomographyFit.rms_px`. |
| `corrected_height` | you already called `project(target_height_m=)` | then `offset_m` stops counting and only `offset_sigma_m` remains |
| `corrected_curvature` | you already called `project(curvature=True)` | likewise |

**"Boresighting"**, if you want to reduce that term rather than price it, means
determining the mounting misalignment and folding it into the calibration. Here
that is not a separate procedure: re-fit the homography from control points and
the current orientation is absorbed into the new matrix. `pointing_sigma_deg`
prices what *drifts* after that fit.

### ErrorBudget — what comes back

Every field is **metres of horizontal position error**, one entry per pixel,
named for its cause. Do not confuse them with the `Surface` inputs of similar
name, which are heights.

| field | bias or random | from |
|---|---|---|
| `pixel_m` | random | `pixel_sigma_px` × conditioning |
| `height_bias_m` | **bias** | `offset_m` — correctable |
| `height_sigma_m` | random | `offset_sigma_m` |
| `roughness_m` | random | `roughness_m` |
| `curvature_m` | **bias** | earth curvature — correctable |
| `pointing_m` | random | `pointing_sigma_deg` |
| `control_m` | random | `control_rms_px` × conditioning |
| `bias_m` | | the biases, added **linearly** |
| `random_m` | | the random terms, added **in quadrature** |
| `total_m` | | `bias_m + random_m` |

Biases and random terms are kept apart deliberately — quadrature is only valid
for independent zero-mean errors, and a known swell height is neither. See
LIMITATIONS §6.

`range_m` comes back too, and is NaN wherever the pixel could not be placed,
which makes every other term NaN with it.

## demo

Three named synthetic setups so every example here runs with no calibration of
your own: `coastal_mast()`, `moorland_mast()`, `pole_camera()`. Each returns a
`Setup` with `.station`, `.view`, `.surface`, `.name` and `.description`. They are
also what the error tables and figures are computed from.
