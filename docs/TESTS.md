# What the tests cover

203 tests, all synthetic. Run them with `uv run pytest`.

The point of this document is to make the *gaps* visible. A green suite here
means the geometry is self-consistent and matches independent implementations of
the same maths — it does not mean the library has met a real camera.

## The fixture everything is built on

`tests/conftest.py` defines one camera whose truth is known exactly: an ideal
pinhole 10 m above a metric plane, looking north, tilted 20° below horizontal,
1920×1080 at 1000 px focal length. That puts the horizon at image row **176.03**,
which is what the NaN and conditioning tests key off.

It works in **metres**, not degrees. In a geodetic frame a one-pixel finite
difference at latitude 10 loses about eight significant figures to cancellation,
which would force loose tolerances everywhere; in metres the same comparisons
hold at `1e-12`.

## By file

| file | tests | what it defends |
|---|---:|---|
| `test_homography.py` | 62 | The projection contract, the horizon rule, conditioning, the Jacobian, footprints, fingerprints |
| `test_demo.py` | 30 | The demo setups, including the one with a real lens |
| `test_station.py` | 27 | Pose-derived vs fitted stations, re-aiming, range, shared origins |
| `test_surface.py` | 25 | The error budget, and the bias/random split |
| `test_lens.py` | 15 | Undistortion, and NaN where the model has no inverse |
| `test_fit.py` | 14 | Fitting from control points, residuals, outliers, surface tilt |
| `test_anchors.py` | 12 | Box anchors, including rotated boxes |
| `test_frames.py` | 11 | Metric and geodetic plane units |
| `test_oracles.py` | 7 | **Our maths against other people's implementations** |

## The guarantees, and the tests that hold them

**One output per input, always.** Empty input gives empty output; a 2-D grid of
pixels comes back as a 2-D grid; three sky pixels give three NaNs, not zero rows.

**NaN is the only failure marker.** `test_the_mirrored_point_is_what_we_are_refusing_to_return`
computes the naive inverse of a sky pixel by hand, asserts it is finite and
behind the camera, and then asserts the library returns NaN for it. The trap is
demonstrated, not just avoided.

**Scale and sign of the matrix do not matter.** Parametrised over ×−1, ×2,
×−0.001 and ×1000, all agreeing to `1e-12`.

**The horizon line and the NaN boundary are the same boundary.** Checked row by
row down the whole frame.

**The ground side is an input, and a bad one is not always detectable.** Two
tests, deliberately: an anchor within a pixel of the horizon *raises*, and an
anchor above the horizon **silently inverts everything** — sky valid, ground NaN.
That second test pins a limitation rather than a feature.

**Conditioning is local, not a constant.** Four tests assert the scale varies
more than tenfold down the frame, varies along a single row, is anisotropic
within one pixel, and that the reported figure is the *worse* axis rather than
the mean.

**Corrections are exact, not approximate.** The relief-displacement factor
`1 − h/H` is checked to `1e-12`, including for a camera that is not at the plane
origin — the correction is radial about the nadir, not about the origin.

**A surface tilt costs nothing.** A synthetic scene with 1.5% crossfall is fitted
and the residuals stay below half a pixel, which is the test that stops anyone
correcting for camber twice.

**Biases and random errors combine differently.** One test asserts the random
terms are the quadrature sum; another asserts the bias sum is strictly larger
than their quadrature sum, which is the whole reason they are kept apart.

## Oracle tests: checked against other implementations

The conditioning code is the one part with no off-the-shelf equivalent to call,
so it answers to other people's code instead.

| ours | oracle | agreement |
|---|---|---|
| `jacobian` | `scipy.differentiate.jacobian` | better than **1e-11 absolute**, at every probe row, for a rolled camera too |
| `project` | `cv2.perspectiveTransform` | full double precision on the ground side |
| `unproject` | `cv2.perspectiveTransform` | `rtol=1e-12` |
| `fit_homography` | `cv2.findHomography` | `rtol=1e-9` after normalising for scale |
| `Lens.undistort` | `cv2.projectPoints` round trip | `atol=1e-6` px |
| `GeodeticPlane.distance_m` | `pymap3d.geodetic2ned` | `rtol=1e-12` |

One of these earns its place twice.
`test_opencv_returns_the_mirrored_point_beyond_the_horizon` shows OpenCV agreeing
with us exactly on the ground side **and** returning a finite wrong point beyond
the horizon with no error — which is the executable justification for why the
implementation keeps its own homogeneous divide instead of calling
`perspectiveTransform`.

A tolerance note found while writing these: for an unrolled camera the Jacobian's
off-diagonals are structural zeros, so a relative comparison reads 0-against-1e-14
as 100% error. The tests compare singular values with `rtol` and the matrix with
an `atol` scaled to its largest entry, and include a rolled camera where those
entries are real numbers.

## Bugs these tests were written for

Two were found by running the worked example rather than by reasoning:

- **An exact fit rejected every point.** `cv2.findHomography` with `method=0`
  returns an all-zero mask, which was read as "no inliers", and the RMS then
  crashed on an empty array. Pinned by `test_an_exact_fit_keeps_every_point`.
- **A pose could be built in degrees.** `homography_from_pose` composes the
  camera's position with its height in one rotation, so both must be metres;
  pairing it with a `GeodeticPlane` produced a silently wrong camera. Now
  refused, and pinned by `test_a_geodetic_frame_is_refused`.

## What is **not** covered

Read this part before trusting the suite.

- **No real camera, ever.** Every matrix in the tests is synthesised from a known
  pinhole. Nothing here has met a lens with manufacturing scatter, a rolling
  shutter, or a mount that moves in wind.
- **No hand-fitted homography.** Real control-point surveys have correlated
  errors, points clustered in the near field, and marks whose charted positions
  are themselves wrong. The fit tests use well-spread points with clean noise.
- **No non-planar ground.** The one surface departure exercised is a uniform
  tilt. Undulation, berms, stepped surfaces and terrain relief appear only as
  numbers in the error budget, never as geometry under test.
- **No real distortion model.** `wide_quay` uses a single `k1` term. Tangential
  distortion, `k2`/`k3`, and fisheye projections are implemented but untested
  against a real calibration.
- **The Surface presets are not validated.** They are starting points with cited
  provenance, not measurements. No test asserts that `SURFACES["sea"]` describes
  any actual sea.
- **No accuracy claim.** Every test here is a consistency or agreement check.
  Nothing in this repository establishes that a position produced by this library
  is *correct* against an independent ground truth — because a homography is a
  measurement with nothing behind it. See LIMITATIONS.md §8.
- **Numerics near the horizon.** Tests assert behaviour is *unbounded* there
  (`inf`, or NaN), not that any particular value is right. There is no right
  value.
