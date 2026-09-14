# Limitations and the error budget

What a plane homography can and cannot tell you, priced in metres.

Every number below was computed by `docs/make_error_tables.py` over the synthetic
setups in `cam2geo.demo`, not written by hand. Re-run it and the tables should
reproduce exactly. If one looks wrong, fix the library or the script — not the
prose.

```
uv run python docs/make_error_tables.py
```

The thesis, which the tables exist to support:

> **Pixel precision is almost never what limits you.** At 269 m on the coastal
> setup one pixel is worth 1.7 m, while half a metre of uncertainty about the
> target's height is worth 4.5 m and a twentieth of a degree of mast sway is
> worth 2.1 m. Buying a sharper camera changes the smallest term in the budget.

![Which term dominates, against range](figures/budget.svg)

---

## 1. What fixes the ground side

A homography is defined up to scale — **including sign**. Nothing in the matrix
says which side of the horizon is ground, so `ground_pixel` does: the sign of the
third homogeneous coordinate there becomes the definition of "ground".

Two consequences, and the second is the one that bites.

**The far side returns NaN, not a number.** Inverting a pixel above the horizon
gives a finite, plausible point *behind the camera*. `cv2.perspectiveTransform`
returns it without complaint — verified in `tests/test_oracles.py`, where for one
sky pixel OpenCV hands back a position 300 m behind the mast. `project` returns
NaN there and counts it in `beyond_horizon`.

**An anchor chosen in the sky inverts everything, silently.** The library cannot
detect it: the inverted answer is entirely self-consistent, with the sky valid and
the ground NaN. Only you know where the ground is. The one case that *is* caught
is an anchor within a pixel of the horizon, where the side is genuinely
undecidable and construction raises. `tests/test_homography.py` pins both
behaviours deliberately.

![The vanishing line and the mirrored point](figures/horizon.svg)

## 2. Conditioning: why there is no "ground sample distance"

A pixel does not cover a fixed patch of ground. It is not even a square.

| | `coastal_mast` | `pole_camera` |
|---|---:|---:|
| variation down the frame | **91×** | 14× |
| variation along one row, centre to edge | 0.34 → 0.40 m/px | 0.033 → 0.045 m/px |
| anisotropy **within one pixel** near the horizon | **18.4 : 1** | 4.4 : 1 |

That is why the method is `ground_m_per_px` and not `ground_sample_distance` —
the latter is an ortho-imagery term implying a uniformity this geometry never
has. What is reported is the **worse axis**, a deliberate worst case; `jacobian`
returns the full 2×2 if you need the shape and not just the bound.

![Conditioning against image row](figures/conditioning.svg)

`max_m_per_px` cuts the tail. It is not optional: without it a detection a few
pixels below the horizon produces a position kilometres away, with no other
signal that anything is wrong. Cut points keep their `ground_m_per_px`, so you
can always see how bad the one you rejected was.

### The Jacobian, and its extra term

With `g = H⁻¹·(u, v, 1)` and `x = g₀/g₂`:

```
∂x/∂u  =  (H⁻¹₀₀ − x·H⁻¹₂₀) / g₂
                   ^^^^^^^^^ the perspective-divide term
```

The naive answer is `H⁻¹₀₀/g₂` — the numerator's derivative alone. The second
piece is the quotient rule applied to the perspective divide, and it is not a
correction but the entire phenomenon: without it the Jacobian is effectively the
affine part, which claims the scale is constant. Measured on the reference
camera, dropping it is wrong by **5 to 6 orders of magnitude** (195,000× at the
bottom row, 5,190× at row 200) and reverses the direction — the naive version
shrinks toward the horizon where the truth diverges.

No library hands you this. `perspectiveTransform` performs the divide but
discards its derivative; `projectPoints`' Jacobian differentiates with respect to
pose and intrinsics while holding the ground point fixed, which is the other
variable. So it is computed here, and checked against
`scipy.differentiate.jacobian` to better than 1e-11 absolute.

## 3. A target above the plane

The largest correctable term, and the reason the default anchor is the bottom of
the box.

For a camera at height `H` and a target whose base sits at height `h` above the
fitted plane, the measured point lands too far out, and the radius from the
camera's nadir is corrected by:

```
r_true = r_measured · (1 − h/H)
```

This is **relief displacement**, standard photogrammetry, exact for any camera
orientation because it depends only on the ray through the camera centre. See
Mikhail, Bethel & McGlone, or Wolf, DeWitt & Wilkinson, below.

`CameraStation.project(target_height_m=)` applies it. What it cannot remove is the
*variability* of `h` — swell that is 0.5 m one minute and 1.5 m the next — which
stays in the budget as `offset_sigma_m`.

![The anchor and relief displacement](figures/anchor.svg)

**A tilt of the surface is the benign case.** Runway crossfall, pitch camber, a
drainage grade: a uniformly sloping surface is still a plane, and
`fit_homography` absorbs it into the matrix at no cost, with no residual left
over. Do **not** correct for it again — that is double-counting, and
`tests/test_fit.py` pins it.

## 4. Earth curvature and refraction

Over open water and essentially nowhere else. The surface falls below the tangent
plane, so the target is further away than the flat model says. The standard
surveying form is:

```
drop = (1 − k) · d² / (2R)
```

with `k` the coefficient of refraction. **If you take `k` from a surveying text,
use 0.14** — that is the value implied by the familiar constants `0.0675 K²`
(metres, km) and `0.0206 F²` (feet, thousands of feet) in Ghilani's *Elementary
Surveying*. The classical Gaussian 0.13 belongs to a different tradition and is
inconsistent with those constants.

**Over water, do not trust any value of `k`.** Hirt et al. (2010) state outright
that the Gaussian 0.13 is unsuited to the lower atmosphere, and report measured
coefficients from −4 to +16 on clear days; for a line of sight a few metres over
water, Kabashi measured `k` from **+1 to +18**. A refraction correction over the
sea is a guess with a plausible-looking number attached. `refraction_k` therefore
defaults to **0.0** — no refraction assumed, which overstates curvature slightly
and is the honest direction to be wrong in.

## 5. Why the station is an input

`CameraStation` takes height and position as explicit inputs and never recovers
them from the matrix.

`cv2.decomposeHomographyMat` exists and is the wrong tool here. On a real set of
65 hand-fitted traffic-camera homographies, the two independent focal-length
estimates from Zhang's single-homography constraints disagreed by up to **95%**,
and **11 of the 65 admitted no real solution at all**. A hand-fitted homography
carries its fitting error into any pose you extract from it. You know your own
mast height; the matrix does not.

The same logic covers orientation: a station derived from a `CameraPose` can be
re-aimed and its view rebuilt, because the pose is the source of truth. A station
wrapping a *fitted* matrix cannot, and `updated()` raises rather than returning a
view that no longer matches its stated pose.

## 6. How the terms combine

Adding independent uncertainties in quadrature is the standard move -- it is the
law of propagation of uncertainty in the GUM (JCGM 100:2008), and it is what
photogrammetric error budgets and positional-accuracy standards do. But it is
only valid for **independent, zero-mean random** terms, and two of the terms here
are neither.

- **A known mean height** (`Surface.offset_m` -- the average swell, the height of
  a container, a crop) and **earth curvature** are *biases*. They are
  deterministic given range and they displace the point the same way every time.
  Adding a bias in quadrature flatters it.
- Worse, height, roughness and curvature all act through the same lever
  (`range / camera height`) and all displace the point *radially outward from the
  nadir*. They are collinear, not independent.

So `position_error` keeps them apart:

```
bias_m   = height_bias_m + curvature_m           # linear: same direction, every time
random_m = sqrt( pixel^2 + height_sigma^2 + roughness^2
                 + pointing^2 + control^2 )      # quadrature: independent
total_m  = bias_m + random_m
```

The right answer to a bias is to **correct it**, not to carry it:
`CameraStation.project(target_height_m=, curvature=)` removes both, and then
`corrected_height=True` / `corrected_curvature=True` tells the budget they are
gone. What remains is `height_sigma_m` -- the part of the height you never knew.

Two caveats this still does not fix. `random_m` treats roughness and height
spread as independent when they are both radial, so the true combined radial
error is somewhere between the quadrature sum and the linear one. And `total_m`
is a one-sigma figure, not a confidence bound -- multiply it if you need one.

## 7. The tables

<!-- generated by docs/make_error_tables.py -- do not hand-edit -->

Assumptions: 1 px of detection error, 0.05 deg of pointing uncertainty,
2 px of control-point residual, and at least 0.5 m of height uncertainty.
The preset biases are set to zero here so the random terms are visible;
a real site with a known mean swell would show a `bias` column.

### Reach

| setup | horizon row | 0.1 m/px | 1 m/px | 10 m/px |
|---|---:|---:|---:|---:|
| `coastal_mast` | 343 | 61 m | 203 m | 647 m |
| `moorland_mast` | 393 | 57 m | 183 m | 583 m |
| `pole_camera` | 176 | 27 m | 92 m | 299 m |


### coastal_mast

_30 m mast over the sea, 8 deg down-tilt_

| image row | range m | m/px | 1 px | height sigma | roughness | pointing | control | random | bias | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1079 | 54 | 0.079 | 0.08 | 0.90 | 0.90 | 0.11 | 0.16 | 1.29 | 0.00 | **1.29** |
| 979 | 63 | 0.106 | 0.11 | 1.05 | 1.05 | 0.14 | 0.21 | 1.51 | 0.00 | **1.51** |
| 878 | 76 | 0.149 | 0.15 | 1.26 | 1.26 | 0.19 | 0.30 | 1.83 | 0.00 | **1.83** |
| 777 | 95 | 0.227 | 0.23 | 1.58 | 1.58 | 0.29 | 0.45 | 2.30 | 0.00 | **2.31** |
| 676 | 124 | 0.386 | 0.39 | 2.07 | 2.07 | 0.48 | 0.77 | 3.10 | 0.01 | **3.10** |
| 575 | 181 | 0.794 | 0.79 | 3.01 | 3.01 | 0.97 | 1.59 | 4.71 | 0.02 | **4.73** |
| 474 | 323 | 2.486 | 2.49 | 5.39 | 5.39 | 3.07 | 4.97 | 9.92 | 0.09 | **10.01** |
| 373 | 1435 | 46.796 | 46.80 | 23.92 | 23.92 | 59.93 | 93.59 | 125.24 | 7.73 | **132.97** |

Dominant term by range:

| range m | dominant term | it is worth | total |
|---:|---|---:|---:|
| 54 | height sigma | 0.90 m | 1.29 m |
| 68 | height sigma | 1.13 m | 1.63 m |
| 90 | height sigma | 1.50 m | 2.19 m |
| 133 | height sigma | 2.21 m | 3.33 m |
| 245 | height sigma | 4.09 m | 6.89 m |
| 1435 | control | 93.59 m | 132.97 m |


### moorland_mast

_12 m mast over moorland, 3 deg down-tilt, long lens_

| image row | range m | m/px | 1 px | height sigma | roughness | pointing | control | random | bias | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1079 | 49 | 0.072 | 0.07 | 2.02 | 4.04 | 0.18 | 0.14 | 4.53 | 0.00 | **4.53** |
| 985 | 56 | 0.096 | 0.10 | 2.35 | 4.69 | 0.24 | 0.19 | 5.26 | 0.00 | **5.26** |
| 891 | 67 | 0.136 | 0.14 | 2.79 | 5.59 | 0.34 | 0.27 | 6.26 | 0.00 | **6.26** |
| 797 | 83 | 0.206 | 0.21 | 3.45 | 6.90 | 0.51 | 0.41 | 7.75 | 0.00 | **7.75** |
| 702 | 108 | 0.352 | 0.35 | 4.52 | 9.04 | 0.87 | 0.70 | 10.18 | 0.00 | **10.18** |
| 608 | 156 | 0.727 | 0.73 | 6.51 | 13.02 | 1.79 | 1.45 | 14.76 | 0.00 | **14.76** |
| 514 | 278 | 2.292 | 2.29 | 11.60 | 23.20 | 5.65 | 4.58 | 27.04 | 0.00 | **27.04** |
| 419 | 1308 | 48.944 | 48.94 | 54.51 | 109.02 | 124.47 | 97.89 | 205.74 | 0.00 | **205.74** |

Dominant term by range:

| range m | dominant term | it is worth | total |
|---:|---|---:|---:|
| 49 | roughness | 4.04 m | 4.53 m |
| 60 | roughness | 5.02 m | 5.62 m |
| 79 | roughness | 6.60 m | 7.41 m |
| 116 | roughness | 9.64 m | 10.86 m |
| 213 | roughness | 17.75 m | 20.34 m |
| 1308 | pointing | 124.47 m | 205.74 m |


### pole_camera

_8 m pole over a yard, 20 deg down-tilt_

| image row | range m | m/px | 1 px | height sigma | roughness | pointing | control | random | bias | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1079 | 7 | 0.011 | 0.01 | 0.45 | 0.03 | 0.01 | 0.02 | 0.45 | 0.00 | **0.45** |
| 952 | 9 | 0.015 | 0.02 | 0.55 | 0.03 | 0.02 | 0.03 | 0.55 | 0.00 | **0.55** |
| 825 | 11 | 0.021 | 0.02 | 0.69 | 0.04 | 0.02 | 0.04 | 0.69 | 0.00 | **0.69** |
| 698 | 14 | 0.033 | 0.03 | 0.90 | 0.05 | 0.03 | 0.07 | 0.91 | 0.00 | **0.91** |
| 571 | 20 | 0.058 | 0.06 | 1.25 | 0.08 | 0.05 | 0.12 | 1.26 | 0.00 | **1.26** |
| 444 | 31 | 0.126 | 0.13 | 1.93 | 0.12 | 0.11 | 0.25 | 1.96 | 0.00 | **1.96** |
| 317 | 61 | 0.453 | 0.45 | 3.83 | 0.23 | 0.42 | 0.91 | 3.99 | 0.00 | **3.99** |
| 189 | 696 | 50.000 | 50.00 | 43.47 | 2.61 | 52.79 | 100.00 | 131.08 | 0.00 | **131.08** |

Dominant term by range:

| range m | dominant term | it is worth | total |
|---:|---|---:|---:|
| 7 | height sigma | 0.45 m | 0.45 m |
| 10 | height sigma | 0.60 m | 0.60 m |
| 14 | height sigma | 0.85 m | 0.86 m |
| 22 | height sigma | 1.35 m | 1.36 m |
| 45 | height sigma | 2.78 m | 2.85 m |
| 696 | control | 100.00 m | 131.08 m |

Read the last row of each table as a warning, not a result: past the conditioning
cutoff every term explodes together and the position means nothing.

## 8. What the model cannot represent at all

- **Long-wavelength relief.** A plane is local. Moorland that looks flat over
  100 m is not flat over 1 km, and `roughness_m` prices the residual but does not
  fix it. If the surface has structure, you need a terrain model, not a
  homography.
- **A non-planar shoreline or a stepped surface.** Quarry benches, container
  stacks, a beach with a berm: flat *per level*, not across them. Fit one
  homography per level.
- **Anything above the plane, beyond the single-height correction.** The
  correction assumes you know `h`. A scene with targets at many unknown heights
  is not solvable from one view without more information.
- **Independent validation.** A homography is a measurement with nothing behind
  it. Cross-camera agreement is a *consistency* check, not an accuracy one: two
  cameras agreeing does not prove either is right, though disagreement does prove
  at least one is wrong.
- **Time.** Nothing here models a surface that moves — tide, a settling mast, a
  seasonal crop. Re-fit, or carry the change as `offset_sigma_m`.

## References

The geometry is textbook and the error terms were derived independently in three
fields. Every entry below was checked against a primary source; where a detail
could not be confirmed it is marked rather than guessed.

**Projective geometry and estimation**

- Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*,
  2nd ed., Cambridge University Press, 2004. doi:10.1017/CBO9780511811685.
  Chapter 2 for plane projective transformations, chapter 4 for the normalised
  DLT and RANSAC, chapter 5 for covariance propagation — the published basis for
  `jacobian` — and chapter 8 for vanishing points and vanishing lines.
  *Chapter 13, "Scene Planes and Homographies", is a different result: the
  homography induced between two views by a scene plane, not the single-view
  image-to-ground map used here.*
- Abdel-Aziz, Y. I. and Karara, H. M. "Direct Linear Transformation from
  Comparator Coordinates into Object Space Coordinates in Close-Range
  Photogrammetry." Originally presented at the Symposium on Close-Range
  Photogrammetry, Urbana, Illinois, 1971; reprinted in *Photogrammetric
  Engineering & Remote Sensing* 81(2):103–107, 2015,
  doi:10.14358/PERS.81.2.103. The origin of the method `fit_homography` rests on.

**Relief displacement and error propagation**

- Mikhail, E. M., Bethel, J. S. and McGlone, J. C. *Introduction to Modern
  Photogrammetry*, Wiley, 2001. ISBN 978-0-471-30924-6.
- Wolf, P. R., DeWitt, B. A. and Wilkinson, B. E. *Elements of Photogrammetry
  with Applications in GIS*, 4th ed., McGraw-Hill, 2014.
  ISBN 978-0-07-176112-3. The canonical treatment of relief displacement,
  `d = r·h/H` — section 3 above.
- Criminisi, A., Reid, I. and Zisserman, A. "Single View Metrology."
  *International Journal of Computer Vision* 40(2):123–148, 2000,
  doi:10.1023/A:1026598000963. Affine 3D measurement from one uncalibrated view
  using a vanishing line and a vanishing point, with first-order error
  propagation giving a confidence interval per measurement.
  *Cited for single-view uncertainty, not for the `1 − h/H` correction: that
  paper is deliberately uncalibrated and does not use a known camera height or a
  nadir point.*

**The maritime case**

- Holland, K. T., Holman, R. A., Lippmann, T. C., Stanley, J. and Plant, N.
  "Practical use of video imagery in nearshore oceanographic field studies."
  *IEEE Journal of Oceanic Engineering* 22(1):81–92, 1997,
  doi:10.1109/48.557542. A shore-based camera, the photogrammetric transform
  onto a horizontal water plane, and how its error behaves — the closest
  published analogue to this library's README.
- Holman, R. A. and Stanley, J. "The history and technical capabilities of
  Argus." *Coastal Engineering* 54(6–7):477–491, 2007,
  doi:10.1016/j.coastaleng.2007.01.003. Three decades of operational
  shore-camera georeferencing.

**The traffic-camera case**

- Dubská, M., Herout, A. and Sochor, J. "Automatic Camera Calibration for
  Traffic Understanding." *BMVC 2014*, doi:10.5244/C.28.42. Calibrating a fixed
  roadside camera from vehicle motion alone; mean speed and distance accuracy
  below 2% in favourable conditions.
- Sochor, J., Juránek, R., Špaňhel, J., Maršík, L., Široký, A., Herout, A. and
  Zemčík, P. "Comprehensive Data Set for Automatic Single Camera Visual Speed
  Measurement." *IEEE Transactions on Intelligent Transportation Systems*
  20(5):1633–1643, 2019, doi:10.1109/TITS.2018.2825609. Roughly 1.4–1.7 km/h
  mean speed error for their calibrated variants, against about 8.6 km/h fully
  automatic.

**Curvature, refraction and sea state**

- Ghilani, C. D. *Elementary Surveying: An Introduction to Geomatics*, 16th ed.,
  Pearson, 2021, chapter 4 §4.4 "Curvature and Refraction". *Paul R. Wolf is the
  founding co-author and appears on earlier editions.*
- Hirt, C., Guillaume, S., Wisbar, A., Bürki, B. and Sternberg, H. "Monitoring
  of the refraction coefficient in the lower atmosphere using a controlled
  setup of simultaneous reciprocal vertical angle measurements." *Journal of
  Geophysical Research* 115, D21102, 2010, doi:10.1029/2010JD014067. States that
  the Gaussian 0.13 is unsuited to the lower atmosphere, and surveys measured
  coefficients over water and ice.
- Yang, X. and Zou, Y. "Precise levelling in crossing river over 5 km using
  total station and GNSS." *Scientific Reports* 11, 7492, 2021,
  doi:10.1038/s41598-021-86929-1. Why only simultaneous reciprocal observation
  reliably cancels refraction over water.
- World Meteorological Organization, *Manual on Codes*, Volume I.1, Part A —
  Alphanumeric Codes, WMO-No. 306, 2019 edition, Code Table 3700 "Sea state".
  Codes 0–9 against wave-height ranges, for the `SURFACES["sea"]` figures.
  *The table applies to well-developed wind waves; it does not state that the
  figures are significant wave heights.* The Douglas sea scale is a separate,
  older scheme (Captain H. P. Douglas, c. 1917–1921) whose swell code is
  qualitative rather than a height range.

**One standard deliberately not quoted.** The README's terrain catalogue
describes runway crossfall and pitch camber qualitatively rather than with
figures. Those belong to ICAO Annex 14 and the relevant sport's governing body
respectively, and are not reproduced here because they were not verified against
the published standards. Consult the standard for your site; whatever number it
gives is a *tilt*, which section 3 explains costs nothing.
