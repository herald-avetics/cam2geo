# Examples

Every one is runnable and self-contained. None needs an image, a camera or a
survey — where real data would be needed, a synthetic camera stands in for it and
says so.

```
uv run python examples/four_points.py
```

| example | what it answers | needs |
|---|---|---|
| [`four_points.py`](four_points.py) | I have four points whose coordinates I know. How do I get world coordinates for a pixel? | `lens`, `geodetic` |
| [`from_gps_control_points.py`](from_gps_control_points.py) | The same, done properly: box anchors, an error budget, a footprint, and a detection the geometry refuses. | `lens`, `geodetic` |
| [`boresight_from_landmarks.py`](boresight_from_landmarks.py) | The camera has been knocked and I cannot re-survey it. How do I measure the drift and remove it? | `lens` |

```
pip install 'cam2geo[lens,geodetic]'
```

## Where to start

**`four_points.py`** is the minimum: four `(longitude, latitude)` ↔ `(column, row)`
pairs, one call to `fit_homography`, and every pixel below the horizon becomes a
position. It also makes the point that four marks fit *exactly* — an rms of
0.00 px however wrong the survey is — so the residual only starts telling you
something at the fifth.

**`from_gps_control_points.py`** is the same workflow as you would actually run
it: six marks, boxes rather than bare pixels, a conditioning cutoff, the error
budget, and the usable footprint as a ring you can plot on a chart. One of its
four detections is beyond the horizon and comes back reported rather than
dropped.

**`boresight_from_landmarks.py`** is the maintenance half, and the one most
pipelines are missing. A fitted homography records where the camera pointed *on
the day it was fitted*; a gale, a ladder or a warm afternoon moves it, and every
position afterwards is silently and systematically wrong. The example measures
92 m of error at 820 m range from 0.6° of drift, then removes it using landmarks
whose coordinates are never known — only that they have not moved.

## Using a real image

The examples above synthesise their pixels so they run anywhere with no
downloads. To point them at a real camera you need two things, and they are
separable:

1. **The image.** Intrinsics are optional — `fit_homography` needs none, and a
   `Lens` only matters if the lens visibly barrels. Pixel coordinates come from
   clicking the frame.
2. **Ground truth for the marks.** Openly-licensed satellite or aerial imagery is
   the usual source: find the same harbour-wall corner or slipway edge in both,
   and read its coordinates off the georeferenced one.

`images/MANIFEST.yaml` records the provenance of anything placed in `images/` —
source URL, licence, attribution and a checksum — and the images themselves are
**not committed**, for licence reasons. Add an entry before adding a file, and
keep the two in step.
