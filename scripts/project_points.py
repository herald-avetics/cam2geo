"""Project the marked points with cam2geo and draw the coordinates on the image.

    uv run --all-extras --group images python scripts/project_points.py

Takes each labelled image in examples/images, fits a homography to its marks
with cam2geo, projects those marks back to longitude and latitude through the
fitted matrix, and writes an annotated PNG to output/.

The marks are the only input. Their coordinates in the points file came from the
GeoTIFF's own georeferencing, so this is a closed loop: the fit is given the same
numbers it is then asked to reproduce, and at exactly four marks a homography has
eight degrees of freedom to reproduce eight, so its residual is zero by
construction and means nothing at all.

What does mean something is the grid check. The fitted matrix is asked for a
coordinate at every node of a grid across the frame, none of which it was fitted
on, and compared against the affine transform stored in the file. That number is
the honest one: it is how far a homography fitted to four corners drifts from the
map in between them.

There is no terrain correction here, on purpose -- see the commented call in
placed_points. Sentinel-2 L2A is orthorectified.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402
from rasterio.warp import transform  # noqa: E402

from cam2geo import (  # noqa: E402
    CameraPose,
    CameraStation,
    GeodeticPlane,
    GroundPoints,
    fit_homography,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGES = ROOT / "examples" / "images"
OUTPUT = ROOT / "output"
SUFFIX = ".points.yaml"

#: Sentinel-2's orbit. Only the corrections would use it, and they are off.
ORBIT_HEIGHT_M = 786_000.0
#: Nodes per axis for the check against the file's own affine transform.
GRID = 16

MARK = "#ffd166"


def labelled(image: pathlib.Path) -> dict:
    path = image.with_suffix("").with_suffix(SUFFIX)
    if not path.exists():
        raise SystemExit(path.name + " is missing: run scripts/annotate.py")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def fitted(document: dict, width: int, height: int) -> CameraStation:
    """A station whose view is fitted to the marks, in latitude and longitude."""
    points = document["points"]
    # GeodeticPlane is x latitude, y longitude. The points file stores lon, lat.
    latlon = np.array([point["lonlat"][::-1] for point in points], dtype=float)
    pixels = np.array([point["px"] for point in points], dtype=float)
    fit = fit_homography(latlon, pixels, width, height, frame=GeodeticPlane(),
                         method="exact")
    pose = CameraPose(height_m=ORBIT_HEIGHT_M, x=float(latlon[:, 0].mean()),
                      y=float(latlon[:, 1].mean()))
    return CameraStation(fit.homography, pose, name=document["image"]["file"])


def placed_points(station: CameraStation, pixels: np.ndarray) -> GroundPoints:
    placed = station.project(pixels[:, 0], pixels[:, 1])

    # The terrain correction belongs here and is deliberately not applied.
    # Sentinel-2 L2A is orthorectified: every pixel has already been moved onto
    # a DEM, so the relief displacement this would undo was taken out once
    # already and applying it again would introduce the error it removes. For a
    # camera looking at ground it has not been draped over, uncomment this and
    # supply target_height_m from the points file's estimated block.
    #
    # placed = station.project(pixels[:, 0], pixels[:, 1],
    #                          target_height_m=height_above_the_plane_m,
    #                          curvature=True)

    return placed


def grid_error_m(station: CameraStation, source) -> tuple[float, float]:
    """How far the fitted view drifts from the file's affine, in metres."""
    columns, rows = np.meshgrid(np.linspace(0.5, source.width - 0.5, GRID),
                                np.linspace(0.5, source.height - 0.5, GRID))
    placed = station.project(columns.ravel(), rows.ravel())
    east, north = source.xy(rows.ravel(), columns.ravel())
    lon, lat = transform(source.crs, "EPSG:4326", east, north)
    metres = GeodeticPlane().distance_m(placed.x, placed.y,
                                        np.array(lat), np.array(lon))
    return float(np.sqrt(np.mean(metres**2))), float(metres.max())


def coordinate(lon: float, lat: float) -> str:
    return (format(abs(lat), ".6f") + ("N  " if lat >= 0 else "S  ")
            + format(abs(lon), ".6f") + ("E" if lon >= 0 else "W"))


def draw(image: pathlib.Path, document: dict, placed: GroundPoints,
         caption: str) -> pathlib.Path:
    with rasterio.open(image) as source:
        picture = np.transpose(source.read([1, 2, 3]), (1, 2, 0))
        width, height = source.width, source.height

    fig, ax = plt.subplots(figsize=(9.0, 9.4))
    ax.imshow(picture)
    ax.set_axis_off()
    for i, point in enumerate(document["points"]):
        column, row = point["px"]
        ax.plot(column, row, marker="o", ms=11, mfc="none", mec=MARK, mew=1.8)
        ax.plot(column, row, marker="+", ms=15, color=MARK, mew=1.0)
        # Away from the centre, so marks near the corners do not stack their
        # labels over the middle of the scene or over each other.
        right = column >= width / 2.0
        below = row >= height / 2.0
        ax.annotate(
            point["label"] + "\n" + coordinate(placed.y[i], placed.x[i]),
            (column, row), textcoords="offset points",
            xytext=(34 if right else -34, -34 if below else 34), fontsize=8,
            ha="left" if right else "right", va="top" if below else "bottom",
            color="#101418", annotation_clip=False,
            bbox={"boxstyle": "round,pad=0.35", "fc": MARK, "ec": "none",
                  "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": MARK, "lw": 1.0,
                        "shrinkA": 2.0, "shrinkB": 7.0},
        )
    ax.set_title(caption, fontsize=9, loc="left", family="monospace")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / (image.stem + ".points.png")
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def run(image: pathlib.Path) -> None:
    document = labelled(image)
    points = document.get("points") or []
    if len(points) < 4:
        print(image.stem + ": " + str(len(points)) + " marks, and a homography "
              "needs four. Mark more in " + image.stem + SUFFIX)
        return

    with rasterio.open(image) as source:
        station = fitted(document, source.width, source.height)
        rms_m, worst_m = grid_error_m(station, source)

    pixels = np.array([point["px"] for point in points], dtype=float)
    placed = placed_points(station, pixels)

    print("\n" + document["measured"]["site"] + "  (" + document["image"]["scene"] + ")")
    print(f"  {'mark':<18} {'longitude':>11} {'latitude':>10} {'off by':>8}")
    for point, lat, lon in zip(points, placed.x, placed.y, strict=True):
        off = GeodeticPlane().distance_m(np.array([lat]), np.array([lon]),
                                         np.array([point["lonlat"][1]]),
                                         np.array([point["lonlat"][0]]))[0]
        print(f"  {point['label']:<18} {lon:11.6f} {lat:10.6f} {off:7.2f} m")
    print(f"  against the file's own affine, over {GRID * GRID} pixels it never "
          f"saw: {rms_m:.2f} m rms, {worst_m:.2f} m worst")

    caption = (document["measured"]["site"] + "  --  " + document["image"]["scene"]
               + "\n" + str(len(points)) + " marks fitted, "
               + format(worst_m, ".1f") + " m worst against the georeferencing"
               + "\northorectified, so no terrain correction applied")
    print("  wrote " + str(draw(image, document, placed, caption).relative_to(ROOT)))


def main() -> None:
    images = sorted(IMAGES.glob("*.tif"))
    if not images:
        raise SystemExit("no imagery in examples/images: run "
                         "scripts/fetch_images.py first")
    for image in images:
        run(image)


if __name__ == "__main__":
    main()
