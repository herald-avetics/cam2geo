"""Points marked on the example imagery: the file format, and everything derivable.

    uv run --group images python scripts/annotate.py

scripts/fetch_images.py writes one <image>.points.yaml per image as it downloads,
empty of points. You add labels and pixels; this fills in the rest and rewrites the
file, every run. Comments are regenerated too, so nothing you write in a comment
survives -- put prose in a point's `note`, which is data.

Three kinds of number end up in there, kept in separate blocks because confusing
them is how error budgets go wrong:

    measured     read straight out of the product. The georeferencing is exact
                 arithmetic; the acquisition angles are what the granule metadata
                 reports. No assumption in any of it.
    inferred     arithmetic on the measured angles. True given the geometry, but
                 they need a height you supply before they mean anything.
    estimated    defaults chosen to be defensible, not measurements. pick_sigma_m
                 is half a Sentinel-2 pixel because that is roughly how well a
                 person places a hard edge at 10 m. Replace any you actually know.

And a fourth kind, listed as `unavailable`, which is the honest part: a satellite
ephemeris, the tide at the moment of acquisition, the height of whatever stands at
your point, and the camera-frame pixel that would make each of these a control
point rather than half of one.

Elevation is sampled from Copernicus DEM GLO-30. It is a *surface* model, so over
a yard or a town it includes what is standing there rather than the ground under
it, and at 30 m posting it is smoothed across three of the image's pixels: good
for a plane's height, useless for an edge. Over water it reports a nominal flat
surface, not the level on the day.

Labels going stale. The check is on scene, crs, bbox and size -- what actually
determines which ground a pixel is -- and not on the checksum, so re-fetching the
same scene keeps your labels while a different scene refuses them.

Licences. The elevations are adapted Copernicus WorldDEM-30 and Art. 6(b) of its
licence fixes the notice, written into each file. Redistributing these files also
pulls in the Art. 6(c) liability sentence, recorded in MANIFEST.yaml.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import pathlib

import rasterio
import yaml
from rasterio.warp import transform

IMAGES = pathlib.Path(__file__).resolve().parents[1] / "examples" / "images"
SUFFIX = ".points.yaml"

DEM_URL = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
DEM_SOURCE = "Copernicus DEM GLO-30"
DEM_ATTRIBUTION = ("produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 "
                   "and © Airbus Defence and Space GmbH 2014-2018 provided under "
                   "COPERNICUS by the European Union and ESA; all rights reserved")

PICK_SIGMA_M = 5.0
GDAL = {"AWS_NO_SIGN_REQUEST": "YES", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}

MEASURED_TAGS = ("site", "terrain", "acquired", "platform", "relative_orbit",
                 "orthorectified", "resolution_m", "view_zenith_deg",
                 "view_azimuth_deg", "sun_zenith_deg", "sun_azimuth_deg")

UNAVAILABLE = [
    "satellite position: the ephemeris is a separate POD product, not in the COG "
    "bucket, and an orthorectified image needs none of it",
    "water level at acquisition: alt_m over water is the DEM's nominal flat "
    "surface, not the tide on the day",
    "height of what stands at the point: you supply it, and the inferred block "
    "turns it into metres of lean and shadow",
    "camera-frame pixel: the other half of a control point, matched by label",
    "scene geolocation bias: real, shared by every point here, so it shifts a fit "
    "rather than showing up in its residuals; the figure is not verified",
]

HEADER = """\
# Points marked on {image}.
#
# You write label and px. Everything else scripts/annotate.py writes on every
# run, comments included -- hand-edit a derived field and it goes. Prose belongs
# in a point's note.
#
#   points:
#     - label: "breakwater west end"
#       px: [212.5, 331.0]
#
# px is [column, row] from the top left, and [0.5, 0.5] is the first pixel's
# centre. One decimal place is already a metre at 10 m per pixel; two is
# pretending. A homography needs four points, and at exactly four its residual is
# zero whatever you clicked, so mark five or six.

"""

COMMENTS = {
    "measured": "# Read out of the product. No assumption in any of it.\n",
    "inferred": ("# Arithmetic on the angles above, per metre of height above the\n"
                 "# ground. A thing h metres tall leans relief_m_per_m * h toward\n"
                 "# relief_bearing_deg and throws shadow_m_per_m * h toward\n"
                 "# shadow_bearing_deg. Both are why you click the base, not the top.\n"),
    "estimated": ("# Defaults, not measurements. Replace any you actually know; a\n"
                  "# point's own sigma_m overrides pick_sigma_m for that point.\n"),
    "unavailable": "# Not in anything fetched here, at any price.\n",
    "elevation": "# Sampled per point into alt_m below. A surface model, 30 m posting.\n",
    "points": "# lonlat, en_m and alt_m are derived from px. Do not hand-edit them.\n",
}


def _scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return repr(int(value))
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _mapping(name: str, data: dict) -> str:
    out = COMMENTS.get(name, "") + name + ":\n"
    for key, value in data.items():
        out += "  " + key + ": " + _scalar(value) + "\n"
    return out + "\n"


def _sequence(name: str, items: list[str]) -> str:
    out = COMMENTS.get(name, "") + name + ":\n"
    for item in items:
        out += "  - " + _scalar(item) + "\n"
    return out + "\n"


def _points(points: list[dict]) -> str:
    out = COMMENTS["points"] + "points:"
    if not points:
        return out + " []\n"
    out += "\n"
    for point in points:
        for i, (key, value) in enumerate(point.items()):
            out += ("  - " if i == 0 else "    ") + key + ": " + _scalar(value) + "\n"
    return out


def dem_tile(lon: float, lat: float) -> str:
    return DEM_URL.format(ns="N" if lat >= 0 else "S", lat=abs(int(lat // 1)),
                          ew="E" if lon >= 0 else "W", lon=abs(int(lon // 1)))


def elevations(lonlat: list[list[float]]) -> list[float | None]:
    """Ground height above the EGM2008 geoid, None where the DEM has nothing."""
    heights: list[float | None] = [None] * len(lonlat)
    tiles: dict[str, list[int]] = {}
    for i, (lon, lat) in enumerate(lonlat):
        tiles.setdefault(dem_tile(lon, lat), []).append(i)
    with rasterio.Env(**GDAL):
        for url, indices in tiles.items():
            with rasterio.open(url) as dem:
                samples = dem.sample([lonlat[i] for i in indices])
                for i, value in zip(indices, samples, strict=True):
                    height = float(value[0])
                    heights[i] = None if height == dem.nodata else round(height, 2)
    return heights


def inferred(measured: dict) -> dict:
    return {
        "relief_m_per_m": round(math.tan(math.radians(measured["view_zenith_deg"])), 4),
        "relief_bearing_deg": round((measured["view_azimuth_deg"] + 180.0) % 360.0, 3),
        "shadow_m_per_m": round(math.tan(math.radians(measured["sun_zenith_deg"])), 4),
        "shadow_bearing_deg": round((measured["sun_azimuth_deg"] + 180.0) % 360.0, 3),
    }


def read_image(image: pathlib.Path) -> tuple[dict, dict]:
    """What the fetch tagged onto the image, and the frame it was written in."""
    with rasterio.open(image) as source:
        tags = source.tags()
        frame = {
            "scene": tags.get("scene"),
            "crs": str(source.crs),
            "bbox": [round(v, 1) for v in source.bounds],
            "size_px": [source.width, source.height],
        }
    if "view_zenith_deg" not in tags:
        raise SystemExit(image.name + " carries no acquisition tags: re-run "
                         "scripts/fetch_images.py")
    measured = {}
    for key in MEASURED_TAGS:
        if key not in tags:
            continue
        value = tags[key]
        if key.endswith("_deg") or key.endswith("_m"):
            measured[key] = float(value)
        elif value in ("True", "False"):
            measured[key] = value == "True"
        else:
            measured[key] = value
    return frame, measured


def locate(image: pathlib.Path, points: list[dict]) -> None:
    """World coordinates for every point, in the image's own frame and in WGS84."""
    with rasterio.open(image) as source:
        for point in points:
            column, row = point["px"]
            if not (0 <= column <= source.width and 0 <= row <= source.height):
                raise SystemExit(str(point["label"]) + " at " + str(point["px"])
                                 + " is outside " + image.name)
            east, north = source.xy(row, column)
            point["en_m"] = [round(east, 2), round(north, 2)]
        crs = source.crs
    longitude, latitude = transform(crs, "EPSG:4326",
                                    [p["en_m"][0] for p in points],
                                    [p["en_m"][1] for p in points])
    for point, lon, lat in zip(points, longitude, latitude, strict=True):
        point["lonlat"] = [round(lon, 6), round(lat, 6)]


def ensure(image: pathlib.Path) -> int:
    """Create the points file if it is missing, then fill in everything derived."""
    path = image.with_suffix("").with_suffix(SUFFIX)
    document = {}
    if path.exists():
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    frame, measured = read_image(image)
    was = document.get("image") or {}
    stale = [key for key in frame if key in was and was[key] != frame[key]]
    if stale:
        raise SystemExit(path.name + " was labelled against different ground ("
                         + ", ".join(sorted(stale)) + " changed). Its pixels no "
                         "longer mean what they meant; re-label, or restore the "
                         "image it was marked on.")

    points = [dict(point) for point in document.get("points") or []]
    labels = [point["label"] for point in points]
    if len(set(labels)) != len(labels):
        raise SystemExit(path.name + " repeats a label, and the label is what "
                         "joins these points to a camera frame")

    if points:
        locate(image, points)
        heights = elevations([point["lonlat"] for point in points])
        for point, height in zip(points, heights, strict=True):
            point["alt_m"] = height
    ordered = ["label", "px", "sigma_m", "lonlat", "en_m", "alt_m", "note"]

    path.write_text(
        HEADER.format(image=image.name)
        + _mapping("image", {"file": image.name, **frame,
                             "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                             "written": dt.date.today().isoformat()})
        + _mapping("measured", measured)
        + _mapping("inferred", inferred(measured))
        + _mapping("estimated", {"pick_sigma_m": PICK_SIGMA_M,
                                 "target_height_m": None,
                                 "geolocation_bias_m": None})
        + _sequence("unavailable", UNAVAILABLE)
        + _mapping("elevation", {"source": DEM_SOURCE,
                                 "vertical_datum": "EGM2008 geoid",
                                 "attribution": DEM_ATTRIBUTION})
        + _points([{key: point[key] for key in ordered if key in point}
                   for point in points]),
        encoding="utf-8")
    return len(points)


def main() -> None:
    images = sorted(IMAGES.glob("*.tif"))
    if not images:
        raise SystemExit("no imagery in examples/images: run "
                         "scripts/fetch_images.py first")
    for image in images:
        count = ensure(image)
        print(image.stem + SUFFIX + ": " + str(count) + " points"
              + ("" if count >= 5 else "  (mark five or six -- at four the fit is "
                                       "exact and its residual tells you nothing)"))


if __name__ == "__main__":
    main()
