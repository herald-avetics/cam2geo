"""Fetch the Sentinel-2 windows the terrain examples read their control points off.

    uv run --group images python scripts/fetch_images.py

Each site in examples/terrain_sites.py names a real place and a terrain preset.
This finds the least cloudy Sentinel-2 L2A scene over it, reads an 8 km window
out of that scene's true-colour COG by HTTP range request, writes it to
examples/images/ as a georeferenced GeoTIFF, and rewrites MANIFEST.yaml from
what is then on disk. Delete a file, re-run, and its entry goes with it.

Provenance, which is the whole point of the manifest:

    file          name under examples/images/
    description   what it shows, and why it is useful here
    source_url    the COG the pixels were read from
    page_url      the human-readable page that states the terms
    licence       SPDX identifier where one exists, else the terms' own name
    attribution   the credit line the licence requires, verbatim
    redistribute  true only if the licence permits us to ship the file itself
    retrieved     ISO date the URL was last read
    sha256        checksum of the file as written here, not as served -- these
                  are windows out of a 343 MB scene rather than whole downloads,
                  so scene, bbox and size_px are what make one reproducible
    crs           coordinate reference system the window is written in
    resolution_m  ground sample distance, in metres

Acquisition geometry, recorded per image and also written into the GeoTIFF's own
tags so it travels with the file:

    view_zenith_deg    angle off vertical at which the sensor saw this ground,
                       averaged over the three bands the true-colour image is
                       built from (B04, B03, B02; they differ by under 0.2 deg)
    view_azimuth_deg   compass bearing of the satellite as seen from the ground
    sun_zenith_deg     angle off vertical of the sun at acquisition
    sun_azimuth_deg    compass bearing of the sun

There is no satellite position here and no perspective to undo. L2A is
orthorectified: the product has already been projected onto a DEM, so the image
is a map with one affine transform and no projective term. That is exactly why
it works as ground truth, and why no homography can be recovered from it.

The viewing angles still matter for one thing -- relief displacement. The ortho
step puts the DEM surface in the right place, so anything standing *above* that
surface leans away from the satellite by height * tan(view_zenith). At the three
kilometres and change of view zenith here that is 0.05 m per metre of height:
nothing for a container stack, a metre for a gantry crane. The sun angles matter
for the opposite reason: a shadow is height * tan(sun_zenith) long, cast away
from the sun's bearing, and at 10 m per pixel clicking a shadow edge instead of
the thing casting it is a real and easy mistake.

Ground elevation is not in these files. The GeoTIFF carries no Z, no elevation
band, and the DEM the ortho step used is not shipped with the product. Copernicus
DEM GLO-30 is the matching open source for it -- 30 m posting, heights above the
EGM2008 geoid, so above mean sea level in the sense that matters here. Its licence
is separate from the Sentinel one and is recorded in ground_truth_sources;
scripts/annotate.py is what samples it.

The licence terms below were read first-hand. Art. 7 of Regulation (EU)
1159/2013 grants, in so far as lawful, reproduction, distribution,
communication to the public, adaptation and combination with other data; Art. 8
sets the credit wording; Arts. 3 and 9 disclaim all warranty, including fitness
for purpose -- which for this library means coordinates read off it carry no
accuracy guarantee whatsoever.

The caveat that matters more than the licence: 10 m per pixel. A control point
picked off Sentinel-2 is good to perhaps half a pixel, so about +/- 5 m, which
on a 200 m coastal scene is the largest term in the whole error budget and
swamps everything docs/LIMITATIONS.md prices. Fine for demonstrating the
workflow; not a survey. For real control points use sub-metre national aerial
imagery and check that country's licence separately.

Two things remain unverified and both need a human. Sub-metre open aerial
imagery per country: USGS NAIP is about 0.6 m but blocks automated fetches, so
its terms have to be read by hand. And a fixed-camera image with published
intrinsics under a licence permitting redistribution: expect non-commercial
terms to dominate, and if none is found the honest answer is that you supply
your own frame and the manifest records only what you fetched.

The files themselves stay out of git. Copernicus would let us ship them; ten
megabytes of binary in a library this size is the reason not to.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import annotate
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

IMAGES = pathlib.Path(__file__).resolve().parents[1] / "examples" / "images"
MANIFEST = IMAGES / "MANIFEST.yaml"

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
RESOLUTION_M = 10.0
HALF_WIDTH_M = 4000.0
MAX_CLOUD_PCT = 10.0

LICENCE_PAGE = ("https://sentinels.copernicus.eu/documents/247904/690755/"
                "Sentinel_Data_Legal_Notice")
LICENCE_NAME = "Copernicus Sentinel Data Legal Notice (EU law)"

TCI_BAND_IDS = ("3", "2", "1")
SITE_FIELDS = ("site", "terrain", "scene", "acquired", "resolution_m")

DEM_LICENCE_PAGE = ("https://docs.sentinel-hub.com/api/latest/static/files/data/"
                    "dem/resources/license/License-COPDEM-30.pdf")
DEM_ATTRIBUTION = ("© DLR e.V. 2010-2014 and © Airbus Defence and Space "
                   "GmbH 2014-2018 provided under COPERNICUS by the European Union "
                   "and ESA; all rights reserved")
DEM_LIABILITY_NOTICE = ("The organisations in charge of the Copernicus programme by "
                        "law or by delegation do not incur any liability for any use "
                        "of the Copernicus WorldDEM-30")

GROUND_TRUTH_SOURCES = [{
    "name": "Copernicus Sentinel-2",
    "page_url": LICENCE_PAGE,
    "licence": LICENCE_NAME,
    "legal_basis": ("Regulation (EU) No 377/2014; Commission Delegated "
                    "Regulation (EU) No 1159/2013"),
    "verified": "2026-09-14",
    "resolution_m": RESOLUTION_M,
    "redistribute": True,
    "commercial_use": True,
    "attribution": "Copernicus Sentinel data [Year]",
    "attribution_if_modified": "Contains modified Copernicus Sentinel data [Year]",
}, {
    "name": "Copernicus DEM GLO-30",
    "provides": "ground elevation, metres above the EGM2008 geoid",
    "page_url": DEM_LICENCE_PAGE,
    "licence": "Licence for Copernicus DEM instance COP-DEM-GLO-30-F Global 30m "
               "Full, Free & Open",
    "legal_basis": ("Regulation (EU) No 377/2014; Commission Delegated "
                    "Regulation (EU) No 1159/2013"),
    "verified": "2026-09-17",
    "resolution_m": 30.0,
    "model": "DSM, not DTM: buildings and vegetation are part of the surface",
    "redistribute": True,
    "commercial_use": True,
    "attribution": DEM_ATTRIBUTION,
    "attribution_if_modified": "produced using Copernicus WorldDEM-30 "
                               + DEM_ATTRIBUTION,
    "liability_notice_required_on_distribution": DEM_LIABILITY_NOTICE,
}]


@dataclass(frozen=True)
class Site:
    """A place in examples/terrain_sites.py, and where to look for it."""

    key: str
    site: str
    terrain: str
    features: str
    lat: float
    lon: float
    season: tuple[str, str]


SITES = [
    Site("camargue_salt_pans", "Camargue salt pans", "salt_flat",
         "evaporation-pond dykes, a geometric grid crisp at 10 m",
         43.39, 4.74, ("2024-06-01", "2024-09-30")),
    Site("maasvlakte_container_terminal", "Maasvlakte container terminal",
         "container_yard",
         "quay edges and stack corners, straight lines at high contrast",
         51.95, 4.05, ("2024-04-01", "2024-09-30")),
    Site("plymouth_sound_breakwater", "Plymouth Sound breakwater", "sea",
         "breakwater ends and fort walls, isolated and unambiguous",
         50.33, -4.16, ("2024-05-01", "2024-09-30")),
]


def find_scene(site: Site) -> dict:
    """The least cloudy scene over the site in its season, ties broken by id."""
    body = json.dumps({
        "collections": [COLLECTION],
        "bbox": [site.lon - 0.05, site.lat - 0.05, site.lon + 0.05, site.lat + 0.05],
        "datetime": site.season[0] + "T00:00:00Z/" + site.season[1] + "T00:00:00Z",
        "query": {"eo:cloud_cover": {"lt": MAX_CLOUD_PCT}},
        "limit": 50,
    }).encode()
    request = urllib.request.Request(STAC_URL, data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        features = json.load(response)["features"]
    if not features:
        raise SystemExit("no clear scene for " + site.site)
    return min(features, key=lambda f: (f["properties"]["eo:cloud_cover"], f["id"]))


def acquisition(scene: dict) -> dict:
    """Sensor and sun geometry, from the granule metadata beside the COG."""
    url = scene["assets"]["granule_metadata"]["href"]
    with urllib.request.urlopen(url, timeout=120) as response:
        root = ET.fromstring(response.read())

    def angles(element) -> dict:
        return {child.tag.split("}")[-1]: float(child.text) for child in element}

    view = []
    sun = {}
    for element in root.iter():
        name = element.tag.split("}")[-1]
        if name == "Mean_Sun_Angle":
            sun = angles(element)
        elif (name == "Mean_Viewing_Incidence_Angle"
              and element.attrib.get("bandId") in TCI_BAND_IDS):
            view.append(angles(element))
    if len(view) != len(TCI_BAND_IDS) or not sun:
        raise SystemExit("no acquisition geometry in " + url)

    def mean(key: str) -> float:
        return round(sum(a[key] for a in view) / len(view), 3)

    return {
        "scene": scene["id"],
        "acquired": scene["properties"]["datetime"],
        "resolution_m": RESOLUTION_M,
        "platform": scene["properties"]["platform"],
        "relative_orbit": scene["properties"]["s2:product_uri"].split("_")[4],
        "orthorectified": True,
        "view_zenith_deg": mean("ZENITH_ANGLE"),
        "view_azimuth_deg": mean("AZIMUTH_ANGLE"),
        "sun_zenith_deg": round(sun["ZENITH_ANGLE"], 3),
        "sun_azimuth_deg": round(sun["AZIMUTH_ANGLE"], 3),
    }


def read_window(url: str, site: Site, path: pathlib.Path, tags: dict) -> dict:
    """Cut an 8 km box about the site out of the scene, keeping it georeferenced."""
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES",
                      GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(url) as source:
            east, north = transform_bounds("EPSG:4326", source.crs,
                                           site.lon, site.lat,
                                           site.lon, site.lat)[:2]
            bounds = (east - HALF_WIDTH_M, north - HALF_WIDTH_M,
                      east + HALF_WIDTH_M, north + HALF_WIDTH_M)
            window = from_bounds(*bounds, source.transform).round_lengths()
            pixels = source.read(window=window)
            profile = source.profile | {
                "driver": "GTiff",
                "height": pixels.shape[1],
                "width": pixels.shape[2],
                "transform": source.window_transform(window),
                "compress": "deflate",
                "tiled": True,
            }
            crs = str(source.crs)

    with rasterio.open(path, "w", **profile) as out:
        out.write(pixels)
        out.update_tags(**{key: str(value) for key, value in tags.items()})

    return {"crs": crs, "size_px": [pixels.shape[2], pixels.shape[1]],
            "bbox": [round(v, 1) for v in bounds]}


def entry(site: Site, scene: dict, geometry: dict, tags: dict,
          path: pathlib.Path) -> dict:
    """One manifest record, built from the scene actually fetched."""
    return {
        "file": path.name,
        "description": site.site + " (" + site.terrain + "): " + site.features,
        "terrain": site.terrain,
        "scene": scene["id"],
        "acquired": scene["properties"]["datetime"],
        "cloud_cover_pct": round(scene["properties"]["eo:cloud_cover"], 3),
        "source_url": scene["assets"]["visual"]["href"],
        "page_url": LICENCE_PAGE,
        "licence": LICENCE_NAME,
        "attribution": ("Contains modified Copernicus Sentinel data "
                        + scene["properties"]["datetime"][:4]),
        "redistribute": True,
        "retrieved": dt.date.today().isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "crs": geometry["crs"],
        "resolution_m": RESOLUTION_M,
        "size_px": geometry["size_px"],
        "bbox": geometry["bbox"],
        **{key: value for key, value in tags.items() if key not in SITE_FIELDS},
    }


def _scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _block(records: list[dict]) -> str:
    if not records:
        return " []\n"
    out = "\n"
    for record in records:
        for i, (key, value) in enumerate(record.items()):
            out += ("  - " if i == 0 else "    ") + key + ": " + _scalar(value) + "\n"
    return out


def write_manifest(images: list[dict]) -> None:
    """Rewrite the manifest from the records just built. Data only, no prose."""
    MANIFEST.write_text("images:" + _block(images)
                        + "\nground_truth_sources:" + _block(GROUND_TRUTH_SOURCES),
                        encoding="utf-8")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    images = []
    for site in SITES:
        path = IMAGES / (site.key + ".tif")
        scene = find_scene(site)
        cloud = round(scene["properties"]["eo:cloud_cover"], 2)
        print(site.site + ": " + scene["id"] + "  " + str(cloud) + "% cloud")
        tags = {"site": site.site, "terrain": site.terrain, **acquisition(scene)}
        geometry = read_window(scene["assets"]["visual"]["href"], site, path, tags)
        images.append(entry(site, scene, geometry, tags, path))
        marked = annotate.ensure(path)
        print("  " + path.name + "  "
              + "x".join(str(n) for n in geometry["size_px"]) + " px  "
              + str(round(path.stat().st_size / 1e6, 1)) + " MB  "
              + "view " + str(tags["view_zenith_deg"]) + " deg, sun "
              + str(tags["sun_zenith_deg"]) + " deg")
        print("  " + path.stem + annotate.SUFFIX + "  "
              + str(marked) + " points marked")

    write_manifest(images)
    print("\n" + str(len(images)) + " images, manifest rewritten")


if __name__ == "__main__":
    main()
