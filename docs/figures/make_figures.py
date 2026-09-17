"""Regenerate the data-driven figures. The schematics beside them are authored SVG.

    uv run --group docs python docs/figures/make_figures.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cam2geo import position_error  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from setups import SETUPS  # noqa: E402

HERE = pathlib.Path(__file__).parent
# Readable on a white or a dark page, which is what a README gets.
COLOURS = ("#2f6f9f", "#c2571a", "#3f8f5f")
GRID = "#9aa3ab"


def conditioning() -> None:
    """Ground sample distance against image row, for all three demo setups."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for colour, make in zip(COLOURS, SETUPS, strict=True):
        setup = make()
        view = setup.station.view
        rows = np.arange(view.height - 1.0, 0.0, -1.0)
        cols = np.full(rows.size, view.width / 2.0)
        gsd = view.ground_m_per_px(cols, rows)
        keep = np.isfinite(gsd) & (gsd < 1e4)
        ax.plot(rows[keep], gsd[keep], color=colour, lw=2.0, label=setup.name)

    ax.axhline(1.0, color=GRID, ls="--", lw=1.2)
    ax.text(30, 1.15, "max_m_per_px = 1.0", color=GRID, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("image row (bottom of frame on the right)")
    ax.set_ylabel("ground_m_per_px")
    ax.set_title("One pixel is worth metres near the horizon")
    ax.invert_xaxis()
    ax.grid(True, which="both", color=GRID, alpha=0.25)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "conditioning.svg", format="svg", transparent=True)
    plt.close(fig)


def budget() -> None:
    """Which error term dominates, against range, for the coastal setup."""
    setup = SETUPS[0]()
    view = setup.station.view
    rows = np.arange(view.height - 1.0, 0.0, -1.0)
    cols = np.full(rows.size, view.width / 2.0)
    surface = setup.surface.but(offset_sigma_m=0.5)
    b = position_error(setup.station, cols, rows, surface, pixel_sigma_px=1.0,
                       pointing_sigma_deg=0.05, control_rms_px=2.0)
    keep = np.isfinite(b.range_m) & (b.range_m < 700.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for colour, (label, term) in zip(
        (*COLOURS, "#7a5ea8"),
        [("1 px of detection error", b.pixel_m), ("0.5 m of swell", b.height_sigma_m),
         ("0.05 deg of mast sway", b.pointing_m), ("total", b.total_m)],
        strict=True,
    ):
        ax.plot(b.range_m[keep], term[keep], color=colour, lw=2.2 if label == "total"
                else 1.8, ls="-" if label == "total" else "--", label=label)

    ax.set_xlabel("range from the camera (m)")
    ax.set_ylabel("position error (m)")
    ax.set_title("Pixel precision is not what limits you")
    ax.set_yscale("log")
    ax.grid(True, which="both", color=GRID, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "budget.svg", format="svg", transparent=True)
    plt.close(fig)


def _frame(ax, width, height, view):
    """An empty camera frame with its horizon drawn, to overlay marks on."""
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, color=GRID, lw=1.0))
    a, b, c = view.horizon_line
    rows = [-(a * u + c) / b for u in (0.0, width)]
    ax.plot([0, width], rows, color="#c2571a", lw=1.4, ls="--")
    ax.text(width * 0.02, rows[0] - 26, "horizon -- no ground above this line",
            color="#c2571a", fontsize=8)
    ax.set_xlim(-40, width + 40)
    ax.set_ylim(height + 40, -40)          # image rows run downward
    ax.set_aspect("equal")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def marks_and_drift() -> None:
    """The eight real-image marks, and how far each moved by the later frame.

    Built from the literals in examples/from_a_real_image.py so the figure and
    the example can never disagree.
    """
    import sys

    sys.path.insert(0, str(HERE.parent.parent / "examples"))
    from from_a_real_image import (
        BOXES,
        HEIGHT,
        LATER,
        LENS,
        MARKS,
        NAMES,
        WIDTH,
    )

    from cam2geo import align_pixels, anchor_pixel, fit_homography
    from cam2geo.frames import GeodeticPlane

    marks = np.array(MARKS)
    pixels, degrees = marks[:, :2], marks[:, 2:]
    later = np.array(LATER)
    view = fit_homography(degrees, pixels, WIDTH, HEIGHT, frame=GeodeticPlane(),
                          method="exact").homography
    drift = align_pixels(pixels, later, lens=LENS)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    _frame(ax, WIDTH, HEIGHT, view)

    # The detections, to show what the lower frame is actually for: the marks
    # are all far away and so all crammed near the horizon, while the things
    # being tracked are near and spread down the frame.
    anchors = anchor_pixel(BOXES[:, 0], BOXES[:, 1], BOXES[:, 2], BOXES[:, 3])
    ax.scatter(anchors[:, 0], anchors[:, 1], s=70, marker="v", zorder=3,
               facecolors="none", edgecolors=GRID, lw=1.4, label="detections")

    ax.scatter(pixels[:, 0], pixels[:, 1], s=70, facecolors="none", lw=1.6,
               edgecolors=COLOURS[0], zorder=4, label="mark, March frame")
    ax.scatter(later[:, 0], later[:, 1], s=22, color=COLOURS[2], marker="s",
               zorder=5, label="same mark, September")
    # Motion is a few pixels, so it needs exaggerating to be visible at all.
    scale = 14.0
    for i, name in enumerate(NAMES):
        step = (later[i] - pixels[i]) * scale
        ax.annotate("", xy=pixels[i] + step, xytext=pixels[i],
                    arrowprops={"arrowstyle": "->", "color": "#7a5ea8", "lw": 1.3})
        # Three tiers, not two: eight labels inside ninety rows collide even
        # when alternated, because several marks share a column.
        offset = (-38, 50, 74)[i % 3]
        ax.text(pixels[i, 0], pixels[i, 1] + offset, name, fontsize=7,
                ha="center", va="center", color="#444")

    ax.annotate("every mark sits within 90 rows of the horizon:\n"
                "at 300-870 m a shallow view squeezes them all together",
                xy=(WIDTH * 0.5, 640), fontsize=8, color="#444", ha="center")
    ax.set_title(f"Eight fixed features, and {drift.drift_deg:.2f} deg of drift "
                 f"between two frames (arrows x{scale:.0f})")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "marks.svg", format="svg", transparent=True)
    plt.close(fig)


def corrections() -> None:
    """What each correction is worth against range, over the terrain sites.

    Averaged over independent surveys, for the reason terrain_sites.py explains:
    a correction removes a bias while the control error is noise, so one survey
    shows nothing.
    """
    import sys

    sys.path.insert(0, str(HERE.parent.parent / "examples"))
    from terrain_sites import (
        HEIGHT,
        SITES,
        WIDTH,
        metres_apart,
        survey_off_sentinel,
        truth_lonlat,
    )

    from cam2geo import CameraPose, CameraStation, fit_homography
    from cam2geo.frames import GeodeticPlane

    fig, axes = plt.subplots(1, len(SITES), figsize=(11.0, 3.8), sharey=True)
    for ax, colour, site in zip(axes, COLOURS, SITES, strict=True):
        rows = np.linspace(site.rows.max(), site.rows.min(), 14)
        cols = np.full(rows.size, WIDTH / 2.0 + 120.0)
        lon_true, lat_true = truth_lonlat(site, cols, rows)
        stages = {"nothing to correct" if not site.target_height_m
                  else "no correction": {}}
        if site.target_height_m:
            stages["height corrected"] = {"target_height_m": site.target_height_m}

        totals = {k: np.zeros(rows.size) for k in stages}
        draws = 120
        for draw in range(draws):
            deg, px = survey_off_sentinel(site, np.random.default_rng(500 + draw))
            fitted = fit_homography(deg, px, WIDTH, HEIGHT,
                                    frame=GeodeticPlane(), method="exact")
            station = CameraStation(fitted.homography,
                                    CameraPose(height_m=site.height_m,
                                               x=site.lon, y=site.lat))
            ranges = station.range_m(cols, rows)
            for label, kwargs in stages.items():
                placed = station.project(cols, rows, **kwargs)
                totals[label] += metres_apart(site, placed.x, placed.y,
                                              lon_true, lat_true)
        ranges = np.asarray(ranges)

        for style, (label, _) in zip(("--", "-"), stages.items(), strict=False):
            ax.plot(ranges, totals[label] / draws, color=colour, lw=2.0, ls=style,
                    label=label)
        ax.set_title(site.name.replace(" ", "\n", 1), fontsize=9)
        ax.set_xlabel("range (m)")
        ax.grid(True, color=GRID, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("mean position error (m)")
    fig.suptitle("Corrections against range, mean of 120 surveys off 10 m imagery",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "corrections.svg", format="svg", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    conditioning()
    budget()
    marks_and_drift()
    corrections()
    print("wrote conditioning.svg, budget.svg, marks.svg and corrections.svg")
