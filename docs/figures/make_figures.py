"""Regenerate the data-driven figures. The schematics beside them are authored SVG.

    uv run --group docs python docs/figures/make_figures.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cam2geo import SETUPS, position_error  # noqa: E402

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


if __name__ == "__main__":
    conditioning()
    budget()
    print("wrote conditioning.svg and budget.svg")
