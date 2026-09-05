#!/usr/bin/env python3
"""Boxplots of the per-comparison Dice values behind the reported means.

A mean hides two things this benchmark cares about: how wide the spread is
relative to the differences between methods, and whether a method is beaten
everywhere or only on average.  Each box is the full population its reported
mean summarises -- every region x comparison value -- with the mean marked
separately from the median.

Colours are the documented categorical slots, assigned per method and held
across panels so a method keeps its colour; identity is also carried by the
axis label, never by colour alone.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1, 2, 3, 5, 7 of the reference palette (light mode).
# Slot 4 (yellow) is deliberately unused: beside slot 2 (orange) it fails the
# normal-vision separation floor, and the two ANTs arms would sit adjacent.
BLUE, ORANGE, AQUA, MAGENTA, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#4a3aa7")
GREY = "#8a8f98"          # the affine reference, not a competing method

INK, MUTED, GRID = "#1c1e21", "#5c6169", "#dcdfe4"

VOLUME = [
    ("affine15", "Affine only", GREY),
    ("ants_syn100", "ANTs SyN (MI)", VIOLET),
    ("ants_fmriprep", "ANTs\n(fMRIPrep config)", ORANGE),
    ("mni", "T1Prep", BLUE),
    ("cat12", "CAT12\n(geodesic shooting)", AQUA),
]
SURFACE = [
    ("surf_newmsm", "FSL newMSM\n(fMRIPrep config)", MAGENTA),
    ("surf_fsavg164k", "T1Prep", BLUE),
]


def load(results: Path, name: str, protocol: str) -> np.ndarray:
    with open(results / f"d20_{name}.csv") as fh:
        return np.array([float(r["dice"]) for r in csv.DictReader(fh)
                         if r["protocol"] == protocol])


def panel(ax, results: Path, arms, protocol: str, title: str) -> None:
    data = [load(results, code, protocol) for code, _, _ in arms]
    pos = np.arange(len(arms))
    bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                    showfliers=False, whis=(5, 95),
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=MUTED, linewidth=1),
                    capprops=dict(color=MUTED, linewidth=1))
    for patch, (_, _, colour) in zip(bp["boxes"], arms):
        patch.set_facecolor(colour)
        patch.set_edgecolor("white")      # 2px surface gap between fills
        patch.set_linewidth(2)
    # The mean is what the tables report; mark it so the box explains them.
    ax.scatter(pos, [d.mean() for d in data], marker="D", s=26,
               facecolor="white", edgecolor=INK, linewidth=1.4, zorder=5)
    # Label above the upper whisker, never over the box: at this many boxes the
    # mean often sits within a hair of the median line.
    for x, d in zip(pos, data):
        ax.annotate(f"{d.mean():.3f}", (x, np.percentile(d, 95)),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=9, color=INK, fontweight="medium")

    ax.set_xticks(pos)
    ax.set_xticklabels([label for _, label, _ in arms], fontsize=8.5,
                       color=MUTED)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(here.parent / "results"))
    ap.add_argument("--out", default=str(here.parent / "results" / "dice_boxplots.png"))
    a = ap.parse_args()
    results = Path(a.results)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4),
                             gridspec_kw=dict(width_ratios=[5, 2.2],
                                              hspace=0.42, wspace=0.18))
    fig.patch.set_facecolor("white")
    for row, protocol in enumerate(("loo", "pairs")):
        panel(axes[row][0], results, VOLUME, protocol,
              f"Volume registration - {protocol}")
        panel(axes[row][1], results, SURFACE, protocol,
              f"Surface registration - {protocol}")
        axes[row][0].set_ylabel("Dice", fontsize=9.5, color=MUTED)

    fig.suptitle("Dice per region and comparison, 20 Mindboggle-101 subjects, "
                 "manual DKT31 labels", fontsize=12.5, color=INK, x=0.037,
                 ha="left", y=0.975)
    fig.text(0.037, 0.932,
             "Box = interquartile range, whiskers 5th-95th percentile, white "
             "line = median, white diamond = mean (the value reported in the "
             "tables).",
             fontsize=8.8, color=MUTED, ha="left")
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.062, right=0.985)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
