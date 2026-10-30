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
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1, 2, 3, 5, 6, 7 of the reference palette (light mode).
# Slot 4 (yellow) is deliberately unused: beside slot 2 (orange) it fails the
# normal-vision separation floor, and the two ANTs arms would sit adjacent.
BLUE, ORANGE, AQUA, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#008300", "#4a3aa7")
GREY = "#8a8f98"          # the affine reference, not a competing method
# T1Prep's registration up to 0.8.1 in a lighter step (300) of T1Prep's own
# blue, so the two T1Prep arms read as one method family; between orange and
# aqua it clears the CVD (dE 15.7) and normal-vision (19.2) floors.
BLUE_EARLIER = "#6da7ec"

INK, MUTED, GRID = "#1c1e21", "#5c6169", "#dcdfe4"

# (csv stem, label, colour[, hatched]).  The surface arms are the 19 subjects
# FreeSurfer processed, so that all three are scored on the same ones.
# T1Prep's --fmriprep setting is hatched in T1Prep's own blue: the same method
# with another setting, not another method, so it gets no hue of its own.
VOLUME = [
    ("d20_affine15", "Affine only", GREY),
    ("d20_ants_syn100", "ANTs SyN (MI)", VIOLET),
    ("d20_ants_fmriprep", "ANTs\n(as in sMRIPrep)", ORANGE),
    ("d20_mni", "T1Prep ≤ 0.8.1\n(deepmriprep warp)", BLUE_EARLIER),
    ("d20_cat12", "CAT12\n(geodesic shooting)", AQUA),
    ("d20_t1prep_shoot", "T1Prep default\n(for VBM)", BLUE),
    ("d20_t1prep_shoot_fmriprep", "T1Prep\n(--fmriprep)", BLUE, True),
]
# Neuromorphometrics, 35 subjects, every tissue label; gzipped, since 595
# pairs x 132 labels make each file eight times a Mindboggle one.
NMM = [
    ("n35_ants_fmriprep", "ANTs\n(as in sMRIPrep)", ORANGE),
    ("n35_t1prep_shoot", "T1Prep default\n(for VBM)", BLUE),
    ("n35_t1prep_shoot_fmriprep", "T1Prep\n(--fmriprep)", BLUE, True),
]
SURFACE = [
    ("d19_newmsm", "FSL newMSM\n(fMRIPrep config)", MAGENTA),
    ("d19_t1p_central", "T1Prep", BLUE),
    ("d19_fs81_mid", "FreeSurfer 8.1\n(recon-all)", GREEN),
]


def load(results: Path, name: str, protocol: str) -> np.ndarray:
    path = results / f"{name}.csv"
    if not path.exists():
        path = results / f"{name}.csv.gz"
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as fh:
        return np.array([float(r["dice"]) for r in csv.DictReader(fh)
                         if r["protocol"] == protocol])


def panel(ax, results: Path, arms, protocol: str, title: str) -> None:
    data = [load(results, arm[0], protocol) for arm in arms]
    pos = np.arange(len(arms))
    bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                    showfliers=False, whis=(5, 95),
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=MUTED, linewidth=1),
                    capprops=dict(color=MUTED, linewidth=1))
    for patch, median, arm in zip(bp["boxes"], bp["medians"], arms):
        colour, hatched = arm[2], len(arm) > 3 and arm[3]
        if hatched:
            patch.set_facecolor("white")
            patch.set_edgecolor(colour)   # the hatch takes the edge colour
            patch.set_hatch("////")
            patch.set_linewidth(1.5)
            median.set_color(colour)
        else:
            patch.set_facecolor(colour)
            patch.set_edgecolor("white")  # 2px surface gap between fills
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
    ax.set_xticklabels([arm[1] for arm in arms], fontsize=8.5, color=MUTED)
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

    fig, axes = plt.subplots(2, 3, figsize=(19.2, 8.8),
                             gridspec_kw=dict(width_ratios=[7, 3.2, 3.2],
                                              hspace=0.5, wspace=0.16))
    fig.patch.set_facecolor("white")
    for row, protocol in enumerate(("loo", "pairs")):
        panel(axes[row][0], results, VOLUME, protocol,
              f"Mindboggle-101, volume registration\n{protocol} (20 subjects)")
        panel(axes[row][1], results, NMM, protocol,
              f"Neuromorphometrics, volume\n{protocol} (35 subjects)")
        panel(axes[row][2], results, SURFACE, protocol,
              f"Mindboggle-101, surface registration\n{protocol} (19 subjects)")
        axes[row][0].set_ylabel("Dice", fontsize=9.5, color=MUTED)

    fig.suptitle("Dice per region and comparison: Mindboggle-101 (manual "
                 "DKT31 labels) and Neuromorphometrics (manual whole-brain "
                 "labels)", fontsize=12.5, color=INK, x=0.031, ha="left",
                 y=0.975)
    fig.text(0.031, 0.932,
             "Box = interquartile range, whiskers 5th-95th percentile, line = "
             "median, white diamond = mean (the value reported in the tables); "
             "hatched = T1Prep's --fmriprep setting.",
             fontsize=8.8, color=MUTED, ha="left")
    fig.subplots_adjust(top=0.855, bottom=0.07, left=0.05, right=0.988)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
