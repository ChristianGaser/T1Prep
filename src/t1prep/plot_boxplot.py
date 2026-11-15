#!/usr/bin/env python3
"""Box and violin plots of one or more volume, surface or text data sets.

Python counterpart of CAT12's ``cat_plot_boxplot.m``.  It reads the same files
as ``CAT_PlotHistogram`` -- NIfTI volumes, GIFTI surface overlays, FreeSurfer
morphometry data and plain text -- and draws one box or violin per input.
``--columns`` makes every column of a text file a box of its own, which is
what the matrix input of the MATLAB original does.

The statistics are those of ``cat_plot_boxplot``: quartiles as
``cat_stat_prctile`` computes them, whiskers out to the last value within
``whisker`` times the interquartile range (IQR) of the box, mild outliers up
to twice that distance and extreme outliers beyond it, and the notch at the
median +/- 1.57 IQR / sqrt(n), kept inside the box.  The drawing is
matplotlib's own -- ``Axes.bxp`` for the boxes, ``Axes.violin`` for the
violins -- and the violins show the Gaussian kernel density estimate of
``CAT_PlotHistogram``: the Improved Sheather-Jones bandwidth of Botev et al.
(2010), the estimator of the MATLAB original as well.

``--style`` picks one of the predefined styles of ``opt.style``; every other
option changes one thing of it:

==========  =====  ======================================================
box         0      box plot
violin      1      violin plot, the values drawn as a barcode
violinbox   2      violin plot with a thin box plot inside
density     3      horizontal half violins, the values along the baseline
jitter      4      box plot with the values as points, spread by density
==========  =====  ======================================================

Volumes and GIFTI overlays lose their zero background as in
``CAT_PlotHistogram``.  Text files keep their zeros: a list of values has no
background, and dropping zeros would shift the quartiles.

Usage::

    # one box per volume
    CAT_PlotBoxplot mwp1*.nii

    # violins with a box inside, ordered by the median
    CAT_PlotBoxplot --style violinbox --sort ascending lh.thickness.*

    # one value per subject; the outliers listed by their place in the file
    CAT_GrepJson --field subjectmeasures.vol_TIV --out tiv.txt report/log_*.json
    CAT_PlotBoxplot --style jitter --list-outliers tiv.txt

    # one box per column: CSF, GM and WM volume
    CAT_GrepJson --field subjectmeasures.vol_abs_CGW --out cgw.txt report/log_*.json
    CAT_PlotBoxplot --columns --names CSF,GM,WM cgw.txt

    # from a source checkout, without the entry point on PATH
    python -m t1prep.plot_boxplot mwp1*.nii

The same thing from Python, where it returns the statistics and the figure::

    from t1prep.plot_boxplot import plot_boxplot

    result = plot_boxplot(["tiv_controls.txt", "tiv_patients.txt"], notch=True)
    print(result.matrix())        # the s of cat_plot_boxplot
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .cli_help import ArgumentParser
from .plot_histogram import (
    DEFAULT_PALETTE,
    PALETTES,
    _ellipsize,
    _figure,
    _finite,
    colormap,
    kernel_density,
    load_data,
    shorten_names,
)

__all__ = [
    "SHOW_DATA",
    "SORTS",
    "STYLES",
    "BoxStats",
    "BoxplotResult",
    "box_stats",
    "main",
    "plot_boxplot",
    "prctile",
    "read_text",
]


# ----------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Style:
    """The defaults a ``style`` sets; the single options override them."""

    #: Draw a violin, the kernel density estimate of the values.
    violin: bool
    #: The box: ``"full"``, ``"thin"`` (inside a violin) or ``"none"``.
    box: str
    #: How the values themselves are drawn, one of :data:`SHOW_DATA`.
    show_data: str
    #: Stack the boxes from top to bottom instead of left to right.
    horizontal: bool
    #: Width of a box, or of a violin at its widest, in box spacings.
    width: float
    #: Draw only the upper half of every violin.
    half: bool = False
    #: Mark the outliers.  A violin shows the tails itself, and with the
    #: many values of a volume or a surface the markers would bury it.
    outliers: bool = True


#: The predefined styles, in the order of ``opt.style`` (0-4).
STYLES: Dict[str, Style] = {
    "box": Style(violin=False, box="full", show_data="none",
                 horizontal=False, width=0.6),
    "violin": Style(violin=True, box="none", show_data="barcode",
                    horizontal=False, width=0.8, outliers=False),
    # The barcode of cat_plot_boxplot's style 2 would cover the thin box
    "violinbox": Style(violin=True, box="thin", show_data="none",
                       horizontal=False, width=0.8, outliers=False),
    "density": Style(violin=True, box="none", show_data="strip",
                     horizontal=True, width=1.7, half=True, outliers=False),
    "jitter": Style(violin=False, box="full", show_data="jitter",
                    horizontal=False, width=0.6),
}

#: How the values can be drawn: not at all, as points on the centre line, as
#: points spread sideways by their density, or as one short line each.
SHOW_DATA: Tuple[str, ...] = ("none", "strip", "jitter", "barcode")

#: Orders of the boxes: as given, or by their median.
SORTS: Tuple[str, ...] = ("none", "ascending", "descending")

#: Points at which a violin's density is evaluated.  matplotlib's default of
#: 100 cannot follow the narrow peaks of a p1 map.
_VIOLIN_POINTS = 512

#: Values drawn per box at most.  More points add nothing to a cloud that is
#: already solid but take long to render, so a random subset is drawn; the
#: statistics always use every value.
_MAX_POINTS = 5000

#: Point artists with more markers than that are rasterized, which keeps a
#: figure saved as PDF or SVG small.
_RASTERIZE_ABOVE = 2000


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------


def prctile(values, percentiles) -> np.ndarray:
    """Return *percentiles* of *values* as CAT12's ``cat_stat_prctile`` does.

    The p-th percentile sits at the 1-based position ``p/100 n + 0.5`` of the
    sorted values, limited to ``[1, n]`` and linearly interpolated between
    neighbours: MATLAB's ``prctile`` and numpy's ``"hazen"`` method.
    matplotlib's ``boxplot`` uses numpy's default ``"linear"`` method instead,
    position ``p/100 (n - 1) + 1``, which gives other quartiles for small
    samples.

    Args:
        values: Finite values.
        percentiles: Percentiles in ``[0, 100]``.

    Returns:
        The percentiles, shaped like *percentiles*.
    """
    return np.percentile(np.asarray(values, dtype=np.float64), percentiles,
                         method="hazen")


def _no_index() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


@dataclass
class BoxStats:
    """Statistics of one box.

    The numbers of one column of the ``s`` matrix of ``cat_plot_boxplot``,
    plus the outliers of its ``out`` struct.
    """

    name: str
    #: Number of finite values; NaN and Inf are left out.
    n: int
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    #: Confidence interval of the median, the notch: median +/- 1.57 IQR /
    #: sqrt(n), kept within ``[q1, q3]``.
    ci_low: float
    ci_high: float
    #: Ends of the whiskers: the last values within the whisker reach.
    whisker_low: float
    whisker_high: float
    #: Indices of the outliers, 0-based, into the data set as it was passed
    #: in and flattened in numpy's C order -- NaN included, so that they match
    #: its positions; ``np.unravel_index`` turns them into voxel coordinates.
    #: Mild outliers lie at most one whisker reach beyond the whisker limit,
    #: extreme ones further out.
    mild_low: np.ndarray = field(default_factory=_no_index)
    mild_high: np.ndarray = field(default_factory=_no_index)
    extreme_low: np.ndarray = field(default_factory=_no_index)
    extreme_high: np.ndarray = field(default_factory=_no_index)

    @property
    def outliers(self) -> np.ndarray:
        """Indices of all outliers, in ascending order."""
        return np.sort(np.concatenate([self.mild_low, self.mild_high,
                                       self.extreme_low, self.extreme_high]))


def box_stats(sample, whisker: float = 1.5, name: str = "") -> BoxStats:
    """Compute the statistics of one box as ``cat_plot_boxplot`` does.

    The quartiles come from :func:`prctile`.  With ``reach = whisker * IQR``
    the whiskers end at the last value within *reach* of the box -- never
    inside it, as in matplotlib.  Values beyond that limit are mild outliers
    up to another *reach* further out, extreme outliers beyond; with the
    default 1.5 these are Tukey's inner and outer fences at 1.5 and 3 IQR.

    Args:
        sample: The values; non-finite ones are left out.
        whisker: Length of the whiskers in IQRs; ``0`` makes every value
            outside the box an extreme outlier.
        name: Label of the box.

    Returns:
        The :class:`BoxStats`, all NaN when *sample* has no finite value.
    """
    if not whisker >= 0:
        raise ValueError(f"whisker has to be zero or positive, not {whisker:g}")
    raw = np.asarray(sample, dtype=np.float64).ravel()
    index = np.flatnonzero(np.isfinite(raw))
    values = raw[index]
    n = int(values.size)
    if n == 0:
        nan = float("nan")
        return BoxStats(name, 0, *([nan] * 11))

    q1, median, q3 = (float(v) for v in prctile(values, [25, 50, 75]))
    reach = whisker * (q3 - q1)
    low, high = q1 - reach, q3 + reach
    notch = 1.57 * (q3 - q1) / np.sqrt(n)
    return BoxStats(
        name=name,
        n=n,
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if n > 1 else 0.0,
        min=float(values.min()),
        q1=q1,
        median=median,
        q3=q3,
        max=float(values.max()),
        ci_low=max(median - notch, q1),
        ci_high=min(median + notch, q3),
        # Neither selection can be empty: the extremes lie beyond the box
        whisker_low=min(float(values[values >= low].min()), q1),
        whisker_high=max(float(values[values <= high].max()), q3),
        mild_low=index[(values < low) & (values >= low - reach)],
        mild_high=index[(values > high) & (values <= high + reach)],
        extreme_low=index[values < low - reach],
        extreme_high=index[values > high + reach],
    )


@dataclass
class BoxplotResult:
    """Everything :func:`plot_boxplot` computed and drew."""

    #: One :class:`BoxStats` per box, in the order of the input.
    stats: List[BoxStats] = field(default_factory=list)
    #: Input index of every box in the order drawn: left to right, or top to
    #: bottom with ``horizontal``.
    order: List[int] = field(default_factory=list)
    #: Position of every box on its axis, in the order of the input.
    positions: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: The tick labels, in the order of the input.
    labels: List[str] = field(default_factory=list)
    #: The figure that was drawn.
    figure: object = None

    def matrix(self) -> np.ndarray:
        """Return the statistics as the ``s`` matrix of ``cat_plot_boxplot``.

        Returns:
            Shape ``(7, n)``, one column per box in the order of the input;
            rows minimum, first quartile, median, third quartile, maximum,
            and the lower and upper confidence limit of the median.
        """
        return np.array([[s.min, s.q1, s.median, s.q3, s.max, s.ci_low,
                          s.ci_high] for s in self.stats]).reshape(-1, 7).T


# ----------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------

#: Tokens of a missing value, so that the line keeps its place.  CAT_GrepJson
#: writes ``N/A`` for a field a report lacks and ``null`` for an empty one.
_MISSING = frozenset({"n/a", "na", "null", "none"})

#: A line that starts with one of these is a comment.
_COMMENT = ("#", "%")

#: Extensions of the volumes and surfaces ``load_data`` reads by name; any
#: other file is tried as FreeSurfer morphometry data.
_IMAGE_SUFFIXES = (".nii", ".nii.gz", ".img", ".hdr", ".gii")


def _number(token: str) -> Optional[float]:
    """Return *token* as a number, NaN for a missing value, else ``None``."""
    try:
        return float(token)
    except ValueError:
        return float("nan") if token.lower() in _MISSING else None


def read_text(path: str, columns: bool = False
              ) -> Tuple[List[np.ndarray], Optional[List[str]]]:
    """Read the numbers of a whitespace-separated text file.

    Words -- a header, a subject name in front of the values -- are skipped,
    as are lines that start with ``#`` or ``%``.  ``N/A``, ``NA``, ``null``
    and ``none`` count as missing values, so that the row keeps its place.
    Unlike ``CAT_PlotHistogram``, zeros stay: a list of values has no
    background.

    Args:
        path: The text file.
        columns: Return every column as a data set of its own, the matrix
            input of ``cat_plot_boxplot``.  All rows then need the same
            number of values, except for a row of missing values only:
            CAT_GrepJson writes a single ``N/A`` for a report that lacks a
            field of several values, and it stands for the whole row.

    Returns:
        The data sets -- all numbers of the file in one, or one per column --
        and, with *columns*, their names: the words of a header line above
        the first row of values whose fields line up with it, else the
        column numbers.
    """
    rows: List[np.ndarray] = []
    lines: List[int] = []
    words: Optional[List[str]] = None
    header: Optional[List[str]] = None
    named = False
    # utf-8-sig drops the byte order mark some editors write, which would
    # otherwise stick to the first number and make it unreadable
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as handle:
        for number, line in enumerate(handle, start=1):
            tokens = line.split()
            if not tokens or tokens[0].startswith(_COMMENT):
                continue
            values = [_number(token) for token in tokens]
            fields = [i for i, value in enumerate(values) if value is not None]
            if not fields:
                if not rows:
                    words = tokens
                continue
            row = np.array([values[i] for i in fields], dtype=np.float64)
            if not named and not np.isnan(row).all():
                # A header names the fields of the first row of values
                named = True
                if words is not None and len(words) == len(tokens):
                    header = [words[i] for i in fields]
            rows.append(row)
            lines.append(number)
    if not rows:
        raise ValueError(f"no numbers found in {path}")
    if not columns:
        return [np.concatenate(rows)], None

    # The width of most rows with values; a row of missing values only is
    # widened to it
    widths = [len(row) for row in rows if not np.isnan(row).all()]
    width = Counter(widths or [len(rows[0])]).most_common(1)[0][0]
    for k, (row, number) in enumerate(zip(rows, lines)):
        if len(row) == width:
            continue
        if np.isnan(row).all():
            rows[k] = np.full(width, np.nan)
            continue
        raise ValueError(
            f"{path}, line {number}: {len(row)} values where the other rows "
            f"have {width}; columns need the same number in every row"
        )
    table = np.vstack(rows)
    if header is None or len(header) != width:
        header = [str(j + 1) for j in range(width)]
    return [table[:, j] for j in range(width)], header


def _read_files(paths: Sequence[str], columns: bool
                ) -> Tuple[List[np.ndarray], List[str]]:
    """Read every file, returning the data sets and their labels."""
    file_labels = shorten_names(list(paths))
    samples: List[np.ndarray] = []
    labels: List[str] = []
    for path, label in zip(paths, file_labels):
        if not os.path.isfile(path):
            # Before the readers, which would call it an unknown format
            raise FileNotFoundError(f"no such file: {path}")
        if path.lower().endswith(".txt"):
            sets, names = read_text(path, columns)
        else:
            try:
                sets, names = [load_data(path)[0]], None
            except Exception as exc:
                hint = ("" if path.lower().endswith(_IMAGE_SUFFIXES)
                        else "; text files need the extension .txt")
                raise ValueError(f"cannot read {path}: {exc}{hint}") from exc
        if names is None or len(sets) == 1:
            labels.append(label)
        elif len(paths) == 1:
            labels.extend(names)
        else:
            labels.extend(f"{label}:{name}" for name in names)
        samples.extend(sets)
    return samples, labels


DataLike = Union[str, os.PathLike, np.ndarray, Mapping,
                 Sequence[Union[str, os.PathLike, np.ndarray, Sequence[float]]]]


def _normalise_inputs(data: DataLike
                      ) -> Tuple[list, bool, Optional[List[str]]]:
    """Bring the accepted input forms into one list.

    Accepts a file name, a list of file names, a vector or a list of
    numbers (one data set), a matrix or a pandas DataFrame with one column
    per data set as in ``cat_plot_boxplot``, a list of arrays, or a mapping
    of names to arrays.

    Returns:
        The list of inputs, whether they are file names, and the labels the
        input brings along: the keys of a mapping, the columns of a
        DataFrame, the name of a Series -- else ``None``.
    """
    if isinstance(data, (str, os.PathLike)):
        return [os.fspath(data)], True, None
    if isinstance(data, Mapping):
        labels = [str(key) for key in data]
        return ([np.asarray(value, dtype=np.float64)
                 for value in data.values()], False, labels)
    labels: Optional[List[str]] = None
    if hasattr(data, "to_numpy") and not isinstance(data, np.ndarray):
        # pandas: a DataFrame is a matrix, a Series a vector
        if hasattr(data, "columns"):
            labels = [str(column) for column in data.columns]
        elif getattr(data, "name", None) is not None:
            labels = [str(data.name)]
        data = data.to_numpy(dtype=np.float64)
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            return [column for column in data.T], False, labels
        # A volume keeps its shape, so that outliers can be listed by voxel
        return [data], False, labels
    items = list(data)
    if not items:
        raise ValueError("no input data specified")
    paths = [isinstance(item, (str, os.PathLike)) for item in items]
    if all(paths):
        return [os.fspath(item) for item in items], True, None
    if any(paths):
        raise ValueError("cannot mix file names and arrays in one call")
    if all(np.ndim(item) == 0 for item in items):
        # A list of numbers is one data set, not one per number
        return [np.asarray(items, dtype=np.float64)], False, None
    return [np.asarray(item, dtype=np.float64) for item in items], False, None


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------


def _accepts(method, name: str) -> bool:
    """True when the matplotlib *method* takes a keyword *name*."""
    try:
        return name in inspect.signature(method).parameters
    except (TypeError, ValueError):  # pragma: no cover - C or odd wrappers
        return False


def _orientation(method, horizontal: bool) -> Dict[str, object]:
    """The keyword that orients ``bxp`` and ``violin``.

    matplotlib 3.10 replaced ``vert`` by ``orientation`` and deprecated the
    former; the oldest supported matplotlib only knows ``vert``.
    """
    if _accepts(method, "orientation"):
        return {"orientation": "horizontal" if horizontal else "vertical"}
    return {"vert": not horizontal}


def _xy(across, along, horizontal: bool):
    """Turn coordinates across and along the value axis into ``(x, y)``."""
    return (along, across) if horizontal else (across, along)


def _shade(rgb, factor: float = 0.55) -> Tuple[float, ...]:
    """A darker shade of *rgb*, for lines and points on a filled box."""
    return tuple(float(c) * factor for c in rgb[:3])


def _subset(index: np.ndarray, rng) -> np.ndarray:
    """At most :data:`_MAX_POINTS` of *index*, drawn at random, in order."""
    if index.size <= _MAX_POINTS:
        return index
    return np.sort(rng.choice(index, _MAX_POINTS, replace=False))


def _value_range(ylim: Sequence[float]) -> Tuple[float, float]:
    """Return ``(low, high)`` of *ylim*, an end that is NaN or Inf open.

    ``opt.ylim`` of the MATLAB original takes ``-inf``/``inf`` for an end
    left to the data; on the command line ``nan`` does that, because
    argparse reads ``-inf`` as an option.
    """
    first, second = (float(v) for v in ylim)
    if np.isfinite(first) and np.isfinite(second):
        return min(first, second), max(first, second)
    return (first if np.isfinite(first) else -np.inf,
            second if np.isfinite(second) else np.inf)


def _density(values: np.ndarray, window: Optional[Sequence[float]] = None
             ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the kernel density of *values* between their extremes.

    Cut at the smallest and the largest value, as matplotlib's violins are,
    and at the edges of *window*: a violin is scaled to its widest point, so
    that a peak cut off by the value axis -- the near-zero values of a p1 map
    -- would otherwise flatten what is left in view.  Every value enters the
    estimate either way.

    Returns:
        ``(coords, density)``, or ``None`` when there is nothing to estimate:
        fewer than two values, all of them equal, or none in *window*.
    """
    if values.size < 2:
        return None
    low, high = float(values.min()), float(values.max())
    if window is not None:
        low, high = max(low, min(window)), min(high, max(window))
    if not high > low:
        return None
    coords = np.linspace(low, high, _VIOLIN_POINTS)
    density = kernel_density(values, coords)
    if not density.max() > 0:
        return None
    return coords, density


def _draw_violins(axes, curves, positions, widths, horizontal: bool,
                  half: bool, colors, alpha: float, fill: bool) -> None:
    """Draw the violins with matplotlib's ``Axes.violin`` and colour them."""
    drawn = [i for i, curve in enumerate(curves) if curve is not None]
    if not drawn:
        return
    nan = float("nan")
    vpstats = [{"coords": curves[i][0], "vals": curves[i][1], "mean": nan,
                "median": nan, "min": nan, "max": nan} for i in drawn]
    where = [float(positions[i]) for i in drawn]
    size = [float(widths[i]) for i in drawn]
    if half and not _accepts(axes.violin, "side"):
        # matplotlib < 3.9 draws whole violins only: the upper halves are
        # drawn here the way Axes.violin draws them
        fill_area = axes.fill_between if horizontal else axes.fill_betweenx
        bodies = []
        for stats, pos, width in zip(vpstats, where, size):
            reach = 0.5 * width * stats["vals"] / stats["vals"].max()
            bodies.append(fill_area(stats["coords"], pos, pos + reach))
    else:
        options: Dict[str, object] = {
            "positions": where, "widths": size, "showmeans": False,
            "showextrema": False, "showmedians": False}
        options.update(_orientation(axes.violin, horizontal))
        if half:
            options["side"] = "high"
        bodies = axes.violin(vpstats, **options)["bodies"]
    for body, i in zip(bodies, drawn):
        # A set alpha would override the opaque edge
        body.set_alpha(None)
        body.set_facecolor((*colors[i], alpha) if fill else "none")
        body.set_edgecolor(colors[i])
        body.set_linewidth(1.0)
        body.set_zorder(1)


def _draw_violin_medians(axes, stats, curves, positions, widths,
                         horizontal: bool, half: bool, colors,
                         flat_only: bool = False) -> None:
    """Draw the median of every violin as a line out to its outline.

    A data set too small or too uniform for a violin gets a line of a
    quarter of the width, so that it does not look empty.  With *flat_only*
    -- when a box shows the medians -- only those sets get one.
    """
    for i, (entry, curve) in enumerate(zip(stats, curves)):
        if not entry.n or (flat_only and curve is not None):
            continue
        if curve is None:
            # One value, or all equal: there is no outline to reach
            reach = 0.25 * widths[i]
        else:
            coords, density = curve
            reach = (0.5 * widths[i] * np.interp(entry.median, coords, density)
                     / density.max())
        start = positions[i] if half else positions[i] - reach
        axes.plot(*_xy([start, positions[i] + reach], [entry.median] * 2,
                       horizontal),
                  color=_shade(colors[i]), linewidth=2, zorder=2.5,
                  solid_capstyle="butt")


def _draw_boxes(axes, stats, positions, widths, horizontal: bool, colors,
                alpha: float, fill: bool, notch: bool, thin: bool) -> None:
    """Draw the boxes with matplotlib's ``Axes.bxp`` and colour them.

    A *thin* box sits inside a violin: dark, without caps and notch, with a
    white median.
    """
    drawn = [i for i, entry in enumerate(stats) if entry.n]
    entries = [{"med": s.median, "q1": s.q1, "q3": s.q3,
                "whislo": s.whisker_low, "whishi": s.whisker_high,
                "cilo": s.ci_low, "cihi": s.ci_high, "fliers": []}
               for s in (stats[i] for i in drawn)]
    options: Dict[str, object] = {
        "positions": [float(positions[i]) for i in drawn],
        "widths": [(0.1 if thin else 1.0) * float(widths[i]) for i in drawn],
        "patch_artist": True, "shownotches": notch and not thin,
        "showcaps": not thin, "showfliers": False, "manage_ticks": False,
        "zorder": 2}
    options.update(_orientation(axes.bxp, horizontal))
    artists = axes.bxp(entries, **options)

    for k, i in enumerate(drawn):
        box = artists["boxes"][k]
        if thin:
            box.set_facecolor("0.25")
            box.set_edgecolor("0.25")
            line_color, median_color = "0.25", "white"
        else:
            box.set_facecolor((*colors[i], alpha) if fill else "none")
            box.set_edgecolor(colors[i])
            line_color, median_color = colors[i], _shade(colors[i])
        box.set_linewidth(1.0)
        for line in (artists["whiskers"][2 * k:2 * k + 2]
                     + artists["caps"][2 * k:2 * k + 2]):
            line.set_color(line_color)
        median = artists["medians"][k]
        median.set_color(median_color)
        median.set_linewidth(2)
        median.set_solid_capstyle("butt")


def _jitter(values: np.ndarray, curve, width: float, half: bool,
            rng) -> np.ndarray:
    """Sideways offsets of *values*, wide where they are dense.

    A sina plot: uniform within the outline the violin would have at that
    value, 80% of it, so that the points stay inside a violin and spread
    over a box.  Only upwards for a half violin.
    """
    if curve is None:
        scale = np.ones(values.size)
    else:
        coords, density = curve
        scale = np.interp(values, coords, density) / density.max()
    spread = rng.uniform(0.0 if half else -1.0, 1.0, values.size)
    return 0.4 * width * scale * spread


def _draw_values(axes, kind: str, samples, curves, positions, widths,
                 horizontal: bool, half: bool, colors, connect: bool,
                 order: Sequence[int], rng) -> None:
    """Draw the values themselves, and the lines connecting them.

    With *connect* the same values of every data set are drawn, so that the
    lines run through the drawn points.
    """
    count = len(samples)
    common = None
    if connect:
        # Pick among the positions with a value in any data set: the NaN
        # background of a volume would otherwise take most of the picks
        valid = np.zeros(np.asarray(samples[0]).size, dtype=bool)
        for sample in samples:
            valid |= np.isfinite(np.asarray(sample, dtype=np.float64).ravel())
        common = _subset(np.flatnonzero(valid), rng)

    across_all: List[np.ndarray] = []
    along_all: List[np.ndarray] = []
    for i in range(count):
        raw = np.asarray(samples[i], dtype=np.float64).ravel()
        chosen = common if connect else _subset(np.flatnonzero(np.isfinite(raw)), rng)
        values = raw[chosen]
        offsets = (_jitter(values, curves[i], widths[i], half, rng)
                   if kind == "jitter" else np.zeros(values.size))
        across = positions[i] + offsets
        across_all.append(across)
        along_all.append(values)
        if not values.size:
            continue

        dark = _shade(colors[i])
        rasterized = values.size > _RASTERIZE_ABOVE
        if kind in ("strip", "jitter"):
            # Large dots for a few values, small ones for a dense cloud
            size = float(np.clip(12.0 / values.size ** 0.25, 1.5, 5.0))
            axes.plot(*_xy(across, values, horizontal), linestyle="none",
                      marker="o", markersize=size, markeredgewidth=0,
                      color=dark, alpha=0.7, zorder=3, rasterized=rasterized)
        elif kind == "barcode":
            reach = 0.05 * widths[i]
            start = positions[i] if half else positions[i] - reach
            end = positions[i] + (2 if half else 1) * reach
            draw = axes.vlines if horizontal else axes.hlines
            draw(values[np.isfinite(values)], start, end, colors=[dark],
                 linewidth=0.6, alpha=0.5, zorder=1.5, rasterized=rasterized)

    if connect and count > 1:
        from matplotlib.collections import LineCollection

        # One line per value, through the boxes in the order they are drawn;
        # a missing value leaves a gap
        across = np.stack([across_all[i] for i in order])
        along = np.stack([along_all[i] for i in order])
        x, y = _xy(across, along, horizontal)
        lines = np.stack([x.T, y.T], axis=-1)
        alpha = max(0.05, min(0.4, 20.0 / max(lines.shape[0], 1)))
        axes.add_collection(LineCollection(
            lines, colors=[(0.0, 0.0, 0.5, alpha)], linewidths=0.8, zorder=2.8,
            rasterized=lines.shape[0] > _RASTERIZE_ABOVE))


def _draw_outliers(axes, samples, stats, positions, horizontal: bool,
                   window: Optional[Tuple[float, float]]) -> None:
    """Mark the mild outliers with ``+`` and the extreme ones with ``o``.

    Within the value range *window* -- ``(low, high)``, open where infinite
    -- they are kept at its edge instead of vanishing beyond it, as in the
    MATLAB original.
    """
    size = max(3.0, min(6.0, 60.0 / len(stats)))
    for marker, groups in (("+", ("mild_low", "mild_high")),
                           ("o", ("extreme_low", "extreme_high"))):
        across: List[np.ndarray] = []
        along: List[np.ndarray] = []
        for i, entry in enumerate(stats):
            index = np.concatenate([getattr(entry, group) for group in groups])
            if index.size:
                raw = np.asarray(samples[i]).ravel()
                along.append(raw[index].astype(np.float64))
                across.append(np.full(index.size, positions[i]))
        if not along:
            continue
        values = np.concatenate(along)
        if window is not None:
            values = np.clip(values, *window)
        axes.plot(*_xy(np.concatenate(across), values, horizontal),
                  linestyle="none", marker=marker, markersize=size,
                  markerfacecolor="none", markeredgecolor="0.15",
                  markeredgewidth=0.8, zorder=4,
                  rasterized=values.size > _RASTERIZE_ABOVE)


#: Share of the figure the tick labels may take; longer labels are cut in
#: the middle.  File names of BIDS data easily fill half of it.
_LABEL_ROOM = 0.3


def _fit_tick_labels(axes, horizontal: bool, labels: Sequence[str]) -> None:
    """Set the box labels so that many or long names do not run together.

    A font larger than the spacing of the boxes is shrunk.  Side by side,
    labels wider than their box are turned by 45 degrees.  Labels that would
    still take more than a third of the figure are cut in the middle, unless
    that would make two of them equal.
    """
    figure = axes.figure
    box = axes.get_position()
    extent = (box.height * figure.get_figheight() if horizontal
              else box.width * figure.get_figwidth())
    spacing = 72.0 * extent / max(len(labels), 1)
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties

    # rcParams may name the size ("medium") instead of giving points
    size = mpl.rcParams["ytick.labelsize" if horizontal else "xtick.labelsize"]
    fontsize = min(FontProperties(size=size).get_size_in_points(),
                   max(4.0, 0.9 * spacing))
    # Width of an average character, in points
    char = 0.6 * fontsize
    longest = max(len(label) for label in labels)
    rotate = not horizontal and char * longest > 0.9 * spacing
    if horizontal:
        limit = _LABEL_ROOM * 72.0 * figure.get_figwidth() / char
    elif rotate:
        limit = _LABEL_ROOM * 72.0 * figure.get_figheight() / (char * np.sqrt(0.5))
    else:
        limit = float("inf")
    shown = list(labels)
    if longest > limit:
        cut = [_ellipsize(label, max(12, int(limit))) for label in labels]
        if len(set(cut)) == len(set(labels)):
            shown = cut

    if horizontal:
        axes.set_yticklabels(shown, fontsize=fontsize)
    else:
        axes.set_xticklabels(shown, fontsize=fontsize)
    if rotate:
        for label in axes.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")


# ----------------------------------------------------------------------
# The plot
# ----------------------------------------------------------------------


def plot_boxplot(
    data: DataLike,
    fig=None,
    *,
    style: str = "box",
    names: Optional[Sequence[str]] = None,
    color: Union[str, np.ndarray] = DEFAULT_PALETTE,
    horizontal: Optional[bool] = None,
    notch: bool = False,
    whisker: float = 1.5,
    outliers: Optional[bool] = None,
    show_data: Optional[str] = None,
    sort: str = "none",
    width: Optional[float] = None,
    varwidth: bool = False,
    groupnum: bool = False,
    fill: bool = True,
    alpha: float = 0.5,
    grid: bool = True,
    ylim: Optional[Sequence[float]] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    connect: bool = False,
    columns: bool = False,
    winsize: Sequence[float] = (750, 500),
    verbose: bool = True,
    list_outliers: bool = False,
) -> BoxplotResult:
    """Draw box or violin plots of one or more data sets.

    Every argument after *fig* has to be given by name.

    Args:
        data: File name, list of file names, vector or list of numbers (one
            data set), matrix or pandas DataFrame with one column per data
            set as in ``cat_plot_boxplot``, list of arrays, or a mapping of
            names to arrays.
        fig: Figure or figure number to draw into; a new figure when ``None``.
        style: One of :data:`STYLES`: ``"box"``, ``"violin"``,
            ``"violinbox"``, ``"density"`` or ``"jitter"`` -- ``opt.style``
            0 to 4.  The options below that default to ``None`` take the
            style's setting.
        names: Box labels, one per data set; derived from the file names or
            numbered when ``None``.
        color: Name of a palette in :data:`~t1prep.plot_histogram.PALETTES`,
            or an ``(n, 3)`` array of colours; a single row is used for all.
            Each colour stays with its data set when the boxes are sorted.
        horizontal: Stack the boxes from top to bottom, the first on top,
            instead of left to right.
        notch: Notch the boxes at the confidence interval of the median.
        whisker: Reach of the whiskers in interquartile ranges, see
            :func:`box_stats`.
        outliers: Mark the mild outliers with ``+`` and the extreme ones
            with ``o``; by default only the box styles do.
        show_data: Draw the values themselves: ``"none"``, ``"strip"``
            (points on the centre line), ``"jitter"`` (points spread
            sideways by their density) or ``"barcode"`` (a short line each).
            At most 5000 random values per box are drawn.
        sort: ``"none"``, ``"ascending"`` or ``"descending"`` by the median.
        width: Width of a box, or of a violin at its widest, in units of the
            spacing between boxes.
        varwidth: Scale the widths with the square root of the number of
            values, as R's ``boxplot(varwidth=TRUE)``.
        groupnum: Add the number of values to every label.
        fill: Fill the boxes and violins.
        alpha: Opacity of the fill.
        grid: Draw grid lines along the value axis.
        ylim: ``(min, max)`` of the value axis -- the horizontal axis when
            the boxes are horizontal -- as ``opt.ylim``; an end given as NaN
            or +/-Inf is left to the data.  Outliers beyond it are drawn at
            its edge, and a violin is scaled to its widest point within it,
            so that cutting off a dominant peak brings out the rest.
        ylabel: Label of the value axis.
        title: Title of the plot.
        connect: Connect the n-th value of every data set with a line, such
            as one subject over several time points.  Needs data sets of
            equal size; *show_data* then defaults to ``"strip"`` where the
            style draws no values.
        columns: Make every column of a text file a data set of its own.
        winsize: Figure size in pixels.
        verbose: Print the summary table.
        list_outliers: Print every outlier with its 1-based position among
            the values of its data set -- the n-th value of a one-column
            text file -- or, for a volume, its voxel coordinates counted
            from 1.

    Returns:
        A :class:`BoxplotResult` with the statistics and the figure.
    """
    import matplotlib.pyplot as plt

    try:
        preset = STYLES[style]
    except KeyError:
        raise ValueError(
            f"unknown style '{style}'; choose one of {', '.join(STYLES)}"
        ) from None
    if show_data is None:
        # Connected values are drawn as points, as in the MATLAB original
        show_data = ("strip" if connect and preset.show_data == "none"
                     else preset.show_data)
    if show_data not in SHOW_DATA:
        raise ValueError(f"show_data has to be one of {', '.join(SHOW_DATA)}, "
                         f"not '{show_data}'")
    if sort not in SORTS:
        raise ValueError(f"sort has to be one of {', '.join(SORTS)}, not '{sort}'")
    horizontal = preset.horizontal if horizontal is None else bool(horizontal)
    outliers = preset.outliers if outliers is None else bool(outliers)
    width = preset.width if width is None else float(width)
    if not width > 0:
        raise ValueError(f"width has to be positive, not {width:g}")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha has to lie in [0, 1], not {alpha:g}")
    if ylim is not None and len(ylim) != 2:
        raise ValueError("ylim needs exactly two entries")
    # The value range for clipping the outliers and windowing the violins,
    # open where an end is not finite
    window = None if ylim is None else _value_range(ylim)

    # -- read every data set ------------------------------------------------
    inputs, is_files, carried = _normalise_inputs(data)
    if is_files:
        samples, labels = _read_files(inputs, columns)
    else:
        samples = inputs
        labels = carried or [str(i + 1) for i in range(len(samples))]
    count = len(samples)
    if names is not None:
        if isinstance(names, str):
            names = [names]
        labels = [str(name) for name in names]
        if len(labels) != count:
            raise ValueError(f"names has {len(labels)} entries for {count} boxes")
    if connect and len({np.asarray(s).size for s in samples}) > 1:
        sizes = ", ".join(str(np.asarray(s).size) for s in samples)
        raise ValueError(f"connecting the values needs data sets of equal "
                         f"size, not {sizes}")

    stats = [box_stats(sample, whisker, label)
             for sample, label in zip(samples, labels)]
    if not any(entry.n for entry in stats):
        raise ValueError("no finite values in any data set")

    # -- colours, by data set, so that they stay with it when sorted ------
    if isinstance(color, str):
        colors = colormap(color, count)
    else:
        colors = np.atleast_2d(np.asarray(color, dtype=float))
        if len(colors) == 1:
            colors = np.repeat(colors, count, axis=0)
        if len(colors) < count:
            raise ValueError(f"color has {len(colors)} rows for {count} boxes")
    colors = [tuple(float(c) for c in row[:3]) for row in colors[:count]]

    # -- where every box goes ------------------------------------------------
    order = list(range(count))
    if sort != "none":
        medians = np.array([entry.median for entry in stats])
        key = medians if sort == "ascending" else -medians
        # Empty data sets have a NaN median and go last either way
        order = [int(i) for i in np.argsort(key, kind="stable")]
    slots = np.arange(1, count + 1, dtype=float)
    if horizontal:
        # The first box on top, as in the MATLAB original
        slots = slots[::-1]
    positions = np.empty(count)
    positions[order] = slots
    sizes = np.array([entry.n for entry in stats], dtype=float)
    widths = (width * np.sqrt(sizes / sizes.max()) if varwidth
              else np.full(count, width))

    # -- draw ----------------------------------------------------------------
    need_density = preset.violin or show_data == "jitter"
    curves = [_density(_finite(sample), window) if need_density else None
              for sample in samples]
    rng = np.random.default_rng(0)
    figure = _figure(plt, fig, winsize)
    axes = figure.gca()

    if preset.violin:
        _draw_violins(axes, curves, positions, widths, horizontal,
                      preset.half, colors, alpha, fill)
        # Without a box the line shows the median of every violin; with one,
        # it still marks a set too small or too uniform for a violin, whose
        # flat box the white median would hide
        _draw_violin_medians(axes, stats, curves, positions, widths,
                             horizontal, preset.half, colors,
                             flat_only=preset.box != "none")
    if preset.box != "none":
        _draw_boxes(axes, stats, positions, widths, horizontal, colors, alpha,
                    fill, notch, thin=preset.box == "thin")
    if show_data != "none" or connect:
        _draw_values(axes, show_data, samples, curves, positions, widths,
                     horizontal, preset.half, colors, connect, order, rng)
    if outliers:
        _draw_outliers(axes, samples, stats, positions, horizontal, window)

    # -- axes ----------------------------------------------------------------
    ticks = list(labels)
    if groupnum:
        ticks = [f"{label} (n={entry.n})" for label, entry in zip(ticks, stats)]
    by_position = np.argsort(positions)
    tick_positions = positions[by_position]
    if preset.half:
        # Room for the half violin above the top baseline, and for the
        # values on the lowest one
        limits = (positions.min() - 0.3,
                  positions.max() + 0.5 * float(np.max(widths)) + 0.1)
    else:
        limits = (positions.min() - 0.5, positions.max() + 0.5)
    # An open end of ylim is left to matplotlib's autoscaling
    value_limits = (None if ylim is None else
                    [float(v) if np.isfinite(v) else None for v in ylim])
    if horizontal:
        axes.set_yticks(tick_positions)
        axes.set_ylim(*limits)
        axes.tick_params(axis="y", length=0)
        if value_limits is not None:
            axes.set_xlim(*value_limits)
        if ylabel:
            axes.set_xlabel(ylabel)
    else:
        axes.set_xticks(tick_positions)
        axes.set_xlim(*limits)
        axes.tick_params(axis="x", length=0)
        if value_limits is not None:
            axes.set_ylim(*value_limits)
        if ylabel:
            axes.set_ylabel(ylabel)
    if grid:
        axes.set_axisbelow(True)
        axes.grid(True, axis="x" if horizontal else "y", alpha=0.4)
    if title:
        axes.set_title(title)
    _fit_tick_labels(axes, horizontal, [ticks[i] for i in by_position])
    if fig is None or isinstance(fig, (int, np.integer)):
        figure.tight_layout()

    if verbose:
        _print_table(stats, order)
    if list_outliers:
        _print_outliers(samples, stats, order)

    return BoxplotResult(stats=stats, order=order, positions=positions,
                         labels=ticks, figure=figure)


def _print_table(stats: Sequence[BoxStats], order: Sequence[int]) -> None:
    """Print the statistics of every box, in the order drawn."""
    header = ("name", "n", "mean", "std", "min", "Q1", "median", "Q3", "max",
              "outliers")
    rows = [stats[i] for i in order]
    pad = max(len(header[0]), *(len(entry.name) for entry in rows))
    print()
    print(f"{header[0]:>{pad}}\t" + " ".join(f"{h:>10}" for h in header[1:]))
    for entry in rows:
        numbers = (entry.mean, entry.std, entry.min, entry.q1, entry.median,
                   entry.q3, entry.max)
        print(f"{entry.name:>{pad}}\t{entry.n:>10d} "
              + " ".join(f"{f'{v:8g}':>10}" for v in numbers)
              + f" {entry.outliers.size:>10d}")


def _print_outliers(samples, stats: Sequence[BoxStats],
                    order: Sequence[int]) -> None:
    """Print every outlier with its position and its kind.

    The position counts from 1: the n-th value of a vector, or the voxel
    coordinates ``i,j,k`` of a volume, as SPM shows them.
    """
    kinds = (("mild_low", "mild, low"), ("mild_high", "mild, high"),
             ("extreme_low", "extreme, low"), ("extreme_high", "extreme, high"))
    rows: List[Tuple[str, str, float, str]] = []
    for i in order:
        shape = np.shape(samples[i])
        raw = np.asarray(samples[i]).ravel()
        found = [(int(k), name) for group, name in kinds
                 for k in getattr(stats[i], group)]
        for k, name in sorted(found):
            if len(shape) > 1:
                where = ",".join(str(c + 1) for c in np.unravel_index(k, shape))
            else:
                where = str(k + 1)
            rows.append((stats[i].name, where, float(raw[k]), name))
    print()
    if not rows:
        print("No outliers.")
        return
    pad = max(len("name"), *(len(row[0]) for row in rows))
    span = max(len("position"), *(len(row[1]) for row in rows))
    print(f"{'name':>{pad}}\t{'position':>{span}} {'value':>10}  outlier")
    for name, where, value, kind in rows:
        print(f"{name:>{pad}}\t{where:>{span}} {f'{value:8g}':>10}  {kind}")


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

_DESCRIPTION = """\
Box and violin plots of one or more volume, surface or text data sets.

Python counterpart of CAT12's cat_plot_boxplot.m.  Reads the same files as
CAT_PlotHistogram -- NIfTI volumes, GIFTI surfaces, FreeSurfer morphometry
data and plain text -- and draws one box or violin per file, or per column
of a text file with --columns.  Volumes and GIFTI files lose their zero
background; FreeSurfer data and text files keep their zeros.

styles (opt.style of cat_plot_boxplot):
  box        0  box plot
  violin     1  violin plot, the values drawn as a barcode
  violinbox  2  violin plot with a thin box plot inside (without the barcode)
  density    3  horizontal half violins, the values along the baseline
  jitter     4  box plot with the values as points, spread by their density

Whiskers reach the last value within --whisker times the interquartile
range (IQR) of the box; mild outliers (+) lie up to twice that far from the
box, extreme outliers (o) beyond.  Quartiles follow cat_stat_prctile."""

_EPILOG = """\
examples:
  CAT_PlotBoxplot mwp1*.nii
  CAT_PlotBoxplot --style violinbox --sort ascending lh.thickness.*
  CAT_GrepJson --field subjectmeasures.vol_TIV --out tiv.txt report/log_*.json
  CAT_PlotBoxplot --style jitter --list-outliers tiv.txt
  CAT_PlotBoxplot --columns --names CSF,GM,WM cgw.txt"""


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    # No abbreviations: CAT_GrepJson's --out, two lines up in the same
    # workflow, would otherwise be taken for --outliers
    parser = ArgumentParser(prog="CAT_PlotBoxplot", description=_DESCRIPTION,
                            epilog=_EPILOG, allow_abbrev=False)
    parser.add_argument("files", nargs="*", metavar="FILE",
                        help="volumes, surfaces or text files, one box each")
    parser.add_argument("--style", default="box", choices=tuple(STYLES),
                        help="predefined style, see above")
    parser.add_argument("--horizontal", dest="horizontal", action="store_const",
                        const=True, default=None,
                        help="stack the boxes from top to bottom, the first on "
                             "top (default of the density style)")
    parser.add_argument("--vertical", dest="horizontal", action="store_const",
                        const=False,
                        help="place the boxes side by side (default of the "
                             "other styles)")
    parser.add_argument("--show-data", choices=SHOW_DATA, default=None,
                        help="draw the values themselves: none, strip (points "
                             "on the centre line), jitter (points spread by "
                             "their density) or barcode (a short line each); "
                             "default: the style's")
    parser.add_argument("--notch", action="store_true",
                        help="notch the boxes at the confidence interval of "
                             "the median (box and jitter styles)")
    parser.add_argument("--whisker", type=float, default=1.5, metavar="K",
                        help="whiskers reach the last value within K IQR of "
                             "the box; 0 makes every value outside the box an "
                             "outlier")
    parser.add_argument("--outliers", dest="outliers", action="store_const",
                        const=True, default=None,
                        help="mark the outliers (default of the box and "
                             "jitter styles)")
    parser.add_argument("--no-outliers", dest="outliers", action="store_const",
                        const=False,
                        help="do not mark them (default of the violin styles)")
    parser.add_argument("--list-outliers", action="store_true",
                        help="print every outlier with its 1-based position "
                             "among the values of its box (the n-th value of "
                             "a one-column text file), or its voxel "
                             "coordinates for a volume")
    parser.add_argument("--sort", choices=SORTS, default="none",
                        help="order the boxes by their median")
    parser.add_argument("--names", metavar="A,B,...",
                        help="comma-separated box labels (default: derived "
                             "from the file names)")
    parser.add_argument("--groupnum", action="store_true",
                        help="add the number of values to every label")
    parser.add_argument("--columns", action="store_true",
                        help="one box per column of a text file instead of "
                             "one per file")
    parser.add_argument("--connect", action="store_true",
                        help="connect the n-th value of every box, e.g. a "
                             "subject over time points; needs data sets of "
                             "equal size")
    parser.add_argument("--width", type=float, metavar="W",
                        help="width of a box, or of a violin at its widest, "
                             "in box spacings (default: the style's)")
    parser.add_argument("--varwidth", action="store_true",
                        help="scale the widths with the square root of the "
                             "number of values")
    parser.add_argument("--no-fill", dest="fill", action="store_false",
                        help="draw outlines only")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="opacity of the fill, 0 to 1")
    parser.add_argument("--no-grid", dest="grid", action="store_false",
                        help="no grid lines along the value axis")
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="range of the value axis (the horizontal one "
                             "when the boxes are horizontal); nan leaves an "
                             "end to the data. Outliers beyond it are drawn "
                             "at its edge, violins are scaled within it")
    parser.add_argument("--ylabel", metavar="TEXT",
                        help="label of the value axis")
    parser.add_argument("--title", metavar="TEXT", help="title of the plot")
    parser.add_argument("--color", default=DEFAULT_PALETTE, metavar="NAME",
                        choices=sorted(PALETTES),
                        help="categorical palette: " + ", ".join(sorted(PALETTES)))
    parser.add_argument("--winsize", type=int, nargs=2, default=[750, 500],
                        metavar=("W", "H"), help="figure size in pixels")
    parser.add_argument("--fig", type=int, metavar="N",
                        help="number of the figure to draw into")
    parser.add_argument("--save", metavar="FILE",
                        help="write the figure to FILE instead of showing it; "
                             "the extension picks the format, .png if it has "
                             "none")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                        help="do not print the summary table")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of the command-line tool."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    parser.exit_without_arguments(argv)
    args = parser.parse_args(argv)
    if not args.files:
        parser.error("no input files specified")

    try:
        import matplotlib
    except ImportError:
        print(f"{parser.prog} needs matplotlib: pip install matplotlib",
              file=sys.stderr)
        return 1
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = None
    if args.names is not None:
        names = [name.strip() for name in args.names.split(",")]
    missing = [path for path in args.files if not os.path.isfile(path)]
    if missing:
        message = f"no such file: {missing[0]}"
        if args.names is not None and "," not in args.names:
            # "--names CSF GM WM file" leaves GM and WM as input files
            message += ("; --names takes one comma-separated list, "
                        "e.g. --names CSF,GM,WM")
        print(f"{parser.prog}: error: {message}", file=sys.stderr)
        return 1
    try:
        result = plot_boxplot(
            args.files, fig=args.fig, style=args.style, names=names,
            color=args.color, horizontal=args.horizontal, notch=args.notch,
            whisker=args.whisker, outliers=args.outliers,
            show_data=args.show_data, sort=args.sort, width=args.width,
            varwidth=args.varwidth, groupnum=args.groupnum, fill=args.fill,
            alpha=args.alpha, grid=args.grid, ylim=args.ylim,
            ylabel=args.ylabel, title=args.title, connect=args.connect,
            columns=args.columns, winsize=args.winsize, verbose=args.verbose,
            list_outliers=args.list_outliers,
        )
    except (OSError, ValueError) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 1

    if args.save:
        name = _figure_file(args.save, result.figure)
        try:
            result.figure.savefig(name, dpi=100, bbox_inches="tight")
        except (OSError, ValueError) as exc:
            print(f"{parser.prog}: error: cannot save {name}: {exc}",
                  file=sys.stderr)
            return 1
        print(f"saved {name}")
    else:
        plt.show()
    return 0


def _figure_file(name: str, figure) -> str:
    """Return *name*, with ``.png`` added unless it ends in a known format.

    A dot in a name such as ``qc_v1.2`` is not taken for an extension.
    """
    ext = os.path.splitext(name)[1].lstrip(".").lower()
    if ext and ext in figure.canvas.get_supported_filetypes():
        return name
    return name + ".png"


if __name__ == "__main__":
    raise SystemExit(main())
