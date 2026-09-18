#!/usr/bin/env python3
"""Histogram of one or more volume, surface or text data sets.

Python counterpart of CAT12's ``cat_plot_histogram.m``.  Every input is
reduced to a one-dimensional sample, binned on a grid shared by all inputs,
and drawn as a line — one line per input — so that distributions can be
compared directly.  By default a smooth curve is fitted through the sample
(a Gaussian kernel density estimate) and the raw histogram is added as a
dotted line of the same colour.

For ``spmT*`` maps, and for files whose name starts with ``D`` (effect size
maps), the printed table adds the upper 5% tail cutoff (``TH5``) and the
legend is labelled with it.

Usage::

    # one or more volumes, surfaces or text files
    CAT_PlotHistogram p1*.nii.gz

    # raw histograms without a fitted curve, fixed range, saved to disk
    CAT_PlotHistogram --dist none --xrange 0 1 --save /tmp/hist lh.thickness.*

    # average histogram with its standard error in a second figure
    CAT_PlotHistogram --mean p1*.nii.gz

    # from a source checkout, without the entry point on PATH
    python -m t1prep.plot_histogram p1*.nii.gz

The same thing from Python, where it returns the binned data, the statistics
and the figures instead of only drawing them::

    from t1prep.plot_histogram import plot_histogram

    result = plot_histogram(["p1a.nii.gz", "p1b.nii.gz"], dist="kernel")
    print([(s.name, s.mean, s.std) for s in result.stats])

Two inputs of identical shape additionally get a density scatter plot and,
for volumes, the three orthogonal projections of their difference — as the
MATLAB original does.  The surface rendering of a difference between two
GIFTI files is not reproduced here; use ``CAT_SurfView`` for that.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .cli_help import ArgumentParser


# ----------------------------------------------------------------------
# Categorical colormaps (cat_io_colormaps.m)
# ----------------------------------------------------------------------

#: The categorical palettes of ``cat_io_colormaps``, as hex strings.  ggsci
#: (https://nanx.me/ggsci) for nejm/jco/jama/d3, RColorBrewer for the rest.
PALETTES: Dict[str, Tuple[str, ...]] = {
    "nejm": ("#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD",
             "#FFDC91", "#EE4C97", "#8C564B", "#BCBD22", "#00A1D5", "#374E55",
             "#003C67", "#8F7700", "#7F7F7F", "#353535"),
    "jco": ("#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67",
            "#8F7700", "#3B3B3B", "#A73030", "#4A6990"),
    "jama": ("#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", "#6A6599",
             "#80796B"),
    "d3": ("#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B",
           "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"),
    "accent": ("#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F",
               "#BF5B17", "#666666"),
    "dark2": ("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#669B1E", "#E6AB02",
              "#A6761D", "#666666"),
    "paired": ("#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
               "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928"),
    "set1": ("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
             "#A65628", "#F781BF", "#999999"),
    "set2": ("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F",
             "#E5C494", "#B3B3B3"),
    "set3": ("#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462",
             "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F"),
}

DEFAULT_PALETTE = "nejm"


def colormap(name: str = DEFAULT_PALETTE, ncolors: Optional[int] = None) -> np.ndarray:
    """Return *ncolors* RGB rows of the categorical palette *name*.

    Mirrors ``cat_io_colormaps``: a categorical palette is truncated when
    fewer colours are asked for than it holds, and linearly interpolated when
    more are needed.

    Args:
        name: Key of :data:`PALETTES`.
        ncolors: Number of colours; the full palette when ``None``.

    Returns:
        Array of shape ``(ncolors, 3)`` with values in ``[0, 1]``.
    """
    try:
        hexes = PALETTES[name]
    except KeyError:
        raise ValueError(
            f"unknown colormap '{name}'; choose one of {', '.join(sorted(PALETTES))}"
        ) from None
    base = np.array([[int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for h in hexes])
    if ncolors is None or ncolors == len(base):
        return base
    if ncolors < 1:
        raise ValueError("need at least one colour")
    if ncolors < len(base):
        return base[:ncolors]
    # More colours than the palette holds: interpolate between its entries
    src = np.arange(len(base))
    dst = np.linspace(0, len(base) - 1, ncolors)
    return np.stack([np.interp(dst, src, base[:, c]) for c in range(3)], axis=1)


# ----------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------

#: Fraction of zero entries above which zeros are treated as background and
#: dropped (``loadsingle`` in the MATLAB original).
_ZERO_BACKGROUND = 0.01


def _drop_zero_background(data: np.ndarray) -> np.ndarray:
    """Replace zeros by NaN when they make up more than 1% of the data.

    Volume and surface files carry a zero background that would otherwise
    dominate the first bin; a handful of genuine zeros is kept.
    """
    zeros = data == 0
    if zeros.any() and zeros.sum() > _ZERO_BACKGROUND * data.size:
        data = data.astype(np.float32, copy=True)
        data[zeros] = np.nan
    return data


def _finite(data: np.ndarray) -> np.ndarray:
    """Return the finite entries of *data* as a flat float64 array."""
    flat = np.asarray(data, dtype=np.float64).ravel()
    return flat[np.isfinite(flat)]


def _classify(path: str) -> int:
    """Return the file type code used by the MATLAB original.

    ``0`` nii.gz, ``1`` text, ``2`` volume, ``3`` GIFTI, ``4`` FreeSurfer.
    """
    lower = path.lower()
    if lower.endswith(".nii.gz"):
        return 0
    if lower.endswith(".txt"):
        return 1
    if lower.endswith((".nii", ".img", ".hdr")):
        return 2
    if lower.endswith(".gii"):
        return 3
    return 4


def _load_gifti(path: str) -> np.ndarray:
    """Return the first scalar data array of a GIFTI file."""
    import nibabel as nib

    image = nib.load(path)
    arrays = []
    for darray in image.darrays:
        # Skip the geometry arrays (NIFTI_INTENT_POINTSET / TRIANGLE)
        code = getattr(darray, "intent", getattr(darray, "intent_code", -1))
        if int(code or -1) in (1008, 1009):
            continue
        arrays.append(np.asarray(darray.data).ravel())
    if not arrays:
        raise ValueError(f"no scalar data array in {path}")
    return np.concatenate(arrays) if len(arrays) > 1 else arrays[0]


def _load_freesurfer(path: str) -> np.ndarray:
    """Return FreeSurfer morphometry data (thickness, curv, sulc, ...)."""
    from nibabel.freesurfer.io import read_morph_data

    return np.asarray(read_morph_data(path))


def _load_text(path: str) -> np.ndarray:
    """Return the numbers of a whitespace-separated text file."""
    values: List[float] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            for token in line.split():
                try:
                    values.append(float(token))
                except ValueError:
                    # Header words and comments are simply skipped
                    continue
    if not values:
        raise ValueError(f"no numbers found in {path}")
    return np.array(values, dtype=np.float64)


def load_data(path: str) -> Tuple[np.ndarray, int]:
    """Load one input file and return ``(data, filetype)``.

    Volumes, GIFTI surfaces and text files lose their zero background;
    FreeSurfer morphometry data keeps its zeros, where zero is a value like
    any other.

    Args:
        path: File to read.

    Returns:
        The data as a float32 array (NaN where the background was dropped)
        and its file type code.
    """
    filetype = _classify(path)
    if filetype in (0, 2):
        import nibabel as nib

        data = np.asanyarray(nib.load(path).dataobj, dtype=np.float32)
    elif filetype == 1:
        data = _load_text(path)
    elif filetype == 3:
        data = _load_gifti(path)
    else:
        try:
            data = _load_freesurfer(path)
        except Exception as exc:
            raise ValueError(f"unknown data format: {path}") from exc
        return np.asarray(data, dtype=np.float32), filetype
    return _drop_zero_background(np.asarray(data, dtype=np.float32)), filetype


# ----------------------------------------------------------------------
# Binning and curve fitting
# ----------------------------------------------------------------------


def hist_centers(sample: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Bin *sample* into bins described by their *centers*.

    Reproduces MATLAB's ``hist(y, centers)``: the bin borders are the
    midpoints between neighbouring centers, and the two outer bins are open,
    so every value is counted.  That is why the first and the last bin are
    dropped again before plotting — they hold everything outside the range.

    Args:
        sample: Values to bin.
        centers: Monotonically increasing bin centers.

    Returns:
        Counts, one per center.
    """
    borders = 0.5 * (centers[:-1] + centers[1:])
    index = np.searchsorted(borders, sample, side="right")
    return np.bincount(index, minlength=len(centers)).astype(np.float64)


#: ``--dist`` names mapped to their :mod:`scipy.stats` distribution, with a
#: flag saying whether the location is pinned to zero.  MATLAB's ``fitdist``
#: has no shift parameter for the distributions with positive support, so
#: pinning it keeps the fits comparable.
_DISTRIBUTIONS: Dict[str, Tuple[str, bool]] = {
    "normal": ("norm", False),
    "logistic": ("logistic", False),
    "tlocationscale": ("t", False),
    "gamma": ("gamma", True),
    "rician": ("rice", True),
    "rayleigh": ("rayleigh", True),
    "weibull": ("weibull_min", True),
    "lognormal": ("lognorm", True),
    "exponential": ("expon", True),
    "beta": ("beta", True),
}

#: Everything ``--dist`` accepts.  ``kernel`` is the default and needs no
#: parametric family; ``none`` draws the plain histogram instead of a curve.
DISTRIBUTIONS: Tuple[str, ...] = ("kernel", "none") + tuple(sorted(_DISTRIBUTIONS))


#: Above this many values a kernel density estimate is computed on a thinned
#: sample -- see :func:`fit_density`.
_KDE_MAX_POINTS = 20000


def fit_density(sample: np.ndarray, dist: str, x: np.ndarray) -> np.ndarray:
    """Evaluate the density of *dist* fitted to *sample* at the points *x*.

    Args:
        sample: The values to fit.
        dist: ``"kernel"`` for a Gaussian kernel density estimate, or one of
            the parametric families in :data:`DISTRIBUTIONS`.
        x: Points at which the fitted density is evaluated.

    Returns:
        The probability density at *x*.
    """
    from scipy import stats

    if dist == "kernel":
        # A KDE over millions of voxels evaluates in O(n * len(x)); the
        # estimate itself is stable long before that, so the sample is
        # thinned first.  The bandwidth still follows Scott's rule on the
        # full count, which is what MATLAB's kernel fitdist uses as well.
        values = sample
        if values.size > _KDE_MAX_POINTS:
            step = int(np.ceil(values.size / _KDE_MAX_POINTS))
            values = values[::step]
        kde = stats.gaussian_kde(values, bw_method=sample.size ** (-1.0 / 5.0))
        return kde(x)

    try:
        name, pin_loc = _DISTRIBUTIONS[dist]
    except KeyError:
        raise ValueError(
            f"unknown distribution '{dist}'; choose one of {', '.join(DISTRIBUTIONS)}"
        ) from None
    family = getattr(stats, name)
    if pin_loc:
        if sample.min() <= 0:
            raise ValueError(
                f"the '{dist}' distribution is only defined for positive data, "
                f"but the sample starts at {sample.min():g}"
            )
        params = family.fit(sample, floc=0)
    else:
        params = family.fit(sample)
    return family.pdf(x, *params)


# ----------------------------------------------------------------------
# Legend labels
# ----------------------------------------------------------------------


def shorten_names(paths: Sequence[str]) -> List[str]:
    """Strip the part every path has in common, as ``spm_str_manip(...,'C')``.

    With one path, or when the paths differ only in their directory, there is
    no informative middle part left; the file name is used instead.

    Args:
        paths: The input file names.

    Returns:
        One short label per path.
    """
    if len(paths) == 1:
        return [os.path.basename(paths[0])]
    head = os.path.commonprefix(paths)
    tail = os.path.commonprefix([p[::-1] for p in paths])[::-1]
    middles = [p[len(head):len(p) - len(tail)] for p in paths]
    if all(m.strip(os.sep + "._-") for m in middles):
        return [m.strip(os.sep) or os.path.basename(p) for m, p in zip(middles, paths)]
    return [os.path.basename(p) for p in paths]


def _is_statistic_map(path: str) -> bool:
    """True for SPM T maps and effect size maps, which get a tail cutoff.

    Follows the MATLAB original: the name contains ``spmT``, or it starts
    with ``D`` as CAT12's effect size maps do.
    """
    name = os.path.basename(path)
    for suffix in (".nii.gz", ".nii", ".img", ".hdr", ".gii", ".txt"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return "spmT" in name or name.startswith("D")


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------


@dataclass
class HistogramStats:
    """Summary of one input, the counterpart of the MATLAB ``out2`` struct."""

    name: str
    mean: float
    median: float
    std: float
    effect_size: float
    #: Peak of the histogram, the ``maxFreq`` column of the printed table.
    max_freq: float
    #: Upper 5% tail cutoff; only filled for statistic maps.
    th5: Optional[float] = None


@dataclass
class HistogramResult:
    """Everything :func:`plot_histogram` computed, the MATLAB ``out``/``out2``."""

    #: Bin centers of the plotted range (the open outer bins are dropped).
    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Histogram per input, shape ``(n, len(x))``.
    hist: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    #: Fitted curve per input; equal to :attr:`hist` when ``dist="none"``.
    fit: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    #: One :class:`HistogramStats` per input.
    stats: List[HistogramStats] = field(default_factory=list)
    #: The legend labels.
    labels: List[str] = field(default_factory=list)
    #: The figures that were drawn, by role (``"histogram"``, ``"mean"``,
    #: ``"pooled"``, ``"scatter"``, ``"difference"``).
    figures: Dict[str, object] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------

DataLike = Union[str, np.ndarray, Sequence[Union[str, np.ndarray]]]


def _normalise_inputs(data: DataLike) -> Tuple[List[Union[str, np.ndarray]], bool]:
    """Bring the accepted input forms into one list.

    Accepts a file name, a list of file names, a vector, a matrix whose
    smaller dimension counts the data sets, or a list of arrays.

    Returns:
        The list of inputs and whether they are file names.
    """
    if isinstance(data, str):
        return [data], True
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return [data], False
        if data.ndim == 2:
            # As in the MATLAB original the smaller dimension counts the
            # data sets, so both orientations of a matrix work
            if data.shape[0] <= data.shape[1]:
                return [row for row in data], False
            return [col for col in data.T], False
        return [data.ravel()], False
    items = list(data)
    if not items:
        raise ValueError("no input data specified")
    if all(isinstance(item, str) for item in items):
        return items, True
    if any(isinstance(item, str) for item in items):
        raise ValueError("cannot mix file names and arrays in one call")
    return [np.asarray(item) for item in items], False


def _figure(plt, handle, winsize: Sequence[float]):
    """Create or reuse a figure of *winsize* pixels."""
    size = (winsize[0] / 100.0, winsize[1] / 100.0)
    if handle is None:
        return plt.figure(figsize=size, dpi=100)
    if isinstance(handle, (int, np.integer)):
        return plt.figure(num=int(handle), figsize=size, dpi=100)
    return handle


# ----------------------------------------------------------------------
# The plot
# ----------------------------------------------------------------------


def plot_histogram(
    data: DataLike,
    fig=None,
    color: Union[str, np.ndarray] = DEFAULT_PALETTE,
    norm_frequency: bool = True,
    winsize: Sequence[float] = (750, 500),
    xrange: Optional[Sequence[float]] = None,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    dist: str = "kernel",
    mean: bool = False,
    bins: int = 500,
    alpha: float = 2.0,
    rawline: int = 2,
    scatter: bool = True,
    names: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> HistogramResult:
    """Draw the histogram of one or more data sets.

    Args:
        data: File name, list of file names, vector, matrix whose smaller
            dimension counts the data sets, or list of arrays.
        fig: Figure or figure number to draw into; a new figure when ``None``.
        color: Name of a palette in :data:`PALETTES`, or an ``(n, 3)`` /
            ``(n, 4)`` array of colours.
        norm_frequency: Divide every histogram by its own total, so that
            inputs of different size stay comparable.
        winsize: Figure size in pixels.
        xrange: ``(min, max)`` of the binning; the range of the data when
            ``None``.
        xlim: ``(min, max)`` of the drawn x axis.
        ylim: ``(min, max)`` of the drawn y axis.
        dist: Curve fitted through each sample, one of :data:`DISTRIBUTIONS`.
            ``"kernel"`` is a Gaussian kernel density estimate, ``"none"``
            draws the plain histogram and adds a second figure with all
            inputs pooled.
        mean: Also draw the average of all histograms with its standard
            error as a shaded band, in its own figure (number 11).  Repeated
            calls overlay their averages there in the next colour and have
            to be labelled by the caller, because the bands are kept out of
            the legend.
        bins: Upper limit on the number of bins.
        alpha: Opacity of the lines; outside ``[0, 1]`` it is derived from
            the number of inputs.
        rawline: Draw the raw histogram as a dotted line next to a fitted
            curve -- ``0`` never, ``1`` always, ``2`` only for fewer than
            six inputs.
        scatter: For exactly two inputs of equal shape, add the density
            scatter plot and the difference projections.
        names: Legend labels, one per input; derived from the file names or
            numbered when ``None``.
        verbose: Print the summary table.

    Returns:
        A :class:`HistogramResult` with the binned data, the statistics and
        the figures.
    """
    import matplotlib.pyplot as plt

    inputs, is_files = _normalise_inputs(data)
    count = len(inputs)

    # -- read every input, and remember the range it spans --------------
    samples: List[np.ndarray] = []
    lower = np.empty(count)
    upper = np.empty(count)
    filetypes: List[int] = []
    for i, item in enumerate(inputs):
        if is_files:
            values, filetype = load_data(item)
        else:
            # Direct data keeps its zeros: nothing here is a file background
            values, filetype = np.asarray(item, dtype=np.float64), -1
        samples.append(values)
        filetypes.append(filetype)
        finite = _finite(values)
        if finite.size == 0:
            raise ValueError(
                f"{item if is_files else f'data set {i + 1}'} has no finite values"
            )
        lower[i] = finite.min()
        upper[i] = finite.max()

    # -- legend labels --------------------------------------------------
    if names is not None:
        labels = [str(name) for name in names]
        if len(labels) != count:
            raise ValueError(f"names has {len(labels)} entries for {count} inputs")
    elif is_files:
        labels = shorten_names(inputs)
    else:
        labels = [str(i + 1) for i in range(count)]

    # -- colours --------------------------------------------------------
    if isinstance(color, str):
        colors = colormap(color, count)
    else:
        colors = np.atleast_2d(np.asarray(color, dtype=float))
        if len(colors) < count:
            raise ValueError(f"color has {len(colors)} rows for {count} inputs")
        colors = colors[:count]
    if alpha > 1 or alpha < 0:
        # Automatic: the more lines overlap, the more transparent they get
        alpha = max(0.2, 1.0 / max(1, count - 5) ** 0.5)
    if alpha != 1 and colors.shape[1] < 4:
        colors = np.column_stack([colors, np.full(count, alpha)])

    if dist == "none":
        dist = ""

    # -- the shared bin centers ----------------------------------------
    if xrange is None:
        # About one bin per 100 values of the first input, but never fewer
        # than 10 and never more than requested
        first = _finite(samples[0]).size
        npoints = max(min(int(np.floor(first / 100.0 + 0.5)), bins), 10)
        centers = np.linspace(lower.min(), upper.max(), npoints)
    elif len(xrange) == 2:
        centers = np.linspace(xrange[0], xrange[1], bins)
    else:
        raise ValueError("xrange needs exactly two entries")
    if not np.isfinite(centers).all() or centers[-1] <= centers[0]:
        raise ValueError(
            f"cannot bin over the range [{centers[0]:g}, {centers[-1]:g}]"
        )
    width = centers[1] - centers[0]

    figures: Dict[str, object] = {}

    # -- the scatter plot of two matching inputs ------------------------
    if scatter and count == 2:
        figures.update(
            _compare_two(plt, samples, inputs, filetypes, is_files, verbose)
        )

    # Claim the figure of the average before the one of the histograms, so
    # that repeated calls keep overlaying their averages in the same figure
    mean_number = 11
    if mean:
        if isinstance(fig, (int, np.integer)) and int(fig) == mean_number:
            mean_number += 1
        mean_figure = plt.figure(num=mean_number, figsize=(winsize[0] / 100.0,
                                                          winsize[1] / 100.0),
                                 dpi=100)

    figure = _figure(plt, fig, winsize)
    axes = figure.gca()

    # -- bin, fit and summarise every input ----------------------------
    histograms = np.empty((count, len(centers)))
    curves = np.empty((count, len(centers)))
    stats: List[HistogramStats] = []
    pooled: List[np.ndarray] = []
    table: List[Tuple[str, ...]] = []
    statistic_map = is_files and _is_statistic_map(inputs[0])

    for j, sample in enumerate(samples):
        values = _finite(sample)
        counts = hist_centers(values, centers)
        total = counts.sum()
        if dist:
            # Scale the density to the area of the histogram, so that the
            # curve and the counts share one y axis
            curve = values.size * width * fit_density(values, dist, centers)
        else:
            curve = None
            pooled.append(values)

        if norm_frequency and total > 0:
            if curve is not None:
                curve = curve / total
            counts = counts / total
        if curve is None:
            curve = counts.copy()
        histograms[j] = counts
        curves[j] = curve

        average = float(values.mean())
        spread = float(values.std(ddof=1)) if values.size > 1 else 0.0
        effect = average / spread if spread else float("nan")
        entry = HistogramStats(
            name=labels[j], mean=average, median=float(np.median(values)),
            std=spread, effect_size=effect, max_freq=float(counts.max()),
        )
        if statistic_map:
            # Upper 5% tail: the first bin the cumulative histogram passes.
            # counts may already be normalized, so it is summed again here
            area = counts.sum()
            cumulative = np.cumsum(counts) / area if area else np.zeros_like(counts)
            above = np.flatnonzero(cumulative > 0.95)
            entry.th5 = float(centers[above[0]]) if above.size else float("nan")
            labels[j] = f"TH5={entry.th5:.4f} {labels[j]}"
            entry.name = labels[j]
            table.append((labels[j], average, spread, effect, entry.th5))
        else:
            table.append((labels[j], average, entry.median, spread, effect,
                          entry.max_freq))
        stats.append(entry)

    if verbose:
        header = (("file", "mean", "std", "ES", "TH5") if statistic_map
                  else ("file", "mean", "median", "std", "ES", "maxFreq"))
        pad = max(len(row[0]) for row in table)
        print()
        print(f"{header[0]:>{pad}}\t" + " ".join(f"{h:>10}" for h in header[1:]))
        for row in table:
            print(f"{row[0]:>{pad}}\t"
                  + " ".join(f"{f'{v:8g}':>10}" for v in row[1:]))

    # -- draw ------------------------------------------------------------
    inner = slice(1, -1)  # the open outer bins hold everything off-range
    lines = []
    for j in range(count):
        line, = axes.plot(centers[inner], (curves if dist else histograms)[j][inner],
                          linewidth=1, color=colors[j], label=labels[j])
        lines.append(line)
    if dist and (rawline == 1 or (rawline == 2 and count < 6)):
        for j in range(count):
            axes.plot(centers[inner], histograms[j][inner], linewidth=1,
                      linestyle=":", color=colors[j], label="_nolegend_")

    axes.set_ylabel("Normalized Frequency" if norm_frequency else "Frequency")
    axes.grid(True, which="major")
    axes.minorticks_on()
    axes.grid(True, which="minor", alpha=0.3)
    # A legend with more entries than that hides the plot instead of
    # explaining it, so it is left off -- as the MATLAB original does
    if count <= 20:
        axes.legend()
    if xlim is not None and len(xlim) == 2:
        axes.set_xlim(xlim)
    if ylim is not None and len(ylim) == 2:
        axes.set_ylim(ylim)
    figures["histogram"] = figure

    # -- all inputs pooled into one histogram, when nothing was fitted ---
    if not dist and pooled:
        pooled_figure = plt.figure(figsize=(winsize[0] / 100.0, winsize[1] / 100.0),
                                   dpi=100)
        pooled_axes = pooled_figure.gca()
        pooled_axes.bar(centers, hist_centers(np.concatenate(pooled), centers),
                        width=width, color="0.4")
        pooled_axes.set_ylabel("Frequency")
        pooled_axes.set_title("All data")
        pooled_axes.grid(True)
        if xlim is not None and len(xlim) == 2:
            pooled_axes.set_xlim(xlim)
        figures["pooled"] = pooled_figure

    # -- the average of all histograms ----------------------------------
    if mean:
        mean_axes = mean_figure.gca()
        x_mean = centers[inner]
        y_mean = histograms[:, inner].mean(axis=0)
        # Keep the automatic colour cycle, so that repeated calls into this
        # figure distinguish their averages
        mean_line, = mean_axes.plot(x_mean, y_mean, linewidth=1,
                                    label="Average histogram")
        first_average = len(mean_axes.lines) == 1
        band = None
        if count > 1:
            error = histograms[:, inner].std(axis=0, ddof=1) / np.sqrt(count)
            if np.any(error > 0):
                # Behind the averages and out of the legend, so that repeated
                # calls can be labelled with one entry per average
                band = mean_axes.fill_between(
                    x_mean, y_mean - error, y_mean + error,
                    color=mean_line.get_color(), alpha=0.2, linewidth=0,
                    zorder=mean_line.get_zorder() - 1, label="_nolegend_")
        if first_average:
            handles = [mean_line]
            entries = ["Average histogram"]
            if band is not None:
                handles.append(band)
                entries.append("Standard error")
            mean_axes.legend(handles, entries)
        mean_axes.set_ylabel("Normalized Frequency" if norm_frequency else "Frequency")
        mean_axes.grid(True, which="major")
        mean_axes.minorticks_on()
        mean_axes.grid(True, which="minor", alpha=0.3)
        if xlim is not None and len(xlim) == 2:
            mean_axes.set_xlim(xlim)
        if ylim is not None and len(ylim) == 2:
            mean_axes.set_ylim(ylim)
        figures["mean"] = mean_figure

    return HistogramResult(
        x=centers[inner], hist=histograms[:, inner], fit=curves[:, inner],
        stats=stats, labels=labels, figures=figures,
    )


# ----------------------------------------------------------------------
# The special case of exactly two inputs
# ----------------------------------------------------------------------

#: Points drawn in the density scatter; more than that turns the plot into
#: a solid block and takes minutes to render, so the data is thinned.
_SCATTER_MAX_POINTS = 200000


def _difference_colormap():
    """Return the diverging map of the MATLAB original: inverted hot, hot."""
    from matplotlib.colors import ListedColormap
    from matplotlib import colormaps

    hot = colormaps["hot"](np.linspace(0, 1, 64))[:, :3]
    return ListedColormap(np.vstack([1.0 - hot, hot]))


def _density_scatter(axes, x: np.ndarray, y: np.ndarray, bins: int = 200) -> None:
    """Scatter *x* against *y*, coloured by the local density of the points."""
    from scipy.ndimage import gaussian_filter

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    # Smooth over a few bins, as dscatter's spline smoothing does, so that
    # the colour follows the density rather than the binning
    counts = gaussian_filter(counts, bins / 50.0)
    ix = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    density = counts[ix, iy]

    if x.size > _SCATTER_MAX_POINTS:
        step = int(np.ceil(x.size / _SCATTER_MAX_POINTS))
        x, y, density = x[::step], y[::step], density[::step]
    axes.scatter(x, y, c=density, s=4, marker="s", linewidths=0, cmap="viridis")


def _compare_two(plt, samples, inputs, filetypes, is_files,
                 verbose) -> Dict[str, object]:
    """Draw the scatter plot and the difference of two matching inputs.

    Returns:
        The figures that were drawn, by role.
    """
    first, second = samples
    if first.shape != second.shape:
        if verbose:
            print("No 2D histogram plotted because size differs between data.")
        return {}

    figures: Dict[str, object] = {}
    x = np.asarray(first, dtype=np.float64).ravel()
    y = np.asarray(second, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    figure = plt.figure(figsize=(5, 5), dpi=100)
    axes = figure.gca()
    _density_scatter(axes, x[keep], y[keep])
    axes.set_title("Histogram")
    if is_files:
        axes.set_xlabel(os.path.basename(inputs[0]))
        axes.set_ylabel(os.path.basename(inputs[1]))
    else:
        axes.set_xlabel("Data 1")
        axes.set_ylabel("Data 2")
    axes.set_aspect("auto")
    figures["scatter"] = figure

    if not is_files:
        return figures

    difference = np.nan_to_num(np.asarray(second, dtype=np.float64)
                               - np.asarray(first, dtype=np.float64))

    # -- three orthogonal projections of a volume difference ------------
    if filetypes[0] in (0, 2) and difference.ndim == 3:
        peak = float(np.abs(difference).max())
        if peak == 0.0:
            if verbose:
                print("Images are identical!")
            return figures
        if verbose:
            print(f"Blue: i1>i2; Red: i2>i1\ni1 = {inputs[0]}\ni2 = {inputs[1]}")
        projections = [np.rot90(difference.sum(axis=0)),
                       np.rot90(difference.sum(axis=1)),
                       difference.sum(axis=2)]
        scale = max(float(np.abs(p).max()) for p in projections) or np.finfo(float).eps
        cmap = _difference_colormap()
        diff_figure, panels = plt.subplots(2, 2, figsize=(7, 7), dpi=100)
        for axis, projection in zip(panels.ravel(), projections):
            axis.imshow(projection, vmin=-scale, vmax=scale, cmap=cmap,
                        aspect="equal")
            axis.axis("off")
        panels[1, 1].axis("off")
        image = panels[1, 1].imshow(np.zeros((1, 1)), vmin=-peak, vmax=peak, cmap=cmap)
        image.set_visible(False)
        diff_figure.colorbar(image, ax=panels[1, 1], fraction=0.6)
        figures["difference"] = diff_figure

    # -- the surface counterpart is a rendering, which lives elsewhere ---
    elif filetypes[0] == 3 and verbose:
        span = float(np.abs(difference).max())
        print(f"Blue: i1>i2; Red: i2>i1\ni1 = {inputs[0]}\ni2 = {inputs[1]}")
        print(f"Difference range +/-{span:g}; render it with "
              f"CAT_SurfView --overlay <difference>.gii")

    return figures


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = ArgumentParser(
        prog="CAT_PlotHistogram",
        description="Histogram of one or more volume, surface or text data "
                    "sets.\n\n"
                    "Python counterpart of CAT12's cat_plot_histogram.m.  "
                    "Reads NIfTI\nvolumes, GIFTI surfaces, FreeSurfer "
                    "morphometry data and plain text,\ndrops the zero "
                    "background of the files that have one, and draws one\n"
                    "line per input on a shared set of bins.",
    )
    parser.add_argument("files", nargs="*", metavar="FILE",
                        help="volumes, surfaces or text files to plot")
    parser.add_argument("--dist", default="kernel", choices=DISTRIBUTIONS,
                        metavar="NAME",
                        help="curve fitted through each sample: "
                             + ", ".join(DISTRIBUTIONS))
    parser.add_argument("--bins", type=int, default=500,
                        help="upper limit on the number of bins")
    parser.add_argument("--xrange", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="range to bin over (default: the data range)")
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="range of the drawn x axis")
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="range of the drawn y axis")
    parser.add_argument("--winsize", type=int, nargs=2, default=[750, 500],
                        metavar=("W", "H"), help="figure size in pixels")
    parser.add_argument("--color", default=DEFAULT_PALETTE, metavar="NAME",
                        choices=sorted(PALETTES),
                        help="categorical palette: " + ", ".join(sorted(PALETTES)))
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="line opacity; outside [0,1] it follows the "
                             "number of inputs")
    parser.add_argument("--rawline", type=int, default=2, choices=(0, 1, 2),
                        help="dotted raw histogram next to a fitted curve: "
                             "0 never, 1 always, 2 only for fewer than six inputs")
    parser.add_argument("--no-norm-frequency", dest="norm_frequency",
                        action="store_false",
                        help="plot counts instead of normalizing each "
                             "histogram by its own total")
    parser.add_argument("--mean", action="store_true",
                        help="also plot the average histogram with its "
                             "standard error in a second figure")
    parser.add_argument("--no-scatter", dest="scatter", action="store_false",
                        help="skip the scatter plot and the difference "
                             "projections drawn for two matching inputs")
    parser.add_argument("--fig", type=int, metavar="N",
                        help="number of the figure to draw into")
    parser.add_argument("--save", metavar="PREFIX",
                        help="write the figures to PREFIX.png, "
                             "PREFIX_mean.png, ... instead of showing them")
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

    try:
        result = plot_histogram(
            args.files, fig=args.fig, color=args.color,
            norm_frequency=args.norm_frequency, winsize=args.winsize,
            xrange=args.xrange, xlim=args.xlim, ylim=args.ylim, dist=args.dist,
            mean=args.mean, bins=args.bins, alpha=args.alpha,
            rawline=args.rawline, scatter=args.scatter, verbose=args.verbose,
        )
    except (OSError, ValueError) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 1

    if args.save:
        base, ext = os.path.splitext(args.save)
        ext = ext or ".png"
        for role, figure in result.figures.items():
            name = f"{base}{ext}" if role == "histogram" else f"{base}_{role}{ext}"
            figure.savefig(name, dpi=100, bbox_inches="tight")
            print(f"saved {name}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
