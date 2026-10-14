#!/usr/bin/env python3
"""Histogram of one or more volume, surface or text data sets.

Python counterpart of CAT12's ``cat_plot_histogram.m``.  Every input is
reduced to a one-dimensional sample, binned on a grid shared by all inputs,
and drawn as a line — one line per input — so that distributions can be
compared directly.  By default a smooth curve is fitted through the sample
(a Gaussian kernel density estimate) and the raw histogram is added as a
dotted line of the same colour.  ``--ridgeline`` stacks the curves instead,
one filled ridge per input, which keeps many inputs apart.

The kernel density estimate uses a Gaussian kernel whose bandwidth comes
from the Improved Sheather-Jones selector, which resolves multimodal
densities -- see :func:`kernel_bandwidth`.  ``--dist gaussian_kde`` takes the
bandwidth of scipy's ``gaussian_kde`` instead, Scott's rule.  Either is
computed from every value, not a subsample, and the density scatter of two
inputs uses the same kernel in two dimensions.

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

    # one ridge per input, stacked from the first input downwards
    CAT_PlotHistogram --ridgeline lh.thickness.*

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

#: Everything ``--dist`` accepts.  ``kernel`` is the default and
#: ``gaussian_kde`` the estimate of scipy's ``gaussian_kde``; neither needs a
#: parametric family.  ``none`` draws the plain histogram instead of a curve.
DISTRIBUTIONS: Tuple[str, ...] = (
    ("kernel", "gaussian_kde", "none") + tuple(sorted(_DISTRIBUTIONS))
)


#: Grid points per bandwidth on which :func:`kernel_density` bins the
#: sample.  Binning and interpolation errors fall with the square of the
#: spacing and stay below 0.1% of the peak at this density.
_KDE_GRID_PER_BANDWIDTH = 20

#: Upper limit on that grid, reached only by data whose range is tens of
#: thousands of bandwidths wide.
_KDE_MAX_GRID = 2 ** 20

#: The Gaussian is cut off this many bandwidths from its center.
_KDE_TRUNCATE = 5.0

#: Bins of the Improved Sheather-Jones selector, spread over the data range
#: padded by a tenth on either side -- the setting of Botev et al.
_ISJ_GRID = 2 ** 14

#: Lowest bandwidth :func:`kernel_bandwidth` accepts, as a fraction of
#: Silverman's rule of thumb.  On smooth densities the ISJ bandwidth hardly
#: reaches it -- three times below the rule for two well separated modes,
#: 4.8 times only for four narrow ones -- but it collapses by a factor of
#: thousands on data that are not smooth: values stored on a lattice (uint8
#: p1 maps), ties, or the steep pile-up of near-zero probabilities in a p1
#: map.
_ISJ_FLOOR = 0.25


def _rule_of_thumb(values: np.ndarray, ndim: int = 1) -> float:
    """Return Silverman's rule-of-thumb bandwidth of the finite *values*.

    The scale is the robust ``a = min(std, IQR / 1.349)``.  One dimension
    gets Silverman's ``0.9 a n^(-1/5)``, the default of R's ``density()``;
    one axis of a two-dimensional estimate the normal reference
    ``a n^(-1/6)``.
    """
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    q25, q75 = np.percentile(values, [25, 75])
    scale = min(std, float(q75 - q25) / 1.349)
    if not scale > 0:
        # Half the values or more tied
        scale = std
    if not scale > 0:
        # A constant sample: any width will do, so take its magnitude
        scale = abs(float(values[0])) or 1.0
    if ndim == 2:
        return scale * values.size ** (-1.0 / 6.0)
    return 0.9 * scale * values.size ** -0.2


def _isj_bandwidth(values: np.ndarray) -> Optional[float]:
    """Return the Improved Sheather-Jones bandwidth of the finite *values*.

    Botev, Grotowski & Kroese (2010), "Kernel density estimation via
    diffusion", Ann. Statist. 38(5): the fixed point ``t = xi gamma^[7](t)``
    of the squared bandwidth, solved on the discrete cosine transform of the
    binned sample.

    Returns:
        The bandwidth, or ``None`` when the equation has no root -- for a
        constant sample or one with very few values.
    """
    from scipy.fft import dct
    from scipy.optimize import brentq

    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) / 10.0
    if not pad > 0:
        return None
    counts, _ = np.histogram(values, bins=_ISJ_GRID, range=(lo - pad, hi + pad))
    power = (np.asarray(dct(counts / values.size, type=2))[1:] / 2.0) ** 2
    k2 = np.arange(1, _ISJ_GRID, dtype=np.float64) ** 2
    k2_powers = {s: k2 ** s for s in range(2, 8)}
    n = values.size

    def functional(s: int, time: float) -> float:
        """Estimate of the squared norm of the s-th density derivative."""
        weights = np.exp(-k2 * np.pi ** 2 * time)
        return 2.0 * np.pi ** (2 * s) * float(np.sum(k2_powers[s] * power * weights))

    def fixed_point(t: float) -> float:
        f = functional(7, t)
        for s in range(6, 1, -1):
            if not f > 0:
                return -1.0
            k0 = float(np.prod(np.arange(1, 2 * s, 2))) / np.sqrt(2.0 * np.pi)
            const = (1.0 + 0.5 ** (s + 0.5)) / 3.0
            f = functional(s, (2.0 * const * k0 / (n * f)) ** (2.0 / (3 + 2 * s)))
        if not f > 0:
            return -1.0
        return t - (2.0 * n * np.sqrt(np.pi) * f) ** -0.4

    try:
        t_star = brentq(fixed_point, 0.0, 0.1)
    except ValueError:
        return None
    return float(np.sqrt(t_star)) * (hi - lo + 2.0 * pad)


def kernel_bandwidth(sample: np.ndarray) -> float:
    """Return the bandwidth of the Gaussian kernel for *sample*.

    The Improved Sheather-Jones (ISJ) selector of Botev et al. (2010).  It
    makes no normal-reference assumption, so it resolves multimodal and
    skewed densities that Silverman's and Scott's rules oversmooth: on
    simulated multimodal mixtures its error was up to 17 times lower than
    Silverman's, and on normal data it finds the optimal bandwidth.

    Plug-in selectors collapse where the data are not a smooth density --
    the lattice of a uint8 p1 map, its saturated end, the pile-up of
    near-zero probabilities -- so the result is never taken below a quarter
    of Silverman's rule of thumb, and that rule is used outright where the
    ISJ equation has no solution.

    Args:
        sample: The values; non-finite ones are ignored.

    Returns:
        The standard deviation of the Gaussian kernel.
    """
    values = _finite(sample)
    if values.size == 0:
        raise ValueError("cannot estimate a bandwidth without finite values")
    rule = _rule_of_thumb(values)
    isj = _isj_bandwidth(values)
    return rule if isj is None else max(isj, _ISJ_FLOOR * rule)


def kernel_density(sample: np.ndarray, x: np.ndarray,
                   bandwidth: Optional[float] = None) -> np.ndarray:
    """Evaluate a Gaussian kernel density estimate of *sample* at *x*.

    Computed on a regular grid, as R's ``density()`` and statsmodels do: the
    sample is linearly binned onto it and convolved with the Gaussian.  That
    costs ``O(n + m log m)`` instead of ``O(n len(x))``, so every value of a
    whole-brain map is used and none has to be thinned out.

    Args:
        sample: The values; non-finite ones are ignored.
        x: Points at which the density is evaluated.
        bandwidth: Standard deviation of the kernel; :func:`kernel_bandwidth`
            when ``None``.

    Returns:
        The probability density at *x*.
    """
    from scipy.signal import fftconvolve

    values = _finite(sample)
    x = np.asarray(x, dtype=np.float64)
    width = kernel_bandwidth(values) if bandwidth is None else float(bandwidth)
    if not width > 0:
        raise ValueError(f"the bandwidth has to be positive, not {width:g}")

    # The grid spans x and the reach of the kernel around it; values beyond
    # that add nothing measurable at x
    reach = _KDE_TRUNCATE * width
    lo = float(x.min()) - reach
    hi = float(x.max()) + reach
    size = int(min(np.ceil((hi - lo) / width * _KDE_GRID_PER_BANDWIDTH),
                   _KDE_MAX_GRID)) + 1
    step = (hi - lo) / (size - 1)

    # Linear binning: each value is shared between its two grid neighbours
    position = (values - lo) / step
    position = position[(position >= 0) & (position <= size - 1)]
    left = np.minimum(position.astype(np.int64), size - 2)
    frac = position - left
    counts = (np.bincount(left, 1.0 - frac, size)
              + np.bincount(left + 1, frac, size))

    half = int(np.ceil(reach / step))
    t = np.arange(-half, half + 1) * (step / width)
    kernel = np.exp(-0.5 * t * t) / (np.sqrt(2.0 * np.pi) * width * values.size)
    # The FFT leaves round-off of either sign where the density is zero
    density = np.maximum(fftconvolve(counts, kernel, mode="same"), 0.0)
    return np.interp(x, lo + step * np.arange(size), density)


def gaussian_kde_bandwidth(sample: np.ndarray) -> float:
    """Return the bandwidth scipy's ``gaussian_kde`` picks for *sample*.

    Its default, Scott's rule ``std n^(-1/5)``, which seaborn's ``kdeplot``
    uses as well.  The rule assumes a normal density, so it is smoother than
    :func:`kernel_bandwidth` and flattens sharp or close peaks.

    Args:
        sample: The values; non-finite ones are ignored.

    Returns:
        The standard deviation of the Gaussian kernel.
    """
    from scipy import stats

    values = _finite(sample)
    try:
        kde = stats.gaussian_kde(values)
    except (np.linalg.LinAlgError, ValueError):
        # scipy refuses a constant sample and a single value
        if values.size == 0:
            raise ValueError("cannot estimate a bandwidth without finite values")
        return _rule_of_thumb(values)
    return float(np.sqrt(kde.covariance[0, 0]))


def fit_density(sample: np.ndarray, dist: str, x: np.ndarray) -> np.ndarray:
    """Evaluate the density of *dist* fitted to *sample* at the points *x*.

    Args:
        sample: The values to fit.
        dist: ``"kernel"`` for a Gaussian kernel density estimate (see
            :func:`kernel_density`), ``"gaussian_kde"`` for the one of
            scipy's ``gaussian_kde``, or one of the parametric families in
            :data:`DISTRIBUTIONS`.
        x: Points at which the fitted density is evaluated.

    Returns:
        The probability density at *x*.
    """
    from scipy import stats

    if dist == "kernel":
        return kernel_density(sample, x)
    if dist == "gaussian_kde":
        # scipy's own evaluation costs seconds per whole-brain map; the
        # binned one is the same curve, to within 0.03% of its peak
        return kernel_density(sample, x, gaussian_kde_bandwidth(sample))

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


#: How :func:`plot_histogram` scales the ridges of a ridgeline plot: all on
#: one scale, or each to its own maximum.
RIDGE_SCALES: Tuple[str, ...] = ("shared", "own")

#: Ridge labels longer than this are shortened in the middle; the y axis
#: would otherwise squeeze the plot into a corner of the figure.
_RIDGE_LABEL_LENGTH = 40


def _ellipsize(label: str, length: int = _RIDGE_LABEL_LENGTH) -> str:
    """Shorten *label* to *length* characters by cutting out its middle.

    The start keeps a ``TH5=`` prefix, the end keeps the file name.
    """
    if len(label) <= length:
        return label
    head = (length - 1) * 2 // 5
    return label[:head] + "…" + label[len(label) - (length - 1 - head):]


def _draw_ridgeline(axes, x: np.ndarray, curves: np.ndarray,
                    raw: Optional[np.ndarray], colors: np.ndarray,
                    labels: Sequence[str], height: float, alpha: float,
                    xlim: Optional[Sequence[float]] = None,
                    own_scale: bool = False) -> None:
    """Draw *curves* as a ridgeline plot into *axes*.

    The first curve gets the top baseline.  All ridges share one scale, set
    so that the tallest reaches *height* baseline spacings within *xlim* --
    or, with *own_scale*, each ridge reaches that height on its own -- and
    each one is drawn in front of the ridge above it.

    Args:
        axes: The axes to draw into.
        x: The shared x positions.
        curves: One curve per row.
        raw: Histograms drawn as dotted lines on the same scale, or ``None``.
        colors: One RGB row per curve.
        labels: The y tick label of every baseline.
        height: Height of the tallest ridge in baseline spacings.
        alpha: Opacity of the fill.
        xlim: The drawn x range, which the scale is taken from; all of *x*
            when ``None``.  A peak cut off there, such as the saturated end
            of a p1 map, then no longer flattens the rest.
        own_scale: Scale every ridge to its own maximum instead of the
            maximum of all of them.
    """
    from matplotlib.ticker import AutoMinorLocator

    count = len(curves)
    visible = np.ones(len(x), dtype=bool)
    if xlim is not None:
        visible = (x >= min(xlim)) & (x <= max(xlim))
    peaks = np.zeros(count)
    for j in range(count):
        finite = curves[j, visible]
        finite = finite[np.isfinite(finite)]
        peaks[j] = finite.max() if finite.size else 0.0
    if not own_scale:
        peaks[:] = peaks.max() if count else 0.0
    baselines = np.arange(count - 1, -1, -1, dtype=float)
    background = axes.get_facecolor()
    top = 0.0
    for j in range(count):
        scale = height / peaks[j] if peaks[j] > 0 else 0.0
        ridge = baselines[j] + scale * curves[j]
        # An opaque backing hides the ridges behind, so that the tinted fill
        # on top of it stays light where it overlaps them
        zorder = 2 + 3 * j
        axes.fill_between(x, baselines[j], ridge, color=background,
                          linewidth=0, zorder=zorder)
        axes.fill_between(x, baselines[j], ridge, color=colors[j], alpha=alpha,
                          linewidth=0, zorder=zorder + 1)
        axes.plot(x, ridge, color=colors[j], linewidth=1, zorder=zorder + 2)
        if raw is not None:
            ridge = baselines[j] + scale * raw[j]
            axes.plot(x, ridge, color=colors[j], linewidth=1, linestyle=":",
                      zorder=zorder + 2)
        drawn = ridge[visible & np.isfinite(ridge)]
        if drawn.size:
            top = max(top, float(drawn.max()))

    axes.set_ylim(-0.1, max(top, baselines[0]) + 0.1)
    axes.set_yticks(baselines)
    short = [_ellipsize(label) for label in labels]
    # Two labels made identical by the cut would not tell the ridges apart
    axes.set_yticklabels(short if len(set(short)) == len(set(labels)) else labels)
    # Shrink the labels when the rows get narrower than the text
    row = (axes.get_position().height * axes.figure.get_figheight() * 72.0
           / (count - 1 + height + 0.2))
    for label in axes.get_yticklabels():
        label.set_fontsize(min(label.get_fontsize(), max(4.0, 0.9 * row)))
    axes.tick_params(axis="y", which="both", length=0)
    for side in ("left", "right", "top"):
        axes.spines[side].set_visible(False)
    axes.set_axisbelow(True)
    axes.xaxis.set_minor_locator(AutoMinorLocator())
    axes.grid(True, axis="x", which="major")
    axes.grid(True, axis="x", which="minor", alpha=0.3)


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
    ridgeline: bool = False,
    ridge_height: float = 1.5,
    ridge_scale: str = "shared",
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
            ``"kernel"`` is a Gaussian kernel density estimate with the ISJ
            bandwidth, ``"gaussian_kde"`` the one of scipy's
            ``gaussian_kde`` (Scott's rule), ``"none"`` draws the plain
            histogram and adds a second figure with all inputs pooled.
        mean: Also draw the average of all histograms with its standard
            error as a shaded band, in its own figure (number 11).  Repeated
            calls overlay their averages there in the next colour and have
            to be labelled by the caller, because the bands are kept out of
            the legend.
        bins: Upper limit on the number of bins.
        alpha: Opacity of the lines; outside ``[0, 1]`` it is derived from
            the number of inputs.  With *ridgeline* it is the opacity of the
            filled ridges instead, 0.6 when automatic.
        rawline: Draw the raw histogram as a dotted line next to a fitted
            curve -- ``0`` never, ``1`` always, ``2`` only for fewer than
            six inputs.
        scatter: For exactly two inputs of equal shape, add the density
            scatter plot and the difference projections.
        names: Legend labels, one per input; derived from the file names or
            numbered when ``None``.
        verbose: Print the summary table.
        ridgeline: Draw the curves as a ridgeline plot -- one filled curve
            per input, stacked from the first input at the top downwards
            and labelled on the y axis -- instead of overlaying them.  All
            ridges share one height scale, taken within *xlim* when that is
            given; *ylim* does not apply.
        ridge_height: Height of the tallest ridge -- of every ridge with
            *ridge_scale* ``"own"`` -- in units of the spacing between
            neighbouring baselines; at ``1`` it just reaches the baseline
            above.
        ridge_scale: ``"shared"`` puts all ridges on one height scale, so
            that their heights compare -- the default of joypy, seaborn's
            ridge plot and ggridges.  ``"own"`` scales every ridge to its own
            maximum, so that all reach *ridge_height* and only their shapes
            compare; one sharp peak then no longer flattens the others.

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
    if ridgeline and not ridge_height > 0:
        raise ValueError(f"ridge_height has to be positive, not {ridge_height:g}")
    if ridge_scale not in RIDGE_SCALES:
        raise ValueError(f"ridge_scale has to be one of {', '.join(RIDGE_SCALES)}, "
                         f"not '{ridge_scale}'")
    # Ridges do not overlap the way lines do, so they fill at a fixed opacity
    ridge_alpha = alpha if 0 <= alpha <= 1 else 0.6
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
    shown = curves if dist else histograms
    raw = bool(dist) and (rawline == 1 or (rawline == 2 and count < 6))
    if ridgeline:
        _draw_ridgeline(axes, centers[inner], shown[:, inner],
                        histograms[:, inner] if raw else None, colors[:, :3],
                        labels, ridge_height, ridge_alpha,
                        xlim if xlim is not None and len(xlim) == 2 else None,
                        own_scale=ridge_scale == "own")
    else:
        for j in range(count):
            axes.plot(centers[inner], shown[j][inner], linewidth=1,
                      color=colors[j], label=labels[j])
        if raw:
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
        if ylim is not None and len(ylim) == 2:
            axes.set_ylim(ylim)
    if xlim is not None and len(xlim) == 2:
        axes.set_xlim(xlim)
    if ridgeline:
        # The labels sit left of the axes, where the default margin is too
        # narrow for file names
        figure.tight_layout()
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
    """Scatter *x* against *y*, coloured by the local density of the points.

    The density is the two-dimensional counterpart of :func:`kernel_density`:
    the binned points smoothed with a Gaussian whose width on each axis is
    the two-dimensional rule of thumb.  The colour does not need the ISJ
    selector, which would cost a two-dimensional fixed point.
    """
    from scipy.ndimage import gaussian_filter

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    sigma = [_rule_of_thumb(values, ndim=2) / (edges[1] - edges[0])
             for values, edges in ((x, x_edges), (y, y_edges))]
    counts = gaussian_filter(counts, sigma, mode="constant")
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
                        help="curve fitted through each sample: kernel "
                             "(Gaussian KDE with the ISJ bandwidth, default), "
                             "gaussian_kde (scipy's, Scott's rule), none "
                             "(the plain histogram) or a parametric family: "
                             + ", ".join(sorted(_DISTRIBUTIONS)))
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
    parser.add_argument("--ridgeline", action="store_true",
                        help="stack the curves as a ridgeline plot, the first "
                             "input at the top, instead of overlaying them")
    parser.add_argument("--ridge-height", type=float, default=1.5, metavar="H",
                        help="height of the tallest ridge (of every ridge "
                             "with --ridge-scale own) in baseline spacings; "
                             "1 just reaches the baseline above (default 1.5)")
    parser.add_argument("--ridge-scale", default="shared", choices=RIDGE_SCALES,
                        help="shared: one height scale for all ridges, so "
                             "heights compare (default); own: every ridge "
                             "scaled to its own maximum")
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
            ridgeline=args.ridgeline, ridge_height=args.ridge_height,
            ridge_scale=args.ridge_scale,
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
