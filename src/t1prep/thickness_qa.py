"""Shape of the cortical thickness distribution as a quality measure.

Residual overestimation -- a glued sulcus the barrier did not open, a thick
blob in the insula -- stretches the *upper* tail of the thickness histogram,
so the skewness of that histogram is a natural per-hemisphere flag.  Taken
over all vertices it measures something else, though: the lower tail
dominates it, because the medial wall and other near-zero values (2-15% of
the vertices on 38 test hemispheres) are part of the surface.  Plain
skewness came out negative on every one of them.

Dropping the values below :data:`LOWER_CUT` turns it into an upper-tail
measure.  On the test hemispheres it followed independent signs of fused
sulci (the fused fraction of the cortical band, and the share of vertices
where PBT exceeds the pial-white distance by more than 1 mm) with a partial
Spearman correlation of +0.57 / +0.47 after controlling for the median
thickness, and a fixed cut kept it less coupled to the thickness itself than
one relative to the median (+0.36 against +0.66).

What it can and cannot see, measured by injecting patches of known size:
implausible values (+2.5 mm) on 1% of the vertices shift it by about two
between-subject standard deviations and six times the typical difference
between the hemispheres of a subject; +1.5 mm on 2% stays inside the normal
spread.  It does not localize anything, and overestimation spread over the
whole hemisphere shifts the histogram rather than its tail, so it goes
unnoticed.  The two hemispheres of a subject agree closely (rank correlation
0.95), which makes their difference the most sensitive use.

:data:`HIGH_FACTOR` defines the companion measure, the share of vertices
thicker than that multiple of the hemisphere median.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

__all__ = ["thickness_shape", "LOWER_CUT", "HIGH_FACTOR"]

#: Values below this thickness in mm are left out of the skewness.
LOWER_CUT = 1.0

#: Multiple of the median above which a vertex counts as implausibly thick.
HIGH_FACTOR = 1.6


def _skewness(x: np.ndarray) -> Optional[float]:
    """Sample skewness g1 = m3 / m2**1.5 (as ``scipy.stats.skew``)."""
    if x.size < 3:
        return None
    d = x - x.mean()
    m2 = np.mean(d * d)
    if m2 <= 0.0:
        return 0.0
    return float(np.mean(d ** 3) / m2 ** 1.5)


def thickness_shape(
    values: Any,
    lower_cut: float = LOWER_CUT,
    high_factor: float = HIGH_FACTOR,
) -> Dict[str, Any]:
    """Upper-tail shape of a per-vertex thickness map.

    Parameters
    ----------
    values : array_like, shape (V,)
        Cortical thickness in mm, one value per vertex.  Non-finite values
        are ignored.
    lower_cut : float
        Values below this are left out of the skewness (medial wall).
    high_factor : float
        A vertex thicker than ``high_factor`` times the median counts as
        implausibly thick.

    Returns
    -------
    dict
        ``n_vertices``, ``median`` (mm, over all finite values),
        ``upper_skewness`` (skewness of the values >= ``lower_cut``, or
        ``None`` when fewer than three remain) and ``high_fraction``
        (fraction of the finite values above ``high_factor * median``).
    """
    t = np.asarray(values, dtype=np.float64).ravel()
    t = t[np.isfinite(t)]
    if t.size == 0:
        return {"n_vertices": 0, "median": None,
                "upper_skewness": None, "high_fraction": None}
    med = float(np.median(t))
    return {
        "n_vertices": int(t.size),
        "median": med,
        "upper_skewness": _skewness(t[t >= lower_cut]),
        "high_fraction": float(np.mean(t > high_factor * med)),
    }
