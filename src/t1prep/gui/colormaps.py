"""Colormaps shared by the surface and volume viewers.

Both build the same lookup tables, so they live here rather than in either
viewer — cat_surf_view imports cat_vol_view, which would make the reverse
direction circular.
"""

from __future__ import annotations

from typing import List

from vtkmodules.vtkCommonCore import vtkLookupTable

#: Colormap ids, in the order the viewers offer them
C1, C2, C3, JET, HOT, FIRE, BIPOLAR, GRAY = range(8)

#: Names shown in the control panel, indexed the way the combo box lists them
COLORMAP_NAMES = ("JET", "HOT", "FIRE", "BIPOLAR", "GRAY", "C1", "C2", "C3")
COLORMAP_ORDER = (JET, HOT, FIRE, BIPOLAR, GRAY, C1, C2, C3)


# ---- Lookup table helper ----
class LookupTableWithEnabling(vtkLookupTable):
    """Plain VTK lookup table; clipping is applied by zeroing table alphas."""


def _fill_from_ctf(lut: LookupTableWithEnabling, points, alpha: float):
    # points: list of (pos0..100, r,g,b)
    xs = [p[0] for p in points]
    rs = [p[1] for p in points]
    gs = [p[2] for p in points]
    bs = [p[3] for p in points]
    def interp(x, xs, ys):
        if x <= xs[0]: return ys[0]
        if x >= xs[-1]: return ys[-1]
        for i in range(1, len(xs)):
            if x <= xs[i]:
                t = (x - xs[i-1])/(xs[i]-xs[i-1])
                return ys[i-1]*(1-t) + ys[i]*t
        return ys[-1]
    lut.SetNumberOfTableValues(256)
    for i in range(256):
        val = (i/255.0)*100.0
        r = interp(val, xs, rs); g = interp(val, xs, gs); b = interp(val, xs, bs)
        lut.SetTableValue(i, r, g, b, alpha)

def get_lookup_table(colormap: int, alpha: float) -> LookupTableWithEnabling:
    lut = LookupTableWithEnabling()
    if colormap == C1:
        pts = [
            (0.0, 50/255.0,136/255.0,189/255.0), (12.5, 102/255.0,194/255.0,165/255.0),
            (25, 171/255.0,221/255.0,164/255.0), (37.5, 230/255.0,245/255.0,152/255.0),
            (50, 1.0,1.0,191/255.0), (62.5, 254/255.0,224/255.0,139/255.0),
            (75, 253/255.0,174/255.0,97/255.0), (82.5, 244/255.0,109/255.0,67/255.0),
            (100.0,213/255.0,62/255.0,79/255.0),
        ]; _fill_from_ctf(lut, pts, alpha)
    elif colormap == C2:
        pts = [(0,0,0.6,1),(25,0.5,1,0.5),(50,1,1,0.5),(75,1,0.75,0.5),(100,1,0.5,0.5)]; _fill_from_ctf(lut, pts, alpha)
    elif colormap == C3:
        pts = [(0,0/255,143/255,213/255),(25,111/255,190/255,70/255),(50,1,220/255,45/255),(75,252/255,171/255,23/255),(100,238/255,28/255,58/255)]; _fill_from_ctf(lut, pts, alpha)
    elif colormap == JET:
        pts = [(0,0,0,0.5625),(16.67,0,0,1),(33.33,0,1,1),(50,0.5,1,0.5),(66.67,1,1,0),(83.33,1,0,0),(100,0.5,0,0)]; _fill_from_ctf(lut, pts, alpha)
    elif colormap == HOT:
        # Classic HOT: black -> red -> yellow -> white
        pts = [(0,0,0,0),(33.33,1,0,0),(66.67,1,1,0),(100,1,1,1)]; _fill_from_ctf(lut, pts, alpha)
    elif colormap == FIRE:
        pts = [(0,0,0,0),(25,0.5,0,0),(50,1,0,0),(75,1,0.5,0),(100,1,1,0)]; _fill_from_ctf(lut, pts, alpha)
    elif colormap == BIPOLAR:
        pts = [(0,0,1,1),(25,0,0,1),(50,0.1,0.1,0.1),(62.5,0.5,0,0),(75,1,0,0),(87.5,1,0.5,0),(100,1,1,0)]; _fill_from_ctf(lut, pts, alpha)
    elif colormap == GRAY:
        lut.SetHueRange(0.0, 0.0); lut.SetSaturationRange(0.0, 0.0); lut.SetValueRange(0.0, 1.0); lut.Build()
    else:
        lut.Build()
    return lut


def invert_lut(lut: vtkLookupTable) -> None:
    """Reverse the order of colours in a table, keeping each entry's alpha.

    This flips how values map to colours without touching the data or the
    table range.
    """
    n = int(lut.GetNumberOfTableValues())
    for i in range(n // 2):
        first = lut.GetTableValue(i)
        last = lut.GetTableValue(n - 1 - i)
        lut.SetTableValue(i, *last)
        lut.SetTableValue(n - 1 - i, *first)


def apply_discrete(lut: vtkLookupTable, levels: int) -> None:
    """Flatten a table into *levels* bands of constant colour.

    The table is split into equally wide segments, each filled with the colour
    of its first entry.
    """
    levels = int(levels or 0)
    if levels <= 0:
        return
    n = int(lut.GetNumberOfTableValues())
    levels = max(1, min(n, levels))
    for band in range(levels):
        start = int(band * n / levels)
        end = int((band + 1) * n / levels) if band < levels - 1 else n
        r, g, b, a = lut.GetTableValue(start)
        for i in range(start, end):
            lut.SetTableValue(i, r, g, b, a)


def clipped_lut_indices(n: int, smin: float, smax: float,
                        c0: float, c1: float) -> List[int]:
    """Return the table indices covered by the clip window (c0, c1).

    Entry ``i`` of an ``n``-entry table represents the value
    ``smin + i/(n-1) * (smax - smin)``.  VTK clamps values outside
    ``[smin, smax]`` onto the first/last entry, so those two entries are also
    clipped once the clip window reaches the corresponding range boundary.
    That makes ``-range 6 16 -clip -100 6`` behave like ``-clip -100 6.00001``
    instead of clipping nothing.
    """
    if n <= 0 or not (c1 > c0):
        return []
    span = abs(smax - smin)
    eps = (span if span > 0 else 1.0) * 1e-9
    out: List[int] = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        val = smin + t * (smax - smin)
        inside = (c0 < val < c1)
        if not inside and i == 0:
            # Everything below smin is clamped onto entry 0
            inside = (c0 - eps) <= smin <= (c1 + eps)
        if not inside and i == n - 1:
            # Everything above smax is clamped onto the last entry
            inside = (c0 - eps) <= smax <= (c1 + eps)
        if inside:
            out.append(i)
    return out


def apply_clip_alpha(lut: vtkLookupTable, value_range, clip) -> None:
    """Make the entries inside the clip window fully transparent."""
    lo, hi = float(value_range[0]), float(value_range[1])
    if not (hi > lo):
        return
    n = int(lut.GetNumberOfTableValues())
    for i in clipped_lut_indices(n, lo, hi, float(clip[0]), float(clip[1])):
        r, g, b, _ = lut.GetTableValue(i)
        lut.SetTableValue(i, r, g, b, 0.0)


def build_overlay_lut(colormap: int, opacity: float, value_range=None,
                      clip=(0.0, -1.0), inverse: bool = False,
                      discrete: int = 0) -> LookupTableWithEnabling:
    """The lookup table an overlay is drawn with.

    One place for what "overlay colours" means, shared by the surface and the
    volume viewer: the colormap at *opacity*, optionally reversed and flattened
    into bands, with the values inside *clip* made transparent.
    """
    lut = get_lookup_table(colormap, opacity)
    # Voxels a map has no value for — statistic maps carry NaN outside their
    # mask — must not be painted at all; VTK would use its dark red NaN colour
    lut.SetNanColor(0.0, 0.0, 0.0, 0.0)
    if inverse:
        invert_lut(lut)
    if discrete:
        apply_discrete(lut, discrete)
    if value_range is not None and value_range[1] > value_range[0]:
        lut.SetTableRange(float(value_range[0]), float(value_range[1]))
        apply_clip_alpha(lut, value_range, clip)
    return lut
