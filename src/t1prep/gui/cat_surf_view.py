#!/usr/bin/env python3
"""
CAT_SurfView — PySide6 + VTK port with right-side control panel

Features:
  • Load a hemisphere mesh (.gii).  The other one is found from the name in either
    direction ("lh."↔"rh.", "left"↔"right", "_hemi-L_"↔"_hemi-R_"), or split off a
    combined surface holding both hemispheres (CAT12 "mesh.central.*").
  • Optional overlay scalars (.gii; FreeSurfer morph: thickness/curv/sulc; or text one value/line).
  • Optional background scalars for curvature shading.
  • Six-view montage (lat/med/sup/inf/ant/post) by cloning actors with transforms.
  • Colormaps: C1, C2, C3, JET, FIRE, BIPOLAR, GRAY. Discrete levels, inverse, clip window.
  • Colorbar (VTK 9.5-compatible AddViewProp), optional stats in title.
    Overlays holding -log10(p) values (name contains "log", or -log) are labelled
    with the p-values they stand for (1.3 -> 0.05, 2 -> 0.01, 3 -> 0.001).
  • Right-side docked control panel: range, clip, colorbar toggle, overlay picker, opacity, bkg range, stats, inverse.
    For -log10(p) overlays it also offers the p<0.05 / p<0.01 / p<0.001 thresholds.
  • Optional volume window (-volume) with three orthogonal slices, linked to the
    surface in both directions: clicking the surface moves the slices, clicking
    a slice marks the closest surface point.  Right-click for the zoom levels.
    It is the same viewer as the standalone CAT_VolView tool (cat_vol_view.py).
  • Clicking a vertex reports it in the status bar: hemisphere, vertex number,
    mm position, overlay value (as a p-value for -log10(p) maps) and — with an
    atlas selected from the right-click menu — the region it belongs to.  The
    atlases are the .annot files shipped with T1Prep, and their region borders
    can be drawn on the surface (Atlas > Show region borders), as in
    cat_surf_results.
  • Clusters: the suprathreshold regions of an overlay as a table (peak, p, mm,
    vertices, mm², region), with a threshold to change, rows that mark the peak
    on the surface, and CSV export.
  • 'm' marks the strongest vertex of the overlay.
  • The right-click menu switches the surface (central, inflated, patch) and,
    separately, what it is shaded with — mean curvature, sulcal depth or
    nothing — as in cat_surf_results, whose shading it follows: a signed
    square root of the curvature, inverted for sulcal depth.  The shading is
    always that of the folded surface, so an inflated or flattened one keeps
    the relief instead of turning blank, and it follows a change of surface
    without being reselected.
  • The default value range follows cat_surf_results as well: symmetric for
    two-sided data, whole numbers for -log10(p) maps, and starting at the
    threshold when there is nothing below the negative one, so the whole
    colormap is spent on the values that are actually shown.  It also saves a screenshot; dropping a
    surface, an overlay or an annotation on the window opens it.  A flat patch
    is shown once per hemisphere rather than rotated into six overlapping
    views — the two mirroring each other, as cat_surf_results shows flatmaps —
    and each view shows only the hemisphere it stands for.
  • The mouse does not change the zoom (a right-click would otherwise leave the
    view zooming on every move, so a vertex could not be clicked); use '+'/'-',
    the Zoom entries of the menu, or '-free-zoom' to allow it again.
  • Keyboard: u/d/l/r rotate (Shift=±1°, Ctrl=180°), b flip, o reset, m peak,
    g screenshot, h for the list, plus the standard VTK keys the viewer does
    not claim (w/s wireframe/shaded).

Requires: vtk (>=9), PySide6; nibabel (for GIFTI fallback + FreeSurfer textures if VTK lacks vtkGIFTIReader).

Usage
-----
Preferred (uses the repo's venv wrapper):

    scripts/CAT_SurfView <mesh_or_overlay> [more_overlays...] [options]

Direct invocation:

    python src/t1prep/gui/cat_surf_view.py <mesh_or_overlay> [more_overlays...] [options]
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import re
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph  # noqa: F401  (sparse.csgraph)
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Headless / software-rendering guard.
# Must run BEFORE any Qt or VTK import because libGL.so / libEGL.so are
# loaded as a side-effect of importing those modules.  Setting the env vars
# afterwards has no effect.
# The shell wrapper (CAT_SurfView) also sets these before launching Python;
# the block below is a belt-and-suspenders fallback for direct invocations.
# ---------------------------------------------------------------------------
def _x_display_works() -> bool:
    """Return True only when an X server is actually reachable."""
    display = os.environ.get("DISPLAY")
    if not display:
        return False
    import subprocess
    try:
        return subprocess.run(
            ["xdpyinfo"], capture_output=True, timeout=2
        ).returncode == 0
    except Exception:
        return False


_headless = not _x_display_works() and not os.environ.get("WAYLAND_DISPLAY")
if _headless:
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "330")

# --- Qt setup (PySide6 only) ---
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QPainter, QColor, QPen, QBrush, QSurfaceFormat

# Qt compatibility shims
ORIENT_H = Qt.Orientation.Horizontal
DOCK_RIGHT = Qt.DockWidgetArea.RightDockWidgetArea
DOCK_LEFT = Qt.DockWidgetArea.LeftDockWidgetArea


# --- Import naming utilities ---
# (No local utils needed in this module)

# --- VTK imports (module-accurate for common wheels) ---
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkDoubleArray, vtkPoints, vtkVariant
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkFiltersGeneral import vtkCurvatures
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkWindowToImageFilter,  # on many wheels this class lives here
)
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
# Optional GIFTI reader (not on all wheels)
try:
    from vtkmodules.vtkIOGeometry import vtkGIFTIReader
    HAVE_VTK_GIFTI = True
except Exception:
    vtkGIFTIReader = None
    HAVE_VTK_GIFTI = False

# Saving screenshots
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkCellPicker
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter

# The volume window is shared with the standalone CAT_VolView tool.  It also
# selects the QVTKRWIBase used below, so it is imported before the widget.
try:
    from .cat_vol_view import (
        VolumeViewerWindow, ask_for_files, install_qt_message_filter,
        qt_application, running_as_app,
    )
except ImportError:  # direct invocation as a script (no package context)
    from cat_vol_view import (
        VolumeViewerWindow, ask_for_files, install_qt_message_filter,
        qt_application, running_as_app,
    )

# The control panel is shared with the volume viewer
try:
    from .make_apps import ensure_apps_exist
except ImportError:  # direct invocation as a script
    from make_apps import ensure_apps_exist

try:
    from .controls import ControlPanel, LOGP_THRESHOLDS
except ImportError:  # direct invocation as a script
    from controls import ControlPanel, LOGP_THRESHOLDS

# Qt interactor & backends
import vtkmodules.qt as vtk_qt
vtk_qt.QVTKRWIBase = "QOpenGLWidget"
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkRenderingOpenGL2   # registers OpenGL2 backend (fixes vtkShaderProperty)
import vtkmodules.vtkRenderingFreeType  # text rendering for labels/ScalarBar

# --- Defaults ---
DEFAULT_WINDOW_SIZE = (1800, 800)

# Colormaps live in a shared module (the volume viewer needs them too)
try:
    from .colormaps import (
        C1, C2, C3, JET, HOT, FIRE, BIPOLAR, GRAY,
        COLORMAP_NAMES, COLORMAP_ORDER,
        LOG10_P005, LookupTableWithEnabling, apply_discrete, build_overlay_lut,
        clipped_lut_indices, format_p_value_label, get_lookup_table, invert_lut,
        logp_colorbar_ticks,
    )
except ImportError:  # direct invocation as a script
    from colormaps import (
        C1, C2, C3, JET, HOT, FIRE, BIPOLAR, GRAY,
        COLORMAP_NAMES, COLORMAP_ORDER,
        LOG10_P005, LookupTableWithEnabling, apply_discrete, build_overlay_lut,
        clipped_lut_indices, format_p_value_label, get_lookup_table, invert_lut,
        logp_colorbar_ticks,
    )

# ---- Naming helpers ----
# Surface types that CAT12/FreeSurfer put in the second dot-token of a mesh
# file name (lh.central.subj.gii, mesh.inflated.freesurfer.gii, …).
MESH_TYPE_TOKENS = frozenset(
    {'central', 'pial', 'white', 'inflated', 'sphere', 'patch', 'mc', 'sqrtsulc'}
)

#: The even grey a surface gets when there is nothing to shade it with.
#: cat_surf_results paints 0.5 there; the 3D pipeline gamma-encodes what it is
#: given, so 0.5 would come out almost white — this value is what renders as
#: the same mid-grey (and matches the clipped band of the colorbar).
UNDERLAY_PLAIN_GREY = (0.33, 0.33, 0.33)

#: Darkest and lightest grey the shading uses.  The relief comes out of
#: shade_from_curvature(); this only says how much of it to show.
UNDERLAY_GREYS = (0.30, 0.80)

#: Surfaces the viewer offers to switch between, in the order they are shown.
#: A subset of the tokens above: 'mc' and 'sqrtsulc' name scalar files rather
#: than meshes, and sphere/pial/white add little next to these three.
SWITCHABLE_SURFACES = ('central', 'inflated', 'patch')


def detect_naming_scheme(filename: str) -> bool:
    """
    Detect whether a filename uses BIDS naming convention.
    
    Args:
        filename: The filename to analyze
        
    Returns:
        bool: True if BIDS naming, False if FreeSurfer naming
    """
    # BIDS naming patterns
    bids_patterns = [
        '_hemi-L_', '_hemi-R_',  # hemisphere indicators
        '_space-',  # space indicators
        '_desc-',   # description indicators
        '_label-',  # label indicators
        '.surf.gii', '.shape.gii', '.label.gii'  # BIDS surface extensions
    ]
    
    # FreeSurfer naming patterns
    freesurfer_patterns = [
        'lh.', 'rh.',  # hemisphere prefixes
        '.central.', '.pial.', '.white.',  # surface types
        '.thickness.', '.pbt.',  # shape types
        '.annot'  # annotation files
    ]
    
    filename_lower = filename.lower()
    # Prefer FreeSurfer when both styles appear (e.g., lh.* files carrying _desc- tags)
    if any(p in filename_lower for p in freesurfer_patterns):
        return False
    if any(p in filename_lower for p in bids_patterns):
        return True
    # Default to FreeSurfer if no clear pattern found
    return False

def convert_filename_to_mesh(overlay_filename: str) -> str:
    """
    Convert an overlay filename to the corresponding mesh filename.
    
    Args:
        overlay_filename: The overlay file (e.g., thickness, pbt, etc.)
        
    Returns:
        str: The corresponding mesh filename
    """
    overlay_path = Path(overlay_filename)

    # SPM analysis overlays map to the template mesh they were computed on
    if is_spm_surface_overlay(overlay_filename):
        tpl = _template_mesh_for_points(_gifti_scalar_count(overlay_filename))
        if tpl is not None:
            return str(tpl)
        template_dir = _get_template_surface_dir()
        lh_template = template_dir / 'lh.central.freesurfer.gii'
        if lh_template.exists():
            return str(lh_template)

    # Combined-hemisphere 'mesh.' overlays
    # e.g. s12.mesh.thickness.resampled_32k.HR075_MPRAGE.gii
    #   -> try lh.central.<suffix>.gii, then progressively strip prefix tokens,
    #      then fall back to the 32k template mesh.
    parsed = _parse_mesh_combined_overlay(overlay_filename)
    if parsed is not None:
        _ov_type, suffix = parsed
        # Try exact suffix first, then progressively strip leading tokens
        # e.g. suffix = 'resampled_32k.HR075_MPRAGE'
        # try: lh.central.resampled_32k.HR075_MPRAGE.gii
        # try: lh.central.HR075_MPRAGE.gii
        suffix_tokens = suffix.split('.') if suffix else []
        for start_idx in range(len(suffix_tokens) + 1):
            sub_suffix = '.'.join(suffix_tokens[start_idx:])
            if sub_suffix:
                mesh_name = f"lh.central.{sub_suffix}.gii"
            else:
                mesh_name = "lh.central.gii"
            cand = overlay_path.parent / mesh_name
            if cand.exists():
                return str(cand)
        # Fall back to 32k template (resampled_32k data lives on this mesh)
        template_dir = _get_template_surface_dir()
        lh_template = template_dir / 'lh.central.freesurfer.gii'
        if lh_template.exists():
            return str(lh_template)

    # Prefer explicit non-BIDS mapping for lh/rh thickness (and pbt) overlays
    # Examples:
    #   lh.thickness.name     -> lh.central.name.gii
    #   rh.thickness.name.gii -> rh.central.name.gii
    #   lh.thickness          -> lh.central.gii
    def _fs_thickness_to_mesh(nm: str) -> Optional[str]:
        base = nm
        if base.lower().endswith('.gii'):
            base = Path(base).stem
        m = re.match(r'^(lh|rh)\.(thickness|pbt)(?:\.(.+))?$', base, flags=re.IGNORECASE)
        if not m:
            return None
        hemi = m.group(1).lower()
        suffix = m.group(3)
        if suffix:
            return f"{hemi}.central.{suffix}.gii"
        return f"{hemi}.central.gii"

    if not detect_naming_scheme(overlay_filename):
        mesh_name = _fs_thickness_to_mesh(overlay_path.name)
        if mesh_name is not None:
            return str(overlay_path.parent / mesh_name)

    # Try BIDS-style conversion first; only succeed when a hemisphere token exists
    mesh_candidate: Optional[Path] = None
    if detect_naming_scheme(overlay_filename):
        name_parts = overlay_path.stem.split('_')
        hemi_part = next((p for p in name_parts if p.startswith('hemi-')), None)
        if hemi_part:
            base_parts = [p for p in name_parts if not p.startswith('hemi-') and not p.startswith('desc-')]
            base_name = '_'.join(base_parts)
            mesh_filename = f"{base_name}_{hemi_part}_space-MNI152NLin2009cAsym_desc-midthickness.surf.gii"
            mesh_candidate = overlay_path.parent / mesh_filename

    # FreeSurfer naming: convert overlay to central surface
    # Accept both with and without a subject token:
    #  - lh.thickness.name   -> lh.central.name.gii
    #  - lh.thickness        -> lh.central.gii
    def _fs_overlay_to_mesh(nm: str) -> Optional[str]:
        # Handle overlays that already include a .gii extension.
        # Example: lh.thickness.subj.gii -> lh.central.subj.gii
        try:
            if nm.lower().endswith('.gii'):
                nm = Path(nm).stem
        except Exception:
            pass
        hemi = None
        remaining = None
        if nm.startswith('lh.'):
            hemi = 'lh'; remaining = nm[3:]
        elif nm.startswith('rh.'):
            hemi = 'rh'; remaining = nm[3:]
        else:
            parts_f = nm.split('.')
            if len(parts_f) >= 2 and parts_f[0] in ('lh', 'rh'):
                hemi = parts_f[0]
                remaining = '.'.join(parts_f[1:])
            else:
                return None
        tokens = [t for t in remaining.split('.') if t]
        if not tokens:
            return None
        mesh_types = {'central','pial','white','inflated','sphere','patch','mc','sqrtsulc'}
        if tokens[0] in mesh_types:
            return None
        base = '.'.join(tokens[1:]) if len(tokens) > 1 else ''
        if base:
            return f"{hemi}.central.{base}.gii"
        return f"{hemi}.central.gii"

    if mesh_candidate is None:
        mesh_name = _fs_overlay_to_mesh(overlay_path.name)
        if mesh_name is not None:
            mesh_candidate = overlay_path.parent / mesh_name

    result = mesh_candidate or overlay_path
    # No usable surface from the name — statistic results (TFCE_*, logP_*, …)
    # follow no convention and may sit in a folder without any surface.  Their
    # value count still identifies the template they were computed on.
    if _gifti_point_count(str(result)) <= 0:
        tpl = _template_mesh_for_points(_gifti_scalar_count(str(overlay_path)))
        if tpl is not None:
            return str(tpl)
    return str(result)

def is_overlay_file(filename: str) -> bool:
    """Heuristic check whether a path is an overlay (texture/label) rather than a mesh."""
    filename_only = Path(filename).name
    filename_lower = filename_only.lower()

    parts = filename_lower.split('.')
    mesh_types = MESH_TYPE_TOKENS
    # Case A: lh.kind.subject (>=3 parts)
    if len(parts) >= 3 and parts[0] in ('lh', 'rh'):
        if parts[1] not in mesh_types:
            return True
    # Case B: lh.kind (exactly 2 parts, no subject)
    if len(parts) == 2 and parts[0] in ('lh', 'rh'):
        if parts[1] not in mesh_types:
            return True

    # SPM surface analysis overlays
    if is_spm_surface_overlay(filename):
        return True

    # A .gii that holds values but no surface can only be an overlay
    if _is_scalars_only_gifti(filename):
        return True

    # Combined-hemisphere 'mesh.' overlays (e.g. s12.mesh.thickness.rest.gii)
    if _parse_mesh_combined_overlay(filename) is not None:
        return True

    overlay_patterns = [
        '_desc-thickness.', '_desc-pbt.',  # BIDS shape files
        '.annot',  # FreeSurfer annotation
        '_label-',  # BIDS label files
        '.txt'  # Text overlays
    ]
    return any(p in filename_lower for p in overlay_patterns)

def detect_overlay_kind(filename: str) -> Optional[str]:
    """Detect overlay kind such as 'thickness' or 'pbt' from filename."""
    name = Path(filename).name.lower()
    if '_desc-thickness' in name or '.thickness.' in name or name.endswith('thickness'):
        return 'thickness'
    if '_desc-pbt' in name or '.pbt.' in name or name.endswith('pbt'):
        return 'pbt'
    return None


def default_overlay_range(values, threshold: float = 0.0,
                          logp: bool = False) -> Optional[Tuple[float, float]]:
    """The value range an overlay is coloured over, as cat_surf_results picks it.

    Three rules, taken from ``cat_surf_results.m``:

    * data with negative values is scaled symmetrically, so the two tails get
      the same amount of colour;
    * a -log10(p) map is rounded outwards to whole numbers;
    * when nothing lies below the negative threshold, the range *starts* at
      the threshold — otherwise the lower part of the colormap is spent on
      values that are hidden anyway.

    Args:
        values: The overlay values (both hemispheres together).
        threshold: Where the map is thresholded, i.e. the upper edge of the
            clip window; 0 for an unthresholded map.
        logp: True for a -log10(p) map.

    Returns:
        (low, high), or None when there is nothing to scale.
    """
    data = np.asarray(values, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    low, high = float(data.min()), float(data.max())
    if low < 0:
        edge = max(abs(low), abs(high))
        low, high = -edge, edge
    if logp:
        low, high = float(math.floor(low)), float(math.ceil(high))
    threshold = abs(float(threshold))
    if threshold > 0 and float(data.min()) > -threshold:
        # One-sided: the colours start where the map does
        low = threshold
    if not high > low:
        return None
    return (low, high)


def atlas_border_lines(poly, labels) -> vtkPolyData:
    """The boundaries between atlas regions, as line segments on the surface.

    cat_surf_results draws the 0.5-isocontour of every region in turn; the
    same lines come out of one pass over the triangles, taking the segment
    between the midpoints of the edges whose ends belong to different regions
    (a triangle with three different regions is closed through its centre).

    Args:
        poly: The surface the labels belong to.
        labels: One region number per vertex.

    Returns:
        Polydata holding the border segments; empty when there is nothing to
        draw.
    """
    border = vtkPolyData()
    points = vtkPoints()
    lines = vtkCellArray()
    border.SetPoints(points)
    border.SetLines(lines)
    if poly is None or labels is None or poly.GetNumberOfPoints() != len(labels):
        return border
    coordinates = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    faces = vtk_to_numpy(poly.GetPolys().GetData())
    if faces.size == 0:
        return border
    triangles = faces.reshape(-1, 4)[:, 1:]
    labels = np.asarray(labels)
    corners = labels[triangles]

    midpoints = {}
    for first, second in ((0, 1), (1, 2), (2, 0)):
        midpoints[(first, second)] = 0.5 * (coordinates[triangles[:, first]]
                                            + coordinates[triangles[:, second]])
    centres = coordinates[triangles].mean(axis=1)

    segments = []
    differs = {pair: corners[:, pair[0]] != corners[:, pair[1]]
               for pair in ((0, 1), (1, 2), (2, 0))}
    crossings = sum(differs.values())          # 0, 2 or 3 edges per triangle
    two = crossings == 2
    if two.any():
        cut = [pair for pair in differs]
        for i, first in enumerate(cut):
            for second in cut[i + 1:]:
                both = two & differs[first] & differs[second]
                if both.any():
                    segments.append((midpoints[first][both],
                                     midpoints[second][both]))
    three = crossings == 3
    if three.any():
        for pair in differs:
            segments.append((midpoints[pair][three], centres[three]))

    if not segments:
        return border
    starts = np.vstack([start for start, _ in segments])
    ends = np.vstack([end for _, end in segments])
    coords = np.empty((2 * len(starts), 3), dtype=float)
    coords[0::2] = starts
    coords[1::2] = ends
    points.SetData(numpy_to_vtk(coords, deep=True))
    for index in range(len(starts)):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2 * index)
        lines.InsertCellPoint(2 * index + 1)
    return border


def available_surface_atlases() -> List[Tuple[str, str]]:
    """The surface atlases shipped with T1Prep, as (name, left annot) pairs.

    Only the left file is listed; the right one is found from its name, the
    same way the hemispheres of a mesh are.
    """
    folder = Path(__file__).resolve().parent.parent / 'data' / 'atlases_surfaces_32k'
    out: List[Tuple[str, str]] = []
    if not folder.is_dir():
        return out
    for annot in sorted(folder.glob('lh.*.annot')):
        out.append((annot.name[len('lh.'):-len('.annot')], str(annot)))
    return out


def read_annotation(path: str) -> Tuple["np.ndarray", List[str]]:
    """Vertex labels and region names of a FreeSurfer ``.annot`` file.

    Returns:
        The label of every vertex as an index into the returned names, and the
        names themselves.  Unlabelled vertices carry -1.

    Raises:
        RuntimeError: when the file cannot be read.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError("reading .annot files needs nibabel")
    try:
        labels, _ctab, names = nib.freesurfer.io.read_annot(str(path))
    except Exception as exc:
        raise RuntimeError(f"cannot read {os.path.basename(str(path))}: {exc}")
    decoded = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return np.asarray(labels), [n.strip() for n in decoded]


def vertex_areas(poly) -> "np.ndarray":
    """Surface area belonging to each vertex (a third of its triangles).

    Cluster sizes are reported in mm2 rather than in vertices: a 32k mesh has
    vertices of unequal size, and a count means nothing across resolutions.
    """
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    polys = vtk_to_numpy(poly.GetPolys().GetData())
    areas = np.zeros(len(points), dtype=float)
    if polys.size == 0:
        return areas
    # vtkCellArray stores (n, i0, i1, …) per cell; the meshes here are triangles
    triangles = polys.reshape(-1, 4)[:, 1:]
    a, b, c = (points[triangles[:, 0]], points[triangles[:, 1]],
               points[triangles[:, 2]])
    face = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    for column in range(3):
        np.add.at(areas, triangles[:, column], face / 3.0)
    return areas


def find_surface_clusters(poly, values, threshold: float,
                          min_area: float = 0.0) -> List[dict]:
    """Connected regions of a surface where |values| exceeds *threshold*.

    Both tails are searched, and a cluster never mixes them: neighbouring
    vertices only join when they have the same sign, so a positive and a
    negative blob that touch stay two findings.

    Args:
        poly: The hemisphere the values belong to.
        values: One value per vertex.
        threshold: Vertices with ``|value| > threshold`` take part.
        min_area: Drop clusters smaller than this (mm2).

    Returns:
        One dict per cluster with its peak value, peak vertex, area in mm2 and
        number of vertices, ordered by descending peak.
    """
    values = np.asarray(values, dtype=float)
    polys = vtk_to_numpy(poly.GetPolys().GetData())
    if polys.size == 0 or values.size == 0:
        return []
    triangles = polys.reshape(-1, 4)[:, 1:]
    areas = vertex_areas(poly)

    above = np.abs(values) > float(threshold)
    if not above.any():
        return []
    # Edges of the mesh, kept only where both ends are in the same tail
    edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]],
                       triangles[:, [2, 0]]])
    keep = (above[edges[:, 0]] & above[edges[:, 1]]
            & (np.sign(values[edges[:, 0]]) == np.sign(values[edges[:, 1]])))
    edges = edges[keep]

    count = len(values)
    graph = sparse.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    number, labels = sparse.csgraph.connected_components(graph, directed=False)
    labels = np.where(above, labels, -1)

    clusters = []
    for label in range(number):
        members = np.flatnonzero(labels == label)
        if members.size == 0:
            continue
        area = float(areas[members].sum())
        if area < min_area:
            continue
        peak = int(members[np.argmax(np.abs(values[members]))])
        clusters.append({
            'peak_value': float(values[peak]),
            'peak_vertex': peak,
            'vertices': int(members.size),
            'area': area,
        })
    clusters.sort(key=lambda c: abs(c['peak_value']), reverse=True)
    return clusters


def is_logp_overlay(filename: Optional[str]) -> bool:
    """Return True when the overlay holds -log10(p) values.

    Same heuristic as ``cat_surf_results``: the file name contains 'log'
    (e.g. ``logP_age_pFWE0.05_k0.gii``).
    """
    if not filename:
        return False
    return 'log' in Path(filename).name.lower()


def _is_gifti_mesh_by_name(filename: str) -> bool:
    """Return True when the filename alone indicates a surface mesh file.

    Uses naming conventions instead of reading file contents, so it is fast
    and avoids the expensive I/O of :func:`is_gifti_mesh_file` for common
    FreeSurfer / CAT12 and BIDS mesh filenames.

    Recognised patterns
    -------------------
    * ``lh.central.<subject>.gii``, ``rh.pial.<subject>.gii``, etc.
      (``lh.``/``rh.`` prefix followed by a mesh-type token)
    * ``*.surf.gii``  (BIDS surface files)

    Args:
        filename: Path or basename to inspect.

    Returns:
        bool: True if the name suggests a mesh file.
    """
    name = Path(filename).name.lower()
    # BIDS surface files always end with .surf.gii
    if name.endswith('.surf.gii'):
        return True
    if not name.endswith('.gii'):
        return False
    stem = name[:-4]  # strip .gii
    parts = stem.split('.')
    mesh_types = {'central', 'pial', 'white', 'inflated', 'sphere', 'patch', 'mc', 'sqrtsulc'}
    # FreeSurfer / CAT12: lh.<mesh_type>[.<subject>]
    if len(parts) >= 2 and parts[0] in ('lh', 'rh') and parts[1] in mesh_types:
        return True
    return False


# Vertices per hemisphere of the shipped resampling templates.  Overlays that
# were resampled to one of these meshes can be matched by value count alone,
# which is the only reliable clue when the filename says nothing about the
# underlying surface (CAT12/SPM statistic folders, for example).
TEMPLATE_HEMI_POINTS = {
    4002: 'templates_surfaces_4k',
    32492: 'templates_surfaces_32k',
    163842: 'templates_surfaces_164k',
}
DEFAULT_TEMPLATE_DIR = 'templates_surfaces_32k'


def _get_template_surface_dir(n_points: Optional[int] = None) -> Path:
    """Return the directory holding the resampling template surfaces.

    Args:
        n_points: Optional vertex count of the data to display.  Accepts either
            a per-hemisphere count (e.g. 32492) or a combined LH+RH count
            (e.g. 64984) and selects the matching template resolution.
            Without it, the 32k templates used by T1Prep are returned.

    Returns:
        Path to the ``templates_surfaces_*`` directory.
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    if n_points:
        for hemi_pts, dirname in TEMPLATE_HEMI_POINTS.items():
            if n_points in (hemi_pts, 2 * hemi_pts):
                return data_dir / dirname
    return data_dir / DEFAULT_TEMPLATE_DIR


#: How a file name says which hemisphere it holds, left form and right form
HEMI_MARKERS = (('lh.', 'rh.'), ('_hemi-L_', '_hemi-R_'), ('left', 'right'))


def hemisphere_of(filename) -> Optional[str]:
    """``'lh'`` or ``'rh'`` when the name names a hemisphere, else None.

    Covers the FreeSurfer/CAT12 (``lh.``), BIDS (``_hemi-L_``) and plain
    (``left``) conventions.
    """
    name = Path(filename).name
    for left, right in HEMI_MARKERS:
        if left in name:
            return 'lh'
        if right in name:
            return 'rh'
    return None


def _hemi_counterpart(path: Path) -> Optional[Path]:
    """Return the file of the other hemisphere, in either direction.

    Picking ``rh.central.subj.gii`` looks for ``lh.central.subj.gii`` just as
    the other way round; which side the given file is decides where the pair
    ends up (see :func:`read_mesh_pair`).  The returned path is not checked
    for existence, and files that name no hemisphere return None.
    """
    name = path.name
    for left, right in HEMI_MARKERS:
        if left in name:
            return path.with_name(name.replace(left, right))
        if right in name:
            return path.with_name(name.replace(right, left))
    return None


def order_by_hemisphere(filename, own, other):
    """Sort a pair of loaded objects into (left, right).

    *own* belongs to *filename* and *other* to its counterpart, so the name
    decides which of the two is the left hemisphere.
    """
    if other is None:
        return own, None
    if hemisphere_of(filename) == 'rh':
        return other, own
    return own, other


def _split_scalars(arr: vtkDoubleArray, n_left: int, n_right: int) -> Tuple[vtkDoubleArray, vtkDoubleArray]:
    """Split a concatenated LH+RH scalar array into per-hemisphere arrays."""
    values = vtk_to_numpy(arr)

    def _to_vtk(chunk) -> vtkDoubleArray:
        out = vtkDoubleArray()
        out.SetNumberOfTuples(len(chunk))
        for i, v in enumerate(chunk):
            out.SetValue(i, float(v))
        return out

    return _to_vtk(values[:n_left]), _to_vtk(values[n_left:n_left + n_right])


@lru_cache(maxsize=64)
def _gifti_point_count(filename: str) -> int:
    """Return the number of vertices of a GIFTI surface (0 when unreadable)."""
    try:
        poly = read_gifti_mesh(filename)
        return int(poly.GetNumberOfPoints()) if poly is not None else 0
    except Exception:
        return 0


@lru_cache(maxsize=64)
def _gifti_scalar_count(filename: str) -> int:
    """Return the number of scalar values in a ``.gii`` overlay.

    Returns 0 for surfaces (they hold geometry only) and for unreadable files.
    The array dimensions are taken from the XML header, so the companion
    ``.dat`` of an ExternalFileBinary pair does not have to be read.
    """
    if not str(filename).lower().endswith('.gii'):
        return 0
    try:
        g = _nib_load_gifti(filename)
    except Exception:
        return 0
    for d in getattr(g, 'darrays', []):
        code = int(getattr(d, 'intent', getattr(d, 'intent_code', -1)) or -1)
        if code in (1008, 1009):  # POINTSET / TRIANGLE – geometry, not values
            continue
        try:
            dims = [int(v) for v in getattr(d, 'dims', []) if int(v) > 0]
            if dims:
                return int(np.prod(dims))
            return int(np.asarray(d.data).size)
        except Exception:
            continue
    return 0


def _is_scalars_only_gifti(filename: str) -> bool:
    """True for an existing ``.gii`` that holds values but no surface.

    Such a file can only be an overlay, whatever it is called.  CAT12/TFCE
    statistic results (``TFCE_log_pFWE_0001.gii`` plus its ``.dat``) are the
    common case: the name follows no convention, and the ``SPM.mat`` that
    would otherwise mark the directory is not always kept next to them.
    """
    try:
        if not str(filename).lower().endswith('.gii') or not Path(filename).exists():
            return False
    except Exception:
        return False
    return _gifti_point_count(str(filename)) <= 0 and _gifti_scalar_count(str(filename)) > 0


def _mesh_point_capacity(mesh_path: Path) -> Tuple[int, int]:
    """Return (points of *mesh_path*, points of both hemispheres together).

    The second value equals the first when no opposite-hemisphere file exists,
    so a caller can simply test its scalar count against both numbers.
    """
    n_l = _gifti_point_count(str(mesh_path))
    if n_l <= 0:
        return (0, 0)
    other = _hemi_counterpart(mesh_path)
    n_r = _gifti_point_count(str(other)) if (other is not None and other.exists()) else 0
    return (n_l, n_l + n_r)


def _template_mesh_for_points(n_points: int, hemi: str = 'lh') -> Optional[Path]:
    """Return the template central surface matching *n_points*, if any.

    Args:
        n_points: Per-hemisphere or combined LH+RH vertex count.
        hemi: ``'lh'`` or ``'rh'``.

    Returns:
        Path to ``<hemi>.central.freesurfer.gii`` of the matching template
        resolution, or None when the count matches no shipped template.
    """
    if not n_points:
        return None
    for hemi_pts, dirname in TEMPLATE_HEMI_POINTS.items():
        if n_points in (hemi_pts, 2 * hemi_pts):
            cand = (Path(__file__).resolve().parent.parent / 'data' / dirname
                    / f'{hemi}.central.freesurfer.gii')
            return cand if cand.exists() else None
    return None


def is_spm_surface_overlay(filename: str) -> bool:
    """Check whether a file is a surface result of an SPM/CAT12 analysis.

    SPM stores surface-based analysis results as .gii/.dat pairs where:
    - The .gii file contains the GIFTI metadata referencing external binary data
    - The .dat file contains the actual scalar values
    - Data covers the template mesh the analysis was run on

    Three cues are accepted, in decreasing specificity:

    1. A known SPM result filename (``spmT_0001.gii``, ``con_0001.gii``, …).
    2. A ``.dat`` companion next to an ``SPM.mat``.
    3. Any non-mesh ``.gii`` inside a directory that holds an ``SPM.mat``.
       CAT12 writes its thresholded results there under free-form names such
       as ``logP_age_(polynomial_of_degree_3)_pFWE0.1_k0.gii``, which no
       naming rule can recognise.

    Results copied away from their ``SPM.mat`` are still recognised by
    :func:`is_overlay_file`, which falls back to the file content.

    Args:
        filename: Path to a potential SPM surface overlay file

    Returns:
        True if the file looks like a surface analysis result
    """
    p = Path(filename)
    if p.suffix.lower() != '.gii':
        return False
    # Known SPM analysis filename patterns
    spm_patterns = [
        r'^spmT_\d+\.gii$',     # T-statistic maps
        r'^spmF_\d+\.gii$',     # F-statistic maps
        r'^con_\d+\.gii$',      # Contrast images
        r'^ess_\d+\.gii$',      # Extra sum of squares
        r'^beta_\d+\.gii$',     # Beta (parameter) images
        r'^ResMS\.gii$',        # Residual mean square
        r'^RPV\.gii$',          # Resels per voxel
    ]
    name = p.name
    pattern_match = any(re.match(pat, name, re.IGNORECASE) for pat in spm_patterns)
    if pattern_match:
        return True
    # Any result file living next to an SPM.mat, provided it is not itself a
    # surface that happens to be stored in the analysis directory.
    if (p.parent / 'SPM.mat').exists() and not _is_gifti_mesh_by_name(name):
        if p.with_suffix('.dat').exists():
            return True
        return p.exists()
    return False


def _parse_mesh_combined_overlay(filename: str) -> Optional[Tuple[str, str]]:
    """Parse a 'mesh' combined-hemisphere overlay filename.

    Recognizes CAT12-style patterns where 'mesh' replaces 'lh'/'rh' to
    indicate combined LH+RH data:
        mesh.thickness.rest.gii
        s12.mesh.thickness.rest.gii   (with smoothing prefix)

    Args:
        filename: The overlay filename (basename or full path).

    Returns:
        (overlay_type, suffix) if the file is a combined-hemisphere overlay,
        None otherwise.  *suffix* contains the remaining dot-separated tokens
        after the overlay type (e.g. 'resampled_32k.HR075_MPRAGE').
    """
    name = Path(filename).name
    if name.lower().endswith('.gii'):
        name = name[:-4]
    parts = name.split('.')

    mesh_types = MESH_TYPE_TOKENS

    # Find the 'mesh' token
    mesh_idx = None
    for i, p in enumerate(parts):
        if p.lower() == 'mesh':
            mesh_idx = i
            break
    if mesh_idx is None:
        return None

    # After 'mesh' we expect an overlay-type token (not a mesh type)
    if mesh_idx + 1 >= len(parts):
        return None
    overlay_type = parts[mesh_idx + 1].lower()
    if overlay_type in mesh_types:
        return None  # e.g. mesh.central.* is a mesh, not an overlay

    # Remaining tokens form the suffix
    suffix_parts = parts[mesh_idx + 2:]
    suffix = '.'.join(suffix_parts) if suffix_parts else ''
    return (overlay_type, suffix)


def split_combined_mesh(
    poly: vtkPolyData,
    n_left: int,
) -> Tuple[vtkPolyData, vtkPolyData]:
    """Split a combined LH+RH mesh into separate left and right polydata.

    Vertices 0..n_left-1 are assigned to LH, the rest to RH.
    Faces are partitioned based on whether all their vertex indices belong
    to the left or right range.

    Args:
        poly: Combined polydata with interleaved LH+RH vertices.
        n_left: Number of vertices belonging to the left hemisphere.

    Returns:
        (poly_l, poly_r): Separate vtkPolyData objects.
    """
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy as _v2n

    pts_all = poly.GetPoints()
    n_total = pts_all.GetNumberOfPoints()
    n_right = n_total - n_left

    # --- Extract vertex coordinates ---
    coords = np.empty((n_total, 3), dtype=float)
    for i in range(n_total):
        coords[i] = pts_all.GetPoint(i)

    # --- Extract face indices ---
    polys = poly.GetPolys()
    polys.InitTraversal()
    from vtkmodules.vtkCommonCore import vtkIdList
    id_list = vtkIdList()
    faces_list = []
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            faces_list.append((
                id_list.GetId(0),
                id_list.GetId(1),
                id_list.GetId(2),
            ))
    faces = np.array(faces_list, dtype=np.int64)

    def _build_poly(v_start: int, v_count: int) -> vtkPolyData:
        v_end = v_start + v_count
        c = coords[v_start:v_end]
        # Faces that belong entirely to this vertex range
        mask = np.all((faces >= v_start) & (faces < v_end), axis=1)
        f = faces[mask] - v_start  # reindex to 0-based
        pts_v = vtkPoints()
        pts_v.SetNumberOfPoints(v_count)
        for i in range(v_count):
            pts_v.SetPoint(i, float(c[i, 0]), float(c[i, 1]), float(c[i, 2]))
        cells_v = vtkCellArray()
        for tri in f:
            cells_v.InsertNextCell(3)
            cells_v.InsertCellPoint(int(tri[0]))
            cells_v.InsertCellPoint(int(tri[1]))
            cells_v.InsertCellPoint(int(tri[2]))
        pd = vtkPolyData()
        pd.SetPoints(pts_v)
        pd.SetPolys(cells_v)
        return pd

    return _build_poly(0, n_left), _build_poly(n_left, n_right)


def is_combined_hemisphere_mesh(filename: str) -> bool:
    """True for a surface file name that holds both hemispheres.

    CAT12 replaces the ``lh``/``rh`` token with ``mesh`` for files covering
    both sides, e.g. ``mesh.central.Template_T1.gii`` or
    ``mesh.inflated.freesurfer.gii``.
    """
    parts = [p.lower() for p in Path(filename).name.split('.')]
    return any(
        p == 'mesh' and parts[i + 1] in MESH_TYPE_TOKENS
        for i, p in enumerate(parts[:-1])
    )


def split_hemispheres(poly: vtkPolyData, filename: str) -> Optional[Tuple[vtkPolyData, vtkPolyData]]:
    """Split a surface that stores LH and RH back to back, if it is one.

    Three cues are accepted: the CAT12 ``mesh.*`` naming of surfaces
    (``mesh.central.*``) and of overlays carrying their geometry
    (``s12.mesh.thickness.*``), and a vertex count of twice a shipped
    template.  The split is only used when the two halves are self-contained:
    a single hemisphere would lose every triangle crossing the split, which is
    how a false cue is detected.

    Returns:
        (poly_l, poly_r), or None when the surface is a single hemisphere.
    """
    n_total = poly.GetNumberOfPoints()
    half = n_total // 2
    if n_total < 2 or n_total % 2:
        return None
    if not (
        is_combined_hemisphere_mesh(filename)
        or _parse_mesh_combined_overlay(filename) is not None
        or half in TEMPLATE_HEMI_POINTS
    ):
        return None
    try:
        poly_l, poly_r = split_combined_mesh(poly, half)
    except Exception:
        return None
    kept = poly_l.GetNumberOfPolys() + poly_r.GetNumberOfPolys()
    if poly_r.GetNumberOfPolys() == 0 or kept != poly.GetNumberOfPolys():
        return None
    return poly_l, poly_r


def read_mesh_pair(mesh_path: str) -> Tuple[vtkPolyData, Optional[vtkPolyData]]:
    """Load a surface together with the surface of the other hemisphere.

    The counterpart is either a sibling file (``lh.`` <-> ``rh.``, in either
    direction) or the second half of a combined ``mesh.*`` surface.  Without
    one, the right entry is None and only the left column of the montage is
    drawn.
    """
    poly = read_gifti_mesh(str(mesh_path))
    pair = split_hemispheres(poly, str(mesh_path))
    if pair is not None:
        return pair
    other = _hemi_counterpart(Path(mesh_path))
    if other is not None and other.exists():
        try:
            return order_by_hemisphere(mesh_path, poly, read_gifti_mesh(str(other)))
        except Exception:
            return poly, None
    return poly, None


# ---- I/O helpers ----
def _nib_load_gifti(filename: str):
    """Load a ``.gii`` file with nibabel, repairing a stale external-file
    reference if needed.

    CAT12/CAT-Surface write large GIFTI arrays with ``Encoding=
    "ExternalFileBinary"``, storing the actual bytes in a sibling ``.dat``
    file that the ``.gii`` XML references by name (``ExternalFileName``).
    By convention that ``.dat`` file shares the exact basename of the
    ``.gii`` file.  If the pair gets renamed after the fact (e.g. a subject
    ID gets appended to both filenames by a batch-rename step) without the
    XML being regenerated, the internal reference points at a filename that
    no longer exists — even though the correctly-named ``.dat`` sits right
    next to the ``.gii``.  nibabel then raises ``GiftiParseError: Cannot
    locate external file ...``.

    Detect that case and retry against a patched copy of the XML that
    points ``ExternalFileName`` at ``<gii-basename>.dat`` instead.
    """
    import nibabel as nib
    try:
        return nib.load(filename)
    except Exception as e:
        if 'external file' not in str(e).lower():
            raise
        gii_path = Path(filename)
        actual_dat = gii_path.with_suffix('.dat')
        if not actual_dat.exists():
            raise
        try:
            text = gii_path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            raise e
        patched, n = re.subn(
            r'ExternalFileName="[^"]*"',
            f'ExternalFileName="{actual_dat.resolve()}"',
            text,
        )
        if n == 0:
            raise e
        import tempfile
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.gii', delete=False, encoding='utf-8'
            ) as tf:
                tf.write(patched)
                tmp_name = tf.name
            return nib.load(tmp_name)
        except Exception:
            raise e
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except Exception:
                    pass


def read_gifti_mesh(filename: str) -> vtkPolyData:
    if HAVE_VTK_GIFTI:
        r = vtkGIFTIReader(); r.SetFileName(filename); r.Update()
        out = r.GetOutput()
        if out is None or out.GetNumberOfPoints() == 0:
            raise RuntimeError(f"Failed to read mesh from {filename}")
        return out
    # Fallback: nibabel
    try:
        g = _nib_load_gifti(filename)
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This VTK build lacks vtkGIFTIReader. Install nibabel for fallback.") from e
    coords = None; faces = None
    for d in g.darrays:
        code = int(getattr(d, 'intent', getattr(d, 'intent_code', -1)) or -1)
        if code == 1008 and coords is None:  # POINTSET
            coords = d.data.astype(float)
        elif code == 1009 and faces is None:  # TRIANGLE
            faces = d.data.astype(np.int64)
    if coords is None or faces is None:
        # Heuristics
        for d in g.darrays:
            if coords is None and d.data.ndim == 2 and d.data.shape[1] == 3 and d.data.dtype.kind in 'fc':
                coords = d.data.astype(float)
            if faces is None and d.data.ndim == 2 and d.data.shape[1] == 3 and d.data.dtype.kind in 'iu':
                faces = d.data.astype(np.int64)
    if coords is None or faces is None:
        raise RuntimeError(f"Could not find POINTSET/TRIANGLE arrays in {filename}")
    pts = vtkPoints(); pts.SetNumberOfPoints(coords.shape[0])
    for i, p in enumerate(coords):
        pts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))
    cells = vtkCellArray()
    for tri in faces:
        cells.InsertNextCell(3); cells.InsertCellPoint(int(tri[0])); cells.InsertCellPoint(int(tri[1])); cells.InsertCellPoint(int(tri[2]))
    poly = vtkPolyData(); poly.SetPoints(pts); poly.SetPolys(cells)
    return poly


def is_gifti_mesh_file(filename: str) -> bool:
    """Return True if the .gii file contains a surface mesh (POINTSET/TRIANGLE)."""
    try:
        if not str(filename).lower().endswith('.gii'):
            return False
        poly = read_gifti_mesh(str(filename))
        if poly is None:
            return False
        try:
            npts = int(poly.GetNumberOfPoints())
        except Exception:
            npts = 0
        try:
            ncells = int(poly.GetNumberOfPolys())
        except Exception:
            try:
                polys = poly.GetPolys()
                ncells = int(polys.GetNumberOfCells()) if polys is not None else 0
            except Exception:
                ncells = 0
        return (npts > 0) and (ncells > 0)
    except Exception:
        return False


def discover_spm_overlays(directory: str) -> List[str]:
    """Find all SPM surface analysis overlay files in *directory*.

    Looks for known SPM result patterns (spmT_*.gii, spmF_*.gii,
    con_*.gii, beta_*.gii, etc.) in a directory that contains an
    SPM.mat file.  Returns a sorted list of absolute paths.

    Args:
        directory: Path to an SPM analysis directory.

    Returns:
        Sorted list of SPM overlay .gii files found in the directory.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    spm_mat = d / 'SPM.mat'
    if not spm_mat.exists():
        return []
    spm_patterns = [
        'spmT_*.gii', 'spmF_*.gii', 'con_*.gii',
        'ess_*.gii', 'beta_*.gii', 'ResMS.gii', 'RPV.gii',
    ]
    found: List[str] = []
    for pat in spm_patterns:
        found.extend(str(p) for p in sorted(d.glob(pat)))
    # Also include any other .gii files with a .dat companion
    for gii in sorted(d.glob('*.gii')):
        if str(gii) not in found and gii.with_suffix('.dat').exists():
            found.append(str(gii))
    return found


def _read_spm_overlay(filename: str) -> vtkDoubleArray:
    """Read scalar data from an SPM surface analysis overlay (.gii/.dat pair).

    SPM stores surface analysis results as GIFTI files with
    ExternalFileBinary encoding, where the actual numeric data lives in
    a companion ``.dat`` file.  nibabel handles this transparently.

    This function is used instead of the generic VTK GIFTI reader path
    because ``vtkGIFTIReader`` may not support ExternalFileBinary
    encoding.

    Args:
        filename: Path to the ``.gii`` file.

    Returns:
        vtkDoubleArray with one scalar value per surface vertex.
    """
    g = _nib_load_gifti(filename)
    data_arr = None
    for d in g.darrays:
        code = int(getattr(d, 'intent', getattr(d, 'intent_code', -1)) or -1)
        if code in (1008, 1009):  # POINTSET / TRIANGLE – skip geometry
            continue
        if d.data.ndim == 1:
            data_arr = d.data.astype(float)
            break
        if d.data.ndim == 2 and 1 in d.data.shape:
            data_arr = d.data.reshape(-1).astype(float)
            break
    if data_arr is None:
        raise RuntimeError(f"No scalar data found in SPM overlay {filename}")
    data_arr = np.nan_to_num(data_arr, nan=0.0)
    out = vtkDoubleArray()
    out.SetNumberOfTuples(len(data_arr))
    for i, v in enumerate(data_arr):
        out.SetValue(i, float(v))
    return out


def _is_freesurfer_morph(filename: str) -> bool:
    """Return True when the file starts with the FreeSurfer curv magic.

    ``nibabel.freesurfer.read_morph_data`` also accepts the legacy format,
    which has no magic number, so it silently returns nonsense for text or
    other binary files.  Requiring the magic keeps plain-text overlays
    (which frequently have no file extension either) on the text path.
    """
    try:
        with open(filename, 'rb') as f:
            return f.read(3) == b'\xff\xff\xff'
    except Exception:
        return False


def read_scalars(filename: str) -> vtkDoubleArray:
    ext = Path(filename).suffix.lower()

    # --- SPM surface analysis overlays (.gii with .dat companion) ---
    # Always use nibabel for these; vtkGIFTIReader may not support
    # the ExternalFileBinary encoding used by SPM.
    if ext == ".gii" and is_spm_surface_overlay(filename):
        return _read_spm_overlay(filename)

    # --- Case 1: GIFTI overlays ---
    if ext == ".gii":
        if HAVE_VTK_GIFTI:
            r = vtkGIFTIReader(); r.SetFileName(filename); r.Update()
            img = r.GetOutput(); arr = None
            if img and img.GetPointData() and img.GetPointData().GetScalars():
                arr = img.GetPointData().GetScalars()
            elif img and img.GetPointData() and img.GetPointData().GetNumberOfArrays() > 0:
                arr = img.GetPointData().GetArray(0)
            if arr:
                npv = np.nan_to_num(vtk_to_numpy(arr).astype(float), nan=0.0)
                out = vtkDoubleArray(); out.SetNumberOfTuples(len(npv))
                for i, v in enumerate(npv): out.SetValue(i, float(v))
                return out
        # fallback with nibabel
        try:
            g = _nib_load_gifti(filename)
        except Exception as e:
            raise RuntimeError("vtkGIFTIReader unavailable and nibabel failed to load .gii") from e
        data_arr = None
        for d in g.darrays:
            code = int(getattr(d, 'intent', getattr(d, 'intent_code', -1)) or -1)
            if code in (1008, 1009):
                continue
            if d.data.ndim == 1:
                data_arr = d.data.astype(float); break
            if d.data.ndim == 2 and 1 in d.data.shape:
                data_arr = d.data.reshape(-1).astype(float); break
        if data_arr is None:
            raise RuntimeError(f"No scalar data array found in {filename}")
        data_arr = np.nan_to_num(data_arr, nan=0.0)
        out = vtkDoubleArray(); out.SetNumberOfTuples(len(data_arr))
        for i, v in enumerate(data_arr): out.SetValue(i, float(v))
        return out

    # --- Case 2: FreeSurfer texture data (thickness/curv/sulc, with or without extension) ---
    # Only when the file really carries the curv magic: nibabel happily
    # misreads other content as the legacy format and returns garbage counts.
    if _is_freesurfer_morph(filename):
        try:
            from nibabel.freesurfer.io import read_morph_data
            fs = np.nan_to_num(read_morph_data(filename).astype(float), nan=0.0)
            out = vtkDoubleArray(); out.SetNumberOfTuples(len(fs))
            for i, v in enumerate(fs): out.SetValue(i, float(v))
            return out
        except Exception:
            pass

    # --- Case 3: Plain text file (one value per line) ---
    data: List[float] = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data.append(float(line.split()[0]))
                except Exception:
                    continue
    except UnicodeDecodeError:
        with open(filename, "r", encoding="latin-1", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data.append(float(line.split()[0]))
                except Exception:
                    continue
    if not data:
        raise RuntimeError(f"Unsupported overlay format for {filename}. Use GIFTI, FreeSurfer morph, or text.")
    data_np = np.nan_to_num(np.array(data, dtype=float), nan=0.0)
    out = vtkDoubleArray(); out.SetNumberOfTuples(len(data_np))
    for i, v in enumerate(data_np): out.SetValue(i, float(v))
    return out

# ---- Title helper ----
def _title_from_path(path: str, max_chars: int = 80) -> str:
    """Return a window-title string from *path*, trimmed to the last *max_chars* characters.

    The .gii and .txt suffixes are stripped, then the rightmost *max_chars*
    characters of the result are returned (no ellipsis prefix).
    """
    s = str(path).replace('.gii', '').replace('.txt', '')
    return s[-max_chars:] if len(s) > max_chars else s


# ---- Stats ----
def get_mean(arr: vtkDoubleArray) -> float: return float(np.nanmean(vtk_to_numpy(arr)))
def get_median(arr: vtkDoubleArray) -> float: return float(np.nanmedian(vtk_to_numpy(arr)))
def get_std(arr: vtkDoubleArray) -> float: return float(np.nanstd(vtk_to_numpy(arr)))

# ---- Interactor style ----
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
#: Keys the viewer acts on itself; VTK must not also act on them.  Kept at
#: module level so the interactor observer and the help dialog agree.
VIEWER_KEYS = frozenset({
    'q', 'Q', 'u', 'U', 'd', 'D', 'l', 'L', 'r', 'R', 'o', 'O', 'b', 'B',
    'g', 'G', 'h', 'H', 'm', 'M', 'Left', 'Right',
    'plus', 'equal', 'minus', 'KP_Add', 'KP_Subtract',
})


class CustomInteractorStyle(vtkInteractorStyleTrackballCamera):
    """Trackball style with the viewer's own keys taken out.

    The keys are not taken out here: overriding ``OnKeyPress`` and friends in
    Python has no effect, because the interactor dispatches events to the C++
    implementation.  :class:`Viewer` therefore observes ``KeyPressEvent`` at a
    higher priority and aborts the ones it handles (see VIEWER_KEYS).
    """

    def __init__(self):
        super().__init__()
        self._renderer = None
        self._viewer = None

    def SetRenderer(self, ren: vtkRenderer): self._renderer = ren
    def SetViewer(self, viewer): self._viewer = viewer

    def _volume_open(self) -> bool:
        try:
            return bool(self._viewer and getattr(self._viewer, '_volume_windows', None))
        except Exception:
            return False


# ---- Options & CLI ----
#: Option sets selected with ``-preset``.  Keys are the command-line option
#: names, so a preset is written the way it would be typed; options given
#: explicitly on the command line keep precedence over the preset.
#: (Not called "style": Qt claims -style for its widget style.)
PRESETS = {
    1: {'c3': True, 'discrete': 16},
}

#: One-line description per preset, shown in the help
PRESET_HELP = {
    1: 'C3 colormap with 16 discrete levels',
}

#: The colormap options form one group: a preset must not override a colormap
#: the user asked for, whichever of the flags it was
_COLORMAP_FLAGS = ('fire', 'bipolar', 'c1', 'c2', 'c3')


@dataclass
class Options:
    mesh_left: Optional[str]
    meshes: List[str] = None  # Multiple mesh files (for navigation when no overlay)
    overlay: Optional[str] = None
    overlays: List[str] = None  # Multiple overlays
    overlay_bkg: Optional[str] = None
    volume: Optional[str] = None  # 3D NIfTI image path to show in orthoview
    range: Tuple[float, float] = (0.0, -1.0)
    range_bkg: Tuple[float, float] = (0.0, -1.0)
    clip: Tuple[float, float] = (0.0, -1.0)
    size: Tuple[int, int] = DEFAULT_WINDOW_SIZE
    title: Optional[str] = None
    output: Optional[str] = None
    fontsize: int = 0
    opacity: float = 0.8
    stats: bool = False  # legacy flag; if true and no title_mode set, implies 'stats'
    title_mode: str = 'shape'  # 'shape' | 'stats' | 'none'
    inverse: bool = False
    colorbar: bool = False
    discrete: int = 0
    log: bool = False
    white: bool = False
    panel: bool = False  # start with control dock hidden by default
    colormap: int = JET
    debug: bool = False
    fix_scaling: bool = False  # Fix scaling across overlays
    free_zoom: bool = False    # Let the mouse change the zoom

def parse_args(argv: List[str]) -> Options:
    p = argparse.ArgumentParser(
        prog='CAT_SurfView',
        description='Render LH/RH cortical surfaces with optional overlays (CAT_SurfView).',
        epilog=(
            'Examples:\n'
            '  CAT_SurfView lh.central.subj.gii                 surface mesh\n'
            '  CAT_SurfView lh.thickness.subj                   overlay (mesh found automatically)\n'
            '  CAT_SurfView lh.central.subj.gii -overlay lh.thickness.subj\n'
            '  CAT_SurfView sub-*/lh.thickness.*                many overlays, ←/→ to step through\n'
            '  CAT_SurfView -range 6 16 -clip -100 6 -colorbar stat/logP_*.gii\n'
            '  CAT_SurfView -preset 1 lh.thickness.subj         predefined settings\n'
            '  CAT_SurfView -output view.png lh.thickness.subj  write a PNG and exit\n'
            '\n'
            'How the surface is determined:\n'
            '  An overlay does not reference its surface, so it is looked up in this order:\n'
            '    1. geometry stored inside the overlay file itself (CAT12 mesh.* and\n'
            '       statistic results usually carry it),\n'
            '    2. the mesh matching the overlay name (lh.thickness.subj -> lh.central.subj.gii)\n'
            '       or a central/midthickness surface in the same folder,\n'
            '    3. the number of values, matched against the 4k/32k/164k templates.\n'
            '  Step 3 is what makes free-form names work, e.g. CAT12/SPM statistic folders\n'
            '  (logP_*.gii, TFCE_*.gii), and it is re-run for every overlay, so files\n'
            '  from different folders, subjects or mesh resolutions can be mixed in one call.\n'
            '  The other hemisphere is added when its file sits next to the selected one —\n'
            '  lh./rh., left/right or _hemi-L_/_hemi-R_, in either direction — or when a mesh\n'
            '  (mesh.central.*) or overlay holds both hemispheres back to back.\n'
            '\n'
            'Keys:\n'
            '  ←/→ previous/next overlay (or mesh)   u/d/l/r rotate   o reset view\n'
            '  b flip dorsal views   w/s wireframe/shaded   g screenshot   h key help   q quit\n'
            '\n'
            'Batch use:\n'
            '  -output renders the view, writes the PNG and exits, so the viewer can be\n'
            '  called in a loop.  With several overlays the first one is written.\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Accept one or more positional inputs. If more than one is given, treat all as overlays
    # and derive the mesh from the first overlay via naming rules.
    p.add_argument(
        'inputs', nargs='*',
        help='Mesh and/or overlay files. Several overlays can be stepped through with ←/→.'
    )
    p.add_argument('-overlay','-ov', dest='overlay', help='Overlay scalars (.gii, FreeSurfer morph, or text)')
    p.add_argument('-overlays', dest='overlays', nargs='+', help='Multiple overlay files for navigation')
    p.add_argument('-bkg', dest='overlay_bkg', help='Background scalars for curvature shading (.gii or text)')
    p.add_argument('-volume','-vol','--nifti', dest='volume',
                   help='3D NIfTI volume to show in a linked orthogonal slice window.\n'
                        'Clicking the surface moves the slices and vice versa.')
    p.add_argument('-range','-r', dest='range', nargs=2, type=float, default=[0.0, -1.0],
                   help='Overlay value range (min max); omit for auto-scaling.\n'
                        'Given explicitly, it also overrides the thickness/pbt presets.')
    p.add_argument('-range-bkg','-rb', dest='range_bkg', nargs=2, type=float, default=[0.0, -1.0],
                   help='Background (curvature) value range (min max); omit for auto-scaling.')
    p.add_argument('-clip','-cl', dest='clip', nargs=2, type=float, default=[0.0, -1.0],
                   help='Hide values between min and max (min max); min == max disables it.\n'
                        'Bounds shared with -range are included, so "-range 6 16 -clip -100 6"\n'
                        'hides everything up to 6.')
    p.add_argument('-size','-sz', dest='size', nargs=2, type=int, default=list(DEFAULT_WINDOW_SIZE), help='Window size in pixels (width height)')
    p.add_argument('-title', dest='title', help='Window/title string (overrides auto title)')
    p.add_argument('-output','-save', dest='output',
                   help='Render to this PNG file and exit without user interaction (batch mode)')
    p.add_argument('-fontsize','-fs', dest='fontsize', type=int, default=0, help='Title/font size (0 = auto)')
    p.add_argument('-opacity','-op', dest='opacity', type=float, default=0.8, help='Overlay opacity')
    p.add_argument('-stats', action='store_true', help='Deprecated: same as --title-mode stats when colorbar is shown')
    p.add_argument('-title-mode', dest='title_mode', choices=['shape','stats','none'], default='shape',
                   help='Colorbar title: shape (filename), stats, or none')
    p.add_argument('-inverse', action='store_true', help='Invert the overlay colormap')
    p.add_argument('-colorbar','-cb', dest='colorbar', action='store_true',
                   help='Show the colorbar (only has an effect with an overlay)')
    p.add_argument('-discrete','-dsc', dest='discrete', type=int, default=0,
                   help='Number of discrete color levels (0 = continuous)')
    p.add_argument('-log', action='store_true',
                   help='Label the colorbar with p-values (-log10(p) overlay).\n'
                        'Applied automatically when the file name contains "log".')
    p.add_argument('-white', action='store_true', help='Use a white background')
    # Control panel visibility (default: hidden)
    p.add_argument('-panel', dest='panel', action='store_true', help='Start with the control panel shown')
    p.add_argument('-no-panel', dest='panel', action='store_false', help='Start with the control panel hidden (default)')
    p.set_defaults(panel=False)
    # Colormap selection (default: jet)
    p.add_argument('-fire', action='store_true', help='Use the fire colormap')
    p.add_argument('-bipolar', action='store_true', help='Use the bipolar colormap')
    p.add_argument('-c1', action='store_true', help='Use custom colormap 1')
    p.add_argument('-c2', action='store_true', help='Use custom colormap 2')
    p.add_argument('-c3', action='store_true', help='Use custom colormap 3')
    p.add_argument('-preset', dest='preset', type=int, default=0, metavar='N',
                   help='Predefined settings:\n'
                        + '\n'.join(f'  {n} = {PRESET_HELP.get(n, "")}'
                                    for n in sorted(PRESETS))
                        + '\nOptions given explicitly take precedence.')
    p.add_argument('-free-zoom', dest='free_zoom', action='store_true',
                   help='Allow zooming with the mouse or trackpad (off by default, '
                        'because a right-click then keeps the view zooming)')
    p.add_argument('-fix-scaling', dest='fix_scaling', action='store_true',
                   help='Keep the range of the first overlay for all following ones')
    p.add_argument('-debug', action='store_true', help=argparse.SUPPRESS)  # accepted, not implemented
    # External defaults file for viewer settings (key=value lines)
    p.add_argument('-defaults', dest='defaults', help='Path to a defaults file (key=value) to override built-in defaults')
    # Called without any argument: show the help instead of opening an empty window
    if not argv:
        p.print_help()
        sys.exit(0)
    a = p.parse_args(argv)

    # Optionally load external defaults and apply only for values not explicitly provided on CLI
    def _parse_bool(s: str) -> bool:
        return str(s).strip().lower() in ('1','true','yes','on')

    def _parse_floats_csv(s: str, n_expected: int = None) -> Tuple[float, ...]:
        parts = [p for p in re.split(r'[;,\s]+', str(s).strip()) if p]
        vals = tuple(float(p) for p in parts)
        if n_expected and len(vals) != n_expected:
            raise ValueError(f'Expected {n_expected} numbers, got {len(vals)}')
        return vals

    def _cm_from_name(name: str) -> int:
        name_u = str(name).strip().upper()
        mapping = {
            'JET': JET, 'HOT': HOT, 'FIRE': FIRE, 'BIPOLAR': BIPOLAR, 'GRAY': GRAY,
            'C1': C1, 'C2': C2, 'C3': C3
        }
        if name_u in mapping:
            return mapping[name_u]
        # allow numeric index
        try:
            v = int(name_u)
            return v if v in mapping.values() else JET
        except Exception:
            return JET

    def _load_defaults_file(path: str) -> dict:
        cfg = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    if '=' not in s:
                        continue
                    key, val = s.split('=', 1)
                    cfg[key.strip()] = val.strip().strip('"\'')
        except Exception:
            return {}
        return cfg

    def _apply_defaults_cfg(cfg: dict, ns: argparse.Namespace, defaults_ns: argparse.Namespace):
        if not cfg:
            return
        def _apply_if_default(attr: str, parser: callable):
            if hasattr(ns, attr):
                if getattr(ns, attr) == getattr(defaults_ns, attr):
                    try:
                        setattr(ns, attr, parser(cfg[attr]))
                    except Exception:
                        pass
        # Scalars / toggles
        if 'opacity' in cfg: _apply_if_default('opacity', float)
        if 'discrete' in cfg: _apply_if_default('discrete', int)
        if 'inverse' in cfg: _apply_if_default('inverse', _parse_bool)
        if 'colorbar' in cfg: _apply_if_default('colorbar', _parse_bool)
        if 'fontsize' in cfg: _apply_if_default('fontsize', int)
        if 'panel' in cfg: _apply_if_default('panel', _parse_bool)
        if 'fix_scaling' in cfg: _apply_if_default('fix_scaling', _parse_bool)
        if 'white' in cfg: _apply_if_default('white', _parse_bool)
        if 'log' in cfg: _apply_if_default('log', _parse_bool)
        if 'debug' in cfg: _apply_if_default('debug', _parse_bool)
        # Enums / tuples
        if 'title_mode' in cfg: _apply_if_default('title_mode', str)
        if 'range' in cfg: _apply_if_default('range', lambda s: tuple(_parse_floats_csv(s, 2)))
        if 'range_bkg' in cfg: _apply_if_default('range_bkg', lambda s: tuple(_parse_floats_csv(s, 2)))
        if 'clip' in cfg: _apply_if_default('clip', lambda s: tuple(_parse_floats_csv(s, 2)))
        if 'size' in cfg: _apply_if_default('size', lambda s: tuple(int(x) for x in _parse_floats_csv(s, 2)))
        if 'colormap' in cfg: _apply_if_default('colormap', _cm_from_name)

    # Build a defaults namespace to detect which args were explicitly set by user
    defaults_ns = p.parse_args([])
    if getattr(a, 'defaults', None):
        import re  # lazy import for simple parsing
        cfg = _load_defaults_file(a.defaults)
        _apply_defaults_cfg(cfg, a, defaults_ns)
    else:
        # If no explicit defaults file given, try to load project default
        import re  # for parsing floats
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir.parent / 'data' / 'cat_surf_view_defaults.txt',
            script_dir / 'cat_surf_view_defaults.txt',
            Path.cwd() / 'cat_surf_view_defaults.txt',
        ]
        for c in candidates:
            if c.exists():
                cfg = _load_defaults_file(str(c))
                _apply_defaults_cfg(cfg, a, defaults_ns)
                break

    # A preset sets several options at once, without overriding anything the
    # user gave on the command line
    if a.preset:
        preset = PRESETS.get(int(a.preset))
        if preset is None:
            p.error(f"Unknown -preset {a.preset}; available: "
                    f"{', '.join(str(n) for n in sorted(PRESETS))}")
        preset = dict(preset)
        if any(getattr(a, flag, False) for flag in _COLORMAP_FLAGS):
            for flag in _COLORMAP_FLAGS:
                preset.pop(flag, None)
        for key, value in preset.items():
            if getattr(a, key, None) == getattr(defaults_ns, key, None):
                setattr(a, key, value)

    cm = JET
    if a.fire: cm = FIRE
    if a.bipolar: cm = BIPOLAR
    if a.c1: cm = C1
    if a.c2: cm = C2
    if a.c3: cm = C3

    d = int(a.discrete)
    if d < 0 or d > 256:
        p.error("Parameter -discrete/-dsc should be 0..256")

    # Derive mesh/overlay list from positional inputs (optional)
    pos_inputs: List[str] = list(a.inputs)
    overlays_from_pos: List[str] = []
    meshes_from_pos: List[str] = []
    mesh_left_resolved: str = ''
    overlay_single_from_pos: Optional[str] = None
    if len(pos_inputs) == 1:
        # Single input can be either a mesh or an overlay. Prefer real mesh content when .gii.
        single = pos_inputs[0]
        # Combined 'mesh.' overlays contain both geometry AND scalars;
        # treat them as overlays so the scalar data is loaded.
        if _parse_mesh_combined_overlay(single) is not None:
            try:
                mesh_left_resolved = convert_filename_to_mesh(single)
            except Exception:
                mesh_left_resolved = single
            overlay_single_from_pos = single
        elif _is_gifti_mesh_by_name(single) or (
            str(single).lower().endswith('.gii') and is_gifti_mesh_file(single)
        ):
            mesh_left_resolved = single
            overlay_single_from_pos = None
        elif is_overlay_file(single):
            try:
                mesh_left_resolved = convert_filename_to_mesh(single)
            except Exception:
                mesh_left_resolved = single
            # If it's an overlay, record it unless other overlay flags are used
            overlay_single_from_pos = single
        else:
            mesh_left_resolved = single
            overlay_single_from_pos = None
    elif len(pos_inputs) > 1:
        # Split inputs: collect any .gii that are real meshes, and treat other inputs as overlays
        mesh_candidates: List[str] = []
        non_mesh_inputs: List[str] = []
        for pth in pos_inputs:
            # Use fast name-based heuristic first to avoid reading every file.
            # Fall back to content inspection only when the name is ambiguous.
            if _is_gifti_mesh_by_name(pth) or (
                str(pth).lower().endswith('.gii') and is_gifti_mesh_file(pth)
            ):
                mesh_candidates.append(pth)
            else:
                non_mesh_inputs.append(pth)
        if mesh_candidates:
            # pick first mesh candidate as mesh; do not force the others as overlays
            mesh_left_resolved = mesh_candidates[0]
            meshes_from_pos = list(mesh_candidates)
            # Be permissive: treat any remaining positional args as overlays, even if
            # naming heuristics fail, so that multiple overlays are never dropped.
            overlays_from_pos = list(non_mesh_inputs)
        else:
            overlays_from_pos = pos_inputs
            try:
                mesh_left_resolved = convert_filename_to_mesh(overlays_from_pos[0])
            except Exception:
                mesh_left_resolved = overlays_from_pos[0]
    else:
        # no positional inputs; mesh will be chosen later via GUI
        mesh_left_resolved = ''

    # Priority for overlays: positional list > -overlays > -overlay
    overlay_list_final: List[str] = overlays_from_pos or (a.overlays or [])
    # Prefer an explicit list; else use single overlay from positional if detected; else -overlay flag
    overlay_single_final: Optional[str] = None
    if not overlay_list_final:
        overlay_single_final = (locals().get('overlay_single_from_pos')
                                if 'overlay_single_from_pos' in locals() and locals()['overlay_single_from_pos']
                                else a.overlay)

    # If mesh could not be resolved from positional inputs, attempt to derive it
    # from the first overlay supplied via -overlays or -overlay.
    if (not mesh_left_resolved) and overlay_list_final:
        try:
            mesh_left_resolved = convert_filename_to_mesh(overlay_list_final[0])
        except Exception:
            mesh_left_resolved = overlay_list_final[0]
    elif (not mesh_left_resolved) and overlay_single_final:
        try:
            mesh_left_resolved = convert_filename_to_mesh(overlay_single_final)
        except Exception:
            mesh_left_resolved = overlay_single_final

    # Map legacy -stats flag into title_mode if specified
    title_mode_arg = getattr(a, 'title_mode', 'shape')
    if getattr(a, 'stats', False):
        title_mode_arg = 'stats'
    return Options(
        mesh_left=mesh_left_resolved,
        meshes=meshes_from_pos,
        overlay=overlay_single_final,
        overlays=overlay_list_final,
        overlay_bkg=a.overlay_bkg,
        volume=getattr(a, 'volume', None),
        range=tuple(a.range),
        range_bkg=tuple(a.range_bkg),
        clip=tuple(a.clip),
        size=tuple(a.size),
        title=a.title,
        output=a.output,
        fontsize=a.fontsize,
        opacity=a.opacity,
    stats=bool(a.stats),
    title_mode=title_mode_arg,
        inverse=bool(a.inverse),
        colorbar=bool(a.colorbar),
        discrete=d,
        log=bool(a.log),
        white=bool(a.white),
        panel=bool(a.panel),
        colormap=cm,
        debug=bool(a.debug),
        fix_scaling=bool(a.fix_scaling),
        free_zoom=bool(getattr(a, 'free_zoom', False)),
    )

# ---- Control Panel ----
# ---- Viewer ----
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, opts: Options):
        super().__init__()
        self.opts = opts
        # Cache of overlay path -> resolved mesh path.
        # Prevents ambiguous directory-based mesh resolution from "sticking" to the
        # most recently used mesh when toggling between multiple overlays.
        self._overlay_mesh_cache: dict[str, str] = {}
        self._y_shift_l: float = 0.0
        self._y_shift_r: float = 0.0
        self._hist_win = None  # histogram window reference
        # Mesh navigation state (when multiple input meshes and no overlay)
        self.mesh_list: List[str] = list(self.opts.meshes or [])
        self.current_mesh_index: int = 0
        # Use the original input file for the window title
        self.setWindowTitle(self.opts.title or _title_from_path(self.opts.mesh_left))
        self.resize(*opts.size)

        # central widget with VTK view
        self.frame = QtWidgets.QFrame(); self.vl = QtWidgets.QVBoxLayout(); self.vl.setContentsMargins(0,0,0,0)
        self.frame.setLayout(self.vl); self.setCentralWidget(self.frame)
        self.vtk_widget = QVTKRenderWindowInteractor(self.frame); self.vl.addWidget(self.vtk_widget)
        # Ensure Qt uses a native window for the GL surface on macOS
        try:
            self.vtk_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        except Exception:
            try:
                self.vtk_widget.setAttribute(Qt.WA_NativeWindow, True)
            except Exception:
                pass
        # Ensure the VTK widget can accept keyboard focus
        try:
            self.vtk_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        except Exception:
            self.vtk_widget.setFocusPolicy(Qt.StrongFocus)

        self.ren = vtkRenderer(); self.ren.SetBackground(1,1,1) if opts.white else self.ren.SetBackground(0,0,0)
        self.rw: vtkRenderWindow = self.vtk_widget.GetRenderWindow();
        try:
            self.rw.SetMultiSamples(0)
            self.rw.SetAlphaBitPlanes(0)
        except Exception:
            pass
        # Use two layers: main 3D in layer 0, UI (colorbar) in layer 1 to keep camera bounds stable
        self.rw.SetNumberOfLayers(2)
        self.ren.SetLayer(0)
        self.rw.AddRenderer(self.ren)
        self.ren_ui = vtkRenderer(); self.ren_ui.SetLayer(1); self.ren_ui.SetInteractive(0)
        # Match UI renderer background to transparent overlay-like look
        try:
            self.ren_ui.SetBackgroundAlpha(0.0)
        except Exception:
            pass
        self.rw.AddRenderer(self.ren_ui)

        # interactor style
        self.iren: vtkRenderWindowInteractor = self.rw.GetInteractor()
        style = CustomInteractorStyle(); style.SetRenderer(self.ren); style.SetViewer(self); self.iren.SetInteractorStyle(style)
        # The viewer handles its own keys here.  Overriding the style's
        # OnKeyPress would not do: the interactor dispatches to the C++
        # implementation, which knows nothing about a Python subclass — so the
        # event has to be taken away from the style by aborting it, otherwise
        # VTK acts on it as well (its 'r' resets the camera, which used to
        # throw away the rotation the viewer had just applied).
        def _claim(event: str, handler=None):
            """Handle *event* before the style, and keep our keys from it."""
            tag = [None]

            def callback(obj, _event):
                try:
                    sym = self.iren.GetKeySym()
                except Exception:
                    sym = None
                if handler is not None:
                    handler(sym)
                if sym in VIEWER_KEYS and tag[0] is not None:
                    command = obj.GetCommand(tag[0])
                    if command is not None:
                        command.AbortFlagOn()

            self._key_callbacks.append(callback)
            tag[0] = self.iren.AddObserver(event, callback, 1.0)

        self._key_callbacks: List = []
        _claim("KeyPressEvent", self._handle_key)
        # VTK acts on most keys in OnChar, not on the key press: 'r' resets the
        # camera there, which used to undo the rotation of the viewer's own 'r'
        _claim("CharEvent")

        # The mouse must not change the zoom.  Right-drag is the style's zoom
        # and the context menu opens on the same button, so the menu takes the
        # release the style waits for and every later mouse move keeps
        # zooming — which makes clicking a vertex impossible.
        self.lock_zoom = not bool(getattr(opts, 'free_zoom', False))
        self._zoom_callbacks: List = []
        for event in ("RightButtonPressEvent", "RightButtonReleaseEvent",
                      "MouseWheelForwardEvent", "MouseWheelBackwardEvent",
                      "StartPinchEvent", "PinchEvent"):
            self._guard_zoom(event)

        # Clicking on the surface marks the spot and moves any open slice viewer
        self._cursor_actors: List[vtkActor] = []
        #: (hemisphere, vertex) the cursor sits on, and the atlas naming it
        self._cursor_vertex: Optional[Tuple[int, int]] = None
        #: What the surface is shaded with; see set_underlay
        self.underlay: Optional[str] = 'mc'
        self._atlas: Optional[dict] = None
        self.atlas_path: Optional[str] = None
        #: Draw the boundaries between atlas regions on the surface
        self.show_borders = False
        self.actor_border_l: Optional[vtkActor] = None
        self.actor_border_r: Optional[vtkActor] = None

        def _on_left_click(_obj, _evt):
            try:
                x, y = self.iren.GetEventPosition()
            except Exception:
                return
            mm = self._surface_click_to_mm(x, y)
            if mm is not None:
                self._set_surface_cursor(mm)
                self._broadcast_world_pick(mm)
        self.iren.AddObserver("LeftButtonPressEvent", _on_left_click)

        # Load surfaces (LH + optional RH) if provided; otherwise start empty
        self.poly_l = None
        self.poly_r = None
        if self.opts.mesh_left:
            # Check if the input is an overlay file or mesh file
            input_path = Path(self.opts.mesh_left)
            # If path does not exist in current CWD, try ORIGINAL_CWD from wrapper
            if not input_path.exists():
                try:
                    orig = os.environ.get('ORIGINAL_CWD')
                    if orig:
                        cand = Path(orig) / Path(self.opts.mesh_left)
                        if cand.exists():
                            input_path = cand
                except Exception:
                    pass
            if not input_path.exists(): 
                raise FileNotFoundError(f"File not found: {input_path}")
            # Determine if input is an overlay file; but let .gii meshes pass through.
            # Combined 'mesh.' files contain geometry AND scalars — treat as overlay.
            is_combined_mesh_overlay = (_parse_mesh_combined_overlay(str(input_path)) is not None)
            if is_combined_mesh_overlay or (
                not (str(input_path).lower().endswith('.gii') and is_gifti_mesh_file(str(input_path)))
                and is_overlay_file(str(input_path))
            ):
                # Input is an overlay file, find the corresponding mesh
                mesh_path = convert_filename_to_mesh(str(input_path))
                mesh_path_obj = Path(mesh_path)
                # If derived path is relative and not found, try ORIGINAL_CWD
                if not mesh_path_obj.exists():
                    try:
                        orig = os.environ.get('ORIGINAL_CWD')
                        if orig:
                            cand = Path(orig) / mesh_path
                            if cand.exists():
                                mesh_path_obj = cand
                    except Exception:
                        pass
                if not mesh_path_obj.exists():
                    raise FileNotFoundError(f"Corresponding mesh file not found: {mesh_path}")
                left_mesh_path = mesh_path_obj
                # Set the overlay to the original input file
                opts.overlay = str(input_path)
            else:
                # Input is a mesh file
                left_mesh_path = input_path
            self.poly_l, self.poly_r = read_mesh_pair(str(left_mesh_path))
            # Normalize Y-origin similar to C++ utility
            self._y_shift_l = self._shift_y_to(self.poly_l) or 0.0
            if self.poly_r is not None:
                self._y_shift_r = self._shift_y_to(self.poly_r) or 0.0
            # Initialize mesh navigation list/index so Left/Right can switch meshes when no overlay
            try:
                if not self.mesh_list:
                    self.mesh_list = [str(left_mesh_path)]
                else:
                    left_str = str(left_mesh_path)
                    if left_str not in self.mesh_list:
                        self.mesh_list.insert(0, left_str)
                self.current_mesh_index = max(0, self.mesh_list.index(str(left_mesh_path)))
            except Exception:
                self.mesh_list = [str(left_mesh_path)]
                self.current_mesh_index = 0

        # Background curvature
        self.curv_l = vtkCurvatures(); self.curv_l.SetInputData(self.poly_l); self.curv_l.SetCurvatureTypeToMean(); self.curv_l.Update()
        self.curv_r = None
        if self.poly_r is not None:
            self.curv_r = vtkCurvatures(); self.curv_r.SetInputData(self.poly_r); self.curv_r.SetCurvatureTypeToMean(); self.curv_r.Update()

        # Optional background scalars
        self.bkg_scalar_l = None; self.bkg_scalar_r = None
        if opts.overlay_bkg:
            self.bkg_scalar_l = read_scalars(opts.overlay_bkg)
            other_bkg = _hemi_counterpart(Path(opts.overlay_bkg))
            if other_bkg is not None and other_bkg.exists() and self.poly_r is not None:
                self.bkg_scalar_l, self.bkg_scalar_r = order_by_hemisphere(
                    opts.overlay_bkg, self.bkg_scalar_l, read_scalars(str(other_bkg)))
            elif self.poly_r is not None and self.bkg_scalar_l is not None and self.bkg_scalar_l.GetNumberOfTuples() == (self.poly_l.GetNumberOfPoints()+self.poly_r.GetNumberOfPoints()):
                self.bkg_scalar_l, self.bkg_scalar_r = _split_scalars(
                    self.bkg_scalar_l,
                    self.poly_l.GetNumberOfPoints(),
                    self.poly_r.GetNumberOfPoints(),
                )

        self.curv_l_out = self.curv_l.GetOutput();  
        if self.bkg_scalar_l is not None: self.curv_l_out.GetPointData().SetScalars(self.bkg_scalar_l)
        self.curv_r_out = None
        if self.curv_r is not None:
            self.curv_r_out = self.curv_r.GetOutput();
            if self.bkg_scalar_r is not None: self.curv_r_out.GetPointData().SetScalars(self.bkg_scalar_r)

        # Overlay range (must be known before any LUT/clip computation)
        self.overlay_range = list(opts.range)
        # Track which display ranges the user explicitly provided via CLI so that
        # per-overlay presets (e.g. thickness) never silently override them.
        self._user_set_range = (opts.range[1] > opts.range[0])
        self._user_set_clip = (opts.clip[1] > opts.clip[0])
        self._user_set_range_bkg = (opts.range_bkg[1] > opts.range_bkg[0])

        # Actors and LUTs
        self._actors: List[vtkActor] = []
        self.lut_overlay_l = get_lookup_table(opts.colormap, opts.opacity)
        self.lut_overlay_r = get_lookup_table(opts.colormap, opts.opacity)
        # If inverse is requested, flip the LUTs (do not modify data/ranges)
        if self.opts.inverse:
            self._invert_lut(self.lut_overlay_l)
            self._invert_lut(self.lut_overlay_r)
        # Apply discrete bands to overlay LUTs if requested
        self._apply_discrete_to_overlay_lut(self.lut_overlay_l)
        self._apply_discrete_to_overlay_lut(self.lut_overlay_r)
        # Clip transparency is applied further below, once the overlay range is final
        # The underlay is shading, not data: a band of greys rather than the
        # full black-to-white ramp, so sulci and gyri read as a relief and the
        # overlay colours stay the brightest thing on the surface
        self.lut_bkg = vtkLookupTable(); self.lut_bkg.SetHueRange(0, 0)
        self.lut_bkg.SetSaturationRange(0, 0)
        self.lut_bkg.SetValueRange(0, 1); self.lut_bkg.Build()

        # Background scalar range
        self.range_bkg = list(opts.range_bkg)
        if not (self.range_bkg[1] > self.range_bkg[0]):
            r = [0.0,0.0]; self.curv_l_out.GetScalarRange(r); self.range_bkg = r
        if self.range_bkg[0] < 0 < self.range_bkg[1]:
            m = max(abs(self.range_bkg[0]), abs(self.range_bkg[1])); self.range_bkg = [-m, m]
        self.lut_bkg.SetTableRange(self.range_bkg)
        # Predefine hemisphere actors; they are created as soon as the data
        # they render is known (see _ensure_hemisphere_actors)
        self.actor_bkg_l = None; self.actor_ov_l = None
        self.actor_bkg_r = None; self.actor_ov_r = None

        # Overlay management
        self.overlay_list = []
        self.current_overlay_index = 0
        self.fixed_overlay_range = None  # Store fixed range when fix_scaling is enabled
        
        # Initialize overlay list
        if opts.overlays:
            self.overlay_list = opts.overlays
        elif opts.overlay:
            self.overlay_list = [opts.overlay]
        # Enforce initial fix scaling policy based on overlay count
        self._enforce_fix_scaling_policy()
        
        # Overlay scalars
        self.scal_l = None; self.scal_r = None
        if self.overlay_list:
            self._load_overlay(self.overlay_list[0])
            # After initial load, enforce again since scalars are present now
            self._enforce_fix_scaling_policy()

        # Overlay range (auto if unset)
        if not (self.overlay_range[1] > self.overlay_range[0]) and (self.scal_l is not None):
            r = [0.0,0.0]; self.poly_l.GetScalarRange(r); self.overlay_range = r

        # Apply clip transparency to overlay LUTs (values inside clip become transparent).
        # Done here so the clip window is mapped against the final overlay range.
        self._apply_clip_to_overlay_luts()

        # Mappers/actors for whichever hemispheres carry data
        self._ensure_hemisphere_actors()

        # Build 6-view montage
        self._build_montage()

        # Colorbar: create once and attach/detach based on option
        self.scalar_bar = None
        self._scalar_bar_added = False
        self._ensure_colorbar()
        has_overlay_initial = (self.scal_l is not None or self.scal_r is not None)
        self._colorbar_intent = bool(self.opts.colorbar)
        self.opts.colorbar = bool(self._colorbar_intent and has_overlay_initial)
        if self.opts.colorbar:
            self._attach_colorbar()
        else:
            self._detach_colorbar()
        # Defer initial render/camera setup until the Qt event loop is running
        self._cam_state = None
        self._base_cam_state = None

        # Right-side control panel (dock)
        self._build_control_panel()

        # Start interactor after the window is shown (avoid blocking render)
        QTimer.singleShot(0, self._post_init_render)

        self.vtk_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu
                                             if hasattr(Qt, "ContextMenuPolicy")
                                             else Qt.CustomContextMenu)
        self.vtk_widget.customContextMenuRequested.connect(self._show_view_context_menu)

        # Focus is set during initialization; no extra activation timer needed

        # Optional snapshot: taken in _post_init_render once the scene is
        # rendered, then the application quits so that -output can be used
        # from batch scripts (see _save_and_quit).

        # (Shortcuts/event filter reverted to previous behavior)
        
    # -- Scene construction --
    # Layout of the six montage views: lateral/medial for both hemispheres in
    # the two rows, dorsal views in the middle.  order[i] picks the hemisphere
    # (0 = left, 1 = right) an individual view is cloned from.
    #: Which hemisphere each of the six views shows (0 = left, 1 = right)
    _MONTAGE_ORDER = (0, 1, 0, 1, 0, 1)
    #: The order actually in use; a flat map has a layout of its own
    _montage_order = _MONTAGE_ORDER

    # Marker linking the surface to the slice viewer: radius in mm, and how
    # far from the surface a picked point may be before it is ignored.
    _CURSOR_RADIUS = 3.0
    _CURSOR_MAX_DIST = 20.0

    def _ensure_hemisphere_actors(self) -> bool:
        """Create the background/overlay actors for hemispheres that have data.

        Called at start-up and again after loading an overlay: a file covering
        both hemispheres may follow one that only had a left one, and without
        the missing actors the right-hand views would keep repeating the left
        hemisphere.

        Returns:
            True when at least one actor was created.
        """
        created = False

        def _new_actor(data, lut, scalar_range, ambient) -> vtkActor:
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(data)
            mapper.SetLookupTable(lut)
            if scalar_range is not None and scalar_range[1] > scalar_range[0]:
                mapper.SetScalarRange(scalar_range)
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetAmbient(ambient)
            actor.GetProperty().SetDiffuse(0.7)
            self._actors.append(actor)
            return actor

        if self.actor_bkg_l is None and getattr(self, 'curv_l_out', None) is not None:
            self.actor_bkg_l = _new_actor(self.curv_l_out, self.lut_bkg, self.range_bkg, 0.8)
            created = True
        if self.actor_ov_l is None and self.scal_l is not None and self.poly_l is not None:
            self.actor_ov_l = _new_actor(self.poly_l, self.lut_overlay_l, self.overlay_range, 0.3)
            created = True
        if (self.actor_bkg_r is None and self.poly_r is not None
                and getattr(self, 'curv_r_out', None) is not None):
            self.actor_bkg_r = _new_actor(self.curv_r_out, self.lut_bkg, self.range_bkg, 0.8)
            created = True
        if self.actor_ov_r is None and self.scal_r is not None and self.poly_r is not None:
            self.actor_ov_r = _new_actor(self.poly_r, self.lut_overlay_r, self.overlay_range, 0.3)
            created = True
        return created

    @staticmethod
    def is_flat_surface(poly) -> bool:
        """True for a flattened map, which has no extent across its plane.

        A patch is a cut-open, flattened hemisphere: rotating it into six views
        would draw the same sheet six times on top of itself.
        """
        if poly is None or poly.GetNumberOfPoints() == 0:
            return False
        bounds = poly.GetBounds()
        extents = [bounds[1] - bounds[0], bounds[3] - bounds[2],
                   bounds[5] - bounds[4]]
        largest = max(extents)
        return largest > 0.0 and min(extents) < 0.02 * largest

    def _montage_layout(self):
        """Where the copies of the hemispheres go: (order, x, y, rotX, rotZ).

        Six views of a folded surface — lateral, medial, superior, inferior —
        but a flat map is shown once per hemisphere, side by side.
        """
        if self.is_flat_surface(self.poly_l):
            # Both at the origin; _separate_flat_views() moves the right one
            # aside once its rotated bounds are known.  The two are turned the
            # opposite way (+90 and -90, the lateral pair cat_surf_results
            # uses for flatmaps), so the second map is the mirror image of the
            # first and the pair reads as one figure.
            return ((0, 1), [0.0, 0.0], [0.0, 0.0], [270, 270], [90, -90])
        shifts = (180.0, 180.0)
        posx = [0, 2 * shifts[0], 0.15 * shifts[0], 1.85 * shifts[0], shifts[0], shifts[0]]
        posy = [0, 0, 0.8 * shifts[1], 0.8 * shifts[1], 0.6 * shifts[1], 0.6 * shifts[1]]
        rotx = [270, 270, 270, 270, 0, 0]
        rotz = [90, -90, -90, 90, 0, 0]
        return (self._MONTAGE_ORDER, posx, posy, rotx, rotz)

    def _build_montage(self):
        """(Re)create the six-view montage from the current hemisphere actors.

        Clones of the left/right actors are what actually gets rendered; the
        actors themselves are only templates.  Rebuilding is required whenever
        the set of hemispheres changes (e.g. switching from a single-hemisphere
        overlay to a combined LH+RH one), so previous clones are removed first.
        """
        order, posx, posy, rotx, rotz = self._montage_layout()
        self._montage_order = order
        views = len(order)

        # The cursor markers are placed with the matrices of the old clones
        self._clear_surface_cursor()
        # Drop clones of a previous scene
        for attr in ('_montage_bkg', '_montage_ov', '_montage_border'):
            for actor in (getattr(self, attr, None) or []):
                if actor is not None:
                    try:
                        self.ren.RemoveActor(actor)
                    except Exception:
                        pass
        # Keep track of clones per view index for selective operations (e.g., key 'b')
        self._montage_bkg: List[Optional[vtkActor]] = [None] * views
        self._montage_ov: List[Optional[vtkActor]] = [None] * views
        self._montage_border: List[Optional[vtkActor]] = [None] * views

        def add_clone(actor: vtkActor, px, py, rx, rz) -> vtkActor:
            a = vtkActor(); a.ShallowCopy(actor); a.AddPosition(px, py, 0); a.RotateX(rx); a.RotateZ(rz); self.ren.AddActor(a); return a

        bkg_l = getattr(self, 'actor_bkg_l', None); bkg_r = getattr(self, 'actor_bkg_r', None)
        ov_l = getattr(self, 'actor_ov_l', None); ov_r = getattr(self, 'actor_ov_r', None)
        # Each view shows the hemisphere it stands for, and nothing else: an
        # overlay that covers only one hemisphere used to be cloned into the
        # views of the other, drawing one surface on top of a different one
        for i in range(views):
            if self.poly_r is None and (i % 2 == 1):
                continue
            src = bkg_r if order[i] == 1 else bkg_l
            if src is not None:
                self._montage_bkg[i] = add_clone(src, posx[i], posy[i], rotx[i], rotz[i])
        if ov_l is not None or ov_r is not None:
            for i in range(views):
                if self.poly_r is None and (i % 2 == 1):
                    continue
                src = ov_r if order[i] == 1 else ov_l
                if src is not None:
                    self._montage_ov[i] = add_clone(src, posx[i], posy[i], rotx[i], rotz[i])

        border_l = getattr(self, 'actor_border_l', None)
        border_r = getattr(self, 'actor_border_r', None)
        if border_l is not None or border_r is not None:
            for i in range(views):
                if self.poly_r is None and (i % 2 == 1):
                    continue
                src = border_r if order[i] == 1 else border_l
                if src is not None:
                    self._montage_border[i] = add_clone(src, posx[i], posy[i],
                                                        rotx[i], rotz[i])

        if self.is_flat_surface(self.poly_l):
            self._separate_flat_views()

    def _separate_flat_views(self):
        """Move the right flat map clear of the left one.

        How wide a patch ends up on screen depends on where it was cut and on
        the Y normalization of each hemisphere, so the two are placed by their
        rotated bounds rather than by a guess from the mesh size — a guess is
        what let them overlap.
        """
        left, right = self._montage_bkg[0], self._montage_bkg[1]
        if left is None or right is None:
            return
        left_bounds, right_bounds = left.GetBounds(), right.GetBounds()
        width = max(left_bounds[1] - left_bounds[0],
                    right_bounds[1] - right_bounds[0])
        gap = 0.04 * width
        shift_x = (left_bounds[1] + gap) - right_bounds[0]
        # and level with each other, so the pair reads as one figure
        shift_y = (0.5 * (left_bounds[2] + left_bounds[3])
                   - 0.5 * (right_bounds[2] + right_bounds[3]))
        for actor in (self._montage_bkg[1], self._montage_ov[1],
                      (getattr(self, '_montage_border', None) or [None, None])[1]):
            if actor is not None:
                actor.AddPosition(shift_x, shift_y, 0.0)

    def _set_meshes(self, poly_l: vtkPolyData, poly_r: Optional[vtkPolyData]):
        """Install new hemisphere meshes and rewire everything that renders them.

        Recomputes curvature, re-attaches background scalars, points the
        existing mappers at the new geometry and rebuilds the montage when the
        number of hemispheres changed.  Used by every code path that replaces
        the displayed surface (mesh navigation, overlay-driven mesh switching,
        geometry embedded in an overlay file).
        """
        had_right = self.poly_r is not None
        self.poly_l = poly_l
        self.poly_r = poly_r
        self._y_shift_l = self._shift_y_to(self.poly_l) or 0.0
        self._y_shift_r = (self._shift_y_to(self.poly_r) or 0.0) if self.poly_r is not None else 0.0

        # Curvature / background scalars
        self.curv_l = vtkCurvatures(); self.curv_l.SetInputData(self.poly_l)
        self.curv_l.SetCurvatureTypeToMean(); self.curv_l.Update()
        self.curv_l_out = self.curv_l.GetOutput()
        if self.bkg_scalar_l is not None:
            self.curv_l_out.GetPointData().SetScalars(self.bkg_scalar_l)
        self.curv_r = None; self.curv_r_out = None
        if self.poly_r is not None:
            self.curv_r = vtkCurvatures(); self.curv_r.SetInputData(self.poly_r)
            self.curv_r.SetCurvatureTypeToMean(); self.curv_r.Update()
            self.curv_r_out = self.curv_r.GetOutput()
            if self.bkg_scalar_r is not None:
                self.curv_r_out.GetPointData().SetScalars(self.bkg_scalar_r)

        # The overlay values belong to the vertices, not to the surface they
        # were loaded on: a switch from central to inflated keeps them, as
        # long as the two have the same number of vertices
        for poly, scalars in ((self.poly_l, getattr(self, 'scal_l', None)),
                              (self.poly_r, getattr(self, 'scal_r', None))):
            if poly is None or scalars is None:
                continue
            if scalars.GetNumberOfTuples() == poly.GetNumberOfPoints():
                poly.GetPointData().SetScalars(scalars)

        # Point existing mappers at the new geometry
        if getattr(self, 'actor_bkg_l', None) is not None:
            self.actor_bkg_l.GetMapper().SetInputData(self.curv_l_out)
        if getattr(self, 'actor_ov_l', None) is not None:
            self.actor_ov_l.GetMapper().SetInputData(self.poly_l)
        if getattr(self, 'actor_bkg_r', None) is not None and self.curv_r_out is not None:
            self.actor_bkg_r.GetMapper().SetInputData(self.curv_r_out)
        if getattr(self, 'actor_ov_r', None) is not None and self.poly_r is not None:
            self.actor_ov_r.GetMapper().SetInputData(self.poly_r)

        # A changed hemisphere layout invalidates the montage clones: without
        # this the right-hand views would keep showing the previous surface.
        if had_right != (self.poly_r is not None):
            self._build_montage()

        # Keep slider bounds in sync with the new data
        self._update_slider_bounds()

    def _show_view_context_menu(self, pos):
        """Right-click menu: what to look at, and what to look at it with."""
        menu = QtWidgets.QMenu(self)

        has_overlay = self.scal_l is not None or self.scal_r is not None
        peak = menu.addAction("Go to peak")
        peak.setEnabled(has_overlay)
        peak.triggered.connect(lambda: self.go_to_peak())
        clusters = menu.addAction("Clusters…")
        clusters.setEnabled(has_overlay)
        clusters.triggered.connect(self.show_cluster_table)

        # Naming a region only means something for a surface the atlas fits,
        # so it is chosen by hand rather than guessed
        atlas_menu = menu.addMenu("Atlas")
        none_action = atlas_menu.addAction("None")
        none_action.setCheckable(True)
        none_action.setChecked(self.atlas_path is None)
        none_action.triggered.connect(lambda: self.set_atlas(None))
        atlas_menu.addSeparator()
        for name, path in available_surface_atlases():
            action = atlas_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(self.atlas_path == path)
            action.triggered.connect(lambda _c=False, p=path: self.set_atlas(p))
        atlas_menu.addSeparator()
        atlas_menu.addAction("Other…").triggered.connect(self._choose_atlas)
        atlas_menu.addSeparator()
        borders = atlas_menu.addAction("Show region borders")
        borders.setCheckable(True)
        borders.setChecked(self.show_borders)
        borders.setEnabled(self._atlas is not None)
        borders.triggered.connect(
            lambda checked=False: self.set_atlas_borders(checked))

        surfaces = self.available_surface_types()
        if surfaces:
            surface_menu = menu.addMenu("Surface")
            current = self.current_surface_type()
            for mesh_type, path in surfaces:
                action = surface_menu.addAction(mesh_type)
                action.setCheckable(True)
                action.setChecked(mesh_type == current)
                action.triggered.connect(
                    lambda _c=False, p=path: self.switch_surface_type(p))

        # Which shape is shown and what is painted on it are two questions
        underlay_menu = menu.addMenu("Underlay")
        for label, token in self.available_underlays():
            action = underlay_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.underlay == token)
            action.triggered.connect(lambda _c=False, k=token: self.set_underlay(k))

        menu.addSeparator()
        lock = menu.addAction("Lock zoom (mouse and touchpad)")
        lock.setCheckable(True)
        lock.setChecked(self.lock_zoom)
        lock.triggered.connect(lambda checked=False: self.set_lock_zoom(checked))
        zoom_menu = menu.addMenu("Zoom")
        zoom_menu.addAction("Zoom in").triggered.connect(lambda: self.zoom_by(1.2))
        zoom_menu.addAction("Zoom out").triggered.connect(lambda: self.zoom_by(1 / 1.2))
        zoom_menu.addAction("Reset view").triggered.connect(
            lambda: (self._fit_camera(), self.rw.Render()))

        menu.addAction(self.act_show_controls)
        menu.addAction("Save screenshot…").triggered.connect(self._save_screenshot_dialog)
        menu.addAction("Keyboard shortcuts…").triggered.connect(self.show_shortcut_help)
        menu.exec(self.vtk_widget.mapToGlobal(pos))

    def _choose_atlas(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose a surface atlas", str(Path(self.opts.mesh_left or '.').parent),
            "FreeSurfer annotation (*.annot);;All files (*)")
        if path:
            self.set_atlas(path)

    def _save_screenshot_dialog(self):
        """Ask where to write a PNG of the montage and save it."""
        start = str(Path(self.opts.mesh_left or '.').with_suffix('.png'))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save screenshot", start, "PNG image (*.png)")
        if path:
            self.save_png(path)

    #: Keys, and what they do, for the help dialog
    SHORTCUTS = (
        ("u / d / l / r", "rotate up, down, left, right (shift: 1°, ctrl: 180°)"),
        ("b", "flip the view"),
        ("o", "reset the view"),
        ("m", "go to the strongest vertex of the overlay"),
        ("+ / -", "zoom in and out (the mouse does not zoom, see the menu)"),
        ("← / →", "step through overlays (or meshes)"),
        ("g", "save a screenshot"),
        ("w / s", "wireframe / shaded"),
        ("h", "this list"),
        ("Ctrl/Cmd+D", "show or hide the control panel"),
    )

    def show_shortcut_help(self):
        """List the keys in a dialog — printing them helps nobody in an app."""
        rows = "".join(f"<tr><td><b>{keys}</b>&nbsp;&nbsp;</td><td>{what}</td></tr>"
                       for keys, what in self.SHORTCUTS)
        QtWidgets.QMessageBox.information(
            self, "Keyboard shortcuts",
            f"<table>{rows}</table>"
            "<p>Click the surface to mark a vertex; right-click for the "
            "display settings. Dropping a surface or an overlay on the window "
            "opens it.</p>")

    # ---------- drag and drop ----------
    def dragEnterEvent(self, event):
        if any(self._droppable(url) for url in event.mimeData().urls()):
            event.acceptProposedAction()

    dragMoveEvent = dragEnterEvent

    @staticmethod
    def _droppable(url) -> bool:
        return bool(url.isLocalFile()
                    and str(url.toLocalFile()).lower().endswith(
                        ('.gii', '.annot', '.txt', '.vtk', '.vtp')))

    def dropEvent(self, event):
        """A dropped mesh replaces the surface, anything else becomes overlay."""
        paths = [url.toLocalFile() for url in event.mimeData().urls()
                 if self._droppable(url)]
        if not paths:
            return
        event.acceptProposedAction()
        meshes = [p for p in paths if is_gifti_mesh_file(p)]
        others = [p for p in paths if p not in meshes]
        if meshes:
            self.switch_surface_type(meshes[0])
        annots = [p for p in others if p.lower().endswith('.annot')]
        overlays = [p for p in others if p not in annots]
        if annots:
            self.set_atlas(annots[0])
        if overlays:
            self.overlay_list = list(overlays)
            self.current_overlay_index = 0
            try:
                self.ctrl.overlay_combo.clear()
                for path in self.overlay_list:
                    self.ctrl.overlay_combo.addItem(path)
                self.ctrl.overlay_combo.setCurrentIndex(0)
            except Exception:
                pass
            self._set_overlay_from_path(overlays[0])

    def _fit_camera(self):
        """Frame the whole scene, keeping the direction it is viewed from.

        ResetCamera only moves the camera and adjusts the scale, so a rotated
        montage stays as the user left it.
        """
        self.ren.ResetCamera()
        if self.is_flat_surface(self.poly_l):
            # ResetCamera fits the bounding sphere, which leaves a wide flat
            # map small in the middle of the window
            self._zoom_to_fit()
        else:
            self.ren.GetActiveCamera().Zoom(2.0)
        self._capture_camera_state()

    def _zoom_to_fit(self, margin: float = 0.92):
        """Zoom so the scene fills the window, whatever shape it has."""
        bounds = self.ren.ComputeVisiblePropBounds()
        width = bounds[1] - bounds[0]
        height = bounds[3] - bounds[2]
        if width <= 0 or height <= 0:
            return
        camera = self.ren.GetActiveCamera()
        window = self.rw.GetSize()
        aspect = (window[0] / window[1]) if window[1] else 1.0
        if camera.GetParallelProjection():
            visible_height = 2.0 * camera.GetParallelScale()
        else:
            visible_height = 2.0 * camera.GetDistance() * math.tan(
                math.radians(camera.GetViewAngle() / 2.0))
        visible_width = visible_height * aspect
        camera.Zoom(margin * min(visible_height / height, visible_width / width))
        self.ren.ResetCameraClippingRange()

    def _mesh_title(self, path: str) -> str:
        """Window title for a surface, numbered when several were given."""
        name = self.opts.title or _title_from_path(path)
        meshes = getattr(self, 'mesh_list', None) or []
        if self.opts.title or len(meshes) < 2:
            return name
        return f"{name} ({self.current_mesh_index + 1}/{len(meshes)})"

    def _post_init_render(self):
        try:
            self.vtk_widget.Initialize()
            self.vtk_widget.setFocus()
        except Exception:
            pass
        try:
            self._fit_camera()
            self._base_cam_state = dict(self._cam_state) if self._cam_state else None
            if not self.opts.overlay:
                self.setWindowTitle(self._mesh_title(self.opts.mesh_left or ""))
            self.rw.Render()
        except Exception:
            pass
        # Batch mode: -output writes the rendered view and terminates, so the
        # viewer can be scripted over many files without user interaction.
        if getattr(self.opts, 'output', None):
            QTimer.singleShot(0, self._save_and_quit)

    def _save_and_quit(self):
        """Write the screenshot requested via -output and close the application."""
        status = 0
        try:
            self.rw.Render()
            self.save_png(self.opts.output)
        except Exception as e:
            print(f"Failed to save {self.opts.output}: {e}", file=sys.stderr)
            status = 1
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.exit(status)
        else:  # pragma: no cover - only when run without an event loop
            sys.exit(status)


    # -- Control panel integration --
    def _build_control_panel(self):
        self.ctrl = ControlPanel(self)
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setObjectName("ControlsDock")
        dock.setWidget(self.ctrl)
        # Expose as attribute for any external references
        self.dock_controls = dock
    
        # Dock features: PyQt6 compatibility
        DockFeature = getattr(QtWidgets.QDockWidget, "DockWidgetFeature", QtWidgets.QDockWidget)
        dock.setFeatures(
            getattr(DockFeature, "DockWidgetMovable")
            | getattr(DockFeature, "DockWidgetFloatable")
            | getattr(DockFeature, "DockWidgetClosable")
        )
        dock.setAllowedAreas(DOCK_RIGHT | DOCK_LEFT)
        # Float by default so the dock does not affect the render view layout
        dock.setFloating(True)
        
        # Position the floating dock on the right side of the main window
        def position_dock():
            if dock.isFloating():
                main_geometry = self.geometry()
                dock_width = dock.sizeHint().width()
                dock_height = dock.sizeHint().height()
                dock_x = main_geometry.x() + main_geometry.width() - dock_width
                dock_y = main_geometry.y()
                dock.setGeometry(dock_x, dock_y, dock_width, dock_height)

        # Connect to show event to position the dock with a proper void-returning handler
        def _dock_show_event(event):
            # Call the base implementation correctly, then position
            QtWidgets.QDockWidget.showEvent(dock, event)
            position_dock()
        dock.showEvent = _dock_show_event

        self.addDockWidget(DOCK_RIGHT, dock)
    
        # ---------- local helpers (closures) ----------
        # programmatic toggle (menu/shortcut)
        def _toggle_controls_local(checked: bool):
            if checked == dock.isVisible():
                return
            # Show/hide the dock as a floating window (no layout shift)
            if checked:
                dock.setFloating(True)
                dock.show()
                position_dock()
            else:
                dock.hide()
            # Update status bar hint
            self._update_status_message(checked)
    
        self._toggle_controls = _toggle_controls_local
    
        # user-driven visibility change (dock close button / drag)
        def _on_vis_changed_local(visible: bool):
            # sync menu action
            if hasattr(self, "act_show_controls"):
                self.act_show_controls.blockSignals(True)
                self.act_show_controls.setChecked(visible)
                self.act_show_controls.blockSignals(False)
    
            # No window resizing needed - dock overlays on the right side
            # Update status bar hint
            self._update_status_message(visible)
    
        dock.visibilityChanged.connect(_on_vis_changed_local)
        # ----------------------------------------------
    
        # Seed values
        if self.overlay_range[1] > self.overlay_range[0]:
            self.ctrl.range_min.setValue(float(self.overlay_range[0]))
            self.ctrl.range_max.setValue(float(self.overlay_range[1]))
        if self.opts.clip[1] > self.opts.clip[0]:
            self.ctrl.clip_min.setValue(float(self.opts.clip[0]))
            self.ctrl.clip_max.setValue(float(self.opts.clip[1]))
        self.ctrl.bkg_min.setValue(float(self.range_bkg[0]))
        self.ctrl.bkg_max.setValue(float(self.range_bkg[1]))
        self.ctrl.opacity.setValue(int(self.opts.opacity * 100))
        # Populate overlay selector with current overlays and selection
        try:
            self.ctrl.overlay_combo.clear()
            # If multiple overlays are provided, list them all for direct selection
            if self.overlay_list:
                for p in self.overlay_list:
                    self.ctrl.overlay_combo.addItem(p)
                self.ctrl.overlay_combo.setCurrentIndex(0)
                self._update_overlay_info()
            else:
                # Fall back to a single path if present
                single = self.opts.overlay or ""
                if single:
                    self.ctrl.overlay_combo.addItem(single)
                self.ctrl.overlay_combo.setEditText(single)
        except Exception:
            pass
        # Initialize control states (ensure clean ASCII spaces for indentation)
        self.ctrl.cb_colorbar.setChecked(self.opts.colorbar)
        self.ctrl.set_threshold_visible(self._uses_logp_scale())
        self.ctrl.set_threshold_from_clip(self.opts.clip)
        self.ctrl.title_mode.setCurrentText(self.opts.title_mode)
        self.ctrl.cb_inverse.setChecked(self.opts.inverse)
        self.ctrl.cb_fix_scaling.setChecked(self.opts.fix_scaling)
        # Histogram toggle initial state (off)
        try:
            self.ctrl.cb_histogram.setChecked(False)
        except Exception:
            pass
        # Initialize colormap selector based on opts.colormap
        cm_index_map = {JET: 0, HOT: 1, FIRE: 2, BIPOLAR: 3, GRAY: 4, C1: 5, C2: 6, C3: 7}
        try:
            self.ctrl.colormap.setCurrentIndex(cm_index_map.get(self.opts.colormap, 0))
        except Exception:
            pass
        # Initialize discrete checkbox from opts (consider non-zero as on)
        if hasattr(self.ctrl, 'cb_discrete'):
            disc = int(getattr(self.opts, 'discrete', 0) or 0)
            self.ctrl.cb_discrete.setChecked(disc > 0)
        # Enable/disable overlay controls based on whether overlay is loaded
        has_overlay = bool((self.overlay_list or self.opts.overlay)
                           and self._overlay_scalars() is not None)
        self.ctrl.set_overlay_controls_enabled(has_overlay)
        # Ensure fix scaling checkbox state reflects current overlay count/availability
        self._enforce_fix_scaling_policy()
        # Signals
        # Removed reset button; reset available via keyboard 'o' or menu if needed
        self.ctrl.overlay_btn.clicked.connect(self._pick_overlay)
        # Open NIfTI volume in orthogonal view window
        try:
            self.ctrl.volume_btn.clicked.connect(self._open_volume_dialog)
        except Exception:
            pass
        # Auto-load overlay when selection changes
        try:
            self.ctrl.overlay_combo.currentIndexChanged.connect(self._on_overlay_combo_changed)
        except Exception:
            pass
        # Auto-load overlay when edit text changes (Enter or focus leave)
        try:
            self.ctrl.overlay_combo.lineEdit().editingFinished.connect(self._on_overlay_combo_edited)
        except Exception:
            pass
        # Colormap selection handler
        def _on_colormap_changed(idx: int):
            # Map UI index back to enum
            idx_to_cm = {0: JET, 1: HOT, 2: FIRE, 3: BIPOLAR, 4: GRAY, 5: C1, 6: C2, 7: C3}
            self.opts.colormap = idx_to_cm.get(int(idx), JET)
            # Rebuild LUTs respecting inverse and discrete
            self.lut_overlay_l = get_lookup_table(self.opts.colormap, self.opts.opacity)
            self.lut_overlay_r = get_lookup_table(self.opts.colormap, self.opts.opacity)
            if self.opts.inverse:
                self._invert_lut(self.lut_overlay_l)
                self._invert_lut(self.lut_overlay_r)
            self._apply_discrete_to_overlay_lut(self.lut_overlay_l)
            self._apply_discrete_to_overlay_lut(self.lut_overlay_r)
            self._apply_clip_to_overlay_luts()
            if self.actor_ov_l is not None:
                self.actor_ov_l.GetMapper().SetLookupTable(self.lut_overlay_l)
            if self.actor_ov_r is not None:
                self.actor_ov_r.GetMapper().SetLookupTable(self.lut_overlay_r)
            if self.opts.colorbar:
                self._ensure_colorbar()
            self.rw.Render()
        self.ctrl.colormap.currentIndexChanged.connect(_on_colormap_changed)
        if hasattr(self.ctrl, 'cb_discrete'):
            def _on_discrete_toggled(checked: bool):
                # Use 2 levels by default when checked
                self.opts.discrete = 16 if checked else 0
                # Rebuild overlay LUTs with new discrete setting
                self.lut_overlay_l = get_lookup_table(self.opts.colormap, self.opts.opacity)
                self.lut_overlay_r = get_lookup_table(self.opts.colormap, self.opts.opacity)
                if self.opts.inverse:
                    self._invert_lut(self.lut_overlay_l)
                    self._invert_lut(self.lut_overlay_r)
                self._apply_discrete_to_overlay_lut(self.lut_overlay_l)
                self._apply_discrete_to_overlay_lut(self.lut_overlay_r)
                # Reapply clip transparency so values inside clip window stay transparent
                self._apply_clip_to_overlay_luts()
                if self.actor_ov_l is not None:
                    self.actor_ov_l.GetMapper().SetLookupTable(self.lut_overlay_l)
                    if self.overlay_range[1] > self.overlay_range[0]:
                        self.actor_ov_l.GetMapper().SetScalarRange(self.overlay_range)
                if self.actor_ov_r is not None:
                    self.actor_ov_r.GetMapper().SetLookupTable(self.lut_overlay_r)
                    if self.overlay_range[1] > self.overlay_range[0]:
                        self.actor_ov_r.GetMapper().SetScalarRange(self.overlay_range)
                if self.opts.colorbar:
                    self._ensure_colorbar()
                self.rw.Render()
            self.ctrl.cb_discrete.toggled.connect(_on_discrete_toggled)

        # Live: overlay range (spin + slider)
        def _on_overlay_range_changed():
            r0 = float(self.ctrl.range_min.value()); r1 = float(self.ctrl.range_max.value())
            if r1 > r0:
                self.overlay_range = [r0, r1]
                for actor in (self.actor_ov_l, self.actor_ov_r):
                    if actor:
                        actor.GetMapper().SetScalarRange(self.overlay_range)
                if self.scalar_bar is not None:
                    lut_cb = self.scalar_bar.GetLookupTable()
                    if lut_cb is not None:
                        lut_cb.SetTableRange(self.overlay_range)
                self.rw.Render()
        self.ctrl.range_min.valueChanged.connect(lambda _=None: _on_overlay_range_changed())
        self.ctrl.range_max.valueChanged.connect(lambda _=None: _on_overlay_range_changed())
        self.ctrl.range_slider_min.valueChanged.connect(lambda _=None: _on_overlay_range_changed())
        self.ctrl.range_slider_max.valueChanged.connect(lambda _=None: _on_overlay_range_changed())

        # Live: background range (spin + slider)
        def _on_bkg_range_changed():
            b0 = float(self.ctrl.bkg_min.value()); b1 = float(self.ctrl.bkg_max.value())
            if b1 > b0:
                self.range_bkg = [b0, b1]
                for actor in (getattr(self, 'actor_bkg_l', None), getattr(self, 'actor_bkg_r', None)):
                    if actor:
                        actor.GetMapper().SetScalarRange(self.range_bkg)
                self.rw.Render()
        self.ctrl.bkg_min.valueChanged.connect(lambda _=None: _on_bkg_range_changed())
        self.ctrl.bkg_max.valueChanged.connect(lambda _=None: _on_bkg_range_changed())
        self.ctrl.bkg_slider_min.valueChanged.connect(lambda _=None: _on_bkg_range_changed())
        self.ctrl.bkg_slider_max.valueChanged.connect(lambda _=None: _on_bkg_range_changed())

        # Live: opacity
        def _on_opacity_changed(val: int):
            self.opts.opacity = max(0.0, min(1.0, float(val)/100.0))
            self.lut_overlay_l = get_lookup_table(self.opts.colormap, self.opts.opacity)
            self.lut_overlay_r = get_lookup_table(self.opts.colormap, self.opts.opacity)
            if self.opts.inverse:
                self._invert_lut(self.lut_overlay_l)
                self._invert_lut(self.lut_overlay_r)
            self._apply_discrete_to_overlay_lut(self.lut_overlay_l)
            self._apply_discrete_to_overlay_lut(self.lut_overlay_r)
            # Reapply clip transparency after opacity change
            self._apply_clip_to_overlay_luts()
            if self.actor_ov_l is not None:
                self.actor_ov_l.GetMapper().SetLookupTable(self.lut_overlay_l)
            if self.actor_ov_r is not None:
                self.actor_ov_r.GetMapper().SetLookupTable(self.lut_overlay_r)
            if self.opts.colorbar:
                self._ensure_colorbar()
            self.rw.Render()
        self.ctrl.opacity.valueChanged.connect(_on_opacity_changed)

        # Live: colorbar toggle
        def _on_colorbar_toggled(checked: bool):
            self.opts.colorbar = bool(checked)
            # Ensure the actor exists and is up to date, then attach/detach
            self._ensure_colorbar()
            if bool(checked):
                self._attach_colorbar()
            else:
                self._detach_colorbar()
            # Keep control states in sync with colorbar visibility
            try:
                en = bool(checked) and self.ctrl.cb_colorbar.isEnabled()
                # Title only matters when colorbar is visible
                self.ctrl.title_mode.setEnabled(en)
                # Discrete remains enabled independent of colorbar visibility
            except Exception:
                pass
            self.rw.Render()
        self.ctrl.cb_colorbar.toggled.connect(_on_colorbar_toggled)

        # Live: title mode change
        def _on_title_mode_changed(_text: str):
            self.opts.title_mode = _text
            if self.opts.colorbar:
                self._ensure_colorbar()
                self.rw.Render()
        self.ctrl.title_mode.currentTextChanged.connect(_on_title_mode_changed)

        # Live: inverse toggle
        def _on_inverse_toggled(checked: bool):
            if bool(self.opts.inverse) == bool(checked):
                return
            self.opts.inverse = bool(checked)
            self._apply_inverse()
            self.rw.Render()
        self.ctrl.cb_inverse.toggled.connect(_on_inverse_toggled)

        # Live: fix scaling toggle
        def _on_fix_scaling_toggled(checked: bool):
            self.opts.fix_scaling = bool(checked)
            if self.opts.fix_scaling:
                # capture current as fixed
                self.fixed_overlay_range = list(self.overlay_range)
            else:
                # recompute from current data
                if self.scal_l is not None:
                    r = [0.0, 0.0]
                    self.poly_l.GetScalarRange(r)
                    self.overlay_range = [float(r[0]), float(r[1])] if r[1] > r[0] else self.overlay_range
            # Apply to actors and UI
            for actor in (self.actor_ov_l, self.actor_ov_r):
                if actor and self.overlay_range[1] > self.overlay_range[0]:
                    actor.GetMapper().SetScalarRange(self.overlay_range)
            if hasattr(self, 'ctrl') and self.overlay_range[1] > self.overlay_range[0]:
                self.ctrl.range_min.setValue(float(self.overlay_range[0]))
                self.ctrl.range_max.setValue(float(self.overlay_range[1]))
            if self.opts.colorbar:
                self._ensure_colorbar()
            self.rw.Render()
        self.ctrl.cb_fix_scaling.toggled.connect(_on_fix_scaling_toggled)

        # Live: histogram toggle
        def _on_histogram_toggled(checked: bool):
            self._toggle_histogram(checked)
        self.ctrl.cb_histogram.toggled.connect(_on_histogram_toggled)

        # Live-ish: clip window — apply on slider release or editing finished
        def _apply_clip_live():
            c0 = float(self.ctrl.clip_min.value()); c1 = float(self.ctrl.clip_max.value())
            # Treat (0,0) as disabled, same convention as _apply_controls
            self.opts.clip = (c0, c1) if c1 > c0 else (0.0, 0.0)
            # The colours span from the threshold to the maximum, so moving the
            # threshold moves the lower end with it, the way cat_surf_results
            # sets clim right after clip.  A range the user asked for is kept.
            self._rescale_overlay_to_clip()
            # Re-apply clip by updating LUT alpha (no data mutation)
            self.lut_overlay_l = get_lookup_table(self.opts.colormap, self.opts.opacity)
            self.lut_overlay_r = get_lookup_table(self.opts.colormap, self.opts.opacity)
            if self.opts.inverse:
                self._invert_lut(self.lut_overlay_l)
                self._invert_lut(self.lut_overlay_r)
            self._apply_discrete_to_overlay_lut(self.lut_overlay_l)
            self._apply_discrete_to_overlay_lut(self.lut_overlay_r)
            self._apply_clip_to_overlay_luts()
            if self.actor_ov_l is not None:
                self.actor_ov_l.GetMapper().SetLookupTable(self.lut_overlay_l)
            if self.actor_ov_r is not None:
                self.actor_ov_r.GetMapper().SetLookupTable(self.lut_overlay_r)
            if self.opts.colorbar:
                self._ensure_colorbar()
            # Keep the threshold entry in sync when the clip is edited by hand
            self.ctrl.set_threshold_from_clip(self.opts.clip)
            self.rw.Render()
        self.ctrl.clip_slider_min.sliderReleased.connect(_apply_clip_live)
        self.ctrl.clip_slider_max.sliderReleased.connect(_apply_clip_live)
        self.ctrl.clip_min.editingFinished.connect(_apply_clip_live)
        self.ctrl.clip_max.editingFinished.connect(_apply_clip_live)

        # Threshold for -log10(p) overlays: hides everything below it, which is
        # the clip window the spin boxes above show
        def _on_threshold_changed(index: int):
            try:
                value = LOGP_THRESHOLDS[int(index)][1]
            except Exception:
                return
            self.ctrl.clip_min.blockSignals(True); self.ctrl.clip_max.blockSignals(True)
            self.ctrl.clip_min.setValue(-value)
            self.ctrl.clip_max.setValue(value)
            self.ctrl.clip_min.blockSignals(False); self.ctrl.clip_max.blockSignals(False)
            self.ctrl._spin_to_slider('clip', 'min', -value)
            self.ctrl._spin_to_slider('clip', 'max', value)
            _apply_clip_live()
        self.ctrl.threshold.currentIndexChanged.connect(_on_threshold_changed)
    
        # Start state based on CLI flag --panel (default hidden)
        if self.opts.panel:
            dock.setFloating(True)
            dock.show()
            position_dock()
        else:
            dock.hide()
        # Initial status hint, next to the readout for the marked vertex
        self._build_pick_label()
        self.setAcceptDrops(True)      # a dropped surface or overlay opens
        self._update_status_message(self.opts.panel)

        # Shade the surface the way the menu would, so switching surface or
        # underlay later changes only what was asked for.  A background given
        # with -bkg is the user's own and stays as it is.
        if not self.opts.overlay_bkg:
            self.set_underlay(self.underlay)
        
        # Initialize slider bounds from current data
        self._update_slider_bounds()
    
        # View menu + keyboard shortcut
        menubar = self.menuBar()
        menu = menubar.addMenu("View")
        act = QAction("Show Controls", self)
        act.setCheckable(True)
        act.setChecked(self.opts.panel)
        # No direct shortcut on the QAction — QShortcuts handle the key bindings reliably
        act.triggered.connect(self._toggle_controls)
        menu.addAction(act)
        # Register action with the window so its shortcut is active
        self.addAction(act)
        self.act_show_controls = act
        # Volume menu: open a 3D NIfTI in a separate orthogonal view window
        vol_menu = menubar.addMenu("Volume")
        act_open_vol = QAction("Open NIfTI…", self)
        act_open_vol.triggered.connect(self._open_volume_dialog)
        vol_menu.addAction(act_open_vol)

        # Add a single QShortcut on the main window (Ctrl+D -> mapped to Cmd+D on macOS)
        try:
            self._dock_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
            try:
                self._dock_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            except Exception:
                self._dock_shortcut.setContext(Qt.ApplicationShortcut)
            self._dock_shortcut.activated.connect(lambda: (
                self._toggle_controls(not self.dock_controls.isVisible())
                if hasattr(self, 'dock_controls') and self.dock_controls is not None else None
            ))
        except Exception:
            pass

        # Auto-open volume if provided via CLI
        try:
            if getattr(self.opts, 'volume', None):
                self._open_volume(self.opts.volume)
        except Exception as e:
            print(f"Failed to open volume '{self.opts.volume}': {e}")

    # ---------- what is under the cursor ----------
    def _pick_value(self, side: int, vertex: int) -> Optional[float]:
        """Overlay value of a vertex, or None when there is no overlay."""
        scalars = self.scal_r if side == 1 else self.scal_l
        if scalars is None or vertex is None:
            return None
        if vertex >= scalars.GetNumberOfTuples():
            return None
        try:
            return float(scalars.GetTuple1(int(vertex)))
        except Exception:
            return None

    def set_atlas(self, path: Optional[str]):
        """Name the region under the cursor using the annotation *path*.

        The right hemisphere is taken from the matching file next to it.  An
        atlas only fits a mesh with the same number of vertices (the 32k
        templates T1Prep works with), so a mismatch is reported rather than
        silently naming the wrong region.  None switches the lookup off.
        """
        self._atlas = None
        self.atlas_path = None
        if not path:
            if self.show_borders:
                self._build_border_actors()
                self._build_montage()
                self.rw.Render()
            self._update_pick_label()
            return
        try:
            own, names = read_annotation(path)
            other_labels = None
            other = _hemi_counterpart(Path(path))
            if other is not None and other.exists():
                other_labels, _ = read_annotation(str(other))
            # An rh.* file names the right hemisphere, whichever of the two was
            # chosen: putting it on the left surface would draw every region in
            # the wrong place
            left, right = order_by_hemisphere(path, own, other_labels)
            labels = {0: left}
            if right is not None:
                labels[1] = right
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "Atlas", str(exc))
            self._update_pick_label()
            return

        for side, poly in ((0, self.poly_l), (1, self.poly_r)):
            if poly is None or side not in labels:
                continue
            if poly.GetNumberOfPoints() != len(labels[side]):
                QtWidgets.QMessageBox.warning(
                    self, "Atlas",
                    f"{os.path.basename(path)} has {len(labels[side])} vertices, "
                    f"the surface {poly.GetNumberOfPoints()} — they do not match.")
                self._update_pick_label()
                return
        stem = os.path.basename(path)
        if stem.startswith('lh.'):
            stem = stem[3:]
        self._atlas = {'labels': labels, 'names': names,
                       'name': stem[:-len('.annot')] if stem.endswith('.annot') else stem}
        self.atlas_path = path
        if self.show_borders:
            self._build_border_actors()
            self._build_montage()
            self.rw.Render()
        self._update_pick_label()

    def set_atlas_borders(self, visible: bool):
        """Show or hide the boundaries between the regions of the atlas."""
        self.show_borders = bool(visible)
        self._build_border_actors()
        self._build_montage()
        self.rw.Render()

    def _build_border_actors(self):
        """One actor per hemisphere holding the atlas boundaries."""
        for attribute in ('actor_border_l', 'actor_border_r'):
            actor = getattr(self, attribute, None)
            if actor is not None:
                try:
                    self.ren.RemoveActor(actor)
                except Exception:
                    pass
            setattr(self, attribute, None)
        if not (self.show_borders and self._atlas):
            return
        for side, poly, attribute in ((0, self.poly_l, 'actor_border_l'),
                                      (1, self.poly_r, 'actor_border_r')):
            labels = self._atlas['labels'].get(side)
            if poly is None or labels is None:
                continue
            border = atlas_border_lines(poly, labels)
            if border.GetNumberOfLines() == 0:
                continue
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(border)
            mapper.ScalarVisibilityOff()
            # Just enough bias to win against the surface the lines lie on.
            # VTK's default for lines (-66000) is far larger and lets the
            # borders inside the sulci show through the gyri in front of them,
            # which fills the view with lines cat_surf_results does not draw.
            try:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0.0, -1.0)
            except Exception:
                pass
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.0, 0.0)   # black, as in CAT12
            actor.GetProperty().SetLineWidth(2.0)
            actor.GetProperty().SetLighting(False)
            setattr(self, attribute, actor)

    def _atlas_region(self, side: int, vertex: int) -> Optional[str]:
        """Region name of the selected atlas at a vertex, if any."""
        if not self._atlas or vertex is None:
            return None
        labels = self._atlas['labels'].get(side)
        if labels is None or vertex >= len(labels):
            return None
        index = int(labels[vertex])
        if index < 0 or index >= len(self._atlas['names']):
            return None
        name = self._atlas['names'][index]
        return None if name.lower() in ('unknown', '???', 'none', '') else name

    def _pick_text(self) -> str:
        """The line describing the marked vertex."""
        if not getattr(self, '_cursor_vertex', None):
            return ""
        side, vertex = self._cursor_vertex
        poly = self.poly_r if side == 1 else self.poly_l
        if poly is None or vertex >= poly.GetNumberOfPoints():
            return ""
        shift = getattr(self, '_y_shift_r' if side == 1 else '_y_shift_l', 0.0)
        x, y, z = poly.GetPoint(int(vertex))
        parts = [f"{'rh' if side == 1 else 'lh'} vertex {int(vertex)}",
                 f"({x:.1f}, {y - float(shift):.1f}, {z:.1f}) mm"]
        value = self._pick_value(side, vertex)
        if value is not None:
            parts.append(f"value {value:g}")
            if self._uses_logp_scale():
                parts.append(f"p {format_p_value_label(value)}")
        region = self._atlas_region(side, vertex)
        if self._atlas:
            parts.append(f"{self._atlas['name']}: {region or '-'}")
        return "    ".join(parts)

    def _update_pick_label(self):
        """Put the description of the marked vertex into the status bar."""
        label = getattr(self, '_pick_label', None)
        if label is not None:
            label.setText(self._pick_text())

    def go_to_peak(self):
        """Mark the strongest vertex of the overlay and say where it is.

        The peak of a statistical map is the first thing to look at, and
        rotating a surface until it appears is no way to find it.
        """
        best = None    # (|value|, side, vertex)
        for side, scalars in ((0, self.scal_l), (1, self.scal_r)):
            if scalars is None:
                continue
            values = np.abs(vtk_to_numpy(scalars).astype(float))
            if values.size == 0:
                continue
            vertex = int(np.nanargmax(values))
            if best is None or values[vertex] > best[0]:
                best = (float(values[vertex]), side, vertex)
        if best is None:
            self.statusBar().showMessage("No overlay to find a peak in", 4000)
            return None
        _magnitude, side, vertex = best
        poly = self.poly_r if side == 1 else self.poly_l
        shift = getattr(self, '_y_shift_r' if side == 1 else '_y_shift_l', 0.0)
        x, y, z = poly.GetPoint(vertex)
        mm = (x, y - float(shift), z)
        self._set_surface_cursor(mm)
        self._broadcast_world_pick(mm)
        return mm

    def _build_pick_label(self):
        """The readout for the marked vertex, in the status bar.

        A permanent widget, because showMessage() hides the ordinary ones and
        the hints would otherwise wipe out what was just picked.
        """
        self._pick_label = QtWidgets.QLabel("")
        self._pick_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)   # copy a coordinate
        self.statusBar().addPermanentWidget(self._pick_label)

    def _rescale_overlay_to_clip(self):
        """Recompute the value range after the clip window changed.

        Unless the range was given on the command line or is held fixed across
        overlays: then it is the user's, and the clip only hides values.
        """
        if getattr(self, '_user_set_range', False) or self.opts.fix_scaling:
            return
        scaled = self._auto_overlay_range()
        if scaled is None or not (scaled[1] > scaled[0]):
            return
        self.overlay_range = list(scaled)
        for actor in (self.actor_ov_l, self.actor_ov_r):
            if actor is not None:
                actor.GetMapper().SetScalarRange(self.overlay_range)
        self._apply_clip_to_overlay_luts()
        if hasattr(self, 'ctrl'):
            try:
                self.ctrl.set_overlay_bounds(*self.overlay_range)
                for widget, value in ((self.ctrl.range_min, self.overlay_range[0]),
                                      (self.ctrl.range_max, self.overlay_range[1])):
                    widget.blockSignals(True)
                    widget.setValue(float(value))
                    widget.blockSignals(False)
                self.ctrl._spin_to_slider('overlay', 'min', float(self.overlay_range[0]))
                self.ctrl._spin_to_slider('overlay', 'max', float(self.overlay_range[1]))
            except Exception:
                pass

    def _auto_overlay_range(self) -> Optional[Tuple[float, float]]:
        """The default range for the overlay now loaded, both hemispheres."""
        values = []
        for scalars in (self.scal_l, self.scal_r):
            if scalars is not None:
                values.append(vtk_to_numpy(scalars).astype(float))
        if not values:
            return None
        clip = tuple(self.opts.clip or (0.0, -1.0))
        threshold = float(clip[1]) if clip[1] > clip[0] else 0.0
        return default_overlay_range(np.concatenate(values), threshold,
                                     self._uses_logp_scale())

    # ---------- underlay ----------
    #: What can be shaded under the overlay, as (label, file token).  The
    #: files sit next to the surface: lh.mc.* is the mean curvature,
    #: lh.sqrtsulc.* the sulcal depth, as in cat_surf_results.
    UNDERLAYS = (("Mean curvature", "mc"),
                 ("Sulcal depth", "sqrtsulc"),
                 ("None", None))

    def _underlay_file(self, token: str) -> Optional[Path]:
        """``lh.<token>.<rest>`` next to the current surface, if it exists."""
        current = self.opts.mesh_left
        if not current or not token:
            return None
        path = Path(current)
        parts = path.name.split('.')
        for index, part in enumerate(parts):
            if part not in MESH_TYPE_TOKENS:
                continue
            candidate = list(parts)
            candidate[index] = token
            sibling = path.with_name('.'.join(candidate))
            return sibling if sibling.exists() else None
        return None

    def available_underlays(self) -> List[Tuple[str, Optional[str]]]:
        """The shading options that can actually be shown.

        Mean curvature is always there — it is computed from the surface when
        no file sits next to it — the rest only with their file.
        """
        offered = []
        for label, token in self.UNDERLAYS:
            if token in (None, 'mc') or self._underlay_file(token) is not None:
                offered.append((label, token))
        return offered

    def set_underlay(self, token: Optional[str]):
        """Shade the surface with *token* ('mc', 'sqrtsulc' or None for plain).

        Separate from the surface itself: which shape is shown and what is
        painted on it are two questions, as in cat_surf_results.
        """
        self.underlay = token
        self.bkg_scalar_l = self.bkg_scalar_r = None
        if token:
            path = self._underlay_file(token)
            if path is not None:
                try:
                    self.bkg_scalar_l = read_scalars(str(path))
                    other = _hemi_counterpart(path)
                    if other is not None and other.exists():
                        self.bkg_scalar_l, self.bkg_scalar_r = order_by_hemisphere(
                            str(path), self.bkg_scalar_l, read_scalars(str(other)))
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(
                        self, "Underlay", f"Cannot read {path.name}:\n{exc}")
                    self.bkg_scalar_l = self.bkg_scalar_r = None
        self._apply_underlay()

    @staticmethod
    def shade_from_curvature(values, invert: bool = False) -> "np.ndarray":
        """Turn curvature (or depth) into the grey level CAT12 shades with.

        The mapping of ``cat_surf_results.m``: a signed square root, which
        pulls in the long tails that otherwise make the surface all black and
        white, then shifted and normalised into a light band.  Sulcal depth
        has no negative side and is inverted, so its sulci go dark like the
        curvature ones.

        Args:
            values: One curvature or depth value per vertex.
            invert: Flip the greys (sulcal depth).

        Returns:
            Grey levels between 0 and 1.
        """
        shaded = np.asarray(values, dtype=float).copy()
        negative = shaded < 0
        positive = shaded > 0
        shaded[negative] = -np.sqrt(-shaded[negative])
        shaded[positive] = np.sqrt(shaded[positive])
        shaded -= shaded.min()
        shaded += 0.5
        peak = shaded.max()
        if peak > 0:
            shaded /= peak
        if invert:
            shaded = 1.0 - shaded
        # Matlab draws these greys as they are; here they go through a light
        # that adds its own contrast, so they are kept inside a band instead
        # of spanning black to white
        low, high = UNDERLAY_GREYS
        return low + shaded * (high - low)

    def _underlay_range(self) -> Optional[List[float]]:
        """A shading window that fits the data now under the surface.

        Curvature from a file and curvature computed from the mesh are the
        same quantity on different scales, and sulcal depth on yet another —
        so the window follows the data instead of a fixed pair of numbers.
        The outer percentiles are left out, or a few extreme vertices would
        flatten everything else into one grey.
        """
        if self.curv_l_out is None or vtk_to_numpy is None:
            return None
        values = []
        for output in (self.curv_l_out, self.curv_r_out):
            if output is None:
                continue
            scalars = output.GetPointData().GetScalars()
            if scalars is not None:
                values.append(np.abs(vtk_to_numpy(scalars).astype(float)))
        if not values:
            return None
        edge = float(np.percentile(np.concatenate(values), 98))
        if not np.isfinite(edge) or edge <= 0:
            return None
        return [-edge, edge]

    def _folded_curvature(self) -> List[Optional["np.ndarray"]]:
        """Mean curvature of the folded surface, whatever surface is shown.

        An inflated or flattened surface is smooth by construction, so its own
        curvature carries no relief.  cat_surf_results shades every surface
        with the curvature of the folded one, which the shared vertices make
        possible — so that is what is computed here, from the central sibling
        when the displayed surface is not it.
        """
        source = [self.poly_l, self.poly_r]
        if self.current_surface_type() in ('inflated', 'patch'):
            central = dict(self.available_surface_types()).get('central')
            if central:
                cached = getattr(self, '_folded_meshes', None) or {}
                if central not in cached:
                    try:
                        cached[central] = read_mesh_pair(central)
                    except Exception:
                        cached[central] = (None, None)
                    self._folded_meshes = cached
                left, right = cached[central]
                if left is not None:
                    source = [left, right]

        values: List[Optional["np.ndarray"]] = []
        for poly, shown in zip(source, (self.poly_l, self.poly_r)):
            if poly is None or shown is None:
                values.append(None)
                continue
            if poly.GetNumberOfPoints() != shown.GetNumberOfPoints():
                poly = shown          # not the same subject: use what is shown
            curvature = vtkCurvatures()
            curvature.SetInputData(poly)
            curvature.SetCurvatureTypeToMean()
            curvature.Update()
            scalars = curvature.GetOutput().GetPointData().GetScalars()
            values.append(vtk_to_numpy(scalars).astype(float)
                          if scalars is not None else None)
        return values

    def _apply_underlay(self):
        """Put the current shading on the surface and rescale its colours."""
        # Recompute the curvature, which the previous shading overwrote
        for filt, poly in ((getattr(self, 'curv_l', None), self.poly_l),
                           (getattr(self, 'curv_r', None), self.poly_r)):
            if filt is not None and poly is not None:
                filt.SetInputData(poly)
                filt.Modified()
                filt.Update()
        self.curv_l_out = self.curv_l.GetOutput() if getattr(self, 'curv_l', None) else None
        self.curv_r_out = (self.curv_r.GetOutput()
                           if getattr(self, 'curv_r', None) is not None else None)
        # Both hemispheres are shaded on one scale, or they would not match
        computed = None
        raw = []
        for index, (output, scalars) in enumerate(
                ((self.curv_l_out, self.bkg_scalar_l),
                 (self.curv_r_out, self.bkg_scalar_r))):
            if output is None:
                raw.append(None)
                continue
            if scalars is not None and scalars.GetNumberOfTuples() == output.GetNumberOfPoints():
                raw.append(vtk_to_numpy(scalars).astype(float))
                continue
            # No file for this shading: take the curvature of the folded surface
            if computed is None:
                computed = self._folded_curvature()
            values = computed[index] if index < len(computed) else None
            if values is not None and len(values) != output.GetNumberOfPoints():
                values = None
            raw.append(values)
        present = [values for values in raw if values is not None]
        if present:
            shaded = self.shade_from_curvature(
                np.concatenate(present), invert=(self.underlay == 'sqrtsulc'))
            at = 0
            for output, values in zip((self.curv_l_out, self.curv_r_out), raw):
                if output is None or values is None:
                    continue
                part = numpy_to_vtk(shaded[at:at + len(values)], deep=True)
                part.SetName('shading')
                output.GetPointData().SetScalars(part)
                at += len(values)

        plain = self.underlay is None
        for actor, output in ((getattr(self, 'actor_bkg_l', None), self.curv_l_out),
                              (getattr(self, 'actor_bkg_r', None), self.curv_r_out)):
            if actor is None or output is None:
                continue
            mapper = actor.GetMapper()
            mapper.SetInputData(output)
            # Nothing to shade with: an even mid-grey, so the overlay stands
            # alone.  cat_surf_results uses 0.5 and turns the lighting off,
            # which is what keeps it grey — lit, that value washes out white.
            if plain:
                mapper.ScalarVisibilityOff()
                actor.GetProperty().SetColor(*UNDERLAY_PLAIN_GREY)
                actor.GetProperty().SetLighting(False)
            else:
                mapper.ScalarVisibilityOn()
                actor.GetProperty().SetLighting(True)

        if not plain and self.curv_l_out is not None:
            # shade_from_curvature() already produced grey levels
            self.range_bkg = [0.0, 1.0]
            self.lut_bkg.SetTableRange(self.range_bkg)
            for actor in (getattr(self, 'actor_bkg_l', None),
                          getattr(self, 'actor_bkg_r', None)):
                if actor is not None:
                    actor.GetMapper().SetScalarRange(self.range_bkg)
            if hasattr(self, 'ctrl'):
                try:
                    self.ctrl.set_bkg_bounds(self.range_bkg[0], self.range_bkg[1])
                    for widget, value in ((self.ctrl.bkg_min, self.range_bkg[0]),
                                          (self.ctrl.bkg_max, self.range_bkg[1])):
                        widget.blockSignals(True)
                        widget.setValue(value)
                        widget.blockSignals(False)
                except Exception:
                    pass
        # The montage clones carry copies of the actors
        self._build_montage()
        self.rw.Render()

    # ---------- zoom ----------
    def _guard_zoom(self, event: str):
        """Take *event* away from the interactor style while zoom is locked.

        Overriding the style in Python would not help: the interactor
        dispatches to the C++ implementation.  An observer with a higher
        priority does get called, and aborting there stops the event before
        the style sees it.
        """
        tag = [None]

        def callback(obj, _event):
            if self.lock_zoom and tag[0] is not None:
                command = obj.GetCommand(tag[0])
                if command is not None:
                    command.AbortFlagOn()

        self._zoom_callbacks.append(callback)
        tag[0] = self.iren.AddObserver(event, callback, 1.0)

    def set_lock_zoom(self, locked: bool):
        """Whether the mouse or trackpad may change the zoom.

        Switching it on also ends a zoom drag that lost its button release, so
        the view stops following the mouse.
        """
        self.lock_zoom = bool(locked)
        if self.lock_zoom:
            style = self.iren.GetInteractorStyle()
            if style is not None:
                try:
                    style.EndDolly()      # a no-op unless a drag is still running
                except Exception:
                    pass

    def zoom_by(self, factor: float):
        """Zoom the view, which is what the mouse is no longer allowed to do."""
        camera = self.ren.GetActiveCamera()
        camera.Zoom(float(factor))
        self.ren.ResetCameraClippingRange()
        self.rw.Render()

    # ---------- surface type ----------
    def available_surface_types(self) -> List[Tuple[str, str]]:
        """Sibling surfaces of the current one, as (type, path) pairs.

        ``lh.central.sub-01.gii`` has ``lh.inflated.sub-01.gii`` next to it;
        results buried in a sulcus only become visible on the inflated one, so
        being able to switch without restarting is worth the lookup.

        Only the surfaces worth switching to are offered, and only when the
        file really holds a mesh: ``lh.mc.*`` and ``lh.sqrtsulc.*`` sit next to
        them with the same naming and hold scalars, not geometry.
        """
        current = self.opts.mesh_left
        if not current:
            return []
        path = Path(current)
        parts = path.name.split('.')
        found: List[Tuple[str, str]] = []
        for index, part in enumerate(parts):
            if part not in SWITCHABLE_SURFACES:
                continue
            for mesh_type in SWITCHABLE_SURFACES:
                candidate = list(parts)
                candidate[index] = mesh_type
                sibling = path.with_name('.'.join(candidate))
                if sibling.exists() and is_gifti_mesh_file(str(sibling)):
                    found.append((mesh_type, str(sibling)))
            break
        return found

    def current_surface_type(self) -> Optional[str]:
        """Which of those the viewer is showing."""
        for part in Path(self.opts.mesh_left or '').name.split('.'):
            if part in SWITCHABLE_SURFACES:
                return part
        return None

    def switch_surface_type(self, path: str):
        """Show another surface of the same subject, keeping the overlay."""
        try:
            self._switch_mesh(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Surface",
                f"Cannot show {os.path.basename(str(path))}:\n{exc}")
            return
        self.opts.mesh_left = path
        self._update_pick_label()

    # ---------- clusters ----------
    def _default_cluster_threshold(self) -> float:
        """Where to cut the map when the table is opened.

        The clip window if one is set — that is the threshold already being
        looked at — otherwise p<0.05 for a -log10(p) map, and half the range
        for anything else.
        """
        clip = tuple(self.opts.clip or (0.0, -1.0))
        if clip[1] > clip[0]:
            return float(clip[1])
        if self._uses_logp_scale():
            return float(LOG10_P005)
        low, high = self.overlay_range
        return float(low + 0.5 * (high - low)) if high > low else 0.0

    def collect_clusters(self, threshold: float, min_area: float = 0.0) -> List[dict]:
        """Clusters of both hemispheres, largest peak first."""
        found: List[dict] = []
        for side, poly, scalars in ((0, self.poly_l, self.scal_l),
                                    (1, self.poly_r, self.scal_r)):
            if poly is None or scalars is None:
                continue
            values = vtk_to_numpy(scalars).astype(float)
            if values.size != poly.GetNumberOfPoints():
                continue
            shift = getattr(self, '_y_shift_r' if side == 1 else '_y_shift_l', 0.0)
            for cluster in find_surface_clusters(poly, values, threshold, min_area):
                x, y, z = poly.GetPoint(cluster['peak_vertex'])
                cluster['side'] = side
                cluster['hemi'] = 'rh' if side == 1 else 'lh'
                cluster['mm'] = (x, y - float(shift), z)
                cluster['region'] = self._atlas_region(side, cluster['peak_vertex'])
                found.append(cluster)
        found.sort(key=lambda c: abs(c['peak_value']), reverse=True)
        return found

    def go_to_cluster(self, cluster: dict):
        """Mark the peak of *cluster* on the surface and in a linked volume."""
        self._set_surface_cursor(cluster['mm'])
        self._broadcast_world_pick(cluster['mm'])

    def show_cluster_table(self):
        """Open (or raise) the table of suprathreshold clusters."""
        if self.scal_l is None and self.scal_r is None:
            QtWidgets.QMessageBox.information(
                self, "Clusters", "There is no overlay to find clusters in.")
            return None
        table = getattr(self, '_cluster_table', None)
        if table is None:
            table = ClusterTableDialog(self, parent=self)
            self._cluster_table = table
        else:
            table.refresh()
        table.show()
        table.raise_()
        return table

    def _update_status_message(self, controls_visible: bool):
        """Show a small hint about the controls shortcut in the window status bar."""
        sb = self.statusBar()
        shortcut_hint = "Ctrl/Cmd+D"
        if controls_visible:
            sb.showMessage(f"Controls visible — press {shortcut_hint} to hide")
        else:
            sb.showMessage(f"Controls hidden — press {shortcut_hint} to show")
        # Single application-wide shortcut should suffice

    def _handle_key(self, sym: Optional[str]):
        if not sym:
            return
        # Normalize special names
        # sym can be like 'Left', 'Right', or single letters
        camera: vtkCamera = self.ren.GetActiveCamera()
        shift = self.iren.GetShiftKey(); ctrl = self.iren.GetControlKey()
        def do_render(): self.ren.ResetCameraClippingRange(); self.rw.Render()
        s = str(sym)
        # Overlay navigation
        if s == 'Left':
            if len(self.overlay_list) > 1:
                self._prev_overlay(); return
            if (not self.opts.overlay) and len(self.mesh_list) > 1:
                self._prev_mesh(); return
            return
        if s == 'Right':
            if len(self.overlay_list) > 1:
                self._next_overlay(); return
            if (not self.opts.overlay) and len(self.mesh_list) > 1:
                self._next_mesh(); return
            return
        if s in ('q','Q'):
            # Gracefully close viewer and quit application
            try:
                # Stop interactor loop if running
                if hasattr(self, 'iren') and self.iren is not None:
                    try:
                        self.iren.TerminateApp()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                self.close()
            except Exception:
                pass
            try:
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    app.quit()
                else:
                    raise RuntimeError('No QApplication instance')
            except Exception:
                try:
                    import sys as _sys
                    _sys.exit(0)
                except Exception:
                    pass
            return
        # Camera/control keys (accept both upper/lower)
        if s in ('u','U'):
            camera.Elevation(180 if ctrl else (1.0 if shift else 45.0)); camera.OrthogonalizeViewUp(); do_render(); return
        if s in ('d','D'):
            camera.Elevation(-180 if ctrl else (-1.0 if shift else -45.0)); camera.OrthogonalizeViewUp(); do_render(); return
        if s in ('l','L'):
            camera.Azimuth(180 if ctrl else (1.0 if shift else 45.0)); camera.OrthogonalizeViewUp(); do_render(); return
        if s in ('r','R'):
            # Rotate right only; do not reset view. Keep small step with Shift, large with Ctrl.
            camera.Azimuth(180 if ctrl else (-1.0 if shift else -45.0)); camera.OrthogonalizeViewUp(); do_render(); return
        if s in ('o','O'):
            self.ren.ResetCamera(); camera.OrthogonalizeViewUp(); camera.Zoom(2.0); do_render(); return
        if s in ('b','B'):
            # Flip only the middle views (indices 4 and 5) for both background and overlay clones
            for idx in (4, 5):
                a_b = self._montage_bkg[idx] if hasattr(self, '_montage_bkg') else None
                a_o = self._montage_ov[idx] if hasattr(self, '_montage_ov') else None
                if a_b is not None:
                    a_b.RotateX(180)
                if a_o is not None:
                    a_o.RotateX(180)
            camera.OrthogonalizeViewUp(); self.rw.Render(); return
        if s in ('g','G'):
            name = Path(self.rw.GetWindowName() or 'screenshot').with_suffix('.png')
            self.save_png(str(name)); return
        if s in ('m','M'):
            self.go_to_peak(); return
        if s in ('plus', 'equal', 'KP_Add'):
            self.zoom_by(1.2); return
        if s in ('minus', 'KP_Subtract'):
            self.zoom_by(1.0 / 1.2); return
        if s in ('h','H'):
            self.show_shortcut_help(); return

    

    # --- Geometry normalization helper ---
    def _shift_y_to(self, poly: vtkPolyData, to_value: float = -100.0):
        """Shift mesh in Y so its minimum Y aligns with to_value.

        Keeps montage layout consistent across mesh switches by anchoring meshes
        to a common Y-origin. This mirrors the normalization used at startup.
        """
        if poly is None:
            return 0.0
        b = [0.0]*6
        poly.GetBounds(b)
        y_shift = to_value - b[2]
        if abs(y_shift) < 1e-9:
            return 0.0
        pts = poly.GetPoints()
        if pts is None:
            return 0.0
        n = pts.GetNumberOfPoints()
        for i in range(n):
            x, y, z = pts.GetPoint(i)
            pts.SetPoint(i, x, y + y_shift, z)
        poly.SetPoints(pts)
        return float(y_shift)

    # --- LUT helpers ---
    def _apply_discrete_to_overlay_lut(self, lut: vtkLookupTable):
        """Flatten the LUT into opts.discrete bands (no gaps, unlike the colorbar)."""
        apply_discrete(lut, int(getattr(self.opts, 'discrete', 0) or 0))

    def _apply_discrete_to_colorbar_lut(self, lut: vtkLookupTable):
        """Apply discrete bands with gaps to colorbar LUT only.
        
        Matches VTK C++ implementation: at each step boundary (i % steps == 0),
        insert a gap (background color with full opacity), otherwise preserve the
        existing color (which may be from clip graying or original colormap).
        This creates visual separation in the colorbar display.
        
        Note: The gap is placed at position i-2 (not i-1 or i-3) because VTK's
        scalar bar rendering likely samples/interpolates the LUT in a way that
        makes this specific offset visible. Empirically, i-2 works while other
        offsets don't produce visible gaps in the colorbar display.
        """
        steps = int(getattr(self.opts, 'discrete', 0) or 0)
        if steps <= 0:
            return
        steps = max(1, min(256, steps))
        
        # Determine background color for gaps (white if background is white, else black)
        bkg_white = getattr(self.opts, 'bkg_color', 0) == 1
        gap_color = (1.0, 1.0, 1.0, 1.0) if bkg_white else (0.0, 0.0, 0.0, 1.0)
        
        # Build a temp copy of the LUT colors before modification
        colors = []
        for i in range(256):
            colors.append(lut.GetTableValue(i))
        
        # Apply discrete with gaps at block boundaries
        # At each boundary (i % steps == 0), set gap at position i-2
        block_color = (0.5, 0.5, 0.5, 1.0)  # Initialize with gray
        for i in range(256):
            if i % steps == 0:
                # At boundary: get color for the new block from original
                block_color = colors[i]
                # Set gap at position i-2 (empirically determined to be visible)
                if i > 0:
                    lut.SetTableValue(i-1, *gap_color)
                if i > 1:
                    lut.SetTableValue(i-2, *gap_color)
                if i > 2:
                    lut.SetTableValue(i-3, *gap_color)
            else:
                # Use the block's color
                r, g, b, a = block_color
                lut.SetTableValue(i, r, g, b, a)

    def _invert_lut(self, lut: vtkLookupTable):
        """Reverse the order of colors in a LUT in-place."""
        try:
            invert_lut(lut)
        except Exception:
            pass

    def _rebuild_overlay_luts(self):
        """Rebuild both overlay LUTs from colormap/opacity, honoring inverse and discrete.

        Any clip transparency previously baked into the LUT alpha is dropped;
        call :meth:`_apply_clip_to_overlay_luts` afterwards to re-apply it.
        """
        self.lut_overlay_l = build_overlay_lut(
            self.opts.colormap, self.opts.opacity,
            inverse=self.opts.inverse, discrete=self.opts.discrete)
        self.lut_overlay_r = build_overlay_lut(
            self.opts.colormap, self.opts.opacity,
            inverse=self.opts.inverse, discrete=self.opts.discrete)

    @staticmethod
    def _clipped_lut_indices(n: int, smin: float, smax: float, c0: float, c1: float) -> List[int]:
        """LUT indices covered by the clip window (see colormaps module)."""
        return clipped_lut_indices(n, smin, smax, c0, c1)

    def _apply_clip_to_overlay_luts(self):
        """Make values inside the clip range transparent and gray in colorbar.

        - For overlay LUTs: set alpha=0 for indices mapping to scalars in (clip_min, clip_max).
        - For colorbar LUT: reuse same logic; the gray band is applied in _ensure_colorbar.
        """
        c0, c1 = self.opts.clip
        if not (c1 > c0):
            return
        # Determine scalar range used for mapping
        if self.overlay_range[1] > self.overlay_range[0]:
            smin, smax = float(self.overlay_range[0]), float(self.overlay_range[1])
        else:
            # fallback to data range from poly_l if available
            r = [0.0, 0.0]
            try:
                self.poly_l.GetScalarRange(r)
            except Exception:
                return
            if not (r[1] > r[0]):
                return
            smin, smax = float(r[0]), float(r[1])
        def apply(lut: vtkLookupTable):
            n = int(lut.GetNumberOfTableValues())
            for i in self._clipped_lut_indices(n, smin, smax, c0, c1):
                r, g, b, a = lut.GetTableValue(i)
                # Transparent in overlay
                lut.SetTableValue(i, r, g, b, 0.0)
        if self.lut_overlay_l is not None:
            apply(self.lut_overlay_l)
        if self.lut_overlay_r is not None:
            apply(self.lut_overlay_r)

    # --- Camera state helpers to preserve view across overlay/mesh changes ---
    def _capture_camera_state(self):
        cam = self.ren.GetActiveCamera()
        try:
            self._cam_state = {
                'position': cam.GetPosition(),
                'focal_point': cam.GetFocalPoint(),
                'view_up': cam.GetViewUp(),
                'clipping_range': cam.GetClippingRange(),
                'window_center': cam.GetWindowCenter(),
                'view_angle': cam.GetViewAngle(),
                'parallel_projection': bool(cam.GetParallelProjection()),
                'parallel_scale': cam.GetParallelScale(),
            }
        except Exception:
            self._cam_state = None

    def _apply_camera_state(self):
        if not getattr(self, '_cam_state', None):
            return
        cam = self.ren.GetActiveCamera()
        st = self._cam_state
        try:
            cam.SetPosition(*st['position'])
            cam.SetFocalPoint(*st['focal_point'])
            cam.SetViewUp(*st['view_up'])
            # Keep the same window center to avoid subtle panning shifts
            if 'window_center' in st and isinstance(st['window_center'], (tuple, list)):
                cam.SetWindowCenter(float(st['window_center'][0]), float(st['window_center'][1]))
            # Preserve projection mode and scale/view angle
            if st['parallel_projection']:
                cam.SetParallelProjection(True)
                cam.SetParallelScale(st['parallel_scale'])
            else:
                cam.SetParallelProjection(False)
                cam.SetViewAngle(st['view_angle'])
            # Let VTK manage clipping range for the new scene bounds to reduce jitter
        except Exception:
            pass
        # Ensure proper rendering (avoid ResetCameraClippingRange to prevent subtle shifts)
        self.rw.Render()

    def _pick_overlay(self):
        start_dir = (self.ctrl.overlay_combo.currentText().strip()
                     or str(Path(self.opts.mesh_left).parent))
        dlg = QtWidgets.QFileDialog(self, "Choose overlay(s)", start_dir)
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if dlg.exec():
            paths = dlg.selectedFiles()
            if not paths:
                return
            # If the user picked an SPM overlay, auto-discover siblings
            paths = self._expand_spm_overlays(paths)
            # Update overlay list with selected files
            self.overlay_list = list(paths)
            self.current_overlay_index = 0
            # Populate combo with all and select the first
            try:
                self.ctrl.overlay_combo.clear()
                for p in self.overlay_list:
                    self.ctrl.overlay_combo.addItem(p)
                self.ctrl.overlay_combo.setCurrentIndex(0)
            except Exception:
                pass
            # Load the first selection immediately
            first = self.overlay_list[0]
            self._set_overlay_from_path(first)

    def _expand_spm_overlays(self, paths: List[str]) -> List[str]:
        """If any selected path is an SPM analysis overlay, expand the list
        to include all sibling SPM overlays from the same directory.

        The initially selected files are placed first, followed by any
        additional discoveries not already in the list.

        Args:
            paths: Overlay paths chosen by the user.

        Returns:
            Expanded (and deduplicated) overlay list.
        """
        seen: set = set()
        result: List[str] = []
        for p in paths:
            rp = str(Path(p).resolve())
            if rp not in seen:
                seen.add(rp)
                result.append(p)
        dirs_checked: set = set()
        for p in list(paths):
            pp = Path(p)
            parent = str(pp.parent.resolve())
            if parent in dirs_checked:
                continue
            dirs_checked.add(parent)
            if is_spm_surface_overlay(p):
                siblings = discover_spm_overlays(str(pp.parent))
                for s in siblings:
                    rs = str(Path(s).resolve())
                    if rs not in seen:
                        seen.add(rs)
                        result.append(s)
        return result

    def _on_overlay_combo_changed(self, _idx: int):
        path = self.ctrl.overlay_combo.currentText().strip()
        if path:
            # Keep overlay navigation index consistent with combo selection
            try:
                if self.overlay_list and path in self.overlay_list:
                    self.current_overlay_index = self.overlay_list.index(path)
            except Exception:
                pass
            self._set_overlay_from_path(path)

    def _on_overlay_combo_edited(self):
        path = self.ctrl.overlay_combo.currentText().strip()
        try:
            if self.overlay_list and path in self.overlay_list:
                self.current_overlay_index = self.overlay_list.index(path)
        except Exception:
            pass
        self._set_overlay_from_path(path)

    def _set_overlay_from_path(self, new_overlay: str):
        """Load or clear overlay based on provided path, updating UI and actors.

        - If path is non-empty and different from current, switch meshes if needed and load overlay.
        - If empty and an overlay is present, clear overlay and detach colorbar.
        """
        if new_overlay and new_overlay != (self.opts.overlay or ""):
            self._capture_camera_state()
            self._maybe_switch_mesh_for_overlay(new_overlay)
            self._load_overlay(new_overlay)
            self._apply_camera_state()
            # Ensure fix scaling policy reflects current overlays
            self._enforce_fix_scaling_policy()
            # Update title to current overlay
            try:
                self.setWindowTitle(_title_from_path(new_overlay))
            except Exception:
                pass
            return
        if not new_overlay and self.opts.overlay:
            # Clear overlay and disable controls
            self.opts.overlay = None
            self.scal_l = None
            self.scal_r = None
            # Close histogram if open
            try:
                if getattr(self, '_hist_win', None) is not None:
                    self._hist_win.close()
                    self._hist_win = None
            except Exception:
                pass
            # Remove overlay actors
            for actor in (self.actor_ov_l, self.actor_ov_r):
                if actor:
                    self.ren.RemoveActor(actor)
            self.actor_ov_l = None
            self.actor_ov_r = None
            # Disable overlay controls
            if hasattr(self, 'ctrl'):
                self.ctrl.set_overlay_controls_enabled(False)
            # Detach colorbar if present
            self._detach_colorbar()
            # Enforce fix scaling policy for zero overlays
            self._enforce_fix_scaling_policy()
            # Revert window title to mesh name when overlay cleared
            try:
                self.setWindowTitle(self.opts.title or _title_from_path(self.opts.mesh_left))
            except Exception:
                pass
            self.rw.Render()

    def _next_overlay(self):
        """Switch to next overlay in the list."""
        if len(self.overlay_list) > 1:
            self._capture_camera_state()
            self.current_overlay_index = (self.current_overlay_index + 1) % len(self.overlay_list)
            self._maybe_switch_mesh_for_overlay(self.overlay_list[self.current_overlay_index])
            self._load_overlay(self.overlay_list[self.current_overlay_index])
            self._update_overlay_info()
            # Update control panel with current overlay range
            if hasattr(self, 'ctrl'):
                self.ctrl.range_min.setValue(float(self.overlay_range[0]))
                self.ctrl.range_max.setValue(float(self.overlay_range[1]))
            # Restore camera state
            self._apply_camera_state()

    def _prev_overlay(self):
        """Switch to previous overlay in the list."""
        if len(self.overlay_list) > 1:
            self._capture_camera_state()
            self.current_overlay_index = (self.current_overlay_index - 1) % len(self.overlay_list)
            self._maybe_switch_mesh_for_overlay(self.overlay_list[self.current_overlay_index])
            self._load_overlay(self.overlay_list[self.current_overlay_index])
            self._update_overlay_info()
            # Update control panel with current overlay range
            if hasattr(self, 'ctrl'):
                self.ctrl.range_min.setValue(float(self.overlay_range[0]))
                self.ctrl.range_max.setValue(float(self.overlay_range[1]))
            # Restore camera state
            self._apply_camera_state()

    def _update_overlay_info(self):
        """Update the overlay path display and window title."""
        if self.overlay_list:
            current_overlay = self.overlay_list[self.current_overlay_index]
            try:
                if self.ctrl.overlay_combo.findText(current_overlay) < 0:
                    self.ctrl.overlay_combo.addItem(current_overlay)
                self.ctrl.overlay_combo.setCurrentText(current_overlay)
            except Exception:
                pass
            # Update window title to show current overlay
            overlay_name = _title_from_path(current_overlay)
            if len(self.overlay_list) > 1:
                self.setWindowTitle(f"{overlay_name} ({self.current_overlay_index + 1}/{len(self.overlay_list)})")
            else:
                # If only one overlay chosen, do not show index numbers
                self.setWindowTitle(overlay_name)
            # Also enforce fix scaling policy based on overlay count
            self._enforce_fix_scaling_policy()
        else:
            # Single overlay or none — keep combo text and update title if present
            path = self.opts.overlay or ""
            if path:
                try:
                    if self.ctrl.overlay_combo.findText(path) < 0:
                        self.ctrl.overlay_combo.addItem(path)
                    self.ctrl.overlay_combo.setCurrentText(path)
                except Exception:
                    pass
                self.setWindowTitle(_title_from_path(path))

    # --- Mesh navigation (when no overlay) ---
    def _switch_mesh(self, new_mesh_path: str):
        """Switch the underlying mesh to a new file path and update actors.

        Keeps the Y-origin normalization and the direction the scene is viewed
        from, but re-frames it for the new surface.  Overlay actors are
        preserved as-is (this entrypoint is used when there is no overlay).
        """
        if not new_mesh_path:
            return
        p = Path(new_mesh_path)
        if not p.exists():
            return
        # Update index to the matching entry if present
        try:
            if hasattr(self, 'mesh_list') and self.mesh_list:
                self.current_mesh_index = self.mesh_list.index(str(p))
        except Exception:
            pass
        # Capture camera before changing geometry
        self._capture_camera_state()
        # Load left mesh plus its opposite hemisphere (sibling file or, for
        # combined mesh.* surfaces, the second half of the same file)
        poly_l, poly_r = read_mesh_pair(str(p))
        self._set_meshes(poly_l, poly_r)
        # Update stored mesh_left
        try:
            self.opts.mesh_left = str(p)
        except Exception:
            pass
        # Update window title to mesh name if no overlay is active
        if not self.opts.overlay:
            try:
                self.setWindowTitle(self._mesh_title(new_mesh_path))
            except Exception:
                pass
        # The shading belongs to the surface now shown: its curvature has to be
        # recomputed and put back through the grey mapping
        if getattr(self, 'actor_bkg_l', None) is not None:
            self._apply_underlay()
        if self.show_borders:
            self._build_border_actors()
            self._build_montage()

        # Surfaces differ in size and position, so the view is framed for the
        # new one instead of keeping the framing of the previous surface
        self._fit_camera()
        self.rw.Render()

    def _next_mesh(self):
        if not getattr(self, 'mesh_list', None) or len(self.mesh_list) <= 1:
            return
        self.current_mesh_index = (self.current_mesh_index + 1) % len(self.mesh_list)
        self._switch_mesh(self.mesh_list[self.current_mesh_index])

    def _prev_mesh(self):
        if not getattr(self, 'mesh_list', None) or len(self.mesh_list) <= 1:
            return
        self.current_mesh_index = (self.current_mesh_index - 1) % len(self.mesh_list)
        self._switch_mesh(self.mesh_list[self.current_mesh_index])

    def _enforce_fix_scaling_policy(self):
        """Disable fix scaling when only one overlay is available.

        - If len(overlay_list) <= 1: uncheck and disable the checkbox, clear fixed range, and force opts.fix_scaling=False.
        - If multiple overlays: enable the checkbox (only when overlay controls are enabled).
        """
        multiple = len(self.overlay_list) > 1
        has_overlay = (self.overlay_list or self.opts.overlay) and (getattr(self, 'scal_l', None) is not None)
        if hasattr(self, 'ctrl'):
            # Enable only when multiple overlays and overlay controls are enabled
            self.ctrl.cb_fix_scaling.setEnabled(multiple and has_overlay)
            if not multiple:
                # Uncheck visually without emitting signals
                try:
                    self.ctrl.cb_fix_scaling.blockSignals(True)
                    self.ctrl.cb_fix_scaling.setChecked(False)
                finally:
                    self.ctrl.cb_fix_scaling.blockSignals(False)
        if not multiple:
            self.opts.fix_scaling = False
            self.fixed_overlay_range = None

    def _find_mesh_for_overlay(self, overlay_path: str) -> Optional[Path]:
        """Locate the central/midthickness mesh that corresponds to an overlay.

        Strategy:
          0) SPM analysis overlays use the template LH mesh.
          0b) Combined-hemisphere 'mesh.' overlays: use convert_filename_to_mesh.
          1) Use convert_filename_to_mesh heuristic.
          2) Search overlay directory for likely meshes (central/midthickness) with matching hemi tokens.
        """
        ov_path = Path(overlay_path)

        # SPM surface analysis overlays → template LH mesh
        if is_spm_surface_overlay(str(ov_path)):
            template_dir = _get_template_surface_dir()
            lh_template = template_dir / 'lh.central.freesurfer.gii'
            if lh_template.exists():
                return lh_template

        # Combined-hemisphere 'mesh.' overlays
        if _parse_mesh_combined_overlay(str(ov_path)) is not None:
            try:
                cand_str = convert_filename_to_mesh(str(ov_path))
                cand = Path(cand_str)
                if cand.exists() and is_gifti_mesh_file(str(cand)):
                    return cand
            except Exception:
                pass

        ov_dir = ov_path.parent
        ov_name = ov_path.name.lower()
        hemi = None
        if 'lh.' in ov_name or '_hemi-l_' in ov_name or 'left' in ov_name:
            hemi = 'lh'
        elif 'rh.' in ov_name or '_hemi-r_' in ov_name or 'right' in ov_name:
            hemi = 'rh'

        def _is_mesh(path: Path) -> bool:
            try:
                return path.exists() and path.suffix.lower() == '.gii' and is_gifti_mesh_file(str(path))
            except Exception:
                return False

        # Step 1: convert_filename_to_mesh
        try:
            cand = Path(convert_filename_to_mesh(str(ov_path)))
        except Exception:
            cand = None
        if cand:
            if not cand.is_absolute():
                cand = ov_dir / cand
            if _is_mesh(cand):
                return cand

        # Step 1b: explicit non-BIDS dot-token replacement for lh/rh thickness/pbt
        try:
            if hemi and not detect_naming_scheme(ov_path.name):
                parts = ov_path.name.split('.')
                parts_lower = [p.lower() for p in parts]
                replaced = False
                for i, p in enumerate(parts_lower):
                    if p in ('thickness', 'pbt'):
                        parts[i] = 'central'
                        replaced = True
                        break
                if replaced:
                    direct = '.'.join(parts)
                    if not direct.lower().endswith('.gii'):
                        direct = f"{direct}.gii"
                    cand2 = ov_dir / direct
                    if _is_mesh(cand2):
                        return cand2
        except Exception:
            pass

        # Step 2: glob for meshes near the overlay
        patterns = []
        if hemi:
            patterns.append(f"{hemi}.central*.gii")
            patterns.append(f"*{hemi}*midthickness*.surf.gii")
            patterns.append(f"*{hemi}*midthickness*.gii")
            patterns.append(f"*{hemi}*central*.gii")
        else:
            patterns.extend([
                "*.central*.gii",
                "*midthickness*.surf.gii",
                "*midthickness*.gii",
            ])
        candidates: List[Path] = []
        if ov_dir.exists():
            for pat in patterns:
                candidates.extend(ov_dir.glob(pat))
        # Deduplicate while preserving order
        seen = set(); uniq: List[Path] = []
        for c in candidates:
            if c not in seen:
                seen.add(c); uniq.append(c)
        candidates = uniq
        if not candidates:
            return None

        # Token-based scoring: prefer central/midthickness and higher token overlap
        stem_tokens = [t for t in re.split(r"[._-]+", ov_path.stem.lower()) if t]
        def score(path: Path) -> Tuple[int, int, int]:
            name_tokens = [t for t in re.split(r"[._-]+", path.stem.lower()) if t]
            common = len(set(stem_tokens) & set(name_tokens))
            is_central = int('central' in path.stem.lower())
            is_mid = int('midthickness' in path.stem.lower())
            return (common, is_central + is_mid, -len(name_tokens))

        best = max(candidates, key=score)
        return best if _is_mesh(best) else None

    def _mesh_fits_scalars(self, n_scal: int) -> bool:
        """True when the loaded mesh(es) have exactly *n_scal* vertices in total."""
        if self.poly_l is None or n_scal <= 0:
            return False
        n_l = self.poly_l.GetNumberOfPoints()
        n_r = self.poly_r.GetNumberOfPoints() if self.poly_r is not None else 0
        return n_scal in (n_l, n_l + n_r)

    def _use_embedded_mesh(self, ov_path: Path, n_scal: int) -> bool:
        """Display the geometry stored inside an overlay ``.gii``, if it fits.

        CAT12 writes combined-hemisphere overlays (``mesh.thickness.*``) and
        statistic results with the surface included in the same file, which is
        the most reliable source: no filename guessing involved.

        Returns:
            True when the embedded mesh was adopted.
        """
        if ov_path.suffix.lower() != '.gii':
            return False
        try:
            poly = read_gifti_mesh(str(ov_path))
        except Exception:
            return False
        if poly is None or poly.GetNumberOfPoints() != n_scal:
            return False
        # Split when the file holds both hemispheres back to back
        pair = split_hemispheres(poly, str(ov_path))
        poly_l, poly_r = pair if pair is not None else (poly, None)
        self._set_meshes(poly_l, poly_r)
        return True

    def _ensure_mesh_for_scalars(self, ov_path: Path, n_scal: int):
        """Make the displayed surface match the number of overlay values.

        Overlay files carry no reference to their surface, so it is normally
        derived from the filename.  That heuristic fails for free-form names —
        most notably CAT12/SPM statistic results (``logP_*.gii``,
        ``TFCE_*.gii``) — and when overlays from several folders are passed in
        one call.  The value count is the one unambiguous clue, so when the current
        mesh does not fit, candidates are tried in order of reliability:

        1. geometry embedded in the overlay file itself,
        2. the mesh derived from the overlay name or found next to it,
        3. the shipped resampling template (4k/32k/164k) of matching size.
        """
        if n_scal <= 0 or self.poly_l is None:
            return
        if self._mesh_fits_scalars(n_scal):
            return

        if self._use_embedded_mesh(ov_path, n_scal):
            # Remember that this overlay brings its own geometry, so navigating
            # back to it does not switch to a guessed mesh first.
            self._remember_mesh_for_overlay(ov_path, '')
            return

        cand = self._find_mesh_for_overlay(str(ov_path))
        if cand is not None and n_scal in _mesh_point_capacity(cand):
            self._switch_mesh(str(cand))
            if self._mesh_fits_scalars(n_scal):
                self._remember_mesh_for_overlay(ov_path, self.opts.mesh_left)
                return

        tpl = _template_mesh_for_points(n_scal, hemisphere_of(ov_path) or 'lh')
        if tpl is not None:
            self._switch_mesh(str(tpl))
            if self._mesh_fits_scalars(n_scal):
                self._remember_mesh_for_overlay(ov_path, self.opts.mesh_left)
        if not self._mesh_fits_scalars(n_scal):
            print(
                f"Warning: no surface with {n_scal} vertices found for "
                f"{ov_path.name}; the overlay may not be displayed correctly.",
                file=sys.stderr,
            )

    @staticmethod
    def _overlay_cache_key(ov_path: Path) -> str:
        try:
            return str(ov_path.expanduser().resolve())
        except Exception:
            return str(ov_path)

    def _remember_mesh_for_overlay(self, ov_path: Path, mesh: str):
        """Record which mesh belongs to an overlay ('' = geometry is in the file)."""
        self._overlay_mesh_cache[self._overlay_cache_key(ov_path)] = mesh or ''

    def _maybe_switch_mesh_for_overlay(self, overlay_path: str):
        """Switch to the mesh that belongs to *overlay_path*, if it is another one.

        Overlays selected from different folders (subjects, smoothing levels,
        statistic directories) usually sit next to their own surface, so the
        mesh is re-resolved on every switch.  Results are cached per overlay to
        keep the pairing stable when cycling back and forth: without the cache
        an ambiguous directory could otherwise "stick" to the mesh resolved for
        a previous overlay.
        """
        ov_path = Path(overlay_path)
        overlay_key = self._overlay_cache_key(ov_path)

        if overlay_key in self._overlay_mesh_cache:
            cached_mesh = self._overlay_mesh_cache[overlay_key]
            if not cached_mesh:
                # The overlay carries its own geometry; _load_overlay installs it
                return
            new_mesh_path = Path(cached_mesh)
        else:
            new_mesh_path = self._find_mesh_for_overlay(str(ov_path))
            if new_mesh_path is None:
                # Nothing conclusive from the filename; _load_overlay still
                # reconciles the mesh via the value count.
                return
            try:
                self._overlay_mesh_cache[overlay_key] = str(new_mesh_path.resolve())
            except Exception:
                self._overlay_mesh_cache[overlay_key] = str(new_mesh_path)

        # If the target mesh is the file already shown, do nothing
        if getattr(self.opts, 'mesh_left', None):
            try:
                if new_mesh_path.resolve() == Path(self.opts.mesh_left).resolve():
                    return
            except Exception:
                pass
        self._switch_mesh(str(new_mesh_path))

    def _apply_inverse(self):
        """Flip colormap without changing data or scalar ranges."""
        # Rebuild LUTs to ensure consistent base, then invert
        self._rebuild_overlay_luts()
        # Rebuilding drops the clip alpha, so re-apply the current clip window
        self._apply_clip_to_overlay_luts()
        if self.actor_ov_l is not None:
            self.actor_ov_l.GetMapper().SetLookupTable(self.lut_overlay_l)
        if self.actor_ov_r is not None:
            self.actor_ov_r.GetMapper().SetLookupTable(self.lut_overlay_r)
        if self.opts.colorbar:
            self._ensure_colorbar()

    def _uses_logp_scale(self) -> bool:
        """True when the current overlay should be labelled with p-values."""
        return bool(getattr(self.opts, 'log', False)) or is_logp_overlay(self.opts.overlay)

    def _apply_logp_labels(self, sb: vtkScalarBarActor, lut_cb: vtkLookupTable):
        """Label the colorbar with p-values for -log10(p) overlays.

        The ticks are attached as LUT annotations, so VTK places them at their
        value position instead of at evenly spaced steps.  For every other
        overlay the plain numeric tick labels are restored.
        """
        ticks = []
        if self._uses_logp_scale():
            ticks = logp_colorbar_ticks(
                self.overlay_range[0], self.overlay_range[1], self.opts.clip
            )
        try:
            lut_cb.ResetAnnotations()
            if not ticks:
                sb.SetDrawAnnotations(False)
                sb.SetDrawTickLabels(True)
                return
            # Dense bars need the short exponential form to stay readable
            compact = len(ticks) > 5
            for value in ticks:
                lut_cb.SetAnnotation(
                    vtkVariant(float(value)), format_p_value_label(value, compact)
                )
            sb.SetDrawAnnotations(True)
            sb.SetDrawTickLabels(False)
            sb.SetFixedAnnotationLeaderLineColor(True)
            tp = sb.GetLabelTextProperty()
            atp = sb.GetAnnotationTextProperty()
            atp.SetFontSize(tp.GetFontSize())
            atp.SetColor(tp.GetColor())
            atp.SetItalic(False)
        except Exception:
            pass

    def _ensure_colorbar(self):
        """Create the scalar bar if needed and update its properties.

        Does not force visibility; callers control SetVisibility().
        """
        # Ensure attribute exists for first-time calls during initialization
        if not hasattr(self, 'scalar_bar'):
            self.scalar_bar = None
        if self.scalar_bar is not None:
            # Simple continuous colorbar LUT based on current colormap/opacity and range
            lut_cb = get_lookup_table(self.opts.colormap, self.opts.opacity)
            if self.opts.inverse:
                self._invert_lut(lut_cb)
            if self.overlay_range[1] > self.overlay_range[0]:
                lut_cb.SetTableRange(self.overlay_range)
            # Gray out clip span on the colorbar (keep alpha opaque so the bar shows gray)
            c0, c1 = self.opts.clip
            if c1 > c0 and self.overlay_range[1] > self.overlay_range[0]:
                smin, smax = float(self.overlay_range[0]), float(self.overlay_range[1])
                n = int(lut_cb.GetNumberOfTableValues())
                for i in self._clipped_lut_indices(n, smin, smax, c0, c1):
                    r, g, b, a = lut_cb.GetTableValue(i)
                    gray = 0.5
                    lut_cb.SetTableValue(i, gray, gray, gray, a)
            # Apply discrete bands WITH GAPS to colorbar LUT AFTER range and clip
            try:
                steps = int(getattr(self.opts, 'discrete', 0) or 0)
            except Exception:
                steps = 0
            if steps > 0:
                self._apply_discrete_to_colorbar_lut(lut_cb)
            self.scalar_bar.SetLookupTable(lut_cb)

            # Update title according to title_mode (only if colorbar is enabled)
            title_mode = self.opts.title_mode
            if title_mode == 'none':
                self.scalar_bar.SetTitle(" ")
            elif title_mode == 'stats' or (self.opts.stats and self._overlay_scalars() is not None):
                values = self._overlay_scalars()
                if values is not None:
                    info = f"Mean={get_mean(values):.3f} Median={get_median(values):.3f} SD={get_std(values):.3f}"
                    self.scalar_bar.SetTitle(info)
                else:
                    self.scalar_bar.SetTitle("")
            else:
                self.scalar_bar.SetTitle(Path(self.opts.overlay or self.opts.mesh_left).name)

            # Ensure fonts are normalized even when title is empty
            base_fs = self.opts.fontsize if self.opts.fontsize else 12
            tp = self.scalar_bar.GetLabelTextProperty(); tp.SetFontSize(base_fs)
            tp2 = self.scalar_bar.GetTitleTextProperty(); tp2.SetFontSize(base_fs)
            try:
                self.scalar_bar.SetAnnotationTextScaling(False)
            except Exception:
                pass
            # Statistic overlays (logP_*) are labelled with p-values
            self._apply_logp_labels(self.scalar_bar, lut_cb)
            # Mark actor modified; do not force visibility here
            try:
                self.scalar_bar.Modified()
            except Exception:
                pass
            return

        # Create a new scalar bar actor
        lut_cb = get_lookup_table(self.opts.colormap, self.opts.opacity)
        if self.opts.inverse:
            self._invert_lut(lut_cb)
        if self.overlay_range[1] > self.overlay_range[0]:
            lut_cb.SetTableRange(self.overlay_range)
        c0, c1 = self.opts.clip
        if c1 > c0 and self.overlay_range[1] > self.overlay_range[0]:
            smin, smax = float(self.overlay_range[0]), float(self.overlay_range[1])
            n = int(lut_cb.GetNumberOfTableValues())
            for i in self._clipped_lut_indices(n, smin, smax, c0, c1):
                r, g, b, a = lut_cb.GetTableValue(i)
                gray = 0.5
                lut_cb.SetTableValue(i, gray, gray, gray, a)
        # Apply discrete bands WITH GAPS to colorbar LUT AFTER range and clip
        try:
            steps = int(getattr(self.opts, 'discrete', 0) or 0)
        except Exception:
            steps = 0
        if steps > 0:
            self._apply_discrete_to_colorbar_lut(lut_cb)

        sb = vtkScalarBarActor()
        sb.SetOrientationToHorizontal()
        sb.SetLookupTable(lut_cb)
        sb.SetWidth(0.3)
        sb.SetHeight(0.05)
        sb.SetPosition(0.35, 0.05)

        base_fs = self.opts.fontsize if self.opts.fontsize else 12
        tp = sb.GetLabelTextProperty(); tp.SetFontSize(base_fs)
        tp2 = sb.GetTitleTextProperty(); tp2.SetFontSize(base_fs)

        title_mode = self.opts.title_mode
        if title_mode == 'none':
            sb.SetTitle(" ")
        elif title_mode == 'stats' or (self.opts.stats and self._overlay_scalars() is not None):
            values = self._overlay_scalars()
            if values is not None:
                info = f"Mean={get_mean(values):.3f} Median={get_median(values):.3f} SD={get_std(values):.3f}"
                sb.SetTitle(info)
            else:
                sb.SetTitle("")
        else:
            sb.SetTitle(Path(self.opts.overlay or self.opts.mesh_left).name)

        try:
            sb.SetAnnotationTextScaling(False)
        except Exception:
            pass

        # Statistic overlays (logP_*) are labelled with p-values
        self._apply_logp_labels(sb, lut_cb)

        # Store; caller manages attaching/detaching
        self.scalar_bar = sb
        self._scalar_bar_added = False

    def _attach_colorbar(self):
        """Attach scalar bar to the appropriate renderer if not already attached."""
        if getattr(self, 'scalar_bar', None) is None:
            return
        if getattr(self, '_scalar_bar_added', False):
            return
        try:
            self.ren_ui.AddViewProp(self.scalar_bar)
        except Exception:
            self.ren.AddViewProp(self.scalar_bar)
        self._scalar_bar_added = True
        try:
            self.scalar_bar.Modified()
        except Exception:
            pass

    def _detach_colorbar(self):
        """Detach scalar bar from renderer if attached."""
        if getattr(self, 'scalar_bar', None) is None:
            return
        if not getattr(self, '_scalar_bar_added', False):
            return
        try:
            self.ren_ui.RemoveViewProp(self.scalar_bar)
        except Exception:
            try:
                self.ren.RemoveViewProp(self.scalar_bar)
            except Exception:
                pass
        self._scalar_bar_added = False
    # _remove_colorbar removed; use _detach_colorbar instead

    def _load_overlay(self, overlay_path: str):
        # Capture camera before modifying actors/ranges
        self._capture_camera_state()
        self.opts.overlay = overlay_path
        try:
            ov_path = Path(overlay_path)
            scal_l = read_scalars(str(ov_path))
            scal_r = None
            n_scal = scal_l.GetNumberOfTuples() if scal_l is not None else 0

            # An overlay file does not say which surface it belongs to, so
            # make sure the displayed mesh has room for exactly these values.
            self._ensure_mesh_for_scalars(ov_path, n_scal)

            # Prefer the overlay of the other hemisphere sitting next to the
            # selected file — an rh.* file finds its lh.* partner just as well
            other_path = _hemi_counterpart(ov_path)
            if other_path is not None and other_path.exists() and self.poly_r is not None:
                scal_l, scal_r = order_by_hemisphere(
                    ov_path, scal_l, read_scalars(str(other_path)))
            # Or: a single overlay holds LH and RH values back to back
            elif self.poly_r is not None and scal_l is not None and n_scal == (
                self.poly_l.GetNumberOfPoints() + self.poly_r.GetNumberOfPoints()
            ):
                scal_l, scal_r = _split_scalars(
                    scal_l,
                    self.poly_l.GetNumberOfPoints(),
                    self.poly_r.GetNumberOfPoints(),
                )
            elif (hemisphere_of(ov_path) == 'rh' and self.poly_r is not None
                  and scal_l is not None
                  and n_scal == self.poly_r.GetNumberOfPoints()):
                # A right-hemisphere overlay without its left partner: show it
                # on the right surface rather than on the left one
                scal_l, scal_r = None, scal_l
        except Exception as e:
            # If loading fails, clear the overlay and disable controls
            print(f"Failed to load overlay: {e}")
            self.opts.overlay = None
            if hasattr(self, 'ctrl'):
                self.ctrl.set_overlay_controls_enabled(False)
            return
        # Do not invert scalars; inversion is handled by flipping LUTs
        # Clip is rendered via LUT alpha; do not mutate scalar arrays
        # attach
        self.scal_l = scal_l; self.scal_r = scal_r
        if scal_l is not None: self.poly_l.GetPointData().SetScalars(scal_l)
        if scal_r is not None and self.poly_r is not None: self.poly_r.GetPointData().SetScalars(scal_r)
        # Predefined ranges for recognized overlays (thickness, pbt)
        kind = detect_overlay_kind(overlay_path)
        if kind in ('thickness', 'pbt') and not self.opts.fix_scaling:
            # Apply requested defaults: overlay 0.5..5; clip 0..0; bkg -1..1,
            # but never override values the user gave on the command line.
            if not getattr(self, '_user_set_range', False):
                self.overlay_range = [0.5, 5.0]
            if not getattr(self, '_user_set_clip', False):
                self.opts.clip = (0.0, 0.0)
            if not getattr(self, '_user_set_range_bkg', False):
                self.range_bkg = [-1.0, 1.0]
        elif self.opts.fix_scaling and self.fixed_overlay_range is not None:
            # Use the fixed range across all overlays
            self.overlay_range = list(self.fixed_overlay_range)
        elif getattr(self, '_user_set_range', False):
            # User explicitly provided a range via CLI; keep it
            pass
        else:
            # Auto-scale: recompute the range from the data, the way
            # cat_surf_results does it
            scaled = self._auto_overlay_range()
            if scaled is not None:
                self.overlay_range = list(scaled)
        # The new overlay may cover a hemisphere that had no actor yet
        if self._ensure_hemisphere_actors() and hasattr(self, '_montage_ov'):
            self._build_montage()
        for actor in (self.actor_ov_l, self.actor_ov_r):
            # LUTs are (re)assigned below, after the clip window has been applied
            if actor and self.overlay_range[1] > self.overlay_range[0]:
                actor.GetMapper().SetScalarRange(self.overlay_range)
        # Apply background range to background actors when set (if already created)
        if self.range_bkg[1] > self.range_bkg[0]:
            for actor in (getattr(self, 'actor_bkg_l', None), getattr(self, 'actor_bkg_r', None)):
                if actor:
                    actor.GetMapper().SetScalarRange(self.range_bkg)
        if self.opts.colorbar: self._ensure_colorbar()
        # Rebuild the overlay LUTs from scratch before re-applying clip: the clip
        # window is baked into the LUT alpha, so a stale LUT would keep hiding
        # values of the previous overlay/clip setting.
        self._rebuild_overlay_luts()
        self._apply_clip_to_overlay_luts()
        if self.actor_ov_l is not None:
            self.actor_ov_l.GetMapper().SetLookupTable(self.lut_overlay_l)
        if self.actor_ov_r is not None:
            self.actor_ov_r.GetMapper().SetLookupTable(self.lut_overlay_r)
        # Update histogram window if visible
        self._update_histogram_window()
        
        # Enable overlay controls since we now have an overlay loaded
        if hasattr(self, 'ctrl'):
            self.ctrl.set_overlay_controls_enabled(True)
            # Update spin boxes to current overlay range
            if self.overlay_range[1] > self.overlay_range[0]:
                self.ctrl.range_min.setValue(float(self.overlay_range[0]))
                self.ctrl.range_max.setValue(float(self.overlay_range[1]))
            # If we applied predefined defaults, reflect them in the UI
            if kind in ('thickness', 'pbt') and not self.opts.fix_scaling:
                if not getattr(self, '_user_set_clip', False):
                    self.ctrl.clip_min.setValue(0.0); self.ctrl.clip_max.setValue(0.0)
                if not getattr(self, '_user_set_range_bkg', False):
                    self.ctrl.bkg_min.setValue(-1.0); self.ctrl.bkg_max.setValue(1.0)
            # The p-value thresholds only make sense for -log10(p) overlays
            self.ctrl.set_threshold_visible(self._uses_logp_scale())
            self.ctrl.set_threshold_from_clip(self.opts.clip)
            # Update slider bounds from data and align to spins
            self._update_slider_bounds()
            # Enforce fix scaling enable/disable based on overlay count and availability
            self._enforce_fix_scaling_policy()
            # Auto-show colorbar if user requested it initially
            try:
                if getattr(self, '_colorbar_intent', False) and not getattr(self, '_scalar_bar_added', False):
                    self._ensure_colorbar(); self._attach_colorbar(); self.opts.colorbar = True
                    self.ctrl.cb_colorbar.blockSignals(True)
                    self.ctrl.cb_colorbar.setChecked(True)
                    self.ctrl.cb_colorbar.blockSignals(False)
                    self.ctrl.title_mode.setEnabled(True)
            except Exception:
                pass
        
        # Restore camera and render
        self._apply_camera_state()
        self.rw.Render()

    def _toggle_histogram(self, checked: bool):
        """Show/hide histogram window for current overlay scalars."""
        has_overlay = getattr(self, 'scal_l', None) is not None or getattr(self, 'scal_r', None) is not None
        if checked and has_overlay:
            if self._hist_win is None:
                try:
                    self._hist_win = HistogramWindow(parent=self)
                except Exception:
                    self._hist_win = None
            if self._hist_win is not None:
                self._update_histogram_window()
                try:
                    self._hist_win.show()
                except Exception:
                    pass
        else:
            # Hide/close
            if self._hist_win is not None:
                try:
                    self._hist_win.close()
                except Exception:
                    pass
                self._hist_win = None

    def _overlay_scalars(self):
        """The overlay values, from whichever hemisphere carries them."""
        return self.scal_l if self.scal_l is not None else getattr(self, 'scal_r', None)

    def _update_histogram_window(self):
        """If histogram window is open, refresh with current overlay scalars."""
        hw = getattr(self, '_hist_win', None)
        if hw is None:
            return
        try:
            vals = []
            if getattr(self, 'scal_l', None) is not None:
                try:
                    arr = vtk_to_numpy(self.scal_l).astype(float)
                    vals.append(arr)
                except Exception:
                    pass
            if getattr(self, 'scal_r', None) is not None:
                try:
                    arr = vtk_to_numpy(self.scal_r).astype(float)
                    vals.append(arr)
                except Exception:
                    pass
            if not vals:
                return
            data = np.concatenate(vals)
            # Filter non-finite
            data = data[np.isfinite(data)] if data.size else data
            # Determine range: prefer overlay_range if valid
            rng = None
            if (self.overlay_range[1] - 1.0) > (self.overlay_range[0] + 0.5):
                rng = (float(self.overlay_range[0]) + 0.5, float(self.overlay_range[1]) - 1.0)
            hw.set_data(data, rng)
        except Exception:
            pass

    def _update_slider_bounds(self):
        """Compute data-driven bounds and apply to control panel sliders.

        - Overlay/Clip sliders span the current overlay data range (from left hemi).
        - Background sliders span the current background data range.
        """
        if not hasattr(self, 'ctrl'):
            return
        # Overlay/Clip bounds from current poly_l scalars if present
        ov_bounds = (-1.0, 1.0)
        try:
            r = [0.0, 0.0]
            if self.poly_l is not None and self.poly_l.GetPointData().GetScalars() is not None:
                self.poly_l.GetScalarRange(r)
                ov_bounds = (float(r[0]), float(r[1]))
                if not (ov_bounds[1] > ov_bounds[0]):
                    ov_bounds = (-1.0, 1.0)
        except Exception:
            pass
        self.ctrl.set_overlay_bounds(*ov_bounds)
        self.ctrl.set_clip_bounds(*ov_bounds)

        # Background bounds from curvature/bkg output
        bkg_bounds = (-1.0, 1.0)
        try:
            r2 = [0.0, 0.0]
            if hasattr(self, 'curv_l_out') and self.curv_l_out is not None:
                self.curv_l_out.GetScalarRange(r2)
                bkg_bounds = (float(r2[0]), float(r2[1]))
                if not (bkg_bounds[1] > bkg_bounds[0]):
                    bkg_bounds = (-1.0, 1.0)
        except Exception:
            pass
        self.ctrl.set_bkg_bounds(*bkg_bounds)

    # -- Save PNG --
    def save_png(self, path: str):
        """Write the current view to a PNG file.

        The scene is re-rendered with buffer swapping disabled and captured
        from the back buffer.  Reading the front buffer (the VTK default)
        returns whatever the window server happens to hold, which on a
        freshly-shown window can be just the topmost layer — the colorbar —
        instead of the rendered surfaces.
        """
        swap_off = False
        try:
            self.rw.SwapBuffersOff()
            swap_off = True
        except Exception:
            pass
        try:
            self.rw.Render()
            w2i = vtkWindowToImageFilter()
            w2i.SetInput(self.rw)
            try:
                w2i.SetInputBufferTypeToRGB()
                w2i.ReadFrontBufferOff()
            except Exception:
                pass
            w2i.Modified()
            w2i.Update()
            writer = vtkPNGWriter(); writer.SetFileName(path)
            writer.SetInputConnection(w2i.GetOutputPort()); writer.Write()
        finally:
            if swap_off:
                try:
                    self.rw.SwapBuffersOn()
                except Exception:
                    pass
        print(f"Saved {path}")

    # -- Volume Integration --
    def _open_volume_dialog(self):
        start_dir = str(Path(self.opts.mesh_left).parent) if self.opts.mesh_left else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open 3D NIfTI volume", start_dir, "NIfTI (*.nii *.nii.gz);;All files (*)")
        if path:
            self._open_volume(path)

    def _open_volume(self, volume_path: str):
        try:
            win = VolumeViewerWindow(volume_path, parent=self,
                                     on_position_changed=self._on_volume_pick,
                                     surfaces=self._surface_outlines())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open volume:\n{e}")
            return
        # keep a reference to prevent garbage collection
        if not hasattr(self, '_volume_windows'):
            self._volume_windows = []
        self._volume_windows.append(win)
        # Track close to prune list
        def _on_close(_evt=None, w=win):
            try:
                if hasattr(self, '_volume_windows'):
                    self._volume_windows = [vw for vw in self._volume_windows if (vw is not None and vw is not w and vw.isVisible())]
            except Exception:
                pass
        try:
            win.destroyed.connect(lambda *_: _on_close())
        except Exception:
            pass
        win.show()

    def _surface_outlines(self) -> List[vtkPolyData]:
        """The displayed hemispheres, in the mm space of a volume.

        Drawn as outlines on the slices of the volume window.  The viewer
        normalizes the Y origin of every mesh it displays, which has to be
        undone first.  This is a snapshot taken when the window opens;
        switching the surface afterwards does not redraw the outlines.
        """
        out: List[vtkPolyData] = []
        hemis = ((self.poly_l, getattr(self, '_y_shift_l', 0.0)),
                 (self.poly_r, getattr(self, '_y_shift_r', 0.0)))
        for poly, y_shift in hemis:
            if poly is None or poly.GetNumberOfPoints() == 0:
                continue
            try:
                transform = vtkTransform()
                transform.Translate(0.0, -float(y_shift), 0.0)
                filt = vtkTransformPolyDataFilter()
                filt.SetInputData(poly)
                filt.SetTransform(transform)
                filt.Update()
                out.append(filt.GetOutput())
            except Exception:
                continue
        return out

    def _broadcast_world_pick(self, world_xyz: Tuple[float, float, float], exclude=None):
        """Send a picked world coordinate to all open volume windows.

        *exclude* skips the window a position came from, so a click in one
        slice viewer moves the others without echoing back into itself.
        """
        if not hasattr(self, '_volume_windows'):
            return
        # prune closed windows
        self._volume_windows = [w for w in self._volume_windows if w is not None and w.isVisible()]
        for w in list(self._volume_windows):
            try:
                if w is not None and w is not exclude:
                    w.set_world_position(world_xyz)
            except Exception:
                continue

    def _on_volume_pick(self, world_xyz: Tuple[float, float, float], source=None):
        """Handle a position picked in a slice viewer (volume -> surface)."""
        self._set_surface_cursor(world_xyz)
        self._broadcast_world_pick(world_xyz, exclude=source)

    def _clear_surface_cursor(self):
        """Remove the cursor markers from the montage."""
        self._cursor_vertex = None
        for actor in getattr(self, '_cursor_actors', None) or []:
            try:
                self.ren.RemoveActor(actor)
            except Exception:
                pass
        self._cursor_actors = []

    def _set_surface_cursor(self, world_xyz: Optional[Tuple[float, float, float]]):
        """Mark the surface vertex closest to a world (mm) position.

        The marker is what links the two windows visually: it is placed by a
        click on the surface as well as by a click in the volume viewer, and it
        is repeated in every montage view that shows the hemisphere it belongs
        to.

        Returns:
            The mm position of the marked vertex, or None when nothing was
            close enough (or no surface is loaded).
        """
        self._clear_surface_cursor()
        if world_xyz is None or self.poly_l is None:
            return None

        best = None  # (squared distance, side, vertex id, mesh coordinates)
        hemis = ((0, self.poly_l, getattr(self, '_y_shift_l', 0.0)),
                 (1, self.poly_r, getattr(self, '_y_shift_r', 0.0)))
        for side, poly, y_shift in hemis:
            if poly is None or poly.GetNumberOfPoints() == 0:
                continue
            points = vtk_to_numpy(poly.GetPoints().GetData())
            # Mesh coordinates are the mm position plus the Y normalization
            target = np.array([world_xyz[0], world_xyz[1] + float(y_shift), world_xyz[2]])
            d2 = ((points - target) ** 2).sum(axis=1)
            idx = int(np.argmin(d2))
            if best is None or d2[idx] < best[0]:
                best = (float(d2[idx]), side, idx, tuple(float(v) for v in points[idx]))
        if best is None:
            return None
        dist2, side, vertex_id, vertex = best
        if dist2 > self._CURSOR_MAX_DIST ** 2:
            # The point is nowhere near the surface (e.g. deep white matter);
            # showing a marker on the closest vertex would be misleading.
            return None
        # Which vertex was marked is what the readout, the atlas lookup and the
        # cluster table all work from
        self._cursor_vertex = (side, vertex_id)
        self._update_pick_label()

        sphere = vtkSphereSource()
        sphere.SetCenter(*vertex)
        sphere.SetRadius(self._CURSOR_RADIUS)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        sphere.Update()
        marker = sphere.GetOutput()

        # One marker per montage view showing this hemisphere; the clone's
        # matrix places it exactly like the surface it sits on.
        for i, clone in enumerate(getattr(self, '_montage_bkg', None) or []):
            if clone is None or self._montage_order[i] != side:
                continue
            mapper = vtkPolyDataMapper(); mapper.SetInputData(marker)
            actor = vtkActor(); actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 1.0, 0.0)
            actor.GetProperty().SetAmbient(0.6)
            matrix = vtkMatrix4x4(); matrix.DeepCopy(clone.GetMatrix())
            actor.SetUserMatrix(matrix)
            self.ren.AddActor(actor)
            self._cursor_actors.append(actor)
        try:
            self.rw.Render()
        except Exception:
            pass
        y_shift = self._y_shift_r if side == 1 else self._y_shift_l
        return (vertex[0], vertex[1] - float(y_shift), vertex[2])

    def _surface_click_to_mm(self, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        """Pick a point on the surface and convert to original mm coordinates (undo actor transforms and Y-shift)."""
        try:
            picker = vtkCellPicker(); picker.SetTolerance(0.0005)
            ok = picker.Pick(int(x), int(y), 0, self.ren)
            if not ok:
                return None
            actor = picker.GetActor()
            wx, wy, wz = picker.GetPickPosition()
            
            if actor is None:
                return (wx, wy, wz)
            # Invert actor transform to object coordinates
            try:
                M = actor.GetMatrix()
                Minv = vtkMatrix4x4(); vtkMatrix4x4.Invert(M, Minv)
                v = [wx, wy, wz, 1.0]
                obj = [0.0, 0.0, 0.0, 1.0]
                for i in range(4):
                    obj[i] = sum(Minv.GetElement(i, j) * v[j] for j in range(4))
                ox, oy, oz = obj[0], obj[1], obj[2]
            except Exception:
                ox, oy, oz = wx, wy, wz
            # Determine side from montage order (0=L, 1=R)
            side = 0
            try:
                idx = -1
                if hasattr(self, '_montage_bkg') and self._montage_bkg:
                    for i, a in enumerate(self._montage_bkg):
                        if a is actor:
                            idx = i; break
                if idx < 0 and hasattr(self, '_montage_ov') and self._montage_ov:
                    for i, a in enumerate(self._montage_ov):
                        if a is actor:
                            idx = i; break
                if idx >= 0:
                    order = [0,1,0,1,0,1]
                    side = order[idx] if idx < len(order) else 0
            except Exception:
                side = 0
            y_shift = self._y_shift_r if side == 1 else self._y_shift_l
            # Undo Y normalization to return mesh point in original mm space
            return (ox, oy - float(y_shift), oz)
        except Exception:
            return None

class ClusterTableDialog(QtWidgets.QDialog):
    """The suprathreshold clusters of an overlay, as a table.

    What a statistical map is finally reported as: where the peaks are, how
    big they are and — with an atlas selected — what they are called.  Picking
    a row marks that peak on the surface, so the table and the view stay
    together.
    """

    def __init__(self, viewer: "Viewer", parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Clusters")
        self.resize(760, 420)
        self.clusters: List[dict] = []

        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Threshold |value| >"))
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setDecimals(3)
        self.threshold.setRange(0.0, 1e6)
        self.threshold.setSingleStep(0.1)
        self.threshold.setValue(viewer._default_cluster_threshold())
        self.threshold.setKeyboardTracking(False)
        self.threshold.valueChanged.connect(lambda _v: self.refresh())
        row.addWidget(self.threshold)
        row.addWidget(QtWidgets.QLabel("   Smallest cluster (mm²)"))
        self.min_area = QtWidgets.QDoubleSpinBox()
        self.min_area.setDecimals(1)
        self.min_area.setRange(0.0, 1e6)
        self.min_area.setValue(10.0)
        self.min_area.setKeyboardTracking(False)
        self.min_area.valueChanged.connect(lambda _v: self.refresh())
        row.addWidget(self.min_area)
        row.addStretch(1)
        self.save_button = QtWidgets.QPushButton("Save CSV…")
        self.save_button.clicked.connect(self.save_csv)
        row.addWidget(self.save_button)
        layout.addLayout(row)

        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._row_selected)
        layout.addWidget(self.table, 1)

        self.summary = QtWidgets.QLabel("")
        layout.addWidget(self.summary)
        self.refresh()

    def columns(self) -> List[str]:
        names = ["hemi", "peak", "vertex", "x", "y", "z", "vertices", "area (mm²)"]
        if self.viewer._uses_logp_scale():
            names.insert(2, "p")
        if self.viewer._atlas:
            names.append(self.viewer._atlas['name'])
        return names

    def refresh(self):
        """Find the clusters again and fill the table."""
        self.clusters = self.viewer.collect_clusters(self.threshold.value(),
                                                     self.min_area.value())
        columns = self.columns()
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(self.clusters))
        for row, cluster in enumerate(self.clusters):
            for column, text in enumerate(self._cells(cluster)):
                item = QtWidgets.QTableWidgetItem()
                # Numbers have to sort as numbers, not as text
                try:
                    item.setData(Qt.ItemDataRole.DisplayRole, float(text))
                except (TypeError, ValueError):
                    item.setText(str(text))
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)
        area = sum(c['area'] for c in self.clusters)
        self.summary.setText(
            f"{len(self.clusters)} clusters, {area:.0f} mm² in total"
            if self.clusters else "No cluster above the threshold")

    def _cells(self, cluster: dict) -> List:
        cells = [cluster['hemi'], round(cluster['peak_value'], 4),
                 cluster['peak_vertex'],
                 round(cluster['mm'][0], 1), round(cluster['mm'][1], 1),
                 round(cluster['mm'][2], 1),
                 cluster['vertices'], round(cluster['area'], 1)]
        if self.viewer._uses_logp_scale():
            cells.insert(2, format_p_value_label(cluster['peak_value']))
        if self.viewer._atlas:
            cells.append(cluster.get('region') or '-')
        return cells

    def _row_selected(self):
        """Mark the peak of the selected cluster on the surface."""
        rows = {index.row() for index in self.table.selectedIndexes()}
        if len(rows) != 1:
            return
        vertex_item = self.table.item(rows.pop(), self.columns().index("vertex"))
        if vertex_item is None:
            return
        vertex = int(vertex_item.data(Qt.ItemDataRole.DisplayRole))
        for cluster in self.clusters:
            if cluster['peak_vertex'] == vertex:
                self.viewer.go_to_cluster(cluster)
                return

    def save_csv(self):
        """Write the table as it stands, for the paper or the supplement."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save cluster table", "clusters.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns())
            for cluster in self.clusters:
                writer.writerow(self._cells(cluster))


class HistogramCanvas(QtWidgets.QWidget):
    """Simple widget to draw a histogram of given data using QPainter."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = np.array([], dtype=float)
        self._range = None  # optional (min, max)
        self.setMinimumSize(400, 250)

    def set_data(self, data: np.ndarray, value_range: Optional[Tuple[float, float]] = None):
        try:
            self._data = np.asarray(data, dtype=float)
        except Exception:
            self._data = np.array([], dtype=float)
        self._range = value_range if (value_range and value_range[1] > value_range[0]) else None
        self.update()

    def paintEvent(self, event):
        if self.rect().isEmpty():
            return
        p = QPainter(self)
        if not p.isActive():
            return
        p.fillRect(self.rect(), QColor(30, 30, 30))
        rect = self.rect().adjusted(40, 10, -10, -30)
        # Border
        p.setPen(QPen(QColor(200, 200, 200), 1))
        p.drawRect(rect)
        if self._data.size == 0:
            p.drawText(rect, Qt.AlignCenter, "No data")
            p.end(); return
        # Build histogram
        data = self._data
        if self._range is not None:
            lo, hi = self._range
            data = data[(data >= lo) & (data <= hi)]
        if data.size == 0:
            p.drawText(rect, Qt.AlignCenter, "No data in range")
            p.end(); return
        bins = 64
        lo = np.nanmin(data)
        hi = np.nanmax(data)
        if self._range is not None:
            lo, hi = self._range
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            p.drawText(rect, Qt.AlignCenter, "Invalid range")
            p.end(); return
        hist, edges = np.histogram(data, bins=bins, range=(lo, hi))
        hmax = hist.max() if hist.size else 1
        if hmax <= 0:
            p.drawText(rect, Qt.AlignCenter, "Empty histogram")
            p.end(); return
        # Draw bars
        bw = rect.width() / bins
        for i, h in enumerate(hist):
            x = rect.left() + i * bw
            w = max(1.0, bw - 1.0)
            hpx = int(round((h / hmax) * rect.height()))
            y = rect.bottom() - hpx
            p.fillRect(int(x), int(y), int(w), int(hpx), QBrush(QColor(80, 170, 255)))
        # X-axis ticks (min/mid/max)
        p.setPen(QPen(QColor(220, 220, 220), 1))
        labels = [(lo, rect.left()), ((lo + hi) / 2.0, rect.left() + rect.width() / 2.0), (hi, rect.right())]
        for val, xpos in labels:
            s = f"{val:.3g}"
            p.drawText(int(xpos) - 20, rect.bottom() + 18, 40, 16, Qt.AlignCenter, s)
        # Y-axis label for max count
        p.drawText(rect.left() - 35, rect.top(), 30, 16, Qt.AlignRight | Qt.AlignVCenter, str(int(hmax)))
        p.end()


class HistogramWindow(QtWidgets.QMainWindow):
    """A small window displaying a histogram of current overlay scalars."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Overlay histogram")
        self.resize(520, 340)
        central = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(6, 6, 6, 6)
        self._canvas = HistogramCanvas(central)
        vbox.addWidget(self._canvas, 1)
        self.setCentralWidget(central)

    def set_data(self, data: np.ndarray, value_range: Optional[Tuple[float, float]] = None):
        self._canvas.set_data(data, value_range)

# ---- Entrypoint ----
def main(argv: Optional[List[str]] = None):
    if argv is None:
        argv = sys.argv[1:]
    install_qt_message_filter()
    ensure_apps_exist()      # macOS: leaves the app bundles behind, once
    if not argv and running_as_app():
        # Double-clicked in Finder: ask for a surface or overlay rather than
        # printing the command-line help into nowhere
        app = qt_application()
        argv = ask_for_files(app, "Open surface or overlay",
                             "Surfaces and overlays (*.gii *.annot *.txt);;All files (*)")
        if not argv:
            return
    opts = parse_args(argv)
    # Ensure a compatible OpenGL surface format before QApplication is created.
    try:
        QSurfaceFormat.setDefaultFormat(QVTKRenderWindowInteractor.defaultFormat())
    except Exception:
        pass
    try:
        QtWidgets.QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    except Exception:
        try:
            QtWidgets.QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
        except Exception:
            pass
    app = qt_application()
    win = Viewer(opts); win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main(sys.argv[1:])

