#!/usr/bin/env python3
"""
CAT_ViewSurf — PySide6 + VTK port with right-side control panel

Features:
  • Load LH mesh (.gii). Auto-detect RH mesh via name pattern ("lh."→"rh.", "left"→"right").
  • Optional overlay scalars (.gii; FreeSurfer morph: thickness/curv/sulc; or text one value/line).
  • Optional background scalars for curvature shading.
  • Six-view montage (lat/med/sup/inf/ant/post) by cloning actors with transforms.
  • Colormaps: C1, C2, C3, JET, FIRE, BIPOLAR, GRAY. Discrete levels, inverse, clip window.
  • Colorbar (VTK 9.5-compatible AddViewProp), optional stats in title.
  • Right-side docked control panel: range, clip, colorbar toggle, overlay picker, opacity, bkg range, stats, inverse.
  • Keyboard: u/d/l/r rotate (Shift=±1°, Ctrl=180°), b flip, o reset, g screenshot, plus standard VTK keys.

Requires: vtk (>=9), PySide6; nibabel (for GIFTI fallback + FreeSurfer textures if VTK lacks vtkGIFTIReader).

Usage
-----
Preferred (uses the repo's venv wrapper):

    scripts/cat_viewsurf.sh <mesh_or_overlay> [more_overlays...] [options]

Direct invocation:

    python src/t1prep/gui/cat_viewsurf.py <mesh_or_overlay> [more_overlays...] [options]
"""
from __future__ import annotations
import argparse
import os
import sys
import re
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# --- Qt setup (PySide6 only) ---
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QPainter, QColor, QPen, QBrush

# Qt compatibility shims
ORIENT_H = Qt.Orientation.Horizontal
DOCK_RIGHT = Qt.DockWidgetArea.RightDockWidgetArea
DOCK_LEFT = Qt.DockWidgetArea.LeftDockWidgetArea


# --- Import naming utilities ---
# (No local utils needed in this module)

# --- VTK imports (module-accurate for common wheels) ---
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkDoubleArray, vtkPoints
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
from vtkmodules.vtkIOImage import vtkPNGWriter, vtkNIFTIImageReader
from vtkmodules.vtkRenderingCore import vtkImageActor, vtkPropPicker
from vtkmodules.vtkRenderingCore import vtkCellPicker
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkImagingCore import vtkImageReslice
from vtkmodules.vtkImagingColor import vtkImageMapToWindowLevelColors
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage

# Qt interactor & backends
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkRenderingOpenGL2   # registers OpenGL2 backend (fixes vtkShaderProperty)
import vtkmodules.vtkRenderingFreeType  # text rendering for labels/ScalarBar

# --- Enums & defaults ---
C1, C2, C3, JET, HOT, FIRE, BIPOLAR, GRAY = range(8)
DEFAULT_WINDOW_SIZE = (1800, 800)

# ---- Lookup table helper ----
class LookupTableWithEnabling(vtkLookupTable):
    def __init__(self):
        super().__init__()
        self._enabled_scalars = None
    def SetEnabledArray(self, arr: vtkDoubleArray):
        self._enabled_scalars = arr

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

# ---- Naming helpers ----
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

    return str(mesh_candidate or overlay_path)

def is_overlay_file(filename: str) -> bool:
    """Heuristic check whether a path is an overlay (texture/label) rather than a mesh."""
    filename_only = Path(filename).name
    filename_lower = filename_only.lower()

    parts = filename_lower.split('.')
    mesh_types = ['central','pial','white','inflated','sphere','patch','mc','sqrtsulc']
    # Case A: lh.kind.subject (>=3 parts)
    if len(parts) >= 3 and parts[0] in ('lh', 'rh'):
        if parts[1] not in mesh_types:
            return True
    # Case B: lh.kind (exactly 2 parts, no subject)
    if len(parts) == 2 and parts[0] in ('lh', 'rh'):
        if parts[1] not in mesh_types:
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

# ---- I/O helpers ----
def read_gifti_mesh(filename: str) -> vtkPolyData:
    if HAVE_VTK_GIFTI:
        r = vtkGIFTIReader(); r.SetFileName(filename); r.Update()
        out = r.GetOutput()
        if out is None or out.GetNumberOfPoints() == 0:
            raise RuntimeError(f"Failed to read mesh from {filename}")
        return out
    # Fallback: nibabel
    try:
        import nibabel as nib
        g = nib.load(filename)
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


def read_scalars(filename: str) -> vtkDoubleArray:
    ext = Path(filename).suffix.lower()

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
                npv = vtk_to_numpy(arr).astype(float)
                out = vtkDoubleArray(); out.SetNumberOfTuples(len(npv))
                for i, v in enumerate(npv): out.SetValue(i, float(v))
                return out
        # fallback with nibabel
        try:
            import nibabel as nib
            g = nib.load(filename)
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
        out = vtkDoubleArray(); out.SetNumberOfTuples(len(data_arr))
        for i, v in enumerate(data_arr): out.SetValue(i, float(v))
        return out

    # --- Case 2: FreeSurfer texture data (thickness/curv/sulc, with or without extension) ---
    try:
        from nibabel.freesurfer.io import read_morph_data
        fs = read_morph_data(filename)
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
    out = vtkDoubleArray(); out.SetNumberOfTuples(len(data))
    for i, v in enumerate(data): out.SetValue(i, float(v))
    return out

# ---- Stats ----
def get_mean(arr: vtkDoubleArray) -> float: return float(np.nanmean(vtk_to_numpy(arr)))
def get_median(arr: vtkDoubleArray) -> float: return float(np.nanmedian(vtk_to_numpy(arr)))
def get_std(arr: vtkDoubleArray) -> float: return float(np.nanstd(vtk_to_numpy(arr)))

# ---- Interactor style ----
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
class CustomInteractorStyle(vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        super().__init__(); self._renderer: Optional[vtkRenderer] = None
        self._viewer = None  # Reference to the main viewer
        # Keys handled by the Viewer (suppress default VTK behavior for these)
        self._viewer_keys = {
            'q','Q','u','U','d','D','l','L','r','R','o','O','b','B','g','G','h','H','Left','Right'
        }
    def SetRenderer(self, ren: vtkRenderer): self._renderer = ren
    def SetViewer(self, viewer): self._viewer = viewer
    def _volume_open(self) -> bool:
        try:
            return bool(self._viewer and getattr(self._viewer, '_volume_windows', None))
        except Exception:
            return False
    def OnKeyPress(self):
        # Suppress default handling for keys that our viewer manages (e.g., 'r' reset)
        try:
            sym = self.GetInteractor().GetKeySym()
        except Exception:
            sym = None
        try:
            code = self.GetInteractor().GetKeyCode()
        except Exception:
            code = None
        if sym in self._viewer_keys or (isinstance(code, str) and code in 'uUdDlLrRoObBgGhH'):
            return
        return super().OnKeyPress()

    def OnChar(self):
        # Suppress default handling for keys that our viewer manages
        try:
            sym = self.GetInteractor().GetKeySym()
        except Exception:
            sym = None
        try:
            code = self.GetInteractor().GetKeyCode()
        except Exception:
            code = None
        if sym in self._viewer_keys or (isinstance(code, str) and code in 'uUdDlLrRoObBgGhH'):
            return
        return super().OnChar()

    def OnKeyDown(self):
        # Suppress default handling on key-down as some styles act here
        try:
            sym = self.GetInteractor().GetKeySym()
        except Exception:
            sym = None
        try:
            code = self.GetInteractor().GetKeyCode()
        except Exception:
            code = None
        if sym in self._viewer_keys or (isinstance(code, str) and code in 'uUdDlLrRoObBgGhH'):
            return
        return super().OnKeyDown()

    def OnLeftButtonDown(self):
        # If a volume window is open, consume left-click to avoid starting rotation.
        # The Viewer installs its own LeftButtonPressEvent handler to perform picking and broadcast to the volume window.
        try:
            if self._volume_open():
                return  # do not call super() -> prevents rotate on drag
        except Exception:
            pass
        return super().OnLeftButtonDown()

# ---- Options & CLI ----
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

def parse_args(argv: List[str]) -> Options:
    p = argparse.ArgumentParser(
        prog='cat_viewsurf.py',
        description='Render LH/RH surfaces with optional overlays (CAT_ViewSurf.py).\n\n'
                    'Usage examples:\n'
                    '  • Single mesh: python src/t1prep/gui/cat_viewsurf.py lh.central.name.gii\n'
                    '  • Single overlay: python src/t1prep/gui/cat_viewsurf.py lh.thickness.name1\n'
                    '  • Multiple overlays (navigate with ←/→): python src/t1prep/gui/cat_viewsurf.py lh.thickness.name1 lh.thickness.name2 ...',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Accept one or more positional inputs. If more than one is given, treat all as overlays
    # and derive the mesh from the first overlay via naming rules.
    p.add_argument(
        'inputs', nargs='*',
        help='Mesh or overlay(s). If multiple values are provided, they are treated as overlays.'
    )
    p.add_argument('-overlay','-ov', dest='overlay', help='Overlay scalars (.gii, FreeSurfer morph, or text)')
    p.add_argument('-overlays', dest='overlays', nargs='+', help='Multiple overlay files for navigation')
    p.add_argument('-bkg', dest='overlay_bkg', help='Background scalars for curvature shading (.gii or text)')
    p.add_argument('-volume','-vol','--nifti', dest='volume', help='3D NIfTI volume to display in a separate orthogonal view window')
    p.add_argument('-range','-r', dest='range', nargs=2, type=float, default=[0.0, -1.0], help='Overlay value range (min max). Use -1 for auto.')
    p.add_argument('-range-bkg','-rb', dest='range_bkg', nargs=2, type=float, default=[0.0, -1.0], help='Background value range (min max). Use -1 for auto.')
    p.add_argument('-clip','-cl', dest='clip', nargs=2, type=float, default=[0.0, -1.0], help='Clip range for display (min max). Use -1 for none/auto.')
    p.add_argument('-size','-sz', dest='size', nargs=2, type=int, default=list(DEFAULT_WINDOW_SIZE), help='Window size in pixels (width height)')
    p.add_argument('-title', dest='title', help='Window/title string (overrides auto title)')
    p.add_argument('-output','-save', dest='output', help='Save a screenshot to this path and exit')
    p.add_argument('-fontsize','-fs', dest='fontsize', type=int, default=0, help='Title/font size (0 = auto)')
    p.add_argument('-opacity','-op', dest='opacity', type=float, default=0.8, help='Overlay opacity')
    p.add_argument('-stats', action='store_true', help='Deprecated: same as --title-mode stats when colorbar is shown')
    p.add_argument('--title-mode', dest='title_mode', choices=['shape','stats','none'], default='shape',
                   help='Colorbar title: shape (filename), stats, or none')
    p.add_argument('-inverse', action='store_true', help='Invert the overlay colormap')
    p.add_argument('-colorbar','-cb', dest='colorbar', action='store_true')
    p.add_argument('-discrete','-dsc', dest='discrete', type=int, default=0,
                   help='Number of discrete color levels (0 = continuous)')
    p.add_argument('-log', action='store_true', help='Use logarithmic scaling for overlay display')
    p.add_argument('-white', action='store_true', help='Use a white background')
    # Control panel visibility (default: hidden)
    p.add_argument('--panel', dest='panel', action='store_true', help='Start with the control panel shown')
    p.add_argument('--no-panel', dest='panel', action='store_false', help='Start with the control panel hidden (default)')
    p.set_defaults(panel=False)
    p.add_argument('-fire', action='store_true')
    p.add_argument('-bipolar', action='store_true')
    p.add_argument('-c1', action='store_true')
    p.add_argument('-c2', action='store_true')
    p.add_argument('-c3', action='store_true')
    p.add_argument('-fix-scaling', dest='fix_scaling', action='store_true', help='Fix scaling across all overlays')
    p.add_argument('-debug', action='store_true', help='Enable debug output')
    # External defaults file for viewer settings (key=value lines)
    p.add_argument('--defaults', dest='defaults', help='Path to a defaults file (key=value) to override built-in defaults')
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
            script_dir.parent / 'data' / 'cat_viewsurf_defaults.txt',
            script_dir / 'cat_viewsurf_defaults.txt',
            Path.cwd() / 'cat_viewsurf_defaults.txt',
        ]
        for c in candidates:
            if c.exists():
                cfg = _load_defaults_file(str(c))
                _apply_defaults_cfg(cfg, a, defaults_ns)
                break

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
        if str(single).lower().endswith('.gii') and is_gifti_mesh_file(single):
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
            if str(pth).lower().endswith('.gii') and is_gifti_mesh_file(pth):
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
    )

# ---- Control Panel ----
class ControlPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(10,10,10,10)
        form = QtWidgets.QFormLayout()
        # Internal bounds for slider mapping (min,max)
        self._overlay_bounds = (-1.0, 1.0)
        self._clip_bounds = (-1.0, 1.0)
        self._bkg_bounds = (-1.0, 1.0)

        # Overlay selector (editable combo for long names + direct selection) — FIRST ROW
        self.overlay_combo = QtWidgets.QComboBox()
        self.overlay_combo.setEditable(True)
        self.overlay_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.overlay_combo.setMinimumContentsLength(50)
        try:
            # Widen dropdown popup for long paths
            self.overlay_combo.view().setMinimumWidth(600)
        except Exception:
            pass
        self.overlay_btn = QtWidgets.QPushButton("…")
        ov_box = QtWidgets.QHBoxLayout(); ov_box.addWidget(self.overlay_combo, 1); ov_box.addWidget(self.overlay_btn)
        form.addRow("Overlay", self._wrap(ov_box))

        # Volume (orthogonal view) — simple button
        self.volume_btn = QtWidgets.QPushButton("Open NIfTI…")
        form.addRow("Volume", self.volume_btn)

        # Range (overlay)
        self.range_min = QtWidgets.QDoubleSpinBox(); self.range_min.setDecimals(6); self.range_min.setRange(-1e9, 1e9)
        self.range_max = QtWidgets.QDoubleSpinBox(); self.range_max.setDecimals(6); self.range_max.setRange(-1e9, 1e9)
        self.range_slider_min = QtWidgets.QSlider(ORIENT_H); self.range_slider_min.setRange(0, 1000)
        self.range_slider_max = QtWidgets.QSlider(ORIENT_H); self.range_slider_max.setRange(0, 1000)
        range_box = QtWidgets.QHBoxLayout(); range_box.addWidget(self.range_min); range_box.addWidget(self.range_slider_min); range_box.addWidget(self.range_slider_max); range_box.addWidget(self.range_max)
        form.addRow("Range (overlay)", self._wrap(range_box))

        # Clip
        self.clip_min = QtWidgets.QDoubleSpinBox(); self.clip_min.setDecimals(6); self.clip_min.setRange(-1e9, 1e9)
        self.clip_max = QtWidgets.QDoubleSpinBox(); self.clip_max.setDecimals(6); self.clip_max.setRange(-1e9, 1e9)
        self.clip_slider_min = QtWidgets.QSlider(ORIENT_H); self.clip_slider_min.setRange(0, 1000)
        self.clip_slider_max = QtWidgets.QSlider(ORIENT_H); self.clip_slider_max.setRange(0, 1000)
        clip_box = QtWidgets.QHBoxLayout(); clip_box.addWidget(self.clip_min); clip_box.addWidget(self.clip_slider_min); clip_box.addWidget(self.clip_slider_max); clip_box.addWidget(self.clip_max)
        form.addRow("Clip window", self._wrap(clip_box))

        # Range bkg
        self.bkg_min = QtWidgets.QDoubleSpinBox(); self.bkg_min.setDecimals(6); self.bkg_min.setRange(-1e9, 1e9)
        self.bkg_max = QtWidgets.QDoubleSpinBox(); self.bkg_max.setDecimals(6); self.bkg_max.setRange(-1e9, 1e9)
        self.bkg_slider_min = QtWidgets.QSlider(ORIENT_H); self.bkg_slider_min.setRange(0, 1000)
        self.bkg_slider_max = QtWidgets.QSlider(ORIENT_H); self.bkg_slider_max.setRange(0, 1000)
        bkg_box = QtWidgets.QHBoxLayout(); bkg_box.addWidget(self.bkg_min); bkg_box.addWidget(self.bkg_slider_min); bkg_box.addWidget(self.bkg_slider_max); bkg_box.addWidget(self.bkg_max)
        form.addRow("Range (bkg)", self._wrap(bkg_box))
        # Opacity
        self.opacity = QtWidgets.QSlider(ORIENT_H); self.opacity.setRange(0,100); self.opacity.setValue(80)
        form.addRow("Opacity", self.opacity)
        # Toggles
        self.cb_colorbar = QtWidgets.QCheckBox("Show colorbar")
        self.cb_discrete = QtWidgets.QCheckBox("Discrete")
        # Colormap selector
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(["JET","HOT","FIRE","BIPOLAR","GRAY","C1","C2","C3"])  # order visible to user
        # Title mode selector (shape | stats | none)
        self.title_mode = QtWidgets.QComboBox(); self.title_mode.addItems(["shape","stats","none"])
        self.cb_inverse = QtWidgets.QCheckBox("Inverse (flip sign)")
        self.cb_fix_scaling = QtWidgets.QCheckBox("Fix scaling")
        self.cb_histogram = QtWidgets.QCheckBox("Show histogram")
        # Put Show colorbar and Colorbar title on one row (aligned with other checkboxes)
        row = QtWidgets.QHBoxLayout(); row.setContentsMargins(0,0,0,0); row.setSpacing(8)
        row.addWidget(self.cb_colorbar)
        row.addWidget(self.cb_discrete)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Colormap"))
        row.addWidget(self.colormap)
        row.addWidget(QtWidgets.QLabel("Colorbar title"))
        row.addWidget(self.title_mode)
        form.addRow(self._wrap(row))
        form.addRow(self.cb_inverse)
        form.addRow(self.cb_fix_scaling)
        form.addRow(self.cb_histogram)
        self.layout.addLayout(form)
        # Action buttons (none for now)
        self.layout.addStretch(1)

        # --- Wiring: bidirectional sync between sliders and spin boxes ---
        # Overlay range
        self.range_slider_min.valueChanged.connect(lambda v: self._slider_to_spin('overlay', 'min', v))
        self.range_slider_max.valueChanged.connect(lambda v: self._slider_to_spin('overlay', 'max', v))
        self.range_min.valueChanged.connect(lambda v: self._spin_to_slider('overlay', 'min', float(v)))
        self.range_max.valueChanged.connect(lambda v: self._spin_to_slider('overlay', 'max', float(v)))
        # Clip window
        self.clip_slider_min.valueChanged.connect(lambda v: self._slider_to_spin('clip', 'min', v))
        self.clip_slider_max.valueChanged.connect(lambda v: self._slider_to_spin('clip', 'max', v))
        self.clip_min.valueChanged.connect(lambda v: self._spin_to_slider('clip', 'min', float(v)))
        self.clip_max.valueChanged.connect(lambda v: self._spin_to_slider('clip', 'max', float(v)))
        # Background
        self.bkg_slider_min.valueChanged.connect(lambda v: self._slider_to_spin('bkg', 'min', v))
        self.bkg_slider_max.valueChanged.connect(lambda v: self._slider_to_spin('bkg', 'max', v))
        self.bkg_min.valueChanged.connect(lambda v: self._spin_to_slider('bkg', 'min', float(v)))
        self.bkg_max.valueChanged.connect(lambda v: self._spin_to_slider('bkg', 'max', float(v)))

    def _wrap(self, hbox: QtWidgets.QHBoxLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); w.setLayout(hbox); return w
    
    def set_overlay_controls_enabled(self, enabled: bool):
        """Enable or disable overlay-related controls based on whether an overlay is loaded."""
        # Ensure a strict boolean is used (avoid None/[] leaking from callers)
        enabled = bool(enabled)
        # Range controls
        self.range_min.setEnabled(enabled)
        self.range_max.setEnabled(enabled)
        self.range_slider_min.setEnabled(enabled)
        self.range_slider_max.setEnabled(enabled)
        # Clip controls
        self.clip_min.setEnabled(enabled)
        self.clip_max.setEnabled(enabled)
        self.clip_slider_min.setEnabled(enabled)
        self.clip_slider_max.setEnabled(enabled)
        # Colorbar and title controls
        self.cb_colorbar.setEnabled(enabled)
        # Discrete applies to the overlay LUT regardless of colorbar visibility
        self.cb_discrete.setEnabled(enabled)
        # Title is relevant only when the colorbar is shown
        self.title_mode.setEnabled(enabled and self.cb_colorbar.isChecked())
        # Colormap selector
        self.colormap.setEnabled(enabled)
        # Inverse control
        self.cb_inverse.setEnabled(enabled)
        # Background and opacity are also not meaningful until data is loaded
        self.bkg_min.setEnabled(enabled)
        self.bkg_max.setEnabled(enabled)
        self.bkg_slider_min.setEnabled(enabled)
        self.bkg_slider_max.setEnabled(enabled)
        self.opacity.setEnabled(enabled)
        # Fix scaling only makes sense with at least one overlay
        self.cb_fix_scaling.setEnabled(enabled)
        # Histogram available only when overlay is loaded
        self.cb_histogram.setEnabled(enabled)
        if not enabled:
            try:
                self.cb_histogram.blockSignals(True)
                self.cb_histogram.setChecked(False)
            finally:
                self.cb_histogram.blockSignals(False)

    # ---- Slider helpers ----
    def _bounds(self, which: str):
        return {
            'overlay': self._overlay_bounds,
            'clip': self._clip_bounds,
            'bkg': self._bkg_bounds,
        }[which]

    @staticmethod
    def _to_slider(value: float, bounds: tuple) -> int:
        a, b = bounds
        if b <= a:
            return 0
        t = (float(value) - float(a)) / (float(b) - float(a))
        t = max(0.0, min(1.0, t))
        return int(round(t * 1000.0))

    @staticmethod
    def _from_slider(ticks: int, bounds: tuple) -> float:
        a, b = bounds
        if b <= a:
            return float(a)
        t = max(0, min(1000, int(ticks))) / 1000.0
        return float(a) + t * (float(b) - float(a))

    def _slider_to_spin(self, which: str, part: str, ticks: int):
        bounds = self._bounds(which)
        val = self._from_slider(ticks, bounds)
        if which == 'overlay':
            if part == 'min':
                # Enforce min <= max
                if self.range_slider_min.value() > self.range_slider_max.value():
                    self.range_slider_max.blockSignals(True)
                    self.range_slider_max.setValue(self.range_slider_min.value())
                    self.range_slider_max.blockSignals(False)
                self.range_min.blockSignals(True); self.range_min.setValue(val); self.range_min.blockSignals(False)
            else:
                if self.range_slider_max.value() < self.range_slider_min.value():
                    self.range_slider_min.blockSignals(True)
                    self.range_slider_min.setValue(self.range_slider_max.value())
                    self.range_slider_min.blockSignals(False)
                self.range_max.blockSignals(True); self.range_max.setValue(val); self.range_max.blockSignals(False)
        elif which == 'clip':
            if part == 'min':
                if self.clip_slider_min.value() > self.clip_slider_max.value():
                    self.clip_slider_max.blockSignals(True)
                    self.clip_slider_max.setValue(self.clip_slider_min.value())
                    self.clip_slider_max.blockSignals(False)
                self.clip_min.blockSignals(True); self.clip_min.setValue(val); self.clip_min.blockSignals(False)
            else:
                if self.clip_slider_max.value() < self.clip_slider_min.value():
                    self.clip_slider_min.blockSignals(True)
                    self.clip_slider_min.setValue(self.clip_slider_max.value())
                    self.clip_slider_min.blockSignals(False)
                self.clip_max.blockSignals(True); self.clip_max.setValue(val); self.clip_max.blockSignals(False)
        elif which == 'bkg':
            if part == 'min':
                if self.bkg_slider_min.value() > self.bkg_slider_max.value():
                    self.bkg_slider_max.blockSignals(True)
                    self.bkg_slider_max.setValue(self.bkg_slider_min.value())
                    self.bkg_slider_max.blockSignals(False)
                self.bkg_min.blockSignals(True); self.bkg_min.setValue(val); self.bkg_min.blockSignals(False)
            else:
                if self.bkg_slider_max.value() < self.bkg_slider_min.value():
                    self.bkg_slider_min.blockSignals(True)
                    self.bkg_slider_min.setValue(self.bkg_slider_max.value())
                    self.bkg_slider_min.blockSignals(False)
                self.bkg_max.blockSignals(True); self.bkg_max.setValue(val); self.bkg_max.blockSignals(False)

    def _spin_to_slider(self, which: str, part: str, value: float):
        bounds = self._bounds(which)
        ticks = self._to_slider(value, bounds)
        if which == 'overlay':
            if part == 'min':
                if ticks > self.range_slider_max.value():
                    self.range_slider_max.blockSignals(True)
                    self.range_slider_max.setValue(ticks)
                    self.range_slider_max.blockSignals(False)
                self.range_slider_min.blockSignals(True); self.range_slider_min.setValue(ticks); self.range_slider_min.blockSignals(False)
            else:
                if ticks < self.range_slider_min.value():
                    self.range_slider_min.blockSignals(True)
                    self.range_slider_min.setValue(ticks)
                    self.range_slider_min.blockSignals(False)
                self.range_slider_max.blockSignals(True); self.range_slider_max.setValue(ticks); self.range_slider_max.blockSignals(False)
        elif which == 'clip':
            if part == 'min':
                if ticks > self.clip_slider_max.value():
                    self.clip_slider_max.blockSignals(True)
                    self.clip_slider_max.setValue(ticks)
                    self.clip_slider_max.blockSignals(False)
                self.clip_slider_min.blockSignals(True); self.clip_slider_min.setValue(ticks); self.clip_slider_min.blockSignals(False)
            else:
                if ticks < self.clip_slider_min.value():
                    self.clip_slider_min.blockSignals(True)
                    self.clip_slider_min.setValue(ticks)
                    self.clip_slider_min.blockSignals(False)
                self.clip_slider_max.blockSignals(True); self.clip_slider_max.setValue(ticks); self.clip_slider_max.blockSignals(False)
        elif which == 'bkg':
            if part == 'min':
                if ticks > self.bkg_slider_max.value():
                    self.bkg_slider_max.blockSignals(True)
                    self.bkg_slider_max.setValue(ticks)
                    self.bkg_slider_max.blockSignals(False)
                self.bkg_slider_min.blockSignals(True); self.bkg_slider_min.setValue(ticks); self.bkg_slider_min.blockSignals(False)
            else:
                if ticks < self.bkg_slider_min.value():
                    self.bkg_slider_min.blockSignals(True)
                    self.bkg_slider_min.setValue(ticks)
                    self.bkg_slider_min.blockSignals(False)
                self.bkg_slider_max.blockSignals(True); self.bkg_slider_max.setValue(ticks); self.bkg_slider_max.blockSignals(False)

    # Public: set slider bounds (min,max) and align slider positions to current spin values
    def set_overlay_bounds(self, vmin: float, vmax: float):
        self._overlay_bounds = (float(vmin), float(vmax))
        self._spin_to_slider('overlay', 'min', float(self.range_min.value()))
        self._spin_to_slider('overlay', 'max', float(self.range_max.value()))

    def set_clip_bounds(self, vmin: float, vmax: float):
        self._clip_bounds = (float(vmin), float(vmax))
        self._spin_to_slider('clip', 'min', float(self.clip_min.value()))
        self._spin_to_slider('clip', 'max', float(self.clip_max.value()))

    def set_bkg_bounds(self, vmin: float, vmax: float):
        self._bkg_bounds = (float(vmin), float(vmax))
        self._spin_to_slider('bkg', 'min', float(self.bkg_min.value()))
        self._spin_to_slider('bkg', 'max', float(self.bkg_max.value()))

# ---- Viewer ----
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, opts: Options):
        super().__init__()
        self.opts = opts
        self._y_shift_l: float = 0.0
        self._y_shift_r: float = 0.0
        self._hist_win = None  # histogram window reference
        # Mesh navigation state (when multiple input meshes and no overlay)
        self.mesh_list: List[str] = list(self.opts.meshes or [])
        self.current_mesh_index: int = 0
        # Use the original input file for the window title
        name_part = Path(self.opts.mesh_left).name
        self.setWindowTitle((self.opts.title or name_part).replace('.gii','').replace('.txt',''))
        self.resize(*opts.size)

        # central widget with VTK view
        self.frame = QtWidgets.QFrame(); self.vl = QtWidgets.QVBoxLayout(); self.vl.setContentsMargins(0,0,0,0)
        self.frame.setLayout(self.vl); self.setCentralWidget(self.frame)
        self.vtk_widget = QVTKRenderWindowInteractor(self.frame); self.vl.addWidget(self.vtk_widget)
        # Ensure the VTK widget can accept keyboard focus
        try:
            self.vtk_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        except Exception:
            self.vtk_widget.setFocusPolicy(Qt.StrongFocus)

        self.ren = vtkRenderer(); self.ren.SetBackground(1,1,1) if opts.white else self.ren.SetBackground(0,0,0)
        self.rw: vtkRenderWindow = self.vtk_widget.GetRenderWindow();
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
        # Also observe key events explicitly for overlay navigation only (avoid duplicates)
        def _on_keypress(obj, ev):
            # Prefer handling our toggle here if modifiers are pressed and match
            try:
                sym = None
                try:
                    sym = self.iren.GetKeySym()
                except Exception:
                    sym = None
            except Exception:
                pass
            try:
                if sym is None:
                    sym = self.iren.GetKeySym()
            except Exception:
                sym = None
            self._handle_key(sym)
        self.iren.AddObserver("KeyPressEvent", _on_keypress)

        # Clicking on the surface forwards the picked world position to any open volume windows
        def _on_left_click(_obj, _evt):
            try:
                x, y = self.iren.GetEventPosition()
            except Exception:
                return
            mm = self._surface_click_to_mm(x, y)
            if mm is not None:
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
            # Determine if input is an overlay file; but let .gii meshes pass through
            if (not (str(input_path).lower().endswith('.gii') and is_gifti_mesh_file(str(input_path)))
                and is_overlay_file(str(input_path))):
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
            self.poly_l = read_gifti_mesh(str(left_mesh_path))
            rh_candidate: Optional[Path] = None
            name = left_mesh_path.name
            if 'lh.' in name:
                rh_candidate = left_mesh_path.with_name(name.replace('lh.', 'rh.'))
            elif 'left' in name:
                rh_candidate = left_mesh_path.with_name(name.replace('left', 'right'))
            elif '_hemi-L_' in name:
                rh_candidate = left_mesh_path.with_name(name.replace('_hemi-L_', '_hemi-R_'))
            elif '_hemi-R_' in name:
                rh_candidate = left_mesh_path.with_name(name.replace('_hemi-R_', '_hemi-L_'))
            if rh_candidate and rh_candidate.exists(): 
                self.poly_r = read_gifti_mesh(str(rh_candidate))
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
            rh_overlay_bkg = None
            if 'lh.' in opts.overlay_bkg: rh_overlay_bkg = opts.overlay_bkg.replace('lh.', 'rh.')
            elif 'left' in opts.overlay_bkg: rh_overlay_bkg = opts.overlay_bkg.replace('left', 'right')
            if rh_overlay_bkg and Path(rh_overlay_bkg).exists() and self.poly_r is not None:
                self.bkg_scalar_r = read_scalars(rh_overlay_bkg)
            elif self.poly_r is not None and self.bkg_scalar_l is not None and self.bkg_scalar_l.GetNumberOfTuples() == (self.poly_l.GetNumberOfPoints()+self.poly_r.GetNumberOfPoints()):
                nL = self.poly_l.GetNumberOfPoints(); nR = self.poly_r.GetNumberOfPoints(); arr = self.bkg_scalar_l
                self.bkg_scalar_r = vtkDoubleArray(); self.bkg_scalar_r.SetNumberOfTuples(nR)
                for i in range(nR): self.bkg_scalar_r.SetValue(i, arr.GetValue(i+nL))
                left_only = vtkDoubleArray(); left_only.SetNumberOfTuples(nL)
                for i in range(nL): left_only.SetValue(i, arr.GetValue(i))
                self.bkg_scalar_l = left_only

        self.curv_l_out = self.curv_l.GetOutput();  
        if self.bkg_scalar_l is not None: self.curv_l_out.GetPointData().SetScalars(self.bkg_scalar_l)
        self.curv_r_out = None
        if self.curv_r is not None:
            self.curv_r_out = self.curv_r.GetOutput();
            if self.bkg_scalar_r is not None: self.curv_r_out.GetPointData().SetScalars(self.bkg_scalar_r)

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
        # Apply clip transparency to overlay LUTs (values inside clip become transparent)
        self._apply_clip_to_overlay_luts()
        lut_bkg = vtkLookupTable(); lut_bkg.SetHueRange(0,0); lut_bkg.SetSaturationRange(0,0); lut_bkg.SetValueRange(0,1); lut_bkg.Build()

        # Background scalar range
        self.range_bkg = list(opts.range_bkg)
        if not (self.range_bkg[1] > self.range_bkg[0]):
            r = [0.0,0.0]; self.curv_l_out.GetScalarRange(r); self.range_bkg = r
        if self.range_bkg[0] < 0 < self.range_bkg[1]:
            m = max(abs(self.range_bkg[0]), abs(self.range_bkg[1])); self.range_bkg = [-m, m]
        lut_bkg.SetTableRange(self.range_bkg)
        # Overlay range (init before calling _load_overlay)
        self.overlay_range = list(opts.range)
        # Predefine overlay actors (referenced in _load_overlay)
        self.actor_ov_l = None
        self.actor_ov_r = None

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

        # Mappers/actors
        self.actor_bkg_l = None
        if hasattr(self, 'curv_l_out') and self.curv_l_out is not None:
            mapper_bkg_l = vtkPolyDataMapper(); mapper_bkg_l.SetInputData(self.curv_l_out); mapper_bkg_l.SetLookupTable(lut_bkg); mapper_bkg_l.SetScalarRange(self.range_bkg)
            self.actor_bkg_l = vtkActor(); self.actor_bkg_l.SetMapper(mapper_bkg_l)
            self.actor_bkg_l.GetProperty().SetAmbient(0.8); self.actor_bkg_l.GetProperty().SetDiffuse(0.7)
            self._actors.append(self.actor_bkg_l)

        self.actor_ov_l = None
        if self.scal_l is not None:
            mapper_ov_l = vtkPolyDataMapper(); mapper_ov_l.SetInputData(self.poly_l); mapper_ov_l.SetLookupTable(self.lut_overlay_l)
            if self.overlay_range[1] > self.overlay_range[0]: mapper_ov_l.SetScalarRange(self.overlay_range)
            self.actor_ov_l = vtkActor(); self.actor_ov_l.SetMapper(mapper_ov_l)
            self.actor_ov_l.GetProperty().SetAmbient(0.3); self.actor_ov_l.GetProperty().SetDiffuse(0.7)
            self._actors.append(self.actor_ov_l)

        self.actor_bkg_r = None; self.actor_ov_r = None
        if self.poly_r is not None and hasattr(self, 'curv_r_out') and self.curv_r_out is not None:
            mapper_bkg_r = vtkPolyDataMapper(); mapper_bkg_r.SetInputData(self.curv_r_out); mapper_bkg_r.SetLookupTable(lut_bkg); mapper_bkg_r.SetScalarRange(self.range_bkg)
            self.actor_bkg_r = vtkActor(); self.actor_bkg_r.SetMapper(mapper_bkg_r)
            self.actor_bkg_r.GetProperty().SetAmbient(0.8); self.actor_bkg_r.GetProperty().SetDiffuse(0.7)
            self._actors.append(self.actor_bkg_r)
            if self.scal_r is not None:
                mapper_ov_r = vtkPolyDataMapper(); mapper_ov_r.SetInputData(self.poly_r); mapper_ov_r.SetLookupTable(self.lut_overlay_r)
                if self.overlay_range[1] > self.overlay_range[0]: mapper_ov_r.SetScalarRange(self.overlay_range)
                self.actor_ov_r = vtkActor(); self.actor_ov_r.SetMapper(mapper_ov_r)
                self.actor_ov_r.GetProperty().SetAmbient(0.3); self.actor_ov_r.GetProperty().SetDiffuse(0.7)
                self._actors.append(self.actor_ov_r)

        # Build 6-view montage
        views = 6; shifts = (180.0, 180.0)
        posx = [0, 2*shifts[0], 0.15*shifts[0], 1.85*shifts[0], shifts[0], shifts[0]]
        posy = [0, 0, 0.8*shifts[1], 0.8*shifts[1], 0.6*shifts[1], 0.6*shifts[1]]
        rotx = [270, 270, 270, 270, 0, 0]; rotz = [90, -90, -90, 90, 0, 0]
        order = [0,1,0,1,0,1]
        # Keep track of clones per view index for selective operations (e.g., key 'b')
        self._montage_bkg: List[Optional[vtkActor]] = [None]*views
        self._montage_ov: List[Optional[vtkActor]] = [None]*views
        def add_clone(actor: vtkActor, px, py, rx, rz) -> vtkActor:
            a = vtkActor(); a.ShallowCopy(actor); a.AddPosition(px, py, 0); a.RotateX(rx); a.RotateZ(rz); self.ren.AddActor(a); return a
        for i in range(views):
            if self.poly_r is None and (i % 2 == 1): continue
            src = self.actor_bkg_r if (order[i] == 1 and self.actor_bkg_r is not None) else self.actor_bkg_l
            if src is not None:
                a = add_clone(src, posx[i], posy[i], rotx[i], rotz[i])
                self._montage_bkg[i] = a
        if self.actor_ov_l is not None or self.actor_ov_r is not None:
            for i in range(views):
                if self.poly_r is None and (i % 2 == 1): continue
                src = self.actor_ov_r if (order[i] == 1 and self.actor_ov_r is not None) else self.actor_ov_l
                if src is not None:
                    a = add_clone(src, posx[i], posy[i], rotx[i], rotz[i])
                    self._montage_ov[i] = a

        # Colorbar: create once and attach/detach based on option
        self.scalar_bar = None
        self._scalar_bar_added = False
        self._ensure_colorbar()
        has_overlay_initial = (self.scal_l is not None)
        self._colorbar_intent = bool(self.opts.colorbar)
        self.opts.colorbar = bool(self._colorbar_intent and has_overlay_initial)
        if self.opts.colorbar:
            self._attach_colorbar()
        else:
            self._detach_colorbar()
        # Render once so initial state is applied
        try:
            self.rw.Render()
        except Exception:
            pass

        # Camera
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Zoom(2.0)
        # Track camera state to preserve view across overlay switches
        self._cam_state = None
        # Capture baseline camera right after initial setup
        self._capture_camera_state()
        self._base_cam_state = dict(self._cam_state) if self._cam_state else None

        # Right-side control panel (dock)
        self._build_control_panel()

        # Start interactor
        self.vtk_widget.Initialize(); self.vtk_widget.Start(); self.vtk_widget.setFocus()

        self.vtk_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu
                                             if hasattr(Qt, "ContextMenuPolicy")
                                             else Qt.CustomContextMenu)
        self.vtk_widget.customContextMenuRequested.connect(self._show_view_context_menu)

        # Focus is set during initialization; no extra activation timer needed

        # Optional snapshot
        if opts.output:
            self.save_png(opts.output)

        # (Shortcuts/event filter reverted to previous behavior)
        
    def _show_view_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        # reuse the same action so state stays in sync
        menu.addAction(self.act_show_controls)
        # position relative to the widget, map to global
        global_pos = self.vtk_widget.mapToGlobal(pos)
        menu.exec(global_pos)
    
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
        has_overlay = bool((self.overlay_list or self.opts.overlay) and (self.scal_l is not None))
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
            self.rw.Render()
        self.ctrl.clip_slider_min.sliderReleased.connect(_apply_clip_live)
        self.ctrl.clip_slider_max.sliderReleased.connect(_apply_clip_live)
        self.ctrl.clip_min.editingFinished.connect(_apply_clip_live)
        self.ctrl.clip_max.editingFinished.connect(_apply_clip_live)
    
        # Start state based on CLI flag --panel (default hidden)
        if self.opts.panel:
            dock.setFloating(True)
            dock.show()
            position_dock()
        else:
            dock.hide()
        # Initial status hint
        self._update_status_message(self.opts.panel)
        
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
            w2i = vtkWindowToImageFilter(); w2i.SetInput(self.rw); w2i.Update()
            name = Path(self.rw.GetWindowName() or 'screenshot').with_suffix('.png')
            png = vtkPNGWriter(); png.SetFileName(str(name)); png.SetInputConnection(w2i.GetOutputPort()); png.Write(); print(f"Saved {name}"); return
        if s in ('h','H'):
            hint = '⌘D' if sys.platform == 'darwin' else 'Ctrl+D'
            print(f"KEYS: u/d/l/r rotate, b flip, o reset, w/s wireframe/shaded, g screenshot, ←/→ overlay navigation, toggle controls: {hint}"); return

    

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
        """Flatten LUT into 'levels' discrete bands if discrete > 0.

        Interprets opts.discrete as the number of bands (1..4). For N levels,
        the 256-entry table is divided into N segments and each segment is
        filled with a representative color sampled at its start index.
        
        NOTE: This is for overlay LUTs applied to surfaces (no gaps).
        """
        levels = int(getattr(self.opts, 'discrete', 0) or 0)
        if levels <= 0:
            return
        levels = max(1, min(256, levels))
        n = 256
        for k in range(levels):
            start = int(k * n / levels)
            end = int((k + 1) * n / levels) if k < levels - 1 else n
            r, g, b, a = lut.GetTableValue(start)
            for i in range(start, end):
                lut.SetTableValue(i, r, g, b, a)

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
        """Reverse the order of colors in a LUT in-place.

        This flips how scalars map to colors without changing data values
        or scalar ranges. Alpha values are preserved with their colors.
        """
        try:
            n = int(lut.GetNumberOfTableValues())
            for i in range(n // 2):
                r1, g1, b1, a1 = lut.GetTableValue(i)
                r2, g2, b2, a2 = lut.GetTableValue(n - 1 - i)
                lut.SetTableValue(i, r2, g2, b2, a2)
                lut.SetTableValue(n - 1 - i, r1, g1, b1, a1)
        except Exception:
            pass

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
            for i in range(n):
                t = i / (n - 1 if n > 1 else 1)
                val = smin + t * (smax - smin)
                r, g, b, a = lut.GetTableValue(i)
                if c0 < val < c1:
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

    def _on_overlay_combo_changed(self, _idx: int):
        path = self.ctrl.overlay_combo.currentText().strip()
        if path:
            self._set_overlay_from_path(path)

    def _on_overlay_combo_edited(self):
        path = self.ctrl.overlay_combo.currentText().strip()
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
                self.setWindowTitle(Path(new_overlay).name)
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
                name_part = Path(self.opts.mesh_left).name
                self.setWindowTitle((self.opts.title or name_part).replace('.gii','').replace('.txt',''))
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
            overlay_name = Path(current_overlay).name
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
                self.setWindowTitle(Path(path).name)

    # --- Mesh navigation (when no overlay) ---
    def _switch_mesh(self, new_mesh_path: str):
        """Switch the underlying mesh to a new file path and update actors.

        Preserves camera state and Y-origin normalization. Overlay actors are
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
        # Load left mesh and normalize Y
        self.poly_l = read_gifti_mesh(str(p))
        self._y_shift_l = self._shift_y_to(self.poly_l) or 0.0
        # Find corresponding right mesh (best-effort) and normalize Y
        self.poly_r = None
        rh_candidate: Optional[Path] = None
        name = p.name
        if 'lh.' in name:
            rh_candidate = p.with_name(name.replace('lh.', 'rh.'))
        elif 'left' in name:
            rh_candidate = p.with_name(name.replace('left', 'right'))
        elif '_hemi-L_' in name:
            rh_candidate = p.with_name(name.replace('_hemi-L_', '_hemi-R_'))
        elif '_hemi-R_' in name:
            rh_candidate = p.with_name(name.replace('_hemi-R_', '_hemi-L_'))
        if rh_candidate and rh_candidate.exists():
            self.poly_r = read_gifti_mesh(str(rh_candidate))
            self._y_shift_r = self._shift_y_to(self.poly_r) or 0.0
        else:
            self._y_shift_r = 0.0
        # Rebuild curvature
        self.curv_l = vtkCurvatures(); self.curv_l.SetInputData(self.poly_l); self.curv_l.SetCurvatureTypeToMean(); self.curv_l.Update()
        self.curv_l_out = self.curv_l.GetOutput()
        if self.bkg_scalar_l is not None:
            self.curv_l_out.GetPointData().SetScalars(self.bkg_scalar_l)
        self.curv_r = None; self.curv_r_out = None
        if self.poly_r is not None:
            self.curv_r = vtkCurvatures(); self.curv_r.SetInputData(self.poly_r); self.curv_r.SetCurvatureTypeToMean(); self.curv_r.Update()
            self.curv_r_out = self.curv_r.GetOutput()
            if self.bkg_scalar_r is not None:
                self.curv_r_out.GetPointData().SetScalars(self.bkg_scalar_r)
        # Update inputs on existing actors/mappers
        if getattr(self, 'actor_bkg_l', None) is not None and self.curv_l_out is not None:
            self.actor_bkg_l.GetMapper().SetInputData(self.curv_l_out)
        if getattr(self, 'actor_bkg_r', None) is not None and self.curv_r_out is not None:
            self.actor_bkg_r.GetMapper().SetInputData(self.curv_r_out)
        if getattr(self, 'actor_ov_l', None) is not None and self.poly_l is not None:
            self.actor_ov_l.GetMapper().SetInputData(self.poly_l)
        if getattr(self, 'actor_ov_r', None) is not None and self.poly_r is not None:
            self.actor_ov_r.GetMapper().SetInputData(self.poly_r)
        # Keep slider bounds in sync with new data
        self._update_slider_bounds()
        # Update window title to mesh name if no overlay is active
        if not self.opts.overlay:
            try:
                name_part = Path(new_mesh_path).name
                self.setWindowTitle((self.opts.title or name_part).replace('.gii','').replace('.txt',''))
            except Exception:
                pass
        # Update stored mesh_left
        try:
            self.opts.mesh_left = str(p)
        except Exception:
            pass
        # Restore camera and render
        self._apply_camera_state()
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
          1) Use convert_filename_to_mesh heuristic.
          2) Search overlay directory for likely meshes (central/midthickness) with matching hemi tokens.
        """
        ov_path = Path(overlay_path)
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

    def _maybe_switch_mesh_for_overlay(self, overlay_path: str):
        """If the overlay implies a different mesh, switch meshes and rebuild mappers/actors.

        This allows cycling across overlays from different subjects/runs where the mesh files differ.
        """
        ov_path = Path(overlay_path)
        ov_dir = ov_path.parent
        new_mesh_path = self._find_mesh_for_overlay(str(ov_path))
        if new_mesh_path is None:
            return

        # If the target mesh is the same file, do nothing
        if getattr(self.opts, 'mesh_left', None):
            try:
                if new_mesh_path.resolve() == Path(self.opts.mesh_left).resolve():
                    return
            except Exception:
                pass
        # Load new left mesh
        try:
            self.poly_l = read_gifti_mesh(str(new_mesh_path))
            self.opts.mesh_left = str(new_mesh_path)
        except Exception:
            return
        # Normalize Y-origin to match initial alignment
        self._shift_y_to(self.poly_l)
        # Attempt right mesh detection similar to __init__
        self.poly_r = None
        rh_candidate: Optional[Path] = None
        name = new_mesh_path.name
        if 'lh.' in name:
            rh_candidate = new_mesh_path.with_name(name.replace('lh.', 'rh.'))
        elif 'left' in name:
            rh_candidate = new_mesh_path.with_name(name.replace('left', 'right'))
        elif '_hemi-L_' in name:
            rh_candidate = new_mesh_path.with_name(name.replace('_hemi-L_', '_hemi-R_'))
        elif '_hemi-R_' in name:
            rh_candidate = new_mesh_path.with_name(name.replace('_hemi-R_', '_hemi-L_'))
        if rh_candidate and rh_candidate.exists():
            self.poly_r = read_gifti_mesh(str(rh_candidate))
            # Normalize Y-origin for right mesh too
            self._shift_y_to(self.poly_r)
        # Recompute curvature and background on the new meshes
        self.curv_l = vtkCurvatures(); self.curv_l.SetInputData(self.poly_l); self.curv_l.SetCurvatureTypeToMean(); self.curv_l.Update()
        self.curv_l_out = self.curv_l.GetOutput()
        if self.bkg_scalar_l is not None:
            self.curv_l_out.GetPointData().SetScalars(self.bkg_scalar_l)
        self.curv_r = None; self.curv_r_out = None
        if self.poly_r is not None:
            self.curv_r = vtkCurvatures(); self.curv_r.SetInputData(self.poly_r); self.curv_r.SetCurvatureTypeToMean(); self.curv_r.Update()
            self.curv_r_out = self.curv_r.GetOutput()
            if self.bkg_scalar_r is not None:
                self.curv_r_out.GetPointData().SetScalars(self.bkg_scalar_r)

        # Rewire existing mappers/actors to the newly loaded meshes
        try:
            if self.actor_bkg_l is not None and self.curv_l_out is not None:
                self.actor_bkg_l.GetMapper().SetInputData(self.curv_l_out)
            if self.actor_ov_l is not None:
                self.actor_ov_l.GetMapper().SetInputData(self.poly_l)
            if self.actor_bkg_r is not None and self.curv_r_out is not None:
                self.actor_bkg_r.GetMapper().SetInputData(self.curv_r_out)
            if self.actor_ov_r is not None and self.poly_r is not None:
                self.actor_ov_r.GetMapper().SetInputData(self.poly_r)
            # Update montage clones to reference the refreshed actor mappers
            order = [0,1,0,1,0,1]
            for i, clone in enumerate(getattr(self, '_montage_bkg', []) or []):
                if clone is None:
                    continue
                src = self.actor_bkg_r if (order[i] == 1 and self.actor_bkg_r is not None) else self.actor_bkg_l
                if src is not None:
                    clone.SetMapper(src.GetMapper())
            for i, clone in enumerate(getattr(self, '_montage_ov', []) or []):
                if clone is None:
                    continue
                src = self.actor_ov_r if (order[i] == 1 and self.actor_ov_r is not None) else self.actor_ov_l
                if src is not None:
                    clone.SetMapper(src.GetMapper())
                else:
                    # Hide overlay clones when no overlay actor on that side
                    try:
                        clone.SetVisibility(False)
                    except Exception:
                        pass
        except Exception:
            pass
        # Update mappers for background actors
        if self.actor_bkg_l is not None:
            self.actor_bkg_l.GetMapper().SetInputData(self.curv_l_out)
        if self.actor_bkg_r is not None and self.curv_r_out is not None:
            self.actor_bkg_r.GetMapper().SetInputData(self.curv_r_out)
        # Update mappers for overlay actors if present
        if self.actor_ov_l is not None:
            self.actor_ov_l.GetMapper().SetInputData(self.poly_l)
        if self.actor_ov_r is not None and self.poly_r is not None:
            self.actor_ov_r.GetMapper().SetInputData(self.poly_r)
        # Keep the stored mesh_left updated for subsequent comparisons
        self.opts.mesh_left = str(new_mesh_path)

    

    def _apply_inverse(self):
        """Flip colormap without changing data or scalar ranges."""
        # Rebuild LUTs to ensure consistent base, then invert
        self.lut_overlay_l = get_lookup_table(self.opts.colormap, self.opts.opacity)
        self.lut_overlay_r = get_lookup_table(self.opts.colormap, self.opts.opacity)
        if self.opts.inverse:
            self._invert_lut(self.lut_overlay_l)
            self._invert_lut(self.lut_overlay_r)
        self._apply_discrete_to_overlay_lut(self.lut_overlay_l)
        self._apply_discrete_to_overlay_lut(self.lut_overlay_r)
        if self.actor_ov_l is not None:
            self.actor_ov_l.GetMapper().SetLookupTable(self.lut_overlay_l)
        if self.actor_ov_r is not None:
            self.actor_ov_r.GetMapper().SetLookupTable(self.lut_overlay_r)
        if self.opts.colorbar:
            self._ensure_colorbar()

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
                for i in range(n):
                    t = i / (n - 1 if n > 1 else 1)
                    val = smin + t * (smax - smin)
                    if c0 < val < c1:
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
            elif title_mode == 'stats' or (self.opts.stats and self.scal_l is not None):
                if self.scal_l is not None:
                    info = f"Mean={get_mean(self.scal_l):.3f} Median={get_median(self.scal_l):.3f} SD={get_std(self.scal_l):.3f}"
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
            for i in range(n):
                t = i / (n - 1 if n > 1 else 1)
                val = smin + t * (smax - smin)
                if c0 < val < c1:
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
        elif title_mode == 'stats' or (self.opts.stats and self.scal_l is not None):
            if self.scal_l is not None:
                info = f"Mean={get_mean(self.scal_l):.3f} Median={get_median(self.scal_l):.3f} SD={get_std(self.scal_l):.3f}"
                sb.SetTitle(info)
            else:
                sb.SetTitle("")
        else:
            sb.SetTitle(Path(self.opts.overlay or self.opts.mesh_left).name)

        try:
            sb.SetAnnotationTextScaling(False)
        except Exception:
            pass

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
        cwd = os.getcwd()
        try:
            os.chdir(str(Path(overlay_path).parent or Path('.')))
            baseL = Path(overlay_path).name
            scal_l = read_scalars(baseL); scal_r = None
            rh_name = None
            if 'lh.' in baseL: rh_name = baseL.replace('lh.', 'rh.')
            elif 'left' in baseL: rh_name = baseL.replace('left', 'right')
            if rh_name and Path(rh_name).exists() and self.poly_r is not None:
                scal_r = read_scalars(rh_name)
            elif self.poly_r is not None and scal_l is not None and scal_l.GetNumberOfTuples() == (self.poly_l.GetNumberOfPoints()+self.poly_r.GetNumberOfPoints()):
                nL = self.poly_l.GetNumberOfPoints(); nR = self.poly_r.GetNumberOfPoints(); arr = scal_l
                scal_r = vtkDoubleArray(); scal_r.SetNumberOfTuples(nR)
                for i in range(nR): scal_r.SetValue(i, arr.GetValue(i+nL))
                left_only = vtkDoubleArray(); left_only.SetNumberOfTuples(nL)
                for i in range(nL): left_only.SetValue(i, arr.GetValue(i))
                scal_l = left_only
        except Exception as e:
            # If loading fails, clear the overlay and disable controls
            print(f"Failed to load overlay: {e}")
            self.opts.overlay = None
            if hasattr(self, 'ctrl'):
                self.ctrl.set_overlay_controls_enabled(False)
            return
        finally:
            os.chdir(cwd)
        # Do not invert scalars; inversion is handled by flipping LUTs
        # Clip is rendered via LUT alpha; do not mutate scalar arrays
        # attach
        self.scal_l = scal_l; self.scal_r = scal_r
        if scal_l is not None: self.poly_l.GetPointData().SetScalars(scal_l)
        if scal_r is not None and self.poly_r is not None: self.poly_r.GetPointData().SetScalars(scal_r)
        # Predefined ranges for recognized overlays (thickness, pbt)
        kind = detect_overlay_kind(overlay_path)
        if kind in ('thickness', 'pbt') and not self.opts.fix_scaling:
            # Apply requested defaults: overlay 0.5..5; clip 0..0; bkg -1..1
            self.overlay_range = [0.5, 5.0]
            self.opts.clip = (0.0, 0.0)
            self.range_bkg = [-1.0, 1.0]
        else:
            # recompute overlay range if needed and fix scaling is not enabled
            if not (self.overlay_range[1] > self.overlay_range[0]) and (self.scal_l is not None):
                if not self.opts.fix_scaling:
                    r = [0.0,0.0]; self.poly_l.GetScalarRange(r); self.overlay_range = r
                elif self.fixed_overlay_range is not None:
                    # Use the fixed range
                    self.overlay_range = list(self.fixed_overlay_range)
        for actor in (self.actor_ov_l, self.actor_ov_r):
            if actor:
                actor.GetMapper().SetLookupTable(self.lut_overlay_l)
                if self.overlay_range[1] > self.overlay_range[0]: actor.GetMapper().SetScalarRange(self.overlay_range)
        # Apply background range to background actors when set (if already created)
        if self.range_bkg[1] > self.range_bkg[0]:
            for actor in (getattr(self, 'actor_bkg_l', None), getattr(self, 'actor_bkg_r', None)):
                if actor:
                    actor.GetMapper().SetScalarRange(self.range_bkg)
        if self.opts.colorbar: self._ensure_colorbar()
        # Apply clip transparency and refresh LUTs on actors
        self._apply_clip_to_overlay_luts()
        for actor in (self.actor_ov_l, self.actor_ov_r):
            if actor:
                actor.GetMapper().SetLookupTable(self.lut_overlay_l)
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
                self.ctrl.clip_min.setValue(0.0); self.ctrl.clip_max.setValue(0.0)
                self.ctrl.bkg_min.setValue(-1.0); self.ctrl.bkg_max.setValue(1.0)
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
            if self.overlay_range[1] > self.overlay_range[0]:
                rng = (float(self.overlay_range[0]), float(self.overlay_range[1]))
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
        w2i = vtkWindowToImageFilter(); w2i.SetInput(self.rw); w2i.Update()
        writer = vtkPNGWriter(); writer.SetFileName(path); writer.SetInputConnection(w2i.GetOutputPort()); writer.Write()
        print(f"Saved {path}")

    # -- Volume Integration --
    def _open_volume_dialog(self):
        start_dir = str(Path(self.opts.mesh_left).parent) if self.opts.mesh_left else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open 3D NIfTI volume", start_dir, "NIfTI (*.nii *.nii.gz);;All files (*)")
        if path:
            self._open_volume(path)

    def _open_volume(self, volume_path: str):
        try:
            win = OrthoVolumeWindow(volume_path, parent=self)
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

    def _broadcast_world_pick(self, world_xyz: Tuple[float, float, float]):
        """Send a picked world coordinate to all open OrthoVolumeWindow instances."""
        if not hasattr(self, '_volume_windows'):
            return
        # prune closed windows
        self._volume_windows = [w for w in self._volume_windows if w is not None and w.isVisible()]
        for w in list(self._volume_windows):
            try:
                if w is not None:
                    w.set_world_position(world_xyz)
            except Exception:
                continue

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

class OrthoVolumeWindow(QtWidgets.QMainWindow):
    """Orthogonal slice viewer for a 3D NIfTI volume with three views in one row."""
    def __init__(self, volume_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Volume: {Path(volume_path).name}")
        self.resize(1200, 420)

        # Central layout with three VTK widgets
        central = QtWidgets.QWidget(self)
        hbox = QtWidgets.QHBoxLayout(central)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(2)
        self.setCentralWidget(central)

    # Read the NIfTI image
        reader = vtkNIFTIImageReader()
        reader.SetFileName(volume_path)
        reader.Update()
        img = reader.GetOutput()
        self._img = img

    # Auto window/level mapping from image range
        rng = img.GetScalarRange()
        wl = vtkImageMapToWindowLevelColors()
        wl.SetInputData(img)
        w = float(rng[1] - rng[0]) if rng[1] > rng[0] else 1.0
        l = float(0.5 * (rng[0] + rng[1]))
        wl.SetWindow(w)
        wl.SetLevel(l)
        wl.Update()
        self._wl = wl

        # Build ijk->world from NIfTI header if available (sform preferred, then qform), else fallback
        self._M_ijk_to_world: Optional[vtkMatrix4x4] = None
        self._M_world_to_ijk: Optional[vtkMatrix4x4] = None
        try:
            m_s = reader.GetSFormMatrix()
        except Exception:
            m_s = None
        try:
            m_q = reader.GetQFormMatrix()
        except Exception:
            m_q = None
        if m_s is not None:
            self._M_ijk_to_world = m_s
        elif m_q is not None:
            self._M_ijk_to_world = m_q
        else:
            # Fallback: origin/spacing/direction
            origin = img.GetOrigin(); spacing = img.GetSpacing()
            axes = None
            try:
                axes = img.GetDirectionMatrix()
            except Exception:
                axes = None
            M = vtkMatrix4x4(); M.Identity()
            if axes is not None:
                for i in range(3):
                    for j in range(3):
                        M.SetElement(i, j, axes.GetElement(i, j) * spacing[j])
            else:
                for i in range(3):
                    M.SetElement(i, i, spacing[i])
            for i in range(3):
                M.SetElement(i, 3, origin[i])
            self._M_ijk_to_world = M
        # Invert to get world->ijk
        try:
            self._M_world_to_ijk = vtkMatrix4x4()
            vtkMatrix4x4.Invert(self._M_ijk_to_world, self._M_world_to_ijk)
        except Exception:
            self._M_world_to_ijk = None

        # Extent and center slice indices
        x0, x1, y0, y1, z0, z1 = img.GetExtent()
        self._iex = (x0, x1, y0, y1, z0, z1)
        cx = int(0.5 * (x0 + x1))
        cy = int(0.5 * (y0 + y1))
        cz = int(0.5 * (z0 + z1))
        self._cx, self._cy, self._cz = cx, cy, cz

        # Helper to construct a reslice for each orientation
        def make_reslice(orientation: str, index: int):
            res = vtkImageReslice()
            res.SetInputConnection(self._wl.GetOutputPort())
            res.SetInterpolationModeToLinear()
            res.SetOutputDimensionality(2)
            axes = vtkMatrix4x4()
            # Start as identity
            for i in range(4):
                for j in range(4):
                    axes.SetElement(i, j, 1.0 if i == j else 0.0)
            if orientation == 'axial':
                # Output X->input X, Output Y->input Y, normal->input Z
                axes.SetElement(2, 3, float(index))  # translate along input Z
            elif orientation == 'coronal':
                # Output X->input X, Output Y->input Z, normal->input Y
                axes.SetElement(1, 1, 0.0); axes.SetElement(2, 2, 0.0)
                axes.SetElement(1, 2, 1.0); axes.SetElement(2, 1, 1.0)
                axes.SetElement(1, 3, float(index))  # translate along input Y
            elif orientation == 'sagittal':
                # Output X->input Y, Output Y->input Z, normal->input X
                axes.SetElement(0, 0, 0.0); axes.SetElement(1, 1, 0.0)
                axes.SetElement(0, 1, 1.0); axes.SetElement(1, 2, 1.0)
                axes.SetElement(2, 0, 1.0)
                axes.SetElement(0, 3, float(index))  # translate along input X
            res.SetResliceAxes(axes)
            res.Update()
            return res, axes

        self._views = {}
        configs = [('axial', cz), ('coronal', cy), ('sagittal', cx)]
        for name, idx in configs:
            w = QVTKRenderWindowInteractor(central)
            hbox.addWidget(w, 1)
            ren = vtkRenderer(); ren.SetBackground(0, 0, 0)
            rw = w.GetRenderWindow(); rw.AddRenderer(ren)
            iren = rw.GetInteractor()
            try:
                iren.SetInteractorStyle(vtkInteractorStyleImage())
            except Exception:
                pass
            reslice, axes = make_reslice(name, idx)
            out_img = reslice.GetOutput()
            extent_out = out_img.GetExtent()  # 2D image: (0..Nx-1, 0..Ny-1, 0, 0)
            actor = vtkImageActor(); actor.SetInputData(out_img)
            ren.AddActor(actor)
            ren.ResetCamera(); rw.Render()
            # Center camera on this slice view after initial image is ready
            self._center_view(name)

            # Store view components
            self._views[name] = {
                'widget': w,
                'renderer': ren,
                'iren': iren,
                'reslice': reslice,
                'axes': axes,
                'actor': actor,
                'extent': extent_out,
            }

            # Click picking: update slices to picked world position
            picker = vtkPropPicker()
            def _make_click_handler(local_picker, local_ren):
                def _on_left_button(_obj, _evt):
                    try:
                        x, y = _obj.GetEventPosition()
                    except Exception:
                        return
                    ok = local_picker.Pick(x, y, 0, local_ren)
                    if not ok:
                        return
                    wx, wy, wz = local_picker.GetPickPosition()
                    ijk = self._world_to_index((wx, wy, wz))
                    if ijk is None:
                        return
                    ix, iy, iz = ijk
                    self._set_slices(ix, iy, iz)
                return _on_left_button
            iren.AddObserver("LeftButtonPressEvent", _make_click_handler(picker, ren))

            # Crosshair overlays (two lines) per view
            def build_crosshair():
                lv = vtkLineSource(); lh = vtkLineSource()
                mv = vtkPolyDataMapper(); mh = vtkPolyDataMapper()
                mv.SetInputConnection(lv.GetOutputPort()); mh.SetInputConnection(lh.GetOutputPort())
                av = vtkActor(); ah = vtkActor(); av.SetMapper(mv); ah.SetMapper(mh)
                for a in (av, ah):
                    a.GetProperty().SetColor(1.0, 1.0, 0.0)
                    a.GetProperty().SetLineWidth(1.5)
                ren.AddActor(av); ren.AddActor(ah)
                return lv, lh, av, ah
            lv, lh, av, ah = build_crosshair()
            self._views[name]['cross_v'] = (lv, av)
            self._views[name]['cross_h'] = (lh, ah)
            # Initialize crosshair positions for current indices
            self._update_crosshair_for_view(name)

            # Scroll wheel changes slice for this view
            def _make_wheel(delta_sign: int, vname: str):
                def _on_wheel(_obj, _evt):
                    ix, iy, iz = self._cx, self._cy, self._cz
                    x0, x1, y0, y1, z0, z1 = self._iex
                    if vname == 'axial':
                        iz = max(z0, min(z1, iz + delta_sign))
                    elif vname == 'coronal':
                        iy = max(y0, min(y1, iy + delta_sign))
                    elif vname == 'sagittal':
                        ix = max(x0, min(x1, ix + delta_sign))
                    self._set_slices(ix, iy, iz)
                return _on_wheel
            iren.AddObserver("MouseWheelForwardEvent", _make_wheel(+1, name))
            iren.AddObserver("MouseWheelBackwardEvent", _make_wheel(-1, name))

    def _center_view(self, name: str):
        v = self._views.get(name)
        if not v:
            return
        actor = v['actor']; ren: vtkRenderer = v['renderer']
        b = [0.0]*6
        actor.GetBounds(b)
        cx = 0.5*(b[0]+b[1]); cy = 0.5*(b[2]+b[3]); cz = 0.5*(b[4]+b[5])
        dx = abs(b[1]-b[0]); dy = abs(b[3]-b[2]); dz = abs(b[5]-b[4])
        cam = ren.GetActiveCamera()
        try:
            cam.SetParallelProjection(True)
        except Exception:
            pass
        dist = max(dx, dy, dz) * 2.0 + 1.0
        if name == 'axial':
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(cx, cy, cz + dist)
            cam.SetViewUp(0, 1, 0)
        elif name == 'coronal':
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(cx, cy + dist, cz)
            cam.SetViewUp(0, 0, 1)
        else:  # sagittal
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(cx + dist, cy, cz)
            cam.SetViewUp(0, 0, 1)
        try:
            cam.SetParallelScale(0.5*max(dx, dy))
        except Exception:
            pass
        ren.ResetCameraClippingRange()

    def _world_to_index(self, world_xyz: Tuple[float, float, float]) -> Optional[Tuple[int, int, int]]:
        """Convert a world coordinate (x,y,z) to image IJK indices (ints).
        Uses NIfTI sform/qform if available; falls back to origin/spacing/direction.
        """
        if self._img is None:
            return None
        wx, wy, wz = map(float, world_xyz)
        if self._M_world_to_ijk is not None:
            # Homogeneous multiply
            w = [wx, wy, wz, 1.0]
            ijk_h = [0.0, 0.0, 0.0, 1.0]
            for i in range(4):
                ijk_h[i] = sum(self._M_world_to_ijk.GetElement(i, j) * w[j] for j in range(4))
            ix = int(round(ijk_h[0])); iy = int(round(ijk_h[1])); iz = int(round(ijk_h[2]))
        else:
            # Fallback path (origin/spacing/direction)
            origin = self._img.GetOrigin()
            spacing = self._img.GetSpacing()
            R = None
            try:
                Rm = self._img.GetDirectionMatrix()
                if Rm is not None:
                    R = [[Rm.GetElement(i, j) for j in range(3)] for i in range(3)]
            except Exception:
                R = None
            dx = (wx - origin[0]); dy = (wy - origin[1]); dz = (wz - origin[2])
            if R is not None:
                vx = R[0][0]*dx + R[1][0]*dy + R[2][0]*dz
                vy = R[0][1]*dx + R[1][1]*dy + R[2][1]*dz
                vz = R[0][2]*dx + R[1][2]*dy + R[2][2]*dz
            else:
                vx, vy, vz = dx, dy, dz
            def safe_div(a, b):
                return a / b if abs(b) > 1e-12 else 0.0
            ix = int(round(safe_div(vx, spacing[0])))
            iy = int(round(safe_div(vy, spacing[1])))
            iz = int(round(safe_div(vz, spacing[2])))
        # Clamp to image extent
        x0, x1, y0, y1, z0, z1 = self._img.GetExtent()
        ix = max(x0, min(x1, ix))
        iy = max(y0, min(y1, iy))
        iz = max(z0, min(z1, iz))
        return ix, iy, iz

    def _set_slices(self, ix: int, iy: int, iz: int):
        """Update axial/coronal/sagittal slices to given indices and render."""
        self._cx, self._cy, self._cz = ix, iy, iz
        # Update reslice axes translations
        v = self._views
        if 'axial' in v:
            axes = v['axial']['axes']; axes.SetElement(2, 3, float(iz))
            v['axial']['reslice'].SetResliceAxes(axes); v['axial']['reslice'].Update()
            v['axial']['actor'].SetInputData(v['axial']['reslice'].GetOutput())
        if 'coronal' in v:
            axes = v['coronal']['axes']; axes.SetElement(1, 3, float(iy))
            v['coronal']['reslice'].SetResliceAxes(axes); v['coronal']['reslice'].Update()
            v['coronal']['actor'].SetInputData(v['coronal']['reslice'].GetOutput())
        if 'sagittal' in v:
            axes = v['sagittal']['axes']; axes.SetElement(0, 3, float(ix))
            v['sagittal']['reslice'].SetResliceAxes(axes); v['sagittal']['reslice'].Update()
            v['sagittal']['actor'].SetInputData(v['sagittal']['reslice'].GetOutput())
        # Update crosshair lines per view
        self._update_crosshair_for_view('axial')
        self._update_crosshair_for_view('coronal')
        self._update_crosshair_for_view('sagittal')
        # Render all
        for name in ('axial', 'coronal', 'sagittal'):
            if name in v:
                try:
                    v[name]['renderer'].GetRenderWindow().Render()
                except Exception:
                    pass

    def _update_crosshair_for_view(self, name: str):
        if name not in getattr(self, '_views', {}):
            return
        v = self._views[name]
        lv, av = v.get('cross_v', (None, None))
        lh, ah = v.get('cross_h', (None, None))
        if lv is None or lh is None:
            return
        # Helper: IJK -> world (using sform/qform if available)
        def ijk_to_world(i: float, j: float, k: float):
            M = getattr(self, '_M_ijk_to_world', None)
            if M is None:
                return float(i), float(j), float(k)
            vec = [float(i), float(j), float(k), 1.0]
            out = [0.0, 0.0, 0.0, 1.0]
            for r in range(4):
                out[r] = sum(M.GetElement(r, c) * vec[c] for c in range(4))
            return out[0], out[1], out[2]
        x0, x1, y0, y1, z0, z1 = self._iex
        ix, iy, iz = self._cx, self._cy, self._cz
        # Build endpoints per orientation in world coords
        if name == 'axial':
            p1 = ijk_to_world(ix, y0, iz); p2 = ijk_to_world(ix, y1, iz)  # vertical (Y)
            q1 = ijk_to_world(x0, iy, iz); q2 = ijk_to_world(x1, iy, iz)  # horizontal (X)
        elif name == 'coronal':
            p1 = ijk_to_world(ix, iy, z0); p2 = ijk_to_world(ix, iy, z1)  # vertical (Z)
            q1 = ijk_to_world(x0, iy, iz); q2 = ijk_to_world(x1, iy, iz)  # horizontal (X)
        else:  # sagittal
            p1 = ijk_to_world(ix, y0, iz); p2 = ijk_to_world(ix, y1, iz)  # vertical (Y)
            q1 = ijk_to_world(ix, iy, z0); q2 = ijk_to_world(ix, iy, z1)  # horizontal (Z)
        lv.SetPoint1(*p1); lv.SetPoint2(*p2)
        lh.SetPoint1(*q1); lh.SetPoint2(*q2)
        try:
            lv.Update(); lh.Update()
        except Exception:
            pass

    # Public API for linking from surface picks
    def set_world_position(self, world_xyz: Tuple[float, float, float]):
        ijk = self._world_to_index(world_xyz)
        if ijk is None:
            return
        self._set_slices(*ijk)


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
        p = QPainter(self)
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
def main(argv: List[str]):
    opts = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)
    win = Viewer(opts); win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main(sys.argv[1:])

