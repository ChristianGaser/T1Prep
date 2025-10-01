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

Requires: vtk (>=9), PySide6; nibabel (for GIFTI fallback + FreeSurfer morphs if VTK lacks vtkGIFTIReader).
"""
from __future__ import annotations
import argparse
import os
import sys
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


# --- Numpy ---
import numpy as np

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
    
    # Check for BIDS patterns
    for pattern in bids_patterns:
        if pattern in filename_lower:
            return True
    
    # Check for FreeSurfer patterns
    for pattern in freesurfer_patterns:
        if pattern in filename_lower:
            return False
    
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
    is_bids = detect_naming_scheme(overlay_filename)
    
    if is_bids:
        # BIDS naming: convert to midthickness surface
        # Example: sub-01_hemi-L_desc-thickness.shape.gii -> sub-01_hemi-L_space-MNI152NLin2009cAsym_desc-midthickness.surf.gii
        name_parts = overlay_path.stem.split('_')
        
        # Find hemisphere
        hemi_part = None
        for part in name_parts:
            if part.startswith('hemi-'):
                hemi_part = part
                break
        
        if hemi_part:
            hemi = hemi_part.split('-')[1]  # L or R
            # Build mesh filename
            base_parts = [p for p in name_parts if not p.startswith('hemi-') and not p.startswith('desc-')]
            base_name = '_'.join(base_parts)
            mesh_filename = f"{base_name}_{hemi_part}_space-MNI152NLin2009cAsym_desc-midthickness.surf.gii"
            return str(overlay_path.parent / mesh_filename)
    else:
        # FreeSurfer naming: convert overlay to central surface
        # Accept both with and without a subject token:
        #  - lh.thickness.name   -> lh.central.name.gii
        #  - lh.thickness        -> lh.central.gii
        name = overlay_path.name

        def _fs_overlay_to_mesh(nm: str) -> Optional[str]:
            hemi = None
            remaining = None
            if nm.startswith('lh.'):
                hemi = 'lh'; remaining = nm[3:]
            elif nm.startswith('rh.'):
                hemi = 'rh'; remaining = nm[3:]
            else:
                # Fallback: try parsing as hemi.kind.subject...
                parts_f = nm.split('.')
                if len(parts_f) >= 2 and parts_f[0] in ('lh', 'rh'):
                    hemi = parts_f[0]
                    remaining = '.'.join(parts_f[1:])
                else:
                    return None
            # remaining looks like: kind[.subject[.ext]]
            tokens = [t for t in remaining.split('.') if t]
            if not tokens:
                return None
            # Drop known scalar extensions from the end for subject parsing
            exts = {'gii', 'txt', 'mgh', 'mgz'}
            subj_tokens = tokens[1:]  # after overlay kind
            if subj_tokens and subj_tokens[-1].lower() in exts:
                subj_tokens = subj_tokens[:-1]
            # If there is no subject token, build lh.central.gii
            if not subj_tokens:
                return f"{hemi}.central.gii"
            base = '.'.join(subj_tokens)
            return f"{hemi}.central.{base}.gii"

        mesh_name = _fs_overlay_to_mesh(name)
        if mesh_name is None:
            return str(overlay_path)  # Could not parse; leave unchanged
        return str(overlay_path.parent / mesh_name)
    
    return str(overlay_path)  # Return original if conversion fails

def is_overlay_file(filename: str) -> bool:
    """
    Check if a filename appears to be an overlay file rather than a mesh file.
    
    Args:
        filename: The filename to check
        
    Returns:
        bool: True if it appears to be an overlay file
    """
    # Extract just the filename from the path
    filename_only = Path(filename).name
    filename_lower = filename_only.lower()
    
    # Check for FreeSurfer shape pattern: [l|r]h.shape_type[.subject][.ext]
    parts = filename_lower.split('.')
    mesh_types = ['central', 'pial', 'white', 'inflated', 'sphere', 'patch', 'mc', 'sqrtsulc']
    # Case A: lh.kind.subject (>=3 parts)
    if len(parts) >= 3 and parts[0] in ('lh', 'rh'):
        if parts[1] not in mesh_types:
            return True
    # Case B: lh.kind (exactly 2 parts, no subject) should also be considered overlay
    if len(parts) == 2 and parts[0] in ('lh', 'rh'):
        if parts[1] not in mesh_types:
            return True
    
    # Overlay file patterns
    overlay_patterns = [
        '_desc-thickness.', '_desc-pbt.',  # BIDS shape files
        '.annot',  # Annotation files
        '_label-',  # BIDS label files
        '.txt'  # Text overlay files
    ]
    
    for pattern in overlay_patterns:
        if pattern in filename_lower:
            return True
    
    return False

def detect_overlay_kind(filename: str) -> Optional[str]:
    """Detect overlay kind such as 'thickness' or 'pbt' from filename.

    Returns: 'thickness' | 'pbt' | None
    """
    name = Path(filename).name.lower()
    # BIDS style
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

    # --- Case 2: FreeSurfer morph data (thickness/curv/sulc, with or without extension) ---
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
                    '  • Single mesh: src/cat_viewsurf.py lh.central.name.gii\n'
                    '  • Single overlay: src/cat_viewsurf.py lh.thickness.name1\n'
                    '  • Multiple overlays (navigate with ←/→): src/cat_viewsurf.py lh.thickness.name1 lh.thickness.name2 ...',
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
    p.add_argument('-range','-r', dest='range', nargs=2, type=float, default=[0.0, -1.0])
    p.add_argument('-range-bkg','-rb', dest='range_bkg', nargs=2, type=float, default=[0.0, -1.0])
    p.add_argument('-clip','-cl', dest='clip', nargs=2, type=float, default=[0.0, -1.0])
    p.add_argument('-size','-sz', dest='size', nargs=2, type=int, default=list(DEFAULT_WINDOW_SIZE))
    p.add_argument('-title', dest='title')
    p.add_argument('-output','-save', dest='output')
    p.add_argument('-fontsize','-fs', dest='fontsize', type=int, default=0)
    p.add_argument('-opacity','-op', dest='opacity', type=float, default=0.8)
    p.add_argument('-stats', action='store_true', help='Deprecated: same as --title-mode stats when colorbar is shown')
    p.add_argument('--title-mode', dest='title_mode', choices=['shape','stats','none'], default='shape',
                   help='Colorbar title: shape (filename), stats, or none')
    p.add_argument('-inverse', action='store_true')
    p.add_argument('-colorbar','-cb', dest='colorbar', action='store_true')
    p.add_argument('-discrete','-dsc', dest='discrete', type=int, default=2,
                   help='Number of discrete color levels (0 to disable). Default: 2')
    p.add_argument('-log', action='store_true')
    p.add_argument('-white', action='store_true')
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
    p.add_argument('-debug', action='store_true')
    # External defaults file for viewer settings (key=value lines)
    p.add_argument('--defaults', dest='defaults', help='Path to a defaults file (key=value) to override built-in defaults')
    a = p.parse_args(argv)

    # Optionally load external defaults and apply only for values not explicitly provided on CLI
    def _parse_bool(s: str) -> bool:
        return str(s).strip().lower() in ('1','true','yes','on')

    def _parse_floats_csv(s: str, n_expected: int = None) -> Tuple[float, ...]:
        parts = [p for p in re.split(r'[;,,\s]+', str(s).strip()) if p]
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
                    cfg[key.strip()] = val.strip().strip("'\"")
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
            script_dir.parent.parent / 'cat_viewsurf_defaults.txt',
            script_dir.parent / 'cat_viewsurf_defaults.txt',
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
            overlays_from_pos = [x for x in non_mesh_inputs if is_overlay_file(x)]
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
# --- Viewer ---
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, options: Options, parent=None):
        super().__init__(parent)
        self.setObjectName("ViewerWindow")
        self.setWindowTitle("CAT ViewSurf")
        self.setMinimumSize(800, 600)
        # Central widget (VTK render window)
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)
        # Layouts
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.setLayout(self.main_layout)
        # Menu bar
        self.menu_bar = self.menuBar()
        # File menu
        self.file_menu = self.menu_bar.addMenu("&File")
        self.file_menu.addAction(self._create_action("&Open...", self.open_file, "Ctrl+O"))
        self.file_menu.addAction(self._create_action("&Save Screenshot...", self.save_screenshot, "Ctrl+S"))
        self.file_menu.addSeparator()
        self.file_menu.addAction(self._create_action("&Exit", self.close, "Ctrl+Q"))
        # View menu
        self.view_menu = self.menu_bar.addMenu("&View")
        self.view_menu.addAction(self._create_action("&Reset View", self.reset_view, "Ctrl+R"))
        self.view_menu.addAction(self._create_action("&Toggle Control Panel", self.toggle_control_panel, "Ctrl+T"))
        # Help menu
        self.help_menu = self.menu_bar.addMenu("&Help")
        self.help_menu.addAction(self._create_action("&About", self.show_about))
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        # VTK Renderer
        self.renderer = vtkRenderer()
        self.render_window = vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window_interactor = vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)
        # Custom interactor style
        self.interactor_style = CustomInteractorStyle()
        self.interactor_style.SetViewer(self)
        self.render_window_interactor.SetInteractorStyle(self.interactor_style)
        # Add VTK render window to the main layout
        self.main_layout.addWidget(QVTKRenderWindowInteractor(self.render_window_interactor))
        # Control panel (dockable widget)
        self.control_panel = ControlPanel(options, self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.control_panel)
        # Load initial mesh/overlay
        self.set_options(options)
        # Show the window
        self.show()

    def _create_action(self, text: str, slot, shortcut: str = None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        return action

    def closeEvent(self, event):
        """Handle close event."""
        self.render_window_interactor.GetInteractor().GetRenderWindow().Finalize()
        event.accept()

    def set_options(self, options: Options):
        """Set options and load the corresponding mesh and overlays."""
        self.options = options
        # Update window title
        if options.title:
            self.setWindowTitle(f"CAT ViewSurf - {options.title}")
        else:
            self.setWindowTitle("CAT ViewSurf")
        # Clear existing actors
        self.renderer.RemoveAllActors()
        # Load mesh
        if options.mesh_left:
            self._load_mesh(options.mesh_left)
        # Load overlays
        if options.overlay:
            self._load_overlay(options.overlay)
        if options.overlays:
            for overlay in options.overlays:
                self._load_overlay(overlay)
        # Update control panel
        self.control_panel.set_options(options)
        # Render
        self.render()

    def _load_mesh(self, mesh_file: str):
        """Load a mesh file (GIFTI or FreeSurfer) and add it to the renderer."""
        try:
            mesh_polydata = read_gifti_mesh(mesh_file)
            actor = vtkActor()
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(mesh_polydata)
            actor.SetMapper(mapper)
            self.renderer.AddActor(actor)
            self.status_bar.showMessage(f"Loaded mesh: {mesh_file}")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading mesh: {str(e)}")

    def _load_overlay(self, overlay_file: str):
        """Load an overlay file (scalar data) and apply it to the last loaded mesh."""
        try:
            overlay_data = read_scalars(overlay_file)
            # Find the last added actor (mesh)
            actor = self.renderer.GetActors().GetLastActor()
            if actor is None:
                raise RuntimeError("No mesh actor found to apply the overlay.")
            # Create a lookup table
            lut = get_lookup_table(self.options.colormap, 1.0)
            # Map the overlay data to colors
            mapper = actor.GetMapper()
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToMapScalars()
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(overlay_data.GetRange())
            # Update the actor's property
            actor.GetProperty().SetOpacity(self.options.opacity)
            actor.GetProperty().SetPointSize(2.0)
            # Add color bar
            self._add_color_bar(lut, overlay_file)
            self.status_bar.showMessage(f"Loaded overlay: {overlay_file}")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading overlay: {str(e)}")

    def _add_color_bar(self, lut, overlay_file: str):
        """Add a color bar actor to the renderer."""
        color_bar = vtkScalarBarActor()
        color_bar.SetLookupTable(lut)
        color_bar.SetTitle(os.path.basename(overlay_file))
        color_bar.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        color_bar.SetPosition(0.85, 0.1)
        color_bar.SetWidth(0.03)
        color_bar.SetHeight(0.8)
        self.renderer.AddActor(color_bar)

    def render(self):
        """Render the scene."""
        self.render_window.Render()
        self.status_bar.showMessage("Render complete")

    def open_file(self):
        """Open a mesh or overlay file."""
        options = self.options
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "", "GIFTI Files (*.gii);;All Files (*)")
        if filename:
            if is_gifti_mesh_file(filename):
                # Mesh file
                self.set_options(Options(mesh_left=filename, overlays=options.overlays, overlay=options.overlay,
                                         overlay_bkg=options.overlay_bkg, volume=options.volume,
                                         range=options.range, range_bkg=options.range_bkg, clip=options.clip,
                                         size=options.size, title=options.title, output=options.output,
                                         fontsize=options.fontsize, opacity=options.opacity, stats=options.stats,
                                         title_mode=options.title_mode, inverse=options.inverse, colorbar=options.colorbar,
                                         discrete=options.discrete, log=options.log, white=options.white,
                                         panel=options.panel, colormap=options.colormap, debug=options.debug,
                                         fix_scaling=options.fix_scaling))
            else:
                # Overlay file
                self.set_options(Options(mesh_left=options.mesh_left, overlays=options.overlays, overlay=filename,
                                         overlay_bkg=options.overlay_bkg, volume=options.volume,
                                         range=options.range, range_bkg=options.range_bkg, clip=options.clip,
                                         size=options.size, title=options.title, output=options.output,
                                         fontsize=options.fontsize, opacity=options.opacity, stats=options.stats,
                                         title_mode=options.title_mode, inverse=options.inverse, colorbar=options.colorbar,
                                         discrete=options.discrete, log=options.log, white=options.white,
                                         panel=options.panel, colormap=options.colormap, debug=options.debug,
                                         fix_scaling=options.fix_scaling))

    def save_screenshot(self):
        """Save a screenshot of the current view."""
        options = self.options
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png);;All Files (*)")
        if filename:
            try:
                writer = vtkPNGWriter()
                writer.SetFileName(filename)
                writer.SetInputConnection(self.render_window.GetImageFilter().GetOutputPort())
                writer.Write()
                self.status_bar.showMessage(f"Screenshot saved: {filename}")
            except Exception as e:
                self.status_bar.showMessage(f"Error saving screenshot: {str(e)}")

    def reset_view(self):
        """Reset the camera to the default view."""
        self.renderer.ResetCamera()
        self.render()

    def toggle_control_panel(self):
        """Toggle the visibility of the control panel."""
        if self.control_panel.isVisible():
            self.control_panel.hide()
        else:
            self.control_panel.show()

    def show_about(self):
        """Show the About dialog."""
        QtWidgets.QMessageBox.about(self, "About CAT ViewSurf",
            "<h2>CAT ViewSurf</h2>"
            "<p>PySide6 + VTK port with right-side control panel</p>"
            "<p>For more information, visit the <a href='https://github.com/catmaid/CATMAID'>CATMAID GitHub page</a>.</p>"
            "<p>Version: 1.0</p>"
        )

# --- Control Panel ---
class ControlPanel(QtWidgets.QDockWidget):
    def __init__(self, options: Options, viewer: Viewer, parent=None):
        super().__init__(parent)
        self.setObjectName("ControlPanel")
        self.setWindowTitle("Control Panel")
        self.setMinimumWidth(250)
        self.viewer = viewer
        # Central widget
        self.central_widget = QtWidgets.QWidget(self)
        self.setWidget(self.central_widget)
        # Layout
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.setLayout(self.layout)
        # Range slider
        self.range_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.range_slider.setRange(0, 100)
        self.range_slider.setValue(100)
        self.range_slider.valueChanged.connect(self.update_range)
        self.layout.addWidget(QtWidgets.QLabel("Scalar Range:"))
        self.layout.addWidget(self.range_slider)
        # Clip slider
        self.clip_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.clip_slider.setRange(0, 100)
        self.clip_slider.setValue(0)
        self.clip_slider.valueChanged.connect(self.update_clip)
        self.layout.addWidget(QtWidgets.QLabel("Clip Range:"))
        self.layout.addWidget(self.clip_slider)
        # Opacity slider
        self.opacity_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        self.layout.addWidget(QtWidgets.QLabel("Opacity:"))
        self.layout.addWidget(self.opacity_slider)
        # Color map selector
        self.colormap_combo = QtWidgets.QComboBox(self)
        self.colormap_combo.addItem("JET")
        self.colormap_combo.addItem("HOT")
        self.colormap_combo.addItem("FIRE")
        self.colormap_combo.addItem("BIPOLAR")
        self.colormap_combo.addItem("GRAY")
        self.colormap_combo.currentIndexChanged.connect(self.update_colormap)
        self.layout.addWidget(QtWidgets.QLabel("Color Map:"))
        self.layout.addWidget(self.colormap_combo)
        # Title mode selector
        self.title_mode_combo = QtWidgets.QComboBox(self)
        self.title_mode_combo.addItem("Shape")
        self.title_mode_combo.addItem("Stats")
        self.title_mode_combo.addItem("None")
        self.title_mode_combo.currentIndexChanged.connect(self.update_title_mode)
        self.layout.addWidget(QtWidgets.QLabel("Title Mode:"))
        self.layout.addWidget(self.title_mode_combo)
        # Stats checkbox
        self.stats_checkbox = QtWidgets.QCheckBox("Show Stats", self)
        self.stats_checkbox.stateChanged.connect(self.update_stats)
        self.layout.addWidget(self.stats_checkbox)
        # Inverse checkbox
        self.inverse_checkbox = QtWidgets.QCheckBox("Inverse Colors", self)
        self.inverse_checkbox.stateChanged.connect(self.update_inverse)
        self.layout.addWidget(self.inverse_checkbox)
        # Colorbar checkbox
        self.colorbar_checkbox = QtWidgets.QCheckBox("Show Colorbar", self)
        self.colorbar_checkbox.stateChanged.connect(self.update_colorbar)
        self.layout.addWidget(self.colorbar_checkbox)
        # Discrete levels spinbox
        self.discrete_spinbox = QtWidgets.QSpinBox(self)
        self.discrete_spinbox.setRange(0, 256)
        self.discrete_spinbox.setValue(0)
        self.discrete_spinbox.valueChanged.connect(self.update_discrete)
        self.layout.addWidget(QtWidgets.QLabel("Discrete Levels:"))
        self.layout.addWidget(self.discrete_spinbox)
        # Log scale checkbox
        self.log_checkbox = QtWidgets.QCheckBox("Log Scale", self)
        self.log_checkbox.stateChanged.connect(self.update_log)
        self.layout.addWidget(self.log_checkbox)
        # White background checkbox
        self.white_checkbox = QtWidgets.QCheckBox("White Background", self)
        self.white_checkbox.stateChanged.connect(self.update_white)
        self.layout.addWidget(self.white_checkbox)
        # Fix scaling checkbox
        self.fix_scaling_checkbox = QtWidgets.QCheckBox("Fix Scaling", self)
        self.fix_scaling_checkbox.stateChanged.connect(self.update_fix_scaling)
        self.layout.addWidget(self.fix_scaling_checkbox)
        # Debug checkbox
        self.debug_checkbox = QtWidgets.QCheckBox("Debug Mode", self)
        self.debug_checkbox.stateChanged.connect(self.update_debug)
        self.layout.addWidget(self.debug_checkbox)
        # Apply button
        self.apply_button = QtWidgets.QPushButton("Apply", self)
        self.apply_button.clicked.connect(self.apply_changes)
        self.layout.addWidget(self.apply_button)
        # Set initial values
        self.set_options(options)

    def set_options(self, options: Options):
        """Set options and update the control panel widgets."""
        self.options = options
        # Scalar range
        self.range_slider.setValue(int(options.range[1] * 100))
        # Clip range
        self.clip_slider.setValue(int(options.clip[0] * 100))
        # Opacity
        self.opacity_slider.setValue(int(options.opacity * 100))
        # Color map
        index = options.colormap if options.colormap < 5 else 0
        self.colormap_combo.setCurrentIndex(index)
        # Title mode
        title_mode_index = {'shape': 0, 'stats': 1, 'none': 2}.get(options.title_mode, 0)
        self.title_mode_combo.setCurrentIndex(title_mode_index)
        # Stats
        self.stats_checkbox.setChecked(options.stats)
        # Inverse colors
        self.inverse_checkbox.setChecked(options.inverse)
        # Colorbar
        self.colorbar_checkbox.setChecked(options.colorbar)
        # Discrete levels
        self.discrete_spinbox.setValue(options.discrete)
        # Log scale
        self.log_checkbox.setChecked(options.log)
        # White background
        self.white_checkbox.setChecked(options.white)
        # Fix scaling
        self.fix_scaling_checkbox.setChecked(options.fix_scaling)
        # Debug mode
        self.debug_checkbox.setChecked(options.debug)

    def update_range(self, value: int):
        """Update the scalar range based on the slider value."""
        min_val = -1.0
        max_val = 1.0
        range_val = (min_val + (max_val - min_val) * value / 100.0, max_val)
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=range_val, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_clip(self, value: int):
        """Update the clip range based on the slider value."""
        min_val = -1.0
        max_val = 1.0
        clip_val = (min_val + (max_val - min_val) * value / 100.0, max_val)
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=clip_val,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_opacity(self, value: int):
        """Update the opacity based on the slider value."""
        opacity_val = value / 100.0
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=opacity_val, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_colormap(self, index: int):
        """Update the color map based on the combo box selection."""
        colormap_val = index if index < 5 else 0
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=colormap_val, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_title_mode(self, index: int):
        """Update the title mode based on the combo box selection."""
        title_mode_val = {0: 'shape', 1: 'stats', 2: 'none'}.get(index, 'shape')
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=title_mode_val, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_stats(self, state: int):
        """Update the stats display option."""
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=bool(state),
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_inverse(self, state: int):
        """Update the inverse colors option."""
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=bool(state), colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_colorbar(self, state: int):
        """Update the colorbar visibility option."""
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=bool(state),
                                        discrete=self.options.discrete, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_discrete(self, value: int):
        """Update the number of discrete color levels."""
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=value, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_log(self, state: int):
        """Update the log scale option."""
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=bool(state), white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_white(self, state: int):
        """Update the white background option."""
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=self.options.log, white=bool(state),
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=self.options.fix_scaling))

    def update_fix_scaling(self, state: int):
        """Update the fix scaling option."""
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=self.options.debug,
                                        fix_scaling=bool(state)))

    def update_debug(self, state: int):
        """Update the debug mode option."""
        self.viewer.set_options(Options(mesh_left=self.options.mesh_left, overlays=self.options.overlays, overlay=self.options.overlay,
                                        overlay_bkg=self.options.overlay_bkg, volume=self.options.volume,
                                        range=self.options.range, range_bkg=self.options.range_bkg, clip=self.options.clip,
                                        size=self.options.size, title=self.options.title, output=self.options.output,
                                        fontsize=self.options.fontsize, opacity=self.options.opacity, stats=self.options.stats,
                                        title_mode=self.options.title_mode, inverse=self.options.inverse, colorbar=self.options.colorbar,
                                        discrete=self.options.discrete, log=self.options.log, white=self.options.white,
                                        panel=self.options.panel, colormap=self.options.colormap, debug=bool(state),
                                        fix_scaling=self.options.fix_scaling))

# ---- Entrypoint ----
def main(argv: List[str]):
    opts = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)
    win = Viewer(opts); win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main(sys.argv[1:])
