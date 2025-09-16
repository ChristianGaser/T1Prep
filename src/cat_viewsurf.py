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
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# --- Qt setup (PySide6 only) ---
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut

# Qt compatibility shims
ORIENT_H = Qt.Orientation.Horizontal
DOCK_RIGHT = Qt.DockWidgetArea.RightDockWidgetArea
DOCK_LEFT = Qt.DockWidgetArea.LeftDockWidgetArea


# --- Numpy ---
import numpy as np

# --- Import naming utilities ---
import sys
sys.path.append(str(Path(__file__).parent))
from utils import load_namefile, get_filenames

# --- VTK imports (module-accurate for common wheels) ---
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
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
from vtkmodules.vtkIOImage import vtkPNGWriter

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
        # FreeSurfer naming: convert to central surface
        # Example: lh.thickness.name -> lh.central.name.gii
        name = overlay_path.name
        
        # Extract hemisphere and base name
        if name.startswith('lh.'):
            hemi = 'lh'
            # Remove 'lh.' and find the base name after the overlay type
            remaining = name[3:]  # Remove 'lh.'
            parts = remaining.split('.')
            if len(parts) >= 2:
                # Skip the first part (thickness, pbt, etc.) and use the rest as base name
                # Remove file extension if present
                base_parts = parts[1:]
                if base_parts and base_parts[-1] in ['txt', 'gii']:
                    base_parts = base_parts[:-1]
                base_name = '.'.join(base_parts) if base_parts else parts[1]
            else:
                base_name = remaining
        elif name.startswith('rh.'):
            hemi = 'rh'
            # Remove 'rh.' and find the base name after the overlay type
            remaining = name[3:]  # Remove 'rh.'
            parts = remaining.split('.')
            if len(parts) >= 2:
                # Skip the first part (thickness, pbt, etc.) and use the rest as base name
                # Remove file extension if present
                base_parts = parts[1:]
                if base_parts and base_parts[-1] in ['txt', 'gii']:
                    base_parts = base_parts[:-1]
                base_name = '.'.join(base_parts) if base_parts else parts[1]
            else:
                base_name = remaining
        else:
            # Try to extract from filename
            parts = name.split('.')
            if len(parts) >= 3:
                hemi = parts[0]
                base_name = '.'.join(parts[2:])  # Skip the middle part (thickness, pbt, etc.)
            else:
                return str(overlay_path)  # Return original if can't parse
        
        # Build mesh filename
        mesh_filename = f"{hemi}.central.{base_name}.gii"
        return str(overlay_path.parent / mesh_filename)
    
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
    
    # Check for FreeSurfer shape pattern: [l|r]h.shape_type.name (with or without extension)
    import re
    parts = filename_lower.split('.')
    
    # Check if it matches the pattern [l|r]h.shape_type.name
    if len(parts) >= 3 and parts[0] in ['lh', 'rh']:
        # Check if it's not a mesh file (central, pial, white, etc.)
        mesh_types = ['central', 'pial', 'white', 'inflated', 'sphere', 'patch', 'mc', 'sqrtsulc']
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
            'u','U','d','D','l','L','r','R','o','O','b','B','g','G','h','H','Left','Right'
        }
    def SetRenderer(self, ren: vtkRenderer): self._renderer = ren
    def SetViewer(self, viewer): self._viewer = viewer
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

# ---- Options & CLI ----
@dataclass
class Options:
    mesh_left: str
    overlay: Optional[str] = None
    overlays: List[str] = None  # Multiple overlays
    overlay_bkg: Optional[str] = None
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
        'inputs', nargs='+',
        help='Mesh or overlay(s). If multiple values are provided, they are treated as overlays.'
    )
    p.add_argument('-overlay','-ov', dest='overlay', help='Overlay scalars (.gii, FreeSurfer morph, or text)')
    p.add_argument('-overlays', dest='overlays', nargs='+', help='Multiple overlay files for navigation')
    p.add_argument('-bkg', dest='overlay_bkg', help='Background scalars for curvature shading (.gii or text)')
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
    p.add_argument('-discrete','-dsc', dest='discrete', type=int, default=0)
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
    a = p.parse_args(argv)

    cm = JET
    if a.fire: cm = FIRE
    if a.bipolar: cm = BIPOLAR
    if a.c1: cm = C1
    if a.c2: cm = C2
    if a.c3: cm = C3

    d = a.discrete
    if d < 0 or d > 4:
        p.error("Parameter -discrete/-dsc should be 0..4")
    if d:
        d = int(math.pow(2, d + 2))  # 32,16,8,4

    # Derive mesh/overlay list from positional inputs
    pos_inputs: List[str] = list(a.inputs)
    overlays_from_pos: List[str] = []
    if len(pos_inputs) == 1:
        # Single input can be either a mesh or an overlay. Detect overlay and derive mesh.
        single = pos_inputs[0]
        if is_overlay_file(single):
            try:
                mesh_left_resolved = convert_filename_to_mesh(single)
            except Exception:
                mesh_left_resolved = single
            # If it's an overlay, record it unless other overlay flags are used
            overlay_single_from_pos = single
        else:
            mesh_left_resolved = single
            overlay_single_from_pos = None
    else:
        overlays_from_pos = pos_inputs
        try:
            mesh_left_resolved = convert_filename_to_mesh(overlays_from_pos[0])
        except Exception:
            # Fall back to first argument as-is if conversion fails
            mesh_left_resolved = overlays_from_pos[0]

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
        overlay=overlay_single_final,
        overlays=overlay_list_final,
        overlay_bkg=a.overlay_bkg,
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
        # Overlay picker
        self.overlay_path = QtWidgets.QLineEdit(); self.overlay_btn = QtWidgets.QPushButton("…")
        ov_box = QtWidgets.QHBoxLayout(); ov_box.addWidget(self.overlay_path); ov_box.addWidget(self.overlay_btn)
        form.addRow("Overlay", self._wrap(ov_box))
        # Toggles
        self.cb_colorbar = QtWidgets.QCheckBox("Show colorbar")
        # Title mode selector (shape | stats | none)
        self.title_mode = QtWidgets.QComboBox(); self.title_mode.addItems(["shape","stats","none"])
        self.cb_inverse = QtWidgets.QCheckBox("Inverse (flip sign)")
        self.cb_fix_scaling = QtWidgets.QCheckBox("Fix scaling")
        # Put Show colorbar and Colorbar title on one row (aligned with other checkboxes)
        row = QtWidgets.QHBoxLayout(); row.setContentsMargins(0,0,0,0); row.setSpacing(8)
        row.addWidget(self.cb_colorbar)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Colorbar title"))
        row.addWidget(self.title_mode)
        form.addRow(self._wrap(row))
        form.addRow(self.cb_inverse)
        form.addRow(self.cb_fix_scaling)
        self.layout.addLayout(form)
        # Action buttons
        btns = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.reset_btn = QtWidgets.QPushButton("Reset view")
        btns.addWidget(self.apply_btn); btns.addWidget(self.reset_btn)
        self.layout.addLayout(btns)
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
        self.title_mode.setEnabled(enabled and self.cb_colorbar.isChecked())
        # Keep title selector in sync when user toggles colorbar
        try:
            self.cb_colorbar.toggled.disconnect()
        except Exception:
            pass
        self.cb_colorbar.toggled.connect(lambda v: self.title_mode.setEnabled(bool(v) and self.cb_colorbar.isEnabled()))
        # Inverse control
        self.cb_inverse.setEnabled(enabled)

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
    def _setup_view_menu(self):
        menubar = self.menuBar()
        menu = menubar.addMenu("View")
        act = QAction("Show Controls", self)
        act.setCheckable(True)
        act.setChecked(self.opts.panel)
        act.triggered.connect(self._toggle_controls)
        menu.addAction(act)
        self.act_show_controls = act

    def _toggle_controls(self, checked: bool):
        if not hasattr(self, 'dock_controls'):
            return
        dock = self.dock_controls
        current = dock.isVisible()
        if checked == current:
            return  # nothing to do
    
        # Simply show/hide the dock without resizing the window
        # The dock will overlay on the right side of the window
        if checked:
            dock.setFloating(True)  # Ensure it stays floating
            dock.show()
        else:
            dock.hide()


    def _on_dock_visibility_changed(self, visible: bool):
        if hasattr(self, 'act_show_controls'):
            self.act_show_controls.blockSignals(True)
            self.act_show_controls.setChecked(visible)
            self.act_show_controls.blockSignals(False)
    
        # No window resizing needed - dock overlays on the right side


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, opts: Options):
        super().__init__()
        self.opts = opts
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
            try:
                sym = self.iren.GetKeySym()
            except Exception:
                sym = None
            self._handle_key(sym)
        self.iren.AddObserver("KeyPressEvent", _on_keypress)

        # Load surfaces (LH + optional RH)
        # Check if the input is an overlay file or mesh file
        input_path = Path(opts.mesh_left)
        if not input_path.exists(): 
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Determine if input is an overlay file
        if is_overlay_file(str(input_path)):
            # Input is an overlay file, find the corresponding mesh
            mesh_path = convert_filename_to_mesh(str(input_path))
            mesh_path_obj = Path(mesh_path)
            if not mesh_path_obj.exists():
                raise FileNotFoundError(f"Corresponding mesh file not found: {mesh_path}")
            left_mesh_path = mesh_path_obj
            # Set the overlay to the original input file
            opts.overlay = str(input_path)
        else:
            # Input is a mesh file
            left_mesh_path = input_path
        
        self.poly_l = read_gifti_mesh(str(left_mesh_path))
        self.poly_r = None
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
        self._shift_y_to(self.poly_l)
        if self.poly_r is not None:
            self._shift_y_to(self.poly_r)

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
        # Apply discrete bands to overlay LUTs if requested
        self._apply_discrete_to_overlay_lut(self.lut_overlay_l)
        self._apply_discrete_to_overlay_lut(self.lut_overlay_r)
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
        if self.poly_r is not None:
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

        # Colorbar (lazy create)
        self.scalar_bar = None
        if opts.colorbar:
            self._ensure_colorbar()

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
        # Set initial overlay path
        if self.overlay_list:
            self.ctrl.overlay_path.setText(self.overlay_list[0])
            self._update_overlay_info()
        else:
            self.ctrl.overlay_path.setText(self.opts.overlay or "")
        self.ctrl.cb_colorbar.setChecked(self.opts.colorbar)
        # Initialize title mode combo
        self.ctrl.title_mode.setCurrentText(self.opts.title_mode)
        self.ctrl.cb_inverse.setChecked(self.opts.inverse)
        self.ctrl.cb_fix_scaling.setChecked(self.opts.fix_scaling)
        
        # Enable/disable overlay controls based on whether overlay is loaded
        has_overlay = (self.overlay_list or self.opts.overlay) and self.scal_l is not None
        self.ctrl.set_overlay_controls_enabled(has_overlay)
        # Ensure fix scaling checkbox state reflects current overlay count/availability
        self._enforce_fix_scaling_policy()
    
        # Signals
        self.ctrl.apply_btn.clicked.connect(self._apply_controls)
        self.ctrl.reset_btn.clicked.connect(self._reset_camera)
        self.ctrl.overlay_btn.clicked.connect(self._pick_overlay)
    
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
    
        # View menu + keyboard shortcut (uses QAction shim)
        menubar = self.menuBar()
        menu = menubar.addMenu("View")
        act = QAction("Show Controls", self)
        act.setCheckable(True)
        act.setChecked(self.opts.panel)
        # Do not attach a shortcut to the QAction to avoid double-triggering with QShortcut
        self.addAction(act)               # register action (no keybinding)
        act.triggered.connect(self._toggle_controls)
        menu.addAction(act)
        self.act_show_controls = act

        # Add a direct QShortcut to toggle dock visibility reliably with one key press
        # This avoids any interference from focused VTK interactor or QAction state.
        # Shortcut on main window (Ctrl/Command+D)
        self._dock_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        try:
            self._dock_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        except Exception:
            self._dock_shortcut.setContext(Qt.ApplicationShortcut)
        self._dock_shortcut.activated.connect(lambda d=dock: self._toggle_controls(not d.isVisible()))

    def _update_status_message(self, controls_visible: bool):
        """Show a small hint about the controls shortcut in the window status bar."""
        sb = self.statusBar()
        if controls_visible:
            sb.showMessage("Controls visible — press Ctrl/Cmd+D to hide")
        else:
            sb.showMessage("Controls hidden — press Ctrl/Cmd+D to show")
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
            self._prev_overlay(); return
        if s == 'Right':
            self._next_overlay(); return
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
            print("KEYS: u/d/l/r rotate, b flip, o reset, w/s wireframe/shaded, g screenshot, ←/→ overlay navigation"); return

    def _reset_camera(self):
        self.ren.ResetCamera(); self.ren.GetActiveCamera().Zoom(2.0); self.rw.Render()

    # --- Geometry normalization helper ---
    def _shift_y_to(self, poly: vtkPolyData, to_value: float = -100.0):
        """Shift mesh in Y so its minimum Y aligns with to_value.

        Keeps montage layout consistent across mesh switches by anchoring meshes
        to a common Y-origin. This mirrors the normalization used at startup.
        """
        if poly is None:
            return
        b = [0.0]*6
        poly.GetBounds(b)
        y_shift = to_value - b[2]
        if abs(y_shift) < 1e-9:
            return
        pts = poly.GetPoints()
        if pts is None:
            return
        n = pts.GetNumberOfPoints()
        for i in range(n):
            x, y, z = pts.GetPoint(i)
            pts.SetPoint(i, x, y + y_shift, z)
        poly.SetPoints(pts)

    # --- LUT helpers ---
    def _apply_discrete_to_overlay_lut(self, lut: vtkLookupTable):
        """Flatten LUT into discrete bands if -discrete is enabled.

        This mimics the discrete overlay behavior of the C++ implementation by
        sampling the color at the start of each block and reusing it across the
        block, leaving boundary indices as-is (they can be used as visual gaps
        in the colorbar logic).
        """
        steps = int(getattr(self.opts, 'discrete', 0) or 0)
        if steps <= 0:
            return
        for i in range(256):
            base = (i // steps) * steps
            if base > 255:
                base = 255
            r, g, b, a = lut.GetTableValue(base)
            if (i % steps) != 0:
                lut.SetTableValue(i, r, g, b, a)

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
        start_dir = self.ctrl.overlay_path.text() or str(Path(self.opts.mesh_left).parent)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose overlay", start_dir)
        if path: self.ctrl.overlay_path.setText(path)

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
            self.ctrl.overlay_path.setText(current_overlay)
            # Update window title to show current overlay
            overlay_name = Path(current_overlay).name
            if len(self.overlay_list) > 1:
                self.setWindowTitle(f"{overlay_name} ({self.current_overlay_index + 1}/{len(self.overlay_list)})")
            else:
                # If only one overlay chosen, do not show index numbers
                self.setWindowTitle(overlay_name)
            # Also enforce fix scaling policy based on overlay count
            self._enforce_fix_scaling_policy()

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

    def _maybe_switch_mesh_for_overlay(self, overlay_path: str):
        """If the overlay implies a different mesh, switch meshes and rebuild mappers/actors.

        This allows cycling across overlays from different subjects/runs where the mesh files differ.
        """
        try:
            new_mesh = convert_filename_to_mesh(overlay_path)
        except Exception:
            return
        if not new_mesh:
            return
        new_mesh_path = Path(new_mesh)
        # If the target mesh is the same file, do nothing
        if new_mesh_path.exists() and str(new_mesh_path) == str(Path(self.opts.mesh_left)):
            return
        if not new_mesh_path.exists():
            return
        # Load new left mesh
        self.poly_l = read_gifti_mesh(str(new_mesh_path))
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

    def _apply_controls(self):
        # Overlay range
        r0 = float(self.ctrl.range_min.value()); r1 = float(self.ctrl.range_max.value())
        if r1 > r0: self.overlay_range = [r0, r1]
        # Clip window
        c0 = float(self.ctrl.clip_min.value()); c1 = float(self.ctrl.clip_max.value())
        self.opts.clip = (c0, c1) if c1 > c0 else (0.0, -1.0)
        # Background range
        b0 = float(self.ctrl.bkg_min.value()); b1 = float(self.ctrl.bkg_max.value())
        if b1 > b0:
            self.range_bkg = [b0, b1]
            for actor in (self.actor_bkg_l, self.actor_bkg_r):
                if actor: actor.GetMapper().SetScalarRange(self.range_bkg)
        # Opacity
        self.opts.opacity = max(0.0, min(1.0, self.ctrl.opacity.value()/100.0))
        self.lut_overlay_l = get_lookup_table(self.opts.colormap, self.opts.opacity)
        self.lut_overlay_r = get_lookup_table(self.opts.colormap, self.opts.opacity)
        self._apply_discrete_to_overlay_lut(self.lut_overlay_l)
        self._apply_discrete_to_overlay_lut(self.lut_overlay_r)
        if self.actor_ov_l is not None:
            self.actor_ov_l.GetMapper().SetLookupTable(self.lut_overlay_l)
            if self.overlay_range[1] > self.overlay_range[0]:
                self.actor_ov_l.GetMapper().SetScalarRange(self.overlay_range)
        if self.actor_ov_r is not None:
            self.actor_ov_r.GetMapper().SetLookupTable(self.lut_overlay_r)
            if self.overlay_range[1] > self.overlay_range[0]:
                self.actor_ov_r.GetMapper().SetScalarRange(self.overlay_range)
        # Overlay path
        new_overlay = self.ctrl.overlay_path.text().strip()
        if new_overlay and new_overlay != (self.opts.overlay or ""):
            self._capture_camera_state()
            # If the new overlay maps to a different mesh, switch meshes first
            self._maybe_switch_mesh_for_overlay(new_overlay)
            self._load_overlay(new_overlay)
            self._apply_camera_state()
            # Overlay list may be single; ensure fix scaling disabled when not applicable
            self._enforce_fix_scaling_policy()
        elif not new_overlay and self.opts.overlay:
            # Overlay was cleared, remove overlay and disable controls
            self.opts.overlay = None
            self.scal_l = None
            self.scal_r = None
            # Remove overlay actors
            for actor in (self.actor_ov_l, self.actor_ov_r):
                if actor:
                    self.ren.RemoveActor(actor)
            self.actor_ov_l = None
            self.actor_ov_r = None
            # Disable overlay controls
            self.ctrl.set_overlay_controls_enabled(False)
            # Remove colorbar if it exists
            self._remove_colorbar()
            # Enforce fix scaling policy for zero overlays
            self._enforce_fix_scaling_policy()
        # Toggles
        self.opts.colorbar = self.ctrl.cb_colorbar.isChecked()
        # Persist title mode
        self.opts.title_mode = self.ctrl.title_mode.currentText()
        inv = self.ctrl.cb_inverse.isChecked()
        if inv != self.opts.inverse:
            self.opts.inverse = inv; self._apply_inverse()
        # Fix scaling
        self.opts.fix_scaling = self.ctrl.cb_fix_scaling.isChecked()
        if self.opts.fix_scaling and self.fixed_overlay_range is None:
            # Store the current range as fixed
            self.fixed_overlay_range = list(self.overlay_range)
        # Colorbar
        if self.opts.colorbar:
            self._ensure_colorbar()
        else:
            self._remove_colorbar()
        self.rw.Render()

    def _apply_inverse(self):
        for poly in (self.poly_l, self.poly_r if self.poly_r is not None else None):
            if poly is None: continue
            arr = poly.GetPointData().GetScalars()
            if arr is None: continue
            for i in range(arr.GetNumberOfTuples()): arr.SetValue(i, -float(arr.GetValue(i)))
        if self.overlay_range[1] > self.overlay_range[0]:
            self.overlay_range = [-self.overlay_range[1], -self.overlay_range[0]]
        for actor in (self.actor_ov_l, self.actor_ov_r):
            if actor and self.overlay_range[1] > self.overlay_range[0]:
                actor.GetMapper().SetScalarRange(self.overlay_range)
        # Sync control panel fields to reflect the updated range
        if hasattr(self, 'ctrl') and self.overlay_range[1] > self.overlay_range[0]:
            self.ctrl.range_min.setValue(float(self.overlay_range[0]))
            self.ctrl.range_max.setValue(float(self.overlay_range[1]))
            # Also update slider bounds/positions based on data
            self._update_slider_bounds()
        # Update colorbar if visible
        if self.opts.colorbar:
            self._ensure_colorbar()

    def _ensure_colorbar(self):
        # Ensure attribute exists for first-time calls during initialization
        if not hasattr(self, 'scalar_bar'):
            self.scalar_bar = None
        if self.scalar_bar is not None:
            # Simple continuous colorbar LUT based on current colormap/opacity and range
            lut_cb = get_lookup_table(self.opts.colormap, self.opts.opacity)
            if self.overlay_range[1] > self.overlay_range[0]:
                lut_cb.SetTableRange(self.overlay_range)
            self.scalar_bar.SetLookupTable(lut_cb)

            # Update title according to title_mode (only if colorbar is enabled)
            title_mode = self.opts.title_mode
            if title_mode == 'none':
                # Reserve title space with a single space to keep layout consistent
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
            # Prevent VTK from auto-scaling label text based on available space
            try:
                self.scalar_bar.SetAnnotationTextScaling(False)
            except Exception:
                pass
            self.rw.Render()
            return

        # Create a new scalar bar
        lut_cb = get_lookup_table(self.opts.colormap, self.opts.opacity)
        if self.overlay_range[1] > self.overlay_range[0]:
            lut_cb.SetTableRange(self.overlay_range)
        sb = vtkScalarBarActor()
        sb.SetOrientationToHorizontal()
        sb.SetLookupTable(lut_cb)
        sb.SetWidth(0.3)
        sb.SetHeight(0.05)
        sb.SetPosition(0.35, 0.05)

        base_fs = self.opts.fontsize if self.opts.fontsize else 12
        tp = sb.GetLabelTextProperty(); tp.SetFontSize(base_fs)
        tp2 = sb.GetTitleTextProperty(); tp2.SetFontSize(base_fs)

        # Title according to title_mode
        title_mode = self.opts.title_mode
        if title_mode == 'none':
            # Use a single space to maintain consistent layout even without visible title
            sb.SetTitle(" ")
        elif title_mode == 'stats' or (self.opts.stats and self.scal_l is not None):
            if self.scal_l is not None:
                info = f"Mean={get_mean(self.scal_l):.3f} Median={get_median(self.scal_l):.3f} SD={get_std(self.scal_l):.3f}"
                sb.SetTitle(info)
            else:
                sb.SetTitle("")
        else:
            sb.SetTitle(Path(self.opts.overlay or self.opts.mesh_left).name)

        # Prevent auto-scaling of label text so font size remains consistent
        try:
            sb.SetAnnotationTextScaling(False)
        except Exception:
            pass

        self.scalar_bar = sb
        # Add to UI renderer (layer 1) so it doesn't affect main scene bounds/camera
        try:
            self.ren_ui.AddViewProp(sb)
        except Exception:
            # Fallback to main renderer if UI renderer not available
            self.ren.AddViewProp(sb)

    def _remove_colorbar(self):
        if hasattr(self, 'scalar_bar') and self.scalar_bar is not None:
            try:
                self.ren_ui.RemoveViewProp(self.scalar_bar)
            except Exception:
                self.ren.RemoveViewProp(self.scalar_bar)
            self.scalar_bar = None

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
        # inverse
        if self.opts.inverse and scal_l is not None:
            for i in range(scal_l.GetNumberOfTuples()): scal_l.SetValue(i, -scal_l.GetValue(i))
        if self.opts.inverse and scal_r is not None:
            for i in range(scal_r.GetNumberOfTuples()): scal_r.SetValue(i, -scal_r.GetValue(i))
        # clip
        if self.opts.clip[1] > self.opts.clip[0]:
            def clip_it(arr: vtkDoubleArray):
                for i in range(arr.GetNumberOfTuples()):
                    v = float(arr.GetValue(i))
                    if (self.opts.clip[0] < v < self.opts.clip[1]) or math.isnan(v): arr.SetValue(i, 0.0)
            if scal_l is not None: clip_it(scal_l)
            if scal_r is not None: clip_it(scal_r)
        # attach
        self.scal_l = scal_l; self.scal_r = scal_r
        if scal_l is not None: self.poly_l.GetPointData().SetScalars(scal_l)
        if scal_r is not None and self.poly_r is not None: self.poly_r.GetPointData().SetScalars(scal_r)
        # Predefined ranges for recognized overlays (thickness, pbt)
        kind = detect_overlay_kind(overlay_path)
        if kind in ('thickness', 'pbt') and not self.opts.fix_scaling:
            # Apply requested defaults: overlay 1..5 (or -5..-1 if inverse); clip 0..0; bkg -1..1
            ov_range = [1.0, 5.0]
            if self.opts.inverse:
                ov_range = [-5.0, -1.0]
            self.overlay_range = ov_range
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
        
        # Restore camera and render
        self._apply_camera_state()
        self.rw.Render()

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

# ---- Entrypoint ----
def main(argv: List[str]):
    opts = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)
    win = Viewer(opts); win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main(sys.argv[1:])
