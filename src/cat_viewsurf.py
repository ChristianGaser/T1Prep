#!/usr/bin/env python3
"""
CAT_ViewSurf — PyQt + VTK port with right-side control panel

Features:
  • Load LH mesh (.gii). Auto-detect RH mesh via name pattern ("lh."→"rh.", "left"→"right").
  • Optional overlay scalars (.gii; FreeSurfer morph: thickness/curv/sulc; or text one value/line).
  • Optional background scalars for curvature shading.
  • Six-view montage (lat/med/sup/inf/ant/post) by cloning actors with transforms.
  • Colormaps: C1, C2, C3, JET, FIRE, BIPOLAR, GRAY. Discrete levels, inverse, clip window.
  • Colorbar (VTK 9.5-compatible AddViewProp), optional stats in title.
  • Right-side docked control panel: range, clip, colorbar toggle, overlay picker, opacity, bkg range, stats, inverse.
  • Keyboard: u/d/l/r rotate (Shift=±1°, Ctrl=180°), b flip, o reset, g screenshot, plus standard VTK keys.

Requires: vtk (>=9), PyQt6 or PyQt5; nibabel (for GIFTI fallback + FreeSurfer morphs if VTK lacks vtkGIFTIReader).
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# --- Qt setup (prefer PyQt6, fallback to PyQt5) ---
try:
    from PyQt6 import QtWidgets
    from PyQt6.QtCore import Qt
except Exception:  # pragma: no cover
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import Qt

try:
    from PyQt6.QtGui import QAction, QKeySequence
    from PyQt6.QtWidgets import QShortcut
except Exception:
    from PyQt5.QtWidgets import QAction, QShortcut
    from PyQt5.QtGui import QKeySequence

# PyQt5/6 compatibility shims
try:
    ORIENT_H = Qt.Orientation.Horizontal  # PyQt6
except Exception:
    ORIENT_H = Qt.Horizontal              # PyQt5
try:
    DOCK_RIGHT = Qt.DockWidgetArea.RightDockWidgetArea  # PyQt6
    DOCK_LEFT = Qt.DockWidgetArea.LeftDockWidgetArea    # PyQt6
except Exception:
    DOCK_RIGHT = getattr(Qt, 'RightDockWidgetArea', None) or Qt.RightDockWidgetArea  # PyQt5
    DOCK_LEFT  = getattr(Qt, 'LeftDockWidgetArea', None) or Qt.LeftDockWidgetArea   # PyQt5


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
    def SetRenderer(self, ren: vtkRenderer): self._renderer = ren
    def SetViewer(self, viewer): self._viewer = viewer
    def OnKeyPress(self):
        rwi = self.GetInteractor()
        sym = rwi.GetKeySym()  # e.g., 'Left', 'Right', 'u', 'U', 'w', '3'
        camera: vtkCamera = self._renderer.GetActiveCamera()
        shift = rwi.GetShiftKey(); ctrl = rwi.GetControlKey()
        def do_render(): self._renderer.ResetCameraClippingRange(); rwi.Render()
        # Built-in keys should still work
        if sym in ('q','Q','e','E','p','P','s','S','t','T','j','J','w','W','m','M','f','F','3'):
            super().OnKeyPress(); return
        # Camera controls
        if sym in ('u','U'):
            camera.Elevation(180 if ctrl else (1.0 if shift else 45.0)); camera.OrthogonalizeViewUp(); do_render(); return
        if sym in ('d','D'):
            camera.Elevation(-180 if ctrl else (-1.0 if shift else -45.0)); camera.OrthogonalizeViewUp(); do_render(); return
        if sym in ('l','L'):
            camera.Azimuth(180 if ctrl else (1.0 if shift else 45.0)); camera.OrthogonalizeViewUp(); do_render(); return
        if sym in ('r','R'):
            camera.Azimuth(180 if ctrl else (-1.0 if shift else -45.0)); camera.OrthogonalizeViewUp(); do_render(); return
        if sym in ('o','O'):
            self._renderer.ResetCamera(); camera.OrthogonalizeViewUp(); camera.Zoom(2.0); do_render(); return
        if sym in ('b','B'):
            actors = self._renderer.GetActors(); actors.InitTraversal(); n = actors.GetNumberOfItems(); actors.InitTraversal()
            for _ in range(n):
                a = actors.GetNextActor();
                if a is not None: a.RotateX(180)
            camera.OrthogonalizeViewUp(); do_render(); return
        if sym in ('g','G'):
            win = rwi.GetRenderWindow(); name = Path(win.GetWindowName() or 'screenshot').with_suffix('.png')
            w2i = vtkWindowToImageFilter(); w2i.SetInput(win); w2i.Update()
            png = vtkPNGWriter(); png.SetFileName(str(name)); png.SetInputConnection(w2i.GetOutputPort()); png.Write()
            print(f"Saved {name}"); return
        if sym in ('h','H'):
            print("KEYS: u/d/l/r rotate, b flip, o reset, w/s wireframe/shaded, g screenshot, ←/→ overlay navigation"); return
        # Arrow key navigation for overlays
        if sym == 'Left' and self._viewer:
            self._viewer._prev_overlay(); return
        if sym == 'Right' and self._viewer:
            self._viewer._next_overlay(); return
        # Fallback
        super().OnKeyPress()

    def OnChar(self):
        # Keep default behavior for any remaining character events
        super().OnChar()

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
    stats: bool = False
    inverse: bool = False
    colorbar: bool = False
    discrete: int = 0
    log: bool = False
    white: bool = False
    panel: bool = True  # show right-side control dock at startup
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
    p.add_argument('-stats', action='store_true')
    p.add_argument('-inverse', action='store_true')
    p.add_argument('-colorbar','-cb', dest='colorbar', action='store_true')
    p.add_argument('-discrete','-dsc', dest='discrete', type=int, default=0)
    p.add_argument('-log', action='store_true')
    p.add_argument('-white', action='store_true')
    p.add_argument('--no-panel', dest='panel', action='store_false', help='Start with the control panel hidden')
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
        # Range (overlay)
        self.range_min = QtWidgets.QDoubleSpinBox(); self.range_min.setDecimals(6); self.range_min.setRange(-1e9, 1e9)
        self.range_max = QtWidgets.QDoubleSpinBox(); self.range_max.setDecimals(6); self.range_max.setRange(-1e9, 1e9)
        range_box = QtWidgets.QHBoxLayout(); range_box.addWidget(self.range_min); range_box.addWidget(self.range_max)
        form.addRow("Range (overlay)", self._wrap(range_box))
        # Clip
        self.clip_min = QtWidgets.QDoubleSpinBox(); self.clip_min.setDecimals(6); self.clip_min.setRange(-1e9, 1e9)
        self.clip_max = QtWidgets.QDoubleSpinBox(); self.clip_max.setDecimals(6); self.clip_max.setRange(-1e9, 1e9)
        clip_box = QtWidgets.QHBoxLayout(); clip_box.addWidget(self.clip_min); clip_box.addWidget(self.clip_max)
        form.addRow("Clip window", self._wrap(clip_box))
        # Range bkg
        self.bkg_min = QtWidgets.QDoubleSpinBox(); self.bkg_min.setDecimals(6); self.bkg_min.setRange(-1e9, 1e9)
        self.bkg_max = QtWidgets.QDoubleSpinBox(); self.bkg_max.setDecimals(6); self.bkg_max.setRange(-1e9, 1e9)
        bkg_box = QtWidgets.QHBoxLayout(); bkg_box.addWidget(self.bkg_min); bkg_box.addWidget(self.bkg_max)
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
        self.cb_stats = QtWidgets.QCheckBox("Show stats on colorbar")
        self.cb_inverse = QtWidgets.QCheckBox("Inverse (flip sign)")
        self.cb_fix_scaling = QtWidgets.QCheckBox("Fix scaling")
        form.addRow(self.cb_colorbar)
        form.addRow(self.cb_stats)
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
    def _wrap(self, hbox: QtWidgets.QHBoxLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); w.setLayout(hbox); return w
    
    def set_overlay_controls_enabled(self, enabled: bool):
        """Enable or disable overlay-related controls based on whether an overlay is loaded."""
        # Range controls
        self.range_min.setEnabled(enabled)
        self.range_max.setEnabled(enabled)
        # Clip controls
        self.clip_min.setEnabled(enabled)
        self.clip_max.setEnabled(enabled)
        # Colorbar and stats controls
        self.cb_colorbar.setEnabled(enabled)
        self.cb_stats.setEnabled(enabled)
        # Inverse control
        self.cb_inverse.setEnabled(enabled)

# ---- Viewer ----
class Viewer(QtWidgets.QMainWindow):
    def _setup_view_menu(self):
        menubar = self.menuBar()
        menu = menubar.addMenu("View")
        act = QtWidgets.QAction("Show Controls", self)
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
        self.rw: vtkRenderWindow = self.vtk_widget.GetRenderWindow(); self.rw.AddRenderer(self.ren)

        # interactor style
        self.iren: vtkRenderWindowInteractor = self.rw.GetInteractor()
        style = CustomInteractorStyle(); style.SetRenderer(self.ren); style.SetViewer(self); self.iren.SetInteractorStyle(style)

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
        def shift_y_to(poly: vtkPolyData, to_value: float = -100.0):
            b = [0]*6; poly.GetBounds(b); y_shift = to_value - b[2]
            pts = poly.GetPoints()
            for i in range(pts.GetNumberOfPoints()):
                x,y,z = pts.GetPoint(i); pts.SetPoint(i, x, y+y_shift, z)
            poly.SetPoints(pts)
        shift_y_to(self.poly_l);  
        if self.poly_r is not None: shift_y_to(self.poly_r)

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
        
        # Overlay scalars
        self.scal_l = None; self.scal_r = None
        if self.overlay_list:
            self._load_overlay(self.overlay_list[0])

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
        def add_clone(actor: vtkActor, px, py, rx, rz):
            a = vtkActor(); a.ShallowCopy(actor); a.AddPosition(px, py, 0); a.RotateX(rx); a.RotateZ(rz); self.ren.AddActor(a)
        for i in range(views):
            if self.poly_r is None and (i % 2 == 1): continue
            src = self.actor_bkg_r if (order[i] == 1 and self.actor_bkg_r is not None) else self.actor_bkg_l
            add_clone(src, posx[i], posy[i], rotx[i], rotz[i])
        if self.actor_ov_l is not None or self.actor_ov_r is not None:
            for i in range(views):
                if self.poly_r is None and (i % 2 == 1): continue
                src = self.actor_ov_r if (order[i] == 1 and self.actor_ov_r is not None) else self.actor_ov_l
                if src is not None: add_clone(src, posx[i], posy[i], rotx[i], rotz[i])

        # Colorbar (lazy create)
        self.scalar_bar = None
        if opts.colorbar: self._ensure_colorbar()

        # Camera
        self.ren.ResetCamera(); self.ren.GetActiveCamera().Zoom(2.0)

        # Right-side control panel (dock)
        self._build_control_panel()

        # Start interactor
        self.vtk_widget.Initialize(); self.vtk_widget.Start(); self.vtk_widget.setFocus()

        self.vtk_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu
                                             if hasattr(Qt, "ContextMenuPolicy")
                                             else Qt.CustomContextMenu)
        self.vtk_widget.customContextMenuRequested.connect(self._show_view_context_menu)

        # Optional snapshot
        if opts.output: self.save_png(opts.output)
        
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
    
        # Dock features: PyQt5/6 compatibility
        DockFeature = getattr(QtWidgets.QDockWidget, "DockWidgetFeature", QtWidgets.QDockWidget)
        dock.setFeatures(
            getattr(DockFeature, "DockWidgetMovable")
            | getattr(DockFeature, "DockWidgetFloatable")
            | getattr(DockFeature, "DockWidgetClosable")
        )
        dock.setAllowedAreas(DOCK_RIGHT | DOCK_LEFT)
        
        # Make the dock float by default so it overlays on the content
        # This prevents the central widget from shifting when dock is shown
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
        
        # Connect to show event to position the dock
        dock.showEvent = lambda event: (super(QtWidgets.QDockWidget, dock).showEvent(event), position_dock())
        
        self.addDockWidget(DOCK_RIGHT, dock)
    
        # ---------- local helpers (closures) ----------
        # programmatic toggle (menu/shortcut)
        def _toggle_controls_local(checked: bool):
            if checked == dock.isVisible():
                return
            # Simply show/hide the dock without resizing the window
            # The dock will overlay on the right side of the window
            if checked:
                dock.setFloating(True)  # Ensure it stays floating
                dock.show()
                position_dock()  # Position it correctly
            else:
                dock.hide()
    
        self._toggle_controls = _toggle_controls_local
    
        # user-driven visibility change (dock close button / drag)
        def _on_vis_changed_local(visible: bool):
            # sync menu action
            if hasattr(self, "act_show_controls"):
                self.act_show_controls.blockSignals(True)
                self.act_show_controls.setChecked(visible)
                self.act_show_controls.blockSignals(False)
    
            # No window resizing needed - dock overlays on the right side
    
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
        self.ctrl.cb_stats.setChecked(self.opts.stats)
        self.ctrl.cb_inverse.setChecked(self.opts.inverse)
        self.ctrl.cb_fix_scaling.setChecked(self.opts.fix_scaling)
        
        # Enable/disable overlay controls based on whether overlay is loaded
        has_overlay = (self.overlay_list or self.opts.overlay) and self.scal_l is not None
        self.ctrl.set_overlay_controls_enabled(has_overlay)
    
        # Signals
        self.ctrl.apply_btn.clicked.connect(self._apply_controls)
        self.ctrl.reset_btn.clicked.connect(self._reset_camera)
        self.ctrl.overlay_btn.clicked.connect(self._pick_overlay)
    
        # Start state
        if self.opts.panel:
            dock.setFloating(True)  # Ensure it starts floating
            dock.show()
            position_dock()  # Position it correctly
        else:
            dock.hide()
    
        # View menu + keyboard shortcut (uses QAction shim)
        menubar = self.menuBar()
        menu = menubar.addMenu("View")
        act = QAction("Show Controls", self)
        act.setCheckable(True)
        act.setChecked(self.opts.panel)
        act.setShortcut("Ctrl+D")   # On macOS, Qt maps this to Command+D automatically
        self.addAction(act)               # make shortcut active globally
        act.triggered.connect(self._toggle_controls)
        menu.addAction(act)
        self.act_show_controls = act

        # Add a direct QShortcut to toggle dock visibility reliably with one key press
        # This avoids any interference from focused VTK interactor or QAction state.
        self._dock_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        self._dock_shortcut.activated.connect(lambda d=dock: self._toggle_controls(not d.isVisible()))

    def _reset_camera(self):
        self.ren.ResetCamera(); self.ren.GetActiveCamera().Zoom(2.0); self.rw.Render()

    def _pick_overlay(self):
        start_dir = self.ctrl.overlay_path.text() or str(Path(self.opts.mesh_left).parent)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose overlay", start_dir)
        if path: self.ctrl.overlay_path.setText(path)

    def _next_overlay(self):
        """Switch to next overlay in the list."""
        if len(self.overlay_list) > 1:
            self.current_overlay_index = (self.current_overlay_index + 1) % len(self.overlay_list)
            self._load_overlay(self.overlay_list[self.current_overlay_index])
            self._update_overlay_info()
            # Update control panel with current overlay range
            if hasattr(self, 'ctrl'):
                self.ctrl.range_min.setValue(float(self.overlay_range[0]))
                self.ctrl.range_max.setValue(float(self.overlay_range[1]))

    def _prev_overlay(self):
        """Switch to previous overlay in the list."""
        if len(self.overlay_list) > 1:
            self.current_overlay_index = (self.current_overlay_index - 1) % len(self.overlay_list)
            self._load_overlay(self.overlay_list[self.current_overlay_index])
            self._update_overlay_info()
            # Update control panel with current overlay range
            if hasattr(self, 'ctrl'):
                self.ctrl.range_min.setValue(float(self.overlay_range[0]))
                self.ctrl.range_max.setValue(float(self.overlay_range[1]))

    def _update_overlay_info(self):
        """Update the overlay path display and window title."""
        if self.overlay_list:
            current_overlay = self.overlay_list[self.current_overlay_index]
            self.ctrl.overlay_path.setText(current_overlay)
            # Update window title to show current overlay
            overlay_name = Path(current_overlay).name
            self.setWindowTitle(f"{overlay_name} ({self.current_overlay_index + 1}/{len(self.overlay_list)})")

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
        for actor in (self.actor_ov_l, self.actor_ov_r):
            if actor:
                actor.GetMapper().SetLookupTable(self.lut_overlay_l)
                if self.overlay_range[1] > self.overlay_range[0]:
                    actor.GetMapper().SetScalarRange(self.overlay_range)
        # Overlay path
        new_overlay = self.ctrl.overlay_path.text().strip()
        if new_overlay and new_overlay != (self.opts.overlay or ""):
            self._load_overlay(new_overlay)
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
        # Toggles
        self.opts.colorbar = self.ctrl.cb_colorbar.isChecked()
        self.opts.stats = self.ctrl.cb_stats.isChecked()
        inv = self.ctrl.cb_inverse.isChecked()
        if inv != self.opts.inverse:
            self.opts.inverse = inv; self._apply_inverse()
        # Fix scaling
        self.opts.fix_scaling = self.ctrl.cb_fix_scaling.isChecked()
        if self.opts.fix_scaling and self.fixed_overlay_range is None:
            # Store the current range as fixed
            self.fixed_overlay_range = list(self.overlay_range)
        # Colorbar
        if self.opts.colorbar: self._ensure_colorbar()
        else: self._remove_colorbar()
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

    def _ensure_colorbar(self):
        if self.scalar_bar is not None:
            if self.opts.stats and self.scal_l is not None:
                info = f"Mean={get_mean(self.scal_l):.3f} Median={get_median(self.scal_l):.3f} SD={get_std(self.scal_l):.3f}"
                self.scalar_bar.SetTitle(info)
            self.rw.Render(); return
        lut_cb = get_lookup_table(self.opts.colormap, self.opts.opacity)
        if self.overlay_range[1] > self.overlay_range[0]: lut_cb.SetTableRange(self.overlay_range)
        sb = vtkScalarBarActor(); sb.SetOrientationToHorizontal(); sb.SetLookupTable(lut_cb)
        sb.SetWidth(0.3); sb.SetHeight(0.05); sb.SetPosition(0.35, 0.05)
        if self.opts.fontsize:
            tp = sb.GetLabelTextProperty(); tp.SetFontSize(self.opts.fontsize)
            tp2 = sb.GetTitleTextProperty(); tp2.SetFontSize(self.opts.fontsize)
        if self.opts.stats and self.scal_l is not None:
            info = f"Mean={get_mean(self.scal_l):.3f} Median={get_median(self.scal_l):.3f} SD={get_std(self.scal_l):.3f}"
            sb.SetTitle(info)
        else:
            sb.SetTitle(Path(self.opts.overlay or self.opts.mesh_left).name)
        self.scalar_bar = sb; self.ren.AddViewProp(sb)

    def _remove_colorbar(self):
        if self.scalar_bar is not None:
            self.ren.RemoveViewProp(self.scalar_bar); self.scalar_bar = None

    def _load_overlay(self, overlay_path: str):
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
        if self.opts.colorbar: self._ensure_colorbar()
        
        # Enable overlay controls since we now have an overlay loaded
        if hasattr(self, 'ctrl'):
            self.ctrl.set_overlay_controls_enabled(True)
        
        self.rw.Render()

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
