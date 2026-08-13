"""
cat_vol_view.py

Single-window VTK image viewer with 3 orthogonal slices (axial, coronal,
sagittal) arranged in an SPM12-like layout, with optional surface overlays.

Layout (SPM12-like)::

    +----------+----------+
    | Coronal  | Sagittal |
    |  (top-L) |  (top-R) |
    +----------+----------+
    | Axial    | image    |
    |  (bot-L) | info     |
    +----------+----------+

A second volume on the same voxel grid can be drawn in colour on top
(``--overlay``), with range, clip, colormap, opacity, inversion and the
p-value thresholds set from the control panel the surface viewer uses; the
reported intensity is then the overlay's.

Slices are shown in neurological orientation (left is left).  The information
panel lists file name, dimensions, voxel size, orientation code, data type and
intensity range, plus voxel index, mm position and value under the cursor —
and the region name when an atlas has been selected.

Display intensities are scaled to the 3rd--97th percentile range by default.

Usage (CLI):
    CAT_VolView <image> [more images…] [surf1] [surf2] [surf3] \
        --size 400 [--percentile 3 97]

    Up to three volumes may be given; each opens its own window and their
    cursors are linked, so a click in one moves the others to the same
    millimetre position.

    # source checkout
    python src/t1prep/gui/cat_vol_view.py <image> [surf1] [surf2] [surf3]

Notes:
- Tries to use vtkNIFTIImageReader for NIfTI, vtkMINCImageReader for MINC,
  and vtkImageReader2Factory otherwise.
- Surfaces: supports .gii, .vtp, .vtk, .obj, .stl via appropriate VTK readers.
- The viewer works in the millimetre space of the NIfTI sform/qform, so images
  and surfaces line up without adjustment and each pane shows the anatomical
  plane it is named after, whatever voxel order the image is stored in.
  ``--mirror`` is available for surfaces that need the x flip anyway.
- ``CatImageViewer`` can render into a render window supplied by a host
  application; CAT_SurfView embeds it as its volume window and links the cursor
  to the surface through :meth:`CatImageViewer.set_world_position` and the
  ``on_position_changed`` callback.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Import minimal VTK modules explicitly (avoids large monolithic import)
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkPlane
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonMath import vtkMatrix3x3, vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersCore import vtkCutter
from vtkmodules.vtkIOImage import (
    vtkNIFTIImageReader,
    vtkImageReader2Factory,
)
try:
    from vtkmodules.vtkIOImage import vtkMINCImageReader  # type: ignore
except Exception:  # pragma: no cover - optional
    vtkMINCImageReader = None  # type: ignore
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkOBJReader
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
try:
    # Available in VTK 9+ when GIFTI IO is enabled
    from vtkmodules.vtkIOGeometry import vtkGIFTIReader  # type: ignore
except Exception:  # pragma: no cover - optional
    vtkGIFTIReader = None  # type: ignore
try:
    import nibabel as nib  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional
    nib = None  # type: ignore
    np = None  # type: ignore
try:
    from vtkmodules.util.numpy_support import vtk_to_numpy  # type: ignore
except Exception:  # pragma: no cover - optional
    vtk_to_numpy = None  # type: ignore

from vtkmodules.vtkRenderingCore import (
    VTK_CURSOR_CROSSHAIR,
    vtkActor,
    vtkImageActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkTextActor,
    vtkWindowToImageFilter,
)
import vtkmodules.vtkRenderingFreeType  # noqa: F401  (text rendering)
try:
    from vtkmodules.vtkImagingColor import (
        vtkImageMapToWindowLevelColors,
    )
except Exception:  # pragma: no cover
    vtkImageMapToWindowLevelColors = None  # type: ignore
from vtkmodules.vtkIOImage import vtkPNGWriter
# Ensure rendering backend and interaction styles are registered (VTK 9 modular)
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkImagingCore import vtkImageMapToColors

# Control panel and colormaps are shared with the surface viewer
try:
    from .controls import ControlPanel, LOGP_THRESHOLDS
except ImportError:  # direct invocation as a script
    from controls import ControlPanel, LOGP_THRESHOLDS

# Colormaps are shared with the surface viewer
try:
    from .colormaps import (
        JET, COLORMAP_NAMES, COLORMAP_ORDER, build_overlay_lut,
    )
except ImportError:  # direct invocation as a script
    from colormaps import (
        JET, COLORMAP_NAMES, COLORMAP_ORDER, build_overlay_lut,
    )

# Qt window + interactor.  QVTKRWIBase has to be chosen before the widget is
# imported; CAT_SurfView imports this module and relies on the same setting.
from PySide6 import QtCore, QtWidgets
import vtkmodules.qt as _vtk_qt
_vtk_qt.QVTKRWIBase = "QOpenGLWidget"
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def _guess_image_reader(image_path: str):
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".nii", ".gz") or image_path.lower().endswith(".nii.gz"):
        reader = vtkNIFTIImageReader()
        reader.SetFileName(image_path)
        return reader
    if ext == ".mnc":
        if vtkMINCImageReader is not None:
            reader = vtkMINCImageReader()
            reader.SetFileName(image_path)
            return reader
        # Fallback to factory if MINC reader is unavailable
        factory = vtkImageReader2Factory()
        reader = factory.CreateImageReader2(image_path)
        if reader is None:
            raise RuntimeError(
                "VTK build lacks MINC reader support and factory could not "
                "resolve a reader for this .mnc file."
            )
        reader.SetFileName(image_path)
        return reader
    # Fallback to factory (MHA/MHD/NRRD and others)
    factory = vtkImageReader2Factory()
    reader = factory.CreateImageReader2(image_path)
    if reader is None:
        raise RuntimeError(f"Unsupported image type: {image_path}")
    reader.SetFileName(image_path)
    return reader


def _header_matrix(reader) -> Optional[List[List[float]]]:
    """The sform/qform a NIfTI reader exposes, or None when unset.

    VTK reports it normalized: it maps *data* coordinates (voxel index times
    spacing) to world, which is what the image actors need.
    """
    for getter in ("GetSFormMatrix", "GetQFormMatrix"):
        try:
            m = getattr(reader, getter)()
        except Exception:
            m = None
        if m is None:
            continue
        M = [[float(m.GetElement(r, c)) for c in range(4)] for r in range(4)]
        if all(abs(M[r][c]) < 1e-12 for r in range(3) for c in range(3)):
            continue  # unset form
        return M
    return None


def _voxel_to_world_matrix(reader, image) -> Tuple[List[List[float]], bool]:
    """Voxel index to world (mm) transform, and whether it came from the header.

    NIfTI keeps the anatomical mapping in the sform/qform, which VTK reports
    separately instead of baking it into the image.  Taking it from there is
    what puts slices, crosshair, surfaces and any linked window into the same
    millimetre space; the spacing has to be folded in because the reported
    matrix expects data coordinates, not voxel indices.
    """
    header = _header_matrix(reader)
    if header is not None:
        spacing = image.GetSpacing()
        return ([[header[r][c] * spacing[c] for c in range(3)] + [header[r][3]]
                 for r in range(4)], True)

    # No usable header transform: use the geometry VTK applied to the image
    ox, oy, oz = image.GetOrigin()
    sx, sy, sz = image.GetSpacing()
    try:
        dm = image.GetDirectionMatrix()
        D = [[dm.GetElement(r, c) for c in range(3)] for r in range(3)]
    except Exception:
        D = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return ([
        [D[0][0] * sx, D[0][1] * sy, D[0][2] * sz, ox],
        [D[1][0] * sx, D[1][1] * sy, D[1][2] * sz, oy],
        [D[2][0] * sx, D[2][1] * sy, D[2][2] * sz, oz],
        [0.0, 0.0, 0.0, 1.0],
    ], False)


def _load_surface(surface_path: str) -> vtkPolyData:
    """Load a surface file into vtkPolyData using appropriate reader by extension."""
    ext = os.path.splitext(surface_path)[1].lower()

    if ext == ".gii":
        # Prefer VTK reader if available
        if vtkGIFTIReader is not None:
            try:
                r = vtkGIFTIReader()
                r.SetFileName(surface_path)
                r.Update()
                out = r.GetOutput()
                if out is not None:
                    return out
            except Exception:
                pass
        # Fallback: use nibabel to parse GIFTI and convert to vtkPolyData
        if nib is None or np is None:
            raise RuntimeError(
                "GIFTI reader not available in this VTK build and nibabel is missing."
            )
        img = nib.load(surface_path)
        coords = None
        faces = None
        # Heuristically locate pointset and triangle arrays
        for da in getattr(img, 'darrays', []):
            arr = getattr(da, 'data', None)
            if arr is None:
                continue
            if getattr(arr, 'ndim', 0) == 2 and arr.shape[1] == 3:
                if np.issubdtype(arr.dtype, np.floating) and coords is None:
                    coords = np.asarray(arr, dtype=np.float32)
                elif np.issubdtype(arr.dtype, np.integer) and faces is None:
                    faces = np.asarray(arr, dtype=np.int64)
        if coords is None or faces is None:
            raise RuntimeError("Failed to parse coordinates/faces from GIFTI file: " + surface_path)

        points = vtkPoints()
        # SetData expects a VTK array; set points one-by-one for compatibility
        points.SetNumberOfPoints(int(coords.shape[0]))
        for idx in range(coords.shape[0]):
            x, y, z = float(coords[idx, 0]), float(coords[idx, 1]), float(coords[idx, 2])
            points.SetPoint(idx, x, y, z)

        polys = vtkCellArray()
        for tri in faces:
            polys.InsertNextCell(3)
            polys.InsertCellPoint(int(tri[0]))
            polys.InsertCellPoint(int(tri[1]))
            polys.InsertCellPoint(int(tri[2]))

        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(polys)
        return poly

    if ext == ".vtp":
        r = vtkXMLPolyDataReader()
        r.SetFileName(surface_path)
        r.Update()
        return r.GetOutput()

    if ext == ".vtk":
        r = vtkPolyDataReader()
        r.SetFileName(surface_path)
        r.Update()
        return r.GetOutput()

    if ext == ".stl":
        r = vtkSTLReader()
        r.SetFileName(surface_path)
        r.Update()
        return r.GetOutput()

    if ext == ".obj":
        r = vtkOBJReader()
        r.SetFileName(surface_path)
        r.Update()
        return r.GetOutput()

    raise RuntimeError(f"Unsupported surface type or reader unavailable: {surface_path}")


def _mirror_polydata_x(poly: vtkPolyData) -> vtkPolyData:
    t = vtkTransform()
    t.Scale(-1.0, 1.0, 1.0)
    f = vtkTransformPolyDataFilter()
    f.SetInputData(poly)
    f.SetTransform(t)
    f.Update()
    return f.GetOutput()


class CatImageViewer:
    """Single-window orthogonal viewer with SPM12-like layout.

    The three views are rendered in a single window arranged as::

        +----------+----------+
        | Coronal  | Sagittal |
        |  (top-L) |  (top-R) |
        +----------+----------+
        | Axial    |          |
        |  (bot-L) |  (empty) |
        +----------+----------+

    Viewer index mapping:
        0 → XY  (Axial,     slicing along Z)
        1 → YZ  (Sagittal,  slicing along X)
        2 → XZ  (Coronal,   slicing along Y)
    """

    VIEW_AXIAL = 0
    VIEW_SAGITTAL = 1
    VIEW_CORONAL = 2

    def __init__(
        self,
        window_size: int = 400,
        mirror_surfaces: bool = False,
        verbose: bool = False,
        surface_convention: str = "auto",
        percentile_range: Optional[Tuple[float, float]] = (3.0, 97.0),
        render_window: Optional[vtkRenderWindow] = None,
        interactor: Optional[vtkRenderWindowInteractor] = None,
        show_info: bool = True,
        interpolate: bool = True,
        recenter: bool = True,
    ):
        """Create the viewer.

        Args:
            show_info: Fill the free quadrant with image information (name,
                dimensions, voxel size, orientation, data type, intensity
                range) and the values under the cursor.
            interpolate: Smooth the slices (linear); False draws the raw
                voxels (nearest neighbour).
            recenter: Let a zoomed view follow the cursor.
            render_window: Render window to draw into.  Pass the one of a Qt
                ``QVTKRenderWindowInteractor`` to embed the viewer in another
                application (CAT_SurfView does this); a standalone window is
                created when omitted.
            interactor: Interactor belonging to *render_window*; required
                together with it.
        """
        self.window_size = int(window_size)
        self.verbose = bool(verbose)
        self.surface_convention = surface_convention.lower()
        self.percentile_range = percentile_range
        # Avoid double flipping: if a convention is explicitly provided,
        # do not mirror by default.
        if self.surface_convention in ("ras", "lps") and mirror_surfaces:
            self.mirror_surfaces = False
            if self.verbose:
                print("[cat_vol_view] Disabling mirroring due to "
                      "explicit --surface-convention")
        else:
            self.mirror_surfaces = bool(mirror_surfaces)
        self._image = None
        self._vox2world = None  # 4×4 matrix (list of lists): index -> world
        # Same transform in the form the image actors need (data coordinates,
        # i.e. index times spacing, to world)
        self._actor_matrix = None
        # True when the transform comes from the NIfTI sform/qform, i.e. the
        # world space is anatomical millimetres
        self._world_from_header = False
        # Voxel axis each pane slices along; recomputed per image
        self._pane_axis: List[int] = [2, 0, 1]
        # Edge length in mm the panes are zoomed to (None = whole volume), the
        # position a zoomed view is centred on, and whether it follows the cursor
        self._fov_mm: Optional[float] = None
        self._fov_center: Optional[Tuple[float, float, float]] = None
        self.recenter = bool(recenter)
        # Slices smoothed (linear) or as raw voxels (nearest neighbour)
        self.interpolate = bool(interpolate)
        # Information panel in the free quadrant
        self.show_info = bool(show_info)
        self._info_actor: Optional[vtkTextActor] = None
        self._image_name = ""
        self._orientation: Optional[str] = None
        # Atlas selected for naming the region under the cursor
        self.atlas_path: Optional[str] = None
        self._atlas: Optional[dict] = None
        # Overlay volume drawn on top, with its own colour settings
        self.overlay_path: Optional[str] = None
        self.overlay_name = ""
        self._overlay_image = None
        self._overlay_actors: List[Optional[vtkImageActor]] = [None, None, None]
        self._overlay_colors: List = [None, None, None]
        self.overlay_range: List[float] = [0.0, 0.0]
        self.overlay_clip: Tuple[float, float] = (0.0, 0.0)
        self.overlay_colormap = JET
        self.overlay_opacity = 0.8
        self.overlay_inverse = False
        self.overlay_discrete = 0

        # Three renderers + image actors (one per orthogonal view)
        self.renderers: List[vtkRenderer] = [
            vtkRenderer(), vtkRenderer(), vtkRenderer(),
        ]
        self._image_actors: List[Optional[vtkImageActor]] = [
            None, None, None,
        ]
        self._wl_filters: List = [None, None, None]
        # Single render window and interactor.  When both are supplied the
        # viewer is embedded in a host application, which owns the window
        # geometry and the event loop.
        self.embedded = render_window is not None and interactor is not None
        self.render_window: vtkRenderWindow = render_window or vtkRenderWindow()
        self.interactor: vtkRenderWindowInteractor = (
            interactor or vtkRenderWindowInteractor()
        )
        # Viewport bounds computed in setup – default equal quadrants
        self._viewports: List[Tuple[float, float, float, float]] = [
            (0.0, 0.0, 0.5, 0.5),   # Axial    (bottom-left)
            (0.0, 0.5, 0.5, 1.0),   # Sagittal (top-left)
            (0.5, 0.5, 1.0, 1.0),   # Coronal  (top-right)
        ]

        # Surfaces to overlay: list of (polydata, colour)
        self.surfaces: List[Tuple[vtkPolyData, Tuple[float, float, float]]] = []
        # Cursor: the exact world (mm) position is the truth, the voxel index
        # is derived from it for the displayed slices and the intensity readout
        self._cursor: Optional[List[float]] = None
        self._ijk: Optional[List[int]] = None
        # Surface contour pipelines per view
        self._surface_contours: List[List] = [[], [], []]
        # Interaction callbacks (kept alive)
        self._event_cbs: List = []
        self._left_down = False
        # Called with the world (mm) position whenever the user moves the
        # cursor by clicking or scrolling.  Used to link the viewer to other
        # windows, e.g. the surface view of CAT_SurfView.
        self.on_position_changed = None

    # -------- Geometry helpers --------
    def _voxel_axis_directions(self) -> List[Tuple[float, float, float]]:
        """Unit world direction of each voxel axis."""
        m = self._vox2world
        dirs = []
        for a in range(3):
            v = (m[0][a], m[1][a], m[2][a])
            n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5 or 1.0
            dirs.append((v[0] / n, v[1] / n, v[2] / n))
        return dirs

    def _voxel_axis_lengths(self) -> List[float]:
        """Physical length of the volume along each voxel axis (mm)."""
        m = self._vox2world
        ext = self._image.GetExtent()
        out = []
        for a in range(3):
            n = float(ext[2 * a + 1] - ext[2 * a] + 1)
            step = (m[0][a] ** 2 + m[1][a] ** 2 + m[2][a] ** 2) ** 0.5
            out.append(n * step)
        return out

    def _world_matrix(self) -> vtkMatrix4x4:
        """Transform for the image actors: data coordinates to world."""
        source = self._actor_matrix or self._vox2world
        m = vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                m.SetElement(r, c, float(source[r][c]))
        return m

    def _assign_pane_axes(self):
        """Choose the voxel axis each pane slices along.

        The panes are anatomical (axial, sagittal, coronal) but the voxel axes
        are not — a scan may be stored sagittal-first.  Every pane therefore
        takes the voxel axis whose world direction is most parallel to the
        anatomical axis it cuts, so each pane shows the plane it is named
        after whatever the storage order.  For an image stored in RAS order
        this reproduces the classic k/i/j assignment.
        """
        dirs = self._voxel_axis_directions()
        wanted = (2, 0, 1)  # axial cuts world Z, sagittal X, coronal Y
        best, best_score = (2, 0, 1), -1.0
        for perm in itertools.permutations(range(3)):
            score = sum(abs(dirs[perm[p]][wanted[p]]) for p in range(3))
            if score > best_score:
                best, best_score = perm, score
        self._pane_axis = list(best)
        if self.verbose:
            names = ["Axial", "Sagittal", "Coronal"]
            print("[cat_vol_view] Pane -> voxel axis: "
                  + ", ".join(f"{names[p]}={'ijk'[self._pane_axis[p]]}"
                              for p in range(3)))

    def _world_bbox_lengths(self) -> Tuple[float, float, float]:
        """Size of the volume along the world axes (mm)."""
        ext = self._image.GetExtent()
        corners = [
            self._world_from_index((ext[0 + (i & 1)], ext[2 + ((i >> 1) & 1)],
                                    ext[4 + ((i >> 2) & 1)]))
            for i in range(8)
        ]
        return tuple(
            max(c[ax] for c in corners) - min(c[ax] for c in corners)
            for ax in range(3)
        )

    # -------- Layout helpers --------
    def _compute_viewports(self) -> Tuple[int, int]:
        """Compute SPM12-like viewport fractions from image physical dims.

        Returns the pixel *(width, height)* for the render window.
        """
        # World extents, so the layout does not depend on the storage order
        px, py, pz = self._world_bbox_lengths()

        # Prevent division by zero for degenerate images
        if px + py == 0 or py + pz == 0:
            px = py = pz = 1.0

        col = px / (px + py)   # left-column width fraction
        row = py / (py + pz)   # bottom-row height fraction

        # Panes are placed so their aspect matches the pane they sit in: the
        # left column is as wide as the volume is in x (coronal, axial), the
        # right column as wide as it is in y (sagittal); the bottom row is as
        # tall as the volume is in y, the top row as tall as it is in z.
        gap = 0.003  # thin black border between viewports
        self._viewports = [
            (0.0,        0.0,        col - gap, row - gap),  # Axial    (bot-L)
            (col + gap,  row + gap,  1.0,       1.0),        # Sagittal (top-R)
            (0.0,        row + gap,  col - gap, 1.0),        # Coronal  (top-L)
        ]

        # Window pixel dimensions preserving physical proportions
        total_w = px + py
        total_h = py + pz
        max_phys = max(total_w, total_h)
        scale = (2 * self.window_size) / max_phys
        win_w = max(int(total_w * scale), 200)
        win_h = max(int(total_h * scale), 200)
        return win_w, win_h

    def _apply_viewports(self):
        """Hand the computed viewport rectangles to the renderers."""
        for i, ren in enumerate(self.renderers):
            ren.SetViewport(*self._viewports[i])
        if getattr(self, '_info_renderer', None) is not None:
            col_edge = self._viewports[0][2] + 0.003
            row_edge = self._viewports[0][3] + 0.003
            self._info_renderer.SetViewport(col_edge, 0.0, 1.0, row_edge)

    # -------- Viewport query --------
    def _get_active_view(self, x: int, y: int) -> int:
        """Return the viewer index whose viewport contains *(x, y)*.

        Returns ``-1`` when the point is outside every viewport (e.g. the
        empty bottom-right quadrant).
        """
        w, h = self.render_window.GetSize()
        if w == 0 or h == 0:
            return -1
        nx, ny = x / w, y / h
        for i, (xmin, ymin, xmax, ymax) in enumerate(self._viewports):
            if xmin <= nx <= xmax and ymin <= ny <= ymax:
                return i
        return -1

    def _get_view_from_renderer(self, renderer: vtkRenderer) -> int:
        """Map a VTK renderer instance to the internal view index."""
        for i, ren in enumerate(self.renderers):
            if ren == renderer:
                return i
        return -1

    # -------- Scroll handler --------
    def _on_scroll(self, view_idx: int, delta: int):
        """Advance the slice in *view_idx* by *delta* steps."""
        index = self.get_index_exact()
        if index is None or not (0 <= view_idx < 3):
            return
        axis = self._pane_axis[view_idx]
        # Step whole slices, keeping the position within the plane
        index = list(index)
        index[axis] = round(index[axis]) + delta
        self._set_cursor(index, notify=True)

    def _dispatch_pointer(self, x: int, y: int):
        """Route a pointer position to the appropriate view and update slices."""
        ren = self.interactor.FindPokedRenderer(x, y)
        view = self._get_view_from_renderer(ren)
        if view < 0:
            view = self._get_active_view(x, y)
        if view >= 0:
            self._on_click(view, x, y)

    def _bind_interaction_events(self):
        """Install robust interactor observers for click/scroll only."""
        self._event_cbs = []

        def _left_down_cb(obj, evt):
            x, y = obj.GetEventPosition()
            self._dispatch_pointer(x, y)

        def _left_up_cb(obj, evt):
            return None

        def _wheel_fwd_cb(obj, evt):
            x, y = obj.GetEventPosition()
            ren = self.interactor.FindPokedRenderer(x, y)
            view = self._get_view_from_renderer(ren)
            if view < 0:
                view = self._get_active_view(x, y)
            if view >= 0:
                self._on_scroll(view, 1)

        def _wheel_back_cb(obj, evt):
            x, y = obj.GetEventPosition()
            ren = self.interactor.FindPokedRenderer(x, y)
            view = self._get_view_from_renderer(ren)
            if view < 0:
                view = self._get_active_view(x, y)
            if view >= 0:
                self._on_scroll(view, -1)

        self._event_cbs.extend([
            _left_down_cb,
            _left_up_cb,
            _wheel_fwd_cb,
            _wheel_back_cb,
        ])

        self.interactor.AddObserver("LeftButtonPressEvent", _left_down_cb)
        self.interactor.AddObserver("LeftButtonReleaseEvent", _left_up_cb)
        self.interactor.AddObserver("MouseWheelForwardEvent", _wheel_fwd_cb)
        self.interactor.AddObserver("MouseWheelBackwardEvent", _wheel_back_cb)

    # -------- Camera setup --------
    def _setup_cameras_spm12(self):
        """Set camera orientation per view, neurological convention.

        Axial   : camera from superior (+Z), view-up = +Y → anterior up,
                   patient-left on screen-left (neurological).
        Sagittal: camera from left (−X), view-up = +Z → superior up,
                   anterior on screen-left.
        Coronal : camera from posterior (−Y), view-up = +Z → superior up,
                   patient-left on screen-left (neurological).

        With the camera looking along *d* and *up* on screen, screen-right is
        ``d × up``; both in-plane views put +x (the patient's right) there,
        which is what "left is left" means.

        These directions are anatomical, which only holds because the world
        space comes from the NIfTI sform/qform (see :meth:`load_image`); the
        slices are then oriented the same way for every image, whatever voxel
        order it is stored in.
        """
        # (camera offset from the focal point, view-up)
        placement = [
            ((0.0, 0.0, 100.0), (0.0, 1.0, 0.0)),   # Axial, from superior
            ((-100.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # Sagittal, from the left
            ((0.0, -100.0, 0.0), (0.0, 0.0, 1.0)),  # Coronal, from posterior
        ]
        for vi in range(3):
            ren = self.renderers[vi]
            cam = ren.GetActiveCamera()
            cam.ParallelProjectionOn()
            fp = list(cam.GetFocalPoint())
            pscale = cam.GetParallelScale()

            offset, view_up = placement[vi]
            cam.SetPosition(fp[0] + offset[0], fp[1] + offset[1], fp[2] + offset[2])
            cam.SetViewUp(*view_up)

            cam.SetParallelScale(pscale)
            ren.ResetCameraClippingRange()

    def _setup_fixed_fov(self):
        """Fix camera focal point and FOV from full-volume dimensions.

        This keeps each view static in its pane while slices change.
        """
        self._apply_field_of_view()

    def get_field_of_view(self) -> Optional[float]:
        """Edge length in mm the panes are zoomed to, None when showing all."""
        return self._fov_mm

    def set_field_of_view(self, mm: Optional[float]):
        """Zoom every pane to an *mm* bounding box around the cursor.

        Follows the zoom of the SPM ortho viewer: the value is the edge length
        of the box in millimetres (e.g. 20 for 20x20 mm), and None shows the
        whole volume again.
        """
        self._fov_mm = float(mm) if mm else None
        self._apply_field_of_view(recenter=True)
        self.render_window.Render()

    def set_recenter(self, recenter: bool):
        """Whether a zoomed view follows the cursor.

        On (the default) the picked point stays in the middle of the pane; off
        keeps the view where it is, so the surroundings do not move away while
        clicking around.
        """
        self.recenter = bool(recenter)
        self._apply_field_of_view(recenter=self.recenter)
        self.render_window.Render()

    def _apply_field_of_view(self, recenter: bool = False):
        """Point every camera at the region the current zoom asks for.

        While zoomed the panes follow the cursor, so the picked point stays in
        the middle of the view; it lands exactly there because the cursor is
        not rounded to the voxel grid (see :meth:`_set_cursor`).  With
        :attr:`recenter` switched off the view stays where it is and only
        picking a zoom level (*recenter*) moves it.
        """
        if self._image is None:
            return
        ext = self._image.GetExtent()

        if self._fov_mm:
            # Zoomed: same box in every pane, centred on the cursor
            if recenter or self.recenter or self._fov_center is None:
                self._fov_center = self.get_world_position()
            focus = self._fov_center
            scales = [0.5 * self._fov_mm] * 3
        else:
            self._fov_center = None
            focus = None
            # Largest in-plane dimension per view, with a small margin
            lengths = self._voxel_axis_lengths()
            margin = 1.05
            scales = []
            for pane in range(3):
                in_plane = [a for a in range(3) if a != self._pane_axis[pane]]
                scales.append(0.5 * max(lengths[a] for a in in_plane) * margin)
        if focus is None:
            focus = self._world_from_index((
                0.5 * (ext[0] + ext[1]),
                0.5 * (ext[2] + ext[3]),
                0.5 * (ext[4] + ext[5]),
            ))

        for vi, ren in enumerate(self.renderers):
            cam = ren.GetActiveCamera()
            old_f = cam.GetFocalPoint()
            old_p = cam.GetPosition()
            offset = [old_p[i] - old_f[i] for i in range(3)]
            cam.SetFocalPoint(*focus)
            cam.SetPosition(*[focus[i] + offset[i] for i in range(3)])
            cam.SetParallelScale(scales[vi])
            ren.ResetCameraClippingRange()

    # -------- Crosshair helpers --------
    def _init_crosshair(self):
        """Create crosshair line actors in each viewer renderer."""
        extent = self._image.GetExtent()

        # Two line actors per viewer (horizontal and vertical in-plane)
        self._line_src = []
        self._line_act = []
        for vi in range(3):
            lr = []
            la = []
            for _ in range(2):
                ls = vtkLineSource()
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(ls.GetOutputPort())
                act = vtkActor()
                act.SetMapper(mapper)
                act.GetProperty().SetColor(1.0, 1.0, 0.0)
                act.GetProperty().SetLineWidth(0.8)
                self.renderers[vi].AddActor(act)
                lr.append(ls)
                la.append(act)
            self._line_src.append(lr)
            self._line_act.append(la)

        # Initial crosshair position: center index
        self._set_cursor([0.5 * (extent[0] + extent[1]),
                          0.5 * (extent[2] + extent[3]),
                          0.5 * (extent[4] + extent[5])])

    def _world_from_index(self, ijk: Tuple[int, int, int]):
        i, j, k = ijk
        # Prefer full 4x4 voxel-to-world
        if self._vox2world is not None:
            m = self._vox2world
            vx, vy, vz = float(i), float(j), float(k)
            wx = m[0][0]*vx + m[0][1]*vy + m[0][2]*vz + m[0][3]
            wy = m[1][0]*vx + m[1][1]*vy + m[1][2]*vz + m[1][3]
            wz = m[2][0]*vx + m[2][1]*vy + m[2][2]*vz + m[2][3]
            return (wx, wy, wz)
        # Fallback to origin/spacing/direction
        ox, oy, oz = self._image.GetOrigin()
        sx, sy, sz = self._image.GetSpacing()
        try:
            dm = self._image.GetDirectionMatrix()
            d = [[dm.GetElement(r, c) for c in range(3)] for r in range(3)]
        except Exception:
            d = None
        vx = i * sx; vy = j * sy; vz = k * sz
        if d is not None:
            wx = ox + d[0][0]*vx + d[0][1]*vy + d[0][2]*vz
            wy = oy + d[1][0]*vx + d[1][1]*vy + d[1][2]*vz
            wz = oz + d[2][0]*vx + d[2][1]*vy + d[2][2]*vz
            return (wx, wy, wz)
        return (ox + vx, oy + vy, oz + vz)

    def _world_from_index_center(self, ijk: Tuple[float, float, float]):
        i, j, k = ijk
        return self._world_from_index((i + 0.5, j + 0.5, k + 0.5))

    def _index_from_world(self, world: Tuple[float, float, float]):
        wx, wy, wz = world
        # Use inverse of voxel->world when available
        if self._vox2world is not None:
            m = self._vox2world
            R = [[m[0][0], m[0][1], m[0][2]],
                 [m[1][0], m[1][1], m[1][2]],
                 [m[2][0], m[2][1], m[2][2]]]
            t = [m[0][3], m[1][3], m[2][3]]
            v = [wx - t[0], wy - t[1], wz - t[2]]
            if np is not None:
                Rm = np.array(R, dtype=float)
                vm = np.array(v, dtype=float)
                try:
                    ijk = np.linalg.solve(Rm, vm)
                    return float(ijk[0]), float(ijk[1]), float(ijk[2])
                except Exception:
                    pass
            # Manual 3x3 inverse fallback
            det = (R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])
                   - R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])
                   + R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0]))
            if det != 0:
                inv = [[0.0]*3 for _ in range(3)]
                inv[0][0] = (R[1][1]*R[2][2]-R[1][2]*R[2][1])/det
                inv[0][1] = (R[0][2]*R[2][1]-R[0][1]*R[2][2])/det
                inv[0][2] = (R[0][1]*R[1][2]-R[0][2]*R[1][1])/det
                inv[1][0] = (R[1][2]*R[2][0]-R[1][0]*R[2][2])/det
                inv[1][1] = (R[0][0]*R[2][2]-R[0][2]*R[2][0])/det
                inv[1][2] = (R[0][2]*R[1][0]-R[0][0]*R[1][2])/det
                inv[2][0] = (R[1][0]*R[2][1]-R[1][1]*R[2][0])/det
                inv[2][1] = (R[0][1]*R[2][0]-R[0][0]*R[2][1])/det
                inv[2][2] = (R[0][0]*R[1][1]-R[0][1]*R[1][0])/det
                i = inv[0][0]*v[0] + inv[0][1]*v[1] + inv[0][2]*v[2]
                j = inv[1][0]*v[0] + inv[1][1]*v[1] + inv[1][2]*v[2]
                k = inv[2][0]*v[0] + inv[2][1]*v[1] + inv[2][2]*v[2]
                return i, j, k
        # Fallback using origin/spacing/direction (approximate inverse assuming orthonormal)
        ox, oy, oz = self._image.GetOrigin()
        sx, sy, sz = self._image.GetSpacing()
        try:
            dm = self._image.GetDirectionMatrix()
            d = [[dm.GetElement(r, c) for c in range(3)] for r in range(3)]
            # transpose for inverse if orthonormal
            vx = wx - ox; vy = wy - oy; vz = wz - oz
            i = (d[0][0]*vx + d[1][0]*vy + d[2][0]*vz) / sx
            j = (d[0][1]*vx + d[1][1]*vy + d[2][1]*vz) / sy
            k = (d[0][2]*vx + d[1][2]*vy + d[2][2]*vz) / sz
            return i, j, k
        except Exception:
            # axis-aligned fallback
            return ( (wx - ox)/sx, (wy - oy)/sy, (wz - oz)/sz )

    def _camera_offset(self, view_idx: int, distance: float = 0.5):
        """Return a small world-space shift towards the camera of *view_idx*.

        The crosshair lies exactly in the slice plane, where the depth buffer
        lets the image win.  Offsetting it towards the camera keeps it visible
        for any image orientation, whereas a fixed index-space offset points
        away from the camera as soon as the corresponding voxel axis is
        flipped — which is what hid the crosshair in the coronal view.
        """
        cam = self.renderers[view_idx].GetActiveCamera()
        px, py, pz = cam.GetPosition()
        fx, fy, fz = cam.GetFocalPoint()
        dx, dy, dz = px - fx, py - fy, pz - fz
        norm = (dx * dx + dy * dy + dz * dz) ** 0.5
        if norm < 1e-12:
            return (0.0, 0.0, 0.0)
        s = distance / norm
        return (dx * s, dy * s, dz * s)

    def _update_crosshair_lines(self):
        extent = self._image.GetExtent()
        exact = self.get_index_exact() or self._ijk

        def w(ijk, off):
            wx, wy, wz = self._world_from_index(tuple(ijk))
            return (wx + off[0], wy + off[1], wz + off[2])

        # Each pane gets one line along either in-plane voxel axis, both
        # crossing at the cursor and spanning the whole image.  In the plane
        # the exact cursor is used, across it the displayed slice, so the lines
        # stay on the image they are drawn over.
        for pane in range(3):
            fixed = self._pane_axis[pane]
            in_plane = [a for a in range(3) if a != fixed]
            off = self._camera_offset(pane)
            base = list(exact)
            base[fixed] = self._ijk[fixed]
            for line, axis in zip(self._line_src[pane], in_plane):
                p1 = list(base); p2 = list(base)
                p1[axis] = extent[2 * axis]
                p2[axis] = extent[2 * axis + 1]
                line.SetPoint1(*w(p1, off))
                line.SetPoint2(*w(p2, off))
                line.Modified()

        # Trigger re-render
        self.render_window.Render()

    def _build_surface_contours(self):
        # Remove existing contour actors
        for vi in range(3):
            ren = self.renderers[vi]
            for entry in self._surface_contours[vi]:
                try:
                    ren.RemoveActor(entry.get('actor'))
                except Exception:
                    pass
            self._surface_contours[vi] = []

        # World-space normal of the voxel axis each pane slices along
        dirs = self._voxel_axis_directions()
        normals = [dirs[self._pane_axis[vi]] for vi in range(3)]
        # Origins must match the actual slice location used by the image actors,
        # which slices at integer voxel indices. Do NOT add the +0.5 voxel-center shift here.
        origins = [self._world_from_index(tuple(self._ijk))] * 3

        for (poly, color) in self.surfaces:
            for vi in range(3):
                plane = vtkPlane()
                plane.SetNormal(*normals[vi])
                plane.SetOrigin(*origins[vi])
                cutter = vtkCutter()
                cutter.SetCutFunction(plane)
                cutter.SetInputData(poly)
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(cutter.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(color)
                actor.GetProperty().SetLineWidth(1.2)
                actor.GetProperty().LightingOff()
                # The contour lies exactly in the slice plane, where the image
                # would win the depth test; nudge it towards the camera
                actor.SetPosition(*self._camera_offset(vi))
                self.renderers[vi].AddActor(actor)
                self._surface_contours[vi].append({'plane': plane, 'cutter': cutter, 'actor': actor})

    def _update_surface_planes(self):
        if not self.surfaces:
            return
        i, j, k = self._ijk
        # Keep plane origins in sync with viewer slice indices (integer index positions)
        origins = [
            self._world_from_index((i, j, k)),
            self._world_from_index((i, j, k)),
            self._world_from_index((i, j, k)),
        ]
        for vi in range(3):
            for entry in self._surface_contours[vi]:
                plane = entry.get('plane')
                if plane is not None:
                    plane.SetOrigin(*origins[vi])
                    plane.Modified()
        # trigger rerender
        self.render_window.Render()

    def _set_slices_from_index(self):
        """Update each image actor's display extent to show the current
        slice, then re-render."""
        ext = list(self._image.GetExtent())

        for pane in range(3):
            axis = self._pane_axis[pane]
            display = list(ext)
            display[2 * axis] = display[2 * axis + 1] = self._ijk[axis]
            self._image_actors[pane].SetDisplayExtent(*display)
            if self._overlay_actors[pane] is not None:
                self._overlay_actors[pane].SetDisplayExtent(*display)

        # A zoomed view stays centred on the cursor
        if self._fov_mm:
            self._apply_field_of_view()

        self._update_info_text()

        # Keep clipping valid but do not move camera/focal point.
        for ren in self.renderers:
            ren.ResetCameraClippingRange()

        self._update_surface_planes()
        self.render_window.Render()

    def _on_click(self, view_idx: int, x: int, y: int):
        """Handle a click/drag at window pixel *(x, y)* in *view_idx*."""
        if self._ijk is None:
            return
        ren = self.renderers[view_idx]

        # --- Build a world-space ray from the click position ----------
        def _display_to_world(dx, dy, dz):
            ren.SetDisplayPoint(dx, dy, dz)
            ren.DisplayToWorld()
            wp = ren.GetWorldPoint()
            ww = wp[3] if wp[3] != 0 else 1.0
            return (wp[0] / ww, wp[1] / ww, wp[2] / ww)

        p1 = _display_to_world(x, y, 0)
        p2 = _display_to_world(x, y, 1)

        # --- Intersect ray with the current slice plane ---------------
        # The plane normal is the world direction of the voxel axis this pane
        # slices along
        n = self._voxel_axis_directions()[self._pane_axis[view_idx]]
        p0 = self._world_from_index(tuple(self._ijk))

        # Line-plane intersection:  p = p1 + t * (p2 - p1)
        v = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        w = (p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2])
        denom = n[0] * v[0] + n[1] * v[1] + n[2] * v[2]
        if abs(denom) < 1e-12:
            return  # ray nearly parallel to slice plane
        t = (n[0] * w[0] + n[1] * w[1] + n[2] * w[2]) / denom
        wx = p1[0] + t * v[0]
        wy = p1[1] + t * v[1]
        wz = p1[2] + t * v[2]

        # --- Convert world position to voxel indices ------------------
        # Kept fractional, so the crosshair sits exactly under the mouse
        self._set_cursor(self._index_from_world((wx, wy, wz)), notify=True)
        if self.verbose:
            print(f"[cat_vol_view] Click -> ijk={self.get_index()}")

    # ---------- Atlas lookup ----------
    @staticmethod
    def available_atlases() -> List[Tuple[str, str]]:
        """The volume atlases shipped with T1Prep as (name, path) pairs.

        Only atlases with a region list (``<name>.csv``) are offered, since
        without it there is nothing to report at the cursor.
        """
        folder = (Path(__file__).resolve().parent.parent / 'data'
                  / 'templates_MNI152NLin2009cAsym')
        out: List[Tuple[str, str]] = []
        if not folder.is_dir():
            return out
        for volume in sorted(folder.glob('*.nii.gz')):
            name = volume.name[:-len('.nii.gz')]
            if (folder / f'{name}.csv').exists():
                out.append((name, str(volume)))
        return out

    @staticmethod
    def _read_region_names(csv_path: str) -> dict:
        """Parse a T1Prep region list.

        The files are semicolon separated and start with a header naming the
        columns; which column holds the name differs between atlases
        (``ROIid;ROIabbr;ROIname;…`` vs ``ROIid;ROIname;…``), so it is taken
        from the header rather than assumed.
        """
        names = {}
        name_col = 2
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = [p.strip() for p in line.strip().split(";")]
                    if len(parts) < 2:
                        continue
                    if parts[0].lower() == "roiid":
                        lowered = [p.lower() for p in parts]
                        if "roiname" in lowered:
                            name_col = lowered.index("roiname")
                        continue
                    try:
                        roi_id = int(parts[0])
                    except ValueError:
                        continue
                    if name_col < len(parts):
                        names[roi_id] = parts[name_col]
        except Exception:
            return {}
        return names

    def set_atlas(self, atlas_path: Optional[str]):
        """Name the region under the cursor using *atlas_path*.

        The atlas is sampled at the world (mm) position of the cursor, so it
        only says something meaningful when the displayed image is registered
        to the space of that atlas — which is why it is chosen explicitly
        rather than guessed.  Pass None to switch the lookup off.
        """
        self._atlas = None
        self.atlas_path = None
        if atlas_path:
            try:
                reader = _guess_image_reader(atlas_path)
                reader.Update()
                image = reader.GetOutput()
                # VTK reports a read failure through empty dimensions
                if image is None or image.GetDimensions() == (0, 0, 0):
                    raise RuntimeError("image is empty or unreadable")
                vox2world, _ = _voxel_to_world_matrix(reader, image)
                world2vox = np.linalg.inv(np.array(vox2world, dtype=float))
                stem = os.path.basename(atlas_path)
                for suffix in ('.nii.gz', '.nii', '.mnc'):
                    if stem.endswith(suffix):
                        stem = stem[:-len(suffix)]
                        break
                names = self._read_region_names(
                    os.path.join(os.path.dirname(atlas_path), stem + '.csv'))
                self._atlas = {'image': image, 'world2vox': world2vox,
                               'names': names, 'name': stem}
                self.atlas_path = atlas_path
            except Exception as exc:
                if self.verbose:
                    print(f"[cat_vol_view] Could not load atlas {atlas_path}: {exc}")
                self._atlas = None
        self._update_info_text()
        self.render_window.Render()

    def _atlas_region(self) -> Optional[str]:
        """Region name of the selected atlas at the cursor, if any."""
        if not self._atlas:
            return None
        world = self.get_world_position()
        if world is None:
            return None
        try:
            ijk = self._atlas['world2vox'] @ np.array([*world, 1.0])
            index = [int(round(v)) for v in ijk[:3]]
            ext = self._atlas['image'].GetExtent()
            for axis in range(3):
                if not (ext[2 * axis] <= index[axis] <= ext[2 * axis + 1]):
                    return None  # outside the atlas
            value = int(round(self._atlas['image'].GetScalarComponentAsDouble(
                index[0], index[1], index[2], 0)))
        except Exception:
            return None
        if value == 0:
            return None
        return self._atlas['names'].get(value, f"label {value}")

    # ---------- Image information panel ----------
    def _orientation_code(self, image_path: str) -> Optional[str]:
        """Anatomical order of the file's voxel axes, e.g. ``LAS``."""
        try:
            if nib is None:
                return None
            return "".join(nib.aff2axcodes(nib.load(image_path).affine))
        except Exception:
            return None

    def _static_info_lines(self) -> List[str]:
        """Properties of the image itself (fixed while it is displayed)."""
        dims = self._image.GetDimensions()
        voxel = self._voxel_axis_lengths()
        ext = self._image.GetExtent()
        size = [voxel[a] / max(1, ext[2 * a + 1] - ext[2 * a] + 1) for a in range(3)]
        lo, hi = self._image.GetScalarRange()
        lines = [
            self._image_name,
            "",
            f"dimensions  {dims[0]} x {dims[1]} x {dims[2]}",
            f"voxel size  {size[0]:.3g} x {size[1]:.3g} x {size[2]:.3g} mm",
        ]
        if self._orientation:
            lines.append(f"orientation {self._orientation}")
        lines.append(f"data type   {self._image.GetScalarTypeAsString()}")
        lines.append(f"intensity   {lo:g} .. {hi:g}")
        return lines

    def _cursor_info_lines(self) -> List[str]:
        """Everything that changes with the cursor."""
        ijk = self.get_index()
        world = self.get_world_position()
        if ijk is None or world is None:
            return []
        value = self.get_value()
        lines = [
            "",
            f"voxel       [{ijk[0]}, {ijk[1]}, {ijk[2]}]",
            f"mm          ({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f})",
            f"value       {value:g}" if value is not None else "value       -",
        ]
        if self._overlay_image is not None:
            # 'value' is the overlay above; keep the image underneath visible
            background = self.get_background_value()
            lines.append(f"background  {background:g}" if background is not None
                         else "background  -")
            lines.insert(1, f"overlay     {self.overlay_name}")
        if self._atlas:
            region = self._atlas_region()
            lines.append(f"atlas       {self._atlas['name']}")
            lines.append(f"region      {region or '-'}")
        return lines

    def _fit_info_font(self, lines: Sequence[str]):
        """Size the panel text so the longest line fits its quadrant.

        Long file names, atlas region names and small windows all change what
        fits, so the size is derived from the text rather than fixed.
        """
        if self._info_actor is None:
            return
        try:
            width = self.render_window.GetSize()[0] * (1.0 - self._viewports[0][2])
            # VTK scales the point size with the window DPI, and Courier glyphs
            # are about 0.6 em wide
            scale = float(self.render_window.GetDPI() or 72) / 72.0
            longest = max((len(line) for line in lines), default=1)
            size = 0.9 * width / max(1.0, longest * 0.6 * scale)
            self._info_actor.GetTextProperty().SetFontSize(int(max(7, min(16, size))))
        except Exception:
            pass

    def _update_info_text(self):
        """Redraw the information panel in the free quadrant."""
        if self._info_actor is None or self._image is None:
            return
        if not self.show_info:
            self._info_actor.SetVisibility(False)
            return
        self._info_actor.SetVisibility(True)
        lines = self._static_info_lines() + self._cursor_info_lines()
        self._fit_info_font(lines)
        self._info_actor.SetInput("\n".join(lines))

    def set_info_visible(self, visible: bool):
        """Show or hide the information panel."""
        self.show_info = bool(visible)
        self._update_info_text()
        self.render_window.Render()

    # ---------- Display range of the image ----------
    def get_window_level(self) -> Tuple[float, float]:
        """Window and level the image is displayed with."""
        return (float(self._wl[0]), float(self._wl[1]))

    def set_window_level(self, window: float, level: float):
        """Change the displayed intensity range of the image."""
        self._wl = (float(window), float(level))
        for filt in self._wl_filters:
            if filt is not None:
                filt.SetWindow(self._wl[0])
                filt.SetLevel(self._wl[1])
        self._update_info_text()
        self.render_window.Render()

    # ---------- Overlay volume ----------
    def set_overlay(self, overlay_path: Optional[str]):
        """Colour a second volume on top of the displayed one.

        The overlay has to sit on the same voxel grid as the image (same
        dimensions and voxel size); anything else would need resampling, which
        this viewer deliberately does not do.  Pass None to remove it.

        Raises:
            ValueError: when the file cannot be read or the grids differ.
        """
        self._remove_overlay_actors()
        self.overlay_path = None
        self.overlay_name = ""
        self._overlay_image = None
        if not overlay_path:
            self._update_info_text()
            self.render_window.Render()
            return

        reader = _guess_image_reader(overlay_path)
        reader.Update()
        image = reader.GetOutput()
        if image is None or image.GetDimensions() == (0, 0, 0):
            raise ValueError(f"cannot read {os.path.basename(overlay_path)}")
        dims, own = image.GetDimensions(), self._image.GetDimensions()
        if dims != own:
            raise ValueError(f"dimensions {dims} differ from {own}")
        spacing, own_spacing = image.GetSpacing(), self._image.GetSpacing()
        if any(abs(a - b) > 1e-4 for a, b in zip(spacing, own_spacing)):
            raise ValueError(
                f"voxel size {tuple(round(v, 3) for v in spacing)} differs from "
                f"{tuple(round(v, 3) for v in own_spacing)}")

        # Same grid, so the geometry of the displayed image applies to both
        image.SetOrigin(0.0, 0.0, 0.0)
        try:
            ident = vtkMatrix3x3()
            ident.Identity()
            image.SetDirectionMatrix(ident)
        except Exception:
            pass
        self._overlay_image = image
        self.overlay_path = overlay_path
        self.overlay_name = os.path.basename(overlay_path)
        lo, hi = image.GetScalarRange()
        self.overlay_range = [float(lo), float(hi)]
        self._build_overlay_actors()
        self._update_info_text()
        self.render_window.Render()

    def _remove_overlay_actors(self):
        for pane, actor in enumerate(self._overlay_actors):
            if actor is not None:
                try:
                    self.renderers[pane].RemoveActor(actor)
                except Exception:
                    pass
        self._overlay_actors = [None, None, None]
        self._overlay_colors = [None, None, None]

    def _overlay_lut(self):
        """Lookup table for the overlay — the same one the surface viewer uses.

        Clipped values are transparent, so the image shows through them, and
        voxels the overlay has no value for (NaN) are not painted at all.
        """
        return build_overlay_lut(
            self.overlay_colormap, self.overlay_opacity,
            value_range=self.overlay_range, clip=self.overlay_clip,
            inverse=self.overlay_inverse, discrete=self.overlay_discrete)

    def _build_overlay_actors(self):
        """One coloured slice actor per pane, drawn over the image."""
        self._remove_overlay_actors()
        if self._overlay_image is None:
            return
        lut = self._overlay_lut()
        for pane in range(3):
            colors = vtkImageMapToColors()
            colors.SetInputData(self._overlay_image)
            colors.SetLookupTable(lut)
            colors.SetOutputFormatToRGBA()
            colors.Update()
            actor = vtkImageActor()
            actor.GetMapper().SetInputConnection(colors.GetOutputPort())
            # Lift it off the image plane so it wins the depth test.  The shift
            # has to go into the world transform: a plain position would be
            # taken in the actor's own frame, where the header transform can
            # turn it away from the camera again.
            offset = self._camera_offset(pane, 0.25)
            if self._world_from_header:
                matrix = self._world_matrix()
                for row in range(3):
                    matrix.SetElement(row, 3, matrix.GetElement(row, 3) + offset[row])
                actor.SetUserMatrix(matrix)
            else:
                actor.SetPosition(*offset)
            self.renderers[pane].AddActor(actor)
            self._overlay_actors[pane] = actor
            self._overlay_colors[pane] = colors
        self._apply_interpolation()
        self._set_slices_from_index()

    def refresh_overlay(self):
        """Re-colour the overlay after range, clip or colormap changed."""
        if self._overlay_image is None:
            return
        lut = self._overlay_lut()
        for colors in self._overlay_colors:
            if colors is not None:
                colors.SetLookupTable(lut)
                colors.Modified()
        self._update_info_text()
        self.render_window.Render()

    # ---------- Display ----------
    def _apply_interpolation(self):
        for actor in self._image_actors:
            if actor is None:
                continue
            try:
                actor.SetInterpolate(1 if self.interpolate else 0)
            except Exception:
                pass
        # The overlay is never smoothed: interpolating a thresholded map would
        # invent values across its edges
        for actor in self._overlay_actors:
            if actor is None:
                continue
            try:
                actor.SetInterpolate(0)
            except Exception:
                pass

    def set_interpolation(self, interpolate: bool):
        """Draw the slices smoothed (linear) or as raw voxels (nearest).

        Nearest neighbour shows the data as it is stored, which is what you
        want when judging segmentation edges or resampling artefacts.
        """
        self.interpolate = bool(interpolate)
        self._apply_interpolation()
        self._update_info_text()   # the reported value follows the display
        self.render_window.Render()

    # ---------- Cursor position ----------
    def _notify_position(self):
        """Report the current cursor to a linked viewer, if one is attached."""
        cb = self.on_position_changed
        if cb is None:
            return
        world = self.get_world_position()
        if world is None:
            return
        try:
            cb(world)
        except Exception:
            pass

    def get_index(self) -> Optional[Tuple[int, int, int]]:
        """Voxel the cursor is in, or None before setup().

        This is the rounded position; the cursor itself is not tied to the
        voxel grid (see :meth:`get_world_position`).
        """
        if self._ijk is None:
            return None
        return (int(self._ijk[0]), int(self._ijk[1]), int(self._ijk[2]))

    def get_index_exact(self) -> Optional[Tuple[float, float, float]]:
        """Cursor in (fractional) voxel index coordinates."""
        if self._cursor is None or self._image is None:
            return None
        return tuple(self._index_from_world(self._cursor))

    def get_world_position(self) -> Optional[Tuple[float, float, float]]:
        """Current cursor in world (mm) coordinates, or None before setup().

        The position is exact: it is where the user clicked, not the centre of
        the voxel that was hit.  Only the slices shown and the intensity read
        out use the rounded index, because those are voxel-wise by nature.
        """
        if self._cursor is None:
            return None
        return tuple(self._cursor)

    def get_value_at_index(self, ijk: Optional[Tuple[int, int, int]] = None):
        """Raw image intensity at *ijk* (default: the cursor's voxel)."""
        if self._image is None:
            return None
        if ijk is None:
            ijk = self.get_index()
        if ijk is None:
            return None
        try:
            return float(self._image.GetScalarComponentAsDouble(
                int(ijk[0]), int(ijk[1]), int(ijk[2]), 0))
        except Exception:
            return None

    def _sample(self, image):
        """Intensity of *image* at the cursor, sampled the way it is drawn.

        With smoothing on this is the trilinear value at the exact cursor
        position, so the number matches what is displayed; with raw voxels
        selected it is the untouched value of the voxel the cursor is in.
        """
        index = self.get_index_exact()
        if image is None or index is None:
            return None
        ext = image.GetExtent()
        if not self.interpolate:
            ijk = [max(ext[2 * a], min(ext[2 * a + 1], int(round(index[a]))))
                   for a in range(3)]
            try:
                return float(image.GetScalarComponentAsDouble(*ijk, 0))
            except Exception:
                return None
        base, frac = [], []
        for axis in range(3):
            lo, hi = ext[2 * axis], ext[2 * axis + 1]
            pos = min(max(index[axis], lo), hi)
            low = min(int(math.floor(pos)), hi - 1) if hi > lo else lo
            base.append(low)
            frac.append(pos - low)
        try:
            value = 0.0
            for di, dj, dk in itertools.product((0, 1), repeat=3):
                weight = ((1.0 - frac[0] if di == 0 else frac[0])
                          * (1.0 - frac[1] if dj == 0 else frac[1])
                          * (1.0 - frac[2] if dk == 0 else frac[2]))
                if weight == 0.0:
                    continue
                value += weight * image.GetScalarComponentAsDouble(
                    min(base[0] + di, ext[1]),
                    min(base[1] + dj, ext[3]),
                    min(base[2] + dk, ext[5]), 0)
            return float(value)
        except Exception:
            return None

    def get_value(self):
        """Intensity at the cursor: the overlay's when one is loaded."""
        if self._overlay_image is not None:
            return self._sample(self._overlay_image)
        return self._sample(self._image)

    def get_background_value(self):
        """Intensity of the displayed image, even when an overlay covers it."""
        return self._sample(self._image)

    def get_overlay_value(self):
        """Overlay intensity at the cursor, or None without an overlay."""
        return self._sample(self._overlay_image)

    def _set_cursor(self, index: Sequence[float], notify: bool = False):
        """Place the cursor at a (fractional) voxel index, clamped to the image.

        The exact position is kept: the crosshair, the reported millimetres and
        a zoomed view all use it, so a click lands where it was made.  The
        rounded index drives the displayed slices and the intensity readout.
        """
        if self._image is None:
            return
        ext = self._image.GetExtent()
        exact = [max(ext[2 * a], min(ext[2 * a + 1], float(index[a])))
                 for a in range(3)]
        self._cursor = list(self._world_from_index(tuple(exact)))
        self._ijk = [int(round(v)) for v in exact]
        self._set_slices_from_index()
        self._update_crosshair_lines()
        if notify:
            self._notify_position()

    def set_index(self, i: float, j: float, k: float, notify: bool = False):
        """Move the cursor to a voxel index, clamped to the image extent."""
        self._set_cursor((i, j, k), notify=notify)

    def set_world_position(self, world: Tuple[float, float, float],
                           notify: bool = False):
        """Move the cursor to a world (mm) position.

        This is the entry point used to link the viewer to another window:
        picking a point on the surface in CAT_SurfView moves the slices here.
        """
        if self._image is None:
            return
        self._set_cursor(self._index_from_world(tuple(world)), notify=notify)

    # ---------- Public API ----------
    def load_image(self, image_path: str):
        if self.verbose:
            print(f"[cat_vol_view] Loading image: {image_path}")
        reader = _guess_image_reader(image_path)
        reader.Update()
        self._image = reader.GetOutput()
        if self._image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        dims = self._image.GetDimensions()
        if self.verbose:
            print(f"[cat_vol_view] Image dimensions: {dims}")
            try:
                print("[cat_vol_view] VTK image origin:", self._image.GetOrigin())
                print("[cat_vol_view] VTK image spacing:", self._image.GetSpacing())
                dm = self._image.GetDirectionMatrix()
                if dm is not None:
                    D = [[dm.GetElement(r, c) for c in range(3)] for r in range(3)]
                    print("[cat_vol_view] VTK image direction:")
                    for r in D:
                        print("    ", r)
            except Exception:
                pass
        if dims == (0, 0, 0):
            raise RuntimeError(
                f"Image appears empty or unreadable (dims={dims}): {image_path}"
            )
        self._vox2world, self._world_from_header = _voxel_to_world_matrix(
            reader, self._image)
        self._actor_matrix = None
        if self._world_from_header:
            # The actors get the transform as the header stores it (they apply
            # the spacing themselves), so the image must not carry an origin or
            # orientation on top of it.
            self._actor_matrix = _header_matrix(reader)
            self._image.SetOrigin(0.0, 0.0, 0.0)
            try:
                ident = vtkMatrix3x3()
                ident.Identity()
                self._image.SetDirectionMatrix(ident)
            except Exception:
                pass
        if self.verbose:
            source = "header" if self._world_from_header else "VTK image properties"
            print(f"[cat_vol_view] Using voxel->world from {source}:")
            for row in self._vox2world:
                print("    ", row)
        self._assign_pane_axes()

        # Details for the information panel
        self._image_name = os.path.basename(image_path)
        self._orientation = self._orientation_code(image_path)

        # Window/level – prefer percentile-based scaling
        rng = self._image.GetScalarRange()
        self._wl = (float(rng[1] - rng[0]),
                     float(0.5 * (rng[1] + rng[0])))
        if (self.percentile_range is not None
                and np is not None and vtk_to_numpy is not None):
            try:
                scalars = self._image.GetPointData().GetScalars()
                if scalars is not None:
                    arr = vtk_to_numpy(scalars).ravel().astype(float)
                    lo_pct, hi_pct = np.percentile(
                        arr, list(self.percentile_range))
                    if hi_pct > lo_pct:
                        self._wl = (float(hi_pct - lo_pct),
                                     float(0.5 * (hi_pct + lo_pct)))
                        if self.verbose:
                            print(
                                f"[cat_vol_view] Percentile "
                                f"{self.percentile_range}: "
                                f"window={self._wl[0]:.1f}, "
                                f"level={self._wl[1]:.1f}")
            except Exception as exc:
                if self.verbose:
                    print(f"[cat_vol_view] Percentile scaling failed: {exc}")
        return self

    def add_surface(self, surface: "str | vtkPolyData", color: Tuple[float, float, float]):
        if isinstance(surface, vtkPolyData):
            poly = surface
        else:
            poly = _load_surface(surface)
        # Optionally convert surface convention (LPS<->RAS)
        poly = self._apply_surface_convention(poly)
        if self.mirror_surfaces:
            poly = _mirror_polydata_x(poly)
            if self.verbose:
                print("[cat_vol_view] Applied surface mirroring (scale -1,1,1)")
        # Heuristic: if the surface center is far from the image center in world coords,
        # translate it so centers coincide (common when surfaces are in 0..FOV vs centered at 0).
        # Skipped once the image carries its own millimetre transform: surface
        # and image then share one space and must not be nudged apart.
        try:
            if (self._image is not None and self._vox2world is not None
                    and not self._world_from_header):
                extent = self._image.GetExtent()
                cx = 0.5 * (extent[0] + extent[1])
                cy = 0.5 * (extent[2] + extent[3])
                cz = 0.5 * (extent[4] + extent[5])
                img_center = self._world_from_index_center((cx, cy, cz))
                # Surface center from bounds
                b = [0.0]*6
                poly.GetBounds(b)
                surf_center = ((b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, (b[4] + b[5]) * 0.5)
                # Compute image world bounding box diagonal length as scale reference
                w000 = self._world_from_index((extent[0], extent[2], extent[4]))
                w111 = self._world_from_index((extent[1], extent[3], extent[5]))
                diag = ((w111[0]-w000[0])**2 + (w111[1]-w000[1])**2 + (w111[2]-w000[2])**2) ** 0.5
                # If centers differ by more than 20% of diag (or 10mm minimum), recenter
                dx = img_center[0] - surf_center[0]
                dy = img_center[1] - surf_center[1]
                dz = img_center[2] - surf_center[2]
                dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                if dist > max(10.0, 0.2 * diag):
                    t = vtkTransform()
                    t.Translate(dx, dy, dz)
                    f = vtkTransformPolyDataFilter()
                    f.SetInputData(poly)
                    f.SetTransform(t)
                    f.Update()
                    poly = f.GetOutput()
                    if self.verbose:
                        print(f"[cat_vol_view] Recentered surface by translation (dx,dy,dz)=({dx:.3f},{dy:.3f},{dz:.3f})")
        except Exception:
            pass
        self.surfaces.append((poly, color))
        return self

    def _apply_surface_convention(self, poly: vtkPolyData) -> vtkPolyData:
        conv = self.surface_convention
        if conv == "none" or conv == "auto":
            return poly
        # We treat image voxel-to-world as RAS (NIfTI q/sform standard). Convert surfaces to RAS if needed.
        if conv == "lps":
            t = vtkTransform()
            # LPS -> RAS: flip X and Y
            t.Scale(-1.0, -1.0, 1.0)
            f = vtkTransformPolyDataFilter()
            f.SetInputData(poly)
            f.SetTransform(t)
            f.Update()
            if self.verbose:
                print("[cat_vol_view] Converted surface from LPS to RAS (flip X,Y)")
            return f.GetOutput()
        # conv == "ras" -> assume already RAS
        return poly

    def setup(self, window_title: Optional[str] = None):
        """Set up the single-window viewer with SPM12-like viewport layout.

        Parameters
        ----------
        window_title : str, optional
            Title for the render window (defaults to ``"Ortho Viewer"``).
        """
        if self._image is None:
            raise RuntimeError("No image loaded. Call load_image() first.")

        # Compute proportional viewports & pixel window size.  An embedded
        # viewer only takes the viewport layout; size and title belong to the
        # host widget.
        win_w, win_h = self._compute_viewports()
        # Kept so an embedding host can size its widget with the same
        # proportions the standalone window would use
        self.window_pixel_size = (win_w, win_h)
        if not self.embedded:
            self.render_window.SetSize(win_w, win_h)
            self.render_window.SetWindowName(window_title or "Ortho Viewer")

        ext = self._image.GetExtent()

        for i in range(3):
            # Create image actor with window/level ---
            actor = vtkImageActor()
            if vtkImageMapToWindowLevelColors is not None:
                wlf = vtkImageMapToWindowLevelColors()
                wlf.SetInputData(self._image)
                wlf.SetWindow(self._wl[0])
                wlf.SetLevel(self._wl[1])
                actor.GetMapper().SetInputConnection(wlf.GetOutputPort())
                self._wl_filters[i] = wlf
            else:
                actor.SetInputData(self._image)
                try:
                    prop = actor.GetProperty()
                    prop.SetColorWindow(self._wl[0])
                    prop.SetColorLevel(self._wl[1])
                except Exception:
                    pass
            if self._world_from_header:
                # The image was reset to voxel geometry in load_image(), so the
                # actor carries the millimetre transform from the NIfTI header
                actor.SetUserMatrix(self._world_matrix())
            self._image_actors[i] = actor

            ren = self.renderers[i]
            ren.AddActor(actor)
            ren.SetBackground(0, 0, 0)
            self.render_window.AddRenderer(ren)

            if self.verbose:
                names = ["Axial", "Sagittal", "Coronal"]
                print(f"[cat_vol_view] Configured {names[i]} "
                      f"viewport {self._viewports[i]}")

        # The free bottom-right quadrant carries the image information
        info_ren = vtkRenderer()
        info_ren.SetBackground(0, 0, 0)
        info_ren.SetInteractive(0)
        self.render_window.AddRenderer(info_ren)
        self._info_renderer = info_ren
        self._apply_viewports()

        info = vtkTextActor()
        prop = info.GetTextProperty()
        prop.SetFontFamilyToCourier()   # keeps the values in a column
        prop.SetFontSize(13)
        prop.SetColor(0.85, 0.85, 0.85)
        prop.SetVerticalJustificationToTop()
        coord = info.GetPositionCoordinate()
        coord.SetCoordinateSystemToNormalizedViewport()
        coord.SetValue(0.06, 0.94)
        info_ren.AddViewProp(info)
        self._info_actor = info

        # Set initial crosshair to the centre of the volume; this also gives
        # the actors a display extent, so ResetCamera has valid bounds
        self._ijk = [(ext[0] + ext[1]) // 2, (ext[2] + ext[3]) // 2,
                     (ext[4] + ext[5]) // 2]
        self._cursor = list(self._world_from_index(tuple(self._ijk)))
        self._set_slices_from_index()

        # Reset cameras to fill each viewport, then enforce SPM12
        for ren in self.renderers:
            ren.ResetCamera()
        self._setup_cameras_spm12()
        self._setup_fixed_fov()

        # Connect interactor AFTER creating actors to avoid any
        # internal event-observer interference
        if not self.embedded:
            self.interactor.SetRenderWindow(self.render_window)
        style = _OrthoStyle(self)
        self.interactor.SetInteractorStyle(style)
        self._ortho_style = style  # prevent garbage collection
        try:
            self.interactor.Initialize()
        except Exception:
            pass
        self._bind_interaction_events()

        self._apply_interpolation()

        # Crosshair overlays & surface contours
        self._init_crosshair()
        if self.surfaces:
            self._build_surface_contours()
        self._set_slices_from_index()

        return self

    def render(self, *, screenshot: Optional[str] = None,
               headless: bool = False):
        """Render the viewer and optionally save a screenshot.

        Parameters
        ----------
        screenshot : str, optional
            Path for a single combined PNG screenshot of all three views.
        headless : bool
            If ``True``, enable off-screen rendering and do not start the
            event loop.
        """
        use_offscreen = bool(headless or screenshot)
        if use_offscreen:
            try:
                self.render_window.SetOffScreenRendering(1)
            except Exception:
                pass
            try:
                self.render_window.SetMultiSamples(0)
            except Exception:
                pass

        self.render_window.Render()

        if screenshot:
            w2i = vtkWindowToImageFilter()
            w2i.SetInput(self.render_window)
            try:
                w2i.ReadFrontBufferOff()
            except Exception:
                pass
            try:
                w2i.SetInputBufferTypeToRGBA()
            except Exception:
                pass
            w2i.Update()
            writer = vtkPNGWriter()
            if not screenshot.lower().endswith(".png"):
                screenshot += ".png"
            writer.SetFileName(screenshot)
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()
            if self.verbose:
                print(f"[cat_vol_view] Wrote screenshot: {screenshot}")

        if not headless:
            if self.verbose:
                print("[cat_vol_view] Starting interactor…")
            self.interactor.Start()


# ------------------------------------------------------------------ #
#  Interaction style                                                  #
# ------------------------------------------------------------------ #

class _OrthoStyle(vtkInteractorStyleImage):
    """Custom interactor style for the combined orthogonal viewer.

    * Left-click: handled by explicit interactor observers in CatImageViewer.
    * Mouse-wheel: handled by explicit interactor observers in CatImageViewer.
    * Middle / right button: default pan / zoom from parent class.

    We intentionally do **not** call ``super().OnLeftButtonDown()`` because
    the base ``vtkInteractorStyleImage`` would enter window/level adjust
    mode and swallow all subsequent events.
    """

    def __init__(self, parent: CatImageViewer):
        super().__init__()
        self._parent = parent

    # -- left button: no-op here; handled by observers --
    def OnLeftButtonDown(self):
        super().OnLeftButtonDown()

    def OnLeftButtonUp(self):
        super().OnLeftButtonUp()

    # -- right button: the host shows a context menu when embedded, so the
    #    inherited drag-to-zoom must not start underneath it --
    def OnRightButtonDown(self):
        if getattr(self._parent, 'embedded', False):
            return
        super().OnRightButtonDown()

    def OnRightButtonUp(self):
        if getattr(self._parent, 'embedded', False):
            return
        super().OnRightButtonUp()

    def OnMouseMove(self):
        super().OnMouseMove()


# ------------------------------------------------------------------ #
#  Qt window                                                          #
# ------------------------------------------------------------------ #

def install_qt_message_filter():
    """Silence the harmless QPainter warning of the VTK widget.

    Qt's backing store touches the paint device of the render-to-texture
    (native OpenGL) widget, which has no paint engine, and complains once per
    paint: "QPainter::begin: Paint device returned engine == 0".  Everything
    else is passed through.  Both viewers install this before their
    QApplication.
    """
    def _filter(mode, context, message):
        if "Paint device returned engine == 0" in message:
            return
        stream = sys.stderr if mode in (QtCore.QtMsgType.QtWarningMsg,
                                        QtCore.QtMsgType.QtCriticalMsg,
                                        QtCore.QtMsgType.QtFatalMsg) else sys.stdout
        print(message, file=stream)

    QtCore.qInstallMessageHandler(_filter)

class VolumeViewerWindow(QtWidgets.QMainWindow):
    """Window around :class:`CatImageViewer`.

    Used by ``CAT_VolView`` and, with *on_position_changed* wired up, as the
    volume window of CAT_SurfView, so both offer the same slices, status line
    and context menu.

    Args:
        image_path: Volume to display.
        surfaces: Surface files or ``vtkPolyData`` drawn as outlines on the
            slices; they must be in the millimetre space of the image.
        on_position_changed: Called as ``callback(world_xyz, window)`` when the
            user moves the cursor by clicking or scrolling.
        viewer_kwargs: Passed on to :class:`CatImageViewer`.
    """

    #: Zoom levels of the context menu: label and bounding-box edge length in
    #: mm, following the ortho viewer of SPM (None = whole volume).
    ZOOM_LEVELS = (
        ("Full volume", None),
        ("160 x 160 mm", 160.0),
        ("80 x 80 mm", 80.0),
        ("40 x 40 mm", 40.0),
        ("20 x 20 mm", 20.0),
        ("10 x 10 mm", 10.0),
    )

    SURFACE_COLORS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.6, 1.0))

    def __init__(self, image_path, parent=None, on_position_changed=None,
                 surfaces: Sequence = (), overlay: Optional[str] = None,
                 **viewer_kwargs):
        super().__init__(parent)
        self.image_path = str(image_path)
        self.setWindowTitle(f"Volume: {os.path.basename(self.image_path)}")
        self.on_position_changed = on_position_changed

        central = QtWidgets.QWidget(self)
        box = QtWidgets.QVBoxLayout(central)
        box.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)
        self.vtk_widget = QVTKRenderWindowInteractor(central)
        box.addWidget(self.vtk_widget, 1)
        try:
            self.vtk_widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow, True)
        except Exception:
            pass

        render_window = self.vtk_widget.GetRenderWindow()
        # A cross is much easier to place on a voxel than the arrow tip.  It is
        # set on the render window as well, because the widget re-derives the
        # Qt cursor from there whenever VTK reports a cursor change.
        try:
            render_window.SetCurrentCursor(VTK_CURSOR_CROSSHAIR)
            self.vtk_widget.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        except Exception:
            pass
        self.viewer = CatImageViewer(
            render_window=render_window,
            interactor=render_window.GetInteractor(),
            **viewer_kwargs,
        )
        self.viewer.load_image(self.image_path)
        for i, surface in enumerate(surfaces):
            self.viewer.add_surface(
                surface, self.SURFACE_COLORS[i % len(self.SURFACE_COLORS)])
        self.viewer.setup(window_title=os.path.basename(self.image_path))
        self.viewer.on_position_changed = self._position_changed

        self._label = QtWidgets.QLabel("")
        self.statusBar().addWidget(self._label)
        self._update_label()

        self._build_control_panel()
        if overlay:
            self.set_overlay(overlay)

        width, height = getattr(self.viewer, 'window_pixel_size', (700, 700))
        self.resize(int(width), int(height) + 24)

        try:
            policy = QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        except AttributeError:
            policy = QtCore.Qt.CustomContextMenu
        self.vtk_widget.setContextMenuPolicy(policy)
        self.vtk_widget.customContextMenuRequested.connect(self._show_context_menu)

        # The interactor may only be initialised once the widget has a window
        QtCore.QTimer.singleShot(0, self._post_show)

    # -------- control panel --------
    def _build_control_panel(self):
        """Dock the shared control panel, wired to the overlay volume."""
        self.ctrl = ControlPanel()
        self.ctrl.configure_for_volume()
        self.ctrl.set_labels_for_volume()
        self.dock_controls = QtWidgets.QDockWidget("Controls", self)
        self.dock_controls.setWidget(self.ctrl)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
                           self.dock_controls)
        # Floating, next to the window: docking it would squeeze the slices
        self.dock_controls.setFloating(True)

        def _place_dock(event, dock=self.dock_controls):
            QtWidgets.QDockWidget.showEvent(dock, event)
            if dock.isFloating():
                own = self.geometry()
                size = dock.sizeHint()
                dock.setGeometry(own.x() + own.width() + 8, own.y(),
                                 size.width(), size.height())
        self.dock_controls.showEvent = _place_dock
        self.dock_controls.hide()          # shown from the context menu

        viewer = self.viewer
        self.ctrl.overlay_btn.clicked.connect(self._choose_overlay)
        self.ctrl.overlay_combo.setEditable(False)
        self.ctrl.overlay_combo.currentIndexChanged.connect(
            lambda i: self.set_overlay(self.ctrl.overlay_combo.itemData(i)))

        def _range_changed():
            viewer.overlay_range = [float(self.ctrl.range_min.value()),
                                    float(self.ctrl.range_max.value())]
            viewer.refresh_overlay()

        def _clip_changed():
            c0 = float(self.ctrl.clip_min.value()); c1 = float(self.ctrl.clip_max.value())
            viewer.overlay_clip = (c0, c1) if c1 > c0 else (0.0, 0.0)
            self.ctrl.set_threshold_from_clip(viewer.overlay_clip)
            viewer.refresh_overlay()

        def _bkg_changed():
            lo = float(self.ctrl.bkg_min.value()); hi = float(self.ctrl.bkg_max.value())
            if hi > lo:
                viewer.set_window_level(hi - lo, 0.5 * (hi + lo))

        for widget, slot in ((self.ctrl.range_min, _range_changed),
                             (self.ctrl.range_max, _range_changed),
                             (self.ctrl.clip_min, _clip_changed),
                             (self.ctrl.clip_max, _clip_changed),
                             (self.ctrl.bkg_min, _bkg_changed),
                             (self.ctrl.bkg_max, _bkg_changed)):
            widget.editingFinished.connect(slot)
        for slider, slot in ((self.ctrl.range_slider_min, _range_changed),
                             (self.ctrl.range_slider_max, _range_changed),
                             (self.ctrl.clip_slider_min, _clip_changed),
                             (self.ctrl.clip_slider_max, _clip_changed),
                             (self.ctrl.bkg_slider_min, _bkg_changed),
                             (self.ctrl.bkg_slider_max, _bkg_changed)):
            slider.sliderReleased.connect(slot)

        def _threshold_changed(index: int):
            try:
                value = LOGP_THRESHOLDS[int(index)][1]
            except Exception:
                return
            for spin, v in ((self.ctrl.clip_min, -value), (self.ctrl.clip_max, value)):
                spin.blockSignals(True); spin.setValue(v); spin.blockSignals(False)
            viewer.overlay_clip = (-value, value) if value else (0.0, 0.0)
            viewer.refresh_overlay()
        self.ctrl.threshold.currentIndexChanged.connect(_threshold_changed)

        def _colormap_changed(index: int):
            viewer.overlay_colormap = COLORMAP_ORDER[int(index)]
            viewer.refresh_overlay()
        self.ctrl.colormap.currentIndexChanged.connect(_colormap_changed)

        def _opacity_changed(value: int):
            viewer.overlay_opacity = max(0.0, min(1.0, value / 100.0))
            viewer.refresh_overlay()
        self.ctrl.opacity.valueChanged.connect(_opacity_changed)

        def _inverse_changed(checked: bool):
            viewer.overlay_inverse = bool(checked)
            viewer.refresh_overlay()
        self.ctrl.cb_inverse.toggled.connect(_inverse_changed)

        def _discrete_changed(checked: bool):
            viewer.overlay_discrete = 16 if checked else 0
            viewer.refresh_overlay()
        self.ctrl.cb_discrete.toggled.connect(_discrete_changed)


        self._sync_control_panel()

    def _sync_control_panel(self):
        """Show the current overlay settings in the panel."""
        viewer = self.viewer
        has_overlay = viewer.overlay_path is not None
        self.ctrl.set_overlay_controls_enabled(has_overlay)
        lo, hi = viewer.overlay_range if has_overlay else (0.0, 0.0)
        image_lo, image_hi = viewer._image.GetScalarRange()
        for widget in (self.ctrl.range_min, self.ctrl.range_max, self.ctrl.clip_min,
                       self.ctrl.clip_max, self.ctrl.bkg_min, self.ctrl.bkg_max,
                       self.ctrl.colormap, self.ctrl.opacity, self.ctrl.threshold):
            widget.blockSignals(True)
        self.ctrl.set_overlay_bounds(lo, hi)
        self.ctrl.set_clip_bounds(lo, hi)
        self.ctrl.set_bkg_bounds(float(image_lo), float(image_hi))
        self.ctrl.range_min.setValue(lo); self.ctrl.range_max.setValue(hi)
        self.ctrl.clip_min.setValue(viewer.overlay_clip[0])
        self.ctrl.clip_max.setValue(viewer.overlay_clip[1])
        window, level = viewer.get_window_level()
        self.ctrl.bkg_min.setValue(level - 0.5 * window)
        self.ctrl.bkg_max.setValue(level + 0.5 * window)
        self.ctrl.colormap.setCurrentIndex(
            COLORMAP_ORDER.index(viewer.overlay_colormap)
            if viewer.overlay_colormap in COLORMAP_ORDER else 0)
        self.ctrl.opacity.setValue(int(round(viewer.overlay_opacity * 100)))
        for widget in (self.ctrl.range_min, self.ctrl.range_max, self.ctrl.clip_min,
                       self.ctrl.clip_max, self.ctrl.bkg_min, self.ctrl.bkg_max,
                       self.ctrl.colormap, self.ctrl.opacity, self.ctrl.threshold):
            widget.blockSignals(False)
        # p-value thresholds only for -log10(p) overlays, as in the surface viewer
        self.ctrl.set_threshold_visible(bool(viewer.overlay_path)
                                        and is_logp_name(viewer.overlay_path))
        self.ctrl.set_threshold_from_clip(viewer.overlay_clip)

    def set_overlay(self, path: Optional[str]):
        """Load (or clear) the overlay volume and refresh the panel."""
        try:
            self.viewer.set_overlay(path)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Overlay", f"Cannot overlay {os.path.basename(path or '')}:\n{exc}")
            return
        if path:
            combo = self.ctrl.overlay_combo
            if combo.findData(path) < 0:
                combo.blockSignals(True)
                combo.addItem(os.path.basename(path), path)
                combo.setCurrentIndex(combo.count() - 1)
                combo.blockSignals(False)
        self._sync_control_panel()
        self._update_label()

    def _choose_overlay(self):
        start = os.path.dirname(self.viewer.overlay_path or self.image_path)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose overlay volume", start,
            "NIfTI (*.nii *.nii.gz);;All files (*)")
        if path:
            self.set_overlay(path)

    def _post_show(self):
        try:
            self.vtk_widget.Initialize()
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self.viewer._update_info_text()   # the panel text follows the size
        except Exception:
            pass

    # -------- context menu --------
    def _show_context_menu(self, pos):
        """Right-click menu of the slice views."""
        menu = QtWidgets.QMenu(self)

        zoom_menu = menu.addMenu("Zoom")
        current = self.viewer.get_field_of_view()
        for label, mm in self.ZOOM_LEVELS:
            action = zoom_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(current == mm)
            action.triggered.connect(lambda _checked=False, v=mm: self.set_zoom(v))
        zoom_menu.addSeparator()
        follow_action = zoom_menu.addAction("Re-centre on cursor")
        follow_action.setCheckable(True)
        follow_action.setChecked(self.viewer.recenter)
        follow_action.triggered.connect(
            lambda checked=False: self.set_recenter(checked))

        # Naming the region under the cursor only makes sense when the image is
        # registered to the atlas, so the atlas is picked by hand
        atlas_menu = menu.addMenu("Atlas")
        selected = self.viewer.atlas_path
        none_action = atlas_menu.addAction("None")
        none_action.setCheckable(True)
        none_action.setChecked(selected is None)
        none_action.triggered.connect(lambda _checked=False: self.set_atlas(None))
        atlas_menu.addSeparator()
        for name, path in self.viewer.available_atlases():
            action = atlas_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(selected == path)
            action.triggered.connect(lambda _checked=False, p=path: self.set_atlas(p))
        atlas_menu.addSeparator()
        atlas_menu.addAction("Other…").triggered.connect(self._choose_atlas)

        menu.addSeparator()
        raw_action = menu.addAction("Raw voxels (nearest neighbour)")
        raw_action.setCheckable(True)
        raw_action.setChecked(not self.viewer.interpolate)
        raw_action.triggered.connect(
            lambda checked=False: self.set_interpolation(not checked))

        overlay_menu = menu.addMenu("Overlay")
        current_overlay = self.viewer.overlay_path
        none_action = overlay_menu.addAction("None")
        none_action.setCheckable(True)
        none_action.setChecked(current_overlay is None)
        none_action.triggered.connect(lambda _checked=False: self.set_overlay(None))
        if current_overlay:
            shown = overlay_menu.addAction(os.path.basename(current_overlay))
            shown.setCheckable(True)
            shown.setChecked(True)
        overlay_menu.addSeparator()
        overlay_menu.addAction("Open…").triggered.connect(self._choose_overlay)

        panel_action = menu.addAction("Controls")
        panel_action.setCheckable(True)
        panel_action.setChecked(self.dock_controls.isVisible())
        panel_action.triggered.connect(
            lambda checked=False: self.dock_controls.setVisible(checked))

        info_action = menu.addAction("Image information")
        info_action.setCheckable(True)
        info_action.setChecked(self.viewer.show_info)
        info_action.triggered.connect(
            lambda checked=False: self.viewer.set_info_visible(checked))
        # Further sections (window/level, overlays, …) go here

        menu.exec(self.vtk_widget.mapToGlobal(pos))

    def set_zoom(self, mm: Optional[float]):
        """Zoom the slices to an mm bounding box around the cursor."""
        try:
            self.viewer.set_field_of_view(mm)
        except Exception:
            pass

    def set_recenter(self, recenter: bool):
        """Whether a zoomed view follows the cursor."""
        try:
            self.viewer.set_recenter(recenter)
        except Exception:
            pass

    def set_interpolation(self, interpolate: bool):
        """Smooth the slices, or draw the raw voxels."""
        try:
            self.viewer.set_interpolation(interpolate)
            self._update_label()   # the reported value follows the display
        except Exception:
            pass

    def set_atlas(self, path: Optional[str]):
        """Use *path* to name the region under the cursor (None switches off)."""
        try:
            self.viewer.set_atlas(path)
        except Exception:
            pass

    def _choose_atlas(self):
        """Pick an atlas volume that is not one of the shipped ones."""
        start = os.path.dirname(self.viewer.atlas_path or self.image_path)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose atlas volume", start, "NIfTI (*.nii *.nii.gz);;All files (*)")
        if path:
            self.set_atlas(path)

    # -------- cursor --------
    def _update_label(self):
        ijk = self.viewer.get_index()
        world = self.viewer.get_world_position()
        if ijk is None or world is None:
            self._label.setText("")
            return
        value = self.viewer.get_value()
        text = (f"voxel [{ijk[0]}, {ijk[1]}, {ijk[2]}]    "
                f"mm ({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f})")
        if value is not None:
            text += f"    value {value:g}"
        self._label.setText(text)

    def _position_changed(self, world_xyz: Tuple[float, float, float]):
        """Report a position picked in a slice to whoever is linked."""
        self._update_label()
        if self.on_position_changed is not None:
            try:
                self.on_position_changed(world_xyz, self)
            except Exception:
                pass

    def set_world_position(self, world_xyz: Tuple[float, float, float]):
        """Move the slices to a world (mm) position picked elsewhere."""
        try:
            self.viewer.set_world_position(world_xyz)
            self._update_label()
        except Exception:
            pass


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def _parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(
        description=(
            "Single-window orthogonal VTK image viewer (SPM12-like "
            "layout) with optional surface overlays."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs", nargs="+",
        help=("Volumes (.nii(.gz), .mnc, .mha/.mhd, .nrrd, …) and up to three "
              "surfaces (.gii, .vtk, .vtp, .obj, .stl) drawn as outlines. "
              "Several volumes are stepped through with the ←/→ keys, "
              "keeping the cursor position."),
    )
    p.add_argument(
        "--size", type=int, default=400,
        help="Approximate half-width of the window in pixels",
    )
    p.add_argument(
        "--mirror", action="store_true",
        help="Mirror surfaces along x (scale -1,1,1)",
    )
    p.add_argument(
        "--no-mirror", action="store_true",
        help=argparse.SUPPRESS,  # now the default; kept for compatibility
    )
    p.add_argument(
        "--overlay", type=str, default=None,
        help=("Volume drawn in colour on top of the image; it must be on the "
              "same voxel grid (same dimensions and voxel size)"),
    )
    p.add_argument(
        "--atlas", type=str, default=None,
        help=("Atlas volume naming the region under the cursor; also "
              "selectable from the right-click menu"),
    )
    p.add_argument(
        "--no-info", action="store_true",
        help="Leave the information panel out of the free quadrant",
    )
    p.add_argument(
        "--nearest", action="store_true",
        help="Draw the raw voxels instead of smoothing the slices",
    )
    p.add_argument(
        "--no-recenter", action="store_true",
        help="Keep a zoomed view in place instead of following the cursor",
    )
    p.add_argument(
        "--headless", action="store_true",
        help="Do not start interactor (no window)",
    )
    p.add_argument(
        "--screenshot", type=str, default=None,
        help="Path (.png) to save a combined screenshot and exit",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print diagnostic information",
    )
    p.add_argument(
        "--surface-convention",
        choices=["auto", "ras", "lps", "none"],
        default="auto",
        help=(
            "Coordinate convention of input surfaces; converted to "
            "match the image world (RAS). Use 'none' to skip."
        ),
    )
    p.add_argument(
        "--percentile", nargs=2, type=float, default=[3, 97],
        metavar=("LOW", "HIGH"),
        help="Percentile range for display intensity scaling",
    )
    p.add_argument(
        "--no-percentile", action="store_true",
        help="Use the full intensity range instead of percentile scaling",
    )
    return p.parse_args(argv)


#: Files that hold a surface rather than a volume
SURFACE_SUFFIXES = ('.gii', '.vtk', '.vtp', '.obj', '.stl', '.ply')

#: Volumes opened at once, one window each; more would not fit side by side
MAX_VOLUMES = 3


def is_logp_name(filename: Optional[str]) -> bool:
    """True when a file name marks a -log10(p) map ('log', as in CAT12)."""
    if not filename:
        return False
    return 'log' in os.path.basename(str(filename)).lower()


def _split_inputs(inputs: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Sort the positional arguments into volumes and surfaces by extension."""
    volumes, surfaces = [], []
    for item in inputs:
        (surfaces if str(item).lower().endswith(SURFACE_SUFFIXES) else volumes).append(str(item))
    return volumes, surfaces


def link_windows(windows: Sequence["VolumeViewerWindow"]):
    """Keep the cursor of several viewer windows on the same world position.

    Each window reports where the user clicked or scrolled, and the others are
    moved to that millimetre position.  They are only told to move, never to
    report back, so this cannot loop.
    """
    windows = list(windows)

    def _follow(world_xyz, source=None):
        for window in windows:
            if window is not source:
                window.set_world_position(world_xyz)

    for window in windows:
        window.on_position_changed = _follow
    return windows


def _place_windows(windows: Sequence["VolumeViewerWindow"]):
    """Lay the windows out side by side, shrinking them to fit the screen.

    Comparing volumes means seeing them next to each other, so the windows are
    scaled down (keeping their proportions) until the row fits; only when that
    would make them unusably small are they cascaded instead.
    """
    if len(windows) < 2:
        return
    try:
        available = QtWidgets.QApplication.primaryScreen().availableGeometry()
    except Exception:
        return

    count = len(windows)
    gap = 12
    width = max(w.width() for w in windows)
    height = max(w.height() for w in windows)
    scale = min(1.0,
                (available.width() - (count - 1) * gap) / float(count * width),
                available.height() / float(height))
    if width * scale < 300:
        for i, window in enumerate(windows):   # too narrow to tile
            window.move(available.left() + 40 * i, available.top() + 40 * i)
        return

    width = int(width * scale)
    height = int(height * scale)
    total = count * width + (count - 1) * gap
    left = available.left() + max(0, (available.width() - total) // 2)
    for i, window in enumerate(windows):
        window.resize(width, height)
        window.move(left + i * (width + gap), available.top())


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry-point."""
    args = _parse_args(argv)

    pct = None if args.no_percentile else tuple(args.percentile)
    options = dict(
        window_size=args.size,
        # Surfaces and image share the millimetre space of the NIfTI header,
        # so mirroring is opt-in
        mirror_surfaces=bool(args.mirror) and not args.no_mirror,
        verbose=args.verbose,
        surface_convention=args.surface_convention,
        percentile_range=pct,
        show_info=not args.no_info,
        interpolate=not args.nearest,
        recenter=not args.no_recenter,
    )
    volumes, surfaces = _split_inputs(args.inputs)
    if not volumes:
        print("[cat_vol_view] No volume among the given files", file=sys.stderr)
        return 2
    if len(volumes) > MAX_VOLUMES:
        print(f"[cat_vol_view] Only the first {MAX_VOLUMES} volumes are shown; "
              f"ignoring {', '.join(os.path.basename(v) for v in volumes[MAX_VOLUMES:])}",
              file=sys.stderr)
        volumes = volumes[:MAX_VOLUMES]
    surfaces = surfaces[:3]

    if args.screenshot or args.headless:
        # Batch mode renders without a window, so no Qt application is needed
        for i, volume in enumerate(volumes):
            viewer = CatImageViewer(**options)
            viewer.load_image(volume)
            for s, surf in enumerate(surfaces):
                viewer.add_surface(surf, VolumeViewerWindow.SURFACE_COLORS[s])
            viewer.setup(window_title=os.path.basename(volume))
            if args.overlay:
                viewer.set_overlay(args.overlay)
            if args.atlas:
                viewer.set_atlas(args.atlas)
            screenshot = args.screenshot
            if screenshot and len(volumes) > 1:
                stem, ext = os.path.splitext(screenshot)
                screenshot = f"{stem}_{i + 1}{ext or '.png'}"
            viewer.render(screenshot=screenshot, headless=True)
        return 0

    install_qt_message_filter()
    # Only the program name goes to Qt: it parses argv itself and would
    # claim options of its own (-style, -stylesheet, -platform, …), which
    # clash with ours
    app = QtWidgets.QApplication(sys.argv[:1])
    # One window per volume, with their cursors linked
    windows = []
    for volume in volumes:
        window = VolumeViewerWindow(volume, surfaces=surfaces,
                                    overlay=args.overlay, **options)
        if args.atlas:
            window.set_atlas(args.atlas)
        windows.append(window)
    link_windows(windows)
    for window in windows:
        window.show()
    _place_windows(windows)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
