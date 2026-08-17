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

A second volume can be drawn in colour on top (``--overlay``), with range,
clip, colormap, opacity, inversion and the p-value thresholds set from the
control panel the surface viewer uses; the reported intensity is then the
overlay's.  It only has to be registered to the image: a different voxel grid
is resampled (nearest neighbour) through the millimetre space of the two
headers.

Other volumes can be outlined on the slices (``--contour``), which is how a
registration or a segmentation is judged; they are resampled the same way.

Slices are shown in neurological orientation (left is left), with the
anatomical direction of each pane edge marked (L/R/A/P/S/I).  The information
panel lists file name, dimensions, voxel size, orientation code, data type and
intensity range, plus voxel index, mm position and value under the cursor —
all in real intensities, i.e. with ``scl_slope`` and ``scl_inter`` applied —
and the region name when an atlas has been selected.  The status bar repeats
the position in editable boxes, so a coordinate from a table can be typed in,
next to buttons for the origin and the strongest voxel.

Display intensities are scaled to the 3rd--97th percentile range by default and
can be set by dragging the two handles over the intensity histogram in the
control panel.

The right-click menu holds the display settings — zoom, atlas, overlay,
contours, raw voxels, crosshair, direction letters, information panel — and
they apply to every open volume.  It also saves a screenshot and opens a
montage of slices for a report figure, whose slices are given either as a list
of millimetre positions or as start, step and stop, with the number of columns
and rows, as in ``cat_vol_slice_overlay``.  The keys are listed under
"Keyboard shortcuts"; dropping files on a window opens, overlays or outlines
them.

Usage (CLI):
    CAT_VolView <image> [more images…] [surf1] [surf2] [surf3] \
        --size 400 [--percentile 3 97]

    A montage can also be produced without opening a window, which is what
    makes it scriptable::

        CAT_VolView T1.nii.gz --montage --slices "25 30 40 80" \
            --overlay spmT_logP.nii.gz --threshold 0.05 --colormap FIRE \
            --colorbar --columns 4 --screenshot figure.png

    Up to six volumes may be given; each opens its own window (tiled three per
    row), titled with the directory the volume comes from.  The windows are
    linked: a click in one moves the others to the same millimetre position,
    and what the context menu changes applies to all of them.

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
import re
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Import minimal VTK modules explicitly (avoids large monolithic import)
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkPlane
from vtkmodules.vtkCommonCore import (
    VTK_FLOAT,
    vtkLookupTable,
    vtkPoints,
    vtkVariant,
)
from vtkmodules.vtkCommonMath import vtkMatrix3x3, vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersCore import vtkCutter, vtkMarchingSquares
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

from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
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
from vtkmodules.vtkImagingCore import (
    vtkImageMapToColors,
    vtkImageReslice,
    vtkImageShiftScale,
)

# Control panel and colormaps are shared with the surface viewer
try:
    from .make_apps import ensure_apps_exist
except ImportError:  # direct invocation as a script
    from make_apps import ensure_apps_exist

try:
    from .controls import ControlPanel, LOGP_THRESHOLDS
    from .viewer_common import (          # noqa: F401 re-exported
        APP_BUNDLE_ENV, VOLUME_SUFFIXES, ZOOM_EVENTS,
        ask_and_save_png, ask_for_files, claim_event, dropped_files,
        droppable_url, files_opened_by_finder, finder_open_files, note,
        qt_application, running_as_app, set_verbose, shorten_path,
        show_shortcuts,
    )
except ImportError:  # direct invocation as a script
    from controls import ControlPanel, LOGP_THRESHOLDS
    from viewer_common import (           # noqa: F401 re-exported
        APP_BUNDLE_ENV, VOLUME_SUFFIXES, ZOOM_EVENTS,
        ask_and_save_png, ask_for_files, claim_event, dropped_files,
        droppable_url, files_opened_by_finder, finder_open_files, note,
        qt_application, running_as_app, set_verbose, shorten_path,
        show_shortcuts,
    )

# Colormaps are shared with the surface viewer
try:
    from .colormaps import (
        JET, COLORMAP_NAMES, COLORMAP_ORDER, build_overlay_lut,
        format_p_value_label, logp_colorbar_ticks,
    )
except ImportError:  # direct invocation as a script
    from colormaps import (
        JET, COLORMAP_NAMES, COLORMAP_ORDER, build_overlay_lut,
        format_p_value_label, logp_colorbar_ticks,
    )

# Qt window + interactor.  QVTKRWIBase has to be chosen before the widget is
# imported; CAT_SurfView imports this module and relies on the same setting.
from PySide6 import QtCore, QtGui, QtWidgets
import vtkmodules.qt as _vtk_qt
_vtk_qt.QVTKRWIBase = "QOpenGLWidget"
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


#: Files that hold a surface rather than a volume
SURFACE_SUFFIXES = ('.gii', '.vtk', '.vtp', '.obj', '.stl', '.ply')


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


def _rescale_factors(reader) -> Tuple[float, float]:
    """The ``scl_slope`` and ``scl_inter`` of a file, or (1, 0) without them."""
    try:
        slope = float(reader.GetRescaleSlope())
        intercept = float(reader.GetRescaleIntercept())
    except Exception:
        return (1.0, 0.0)
    # A MINC reader may have applied them already
    if getattr(reader, "GetRescaleRealValues", None) and reader.GetRescaleRealValues():
        return (1.0, 0.0)
    if slope == 0.0:            # NIfTI: no scaling
        return (1.0, 0.0)
    return (slope, intercept)


def _rescaled_image(reader, image) -> Tuple[object, Tuple[float, float]]:
    """The image in real intensities, and the factors that were applied.

    NIfTI stores intensities as ``value * scl_slope + scl_inter``, and VTK
    hands out the numbers as they sit in the file, reporting the two factors
    separately.  A scaled volume — int16 with a slope, which is how many
    scanners and most statistical maps are written — would otherwise be
    displayed, windowed and reported in storage units.
    """
    slope, intercept = _rescale_factors(reader)
    if (slope, intercept) == (1.0, 0.0):
        return image, (1.0, 0.0)
    shift = vtkImageShiftScale()
    shift.SetInputData(image)
    # VTK computes (value + shift) * scale
    shift.SetShift(intercept / slope)
    shift.SetScale(slope)
    shift.SetOutputScalarTypeToFloat()
    shift.ClampOverflowOff()
    shift.Update()
    return shift.GetOutput(), (slope, intercept)


def _write_png(render_window, path: str, scale: int = 1) -> str:
    """Save what *render_window* shows as a PNG and return the file name."""
    if not path.lower().endswith(".png"):
        path += ".png"
    render_window.Render()
    w2i = vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    try:
        w2i.SetScale(max(1, int(scale)))
    except Exception:
        pass
    try:
        w2i.ReadFrontBufferOff()
        w2i.SetInputBufferTypeToRGB()
    except Exception:
        pass
    w2i.Update()
    writer = vtkPNGWriter()
    writer.SetFileName(path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    return path


def _format_value(value: Optional[float]) -> str:
    """An intensity for the panel; NaN and None are shown as nothing."""
    if value is None or value != value:
        return "-"
    return f"{value:g}"


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
            except Exception as exc:
                note(f"VTK could not read {os.path.basename(surface_path)} "
                     f"({exc}); trying nibabel")
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
        lock_zoom: bool = True,
        show_orientation: bool = True,
    ):
        """Create the viewer.

        Args:
            show_info: Fill the free quadrant with image information (name,
                dimensions, voxel size, orientation, data type, intensity
                range) and the values under the cursor.
            interpolate: Smooth the slices (linear); False draws the raw
                voxels (nearest neighbour).
            recenter: Let a zoomed view follow the cursor.
            lock_zoom: Ignore mouse and trackpad zooming, so the zoom only
                changes through :meth:`set_field_of_view`.
            show_orientation: Mark the edges of the panes with the anatomical
                direction they point at (L/R/A/P/S/I).
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
        #: scl_slope and scl_inter of the file, applied to the values
        self._rescale: Tuple[float, float] = (1.0, 0.0)
        self._file_scalar_type: Optional[str] = None
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
        # Zooming by dragging or pinching is off; the menu sets the zoom
        self.lock_zoom = bool(lock_zoom)
        # Information panel in the free quadrant
        self.show_info = bool(show_info)
        self._info_actor: Optional[vtkTextActor] = None
        # Anatomical direction letters along the pane edges
        self.show_orientation = bool(show_orientation)
        self._orientation_actors: List[List[vtkTextActor]] = [[], [], []]
        # Crosshair, and the pane the keyboard acts on when the mouse is away
        self.show_crosshair = True
        self.last_pane = self.VIEW_AXIAL
        self._image_name = ""
        self._orientation: Optional[str] = None
        # Atlas selected for naming the region under the cursor
        self.atlas_path: Optional[str] = None
        self._atlas: Optional[dict] = None
        # Overlay volume drawn on top, with its own colour settings
        self.overlay_path: Optional[str] = None
        self.overlay_name = ""
        self._overlay_image = None
        #: True when the overlay had to be put on the grid of the image
        self.overlay_resampled = False
        #: Outlines of other volumes drawn over the slices; see add_contour
        self.contours: List[dict] = []
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

        #: Surfaces outlined on the slices; see :meth:`add_surface` for the keys
        self.surfaces: List[dict] = []
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
        self.last_pane = view_idx
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

        # The wheel steps through slices; the style would otherwise zoom on top
        # of that.  Right-drag is its zoom, and the context menu opens on the
        # same button: when the menu takes the release, the style stays in its
        # zoom state and every later mouse move keeps zooming.
        locked = lambda: self.lock_zoom
        claim_event(self.interactor, "MouseWheelForwardEvent", self._event_cbs,
                    locked, lambda obj: _wheel_fwd_cb(obj, None))
        claim_event(self.interactor, "MouseWheelBackwardEvent", self._event_cbs,
                    locked, lambda obj: _wheel_back_cb(obj, None))
        for event in ZOOM_EVENTS:
            if not event.startswith("MouseWheel"):
                claim_event(self.interactor, event, self._event_cbs, locked)

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

    def set_lock_zoom(self, locked: bool):
        """Whether the mouse or trackpad may change the zoom.

        Switching the lock on also repairs the view: a zoom drag that lost its
        button release leaves the style zooming on every mouse move, so the
        state is ended and the zoom the menu asked for is restored.
        """
        self.lock_zoom = bool(locked)
        if not self.lock_zoom:
            return
        style = self.interactor.GetInteractorStyle() if self.interactor else None
        if style is not None:
            style.EndDolly()      # a no-op unless a zoom drag is still running
        self._apply_field_of_view(recenter=False)
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

    def set_crosshair_visible(self, visible: bool):
        """Show or hide the crosshair (a clean view for a screenshot)."""
        self.show_crosshair = bool(visible)
        for pane in getattr(self, '_line_act', []):
            for actor in pane:
                actor.SetVisibility(1 if self.show_crosshair else 0)
        self.render_window.Render()

    # -------- Orientation labels --------
    #: Which anatomical direction the world axes point at.  The world of the
    #: viewer is the millimetre space of the NIfTI sform, i.e. RAS.
    _AXIS_LETTERS = (("L", "R"), ("P", "A"), ("I", "S"))

    @classmethod
    def _direction_letter(cls, vector: Sequence[float]) -> str:
        """Anatomical letter for a world direction, e.g. (0,-1,0) -> 'P'."""
        axis = max(range(3), key=lambda a: abs(vector[a]))
        if abs(vector[axis]) < 1e-6:
            return ""
        return cls._AXIS_LETTERS[axis][1 if vector[axis] > 0 else 0]

    def _init_orientation_labels(self):
        """Put a direction letter on each edge of every pane.

        Which way round a slice is shown is the kind of mistake that is easy to
        make and hard to notice, and it survives into every screenshot — so the
        letters are derived from the cameras rather than written down.
        """
        # (x, y, horizontal justification, vertical justification)
        places = ((0.02, 0.5, "Left", "Centered"),
                  (0.98, 0.5, "Right", "Centered"),
                  (0.5, 0.98, "Centered", "Top"),
                  (0.5, 0.02, "Centered", "Bottom"))
        for pane in range(3):
            actors = []
            for x, y, horizontal, vertical in places:
                actor = vtkTextActor()
                prop = actor.GetTextProperty()
                prop.SetFontFamilyToArial()
                prop.SetBold(True)
                prop.SetColor(0.95, 0.95, 0.55)
                getattr(prop, f"SetJustificationTo{horizontal}")()
                getattr(prop, f"SetVerticalJustificationTo{vertical}")()
                coord = actor.GetPositionCoordinate()
                coord.SetCoordinateSystemToNormalizedViewport()
                coord.SetValue(x, y)
                self.renderers[pane].AddViewProp(actor)
                actors.append(actor)
            self._orientation_actors[pane] = actors
        self._update_orientation_labels()

    def _update_orientation_labels(self):
        """Re-read the letters from the cameras and size them to the panes."""
        if not any(self._orientation_actors):
            return
        # Without a header transform the world is not anatomical, so there is
        # nothing truthful to write
        visible = self.show_orientation and self._world_from_header
        height = self.render_window.GetSize()[1] or 2 * self.window_size
        for pane, actors in enumerate(self._orientation_actors):
            if not actors:
                continue
            camera = self.renderers[pane].GetActiveCamera()
            position = camera.GetPosition()
            focal = camera.GetFocalPoint()
            up = camera.GetViewUp()
            view = [focal[a] - position[a] for a in range(3)]
            # right on screen = view direction x up
            right = [view[1] * up[2] - view[2] * up[1],
                     view[2] * up[0] - view[0] * up[2],
                     view[0] * up[1] - view[1] * up[0]]
            flip = [-v for v in right]
            letters = (self._direction_letter(flip),
                       self._direction_letter(right),
                       self._direction_letter(up),
                       self._direction_letter([-v for v in up]))
            viewport = self._viewports[pane]
            size = int(max(9, min(20, height * (viewport[3] - viewport[1]) / 22)))
            for actor, letter in zip(actors, letters):
                actor.SetInput(letter)
                actor.GetTextProperty().SetFontSize(size)
                actor.SetVisibility(1 if visible and letter else 0)

    def set_orientation_labels(self, visible: bool):
        """Show or hide the L/R/A/P/S/I letters."""
        self.show_orientation = bool(visible)
        self._update_orientation_labels()
        self.render_window.Render()

    # -------- Keyboard-driven navigation --------
    def pane_at(self, x: int, y: int) -> int:
        """Pane at a window pixel position, or -1 outside the three panes."""
        pane = self._get_view_from_renderer(self.interactor.FindPokedRenderer(x, y))
        return pane if pane >= 0 else self._get_active_view(x, y)

    def step_slice(self, delta: int, pane: Optional[int] = None):
        """Move the cursor *delta* slices along the axis *pane* slices through.

        Without a pane the last one the user worked in is used, so the arrow
        keys carry on where the mouse left off.
        """
        self._on_scroll(self.last_pane if pane is None else pane, int(delta))

    # -------- Screenshot --------
    def save_screenshot(self, path: str, scale: int = 1) -> str:
        """Write what the window shows to a PNG file and return its path.

        Args:
            scale: Magnification of the saved image; 2 gives a figure that
                still looks sharp in a report.
        """
        path = _write_png(self.render_window, path, scale)
        if self.verbose:
            print(f"[cat_vol_view] Wrote screenshot: {path}")
        return path

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
        """Cut every surface with the three slice planes, once per pane.

        Each cut is drawn in the colour that identifies its surface, so several
        surfaces can be told apart — the outline of a surface that carries
        per-vertex values gets a second pass over it, coloured through the
        lookup table of those values.  Without a table the values must not
        colour the line at all: a mapper paints by scalars whenever the data
        carries them, which turned the outlines into VTK's default blue-to-red
        rainbow and lost the one colour that said which surface they were.
        """
        # Remove existing contour actors
        for vi in range(3):
            ren = self.renderers[vi]
            for entry in self._surface_contours[vi]:
                for actor in entry.get('actors', ()):
                    try:
                        ren.RemoveActor(actor)
                    except Exception:
                        pass
            self._surface_contours[vi] = []

        # World-space normal of the voxel axis each pane slices along
        dirs = self._voxel_axis_directions()
        normals = [dirs[self._pane_axis[vi]] for vi in range(3)]
        # Origins must match the actual slice location used by the image actors,
        # which slices at integer voxel indices. Do NOT add the +0.5 voxel-center shift here.
        origins = [self._world_from_index(tuple(self._ijk))] * 3

        for surface in self.surfaces:
            for vi in range(3):
                plane = vtkPlane()
                plane.SetNormal(*normals[vi])
                plane.SetOrigin(*origins[vi])
                cutter = vtkCutter()
                cutter.SetCutFunction(plane)
                cutter.SetInputData(surface['poly'])
                actors = [self._surface_contour_actor(
                    cutter, vi, color=surface['color'], distance=0.5)]
                if surface.get('lut') is not None:
                    # Over the plain line, so the clipped and missing values the
                    # table leaves transparent still show which surface it is
                    actors.append(self._surface_contour_actor(
                        cutter, vi, lut=surface['lut'],
                        scalar_range=surface.get('range'), distance=0.6))
                for actor in actors:
                    self.renderers[vi].AddActor(actor)
                self._surface_contours[vi].append(
                    {'plane': plane, 'cutter': cutter, 'actors': actors})

    def _surface_contour_actor(self, cutter, pane: int, color=None, lut=None,
                               scalar_range=None, distance: float = 0.5):
        """One line actor for the cut of a surface, coloured flat or by values."""
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(cutter.GetOutputPort())
        if lut is None:
            mapper.ScalarVisibilityOff()
        else:
            mapper.SetLookupTable(lut)
            mapper.ScalarVisibilityOn()
            mapper.SetColorModeToMapScalars()
            if scalar_range is not None and scalar_range[1] > scalar_range[0]:
                mapper.SetScalarRange(*scalar_range)
        actor = vtkActor()
        actor.SetMapper(mapper)
        if color is not None:
            actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(1.2)
        actor.GetProperty().LightingOff()
        # The contour lies exactly in the slice plane, where the image would win
        # the depth test; nudge it towards the camera
        actor.SetPosition(*self._camera_offset(pane, distance))
        return actor

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

        self._update_contour_slices()

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
        self.last_pane = view_idx
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
        except OSError as exc:
            note(f"no region names for the atlas: {exc}")
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
                image, _ = _rescaled_image(reader, image)   # real label numbers
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
        # The type the file uses, not the float the values were scaled into
        lines.append(f"data type   {self._file_scalar_type or self._image.GetScalarTypeAsString()}")
        slope, intercept = self._rescale
        if (slope, intercept) != (1.0, 0.0):
            sign = "+" if intercept >= 0 else "-"
            lines.append(f"scaling     x {slope:g} {sign} {abs(intercept):g}")
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
            f"value       {_format_value(value)}",
        ]
        if self._overlay_image is not None:
            # 'value' is the overlay above; keep the image underneath visible
            lines.append(f"background  {_format_value(self.get_background_value())}")
            name = self.overlay_name
            if self.overlay_resampled:
                name += " (resampled)"
            lines.insert(1, f"overlay     {name}")
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
        self._update_orientation_labels()
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

        An overlay on the same voxel grid is used as it is.  Anything else is
        resampled into the grid of the displayed image through the millimetre
        space of the two headers, so a map that is merely *registered* to the
        image — an atlas, a template, a statistical map — can be overlaid as
        well.  Pass None to remove it.

        Raises:
            ValueError: when the file cannot be read, or when the grids differ
                and at least one of the two has no anatomical transform to
                resample through.
        """
        self._remove_overlay_actors()
        self.overlay_path = None
        self.overlay_name = ""
        self._overlay_image = None
        self.overlay_resampled = False
        if not overlay_path:
            self._update_info_text()
            self.render_window.Render()
            return

        reader = _guess_image_reader(overlay_path)
        reader.Update()
        image = reader.GetOutput()
        if image is None or image.GetDimensions() == (0, 0, 0):
            raise ValueError(f"cannot read {os.path.basename(overlay_path)}")
        vox2world, from_header = _voxel_to_world_matrix(reader, image)
        # Range, clip and the reported value are then in real intensities
        image, _ = _rescaled_image(reader, image)

        # The image geometry lives in the transform, as for the displayed image
        image.SetOrigin(0.0, 0.0, 0.0)
        try:
            ident = vtkMatrix3x3()
            ident.Identity()
            image.SetDirectionMatrix(ident)
        except Exception:
            pass

        # Value range from the file, before any NaN padding is added
        lo, hi = image.GetScalarRange()

        if not self._same_grid(image, vox2world):
            image = self._resample_to_image(image, vox2world, from_header,
                                            os.path.basename(overlay_path))
            self.overlay_resampled = True

        self._overlay_image = image
        self.overlay_path = overlay_path
        self.overlay_name = os.path.basename(overlay_path)
        self.overlay_range = [float(lo), float(hi)]
        self._build_overlay_actors()
        self._update_info_text()
        self.render_window.Render()

    def _same_grid(self, image, vox2world: Sequence[Sequence[float]]) -> bool:
        """True when *image* has the voxels of the displayed image, mm for mm."""
        if image.GetDimensions() != self._image.GetDimensions():
            return False
        if any(abs(a - b) > 1e-4
               for a, b in zip(image.GetSpacing(), self._image.GetSpacing())):
            return False
        # Same voxel count and size is not the same grid: the two may still sit
        # in different places, which only the transform tells
        return all(abs(vox2world[r][c] - self._vox2world[r][c]) <= 1e-3
                   for r in range(3) for c in range(4))

    def _resample_to_image(self, image, vox2world: Sequence[Sequence[float]],
                           from_header: bool, name: str,
                           interpolate: bool = False,
                           background: float = float("nan")):
        """Put *image* on the voxel grid of the displayed image.

        Both transforms map voxels to anatomical millimetres, so going through
        that space lines the two up wherever they overlap.  Sampling is nearest
        neighbour by default — an overlay is usually a label map or a
        thresholded statistic, and interpolating those invents values — and
        voxels outside the volume become NaN, which the colour table leaves
        transparent.
        """
        if np is None:
            raise ValueError("resampling needs numpy, which is not installed")
        if not (from_header and self._world_from_header):
            which = "the overlay" if not from_header else "the image"
            raise ValueError(
                f"{name} is on a different voxel grid, and {which} has no "
                "sform/qform to resample through")

        own_spacing = self._image.GetSpacing()
        spacing = image.GetSpacing()
        scale_out = np.diag([*own_spacing, 1.0])          # data coords -> index
        scale_in = np.diag([*spacing, 1.0])
        # output data coords -> index -> mm -> overlay index -> overlay data
        transform = (scale_in
                     @ np.linalg.inv(np.array(vox2world, dtype=float))
                     @ np.array(self._vox2world, dtype=float)
                     @ np.linalg.inv(scale_out))

        axes = vtkMatrix4x4()
        for row in range(4):
            for col in range(4):
                axes.SetElement(row, col, float(transform[row, col]))

        reslice = vtkImageReslice()
        reslice.SetInputData(image)
        reslice.SetResliceAxes(axes)
        reslice.SetOutputExtent(*self._image.GetExtent())
        reslice.SetOutputSpacing(*own_spacing)
        reslice.SetOutputOrigin(*self._image.GetOrigin())
        if interpolate:
            reslice.SetInterpolationModeToLinear()
        else:
            reslice.SetInterpolationModeToNearestNeighbor()
        # Float output, so untouched voxels can be NaN rather than a value the
        # overlay would be coloured for
        reslice.SetOutputScalarType(VTK_FLOAT)
        reslice.SetBackgroundLevel(float(background))
        reslice.Update()
        resampled = reslice.GetOutput()
        if resampled is None or resampled.GetDimensions() == (0, 0, 0):
            raise ValueError(f"could not resample {name} onto the image")
        if self.verbose:
            print(f"[cat_vol_view] Resampled {name} onto the image grid")
        return resampled

    # ---------- Contours of other volumes ----------
    #: Colours handed out to contours, in this order
    CONTOUR_COLORS = ((1.0, 1.0, 0.25), (0.35, 1.0, 0.45),
                      (0.4, 0.75, 1.0), (1.0, 0.45, 0.8))

    def add_contour(self, path: str, level: Optional[float] = None,
                    color: Optional[Tuple[float, float, float]] = None) -> dict:
        """Draw the outline of another volume on the slices.

        This is how a registration or a segmentation is judged: the boundary of
        one volume over the grey values of another, as ``CheckReg`` does it.
        The volume is resampled onto the displayed grid when it has one of its
        own, so anything registered to the image can be outlined.

        Args:
            path: Volume to outline.
            level: Intensity the outline follows; halfway through the value
                range by default, which is the tissue boundary of a
                probability map.
            color: Line colour; taken from :attr:`CONTOUR_COLORS` otherwise.

        Returns:
            The contour, as it appears in :attr:`contours`.
        """
        reader = _guess_image_reader(path)
        reader.Update()
        image = reader.GetOutput()
        if image is None or image.GetDimensions() == (0, 0, 0):
            raise ValueError(f"cannot read {os.path.basename(path)}")
        vox2world, from_header = _voxel_to_world_matrix(reader, image)
        image, _ = _rescaled_image(reader, image)   # levels in real values
        image.SetOrigin(0.0, 0.0, 0.0)
        try:
            ident = vtkMatrix3x3()
            ident.Identity()
            image.SetDirectionMatrix(ident)
        except Exception:
            pass
        lo, hi = image.GetScalarRange()
        level = float(level) if level is not None else 0.5 * (lo + hi)
        if not self._same_grid(image, vox2world):
            # Smoothly, unlike the overlay: an outline traced through blocky
            # nearest-neighbour values would look like a staircase.  Outside
            # the volume the value is its minimum, so no outline is drawn
            # around the edge of the data.
            image = self._resample_to_image(
                image, vox2world, from_header, os.path.basename(path),
                interpolate=True, background=float(lo))
            landed = image.GetScalarRange()
            if not landed[0] <= level <= landed[1]:
                # Nothing of the volume reached the image: the two headers put
                # them in different places.  Say so — an outline that is simply
                # not drawn looks like a viewer that ignores the file.
                raise ValueError(
                    f"{os.path.basename(path)} does not land on the image: "
                    "mapped through the two headers it falls outside it, so "
                    f"there is nothing at level {level:g} to outline. Are the "
                    "two registered, and does the header of each say where it "
                    "is?")
        entry = {
            'path': str(path),
            'name': os.path.basename(path),
            'image': image,
            'range': (float(lo), float(hi)),
            'level': level,
            'color': tuple(color) if color else
                     self.CONTOUR_COLORS[len(self.contours) % len(self.CONTOUR_COLORS)],
            'actors': [None, None, None],
            'filters': [None, None, None],
        }
        self.contours.append(entry)
        self._build_contour_actors(entry)
        self._set_slices_from_index()
        self.render_window.Render()
        return entry

    def _build_contour_actors(self, entry: dict):
        """One outline actor per pane for *entry*."""
        for pane in range(3):
            squares = vtkMarchingSquares()
            squares.SetInputData(entry['image'])
            squares.SetValue(0, entry['level'])
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(squares.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*entry['color'])
            actor.GetProperty().SetLineWidth(1.5)
            # In front of the image, like the overlay and the crosshair
            offset = self._camera_offset(pane, 0.35)
            if self._world_from_header:
                matrix = self._world_matrix()
                for row in range(3):
                    matrix.SetElement(row, 3, matrix.GetElement(row, 3) + offset[row])
                actor.SetUserMatrix(matrix)
            else:
                actor.SetPosition(*offset)
            self.renderers[pane].AddActor(actor)
            entry['filters'][pane] = squares
            entry['actors'][pane] = actor

    def _update_contour_slices(self):
        """Point every outline at the slice its pane shows."""
        if not self.contours or self._ijk is None:
            return
        extent = list(self._image.GetExtent())
        for entry in self.contours:
            for pane in range(3):
                squares = entry['filters'][pane]
                if squares is None:
                    continue
                axis = self._pane_axis[pane]
                slice_extent = list(extent)
                slice_extent[2 * axis] = slice_extent[2 * axis + 1] = self._ijk[axis]
                squares.SetImageRange(*slice_extent)
                squares.Modified()

    def contour_lines_shown(self, entry: dict) -> int:
        """How many segments the outline of *entry* draws on the slices shown.

        Zero is not necessarily wrong — the level may be crossed only on other
        slices — but it is what an outline that fails to appear looks like, so
        the window says as much instead of leaving the user guessing.
        """
        total = 0
        for squares in entry['filters']:
            if squares is None:
                continue
            squares.Update()
            total += squares.GetOutput().GetNumberOfLines()
        return total

    def set_contour_level(self, path: str, level: float):
        """Change the intensity the outline of *path* follows."""
        for entry in self.contours:
            if entry['path'] == path:
                entry['level'] = float(level)
                for squares in entry['filters']:
                    if squares is not None:
                        squares.SetValue(0, entry['level'])
                        squares.Modified()
        self.render_window.Render()

    def remove_contour(self, path: str):
        """Remove the outline of *path*."""
        for entry in [e for e in self.contours if e['path'] == path]:
            for pane, actor in enumerate(entry['actors']):
                if actor is not None:
                    try:
                        self.renderers[pane].RemoveActor(actor)
                    except Exception:
                        pass
            self.contours.remove(entry)
        self.render_window.Render()

    def clear_contours(self):
        """Remove every outline."""
        for entry in list(self.contours):
            self.remove_contour(entry['path'])

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

    def _sample(self, image, nearest: bool = False):
        """Intensity of *image* at the cursor, sampled the way it is drawn.

        With smoothing on this is the trilinear value at the exact cursor
        position, so the number matches what is displayed; with raw voxels
        selected — or with *nearest*, which the overlay uses because it is
        drawn that way — it is the untouched value of the voxel the cursor is
        in.
        """
        index = self.get_index_exact()
        if image is None or index is None:
            return None
        ext = image.GetExtent()
        if nearest or not self.interpolate:
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
            return self.get_overlay_value()
        return self._sample(self._image)

    def get_background_value(self):
        """Intensity of the displayed image, even when an overlay covers it."""
        return self._sample(self._image)

    def get_overlay_value(self):
        """Overlay intensity at the cursor, or None without an overlay.

        Always nearest neighbour, since that is how the overlay is drawn — and
        a resampled overlay carries NaN outside itself, which no interpolation
        should smear into the values next to it.
        """
        return self._sample(self._overlay_image, nearest=True)

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

    def go_to_origin(self, notify: bool = True):
        """Put the cursor on the origin of the millimetre space (0, 0, 0).

        In a registered image that is the anterior commissure, which is where
        a coordinate is reported from.
        """
        self.set_world_position((0.0, 0.0, 0.0), notify=notify)

    def go_to_maximum(self, notify: bool = True) -> Optional[Tuple[int, int, int]]:
        """Jump to the strongest voxel — of the overlay when there is one.

        The first thing to look at in a statistical map is its peak, and
        finding it by scrolling is hopeless.

        Returns:
            The voxel jumped to, or None when there is nothing to search.
        """
        image = self._overlay_image if self._overlay_image is not None else self._image
        if image is None or np is None or vtk_to_numpy is None:
            return None
        scalars = image.GetPointData().GetScalars()
        if scalars is None:
            return None
        try:
            values = vtk_to_numpy(scalars)
            if values.ndim > 1:                 # colour or vector data
                values = values[:, 0]
            # A resampled overlay is NaN outside itself
            flat = int(np.nanargmax(values))
        except (ValueError, TypeError):
            return None
        extent = image.GetExtent()
        width = extent[1] - extent[0] + 1
        height = extent[3] - extent[2] + 1
        k, rest = divmod(flat, width * height)
        j, i = divmod(rest, width)
        index = (i + extent[0], j + extent[2], k + extent[4])
        self.set_index(*index, notify=notify)
        return index

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
        """Read a volume and work out the millimetre space it lives in.

        The sform/qform of the header is what puts slices, crosshair, surfaces
        and any linked window into the same space, and ``scl_slope``/
        ``scl_inter`` are applied so the values are the real intensities.

        Raises:
            RuntimeError: when the file cannot be read or is empty.
        """
        if self.verbose:
            print(f"[cat_vol_view] Loading image: {image_path}")
        reader = _guess_image_reader(image_path)
        reader.Update()
        self._image = reader.GetOutput()
        if self._image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        # What the file says the values are, before they become real ones
        self._file_scalar_type = self._image.GetScalarTypeAsString()
        self._image, self._rescale = _rescaled_image(reader, self._image)
        if self.verbose and self._rescale != (1.0, 0.0):
            print(f"[cat_vol_view] Applied scl_slope={self._rescale[0]:g}, "
                  f"scl_inter={self._rescale[1]:g}")
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

    def add_surface(self, surface: "str | vtkPolyData",
                    color: Tuple[float, float, float],
                    lut=None, scalar_range=None):
        """Draw a surface as an outline on the slices, in *color*.

        The surface has to be in the millimetre space of the image; only when
        the image carries no transform of its own is a surface far from it
        nudged onto the image centre.

        Args:
            surface: Surface file or ``vtkPolyData``.
            color: Line colour identifying this surface among the others.
            lut: Lookup table for the per-vertex values the surface carries, if
                any.  The outline is then coloured by them, the way the surface
                viewer colours the surface itself, over a line in *color* that
                shows where the values are clipped or missing.
            scalar_range: Value range *lut* covers; the surface viewer keeps it
                on the mapper rather than in the table.
        """
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
        except Exception as exc:
            note(f"could not check whether the surface needs recentring: {exc}")
        self.surfaces.append({
            'poly': poly,
            'color': tuple(color),
            'lut': lut,
            'range': tuple(scalar_range) if scalar_range is not None else None,
        })
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

        # Crosshair overlays, direction letters & surface contours
        self._init_crosshair()
        self._init_orientation_labels()
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
    """Interactor style for the combined orthogonal viewer.

    Left-click and the mouse wheel are handled by explicit interactor
    observers in :class:`CatImageViewer`; the middle button pans.

    Note that overriding the style's ``On...`` methods here would have no
    effect: the interactor dispatches events to the C++ implementation, which
    knows nothing about a Python subclass.  Anything that has to be taken away
    from the style — the zoom, see :meth:`CatImageViewer._guard` — is stopped
    by a higher-priority observer on the interactor instead.
    """

    def __init__(self, parent: CatImageViewer):
        super().__init__()
        self._parent = parent


# ------------------------------------------------------------------ #
#  Qt window                                                          #
# ------------------------------------------------------------------ #

# APP_BUNDLE_ENV, running_as_app, files_opened_by_finder, ask_for_files,
# finder_open_files and qt_application come from viewer_common; both viewers open
# documents the way macOS hands them over, so that lives next to the other
# shared behaviour.  They are re-exported here because the surface viewer
# imports them from this module.


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

class Montage:
    """A row-and-column sheet of slices through the volume.

    One slice at a time answers a question about a place; a montage answers one
    about the whole volume — which is what goes into a QC report, and what
    ``cat_vol_slice_overlay`` produces in CAT12.  The image is taken from a
    :class:`CatImageViewer` with its display range, overlay and colours, so the
    sheet looks like the slices it was made from.

    The class draws into a render window and knows nothing about Qt, so the
    same code serves :class:`MontageWindow` and the batch call that writes a
    PNG without opening a window (``--montage --screenshot``).
    """

    #: Slice orientations offered, as (label, pane of the ortho viewer)
    AXES = (("Axial", CatImageViewer.VIEW_AXIAL),
            ("Coronal", CatImageViewer.VIEW_CORONAL),
            ("Sagittal", CatImageViewer.VIEW_SAGITTAL))

    #: Never build more tiles than this, whatever step is asked for
    MAX_SLICES = 100

    #: Height of the colorbar strip, as a fraction of the window
    COLORBAR_HEIGHT = 0.13

    def __init__(self, viewer: CatImageViewer, render_window: vtkRenderWindow,
                 pane: int = CatImageViewer.VIEW_AXIAL,
                 slices_mm: Optional[Tuple[float, float, float]] = None,
                 slices: Optional[Sequence[float]] = None,
                 columns: int = 0, rows: int = 0, labels: bool = True,
                 colorbar: bool = False, on_message=None):
        """Prepare a montage; :meth:`build` draws it.

        Args:
            slices_mm: ``(start, step, stop)`` in millimetres along the axis
                being cut, as in ``cat_vol_slice_overlay``; derived from the
                volume when omitted.
            slices: Explicit millimetre positions, which take precedence over
                *slices_mm* — the slices of a figure are usually a hand-picked
                list rather than a regular series.
            columns, rows: Layout of the sheet; 0 means "work it out", and
                giving both fixes the sheet to that many tiles.
            labels: Write the position of each slice into its tile.
            colorbar: Add a colour bar for the overlay (or for the image when
                there is none), labelled with p-values for -log10(p) maps.
            on_message: Called with a line about what the sheet shows.
        """
        self.source = viewer
        self.render_window = render_window
        self.pane = int(pane)
        self.columns = int(columns)
        self.rows = int(rows)
        self.labels = bool(labels)
        self.colorbar = bool(colorbar)
        self.on_message = on_message
        self.slices = [float(v) for v in slices] if slices else None
        self.start_mm, self.step_mm, self.stop_mm = (
            tuple(float(v) for v in slices_mm) if slices_mm
            else self._default_range())

        self._renderers: List[vtkRenderer] = []
        self._keep_alive: List = []
        self._colorbar_renderer: Optional[vtkRenderer] = None
        self.colorbar_actor: Optional[vtkScalarBarActor] = None
        # Each renderer only clears its own viewport, so a sheet with fewer
        # tiles than before would keep the old ones in the empty corner.  This
        # one covers the whole window and is never removed.
        self._background = vtkRenderer()
        self._background.SetViewport(0.0, 0.0, 1.0, 1.0)
        self._background.SetBackground(0, 0, 0)
        self._background.SetInteractive(0)
        self.render_window.AddRenderer(self._background)

    # -------- where the slices are --------
    def _world_axis(self) -> int:
        """World axis the slices step along (x, y or z)."""
        axis = self.source._pane_axis[self.pane]
        direction = self.source._voxel_axis_directions()[axis]
        return max(range(3), key=lambda a: abs(direction[a]))

    def axis_letter(self) -> str:
        """Name of that axis, for labels and for the toolbar."""
        return "xyz"[self._world_axis()]

    def _mm_for_index(self, index: int) -> float:
        """Millimetre position of a slice, along the axis it cuts."""
        axis = self.source._pane_axis[self.pane]
        position = list(self.source._ijk or [0, 0, 0])
        position[axis] = index
        return self.source._world_from_index(tuple(position))[self._world_axis()]

    def _index_for_mm(self, value: float) -> int:
        """Slice that sits at *value* millimetres (rounded to the voxel grid)."""
        axis = self.source._pane_axis[self.pane]
        world = list(self.source.get_world_position() or (0.0, 0.0, 0.0))
        world[self._world_axis()] = float(value)
        return int(round(self.source._index_from_world(tuple(world))[axis]))

    def _extent_mm(self) -> Tuple[float, float]:
        """First and last millimetre position the volume covers on this axis."""
        axis = self.source._pane_axis[self.pane]
        extent = self.source._image.GetExtent()
        ends = (self._mm_for_index(extent[2 * axis]),
                self._mm_for_index(extent[2 * axis + 1]))
        return (min(ends), max(ends))

    def _default_range(self) -> Tuple[float, float, float]:
        """A dozen slices across the volume, in whole millimetres.

        Only a starting point: start, step and stop are what the toolbar edits,
        the way the ``slices`` vector is written in ``cat_vol_slice_overlay``.
        """
        low, high = self._extent_mm()
        margin = 0.12 * (high - low)
        start = round(low + margin)
        stop = round(high - margin)
        step = max(1.0, round((stop - start) / 11.0))
        return (float(start), float(step), float(stop))

    def set_slices(self, slices: Optional[Sequence[float]]):
        """Use an explicit list of millimetre positions (None: back to the series)."""
        self.slices = [float(v) for v in slices] if slices else None

    def slice_positions(self) -> List[float]:
        """The millimetre positions to show.

        An explicit list is taken as it is; otherwise the series start,
        start+step, … up to stop.  Positions outside the volume are dropped
        either way — clamping them would repeat the end slice.
        """
        low, high = self._extent_mm()

        def inside(value: float) -> bool:
            return low - 1e-6 <= value <= high + 1e-6

        if self.slices is not None:
            return [round(v, 3) for v in self.slices if inside(v)][:self.MAX_SLICES]

        step = abs(self.step_mm)
        if step <= 0:
            return []
        start, stop = self.start_mm, self.stop_mm
        if stop < start:
            start, stop = stop, start
        positions = []
        # A hair over stop, so a stop that lands exactly on a step is included
        while start <= stop + 1e-6 and len(positions) < self.MAX_SLICES:
            if inside(start):
                positions.append(round(start, 3))
            start += step
        return positions

    def _slice_indices(self) -> List[int]:
        """Voxel slices for the millimetre positions, in the volume."""
        axis = self.source._pane_axis[self.pane]
        extent = self.source._image.GetExtent()
        indices = []
        for position in self.slice_positions():
            index = self._index_for_mm(position)
            if extent[2 * axis] <= index <= extent[2 * axis + 1]:
                indices.append(index)
        return indices

    def _grid(self, count: Optional[int] = None) -> Tuple[int, int]:
        """Columns and rows of the sheet.

        Fixed where asked for, worked out for the rest: with neither given the
        sheet is kept roughly square, and one of the two determines the other.
        """
        count = len(self._slice_indices()) if count is None else int(count)
        count = max(1, count)
        if self.columns > 0 and self.rows > 0:
            return self.columns, self.rows
        if self.columns > 0:
            return self.columns, int(math.ceil(count / self.columns))
        if self.rows > 0:
            return int(math.ceil(count / self.rows)), self.rows
        columns = max(1, int(math.ceil(math.sqrt(count))))
        return columns, int(math.ceil(count / columns))

    def _slice_label(self, index: int) -> str:
        """Where the slice is, in millimetres along the axis it cuts."""
        return f"{self.axis_letter()} = {self._mm_for_index(index):.0f}"

    # -------- building --------
    def build(self):
        """Lay the slices out, one renderer each."""
        for renderer in self._renderers:
            self.render_window.RemoveRenderer(renderer)
        self._renderers = []
        self._keep_alive = []
        if self._colorbar_renderer is not None:
            self.render_window.RemoveRenderer(self._colorbar_renderer)
            self._colorbar_renderer = None
            self.colorbar_actor = None

        source = self.source
        indices = self._slice_indices()
        columns, rows = self._grid(len(indices))
        dropped = max(0, len(indices) - columns * rows)
        if dropped:
            # Both columns and rows were given, and they do not hold every
            # slice; showing the first ones beats silently changing the layout
            indices = indices[:columns * rows]
        self._report(len(indices), dropped)
        axis = source._pane_axis[self.pane]
        extent = list(source._image.GetExtent())
        template = source.renderers[self.pane].GetActiveCamera()
        floor = self.COLORBAR_HEIGHT if self.colorbar else 0.0

        for position, index in enumerate(indices):
            renderer = vtkRenderer()
            renderer.SetBackground(0, 0, 0)
            column = position % columns
            row = position // columns
            height = (1.0 - floor) / rows
            renderer.SetViewport(column / columns, 1.0 - (row + 1) * height,
                                 (column + 1) / columns, 1.0 - row * height)

            display = list(extent)
            display[2 * axis] = display[2 * axis + 1] = index
            for actor in self._slice_actors(display):
                renderer.AddActor(actor)
                self._keep_alive.append(actor)

            if self.labels:
                label = vtkTextActor()
                label.SetInput(self._slice_label(index))
                prop = label.GetTextProperty()
                prop.SetFontFamilyToArial()
                prop.SetFontSize(12)
                prop.SetColor(0.9, 0.9, 0.6)
                coordinate = label.GetPositionCoordinate()
                coordinate.SetCoordinateSystemToNormalizedViewport()
                coordinate.SetValue(0.04, 0.04)
                renderer.AddViewProp(label)
                self._keep_alive.append(label)

            camera = renderer.GetActiveCamera()
            camera.SetParallelProjection(1)
            # Same view direction as the pane this montage cuts along
            focal = template.GetFocalPoint()
            position_3d = template.GetPosition()
            camera.SetFocalPoint(*focal)
            camera.SetPosition(*position_3d)
            camera.SetViewUp(*template.GetViewUp())
            renderer.ResetCamera()
            self.render_window.AddRenderer(renderer)
            self._renderers.append(renderer)

        # Every tile at the same scale, so the slices are comparable
        scales = [r.GetActiveCamera().GetParallelScale() for r in self._renderers]
        if scales:
            for renderer in self._renderers:
                renderer.GetActiveCamera().SetParallelScale(max(scales))
        if self.colorbar:
            self._build_colorbar()
        self.render_window.Render()

    def _report(self, shown: int, dropped: int = 0):
        """Say what the sheet ended up showing, and why it is not more."""
        if not shown:
            low, high = self._extent_mm()
            message = (f"No slice in that range — the volume covers "
                       f"{low:.0f} … {high:.0f} mm")
        elif self.slices is not None:
            message = f"{shown} slices"
            if len(self.slices) > shown:
                message += f" ({len(self.slices) - shown} outside the volume)"
        else:
            message = f"{shown} slices, {self.step_mm:g} mm apart"
            if dropped:
                message += f" ({dropped} more do not fit the sheet)"
            elif len(self.slice_positions()) >= self.MAX_SLICES:
                message += f" (at most {self.MAX_SLICES})"
        if self.on_message is not None:
            self.on_message(message)
        return message

    def _colorbar_lut(self):
        """Lookup table and value range the bar should show."""
        source = self.source
        if source._overlay_image is not None:
            return source._overlay_lut(), tuple(source.overlay_range)
        window, level = source.get_window_level()
        lut = vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetTableRange(level - 0.5 * window, level + 0.5 * window)
        for i in range(256):
            grey = i / 255.0
            lut.SetTableValue(i, grey, grey, grey, 1.0)
        lut.Build()
        return lut, (level - 0.5 * window, level + 0.5 * window)

    def _build_colorbar(self):
        """A colour bar under the sheet, in p-values for a -log10(p) overlay."""
        lut, value_range = self._colorbar_lut()
        bar = vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetOrientationToHorizontal()
        bar.SetPosition(0.3, 0.25)
        bar.SetWidth(0.4)
        bar.SetHeight(0.55)
        bar.SetTitle(" ")
        try:
            bar.SetAnnotationTextScaling(False)
            bar.UnconstrainedFontSizeOn()
        except Exception:
            pass
        for prop in (bar.GetLabelTextProperty(), bar.GetAnnotationTextProperty()):
            prop.SetFontFamilyToArial()
            prop.SetFontSize(14)
            prop.SetColor(0.92, 0.92, 0.92)
            prop.SetItalic(False)
            prop.SetBold(False)

        # A statistic map is read as p-values, exactly as in the surface viewer
        ticks = []
        if is_logp_name(self.source.overlay_path):
            ticks = logp_colorbar_ticks(value_range[0], value_range[1],
                                        self.source.overlay_clip)
        if ticks:
            compact = len(ticks) > 5
            lut.ResetAnnotations()
            for value in ticks:
                lut.SetAnnotation(vtkVariant(float(value)),
                                  format_p_value_label(value, compact))
            bar.SetDrawAnnotations(True)
            bar.SetDrawTickLabels(False)
            bar.SetFixedAnnotationLeaderLineColor(True)
        else:
            bar.SetDrawAnnotations(False)
            bar.SetDrawTickLabels(True)
            bar.SetNumberOfLabels(5)
            # VTK's default format prints '170.' for 170
            bar.SetLabelFormat("%.4g")

        renderer = vtkRenderer()
        renderer.SetViewport(0.0, 0.0, 1.0, self.COLORBAR_HEIGHT)
        renderer.SetBackground(0, 0, 0)
        renderer.SetInteractive(0)
        renderer.AddViewProp(bar)
        self.render_window.AddRenderer(renderer)
        self._colorbar_renderer = renderer
        #: The bar itself, so its labelling can be inspected
        self.colorbar_actor = bar
        self._keep_alive.append(bar)

    def _slice_actors(self, display_extent: Sequence[int]) -> List[vtkImageActor]:
        """Image (and overlay) actor for one slice of the montage."""
        source = self.source
        actors = []
        window, level = source.get_window_level()

        colours = vtkImageMapToWindowLevelColors()
        colours.SetInputData(source._image)
        colours.SetWindow(window)
        colours.SetLevel(level)
        image_actor = vtkImageActor()
        image_actor.GetMapper().SetInputConnection(colours.GetOutputPort())
        image_actor.SetInterpolate(1 if source.interpolate else 0)
        self._keep_alive.append(colours)
        actors.append(image_actor)

        if source._overlay_image is not None:
            overlay_colours = vtkImageMapToColors()
            overlay_colours.SetInputData(source._overlay_image)
            overlay_colours.SetLookupTable(source._overlay_lut())
            overlay_colours.SetOutputFormatToRGBA()
            overlay_actor = vtkImageActor()
            overlay_actor.GetMapper().SetInputConnection(
                overlay_colours.GetOutputPort())
            overlay_actor.SetInterpolate(0)
            self._keep_alive.append(overlay_colours)
            actors.append(overlay_actor)

        for offset, actor in enumerate(actors):
            actor.SetDisplayExtent(*display_extent)
            if source._world_from_header:
                matrix = source._world_matrix()
                if offset:      # lift the overlay off the image plane
                    shift = source._camera_offset(self.pane, 0.25)
                    for row in range(3):
                        matrix.SetElement(row, 3,
                                          matrix.GetElement(row, 3) + shift[row])
                actor.SetUserMatrix(matrix)
        return actors


def render_montage(viewer: CatImageViewer, path: str,
                   size: Tuple[int, int] = (1200, 900), scale: int = 1,
                   **montage_kwargs) -> str:
    """Write a montage of *viewer* to a PNG without opening a window."""
    window = vtkRenderWindow()
    try:
        window.SetOffScreenRendering(1)
        window.SetMultiSamples(0)
    except Exception:
        pass
    window.SetSize(int(size[0]), int(size[1]))
    montage = Montage(viewer, window, **montage_kwargs)
    montage.build()
    return _write_png(window, path, scale=scale)


class MontageWindow(QtWidgets.QMainWindow):
    """Window around :class:`Montage`, with the sheet settings in a toolbar."""

    AXES = Montage.AXES
    MAX_SLICES = Montage.MAX_SLICES

    def __init__(self, viewer: CatImageViewer, title: str = "Montage",
                 pane: int = CatImageViewer.VIEW_AXIAL,
                 slices_mm: Optional[Tuple[float, float, float]] = None,
                 slices: Optional[Sequence[float]] = None,
                 columns: int = 0, rows: int = 0, labels: bool = True,
                 colorbar: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title} — montage")

        central = QtWidgets.QWidget(self)
        box = QtWidgets.QVBoxLayout(central)
        box.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)
        self.vtk_widget = QVTKRenderWindowInteractor(central)
        box.addWidget(self.vtk_widget, 1)
        self.render_window = self.vtk_widget.GetRenderWindow()

        self.montage = Montage(
            viewer, self.render_window, pane=pane, slices_mm=slices_mm,
            slices=slices, columns=columns, rows=rows, labels=labels,
            colorbar=colorbar,
            on_message=lambda text: self.statusBar().showMessage(text))

        bar = self.addToolBar("Montage")
        bar.setMovable(False)
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems([label for label, _ in Montage.AXES])
        self.axis_combo.setCurrentIndex(
            [p for _, p in Montage.AXES].index(self.montage.pane))
        self.axis_combo.currentIndexChanged.connect(self._axis_changed)
        bar.addWidget(QtWidgets.QLabel(" Slices "))
        bar.addWidget(self.axis_combo)

        # start : step : stop in mm, as the slices are written down in
        # cat_vol_slice_overlay
        self.range_label = QtWidgets.QLabel(
            f"  {self.montage.axis_letter()} mm (start step stop) ")
        bar.addWidget(self.range_label)
        self.range_spins: List[QtWidgets.QDoubleSpinBox] = []
        for value, low in ((self.montage.start_mm, -9999.0),
                           (self.montage.step_mm, 0.1),
                           (self.montage.stop_mm, -9999.0)):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(low, 9999.0)
            spin.setDecimals(1)
            spin.setValue(value)
            spin.setKeyboardTracking(False)
            spin.setFixedWidth(76)
            spin.valueChanged.connect(self._range_changed)
            self.range_spins.append(spin)
            bar.addWidget(spin)
        reset = bar.addAction("Fit")
        reset.setToolTip("Start, step and stop across the whole volume")
        reset.triggered.connect(self.fit_range)

        bar.addSeparator()
        bar.addWidget(QtWidgets.QLabel(" Columns "))
        self.columns_spin = QtWidgets.QSpinBox()
        self.columns_spin.setRange(0, 20)
        self.columns_spin.setSpecialValueText("auto")
        self.columns_spin.setValue(self.montage.columns)
        self.columns_spin.valueChanged.connect(self._layout_changed)
        bar.addWidget(self.columns_spin)
        bar.addWidget(QtWidgets.QLabel(" Rows "))
        self.rows_spin = QtWidgets.QSpinBox()
        self.rows_spin.setRange(0, 20)
        self.rows_spin.setSpecialValueText("auto")
        self.rows_spin.setValue(self.montage.rows)
        self.rows_spin.valueChanged.connect(self._layout_changed)
        bar.addWidget(self.rows_spin)

        self.colorbar_box = QtWidgets.QCheckBox("Colorbar")
        self.colorbar_box.setChecked(self.montage.colorbar)
        self.colorbar_box.toggled.connect(self._colorbar_changed)
        bar.addSeparator()
        bar.addWidget(self.colorbar_box)

        bar.addSeparator()
        save = bar.addAction("Save…")
        save.triggered.connect(self.save_screenshot_dialog)

        self.resize(900, 700)
        self.montage.build()
        QtCore.QTimer.singleShot(0, self._post_show)

    # -------- interaction --------
    def _axis_changed(self, index: int):
        """Another orientation: its millimetres are different ones."""
        self.montage.pane = Montage.AXES[int(index)][1]
        self.fit_range()

    def _range_changed(self, _value=None):
        # Typing a series replaces a list that was passed in
        self.montage.set_slices(None)
        (self.montage.start_mm, self.montage.step_mm,
         self.montage.stop_mm) = (spin.value() for spin in self.range_spins)
        self.montage.build()

    def _layout_changed(self, _value=None):
        self.montage.columns = self.columns_spin.value()
        self.montage.rows = self.rows_spin.value()
        self.montage.build()

    def _colorbar_changed(self, checked: bool):
        self.montage.colorbar = bool(checked)
        self.montage.build()

    def fit_range(self):
        """Put start, step and stop back across the whole volume."""
        self.montage.set_slices(None)
        (self.montage.start_mm, self.montage.step_mm,
         self.montage.stop_mm) = self.montage._default_range()
        for spin, value in zip(self.range_spins,
                               (self.montage.start_mm, self.montage.step_mm,
                                self.montage.stop_mm)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.range_label.setText(
            f"  {self.montage.axis_letter()} mm (start step stop) ")
        self.montage.build()

    def _post_show(self):
        try:
            self.vtk_widget.Initialize()
            self.render_window.Render()
        except Exception:
            pass

    def save_screenshot_dialog(self):
        """Ask where to write the sheet and save it."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save montage", "montage.png", "PNG image (*.png)")
        if path:
            _write_png(self.render_window, path, scale=2)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}", 4000)


class VolumeViewerWindow(QtWidgets.QMainWindow):
    """Window around :class:`CatImageViewer`.

    Used by ``CAT_VolView`` and, with *on_position_changed* wired up, as the
    volume window of CAT_SurfView, so both offer the same slices, status line
    and context menu.

    Args:
        image_path: Volume to display.
        surfaces: Surface files or ``vtkPolyData`` drawn as outlines on the
            slices; they must be in the millimetre space of the image.  A
            ``dict`` with ``poly`` and, optionally, ``color``, ``lut`` and
            ``range`` says how to colour one — that is how the surface viewer
            passes the colours of its overlay on.
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

    #: Colours handed out to surface outlines, in this order.  One per surface,
    #: so the two hemispheres — or a central and a pial surface — can be told
    #: apart on the slices; they stay clear of the yellow of a volume contour.
    SURFACE_COLORS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.6, 1.0),
                      (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.55, 0.0))

    def __init__(self, image_path, parent=None, on_position_changed=None,
                 surfaces: Sequence = (), overlay: Optional[str] = None,
                 **viewer_kwargs):
        super().__init__(parent)
        self.image_path = str(image_path)
        # The directory is what tells volumes apart when comparing subjects;
        # the file name itself is in the information panel
        self.setWindowTitle(shorten_path(os.path.dirname(os.path.abspath(self.image_path))))
        self.on_position_changed = on_position_changed
        #: Windows the context menu settings are applied to; see link_windows
        self.peers: List["VolumeViewerWindow"] = [self]
        #: Surfaces outlined on the slices, so the same one is not added twice
        self.surface_paths: List[str] = [str(s) for s in surfaces
                                         if isinstance(s, (str, os.PathLike))]

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
            self.viewer.add_surface(**_surface_display(
                surface, self.SURFACE_COLORS[i % len(self.SURFACE_COLORS)]))
        self.viewer.setup(window_title=os.path.basename(self.image_path))
        self.viewer.on_position_changed = self._position_changed

        self._build_coordinate_bar()
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
        self._install_shortcuts()
        self.setAcceptDrops(True)

        # The interactor may only be initialised once the widget has a window
        QtCore.QTimer.singleShot(0, self._post_show)

    # -------- keyboard --------
    #: Keys, what they do and how they are described in the help.  Slice steps
    #: and zoom follow the pane under the mouse, everything else the window.
    SHORTCUTS = (
        ("Up, Right", "next slice", "_key_next_slice"),
        ("Down, Left", "previous slice", "_key_previous_slice"),
        ("PgUp, PgDown", "ten slices at a time", None),
        ("+, -", "zoom in, zoom out", None),
        ("0", "show the whole volume", None),
        ("o", "go to the origin (0, 0, 0)", "go_to_origin"),
        ("m", "go to the strongest voxel", "go_to_maximum"),
        ("c", "crosshair on/off", "_key_crosshair"),
        ("a", "orientation letters on/off", "_key_orientation"),
        ("i", "image information on/off", "_key_info"),
        ("n", "raw voxels (nearest neighbour) on/off", "_key_nearest"),
        ("p", "control panel on/off", "_key_panel"),
        ("s", "save a screenshot", "save_screenshot_dialog"),
        ("h, ?", "this list", "show_shortcut_help"),
    )

    def _install_shortcuts(self):
        """Bind the keys of :attr:`SHORTCUTS` to the window.

        They are window shortcuts, not interactor bindings: the VTK widget has
        the focus and would otherwise swallow the keys.
        """
        def bind(sequence: str, slot):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

        self._shortcuts: List[QtGui.QShortcut] = []
        for sequence in ("Up", "Right"):
            bind(sequence, self._key_next_slice)
        for sequence in ("Down", "Left"):
            bind(sequence, self._key_previous_slice)
        bind("PgUp", lambda: self._step_slice(10))
        bind("PgDown", lambda: self._step_slice(-10))
        for sequence in ("+", "="):        # '=' is the unshifted '+' key
            bind(sequence, lambda: self._step_zoom(1))
        bind("-", lambda: self._step_zoom(-1))
        bind("0", lambda: self.set_zoom(None))
        bind("o", self.go_to_origin)
        bind("m", self.go_to_maximum)
        bind("c", self._key_crosshair)
        bind("a", self._key_orientation)
        bind("i", self._key_info)
        bind("n", self._key_nearest)
        bind("p", self._key_panel)
        bind("s", self.save_screenshot_dialog)
        for sequence in ("h", "?", "F1"):
            bind(sequence, self.show_shortcut_help)

    def _pane_under_mouse(self) -> Optional[int]:
        """Pane the mouse is over, or None when it is somewhere else."""
        position = self.vtk_widget.mapFromGlobal(QtGui.QCursor.pos())
        if not self.vtk_widget.rect().contains(position):
            return None
        try:
            ratio = self.vtk_widget.devicePixelRatioF()
        except Exception:
            ratio = 1.0
        # VTK counts pixels from the bottom left
        x = int(position.x() * ratio)
        y = int((self.vtk_widget.height() - position.y()) * ratio)
        pane = self.viewer.pane_at(x, y)
        return pane if pane >= 0 else None

    def _step_slice(self, delta: int):
        self.viewer.step_slice(delta, self._pane_under_mouse())
        self._update_label()

    def _key_next_slice(self):
        self._step_slice(1)

    def _key_previous_slice(self):
        self._step_slice(-1)

    def _step_zoom(self, direction: int):
        """Move one step through :attr:`ZOOM_LEVELS` (+1 zooms in)."""
        levels = [mm for _, mm in self.ZOOM_LEVELS]
        try:
            current = levels.index(self.viewer.get_field_of_view())
        except ValueError:
            current = 0
        self.set_zoom(levels[max(0, min(len(levels) - 1, current + direction))])

    def _key_crosshair(self):
        self.set_crosshair(not self.viewer.show_crosshair)

    def _key_orientation(self):
        self.set_orientation_labels(not self.viewer.show_orientation)

    def _key_info(self):
        self.set_info_visible(not self.viewer.show_info)

    def _key_nearest(self):
        self.set_interpolation(not self.viewer.interpolate)

    def _key_panel(self):
        self.dock_controls.setVisible(not self.dock_controls.isVisible())

    def show_shortcut_help(self):
        """List the keys, since a viewer without a menu bar hides them."""
        show_shortcuts(
            self, self.SHORTCUTS,
            "<p>Slice steps and zoom apply to the pane under the mouse; the "
            "mouse wheel steps through slices as well, and right-click opens "
            "the display settings.</p>"
            "<p>Dropping a volume on the window opens it in a linked window, "
            "<b>shift</b> makes it the overlay and <b>alt</b> outlines it; "
            "a surface is outlined too.</p>")

    # -------- drag and drop --------
    def dragEnterEvent(self, event):
        """Take a drag that carries files this viewer can show."""
        if any(self._droppable(url) for url in event.mimeData().urls()):
            event.acceptProposedAction()

    dragMoveEvent = dragEnterEvent

    #: What this window can open when it is dropped on
    DROP_SUFFIXES = SURFACE_SUFFIXES + VOLUME_SUFFIXES

    @staticmethod
    def _droppable(url) -> bool:
        return droppable_url(url, VolumeViewerWindow.DROP_SUFFIXES)

    def dropEvent(self, event):
        """Open, overlay or outline the dropped files.

        Which of the three is decided by the modifier held while dropping, the
        way file managers pick between copy and link.
        """
        paths = dropped_files(event, self.DROP_SUFFIXES)
        if not paths:
            return
        event.acceptProposedAction()
        modifiers = event.modifiers()
        volumes, surfaces = _split_inputs(paths)
        for surface in surfaces:
            self.add_surface(surface)
        if not volumes:
            return
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.set_overlay(volumes[0])
        elif modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
            for volume in volumes:
                self.add_contour(volume)
        else:
            for volume in volumes:
                self.open_volume(volume)

    def add_surface(self, path: str):
        """Draw a surface as an outline on the slices."""
        self.surface_paths.append(str(path))
        try:
            self.viewer.add_surface(path, self.SURFACE_COLORS[
                len(self.viewer.surfaces) % len(self.SURFACE_COLORS)])
            self.viewer._build_surface_contours()
            self.viewer._set_slices_from_index()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Surface", f"Cannot show {os.path.basename(path)}:\n{exc}")

    def open_volume(self, path: str) -> Optional["VolumeViewerWindow"]:
        """Open *path* in a window of its own, linked to this one."""
        if len(self.peers) >= MAX_VOLUMES:
            self.statusBar().showMessage(
                f"Already showing {MAX_VOLUMES} volumes", 4000)
            return None
        window = VolumeViewerWindow(path, **self._viewer_options())
        windows = list(self.peers) + [window]
        link_windows(windows)
        window.show()
        _place_windows(windows)
        return window

    def _viewer_options(self) -> dict:
        """The display settings a newly opened window should start with."""
        viewer = self.viewer
        return {
            'show_info': viewer.show_info,
            'interpolate': viewer.interpolate,
            'recenter': viewer.recenter,
            'lock_zoom': viewer.lock_zoom,
            'show_orientation': viewer.show_orientation,
        }

    # -------- montage --------
    def open_montage(self, **montage_kwargs) -> "MontageWindow":
        """Open a sheet of slices through this volume, for a report figure."""
        montage = MontageWindow(self.viewer, os.path.basename(self.image_path),
                                parent=self, **montage_kwargs)
        montage.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        montage.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        montage.show()
        self._montage = montage      # keep it alive
        return montage

    # -------- screenshot --------
    def save_screenshot_dialog(self):
        """Ask where to write a PNG of this window and save it."""
        default = os.path.join(
            os.path.dirname(os.path.abspath(self.image_path)),
            os.path.splitext(os.path.basename(self.image_path))[0].replace('.nii', '')
            + ".png")
        written = ask_and_save_png(
            self, default, lambda path: self.viewer.save_screenshot(path, scale=2))
        if written:
            self.statusBar().showMessage(f"Saved {os.path.basename(written)}", 4000)

    # -------- coordinate bar --------
    def _build_coordinate_bar(self):
        """Millimetres and voxel index of the cursor, editable, in the status bar.

        Reading a coordinate off the panel is only half of it — a peak from a
        table has to be typed back in, which is what the boxes in SPM's Display
        are for.
        """
        bar = QtWidgets.QWidget(self)
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(4)

        self.mm_boxes: List[QtWidgets.QDoubleSpinBox] = []
        self.voxel_boxes: List[QtWidgets.QSpinBox] = []
        extent = self.viewer._image.GetExtent()

        row.addWidget(QtWidgets.QLabel("mm"))
        for _ in range(3):
            box = QtWidgets.QDoubleSpinBox()
            box.setDecimals(1)
            box.setRange(-9999.0, 9999.0)
            box.setSingleStep(1.0)
            box.setKeyboardTracking(False)
            box.setFixedWidth(72)
            box.editingFinished.connect(self._mm_entered)
            box.valueChanged.connect(lambda _value: self._mm_entered())
            self.mm_boxes.append(box)
            row.addWidget(box)

        row.addSpacing(8)
        row.addWidget(QtWidgets.QLabel("voxel"))
        for axis in range(3):
            box = QtWidgets.QSpinBox()
            box.setRange(extent[2 * axis], extent[2 * axis + 1])
            box.setKeyboardTracking(False)
            box.setFixedWidth(64)
            box.editingFinished.connect(self._voxel_entered)
            box.valueChanged.connect(lambda _value: self._voxel_entered())
            self.voxel_boxes.append(box)
            row.addWidget(box)

        for text, tip, slot in (
                ("Origin", "Go to 0, 0, 0 (o)", self.go_to_origin),
                ("Max", "Go to the strongest voxel (m)", self.go_to_maximum)):
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)

        self._label = QtWidgets.QLabel("")
        row.addSpacing(8)
        row.addWidget(self._label, 1)
        self.statusBar().addWidget(bar, 1)
        #: Guards the boxes against reacting to their own update
        self._filling_boxes = False

    def _mm_entered(self):
        if getattr(self, '_filling_boxes', False):
            return
        self.viewer.set_world_position(
            tuple(box.value() for box in self.mm_boxes), notify=True)
        self._update_label()

    def _voxel_entered(self):
        if getattr(self, '_filling_boxes', False):
            return
        self.viewer.set_index(*[box.value() for box in self.voxel_boxes], notify=True)
        self._update_label()

    def go_to_origin(self):
        """Move every linked window to the origin of the millimetre space."""
        self.viewer.go_to_origin()
        self._update_label()

    def go_to_maximum(self):
        """Move to the strongest voxel of this window (the others follow)."""
        if self.viewer.go_to_maximum() is None:
            self.statusBar().showMessage("No maximum to go to", 3000)
        self._update_label()

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
                self.ctrl.histogram.set_window(lo, hi)

        def _histogram_dragged(lo: float, hi: float):
            """A handle was moved: the boxes and the slices follow."""
            for spin, value in ((self.ctrl.bkg_min, lo), (self.ctrl.bkg_max, hi)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            if hi > lo:
                viewer.set_window_level(hi - lo, 0.5 * (hi + lo))
        self.ctrl.histogram.windowChanged.connect(_histogram_dragged)
        self._fill_histogram()

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

    def _fill_histogram(self):
        """Show the intensities of the displayed image in the histogram."""
        if vtk_to_numpy is None:
            self.ctrl.set_histogram_visible(False)
            return
        scalars = self.viewer._image.GetPointData().GetScalars()
        if scalars is None:
            return
        try:
            self.ctrl.histogram.set_values(vtk_to_numpy(scalars))
        except Exception:
            self.ctrl.set_histogram_visible(False)
            return
        window, level = self.viewer.get_window_level()
        self.ctrl.histogram.set_window(level - 0.5 * window, level + 0.5 * window)

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
        self.ctrl.histogram.set_window(level - 0.5 * window, level + 0.5 * window)
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
            # Only now does the render window know its real size and DPI, which
            # is what the panel text is sized from
            self.viewer._update_info_text()
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self.viewer._update_info_text()   # the panel text follows the size
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass

    # -------- context menu --------
    def _show_context_menu(self, pos):
        """Right-click menu of the slice views, section by section."""
        menu = QtWidgets.QMenu(self)
        self._add_zoom_menu(menu)
        self._add_atlas_menu(menu)
        self._add_overlay_menu(menu)
        self._add_view_menu(menu)
        menu.addSeparator()
        menu.addAction("Montage…").triggered.connect(self.open_montage)
        menu.addAction("Save screenshot…").triggered.connect(
            self.save_screenshot_dialog)
        menu.addAction("Keyboard shortcuts…").triggered.connect(
            self.show_shortcut_help)
        menu.exec(self.vtk_widget.mapToGlobal(pos))

    def _add_zoom_menu(self, menu):
        """Zoom levels, the lock and the re-centring toggle."""
        zoom_menu = menu.addMenu("Zoom")
        current = self.viewer.get_field_of_view()
        for label, mm in self.ZOOM_LEVELS:
            action = zoom_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(current == mm)
            action.triggered.connect(lambda _checked=False, v=mm: self.set_zoom(v))
        zoom_menu.addSeparator()
        lock_action = zoom_menu.addAction("Lock zoom (mouse and touchpad)")
        lock_action.setCheckable(True)
        lock_action.setChecked(self.viewer.lock_zoom)
        lock_action.triggered.connect(
            lambda checked=False: self.set_lock_zoom(checked))
        follow_action = zoom_menu.addAction("Re-centre on cursor")
        follow_action.setCheckable(True)
        follow_action.setChecked(self.viewer.recenter)
        follow_action.triggered.connect(
            lambda checked=False: self.set_recenter(checked))


    def _add_atlas_menu(self, menu):
        """The atlas that names the region under the cursor."""
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


    def _add_overlay_menu(self, menu):
        """Raw voxels, the overlay volume and the contours."""
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

        # Outlines of other volumes — the quickest check of a registration or a
        # segmentation, as in CheckReg
        contour_menu = menu.addMenu("Contours")
        contour_menu.addAction("Add…").triggered.connect(self._choose_contour)
        if self.viewer.contours:
            contour_menu.addSeparator()
            for entry in self.viewer.contours:
                submenu = contour_menu.addMenu(entry['name'])
                colour = QtGui.QColor.fromRgbF(*entry['color'])
                pixmap = QtGui.QPixmap(12, 12)
                pixmap.fill(colour)
                submenu.setIcon(QtGui.QIcon(pixmap))
                submenu.addAction(f"Level: {entry['level']:g}…").triggered.connect(
                    lambda _checked=False, e=entry: self._ask_contour_level(e))
                submenu.addAction("Remove").triggered.connect(
                    lambda _checked=False, p=entry['path']: self.remove_contour(p))
            contour_menu.addSeparator()
            contour_menu.addAction("Remove all").triggered.connect(self.clear_contours)


    def _add_view_menu(self, menu):
        """Controls, information panel, crosshair and letters."""
        panel_action = menu.addAction("Controls")
        panel_action.setCheckable(True)
        panel_action.setChecked(self.dock_controls.isVisible())
        panel_action.triggered.connect(
            lambda checked=False: self.dock_controls.setVisible(checked))

        info_action = menu.addAction("Image information")
        info_action.setCheckable(True)
        info_action.setChecked(self.viewer.show_info)
        info_action.triggered.connect(
            lambda checked=False: self.set_info_visible(checked))

        cross_action = menu.addAction("Crosshair")
        cross_action.setCheckable(True)
        cross_action.setChecked(self.viewer.show_crosshair)
        cross_action.triggered.connect(
            lambda checked=False: self.set_crosshair(checked))

        letters_action = menu.addAction("Orientation letters")
        letters_action.setCheckable(True)
        letters_action.setChecked(self.viewer.show_orientation)
        letters_action.setEnabled(self.viewer._world_from_header)
        letters_action.triggered.connect(
            lambda checked=False: self.set_orientation_labels(checked))

    def _for_each_window(self, action):
        """Run *action* on every linked window, so they stay in step.

        Everything the context menu offers is a display setting, and comparing
        volumes only works when they are displayed the same way.
        """
        for window in self.peers:
            try:
                action(window)
            except Exception:
                continue

    def set_zoom(self, mm: Optional[float]):
        """Zoom the slices to an mm bounding box around the cursor."""
        self._for_each_window(lambda w: w.viewer.set_field_of_view(mm))

    def set_lock_zoom(self, locked: bool):
        """Whether dragging or pinching may change the zoom."""
        self._for_each_window(lambda w: w.viewer.set_lock_zoom(locked))

    def set_recenter(self, recenter: bool):
        """Whether a zoomed view follows the cursor."""
        self._for_each_window(lambda w: w.viewer.set_recenter(recenter))

    def set_interpolation(self, interpolate: bool):
        """Smooth the slices, or draw the raw voxels."""
        def apply(window):
            window.viewer.set_interpolation(interpolate)
            window._update_label()   # the reported value follows the display
        self._for_each_window(apply)

    def set_info_visible(self, visible: bool):
        """Show or hide the information panel."""
        self._for_each_window(lambda w: w.viewer.set_info_visible(visible))

    def set_crosshair(self, visible: bool):
        """Show or hide the crosshair."""
        self._for_each_window(lambda w: w.viewer.set_crosshair_visible(visible))

    # -------- contours --------
    def add_contour(self, path: str, level: Optional[float] = None):
        """Outline *path* on every linked window.

        Comparing subjects means outlining the same volume in all of them, so
        this follows the other display settings and is broadcast.
        """
        failed = []

        def apply(window):
            try:
                window.viewer.add_contour(path, level)
            except Exception as exc:
                failed.append(str(exc))

        self._for_each_window(apply)
        if failed:
            QtWidgets.QMessageBox.warning(
                self, "Contour",
                f"Cannot outline {os.path.basename(path)}:\n{failed[0]}")
            return
        entry = next((e for e in self.viewer.contours
                      if e['path'] == str(path)), None)
        if entry is not None and not self.viewer.contour_lines_shown(entry):
            self.statusBar().showMessage(
                f"{entry['name']} is outlined, but nothing crosses level "
                f"{entry['level']:g} on these slices — move the cursor, or set "
                "another level from the Contours menu", 8000)

    def remove_contour(self, path: str):
        """Remove the outline of *path* everywhere."""
        self._for_each_window(lambda w: w.viewer.remove_contour(path))

    def clear_contours(self):
        """Remove every outline everywhere."""
        self._for_each_window(lambda w: w.viewer.clear_contours())

    def set_contour_level(self, path: str, level: float):
        """Change the level of an outline everywhere."""
        self._for_each_window(lambda w: w.viewer.set_contour_level(path, level))

    def _choose_contour(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Outline which volume?", os.path.dirname(self.image_path),
            "NIfTI (*.nii *.nii.gz);;All files (*)")
        if path:
            self.add_contour(path)

    def _ask_contour_level(self, entry: dict):
        low, high = entry['range']
        level, accepted = QtWidgets.QInputDialog.getDouble(
            self, "Contour level", f"Intensity to follow ({low:g} … {high:g}):",
            float(entry['level']), -1e9, 1e9, 4)
        if accepted:
            self.set_contour_level(entry['path'], level)

    def set_orientation_labels(self, visible: bool):
        """Show or hide the anatomical direction letters."""
        self._for_each_window(lambda w: w.viewer.set_orientation_labels(visible))

    def set_atlas(self, path: Optional[str]):
        """Use *path* to name the region under the cursor (None switches off)."""
        self._for_each_window(lambda w: w.viewer.set_atlas(path))

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
        self._filling_boxes = True
        try:
            for box, value in zip(self.mm_boxes, world):
                box.setValue(round(float(value), 1))
            for box, value in zip(self.voxel_boxes, ijk):
                box.setValue(int(value))
        finally:
            self._filling_boxes = False
        value = self.viewer.get_value()
        text = f"value {_format_value(value)}"
        if self.viewer.overlay_path:
            text = (f"overlay {_format_value(value)}    "
                    f"image {_format_value(self.viewer.get_background_value())}")
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

#: Options whose value may start with a minus, which argparse would otherwise
#: read as another option.  Slice positions are negative half the time.
_MINUS_VALUE_OPTIONS = ("--slices",)


def _attach_minus_values(argv: Sequence[str]) -> List[str]:
    """Join ``--slices -30:10:30`` into ``--slices=-30:10:30``.

    argparse only lets a value start with '-' when it looks like a plain
    number, so '-30:10:30' and '-30 -15 0' would be taken for options.  The
    attached form is unambiguous, and writing it is not the user's job.
    """
    out: List[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if (token in _MINUS_VALUE_OPTIONS and index + 1 < len(argv)
                and str(argv[index + 1]).startswith("-")):
            out.append(f"{token}={argv[index + 1]}")
            index += 2
            continue
        out.append(token)
        index += 1
    return out


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
              "Every volume opens its own linked window."),
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
        help=("Volume drawn in colour on top of the image; one on a different "
              "grid is resampled through the millimetre space of the headers"),
    )
    p.add_argument(
        "--contour", type=str, action="append", default=None,
        metavar="VOLUME",
        help=("Volume outlined on the slices (repeatable); the fastest check "
              "of a registration or a segmentation"),
    )
    p.add_argument(
        "--atlas", type=str, default=None,
        help=("Atlas volume naming the region under the cursor; also "
              "selectable from the right-click menu"),
    )
    montage = p.add_argument_group(
        "montage",
        "A sheet of slices instead of the three orthogonal views.  With "
        "--screenshot it is written without opening a window, which is what "
        "makes it usable in a script.")
    montage.add_argument(
        "--montage", action="store_true",
        help="Show (or write) a montage of slices",
    )
    montage.add_argument(
        "--slices", type=str, default=None, metavar="SPEC",
        help=("Slices in mm, either a list ('25 30 40 80') or a range "
              "('-40:10:60' = start:step:stop); across the volume by default"),
    )
    montage.add_argument(
        "--orientation", choices=["axial", "coronal", "sagittal"],
        default="axial",
        help="Plane the montage cuts",
    )
    montage.add_argument(
        "--columns", type=int, default=0,
        help="Columns of the sheet (0 works it out)",
    )
    montage.add_argument(
        "--rows", type=int, default=0,
        help="Rows of the sheet (0 works it out)",
    )
    montage.add_argument(
        "--colorbar", action="store_true",
        help=("Draw a colour bar; a -log10(p) overlay ('log' in its name) is "
              "labelled with p-values"),
    )
    montage.add_argument(
        "--no-labels", action="store_true",
        help="Leave out the position label in each tile",
    )
    montage.add_argument(
        "--montage-size", nargs=2, type=int, default=[1200, 900],
        metavar=("WIDTH", "HEIGHT"),
        help="Pixel size of the written montage",
    )

    colours = p.add_argument_group(
        "overlay colours", "How the overlay is coloured, as in the control panel")
    colours.add_argument(
        "--range", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
        help="Value range the colormap covers (the overlay's own by default)",
    )
    colours.add_argument(
        "--clip", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
        help="Hide values in between, so the image shows through",
    )
    colours.add_argument(
        "--threshold", type=float, default=None, metavar="P",
        help=("Threshold a -log10(p) overlay at this p-value, i.e. clip to "
              "+/- -log10(P); 0.05, 0.01 and 0.001 are the usual ones"),
    )
    colours.add_argument(
        "--colormap", type=str, default=None, choices=list(COLORMAP_NAMES),
        help="Colormap of the overlay",
    )
    colours.add_argument(
        "--opacity", type=float, default=None, metavar="0..1",
        help="Opacity of the overlay",
    )
    colours.add_argument(
        "--inverse", action="store_true",
        help="Flip the sign of the overlay",
    )
    colours.add_argument(
        "--discrete", type=int, default=None, metavar="N",
        help="Draw the colormap in N steps instead of continuously",
    )
    colours.add_argument(
        "--range-bkg", nargs=2, type=float, default=None,
        metavar=("LOW", "HIGH"),
        help="Display range of the image underneath (percentiles by default)",
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
        "--no-orientation", action="store_true",
        help="Leave out the L/R/A/P/S/I letters along the pane edges",
    )
    p.add_argument(
        "--free-zoom", action="store_true",
        help="Allow zooming by dragging or pinching (off by default, because "
             "the mouse then keeps changing the zoom)",
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
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # Unknown arguments are looked at first: a list of slices has to be one
    # argument, and unquoted its numbers end up here rather than in --slices
    args, extra = p.parse_known_args(_attach_minus_values(argv))
    numbers = [item for item in list(args.inputs) + list(extra)
               if re.fullmatch(r"[-+]?[0-9]*\.?[0-9]+", str(item))]
    if numbers:
        p.error(f"{' '.join(numbers)}: slice positions have to be quoted, "
                f'e.g. --slices "25 30 40 80"')
    if extra:
        p.error(f"unrecognized arguments: {' '.join(extra)}")
    return args



#: Volumes opened at once, one window each
MAX_VOLUMES = 6

#: Windows per row when they are tiled; the rest go into further rows
WINDOWS_PER_ROW = 3


def parse_slices(text: str) -> Tuple[Optional[List[float]], Optional[Tuple[float, float, float]]]:
    """Read a slice specification into either a list or a start/step/stop series.

    Two forms, both in millimetres, as they are written in
    ``cat_vol_slice_overlay``::

        "25 30 40 80"      an explicit list (commas work as well)
        "-40:10:60"        start:step:stop, or start:stop for 1 mm steps

    Returns:
        ``(list, None)`` or ``(None, (start, step, stop))``.

    Raises:
        ValueError: when the text is neither.
    """
    text = str(text).strip()
    if not text:
        raise ValueError("no slices given")
    if ":" in text:
        parts = [p for p in text.split(":") if p.strip()]
        try:
            values = [float(p) for p in parts]
        except ValueError:
            raise ValueError(f"cannot read the slice range {text!r}")
        if len(values) == 2:
            return None, (values[0], 1.0, values[1])
        if len(values) == 3:
            return None, (values[0], values[1], values[2])
        raise ValueError("a slice range is start:step:stop or start:stop")
    try:
        values = [float(p) for p in text.replace(",", " ").split()]
    except ValueError:
        raise ValueError(f"cannot read the slice list {text!r}")
    if not values:
        raise ValueError("no slices given")
    return values, None


def is_logp_name(filename: Optional[str]) -> bool:
    """True when a file name marks a -log10(p) map ('log', as in CAT12)."""
    if not filename:
        return False
    return 'log' in os.path.basename(str(filename)).lower()


def _surface_display(surface, color) -> dict:
    """The :meth:`CatImageViewer.add_surface` arguments for one surface.

    A file name or a polydata is outlined in *color*.  A dict is how the surface
    viewer hands a hemisphere over: it brings the mesh, and with an overlay on
    it the lookup table and value range it is coloured through, so the outline
    on the slices matches the surface it was taken from.
    """
    if isinstance(surface, dict):
        return {'surface': surface['poly'],
                'color': surface.get('color') or color,
                'lut': surface.get('lut'),
                'scalar_range': surface.get('range')}
    return {'surface': surface, 'color': color}


def _split_inputs(inputs: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Sort the positional arguments into volumes and surfaces by extension."""
    volumes, surfaces = [], []
    for item in inputs:
        (surfaces if str(item).lower().endswith(SURFACE_SUFFIXES) else volumes).append(str(item))
    return volumes, surfaces


def link_windows(windows: Sequence["VolumeViewerWindow"]):
    """Tie several viewer windows together.

    Their cursors stay on the same world position, and what the context menu
    changes — zoom, atlas, interpolation, … — applies to all of them.

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
        window.peers = windows          # display settings apply to the group
    return windows


def _place_windows(windows: Sequence["VolumeViewerWindow"]):
    """Tile the windows over the screen, shrinking them until they fit.

    Comparing volumes means seeing them next to each other: up to three go
    side by side, more fill a second row below (four to six windows give three
    on top and the rest underneath).  Only when the tiles would become
    unusably small are the windows cascaded instead.
    """
    if len(windows) < 2:
        return
    try:
        available = QtWidgets.QApplication.primaryScreen().availableGeometry()
    except Exception:
        return

    count = len(windows)
    columns = min(WINDOWS_PER_ROW, count)
    rows = -(-count // columns)          # ceil
    gap = 12
    width = max(w.width() for w in windows)
    height = max(w.height() for w in windows)
    scale = min(1.0,
                (available.width() - (columns - 1) * gap) / float(columns * width),
                (available.height() - (rows - 1) * gap) / float(rows * height))
    if scale < 1.0 and width * scale < 300:
        for i, window in enumerate(windows):   # shrinking would make them useless
            window.move(available.left() + 40 * i, available.top() + 40 * i)
        return

    width = int(width * scale)
    height = int(height * scale)
    top = available.top() + max(0, (available.height() - rows * height
                                    - (rows - 1) * gap) // 2)
    for i, window in enumerate(windows):
        row, column = divmod(i, columns)
        # A short last row is centred under the ones above
        in_row = min(columns, count - row * columns)
        row_width = in_row * width + (in_row - 1) * gap
        left = available.left() + max(0, (available.width() - row_width) // 2)
        window.resize(width, height)
        window.move(left + column * (width + gap), top + row * (height + gap))


def _montage_options(args) -> dict:
    """Montage settings from the command line, ready for :class:`Montage`.

    Raises:
        ValueError: when the slice specification cannot be read.
    """
    panes = {"axial": CatImageViewer.VIEW_AXIAL,
             "coronal": CatImageViewer.VIEW_CORONAL,
             "sagittal": CatImageViewer.VIEW_SAGITTAL}
    slices = slices_mm = None
    if args.slices:
        slices, slices_mm = parse_slices(args.slices)
    return {
        'pane': panes[args.orientation],
        'slices': slices,
        'slices_mm': slices_mm,
        'columns': args.columns,
        'rows': args.rows,
        'labels': not args.no_labels,
        'colorbar': args.colorbar,
    }


def _apply_overlay_options(viewer: CatImageViewer, args):
    """Colour the overlay the way the command line asks for.

    Called after the overlay is loaded, since its own value range is the
    default for everything here.
    """
    if args.range_bkg:
        low, high = sorted(float(v) for v in args.range_bkg)
        if high > low:
            viewer.set_window_level(high - low, 0.5 * (high + low))
    if viewer.overlay_path is None and not args.overlay:
        return
    if args.range:
        viewer.overlay_range = [float(args.range[0]), float(args.range[1])]
    if args.clip:
        viewer.overlay_clip = (float(args.clip[0]), float(args.clip[1]))
    if args.threshold:
        # A p-value is thresholded on the -log10(p) scale the map is stored in
        edge = -math.log10(float(args.threshold))
        viewer.overlay_clip = (-edge, edge)
    if args.colormap:
        viewer.overlay_colormap = COLORMAP_ORDER[
            list(COLORMAP_NAMES).index(args.colormap)]
    if args.opacity is not None:
        viewer.overlay_opacity = max(0.0, min(1.0, float(args.opacity)))
    if args.inverse:
        viewer.overlay_inverse = True
    if args.discrete is not None:
        viewer.overlay_discrete = max(0, int(args.discrete))
    viewer.refresh_overlay()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry-point."""
    if argv is None and running_as_app() and len(sys.argv) < 2:
        # Double-clicked in Finder: ask for the volume instead of failing on
        # the missing argument
        install_qt_message_filter()
        app = qt_application()
        chosen = ask_for_files(app, "Open volume",
                               "Volumes (*.nii *.nii.gz *.mnc *.mha *.mhd *.nrrd);;"
                               "All files (*)")
        if not chosen:
            return 0
        argv = chosen
    args = _parse_args(argv)

    pct = None if args.no_percentile else tuple(args.percentile)
    set_verbose(args.verbose)
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
        lock_zoom=not args.free_zoom,
        show_orientation=not args.no_orientation,
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

    try:
        montage_options = _montage_options(args)
    except ValueError as exc:
        print(f"[cat_vol_view] {exc}", file=sys.stderr)
        return 2

    if args.screenshot or args.headless:
        return _render_without_a_window(args, options, volumes, surfaces,
                                        montage_options)
    return _open_windows(args, options, volumes, surfaces, montage_options)


def _render_without_a_window(args, options: dict, volumes: Sequence[str],
                             surfaces: Sequence[str], montage_options: dict) -> int:
    """Write the screenshots a batch call asks for, without opening a window."""
    # Batch mode renders without a window, so no Qt application is needed
    for i, volume in enumerate(volumes):
        viewer = CatImageViewer(**options)
        viewer.load_image(volume)
        colors = VolumeViewerWindow.SURFACE_COLORS
        for s, surf in enumerate(surfaces):
            viewer.add_surface(surf, colors[s % len(colors)])
        viewer.setup(window_title=os.path.basename(volume))
        if args.overlay:
            viewer.set_overlay(args.overlay)
        _apply_overlay_options(viewer, args)
        if args.atlas:
            viewer.set_atlas(args.atlas)
        for contour in (args.contour or ()):
            try:
                viewer.add_contour(contour)
            except Exception as exc:
                # A batch figure is worth finishing without its outline, but the
                # reason has to be said rather than left to a blank slice
                print(f"[cat_vol_view] {exc}", file=sys.stderr)
        screenshot = args.screenshot
        if screenshot and len(volumes) > 1:
            stem, ext = os.path.splitext(screenshot)
            screenshot = f"{stem}_{i + 1}{ext or '.png'}"
        if args.montage:
            viewer.render(headless=True)     # the cameras the sheet copies
            if not screenshot:
                print("[cat_vol_view] --montage --headless writes nothing "
                      "without --screenshot", file=sys.stderr)
                return 2
            written = render_montage(
                viewer, screenshot, size=tuple(args.montage_size),
                on_message=lambda text: print(f"[cat_vol_view] {text}"),
                **montage_options)
            if args.verbose:
                print(f"[cat_vol_view] Wrote montage: {written}")
        else:
            viewer.render(screenshot=screenshot, headless=True)
    return 0


def _open_windows(args, options: dict, volumes: Sequence[str],
                  surfaces: Sequence[str], montage_options: dict) -> int:
    """Show one window per volume, link them, and run the application."""
    # Interactive start only: a batch render should have no side effects
    install_qt_message_filter()
    ensure_apps_exist()      # macOS: leaves the app bundles behind, once
    app = qt_application()
    windows = []
    for volume in volumes:
        window = VolumeViewerWindow(volume, surfaces=surfaces, **options)
        _apply_overlay_options(window.viewer, args)
        if args.overlay:
            window.set_overlay(args.overlay)     # the panel follows the file
            _apply_overlay_options(window.viewer, args)
            window._sync_control_panel()
        if args.atlas:
            window.set_atlas(args.atlas)
        windows.append(window)
    link_windows(windows)
    for contour in (args.contour or ()):
        # After linking, so one call outlines it in every window
        windows[0].add_contour(contour)
    for window in windows:
        window.show()
    _place_windows(windows)
    if args.montage:
        for window in windows:
            window.open_montage(**montage_options)
    # A file double-clicked in Finder while the viewer runs is opened as if it
    # had been dropped on the window, rather than being dropped on the floor
    finder_open_files(app).set_handler(
        lambda paths: _open_from_finder(windows, paths))
    return int(app.exec())


def _open_from_finder(windows: List["VolumeViewerWindow"], paths: Sequence[str]):
    """Open what macOS sent in the window that is still there.

    Windows the user has closed are skipped: the first one that is still open
    takes the files, and its peers follow it as they do for a drop.

    A file that is already on screen is raised instead of opened again.  macOS
    sends an open-document event for the files given on the command line as
    well, so without this every one of them would appear twice; and a
    double-click on a volume that is already open should bring it forward
    rather than clone it.
    """
    known = list(windows)
    for window in windows:
        for peer in getattr(window, "peers", ()):
            if peer not in known:
                known.append(peer)     # opened after the start, e.g. by a drop
    alive = [w for w in known if w.isVisible()]
    if not alive:
        return
    volumes, surfaces = _split_inputs(paths)
    for surface in surfaces:
        if not any(_same_file(surface, shown)
                   for w in alive for shown in getattr(w, "surface_paths", ())):
            alive[0].add_surface(surface)
    for volume in volumes:
        shown = next((w for w in alive if _same_file(w.image_path, volume)), None)
        if shown is not None:
            shown.raise_()
            shown.activateWindow()
            continue
        alive[0].open_volume(volume)


def _same_file(one: Optional[str], other: Optional[str]) -> bool:
    """Whether two paths name the same file, symlinks and ``..`` included."""
    if not one or not other:
        return False
    try:
        return os.path.realpath(str(one)) == os.path.realpath(str(other))
    except OSError:
        return str(one) == str(other)


if __name__ == "__main__":
    raise SystemExit(main())
