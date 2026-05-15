"""
Re-export of the ``cat_surf`` package through the ``t1prep`` namespace.

This module exists so that tools building on T1Prep (e.g. Nipype interfaces)
only need ``t1prep`` as a dependency and can access all CAT-Surface
functionality through a single, consistent namespace::

    import t1prep.cat_surf as cs
    v, fcs = cs.read_surface("path/to/surface.gii")

    from t1prep.cat_surf import cli
    cli.surf_reduce(...)

All public symbols from ``cat_surf`` and its ``cli`` sub-module are
available here with identical names and signatures.
"""

import os as _os
import sys as _sys

# ---------------------------------------------------------------------------
# Bootstrap: import the REAL installed cat_surf C-extension, not ourselves.
#
# When src/t1prep/ is on sys.path (e.g. because segment.py was run as a
# script, which causes Python to add its directory to sys.path[0]),
# 'import cat_surf' would resolve to this wrapper directory and trigger a
# circular import.  We temporarily remove every sys.path entry that points
# to our own parent (src/t1prep/) so importlib finds the installed package.
# ---------------------------------------------------------------------------
_our_parent = _os.path.realpath(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_popped: list = []
_i = 0
while _i < len(_sys.path):
    _entry = _sys.path[_i]
    if _entry and _os.path.realpath(_entry) == _our_parent:
        _sys.path.pop(_i)
        _popped.append((_i, _entry))
    else:
        _i += 1
try:
    import cat_surf as _cat_surf
    from cat_surf import cli  # noqa: F401 – re-exported as t1prep.cat_surf.cli
finally:
    for _idx, _entry in _popped:
        _sys.path.insert(_idx, _entry)
    del _i, _popped, _our_parent

# Re-export every public symbol from the top-level cat_surf package.
from cat_surf import (  # noqa: F401
    bbreg,
    bbreg_detect_contrast,
    central_to_pial,
    correct_thickness_folding,
    count_intersections,
    euler_characteristic,
    get_area,
    get_area_normalized,
    hausdorff_distance,
    point_distance,
    point_distance_mean,
    read_surface,
    read_values,
    reduce_mesh,
    remove_intersections,
    resample_annot,
    resample_to_sphere,
    smooth_heatkernel,
    smooth_mesh,
    smoothed_curvatures,
    sphere_radius,
    sulcus_depth,
    surf_average,
    surf_curvature,
    surf_deform,
    surf_to_pial_white,
    surf_to_sphere,
    surf_warp,
    vol2surf,
    vol_amap,
    vol_blood_vessel_correction,
    vol_marching_cubes,
    vol_sanlm,
    vol_thickness_pbt,
    volume_register_nmi,
    volume_register_robust,
    write_surface,
    write_values,
)

__version__ = _cat_surf.__version__

__all__ = [
    # I/O
    "read_surface",
    "write_surface",
    "read_values",
    "write_values",
    # surface metrics
    "get_area",
    "get_area_normalized",
    "euler_characteristic",
    "point_distance",
    "point_distance_mean",
    "hausdorff_distance",
    "sphere_radius",
    "sulcus_depth",
    "smoothed_curvatures",
    "surf_curvature",
    "count_intersections",
    # surface processing
    "smooth_heatkernel",
    "smooth_mesh",
    "reduce_mesh",
    "remove_intersections",
    "surf_average",
    "surf_deform",
    "surf_warp",
    "resample_to_sphere",
    "resample_annot",
    # surface topology
    "surf_to_sphere",
    "surf_to_pial_white",
    "central_to_pial",
    "correct_thickness_folding",
    # volume operations
    "vol_sanlm",
    "vol_blood_vessel_correction",
    "vol_thickness_pbt",
    "vol_amap",
    "vol_marching_cubes",
    "vol2surf",
    # registration
    "bbreg",
    "bbreg_detect_contrast",
    "volume_register_nmi",
    "volume_register_robust",
    # sub-modules
    "cli",
]
