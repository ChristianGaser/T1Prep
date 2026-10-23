"""Tests for :func:`t1prep.surface_estimation.write_curv`.

A sphere of radius R is convex everywhere with mean curvature 1/R, so
FreeSurfer's sign makes its ``curv`` -1/R.  One vertex pulled 5 mm inward
stands in for the degenerate vertices of a white surface: its long edges keep
the heat kernel from mixing it with its neighbours, so without the clip it
stays at +0.62 after smoothing.
"""

import numpy as np
import pytest

# Allow running tests without installing the package (repo checkout / editable dev)
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

cat_surf = pytest.importorskip("cat_surf")
se = pytest.importorskip("t1prep.surface_estimation")

from tests.test_fslr import _icosphere  # noqa: E402

RADIUS = 50.0


def _curv(tmp_path, vertices, faces):
    surface = str(tmp_path / "lh.white.gii")
    cat_surf.write_surface(surface, vertices, faces.astype(np.int32))
    output = str(tmp_path / "lh.curv.gii")
    se.write_curv(surface, output)
    return cat_surf.read_values(output)


def test_sphere_has_freesurfer_sign(tmp_path):
    vertices, faces = _icosphere(subdivisions=6, radius=RADIUS)  # 0.94 mm edges
    curv = _curv(tmp_path, vertices, faces)
    assert np.median(curv) == pytest.approx(-1.0 / RADIUS, rel=0.01)


def test_degenerate_vertex_does_not_survive(tmp_path):
    vertices, faces = _icosphere(subdivisions=6, radius=RADIUS)
    vertices[0] *= 0.9
    curv = _curv(tmp_path, vertices, faces)
    assert np.abs(curv + 1.0 / RADIUS).max() < 0.01
