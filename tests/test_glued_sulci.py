"""Tests for :mod:`t1prep.glued_sulci` and the report merge that consumes it.

A glued sulcus is surface-to-surface *contact*, not a self-intersection, so
the tests below build meshes that touch without crossing and check that the
measure separates them from ordinary geometry.
"""

import json
import os

import numpy as np
import pytest

# Allow running tests without installing the package (repo checkout / editable dev)
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from t1prep.glued_sulci import glued_sulci


def _grid(nx=12, ny=12, z=0.0, spacing=1.0, flip=False):
    """One flat sheet of triangles on the plane ``z``.

    ``flip`` reverses the winding, which reverses the normals -- that is how
    two sheets are made to face each other rather than point the same way.
    """
    xs, ys = np.meshgrid(np.arange(nx) * spacing, np.arange(ny) * spacing,
                         indexing="ij")
    verts = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, z)]
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = a + 1
            c = a + ny
            d = c + 1
            faces += [[a, c, b], [b, c, d]] if not flip else [[a, b, c], [b, d, c]]
    return verts, np.asarray(faces, dtype=np.int64)


def _two_sheets(gap):
    """Two facing sheets separated by ``gap`` mm, as a single mesh."""
    v1, f1 = _grid(z=0.0, flip=False)
    v2, f2 = _grid(z=gap, flip=True)
    return np.vstack([v1, v2]), np.vstack([f1, f2 + len(v1)])


def test_single_sheet_is_not_glued():
    """Ordinary local geometry must not be reported."""
    v, f = _grid()
    assert glued_sulci(v, f)["glued_vertices"] == 0


def test_facing_sheets_in_contact_are_glued():
    """Two banks 0.4 mm apart and facing each other is the defect."""
    v, f = _two_sheets(gap=0.4)
    out = glued_sulci(v, f)
    assert out["glued_vertices"] > 0
    assert out["glued_fraction"] > 0.5, out


def test_open_sulcus_is_not_glued():
    """The same two banks 3 mm apart are a normal sulcus."""
    v, f = _two_sheets(gap=3.0)
    assert glued_sulci(v, f)["glued_vertices"] == 0


def test_contact_requires_facing_normals():
    """Sheets that touch but point the same way are not a glued sulcus.

    Without the normal test any thin structure would be flagged, so this
    pins the discriminator rather than just the distance.
    """
    v1, f1 = _grid(z=0.0, flip=False)
    v2, f2 = _grid(z=0.4, flip=False)          # same winding -> parallel
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    assert glued_sulci(v, f)["glued_vertices"] == 0


@pytest.mark.parametrize("gap,expected", [(0.2, True), (0.4, True),
                                          (1.5, False), (5.0, False)])
def test_radius_threshold(gap, expected):
    v, f = _two_sheets(gap=gap)
    assert (glued_sulci(v, f)["glued_vertices"] > 0) is expected


def test_area_and_counts_are_reported():
    v, f = _grid(nx=5, ny=5, spacing=2.0)
    out = glued_sulci(v, f)
    assert out["n_vertices"] == 25
    # 4x4 quads of 2x2 mm
    assert out["area"] == pytest.approx(64.0)


def test_degenerate_input_is_safe():
    assert glued_sulci(np.zeros((0, 3)), np.zeros((0, 3), int))[
        "glued_vertices"] == 0


def test_report_merge(tmp_path):
    """Sidecars fold into ``qualitymeasures`` and are cleaned up."""
    from t1prep.t1prep import _merge_glued_sulci_qa

    report = tmp_path / "log_SUB01.json"
    report.write_text(json.dumps({"qualitymeasures": {"euler_lh": {"value": 2}}}))
    for hemi, frac, sigma in (("lh", 0.005, 0.6), ("rh", 0.0009, None)):
        (tmp_path / f"SUB01_glued-{hemi}.json").write_text(json.dumps({
            "hemi": hemi, "report_file": "log_SUB01.json",
            "glued_vertices": 1, "glued_fraction": frac,
            "n_vertices": 1000, "area": 1.0, "sulci_sigma_factor": sigma,
        }))
    _merge_glued_sulci_qa(str(tmp_path), "SUB01")

    qa = json.loads(report.read_text())["qualitymeasures"]
    assert qa["glued_lh"]["value"] == pytest.approx(0.5)
    assert qa["glued_rh"]["value"] == pytest.approx(0.09)
    assert qa["glued_lh_sigma"]["value"] == 0.6
    assert "glued_rh_sigma" not in qa          # no escalation -> not recorded
    assert qa["euler_lh"] == {"value": 2}      # existing measures preserved
    assert not list(tmp_path.glob("*_glued-*.json"))


def test_report_merge_without_sidecars_is_a_noop(tmp_path):
    from t1prep.t1prep import _merge_glued_sulci_qa
    report = tmp_path / "log_SUB01.json"
    report.write_text(json.dumps({"qualitymeasures": {}}))
    _merge_glued_sulci_qa(str(tmp_path), "SUB01")
    assert json.loads(report.read_text()) == {"qualitymeasures": {}}
