"""Tests for the shared sulcal-barrier reference of the two hemispheres.

The heavy parts -- vessel correction and the reference estimate itself -- are
replaced by stubs, so these tests only check which references are computed
and how the report picks them up.
"""

import json

import pytest

# Allow running tests without installing the package (repo checkout / editable dev)
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

se = pytest.importorskip("t1prep.surface_estimation")

REFS = {"lh.seg.nii": 2.6, "rh.seg.nii": 2.4}


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    """Label maps on disk and stubs returning a fixed reference for each."""
    for name in REFS:
        (tmp_path / name).write_bytes(b"")
    calls = []

    def fake_input(path, vessel):
        return None, _Path(path).name, (0.5, 0.5, 0.5)

    def fake_reference(vol, voxelsize=None, **kw):
        calls.append(vol)
        return REFS[vol]

    monkeypatch.setattr(se, "_pbt_input", fake_input)
    monkeypatch.setattr(se.cat_surf, "vol_pbt_barrier_reference",
                        fake_reference, raising=False)
    return tmp_path, calls


def _refs(tmp_path, other="rh.seg.nii"):
    return se._barrier_references(own_vol="lh.seg.nii",
                                  own_zooms=(0.5, 0.5, 0.5),
                                  other_src=str(tmp_path / other),
                                  vessel=1, pbt_kw={})


def test_both_references_are_estimated(stubs):
    tmp_path, calls = stubs
    assert _refs(tmp_path) == (pytest.approx(2.6), pytest.approx(2.4))
    assert calls == ["lh.seg.nii", "rh.seg.nii"]


def test_missing_contralateral_hemisphere(stubs):
    tmp_path, calls = stubs
    assert _refs(tmp_path, other="nothere.nii") == (pytest.approx(2.6), None)
    assert calls == ["lh.seg.nii"]


def test_old_cat_surf_falls_back(stubs, monkeypatch):
    tmp_path, _ = stubs
    monkeypatch.delattr(se.cat_surf, "vol_pbt_barrier_reference")
    assert _refs(tmp_path) is None


def test_report_merge(tmp_path):
    from t1prep.t1prep import _merge_glued_sulci_qa

    report = tmp_path / "log_SUB01.json"
    report.write_text(json.dumps({"qualitymeasures": {"euler_lh": {"value": 2}}}))
    for hemi, ref in (("lh", 2.6), ("rh", 2.4)):
        (tmp_path / f"SUB01_glued-{hemi}.json").write_text(json.dumps({
            "hemi": hemi, "report_file": "log_SUB01.json",
            "sulci_sigma_factor": None,
            "barrier_reference": ref, "barrier_reference_shared": 2.5,
        }))
    _merge_glued_sulci_qa(str(tmp_path), "SUB01")

    qa = json.loads(report.read_text())["qualitymeasures"]
    assert qa["barrier_ref_lh"]["value"] == pytest.approx(2.6)
    assert qa["barrier_ref_rh"]["value"] == pytest.approx(2.4)
    assert qa["barrier_ref_shared"]["value"] == pytest.approx(2.5)
    assert qa["euler_lh"] == {"value": 2}
    assert not list(tmp_path.glob("*_glued-*.json"))


def test_report_merge_without_shared_reference(tmp_path):
    from t1prep.t1prep import _merge_glued_sulci_qa

    report = tmp_path / "log_SUB01.json"
    report.write_text(json.dumps({"qualitymeasures": {}}))
    (tmp_path / "SUB01_glued-lh.json").write_text(json.dumps({
        "hemi": "lh", "report_file": "log_SUB01.json",
        "sulci_sigma_factor": None, "barrier_reference": 2.6,
    }))
    _merge_glued_sulci_qa(str(tmp_path), "SUB01")
    qa = json.loads(report.read_text())["qualitymeasures"]
    assert set(qa) == {"barrier_ref_lh"}
