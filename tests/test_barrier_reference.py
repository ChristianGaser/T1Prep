"""Tests for the shared sulcal-barrier reference of the two hemispheres.

The heavy parts -- vessel correction and the reference estimate itself -- are
replaced by stubs, so these tests only check the coordination: which value is
used, when a published value is trusted, and how the report picks it up.
"""

import json
import logging
import os

import pytest

# Allow running tests without installing the package (repo checkout / editable dev)
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

se = pytest.importorskip("t1prep.surface_estimation")

LOG = logging.getLogger("test_barrier_reference")
REFS = {"lh.seg.nii": 2.6, "rh.seg.nii": 2.4}


@pytest.fixture
def hemis(tmp_path, monkeypatch):
    """Two label maps and stubs returning a fixed reference for each."""
    for name in REFS:
        (tmp_path / name).write_bytes(b"")
    calls = []

    def fake_input(path, vessel):
        return None, os.path.basename(path), (0.5, 0.5, 0.5)

    def fake_reference(vol, voxelsize=None, **kw):
        calls.append(vol)
        return REFS[vol]

    monkeypatch.setattr(se, "_pbt_input", fake_input)
    monkeypatch.setattr(se.cat_surf, "vol_pbt_barrier_reference",
                        fake_reference, raising=False)
    return tmp_path, calls


def _shared(tmp_path, other="rh.seg.nii", timeout=5.0):
    return se._shared_barrier_reference(
        log=LOG, own_vol="lh.seg.nii", own_zooms=(0.5, 0.5, 0.5),
        own_src=str(tmp_path / "lh.seg.nii"),
        other_src=str(tmp_path / other),
        own_file=str(tmp_path / "S_barrier-ref-lh.json"),
        other_file=str(tmp_path / "S_barrier-ref-rh.json"),
        fshemi="lh", report_file="log_S.json", vessel=1, pbt_kw={},
        timeout=timeout)


def _publish(tmp_path, reference, pid=None, mtime=None):
    src = tmp_path / "rh.seg.nii"
    (tmp_path / "S_barrier-ref-rh.json").write_text(json.dumps({
        "hemi": "rh", "report_file": "log_S.json",
        "source": os.path.abspath(src),
        "source_mtime": os.path.getmtime(src) if mtime is None else mtime,
        "pid": pid, "reference": reference,
    }))


def test_mean_of_both_hemispheres(hemis):
    tmp_path, calls = hemis
    assert _shared(tmp_path) == pytest.approx(2.5)
    # its own value is published for the other hemisphere
    own = json.loads((tmp_path / "S_barrier-ref-lh.json").read_text())
    assert own["reference"] == pytest.approx(2.6)
    assert calls == ["lh.seg.nii", "rh.seg.nii"]


def test_published_value_is_reused(hemis):
    tmp_path, calls = hemis
    _publish(tmp_path, 2.0)
    assert _shared(tmp_path) == pytest.approx(2.3)
    assert calls == ["lh.seg.nii"]


def test_value_for_another_label_map_is_ignored(hemis):
    tmp_path, calls = hemis
    _publish(tmp_path, 2.0, mtime=0.0)        # written for an older rh.seg
    assert _shared(tmp_path) == pytest.approx(2.5)
    assert calls == ["lh.seg.nii", "rh.seg.nii"]


def test_pending_marker_of_a_dead_process_is_not_waited_for(hemis):
    tmp_path, calls = hemis
    _publish(tmp_path, None, pid=2 ** 22 + 12345)
    assert _shared(tmp_path, timeout=60.0) == pytest.approx(2.5)
    assert calls == ["lh.seg.nii", "rh.seg.nii"]


def test_pending_marker_of_a_live_process_times_out(hemis):
    tmp_path, calls = hemis
    _publish(tmp_path, None, pid=os.getppid())
    assert _shared(tmp_path, timeout=1.0) == pytest.approx(2.5)


def test_missing_contralateral_hemisphere(hemis):
    tmp_path, _ = hemis
    assert _shared(tmp_path, other="nothere.nii") is None


def test_old_cat_surf_falls_back(hemis, monkeypatch):
    tmp_path, _ = hemis
    monkeypatch.delattr(se.cat_surf, "vol_pbt_barrier_reference")
    assert _shared(tmp_path) is None


def test_report_merge(tmp_path):
    from t1prep.t1prep import _merge_barrier_reference_qa

    report = tmp_path / "log_SUB01.json"
    report.write_text(json.dumps({"qualitymeasures": {"euler_lh": {"value": 2}}}))
    for hemi, ref in (("lh", 2.6), ("rh", 2.4)):
        (tmp_path / f"SUB01_barrier-ref-{hemi}.json").write_text(json.dumps({
            "hemi": hemi, "report_file": "log_SUB01.json", "reference": ref,
        }))
    _merge_barrier_reference_qa(str(tmp_path), "SUB01")

    qa = json.loads(report.read_text())["qualitymeasures"]
    assert qa["barrier_ref_lh"]["value"] == pytest.approx(2.6)
    assert qa["barrier_ref_rh"]["value"] == pytest.approx(2.4)
    assert qa["barrier_ref_shared"]["value"] == pytest.approx(2.5)
    assert qa["euler_lh"] == {"value": 2}
    assert not list(tmp_path.glob("*_barrier-ref-*.json"))


def test_report_merge_skips_pending_values(tmp_path):
    from t1prep.t1prep import _merge_barrier_reference_qa

    report = tmp_path / "log_SUB01.json"
    report.write_text(json.dumps({"qualitymeasures": {}}))
    (tmp_path / "SUB01_barrier-ref-lh.json").write_text(json.dumps({
        "hemi": "lh", "report_file": "log_SUB01.json", "reference": None,
    }))
    _merge_barrier_reference_qa(str(tmp_path), "SUB01")
    assert json.loads(report.read_text()) == {"qualitymeasures": {}}
    assert not list(tmp_path.glob("*_barrier-ref-*.json"))
