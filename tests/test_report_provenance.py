"""Tests of the software provenance in the T1Prep JSON report."""

import json
import sys as _sys
from pathlib import Path as _Path

_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from t1prep import report  # noqa: E402
from t1prep.utils import get_filenames  # noqa: E402


def test_package_versions_cover_numerical_dependencies():
    v = report._package_versions()
    assert set(v) == set(report.PROVENANCE_PACKAGES)
    assert v["numpy"] and v["torch"]


def test_model_fingerprint_changes_with_the_weights(tmp_path, monkeypatch):
    import t1prep._models as models

    for name in models.MODEL_FILES:
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(models, "MODEL_DIR", tmp_path)
    report.model_provenance.cache_clear()
    a = report.model_provenance()
    assert a["complete"] and a["dir"] == str(tmp_path)
    assert len(a["sha256"]) == len(models.MODEL_FILES)

    (tmp_path / models.MODEL_FILES[0]).write_bytes(b"other weights")
    report.model_provenance.cache_clear()
    b = report.model_provenance()
    assert b["fingerprint"] != a["fingerprint"]

    (tmp_path / models.MODEL_FILES[1]).unlink()
    report.model_provenance.cache_clear()
    assert report.model_provenance()["complete"] is False
    report.model_provenance.cache_clear()


def test_report_contains_provenance(tmp_path, monkeypatch):
    import t1prep._models as models

    for name in models.MODEL_FILES:
        (tmp_path / name).write_bytes(b"w")
    monkeypatch.setattr(models, "MODEL_DIR", tmp_path)
    report.model_provenance.cache_clear()
    names = get_filenames(False, "sub01", "", "", "", "")
    (tmp_path / names["Report_file"]).write_text(json.dumps({"vol_tiv": {"value": 1500}}))
    report.write_t1prep_report(str(tmp_path), "sub01", False, str(tmp_path / "sub01.nii"))
    sw = json.loads((tmp_path / names["Report_file"]).read_text())["software"]
    assert sw["versions"]["numpy"]
    assert sw["models"]["complete"] and len(sw["models"]["fingerprint"]) == 64
    report.model_provenance.cache_clear()
