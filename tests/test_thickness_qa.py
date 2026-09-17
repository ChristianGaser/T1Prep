"""Tests for :mod:`t1prep.thickness_qa` and its report merge."""

import json

import numpy as np
import pytest

# Allow running tests without installing the package (repo checkout / editable dev)
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from t1prep.thickness_qa import HIGH_FACTOR, LOWER_CUT, thickness_shape


def _cortex(n=20000, seed=0):
    """Symmetric thickness around 2.5 mm, plus a medial wall near zero."""
    rng = np.random.default_rng(seed)
    t = rng.normal(2.5, 0.4, n)
    t[: n // 10] = rng.uniform(0.0, 0.3, n // 10)     # medial wall
    return t


def test_matches_scipy_skewness():
    stats = pytest.importorskip("scipy.stats")
    t = _cortex()
    got = thickness_shape(t)["upper_skewness"]
    assert got == pytest.approx(stats.skew(t[t >= LOWER_CUT]))


def test_medial_wall_does_not_drive_the_skewness():
    t = _cortex()
    # over all vertices the near-zero values make it strongly negative ...
    d = t - t.mean()
    assert np.mean(d ** 3) / np.mean(d ** 2) ** 1.5 < -1.0
    # ... without them the symmetric cortex is left
    assert abs(thickness_shape(t)["upper_skewness"]) < 0.1


def test_thick_patch_raises_skewness_and_high_fraction():
    t = _cortex()
    base = thickness_shape(t)
    t[-400:] += 2.5                                   # 2% implausibly thick
    shifted = thickness_shape(t)
    assert shifted["upper_skewness"] > base["upper_skewness"] + 0.3
    assert shifted["high_fraction"] > base["high_fraction"] + 0.015


def test_high_fraction_is_relative_to_the_median():
    t = np.array([1.0, 2.0, 2.0, 2.0, 2.0 * HIGH_FACTOR + 0.1])
    out = thickness_shape(t)
    assert out["median"] == pytest.approx(2.0)
    assert out["high_fraction"] == pytest.approx(0.2)


def test_non_finite_and_degenerate_input():
    out = thickness_shape([np.nan, np.inf])
    assert out == {"n_vertices": 0, "median": None,
                   "upper_skewness": None, "high_fraction": None}
    out = thickness_shape([0.1, 0.2, 2.0, np.nan])
    assert out["n_vertices"] == 3
    assert out["upper_skewness"] is None              # one value above the cut
    assert thickness_shape([2.0, 2.0, 2.0])["upper_skewness"] == 0.0


def _write_sidecar(path, hemi, **extra):
    data = {"hemi": hemi, "report_file": "log_SUB01.json",
            "sulci_sigma_factor": None}
    data.update(extra)
    path.write_text(json.dumps(data))


def test_report_merge(tmp_path):
    from t1prep.t1prep import _merge_glued_sulci_qa

    report = tmp_path / "log_SUB01.json"
    report.write_text(json.dumps({"qualitymeasures": {}}))
    _write_sidecar(tmp_path / "SUB01_glued-lh.json", "lh",
                   glued_fraction=0.001, thickness_median=2.4,
                   thickness_upper_skewness=0.45, thickness_high_fraction=0.02)
    _write_sidecar(tmp_path / "SUB01_glued-rh.json", "rh",
                   glued_fraction=0.002, thickness_median=2.3,
                   thickness_upper_skewness=0.15, thickness_high_fraction=0.005)
    _merge_glued_sulci_qa(str(tmp_path), "SUB01")

    qa = json.loads(report.read_text())["qualitymeasures"]
    assert qa["thickness_skew_lh"]["value"] == pytest.approx(0.45)
    assert qa["thickness_skew_rh"]["value"] == pytest.approx(0.15)
    assert qa["thickness_high_lh"]["value"] == pytest.approx(2.0)
    assert qa["thickness_high_rh"]["value"] == pytest.approx(0.5)
    assert qa["thickness_skew_asym"]["value"] == pytest.approx(0.30)
    assert qa["glued_lh"]["value"] == pytest.approx(0.1)
    assert not list(tmp_path.glob("*_glued-*.json"))


def test_report_merge_with_partial_sidecars(tmp_path):
    """A sidecar without the thickness measure (or without glued sulci) only
    contributes what it has, and no asymmetry is reported for one side."""
    from t1prep.t1prep import _merge_glued_sulci_qa

    report = tmp_path / "log_SUB01.json"
    report.write_text(json.dumps({"qualitymeasures": {}}))
    _write_sidecar(tmp_path / "SUB01_glued-lh.json", "lh", glued_fraction=0.001)
    _write_sidecar(tmp_path / "SUB01_glued-rh.json", "rh",
                   thickness_upper_skewness=0.2, thickness_high_fraction=0.01)
    _merge_glued_sulci_qa(str(tmp_path), "SUB01")

    qa = json.loads(report.read_text())["qualitymeasures"]
    assert set(qa) == {"glued_lh", "thickness_skew_rh", "thickness_high_rh"}
