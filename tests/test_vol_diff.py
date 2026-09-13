"""Tests for ``scripts/CAT_VolDiff``, the volume difference tool."""

import importlib.machinery
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "CAT_VolDiff"


def _load_script():
    loader = importlib.machinery.SourceFileLoader("cat_vol_diff", str(_SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


vol_diff = _load_script()

needs_vol_calc = pytest.mark.skipif(
    not hasattr(pytest.importorskip("cat_surf.cli"), "vol_calc"),
    reason="installed cat-surf has no vol_calc binding",
)


def _save(path, data, affine=None):
    affine = np.diag([1.5, 1.5, 1.5, 1.0]) if affine is None else affine
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))
    return str(path)


def test_output_name_prefixes_the_file_name(tmp_path):
    src = str(tmp_path / "tp2.nii.gz")
    assert vol_diff.output_name(src) == str(tmp_path / "diff_tp2.nii.gz")
    assert vol_diff.output_name(src, rel=True) == str(
        tmp_path / "diffrel_tp2.nii.gz"
    )


def test_expression_matches_cat_stat_diff():
    assert vol_diff.diff_expression() == "i2-i1"
    assert vol_diff.diff_expression(rel=True).startswith("200*(i2-i1)./(i1+i2+")
    scaled = vol_diff.diff_expression(scale1=0.5, scale2=2.0)
    assert scaled == "(2.0*i2)-(0.5*i1)"


def test_global_mean_drops_the_background(tmp_path):
    data = np.zeros((10, 10, 10))
    data[2:8, 2:8, 2:8] = 100.0
    data[0, 0, 0] = np.nan
    assert vol_diff.global_mean(_save(tmp_path / "g.nii", data)) == pytest.approx(
        100.0
    )


@needs_vol_calc
def test_diff_subject_writes_every_difference_to_the_first(tmp_path):
    rng = np.random.default_rng(0)
    vols = [rng.random((6, 7, 8)) + 1.0 for _ in range(3)]
    files = [_save(tmp_path / f"tp{k}.nii", v) for k, v in enumerate(vols, 1)]

    written = vol_diff.diff_subject(files, verbose=False)

    assert [Path(w).name for w in written] == ["diff_tp2.nii", "diff_tp3.nii"]
    for out, vol in zip(written, vols[1:]):
        img = nib.load(out)
        assert img.get_data_dtype() == np.float32
        np.testing.assert_allclose(img.get_fdata(), vol - vols[0], atol=1e-5)

    rel = vol_diff.diff_subject(files[:2], rel=True, verbose=False)[0]
    expected = 200 * (vols[1] - vols[0]) / (vols[0] + vols[1])
    np.testing.assert_allclose(nib.load(rel).get_fdata(), expected, rtol=1e-5)


@needs_vol_calc
def test_glob_removes_a_global_intensity_factor(tmp_path):
    base = np.zeros((8, 8, 8))
    base[1:7, 1:7, 1:7] = np.random.default_rng(1).random((6, 6, 6)) + 1.0
    files = [
        _save(tmp_path / "a.nii", base),
        _save(tmp_path / "b.nii", 3.0 * base),
    ]

    out = vol_diff.diff_subject(files, glob=True, verbose=False)[0]

    np.testing.assert_allclose(nib.load(out).get_fdata(), 0.0, atol=1e-4)


@needs_vol_calc
def test_main_handles_subjects_and_refuses_single_images(tmp_path, capsys):
    vol = np.ones((4, 4, 4))
    s1 = [_save(tmp_path / f"s1_tp{k}.nii", k * vol) for k in (1, 2)]
    s2 = [_save(tmp_path / f"s2_tp{k}.nii", k * vol) for k in (1, 2)]

    assert vol_diff.main(["-q", "-s", *s1, "-s", *s2]) == 0
    assert (tmp_path / "diff_s1_tp2.nii").exists()
    assert (tmp_path / "diff_s2_tp2.nii").exists()

    with pytest.raises(SystemExit):
        vol_diff.main([s1[0]])


@needs_vol_calc
def test_differing_grids_are_an_error(tmp_path, capsys):
    a = _save(tmp_path / "a.nii", np.ones((4, 4, 4)))
    b = _save(tmp_path / "b.nii", np.ones((4, 4, 5)))

    assert vol_diff.main(["-q", a, b]) == 1
    assert "does not share the grid" in capsys.readouterr().err
