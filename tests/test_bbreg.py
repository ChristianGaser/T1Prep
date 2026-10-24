"""Tests for the files :mod:`t1prep.bbreg` writes for fMRIPrep to reuse.

The head-motion transforms are checked on a synthetic series whose first
volume is the middle one moved by a known amount: realigned onto the middle
volume, its points map back by that amount, and the ITK file stores the
reverse, as it does for the coregistration transform.
"""

import numpy as np
import pytest

# Allow running tests without installing the package (repo checkout / editable dev)
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import nibabel as nib  # noqa: E402

from t1prep import bbreg  # noqa: E402
from t1prep.transforms import save_affine_itk_txt, save_affines_itk_txt  # noqa: E402


def _itk_translations(path):
    """Translations (LPS) of every transform in an ITK text file."""
    return [
        np.array(line.split(":")[1].split(), dtype=float)[9:]
        for line in _Path(path).read_text().splitlines()
        if line.startswith("Parameters:")
    ]


def _series(tmp_path, shift_vox=1.0, voxel=2.0):
    """A smooth texture three times, the first one moved along x."""
    from scipy.ndimage import gaussian_filter, shift

    texture = gaussian_filter(np.random.default_rng(0).standard_normal((32, 32, 24)), 3.0)
    x, y, z = np.meshgrid(*(np.arange(n) - (n - 1) / 2 for n in texture.shape), indexing="ij")
    head = x**2 / 13**2 + y**2 / 13**2 + z**2 / 10**2 < 1
    base = (texture / texture.std() * 300 + 1000) * head
    moved = shift(base, (shift_vox, 0, 0), order=3)
    path = tmp_path / "sub-01_task-rest_run-1_bold.nii.gz"
    nib.Nifti1Image(
        np.stack([moved, base, base], axis=-1).astype(np.float32),
        np.diag([voxel, voxel, voxel, 1.0]),
    ).to_filename(path)
    return path, base


def test_fmriprep_paths_follow_the_bids_run(tmp_path):
    paths = bbreg.fmriprep_paths("sub-01_ses-pre_task-rest_run-2_bold.nii.gz", str(tmp_path))
    func = tmp_path / "sub-01" / "ses-pre" / "func"
    stem = "sub-01_ses-pre_task-rest_run-2"
    assert paths == {
        "hmc_boldref": str(func / f"{stem}_desc-hmc_boldref.nii.gz"),
        "coreg_boldref": str(func / f"{stem}_desc-coreg_boldref.nii.gz"),
        "hmc_xfm": str(func / f"{stem}_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt"),
        "coreg_xfm": str(func / f"{stem}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"),
    }
    # fMRIPrep's own reference names the same run
    assert bbreg.fmriprep_paths(
        "/x/sub-01_ses-pre_task-rest_run-2_desc-coreg_boldref.nii.gz", str(tmp_path)
    ) == paths
    with pytest.raises(ValueError):
        bbreg.fmriprep_paths("bold.nii.gz", str(tmp_path))


def test_one_affine_is_written_as_before(tmp_path):
    affine = np.eye(4)
    affine[:3, 3] = [1.0, -2.0, 3.0]
    save_affine_itk_txt(affine, str(tmp_path / "one.txt"))
    save_affines_itk_txt([affine], str(tmp_path / "list.txt"))
    assert (tmp_path / "one.txt").read_text() == (tmp_path / "list.txt").read_text()

    save_affines_itk_txt([np.eye(4), affine], str(tmp_path / "two.txt"))
    text = (tmp_path / "two.txt").read_text()
    assert text.count("#Transform ") == 2 and "#Transform 1\n" in text


def test_hmc_xfm_maps_reference_points_to_the_volume(tmp_path):
    # The volume is the reference moved 5 mm to the right: its points map
    # 5 mm to the left onto the reference, and the file holds the reverse
    to_reference = np.eye(4)
    to_reference[0, 3] = -5.0
    bbreg.save_hmc_xfm(np.stack([np.eye(4), to_reference]), str(tmp_path / "hmc.txt"))
    identity, moved = _itk_translations(tmp_path / "hmc.txt")
    np.testing.assert_allclose(identity, 0, atol=1e-9)
    np.testing.assert_allclose(moved, [-5.0, 0, 0], atol=1e-9)  # +5 mm right in LPS


def test_realign_onto_the_middle_volume(tmp_path):
    pytest.importorskip("cat_surf")
    path, base = _series(tmp_path, shift_vox=1.0, voxel=2.0)
    reference, transforms = bbreg.realign(str(path))
    assert reference.shape == base.shape
    assert np.corrcoef(reference.get_fdata().ravel(), base.ravel())[0, 1] > 0.99
    np.testing.assert_allclose(transforms[1], np.eye(4), atol=1e-6)
    # The first volume is 2 mm right of the reference, in mm, not voxels
    np.testing.assert_allclose(transforms[0][:3, 3], [-2.0, 0, 0], atol=0.15)


@pytest.mark.parametrize(
    "args, message",
    [
        ([], "give --out or --fmriprep-dir"),
        (["--out", "x.txt"], "a 4-D --bold needs --fmriprep-dir"),
        (["--fmriprep-dir", "deriv", "--hmc", "--reference"], "--hmc needs the 4-D BOLD series"),
    ],
)
def test_cli_refuses_inconsistent_outputs(tmp_path, monkeypatch, capsys, args, message):
    monkeypatch.chdir(tmp_path)
    path, _ = _series(tmp_path)
    if "--reference" in args:
        args = [a for a in args if a != "--reference"]
        img = nib.load(path)
        path = tmp_path / "sub-01_task-rest_run-1_desc-coreg_boldref.nii.gz"
        nib.Nifti1Image(img.get_fdata()[..., 1], img.affine).to_filename(path)
    assert bbreg.main(["--bold", str(path), "--lh-white", "lh.gii", *args]) == 2
    assert message in capsys.readouterr().err
