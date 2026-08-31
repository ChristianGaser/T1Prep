"""Tests for :mod:`t1prep.bids_derivatives` and the BBR transform direction."""

import json
import os

import numpy as np
import pytest

from t1prep.bids_derivatives import (
    derivatives_root,
    write_dataset_description,
    write_sidecar,
)
from t1prep.bbreg import save_boldref_to_t1w_xfm
from t1prep.itk_transforms import load_affine_itk_txt, save_affine_itk_txt


# ---------------------------------------------------------------------------
# Dataset description
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "layout, depth",
    [
        (("sub-01", "anat"), 2),
        (("sub-01", "ses-1", "anat"), 3),
        (("sub-01",), 1),
        (("derivatives",), 0),
    ],
)
def test_dataset_root_is_found_above_the_bids_tree(tmp_path, layout, depth):
    out = tmp_path.joinpath(*layout)
    out.mkdir(parents=True)
    expected = out
    for _ in range(depth):
        expected = expected.parent
    assert derivatives_root(str(out)) == str(expected)


def test_dataset_description_is_written_at_the_root(tmp_path):
    out = tmp_path / "sub-01" / "anat"
    out.mkdir(parents=True)
    path = write_dataset_description(str(out))

    assert path == str(tmp_path / "dataset_description.json")
    content = json.loads(open(path).read())
    # PyBIDS refuses to index a derivatives tree without these two.
    assert content["DatasetType"] == "derivative"
    assert content["GeneratedBy"][0]["Name"] == "T1Prep"


def test_existing_dataset_description_is_kept(tmp_path):
    """A second subject must not clobber the first one's description."""
    (tmp_path / "dataset_description.json").write_text('{"Name": "mine"}')
    write_dataset_description(str(tmp_path))
    assert json.loads((tmp_path / "dataset_description.json").read_text())["Name"] == "mine"

    write_dataset_description(str(tmp_path), overwrite=True)
    assert json.loads((tmp_path / "dataset_description.json").read_text())["Name"] != "mine"


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, expected",
    [
        ("sub-01_desc-brain_mask.nii.gz", "sub-01_desc-brain_mask.json"),
        ("sub-01_desc-preproc_T1w.nii", "sub-01_desc-preproc_T1w.json"),
        ("sub-01_hemi-L_desc-cortex_mask.label.gii", "sub-01_hemi-L_desc-cortex_mask.json"),
        ("sub-01_hemi-L_sphere.surf.gii", "sub-01_hemi-L_sphere.json"),
    ],
)
def test_sidecar_replaces_every_extension(tmp_path, name, expected):
    path = write_sidecar(str(tmp_path / name), Type="ROI")
    assert os.path.basename(path) == expected


def test_sidecar_drops_absent_fields(tmp_path):
    path = write_sidecar(str(tmp_path / "sub-01_desc-ribbon_mask.nii.gz"),
                         Type="ROI", RawSources=None)
    assert json.loads(open(path).read()) == {"Type": "ROI"}


def test_sidecar_ignores_an_empty_path():
    assert write_sidecar("") is None


# ---------------------------------------------------------------------------
# BBR transform direction
# ---------------------------------------------------------------------------
def test_boldref_xfm_stores_the_inverse_mapping(tmp_path):
    """ITK stores output-to-input, the reverse of the BIDS from/to entities.

    Verified against fMRIPrep's own transform for sub-01: writing the BBR
    matrix uninverted disagreed with it by twice the translation.
    """
    matrix = np.eye(4)
    matrix[:3, 3] = [0.5, 1.75, -0.25]        # BOLD point -> T1w point
    out = tmp_path / "xfm.txt"
    save_boldref_to_t1w_xfm(matrix, str(out))

    stored = load_affine_itk_txt(str(out))
    np.testing.assert_allclose(stored, np.linalg.inv(matrix), atol=1e-9)
    np.testing.assert_allclose(stored[:3, 3], [-0.5, -1.75, 0.25], atol=1e-9)


def test_itk_text_format_matches_ants(tmp_path):
    out = tmp_path / "identity.txt"
    save_affine_itk_txt(np.eye(4), str(out))
    lines = out.read_text().splitlines()
    assert lines[0] == "#Insight Transform File V1.0"
    assert lines[2] == "Transform: AffineTransform_float_3_3"
    assert lines[3].split(":")[1].split() == ["1", "0", "0", "0", "1", "0", "0", "0", "1",
                                              "0", "0", "0"]
    assert lines[4] == "FixedParameters: 0 0 0"


def test_itk_text_round_trip_handles_a_rotation_centre(tmp_path):
    """``FixedParameters`` is folded into the translation on read."""
    out = tmp_path / "centred.txt"
    out.write_text(
        "#Insight Transform File V1.0\n#Transform 0\n"
        "Transform: AffineTransform_float_3_3\n"
        "Parameters: 0 -1 0 1 0 0 0 0 1 0 0 0\n"
        "FixedParameters: 10 0 0\n"
    )
    got = load_affine_itk_txt(str(out))
    # LPS rotation about (10, 0, 0) maps that point to itself.
    np.testing.assert_allclose(got[:3, :3] @ [-10, 0, 0] + got[:3, 3], [-10, 0, 0], atol=1e-9)
