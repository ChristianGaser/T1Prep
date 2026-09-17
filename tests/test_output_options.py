"""Tests for :class:`t1prep.segment.OutputOptions`."""

import argparse

import pytest

# Allow running tests without installing the package (repo checkout / editable dev)
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from t1prep.segment import OutputOptions


def _args(**overrides):
    """A namespace as ``parse_arguments`` returns it, with no outputs enabled."""
    values = dict(
        input="/data/sub-01_T1w.nii.gz",
        mri_dir="out/mri",
        label_dir="out/label",
        report_dir="out/report",
        atlas="",
        surf=False,
        csf=False,
        mwp=False,
        wp=False,
        p=False,
        rp=False,
        lesions=False,
        save_h5=False,
        save_fmriprep=False,
        bids=False,
        gz=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/data/sub-01_T1w.nii.gz", "sub-01_T1w"),
        ("/data/sub-01_T1w.nii", "sub-01_T1w"),
        ("relative/scan.nii", "scan"),
    ],
)
def test_out_name_drops_directory_and_extension(path, expected):
    assert OutputOptions.from_args(_args(input=path)).out_name == expected


def test_extension_follows_gz():
    assert OutputOptions.from_args(_args()).ext == "nii"
    assert OutputOptions.from_args(_args(gz=True)).ext == "nii.gz"


@pytest.mark.parametrize(
    "atlas, expected",
    [
        ("", None),
        ("Neuromorphometrics", ("Neuromorphometrics_volumes",)),
        (
            "Neuromorphometrics,LPBA40",
            ("Neuromorphometrics_volumes", "LPBA40_volumes"),
        ),
        # The quoted list form T1Prep_defaults.txt uses.
        (
            "'neuromorphometrics', 'cobra'",
            ("neuromorphometrics_volumes", "cobra_volumes"),
        ),
    ],
)
def test_atlas_list(atlas, expected):
    assert OutputOptions.from_args(_args(atlas=atlas)).atlas_list == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"surf": True},
        {"mwp": True},
        {"wp": True},
        {"save_fmriprep": True},
        {"atlas": "Neuromorphometrics"},
    ],
)
def test_outputs_that_need_the_warp(overrides):
    assert OutputOptions.from_args(_args(**overrides)).needs_warp


@pytest.mark.parametrize(
    "overrides",
    [{}, {"p": True}, {"rp": True}, {"csf": True}, {"lesions": True}, {"bids": True}],
)
def test_outputs_that_do_not(overrides):
    assert not OutputOptions.from_args(_args(**overrides)).needs_warp
