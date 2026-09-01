"""Integrity of the shipped surface atlases.

They are data, not code, so nothing else would notice a truncated file or a
parcellation quietly taken from the wrong space — it would surface only as
subtly wrong parcel statistics in someone's results.

Each atlas is taken from the space it was defined in: DK40 and Destrieux from
fsaverage, HCP-MMP1 and Schaefer2018 from their own fs_LR 32k releases, which
this mesh is vertex-for-vertex.
"""

import re

import nibabel as nib
import numpy as np
import pytest

from t1prep.utils import DATA_PATH_T1PREP

ATLAS_DIR = DATA_PATH_T1PREP / "atlases_surfaces_32k"
TEMPLATE_DIR = DATA_PATH_T1PREP / "templates_surfaces_32k"

#: Atlases taken from their native fs_LR release, with the parcel count each
#: publication defines per hemisphere.
FSLR_NATIVE = [
    ("aparc_HCP_MMP1", 180),
    ("Schaefer2018_100Parcels_17Networks_order", 50),
    ("Schaefer2018_200Parcels_17Networks_order", 100),
    ("Schaefer2018_400Parcels_17Networks_order", 200),
    ("Schaefer2018_600Parcels_17Networks_order", 300),
]

#: HCP's cortical grayordinate counts — the vertices left once fs_LR 32k's
#: medial wall is removed.  The HCP-MMP1 dlabel covers exactly these, so the
#: numbers are a fingerprint of the native release.
HCP_GRAYORDINATES = {"lh": 29696, "rh": 29716}


def _annots():
    return sorted(p.name for p in ATLAS_DIR.glob("?h.*.annot"))


def _read(name):
    labels, ctab, names = nib.freesurfer.read_annot(str(ATLAS_DIR / name))
    return labels, ctab, [n.decode() if isinstance(n, bytes) else n for n in names]


def _area(name):
    """Strip hemisphere and suffix decoration: ``L_V1_ROI`` -> ``V1``."""
    return re.sub(r"_ROI$", "", re.sub(r"^[LR]_", "", name))


@pytest.fixture(scope="module")
def mesh_size():
    surface = nib.load(str(TEMPLATE_DIR / "lh.sphere.freesurfer.gii"))
    return surface.darrays[0].data.shape[0]


def test_atlases_are_present():
    assert _annots(), f"no annot files under {ATLAS_DIR}"


@pytest.mark.parametrize("name", _annots())
def test_annot_is_well_formed(name, mesh_size):
    labels, ctab, names = _read(name)

    assert labels.shape == (mesh_size,), "one label per template vertex"
    assert len(names) == len(ctab), "every colour-table row needs a name"
    # The collection mixes the two FreeSurfer conventions for unassigned
    # vertices: some files reserve colour-table index 0 for the medial wall,
    # others leave those vertices out of the table so nibabel reports -1.
    # Both are valid; only an index past the end of the table is not.
    assert labels.min() >= -1, "-1 is the only admissible out-of-table value"
    assert labels.max() < len(ctab), "labels index the colour table"


@pytest.mark.parametrize("name, per_hemisphere", FSLR_NATIVE)
def test_parcel_count_matches_the_publication(name, per_hemisphere):
    for hemi in ("lh", "rh"):
        labels, _, _ = _read(f"{hemi}.{name}.annot")
        assert len(np.unique(labels[labels > 0])) == per_hemisphere


@pytest.mark.parametrize("hemi, expected", sorted(HCP_GRAYORDINATES.items()))
def test_hcp_mmp1_covers_exactly_the_hcp_grayordinates(hemi, expected):
    """A projection through fsaverage would not land on these counts."""
    labels, _, _ = _read(f"{hemi}.aparc_HCP_MMP1.annot")
    assert int((labels > 0).sum()) == expected


def test_hcp_mmp1_respects_left_right_correspondence():
    """fs_LR pairs vertex i across hemispheres; fsaverage does not.

    Measured 81.6 % for the native fs_LR release against 75.8 % for the
    fsaverage projection T1Prep used to ship, so the threshold separates the
    two.  It is well below 100 % because areas genuinely differ in extent
    between hemispheres — the mesh is symmetric, the anatomy is not.
    """
    left, _, left_names = _read("lh.aparc_HCP_MMP1.annot")
    right, _, right_names = _read("rh.aparc_HCP_MMP1.annot")

    both = (left > 0) & (right > 0)
    homologous = np.mean(
        [_area(left_names[a]) == _area(right_names[b])
         for a, b in zip(left[both], right[both])]
    )
    assert homologous > 0.79, f"only {homologous:.3f} — is this the fsaverage projection?"
