"""Tests for the hemisphere partitioning in :mod:`t1prep._segment_utils`.

``get_partition`` turns the PVE label map into the two ``?h.seg.*`` volumes
that PBT measures cortical thickness on, so anything it writes into the
cortical ribbon lands directly in the thickness estimate.  Its fills are all
driven by a warped atlas, and the tests below pin the invariant that keeps
them honest: a fill may cover the structure it was seeded from, but it must
never relabel cortex or archicortex on its way there.

The regressions these guard against all showed up in the medial temporal
lobe, where the temporal horn, the hippocampus, the ventral diencephalon and
the brainstem sit within a few millimetres of each other and of the ribbon.
"""

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

# Allow running tests without installing the package (repo checkout / editable dev)
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from t1prep._segment_utils import get_partition, _resolve_template_file

SHAPE = (72, 96, 72)
VX = 0.5


def _table():
    return pd.read_csv(_resolve_template_file("IBSR", ".csv"), sep=";")


def _phantom(with_brainstem=True):
    """Build a two-hemisphere label phantom on the pipeline's 0.5 mm grid.

    The geometry is schematic but the adjacencies that matter are real: the
    temporal horn sits directly on the hippocampus with no white matter
    between them, and the brainstem sits within a few millimetres of the
    inferior ribbon.  Those two contacts are what the fills used to cross.
    """
    tbl = _table()
    reg = dict(zip(tbl.ROIabbr, tbl.ROIid))
    lab = np.zeros(SHAPE, np.float32)

    for sd, pre in ((slice(6, 36), "l"), (slice(36, 66), "r")):
        lab[sd, 18:78, 12:60] = reg[pre + "CbrWM"]
        # Cortical ribbon wrapped around the white matter core, deliberately
        # at the thick end of the physiological range.  The final mask
        # cleanup in ``get_partition`` opens the tissue mask with three
        # iterations, which erases whatever is left thinner than ~3.5 mm; a
        # ribbon that survives that on its own keeps these tests measuring
        # the fills rather than the cleanup.
        lab[sd, 10:18, 12:60] = reg[pre + "CbrGM"]
        lab[sd, 78:86, 12:60] = reg[pre + "CbrGM"]
        lab[sd, 18:78, 4:12] = reg[pre + "CbrGM"]
        lab[sd, 18:78, 60:68] = reg[pre + "CbrGM"]
    for sd, pre in ((slice(12, 30), "l"), (slice(42, 60), "r")):
        lab[sd, 30:46, 34:50] = reg[pre + "ThaPro"]
        lab[sd, 50:62, 40:52] = reg[pre + "LatVen"]
        lab[sd, 50:62, 22:30] = reg[pre + "InfLatVen"]
        lab[sd, 50:62, 14:22] = reg[pre + "Hip"]
        lab[sd, 30:46, 26:34] = reg[pre + "VenDC"]
    # Buried in the white matter 1 mm short of the inferior ribbon: inside
    # the reach of a 5-step 26-connected dilation, which is what used to eat
    # the parahippocampal cortex wrapped around the real midbrain, but with
    # enough tissue beneath it that a correct cut leaves the ribbon intact.
    # Replacing the block with white matter rather than dropping it keeps the
    # two runs identical everywhere else, so a differenced result isolates
    # what the brainstem exclusion alone did.
    lab[30:42, 44:64, 14:30] = reg["bBst" if with_brainstem else "lCbrWM"]

    # PVE labels implied by the atlas' own tissue table, so the "subject" is
    # perfectly registered to the atlas unless a test displaces one of them.
    p0 = np.zeros(SHAPE, np.float32)
    for _, r in tbl.iterrows():
        m = lab == r.ROIid
        if r.Vwm and not r.Vgm:
            p0[m] = 3.0
        elif r.Vgm:
            p0[m] = 2.0
        elif r.Vcsf:
            p0[m] = 1.0
    return lab, p0, reg


def _nifti(data):
    affine = np.diag([VX, VX, VX, 1.0])
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.header.set_zooms((VX, VX, VX))
    return img


def _partition(shift_voxels=0, with_brainstem=True):
    """Run ``get_partition``, optionally displacing the atlas from the subject.

    A non-zero ``shift_voxels`` mimics nonlinear registration error, which is
    what turns an over-reaching fill from harmless into destructive.
    """
    lab, p0, reg = _phantom(with_brainstem)
    atlas = lab if not shift_voxels else np.roll(lab, shift_voxels, axis=1)
    lh, rh = get_partition(_nifti(p0), _nifti(atlas))
    return np.maximum(lh, rh), lab, reg


def _mask(lab, reg, *abbrs):
    return np.isin(lab, [reg[a] for a in abbrs])


@pytest.fixture(scope="module")
def aligned():
    return _partition()


def test_hippocampus_is_not_filled_with_wm(aligned):
    """The temporal horn fill must stop at the hippocampus.

    ``ventricle_fill`` propagates through everything that is not white
    matter, which is a wall all around the ventricular system except here:
    the fill used to flow out of the temporal horn and bury a third of the
    hippocampus in WM on every subject, however good the registration.
    """
    seg, lab, reg = aligned
    hip = _mask(lab, reg, "lHip", "rHip")
    blob = (seg[hip] > 2.5)
    assert not blob.any(), (
        f"{blob.sum()} of {hip.sum()} hippocampus voxels "
        f"({blob.mean():.1%}) came out as white matter"
    )


def test_brainstem_cut_spares_cortex_and_hippocampus(aligned):
    """Excluding the brainstem must not carve away nearby grey matter.

    ``exclude`` is applied last and overrides every other label, so anything
    it over-reaches into becomes CSF and truncates the ribbon.

    Measured as a difference against the same phantom with the brainstem
    block replaced by white matter.  The final mask cleanup in
    ``get_partition`` also trims the odd voxel off the rim of the ribbon, and
    that is a separate mechanism -- differencing cancels it out and leaves
    only what the brainstem exclusion is responsible for.
    """
    seg, lab, reg = aligned
    reference, _, _ = _partition(with_brainstem=False)
    spared = _mask(lab, reg, "lCbrGM", "rCbrGM", "lHip", "rHip")
    cut_by_brainstem = (seg < 1.5) & ~(reference < 1.5) & spared
    assert not cut_by_brainstem.any(), (
        f"{cut_by_brainstem.sum()} grey-matter voxels lost to the brainstem cut"
    )


def test_cortex_is_never_relabelled_as_wm(aligned):
    """No fill may push the GM/WM boundary out into the ribbon."""
    seg, lab, reg = aligned
    ctx = _mask(lab, reg, "lCbrGM", "rCbrGM")
    pushed = (seg[ctx] > 2.5)
    assert not pushed.any(), (
        f"{pushed.sum()} cortex voxels relabelled as white matter"
    )


def test_subcortical_structures_never_read_as_grey_matter(aligned):
    """The fills still have to do their job.

    Anything assigned to a hemisphere that is not cortex has to come out as
    white matter (filled) or CSF (cut away).  A subcortical structure left at
    the GM level is somewhere the white surface will dip into, which is the
    mirror image of the blob problem: instead of WM appearing inside grey
    matter, grey matter is left sitting inside the white compartment.

    The ventral diencephalon reached neither the fill list nor the exclusion
    list, so whether it passed came down to a neighbouring fill happening to
    swallow it.
    """
    seg, lab, reg = aligned
    left_at_gm = {}
    for abbr in ("lThaPro", "rThaPro", "lVenDC", "rVenDC"):
        values = seg[_mask(lab, reg, abbr)]
        count = int(((values > 1.5) & (values < 2.5)).sum())
        if count:
            left_at_gm[abbr] = count
    assert not left_at_gm, f"left at GM level: {left_at_gm}"


@pytest.mark.parametrize("shift_voxels", [4, 6])
def test_hippocampus_survives_misregistration(shift_voxels):
    """A few millimetres of atlas error must not flood the hippocampus.

    The medial temporal lobe is where nonlinear registration is worst, so the
    fills have to degrade gracefully rather than cross the structure.
    """
    seg, lab, reg = _partition(shift_voxels)
    hip = _mask(lab, reg, "lHip", "rHip")
    blob = (seg[hip] > 2.5).mean()
    assert blob < 0.25, (
        f"{blob:.1%} of the hippocampus came out as white matter at "
        f"{shift_voxels * VX:.1f} mm of atlas error"
    )
