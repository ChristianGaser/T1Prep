"""Tests for the hemisphere partitioning in :mod:`t1prep._partition`.

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

from t1prep._atlas import resolve_template_file
from t1prep._partition import compute_euler_number, get_partition

SHAPE = (72, 96, 72)
VX = 0.5


def _table():
    return pd.read_csv(resolve_template_file("IBSR", ".csv"), sep=";")


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


def _nifti(data, dtype=np.float32):
    affine = np.diag([VX, VX, VX, 1.0])
    img = nib.Nifti1Image(data.astype(dtype), affine)
    img.header.set_zooms((VX, VX, VX))
    return img


#: IBSR -> Neuromorphometrics, for the labels the guard looks at.  Any
#: Neuromorphometrics id from 100 up is a cortical parcel.
_TO_NMM = {
    "lCbrGM": 101, "rCbrGM": 100,
    "lHip": 48, "rHip": 47,
    "lAmy": 32, "rAmy": 31,
    "lCbrWM": 45, "rCbrWM": 44,
    "lLatVen": 52, "rLatVen": 51,
}


def _guard_atlas(lab, reg):
    """The Neuromorphometrics labelling of an IBSR phantom."""
    nmm = np.zeros(lab.shape, np.uint8)
    for abbr, nid in _TO_NMM.items():
        nmm[lab == reg[abbr]] = nid
    return nmm


def _partition(shift_voxels=0, with_brainstem=True, guard=True):
    """Run ``get_partition``, optionally displacing the atlas from the subject.

    A non-zero ``shift_voxels`` mimics nonlinear registration error, which is
    what turns an over-reaching fill from harmless into destructive.  Both
    atlases move together, as they do when one warp places them.
    """
    lab, p0, reg = _phantom(with_brainstem)
    atlas = lab if not shift_voxels else np.roll(lab, shift_voxels, axis=1)
    return _run(p0, atlas, _guard_atlas(atlas, reg) if guard else None), lab, reg


def _run(p0, atlas, nmm=None):
    guard = None if nmm is None else _nifti(nmm, np.uint8)
    lh, rh = get_partition(_nifti(p0), _nifti(atlas), guard)
    return np.maximum(lh, rh)


def _mask(lab, reg, *abbrs):
    return np.isin(lab, [reg[a] for a in abbrs])


@pytest.fixture(scope="module", params=[True, False], ids=["guard", "noguard"])
def aligned(request):
    return (*_partition(guard=request.param), request.param)


def test_hippocampus_is_not_filled_with_wm(aligned):
    """The temporal horn fill must stop at the hippocampus.

    ``ventricle_fill`` propagates through everything that is not white
    matter, which is a wall all around the ventricular system except here:
    the fill used to flow out of the temporal horn and bury a third of the
    hippocampus in WM on every subject, however good the registration.
    """
    seg, lab, reg, guard = aligned
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
    seg, lab, reg, guard = aligned
    reference, _, _ = _partition(with_brainstem=False, guard=guard)
    spared = _mask(lab, reg, "lCbrGM", "rCbrGM", "lHip", "rHip")
    cut_by_brainstem = (seg < 1.5) & ~(reference < 1.5) & spared
    assert not cut_by_brainstem.any(), (
        f"{cut_by_brainstem.sum()} grey-matter voxels lost to the brainstem cut"
    )


def test_cortex_is_never_relabelled_as_wm(aligned):
    """No fill may push the GM/WM boundary out into the ribbon."""
    seg, lab, reg, guard = aligned
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
    seg, lab, reg, guard = aligned
    left_at_gm = {}
    for abbr in ("lThaPro", "rThaPro", "lVenDC", "rVenDC"):
        values = seg[_mask(lab, reg, abbr)]
        count = int(((values > 1.5) & (values < 2.5)).sum())
        if count:
            left_at_gm[abbr] = count
    assert not left_at_gm, f"left at GM level: {left_at_gm}"


@pytest.mark.parametrize("guard", [True, False], ids=["guard", "noguard"])
@pytest.mark.parametrize("shift_voxels", [4, 6])
def test_hippocampus_survives_misregistration(shift_voxels, guard):
    """A few millimetres of atlas error must not flood the hippocampus.

    The medial temporal lobe is where nonlinear registration is worst, so the
    fills have to degrade gracefully rather than cross the structure.
    """
    seg, lab, reg = _partition(shift_voxels, guard=guard)
    hip = _mask(lab, reg, "lHip", "rHip")
    blob = (seg[hip] > 2.5).mean()
    assert blob < 0.25, (
        f"{blob:.1%} of the hippocampus came out as white matter at "
        f"{shift_voxels * VX:.1f} mm of atlas error"
    )


def _cortex_filled(seg, lab, reg):
    return int((seg[_mask(lab, reg, "lCbrGM", "rCbrGM")] > 2.5).sum())


def test_ventricle_fill_stops_at_cortex_behind_thin_white_matter():
    """The ventricle fill must not cross a partial-volume gap into cortex.

    Behind the atrium and under the frontal horns the white matter between
    ventricle and sulcus is thinner than the blur, so "not white matter" has
    a hole in it.  The fill used to run through and bury the lingual gyrus,
    precuneus and subcallosal area along a straight 5 mm edge.  Here a
    grey-matter-valued channel links the roof of the left ventricle to the
    superior ribbon, and the IBSR cortex label -- which the ventricle fill
    ignores on purpose -- offers no protection.
    """
    lab, p0, reg = _phantom()
    # 4 voxels wide and 8 long: from the ventricle roof (z = 52) up to the
    # ribbon (z = 60), within the 10-step reach of the fill.
    p0[18:24, 54:58, 52:60] = 2.0

    unguarded = _run(p0, lab)
    assert _cortex_filled(unguarded, lab, reg), (
        "phantom no longer exercises the leak: nothing filled without guard"
    )
    guarded = _run(p0, lab, _guard_atlas(lab, reg))
    assert not _cortex_filled(guarded, lab, reg), (
        f"{_cortex_filled(guarded, lab, reg)} cortex voxels filled through "
        "the gap"
    )


def test_seed_overlapping_hippocampus_is_not_filled():
    """An atlas seed lying on the hippocampus must not be filled.

    The IBSR thalamus and ventral DC reach into the hippocampal tail and head
    of real subjects, where Neuromorphometrics still calls the tissue
    hippocampus.  The seeds used to survive every veto, so that overlap was
    filled wholesale.
    """
    lab, p0, reg = _phantom()
    nmm = _guard_atlas(lab, reg)
    hip = _mask(lab, reg, "lHip", "rHip")
    atlas = lab.copy()
    for pre in ("l", "r"):
        tail = (lab == reg[pre + "Hip"]).copy()
        tail[:, 56:, :] = False
        atlas[tail] = reg[pre + "ThaPro"]

    unguarded = _run(p0, atlas)
    assert (unguarded[hip] > 2.5).any(), (
        "phantom no longer exercises the leak: seed not filled without guard"
    )
    guarded = _run(p0, atlas, nmm)
    blob = int((guarded[hip] > 2.5).sum())
    assert not blob, f"{blob} hippocampus voxels filled from the thalamus seed"


def test_guard_lets_the_ventricle_fill_absorb_partial_volume():
    """Thin grey-matter films at the ventricle wall must still be filled.

    The ventricle roof on real data is lined by a partial-volume film that
    reads as grey matter, and misregistration puts cortical parcels over it.
    If the guard held that film, the white surface would dip under the
    corpus callosum -- the roof problem ``ventricle_fill`` exists to solve.
    """
    lab, p0, reg = _phantom()
    roof = np.zeros(lab.shape, bool)
    for pre in ("l", "r"):
        roof |= np.roll(lab == reg[pre + "LatVen"], 2, axis=2)
    roof &= p0 > 2.5
    p0[roof] = 2.0
    nmm = _guard_atlas(lab, reg)
    nmm[roof] = 101

    seg = _run(p0, lab, nmm)
    left = int((seg[roof] < 2.5).sum())
    assert not left, f"{left} of {roof.sum()} roof film voxels left unfilled"


# ---------------------------------------------------------------------------
# Euler number (surface convention)
# ---------------------------------------------------------------------------


def _grid(n=41):
    z, y, x = np.mgrid[:n, :n, :n] - n // 2
    return x, y, z


def _ball(radius, shift=0):
    x, y, z = _grid()
    return (x - shift) ** 2 + y ** 2 + z ** 2 < radius ** 2


def _torus(big=10, small=4, shift=0):
    x, y, z = _grid()
    r = np.sqrt((x - shift) ** 2 + y ** 2)
    return (r - big) ** 2 + z ** 2 < small ** 2


@pytest.mark.parametrize(
    "shape, expected",
    [
        (lambda: _ball(12), 2),                               # sphere
        (lambda: _torus(), 0),                                # one handle
        (lambda: _torus(big=5, small=2, shift=-10)
         | _torus(big=5, small=2, shift=10), 0),              # two rings,
                                                              # 6 voxels apart
        (lambda: _ball(12) & ~_ball(5), 4),                   # enclosed cavity
        (lambda: _ball(6, shift=-10) | _ball(6, shift=10), 4),  # two components
    ],
    ids=["ball", "torus", "two-tori", "cavity", "two-balls"],
)
def test_euler_number_uses_the_surface_convention(shape, expected):
    vol = np.where(shape(), 3.0, 1.0)
    assert compute_euler_number(vol, threshold=2.5) == expected


def test_euler_number_of_a_double_torus():
    # Two rings that overlap in the middle make one genus-2 solid.
    double = _torus(big=8, small=3, shift=-8) | _torus(big=8, small=3, shift=8)
    assert compute_euler_number(np.where(double, 3.0, 1.0)) == -2
