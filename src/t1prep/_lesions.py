"""White-matter hyperintensity detection and tissue-map correction."""

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
    median_filter,
)
from scipy.ndimage import label as label_image

from ._atlas import get_atlas
from ._segment_utils import normalize_to_sum1


def handle_lesions(
    t1: nib.Nifti1Image,
    affine,
    brain_large: nib.Nifti1Image,
    p0_large: nib.Nifti1Image,
    p0_large_orig: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    mri_dir: str,
    out_name: str,
    ext: str,
    use_amap: bool,
    debug: bool,
    device: torch.device,
) -> tuple[
    nib.Nifti1Image,
    nib.Nifti1Image,
    nib.Nifti1Image,
    np.ndarray,
    np.ndarray,
]:
    """Detect lesions and correct tissue probability maps."""

    p0_value = p0_large_orig.get_fdata().copy()
    wm = p0_value >= 2.5
    # Fill WM holes to close potential WMH lesions
    wm = binary_closing(wm, generate_binary_structure(3, 3), 3)
    # Get a conservative WM mask
    wm = binary_erosion(wm, generate_binary_structure(3, 3), 2)
    gm = (p0_value >= 1.5) & (p0_value < 2.5)
    csf = (p0_value < 1.5) & (p0_value > 0)

    if use_amap:
        p0_large_diff_value = (
            p3_large.get_fdata().copy()
            + 2 * p1_large.get_fdata().copy()
            + 3 * p2_large.get_fdata().copy()
            - p0_large_orig.get_fdata().copy()
        )

        # Keep the untouched AMAP maps.  Nifti1Image objects are rebound
        # rather than mutated below, but bind them explicitly so an in-place
        # edit later cannot silently corrupt the reference.
        p1_large_uncorr = nib.Nifti1Image(
            p1_large.get_fdata().copy(), p1_large.affine, p1_large.header
        )
        p2_large_uncorr = nib.Nifti1Image(
            p2_large.get_fdata().copy(), p2_large.affine, p2_large.header
        )
        p3_large_uncorr = nib.Nifti1Image(
            p3_large.get_fdata().copy(), p3_large.affine, p3_large.header
        )

        # Reference GM map built from the deepmriprep label.  Inside the
        # conservative WM mask (and in CSF) it is exactly zero, which is what
        # makes `wmh_value` below equal AMAP's GM probability in deep WM --
        # the lesion signal.  Outside those masks it is only a ramp above the
        # CSF/GM threshold, not a probability, so it must not be used for
        # anything that is not restricted to `deep_wm`.
        p0_value = p0_large_orig.get_fdata().copy()
        p0_value[csf | wm] = 1.5
        p0_value -= 1.5
        p1_large = nib.Nifti1Image(p0_value, p0_large.affine, p0_large.header)

        # Reference CSF map, on the probability scale: the label value 1 is
        # pure CSF, 2 is pure GM, so the CSF fraction is 2 - p0 clipped to
        # [0, 1].  Using the raw label value here would make the reference
        # *rise* across the CSF/GM partial-volume band while the true CSF
        # fraction falls, which turned the comparison below into a systematic
        # brain-wide offset instead of a discrepancy.
        p0_value = np.clip(2.0 - p0_large_orig.get_fdata(), 0.0, 1.0)
        p0_value[~csf] = 0
        p3_large = nib.Nifti1Image(p0_value, p0_large.affine, p0_large.header)
        wmh_value = p1_large_uncorr.get_fdata().copy() - p1_large.get_fdata().copy()
    else:
        # brain_large is for the deepmriprep method the LAS corrected orignal
        # image which can be used here as proxi for p0_large_orig from AMAP
        p0_large_diff_value = (
            p3_large.get_fdata().copy()
            + 2 * p1_large.get_fdata().copy()
            + 3 * p2_large.get_fdata().copy()
            - 3 * brain_large.get_fdata().copy()
        )

        # WMH are where p0_large_diff_value shows a positive difference in WM
        wmh_value = np.zeros_like(p0_value)
        wmh_mask = wm & (p0_large_diff_value > 0)
        wmh_value[wmh_mask] = p0_large_diff_value[wmh_mask]

    # Apply median filter to remove noise
    wmh_value = median_filter(wmh_value, size=3)
    p0_large_diff_value = median_filter(p0_large_diff_value, size=3)
    wmh_value = np.clip(wmh_value, -1, 1)
    p0_large_diff_value = np.clip(p0_large_diff_value, -1, 1)
    p0_large_diff = nib.Nifti1Image(
        p0_large_diff_value, p0_large.affine, p0_large.header
    )

    deep_wm = binary_erosion(wm, generate_binary_structure(3, 3), 2)
    gm_border = binary_dilation(gm, generate_binary_structure(3, 3), 2)

    atlas = get_atlas(
        t1,
        affine,
        p0_large.header,
        p0_large.affine,
        "cat_wmh",
        None,
        device,
        is_label_atlas=False,
    )
    wmh_tpm = atlas.get_fdata().copy()
    wmh_tpm /= np.max(wmh_tpm)

    ind_wmh = ((wmh_value * wmh_tpm) > 0.025) & deep_wm & (~gm_border)

    # Drop small clusters.  Use the same 26-connectivity as the morphology
    # above, otherwise a diagonally connected lesion is split into pieces that
    # can each fall below the threshold.  The size limit is a volume, not a
    # voxel count, so it keeps its meaning if the working grid ever changes.
    min_lesion_mm3 = 62.5
    vx_vol = float(np.prod(p0_large_orig.header.get_zooms()[:3]))
    min_lesion_size = max(1, int(round(min_lesion_mm3 / max(vx_vol, 1e-6))))
    label_map, _ = label_image(ind_wmh, structure=generate_binary_structure(3, 3))
    sizes = np.bincount(label_map.ravel())
    remove = np.isin(label_map, np.where(sizes < min_lesion_size)[0])
    ind_wmh[remove] = 0

    wmh_value[~ind_wmh] = 0

    if use_amap:
        csf_discrep_large = (
            p3_large_uncorr.get_fdata().copy() - p3_large.get_fdata().copy()
        )
        csf_discrep_large = median_filter(csf_discrep_large, size=3)

        # Act only where AMAP and the reference disagree by a meaningful
        # amount, and only inside the CSF band the reference is defined on.
        # Without a threshold this fired on every voxel with even a rounding
        # difference, and the correction is applied brain-wide -- unlike the
        # WMH one, which is confined by deep_wm/gm_border and a size filter.
        min_csf_discrep = 0.05
        ind_csf_discrep = (csf_discrep_large < -min_csf_discrep) & csf

        # Direction: AMAP found less CSF (more GM) than deepmriprep here.
        # deepmriprep is the map that tends to miss lesions and underestimate
        # GM, so AMAP's finding is kept and reinforced rather than pulled back
        # towards the reference.  Flip the two signs below to instead correct
        # AMAP towards deepmriprep.
        tmp_p1 = p1_large_uncorr.get_fdata().copy()
        tmp_p1[ind_wmh] -= wmh_value[ind_wmh]
        tmp_p1[ind_csf_discrep] -= csf_discrep_large[ind_csf_discrep]

        tmp_p2 = p2_large_uncorr.get_fdata().copy()
        tmp_p2[ind_wmh] += wmh_value[ind_wmh]

        tmp_p3 = p3_large_uncorr.get_fdata().copy()
        tmp_p3[ind_csf_discrep] += csf_discrep_large[ind_csf_discrep]

        # We have to normalize all tissue values to overall sum of one
        tmp_p1, tmp_p2, tmp_p3 = normalize_to_sum1(tmp_p1, tmp_p2, tmp_p3)

        # Convert back to nifti
        p1_large = nib.Nifti1Image(tmp_p1, p0_large.affine, p0_large.header)
        p2_large = nib.Nifti1Image(tmp_p2, p0_large.affine, p0_large.header)
        p3_large = nib.Nifti1Image(tmp_p3, p0_large.affine, p0_large.header)

    return p1_large, p2_large, p3_large, p0_large_diff, wmh_value, ind_wmh
