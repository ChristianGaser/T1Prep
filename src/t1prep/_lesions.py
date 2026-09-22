"""White-matter hyperintensity detection and tissue-map correction."""

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    find_objects,
    gaussian_filter,
    generate_binary_structure,
    median_filter,
)
from scipy.ndimage import label as label_image

from ._atlas import get_atlas, resolve_template_file
from ._segment_utils import normalize_to_sum1

#: Calibration of the WMH probability (default path, without AMAP):
#: ``logit p = a + b * hp + c * log(prior + WMH_PRIOR_EPS)``.  ``hp`` is the
#: lesion signal of :func:`lesion_signal` -- three times the intensity deficit
#: of the LAS-normalised image, so 1 is a fully GM-dark voxel -- minus its
#: local level in deep WM (:func:`_highpass`); ``prior`` is the WMH atlas.
#:
#: Fitted (logistic regression on the WMH fraction as soft target) on 14
#: mri_simulate renderings of one anatomy -- SNR 25/50, bias 0/45/90, WMH
#: none/grade 2/grade 4 -- with ``evaluation/tools/eval_phantom.py wmh`` as the
#: measure (``evaluation/tools/fit_wmh_calibration.py`` reproduces it).
#: Leaving out one WMH grade or one noise level moved the signal slope by less
#: than 10%, but the prior weight between 0.7 and 1.45.  One anatomy is thin
#: ground: refit when simulations of others are available.
#:
#: Before 0.7.6 ``p4`` was the signal itself -- about 0.15 inside the lesions
#: of the simulated brains -- so ``p4 > 0.5`` found nothing, and the lesion
#: mask behind it flagged ~19 ml of white matter in every lesion-free brain.
WMH_CALIBRATION = (-3.905, 23.749, 0.959)
WMH_PRIOR_EPS = 0.01

#: Width of the local level (Gaussian sigma, mm) and the largest signal that
#: may contribute to it.  Keeping lesion-strength signal out of the reference
#: stops a large confluent lesion from raising its own level and subtracting
#: itself (a plain clipped average erased the centre of lesions wider than
#: ~2 sigma).
WMH_HP_SIGMA_MM = 5.0
WMH_HP_REF_MAX = 0.2

#: A lesion is a 26-connected component of ``p > WMH_THRESHOLD`` of at least
#: ``WMH_MIN_LESION_MM3``.  On the simulations this found 60% of the lesions
#: with a voxel precision of 0.79 and 1.8 ml of false volume per lesion-free
#: brain (0.7.5: 74%, 0.45 and 19 ml), and the kept volume matched the true
#: WMH volume on average.  A single threshold, not hysteresis: growing
#: components from a low level merged up to 22 lesions into one.
WMH_THRESHOLD = 0.1
WMH_MIN_LESION_MM3 = 30.0


#: Deep grey nuclei, from the affinely placed Neuromorphometrics atlas: kept
#: out of the lesion search.  They are darker than WM, deepmriprep can call
#: their edge WM, and WMHs do not occur in them (what does -- lacunes,
#: perivascular spaces -- is not a WMH).
WMH_EXCLUDE_REGIONS = (
    "Pallidum", "Putamen", "Thalamus", "Caudate", "Accumbens", "Ventral DC",
    "Basal Forebrain",
)


def _deep_gm(t1, affine, p0_large, device):
    """Boolean mask of :data:`WMH_EXCLUDE_REGIONS` on the working grid."""
    atlas = get_atlas(
        t1, affine, p0_large.header, p0_large.affine, "Neuromorphometrics",
        None, device, is_label_atlas=True,
    )
    rois = pd.read_csv(resolve_template_file("Neuromorphometrics", ".csv"), sep=";")
    names = rois.ROIname.astype(str)
    ids = rois.ROIid[names.str.contains("|".join(WMH_EXCLUDE_REGIONS))].tolist()
    return np.isin(np.asanyarray(atlas.dataobj), ids)


def _highpass(diff, candidates, zooms, sigma_mm=None, ref_max=None):
    """``diff`` minus its local level over the candidate WM.

    The level is a normalised Gaussian average (``sigma_mm``) over the
    candidates.  Without ``ref_max`` the signal is clipped to +-0.3 before
    averaging; with it, only voxels below ``ref_max`` contribute, so a large
    confluent lesion does not raise its own reference and subtract itself.
    Computed inside the candidates' bounding box.
    """
    sigma_mm = WMH_HP_SIGMA_MM if sigma_mm is None else sigma_mm
    out = np.zeros(np.shape(diff), dtype=np.float32)
    box = find_objects(candidates.astype(np.uint8))
    if not box:
        return out
    sig = [sigma_mm / float(z) for z in zooms]
    box = tuple(
        slice(max(b.start - int(3 * s) - 1, 0), b.stop + int(3 * s) + 1)
        for b, s in zip(box[0], sig)
    )
    d = np.asarray(diff, dtype=np.float32)[box]
    w = candidates[box].astype(np.float32)
    if ref_max is None:
        v = np.clip(d, -0.3, 0.3) * w
    else:
        w *= d < ref_max
        v = d * w
    level = gaussian_filter(v, sig, truncate=3.0)
    level /= np.maximum(gaussian_filter(w, sig, truncate=3.0), 1e-6)
    out[box] = d - level
    return out


def lesion_probability(hp, prior, candidates, calibration=None):
    """Calibrated probability that a voxel is a WMH (0 outside ``candidates``).

    Args:
        hp: High-passed lesion signal (see :func:`_highpass`), any shape.
        prior: WMH atlas on the same grid, scaled to a maximum of 1.
        candidates: Boolean mask of where lesions are looked for.
        calibration: ``(a, b, c)`` of the logistic model, default
            :data:`WMH_CALIBRATION`.

    Returns:
        ``float32`` array of probabilities in [0, 1].
    """
    a, b, c = WMH_CALIBRATION if calibration is None else calibration
    prob = np.zeros(np.shape(hp), dtype=np.float32)
    logit = (
        a
        + b * np.asarray(hp, dtype=np.float32)[candidates]
        + c * np.log(np.asarray(prior, dtype=np.float32)[candidates] + WMH_PRIOR_EPS)
    )
    prob[candidates] = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
    return prob


def select_lesions(prob, vx_vol, threshold=None, min_lesion_mm3=None):
    """Keep the lesions a probability map supports.

    A lesion is a 26-connected component of ``prob > threshold`` of at least
    ``min_lesion_mm3`` (defaults :data:`WMH_THRESHOLD`,
    :data:`WMH_MIN_LESION_MM3`).  Kept voxels keep their probability.

    Returns:
        ``(prob_kept, mask)``: the map with everything else set to 0, and the
        boolean mask of the kept lesions.
    """
    threshold = WMH_THRESHOLD if threshold is None else threshold
    min_lesion_mm3 = WMH_MIN_LESION_MM3 if min_lesion_mm3 is None else min_lesion_mm3
    lab, n = label_image(prob > threshold, structure=generate_binary_structure(3, 3))
    if n == 0:
        return np.zeros_like(prob), np.zeros(prob.shape, dtype=bool)
    size = np.bincount(lab.ravel(), minlength=n + 1) * vx_vol
    keep = size >= min_lesion_mm3
    keep[0] = False
    mask = keep[lab]
    return np.where(mask, prob, 0.0).astype(np.float32), mask


def lesion_signal(
    t1: nib.Nifti1Image,
    affine,
    brain_large: nib.Nifti1Image,
    p0_large: nib.Nifti1Image,
    p0_large_orig: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    use_amap: bool,
    device: torch.device,
) -> dict:
    """The per-voxel evidence for WMHs, before any lesion is decided on.

    Returns
    -------
    dict
        ``signal`` -- the lesion signal: without AMAP the (median-filtered)
        excess of the label-implied over the observed intensity inside WM,
        ``p3 + 2 p1 + 3 p2 - 3 m``, i.e. three times the intensity deficit of
        the LAS-normalised image (WM = 1, GM = 2/3); with AMAP, AMAP's excess
        GM probability.  ``diff`` -- the same excess everywhere and signed
        (the discrepancy map).  ``prior`` -- the WMH atlas, scaled to a
        maximum of 1.  ``candidates`` -- deep WM away from GM, where lesions
        are looked for.  ``csf`` and, with AMAP, the reference and
        uncorrected tissue maps (``amap``) that the correction needs.
    """
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

    out = {
        "signal": wmh_value,
        "diff": p0_large_diff,
        "prior": wmh_tpm,
        "candidates": deep_wm & ~gm_border,
        "csf": csf,
    }
    if use_amap:
        out["amap"] = {
            "p1": p1_large_uncorr,
            "p2": p2_large_uncorr,
            "p3": p3_large_uncorr,
            "p3_reference": p3_large,
        }
    return out


def _select_legacy(signal, prior, candidates, vx_vol):
    """Lesion mask of T1Prep <= 0.7.5: prior-weighted signal above 0.025.

    Drop small clusters.  Use the same 26-connectivity as the morphology in
    :func:`lesion_signal`, otherwise a diagonally connected lesion is split
    into pieces that can each fall below the threshold.  The size limit is a
    volume, not a voxel count, so it keeps its meaning if the working grid
    ever changes.
    """
    ind_wmh = ((signal * prior) > 0.025) & candidates
    min_lesion_mm3 = 62.5
    min_lesion_size = max(1, int(round(min_lesion_mm3 / max(vx_vol, 1e-6))))
    label_map, _ = label_image(ind_wmh, structure=generate_binary_structure(3, 3))
    sizes = np.bincount(label_map.ravel())
    remove = np.isin(label_map, np.where(sizes < min_lesion_size)[0])
    ind_wmh[remove] = 0
    return ind_wmh


def handle_lesions(
    t1: nib.Nifti1Image,
    affine,
    brain_large: nib.Nifti1Image,
    p0_large: nib.Nifti1Image,
    p0_large_orig: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    use_amap: bool,
    device: torch.device,
) -> tuple[
    nib.Nifti1Image,
    nib.Nifti1Image,
    nib.Nifti1Image,
    nib.Nifti1Image,
    np.ndarray,
    np.ndarray,
]:
    """Detect lesions and correct tissue probability maps.

    Returns
    -------
    tuple
        ``(p1_large, p2_large, p3_large, discrepancy, wmh_value, ind_wmh)``:
        the (with AMAP: corrected) GM, WM and CSF maps, the label discrepancy
        map, the lesion map, and the boolean lesion mask.
    """
    ev = lesion_signal(
        t1, affine, brain_large, p0_large, p0_large_orig,
        p1_large, p2_large, p3_large, use_amap, device,
    )
    zooms = p0_large_orig.header.get_zooms()[:3]
    vx_vol = float(np.prod(zooms))
    csf = ev["csf"]
    p0_large_diff = ev["diff"]
    if use_amap:
        # AMAP's signal is its own excess GM probability, which the tissue
        # correction below moves back to WM; it keeps the 0.7.5 rule.
        wmh_value = ev["signal"]
        ind_wmh = _select_legacy(wmh_value, ev["prior"], ev["candidates"], vx_vol)
        wmh_value[~ind_wmh] = 0
    else:
        # The local level is taken over all candidates, as in the calibration
        # fit; only the probability leaves the deep grey nuclei out.
        hp = _highpass(
            np.asarray(p0_large_diff.dataobj), ev["candidates"], zooms,
            ref_max=WMH_HP_REF_MAX,
        )
        candidates = ev["candidates"] & ~_deep_gm(t1, affine, p0_large, device)
        prob = lesion_probability(hp, ev["prior"], candidates)
        wmh_value, ind_wmh = select_lesions(prob, vx_vol)

    if use_amap:
        p1_large_uncorr = ev["amap"]["p1"]
        p2_large_uncorr = ev["amap"]["p2"]
        p3_large_uncorr = ev["amap"]["p3"]
        p3_large = ev["amap"]["p3_reference"]
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
