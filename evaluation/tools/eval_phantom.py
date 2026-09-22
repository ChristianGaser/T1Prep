#!/usr/bin/env python3
"""Score T1Prep against a simulated brain with a known ground truth.

The phantom in ``evaluation/data/phantom`` is one real anatomy (HR075 MPRAGE)
re-rendered by ``mri_simulate`` (T1-MRI-Phantom) with Rician noise at a WM SNR
of 25, an RF bias field of strength 45 and medium white matter
hyperintensities.  Its ``dseg`` is the partial-volume label the image was
rendered from, in T1Prep's own ``p0`` convention (1 CSF, 2 GM, 3 WM, 4 WMH),
cleaned of vessels and dura -- which the image still shows, on purpose.

What is measured
----------------
Everything that has a ground truth: tissue volumes and Dice, where the
GM/WM and GM/CSF boundaries sit, how much dura and vessel survives as GM, the
WMH map, the bias correction, and the QA noise rating against the noise that
was actually added.

Cortical thickness is not simulated, so it has no ground truth of its own.
It gets a *reference* instead: the surface pipeline also runs on hemisphere
labels built from the ground truth, with the pipeline's own partition (the
same deep-GM and ventricle fills, the same cleared cerebellum, brainstem and
other side, recorded during the run by a hook).  The two arms then differ in
the tissue label alone, so their difference is the thickness error the
segmentation causes -- not PBT's own bias, which a label with known thickness
would have to measure.

Commands
--------
``run``    T1Prep on the phantom plus the ground-truth surface arm, then score.
``score``  (Re)score a finished ``run`` directory.
``pin``    Write the pinned scalars from one or more scored runs.
``check``  Compare a scored run with the pinned scalars; exit 1 on regression.

This is a manual test: a run takes ~15 min.  Typical use::

    python evaluation/tools/eval_phantom.py run --work /tmp/phantom
    python evaluation/tools/eval_phantom.py check --work /tmp/phantom

Pinned values are characterization, not targets: they record what the code
did when they were pinned, with a tolerance derived from the spread between
repeated runs, so ``check`` flags *change*.  An improvement beyond the
tolerance is reported but only fails with ``--strict``; re-pin to accept it.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "evaluation" / "data" / "phantom"
T1_DEFAULT = DATA / "HR075_MPRAGE_desc-snr25Rf45T4Wmh2_T1w.nii.gz"
GT_DEFAULT = DATA / "HR075_MPRAGE_desc-Wmh2Clean_dseg.nii.gz"
PINNED_DEFAULT = REPO / "evaluation" / "results" / "phantom_pinned.json"
HISTORY_DEFAULT = REPO / "evaluation" / "results" / "phantom_history.csv"
RESULTS_NAME = "phantom_results.json"

#: How far a pinned value may move, per metric: ``direction`` says which way
#: is worse ("higher" = higher is better, "lower" = lower is better, "both" =
#: any change is a change), and the tolerance is the largest of 3x the spread
#: seen between the pinned runs, ``abs`` and ``rel`` x |value|.  The floors are
#: what a re-run on another device or thread count moved the value by when
#: they were set; they only matter when the pinned runs happened to agree.
METRICS: dict[str, tuple[str, float, float]] = {
    # tissue label (p0) against the ground truth, WMH counted as WM
    "CSF_dice": ("higher", 0.005, 0.0),
    "GM_dice": ("higher", 0.005, 0.0),
    "WM_dice": ("higher", 0.005, 0.0),
    "CSF_soft_dice": ("higher", 0.005, 0.0),
    "GM_soft_dice": ("higher", 0.005, 0.0),
    "WM_soft_dice": ("higher", 0.005, 0.0),
    "CSF_vol_err_pct": ("both", 1.0, 0.0),
    "GM_vol_err_pct": ("both", 1.0, 0.0),
    "WM_vol_err_pct": ("both", 1.0, 0.0),
    "label_mae": ("lower", 0.005, 0.0),
    # volumes the report states, against the ground truth
    "report_TIV_err_pct": ("both", 1.0, 0.0),
    "report_GM_err_pct": ("both", 1.0, 0.0),
    "report_WM_err_pct": ("both", 1.0, 0.0),
    "report_CSF_err_pct": ("both", 1.0, 0.0),
    # where the boundaries sit (mm, positive = outward of the ground truth)
    "shift_white_mm": ("both", 0.03, 0.0),
    "shift_pial_mm": ("both", 0.03, 0.0),
    # tissue the ground truth rejects
    "fp_gm_far_ml": ("lower", 0.3, 0.2),
    "fp_brain_far_ml": ("lower", 0.5, 0.2),
    "missed_brain_ml": ("lower", 0.5, 0.2),
    # white matter hyperintensities (p4)
    "wmh_soft_dice": ("higher", 0.02, 0.0),
    "wmh_dice_t010": ("higher", 0.02, 0.0),
    "wmh_detected_t010": ("higher", 0.03, 0.0),
    "wmh_pred_ml": ("both", 0.3, 0.1),
    "wmh_mass_in_lesion": ("higher", 0.03, 0.0),
    "wmh_f1": ("higher", 0.03, 0.0),
    "wmh_false_clusters": ("lower", 2.0, 0.25),
    "wmh_dice_t050": ("higher", 0.02, 0.0),
    "wmh_precision": ("higher", 0.03, 0.0),
    "wmh_flagged_ml": ("both", 0.5, 0.15),
    # bias correction: WM uniformity of the corrected image m
    "m_wm_cv": ("lower", 0.002, 0.0),
    "m_wm_lowfreq_cv": ("lower", 0.002, 0.0),
    "m_cjv": ("lower", 0.01, 0.0),
    # QA ratings of the raw image (the truth is recorded next to them)
    "qa_NCR": ("both", 0.0, 0.05),
    "qa_ICR": ("both", 0.0, 0.05),
    "qa_IQR": ("both", 0.05, 0.0),
    # cortical thickness against the ground-truth surface arm
    "thick_diff_median_mm": ("both", 0.02, 0.0),
    "thick_mae_mm": ("lower", 0.01, 0.0),
    "thick_r": ("higher", 0.01, 0.0),
    "central_dist_mm": ("lower", 0.02, 0.0),
    "region_bias_median_mm": ("both", 0.02, 0.0),
    "region_spearman": ("higher", 0.03, 0.0),
    # topology of the hemisphere label (|Euler - 2| / 2 = defects)
    "defects_lh": ("lower", 0.0, 0.25),
    "defects_rh": ("lower", 0.0, 0.25),
}

#: The ``wmh`` command's summary over a set of simulations (same semantics).
WMH_METRICS: dict[str, tuple[str, float, float]] = {
    "f1": ("higher", 0.02, 0.0),
    "found_frac": ("higher", 0.02, 0.0),
    "false_clusters": ("lower", 3.0, 0.25),
    "dice_t050": ("higher", 0.02, 0.0),
    "soft_dice": ("higher", 0.02, 0.0),
    "vol_abs_err_ml": ("lower", 0.3, 0.1),
    "clean_fp_ml": ("lower", 0.2, 0.25),
    "precision": ("higher", 0.03, 0.0),
    "clean_flagged_ml": ("lower", 0.3, 0.25),
}

WMH_PINNED_DEFAULT = REPO / "evaluation" / "results" / "phantom_wmh_pinned.json"

#: Invariants every run must satisfy, whatever is pinned.
MIN_ALIGNMENT = 0.6  # GM soft Dice below this means a misregistered run
MIN_HEMI_CORR = 0.8  # GT vs pipeline hemisphere: the GT arm is valid


# ---------------------------------------------------------------------------
# Metrics (pure functions, tested in tests/test_eval_phantom.py)
# ---------------------------------------------------------------------------


def tissue_fractions(label: np.ndarray) -> dict[str, np.ndarray]:
    """CSF, GM and WM fractions of a ``p0`` label (values above 3 are WM)."""
    lab = np.clip(label, 0, 3)
    return {
        "CSF": np.clip(np.minimum(lab, 2 - lab), 0, 1),
        "GM": np.clip(np.minimum(lab - 1, 3 - lab), 0, 1),
        "WM": np.clip(lab - 2, 0, 1),
    }


def soft_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two fraction maps: overlap is the voxel-wise minimum."""
    denom = float(a.sum() + b.sum())
    return 2.0 * float(np.minimum(a, b).sum()) / denom if denom > 0 else float("nan")


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two boolean masks."""
    denom = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / denom if denom else float("nan")


def boundary_shift(gt: np.ndarray, pred: np.ndarray, thr: float, zooms) -> float:
    """Mean displacement (mm) of the predicted iso-surface from the true one.

    Positive means the predicted object is larger (its surface lies outward).
    The partial-volume content beyond the iso-level -- the WM fraction for
    the GM/WM surface at 2.5, GM + WM for the pial one at 1.5 -- changes by
    area x displacement to first order, so the volume difference divided by
    the true surface area is the mean normal displacement.  It resolves shifts
    well below a voxel, which a voxel-distance measure does not: that one was
    off by up to 15% on shifted balls and asymmetric between in and out.
    Inward and outward displacements in different places cancel, as in any
    mean; the curvature term is displacement / radius (< 1% at 0.2 mm).
    """
    from skimage.measure import marching_cubes, mesh_surface_area

    def content(label):
        return float(np.clip(label - (thr - 0.5), 0, 1).sum())

    dv = (content(pred) - content(gt)) * float(np.prod(zooms))
    verts, faces, _, _ = marching_cubes(gt.astype(np.float32), level=thr, spacing=zooms)
    return dv / float(mesh_surface_area(verts, faces))


def local_noise(image: np.ndarray, mask: np.ndarray) -> float:
    """Noise SD from the residual to the 3x3x3 mean inside ``mask``."""
    from scipy import ndimage as ndi

    resid = image - ndi.uniform_filter(image, 3)
    return float(resid[mask].std() * np.sqrt(27.0 / 26.0))


def lowfreq_cv(image: np.ndarray, mask: np.ndarray, zooms, sigma_mm=4.0) -> float:
    """CV of the image smoothed inside ``mask``: what is left of the bias field.

    Normalised convolution keeps the neighbouring tissue out, and the smoothing
    removes noise and denoising alike, so only low-frequency variation of the
    tissue intensity counts.
    """
    from scipy import ndimage as ndi

    sig = [sigma_mm / z for z in zooms]
    m = mask.astype(np.float32)
    num = ndi.gaussian_filter(image * m, sig)
    den = ndi.gaussian_filter(m, sig)
    inner = ndi.binary_erosion(mask, iterations=2)
    smooth = num[inner] / np.maximum(den[inner], 1e-6)
    return float(smooth.std() / smooth.mean())


def csf_by_zone(label: np.ndarray, gt_csf, pred_csf, zooms) -> dict:
    """Net CSF error (ml) in the outer rim, the ventricles and the sulci.

    A CSF volume error can come from the brain mask (skull-strip, dura
    removal), which acts on the outer rim, or from the segmentation, which acts
    on sulci and ventricle walls.  On the phantom they had opposite signs: the
    rim was right to 3 ml, the sulci 37 ml short.

    - **rim:** within 3 mm of the true intracranial edge, or outside it.
    - **ventricles:** CSF bodies larger than 0.8 ml more than 12 mm inside,
      plus 2 voxels of wall.
    - **sulci:** everything else inside, cisterns included.
    """
    from scipy import ndimage as ndi

    ml = float(np.prod(zooms)) / 1000.0
    inside = label > 0.5
    d_out = ndi.distance_transform_edt(inside, sampling=zooms)
    comp, _ = ndi.label((label > 0.5) & (label < 1.5) & (d_out > 12))
    sizes = np.bincount(comp.ravel()) * ml
    sizes[0] = 0
    vent = ndi.binary_dilation(np.isin(comp, np.flatnonzero(sizes > 0.8)), iterations=2)
    rim = (d_out <= 3) | ~inside
    zones = {"rim": rim, "ventricles": vent & ~rim, "sulci": inside & ~rim & ~vent}
    diff = pred_csf - gt_csf
    return {
        f"csf_err_{k}_ml": round(float(diff[z].sum()) * ml, 2) for k, z in zones.items()
    }


def wmh_scores(label: np.ndarray, p4: np.ndarray, vx_mm3: float) -> dict:
    """Lesion- and voxel-wise agreement of a WMH map with the ground truth.

    A *reported lesion* is a 26-connected component of ``p4 > 0.01`` (the
    method's own positive region, whatever its scale).  A true lesion is
    *found* when a reported lesion overlaps its core (label > 3.5); a reported
    lesion is *false* when it touches no true WMH at all.  Dice at 0.5 asks
    whether ``p4`` can be read as a probability.  Counts alone can be gamed by
    flagging a lot of white matter -- a large enough mask covers every lesion
    -- so the flagged volume and its voxel precision go with them.
    """
    from scipy import ndimage as ndi

    p4 = np.clip(np.nan_to_num(p4), 0, 1)
    frac = np.clip(label - 3, 0, 1)
    core = label > 3.5
    comp, n = ndi.label(core)
    rep, k = ndi.label(p4 > 0.01, structure=np.ones((3, 3, 3)))
    found = int((ndi.maximum(rep > 0, comp, np.arange(1, n + 1)) > 0).sum()) if n else 0
    touching = ndi.maximum(frac > 0, rep, np.arange(1, k + 1)) if k else []
    false = int(k - int(np.sum(touching)))
    denom = 2 * found + (n - found) + false
    near = ndi.binary_dilation(core, iterations=2)
    return {
        "lesions": int(n),
        "found": found,
        "false_clusters": false,
        "f1": 2 * found / denom if denom else 1.0,
        "detected_t010": (
            float(np.mean(ndi.maximum(p4 > 0.1, comp, np.arange(1, n + 1))))
            if n
            else float("nan")
        ),
        "dice_t010": dice(p4 > 0.1, core),
        "dice_t050": dice(p4 > 0.5, core),
        "soft_dice": soft_dice(frac, p4),
        "pred_ml": float(p4.sum()) * vx_mm3 / 1000.0,
        "flagged_ml": float((rep > 0).sum()) * vx_mm3 / 1000.0,
        "precision": float(((rep > 0) & (frac > 0)).sum() / max((rep > 0).sum(), 1)),
        "gt_ml": float(frac.sum()) * vx_mm3 / 1000.0,
        "mass_in_lesion": float(p4[near].sum() / max(p4.sum(), 1e-9)),
        "p4_max": float(p4.max()),
    }


def compare(value: float, pinned: dict, strict: bool = False) -> str:
    """``ok``, ``improved`` or ``REGRESSED`` for one metric against its pin."""
    if value is None or not np.isfinite(value):
        return "REGRESSED"
    delta = value - pinned["value"]
    tol = pinned["tol"]
    direction = pinned["direction"]
    if abs(delta) <= tol:
        return "ok"
    if direction == "both":
        return "REGRESSED"
    better = delta > 0 if direction == "higher" else delta < 0
    if better:
        return "REGRESSED" if strict else "improved"
    return "REGRESSED"


def make_pins(runs: list[dict]) -> dict:
    """Pinned value and tolerance of every metric from one or more runs."""
    table = WMH_METRICS if runs and runs[0].get("kind") == "wmh" else METRICS
    pins = {}
    for name, (direction, abs_floor, rel_floor) in table.items():
        vals = [r["metrics"].get(name) for r in runs]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            continue
        mean = float(np.mean(vals))
        spread = float(np.max(vals) - np.min(vals))
        tol = max(3.0 * spread, abs_floor, rel_floor * abs(mean))
        pins[name] = {
            "value": round(mean, 6),
            "tol": round(tol, 6),
            "direction": direction,
            "spread": round(spread, 6),
            "runs": [round(v, 6) for v in vals],
        }
    return pins


# ---------------------------------------------------------------------------
# Scoring a finished run
# ---------------------------------------------------------------------------


def _one(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def _on_grid(img, ref, order=1) -> np.ndarray:
    """Data of ``img`` on the voxel grid of ``ref`` (world-space resampling)."""
    from nibabel.processing import resample_from_to

    if img.shape[:3] == ref.shape[:3] and np.allclose(
        img.affine, ref.affine, atol=1e-4
    ):
        return np.asarray(img.dataobj, dtype=np.float32)
    out = resample_from_to(img, (ref.shape[:3], ref.affine), order=order)
    return np.asarray(out.dataobj, dtype=np.float32)


def _euler_defects(path: str | None) -> float | None:
    if not path:
        return None
    import nibabel as nib

    sys.path.insert(0, str(REPO / "src"))
    from t1prep._partition import compute_euler_number

    vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return abs(compute_euler_number(vol, threshold=2.5) - 2) / 2.0


def score_volumes(work: Path, t1_path: Path, gt_path: Path) -> tuple[dict, dict]:
    """Volume metrics of a run.  Returns ``(metrics, details)``."""
    import nibabel as nib
    from scipy import ndimage as ndi

    mri = work / "t1prep" / "mri"
    gt_img = nib.load(str(gt_path))
    L = np.asarray(gt_img.dataobj, dtype=np.float32)
    zooms = tuple(float(z) for z in gt_img.header.get_zooms()[:3])
    ml = float(np.prod(zooms)) / 1000.0
    met, det = {}, {}

    p0_path = _one(f"{mri}/p0*.nii*")
    if not p0_path:
        raise SystemExit(f"no p0 label under {mri}: did the run finish?")
    P = _on_grid(nib.load(p0_path), gt_img)
    det["p0_range"] = [float(np.nanmin(P)), float(np.nanmax(P))]
    det["p0_finite"] = bool(np.isfinite(P).all())

    fg, fp = tissue_fractions(L), tissue_fractions(P)
    for k, code in (("CSF", 1), ("GM", 2), ("WM", 3)):
        vg, vp = float(fg[k].sum()) * ml, float(fp[k].sum()) * ml
        det[f"{k}_gt_ml"], det[f"{k}_pred_ml"] = round(vg, 2), round(vp, 2)
        met[f"{k}_vol_err_pct"] = 100.0 * (vp - vg) / vg
        met[f"{k}_soft_dice"] = soft_dice(fg[k], fp[k])
        met[f"{k}_dice"] = dice(
            np.rint(np.clip(L, 0, 3)) == code, np.rint(np.clip(P, 0, 3)) == code
        )
    union = (L > 0) | (P > 0)
    met["label_mae"] = float(np.abs(np.clip(L, 0, 3) - np.clip(P, 0, 3))[union].mean())
    det.update(csf_by_zone(L, fg["CSF"], fp["CSF"], zooms))

    Lc, Pc = np.clip(L, 0, 3), np.clip(P, 0, 3)
    met["shift_white_mm"] = boundary_shift(Lc, Pc, 2.5, zooms)
    met["shift_pial_mm"] = boundary_shift(Lc, Pc, 1.5, zooms)

    # Tissue the ground truth rejects.  Within 2 mm of the true GM it is a
    # boundary disagreement; beyond, it is a structure (dura, vessel, sinus).
    dist_gm = ndi.distance_transform_edt(~(Lc >= 1.5), sampling=zooms)
    fp_gm = (Pc >= 1.5) & (Pc < 2.5) & (L < 1.5)
    met["fp_gm_far_ml"] = float((fp_gm & (dist_gm > 2.0)).sum()) * ml
    det["fp_gm_near_ml"] = round(float((fp_gm & (dist_gm <= 2.0)).sum()) * ml, 2)
    dist_brain = ndi.distance_transform_edt(~(L > 0.5), sampling=zooms)
    met["fp_brain_far_ml"] = float(((P > 0.5) & (dist_brain > 2.0)).sum()) * ml
    met["missed_brain_ml"] = float(((L > 1.5) & (P < 0.5)).sum()) * ml

    # White matter hyperintensities
    p4_path = _one(f"{mri}/p4*.nii*")
    if p4_path:
        w = wmh_scores(L, _on_grid(nib.load(p4_path), gt_img), float(np.prod(zooms)))
        met["wmh_soft_dice"] = w["soft_dice"]
        met["wmh_dice_t010"] = w["dice_t010"]
        met["wmh_dice_t050"] = w["dice_t050"]
        met["wmh_detected_t010"] = w["detected_t010"]
        met["wmh_pred_ml"] = w["pred_ml"]
        met["wmh_mass_in_lesion"] = w["mass_in_lesion"]
        met["wmh_f1"] = w["f1"]
        met["wmh_false_clusters"] = float(w["false_clusters"])
        met["wmh_precision"] = w["precision"]
        met["wmh_flagged_ml"] = w["flagged_ml"]
        det["wmh_gt_ml"] = round(w["gt_ml"], 2)
        det["wmh_gt_lesions"] = w["lesions"]
        det["wmh_found"] = w["found"]
        det["wmh_p4_max"] = round(w["p4_max"], 3)

    # Bias correction and the noise that was actually added
    raw = np.asarray(nib.load(str(t1_path)).dataobj, dtype=np.float32)
    wm = ndi.binary_erosion(np.isclose(L, 3.0), iterations=1)
    gm = np.isclose(L, 2.0)
    sigma = local_noise(raw, ndi.binary_erosion(wm, iterations=1))
    wm_med, gm_med = float(np.median(raw[wm])), float(np.median(raw[gm]))
    det["true_noise_sd"] = round(sigma, 2)
    det["true_snr_wm"] = round(wm_med / sigma, 2)
    det["true_ncr_native"] = round(sigma / (wm_med - gm_med), 4)
    # QA measures noise after block-averaging to ~2.3 mm, as CAT12 does, so
    # its NCR is only comparable with the noise at that resolution (white
    # noise falls with the square root of the block size).
    try:
        sys.path.insert(0, str(REPO / "src"))
        from t1prep.qa import _reduce_factor

        step = _reduce_factor(np.array(zooms), 2.3, shape=raw.shape)
        det["true_ncr_qa_res"] = round(
            sigma / float(np.sqrt(np.prod(step))) / (wm_med - gm_med), 4
        )
    except ImportError:
        pass
    det["raw_wm_cv"] = round(float(raw[wm].std() / raw[wm].mean()), 4)
    det["raw_wm_lowfreq_cv"] = round(lowfreq_cv(raw, wm, zooms), 4)
    m_path = _one(f"{mri}/m[A-Z]*.nii*") or _one(f"{mri}/*desc-corr*.nii*")
    if m_path:
        M = _on_grid(nib.load(m_path), gt_img)
        met["m_wm_cv"] = float(M[wm].std() / M[wm].mean())
        met["m_wm_lowfreq_cv"] = lowfreq_cv(M, wm, zooms)
        met["m_cjv"] = float(
            (M[wm].std() + M[gm].std()) / abs(M[wm].mean() - M[gm].mean())
        )

    # The GM overlap doubles as the check that the run is in the right space
    det["aligned"] = met["GM_soft_dice"] >= MIN_ALIGNMENT
    det["gt_soft_ml"] = {k: round(float(fg[k].sum()) * ml, 2) for k in fg}
    return met, det


def score_report(work: Path, gt_soft_ml: dict) -> tuple[dict, dict]:
    """Volumes and QA from the run's JSON report."""
    met, det = {}, {}
    rep = _one(str(work / "t1prep" / "report" / "log_*.json"))
    if not rep:
        return met, det
    with open(rep) as fh:
        data = json.load(fh)
    sm = data.get("subjectmeasures", {})
    qm = data.get("qualitymeasures", {})
    cgw = sm.get("vol_abs_CGW", {}).get("value")
    if cgw:
        for k, v in zip(("CSF", "GM", "WM"), cgw):
            met[f"report_{k}_err_pct"] = 100.0 * (v - gt_soft_ml[k]) / gt_soft_ml[k]
        tiv_gt = sum(gt_soft_ml.values())
        met["report_TIV_err_pct"] = 100.0 * (sm["vol_TIV"]["value"] - tiv_gt) / tiv_gt
    if "vol_WMH" in sm:
        det["report_wmh_ml"] = round(float(sm["vol_WMH"]["value"]), 3)
    for key in ("NCR", "ICR", "IQR", "res_ECR", "SIQR"):
        if key in qm:
            val = qm[key]["value"]
            det[f"qa_{key}"] = val
            if f"qa_{key}" in METRICS:
                met[f"qa_{key}"] = float(val)
    for side in ("lh", "rh"):
        if f"euler_{side}" in qm:
            det[f"report_euler_{side}"] = qm[f"euler_{side}"]["value"]
    return met, det


def _surface_files(surf: Path, hemi: str) -> tuple[str | None, str | None]:
    central = _one(f"{surf}/{hemi}.central.*.gii")
    thick = _one(f"{surf}/{hemi}.thickness.*")
    return central, thick


def score_thickness(work: Path) -> tuple[dict, dict]:
    """Paired thickness of the T1Prep arm against the ground-truth arm."""
    import nibabel as nib
    from nibabel.freesurfer.io import read_annot, read_morph_data
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    met, det = {}, {}
    pred_surf, gt_surf = work / "t1prep" / "surf", work / "gt_arm" / "surf"
    diffs, pairs, dists, regional = [], [], [], []
    for hemi in ("lh", "rh"):
        pc, pt = _surface_files(pred_surf, hemi)
        gc, gtk = _surface_files(gt_surf, hemi)
        if not all((pc, pt, gc, gtk)):
            det[f"thickness_{hemi}"] = "missing"
            continue
        pv = nib.load(pc).agg_data("pointset")
        gv = nib.load(gc).agg_data("pointset")
        p_th, g_th = read_morph_data(pt), read_morph_data(gtk)
        det[f"thick_finite_{hemi}"] = bool(
            np.isfinite(p_th).all() and (p_th >= 0).all() and len(p_th) == len(pv)
        )
        det[f"thick_median_pred_{hemi}"] = round(float(np.median(p_th)), 3)
        det[f"thick_median_gt_{hemi}"] = round(float(np.median(g_th)), 3)
        dist, idx = cKDTree(pv).query(gv)
        close = dist < 1.0  # compare only where the surfaces coincide
        d = p_th[idx][close] - g_th[close]
        diffs.append(d)
        pairs.append((p_th[idx][close], g_th[close]))
        dists.append(dist)
        det[f"thick_diff_median_{hemi}"] = round(float(np.median(d)), 3)
        det[f"thick_loa95_{hemi}"] = [
            round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])
        ]
        det[f"central_far_frac_{hemi}"] = round(float(np.mean(~close)), 4)

        # Region means by colour-table index: both annots come from the same
        # template, so the index is the region whatever its stored name says.
        pa = _one(f"{pred_surf}/{hemi}.aparc_DK40.*.annot")
        ga = _one(f"{gt_surf}/{hemi}.aparc_DK40.*.annot")
        if pa and ga:
            pl, _, names = read_annot(pa)
            gl, _, _ = read_annot(ga)
            for i, raw in enumerate(names):
                name = raw.decode("latin-1").split("\x00")[0].strip()
                if name.lower() in ("unknown", "corpuscallosum", "medial_wall"):
                    continue
                pm, gm = pl == i, gl == i
                if pm.sum() > 200 and gm.sum() > 200:
                    regional.append(
                        (
                            f"{hemi}.{name}",
                            float(p_th[pm].mean()),
                            float(g_th[gm].mean()),
                        )
                    )

    if diffs:
        d = np.concatenate(diffs)
        p = np.concatenate([a for a, _ in pairs])
        g = np.concatenate([b for _, b in pairs])
        met["thick_diff_median_mm"] = float(np.median(d))
        met["thick_mae_mm"] = float(np.abs(d).mean())
        met["thick_r"] = float(np.corrcoef(p, g)[0, 1])
        met["central_dist_mm"] = float(np.concatenate(dists).mean())
    if len(regional) >= 10:
        rp = np.array([r[1] for r in regional])
        rg = np.array([r[2] for r in regional])
        met["region_bias_median_mm"] = float(np.median(rp - rg))
        met["region_spearman"] = float(spearmanr(rp, rg)[0])
        order = np.argsort(rp - rg)
        det["regions_most_thinned"] = [
            (regional[i][0], round(rp[i] - rg[i], 3)) for i in order[:3]
        ]
        det["regions_most_thickened"] = [
            (regional[i][0], round(rp[i] - rg[i], 3)) for i in order[-3:]
        ]
    det["regions_compared"] = len(regional)
    return met, det


def score_hemispheres(work: Path) -> tuple[dict, dict]:
    """Range invariant and topology of both arms' hemisphere labels."""
    import nibabel as nib

    met, det = {}, {}
    for arm, mri in (
        ("pred", work / "t1prep" / "mri"),
        ("gt", work / "gt_arm" / "mri"),
    ):
        for hemi in ("lh", "rh"):
            path = _one(f"{mri}/{hemi}.seg.*.nii*")
            if not path:
                continue
            vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
            det[f"hemi_max_{arm}_{hemi}"] = round(float(vol.max()), 4)
            defects = _euler_defects(path)
            if arm == "pred":
                met[f"defects_{hemi}"] = defects
            else:
                det[f"defects_gt_{hemi}"] = defects
    for name, key in (
        ("gt_hook.json", "gt_hook"),
        ("gt_hemispheres.json", "gt_hemispheres"),
    ):
        path = work / "gt_arm" / "mri" / name
        if path.exists():
            det[key] = json.loads(path.read_text())
    return met, det


def _git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(REPO), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--", "src", "scripts")
    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(status)}


def _model_hash() -> str | None:
    """SHA-256 over the model weights, so a pin is tied to its models."""
    try:
        sys.path.insert(0, str(REPO / "src"))
        from t1prep._models import MODEL_DIR, MODEL_FILES
    except ImportError:
        return None
    h = hashlib.sha256()
    for name in sorted(MODEL_FILES):
        path = Path(MODEL_DIR) / name
        if not path.exists():
            return None
        h.update(name.encode())
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def _versions() -> dict:
    """T1Prep and cat-surf versions, ``None`` where one cannot be found."""
    from importlib.metadata import PackageNotFoundError, version

    out = {"t1prep": None, "cat_surf": None}
    sys.path.insert(0, str(REPO / "src"))
    try:
        import t1prep

        out["t1prep"] = t1prep.__version__
    except ImportError:
        out["t1prep"] = None
    try:
        out["cat_surf"] = version("cat-surf")
    except PackageNotFoundError:
        out["cat_surf"] = None
    return out


def score(work: Path, t1_path: Path, gt_path: Path) -> dict:
    """Score a run directory and write ``phantom_results.json`` into it."""
    t0 = time.perf_counter()
    met, det = score_volumes(work, t1_path, gt_path)
    m2, d2 = score_report(work, det["gt_soft_ml"])
    m3, d3 = score_thickness(work)
    m4, d4 = score_hemispheres(work)
    for m, d in ((m2, d2), (m3, d3), (m4, d4)):
        met.update(m)
        det.update(d)
    run_info = {}
    info_path = work / "run_info.json"
    if info_path.exists():
        run_info = json.loads(info_path.read_text())
    results = {
        "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "t1": str(t1_path),
        "gt": str(gt_path),
        "versions": _versions(),
        "git": _git_state(),
        "model_hash": _model_hash(),
        "run": run_info,
        "metrics": {k: (None if v is None else float(v)) for k, v in met.items()},
        "details": det,
        "invariants": invariants(met, det),
        "scoring_seconds": round(time.perf_counter() - t0, 1),
    }
    (work / RESULTS_NAME).write_text(json.dumps(results, indent=1, default=float))
    return results


def invariants(met: dict, det: dict) -> dict[str, bool]:
    """Checks that hold for any correct run, independent of the pins."""
    inv = {
        "p0 finite": det.get("p0_finite", False),
        "p0 within [0, 4]": det.get("p0_range", [0, 99])[0] >= 0
        and det.get("p0_range", [0, 99])[1] <= 4.0 + 1e-4,
        "output aligned with the ground truth": bool(det.get("aligned")),
    }
    for key, val in det.items():
        if key.startswith("hemi_max_"):
            arm = key.replace("hemi_max_", "", 1)
            inv[f"{arm} hemisphere within [1, 3]"] = val <= 3.0 + 1e-3
        if key.startswith("thick_finite_"):
            inv[f"thickness finite, >= 0 ({key[-2:]})"] = val
    hemis = det.get("gt_hemispheres")
    if hemis is not None:
        inv["ground-truth hemispheres aligned with the run"] = all(
            h.get("corr_with_pipeline", 0) >= MIN_HEMI_CORR for h in hemis.values()
        )
    hook = det.get("gt_hook")
    if hook is not None:
        inv["partition masks match the run"] = all(
            h.get("fill_agreement", 0) >= 0.999 for h in hook.values()
        )
    return {k: bool(v) for k, v in inv.items()}


# ---------------------------------------------------------------------------
# Running T1Prep with the ground-truth arm
# ---------------------------------------------------------------------------

#: Imported by every Python process T1Prep starts during ``run`` (it sits on
#: PYTHONPATH as ``sitecustomize``).  It only acts in the process that imports
#: ``t1prep._partition``, and never changes what the pipeline computes.
#:
#: It records *where* ``get_partition`` overrode the label: the deep grey
#: nuclei and ventricles it filled with WM, and the tissue it cleared (other
#: hemisphere, cerebellum, brainstem).  A second call on the label lowered by
#: 1e-4 makes both exact -- a fill is then the only way to reach 3.0, and 1.0
#: over tissue can only be a clearance -- and the two masks go through the same
#: reslice as the hemisphere.  The ground-truth hemispheres are then built on
#: that grid from these masks (see ``build_gt_hemispheres``), so the two arms
#: share atlas, fills and clearances and differ in the tissue label alone.
#:
#: Re-partitioning the ground truth itself on the working grid is not an
#: option: that grid lives in the affinely registered space, so its affine does
#: not place native data correctly (a GT resampled onto it correlated at 0.21).
HOOK = r"""
'''Partition masks for evaluation/tools/eval_phantom.py (one run only).'''
import importlib.abc
import importlib.machinery
import json
import os
import re
import sys

_OUT = os.environ.get("T1PREP_EVAL_GT_OUT")
_STASH = {}
_HEMI = re.compile(r"^(lh|rh)\.seg\.|_hemi-(L|R)_seg\.")
_EPS = 1e-4


def _note(name, payload):
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, name), "w") as fh:
        json.dump(payload, fh, indent=1)


def _patch_partition(module):
    original = module.get_partition

    def get_partition(p0_large, atlas, guard_atlas=None):
        result = original(p0_large, atlas, guard_atlas)
        try:
            import nibabel as nib
            import numpy as np

            p0 = np.clip(np.asarray(p0_large.dataobj, np.float32), 0, 3)
            marked = np.where(p0 > 0, p0 - _EPS, p0).astype(np.float32)
            parts = original(nib.Nifti1Image(marked, p0_large.affine, p0_large.header),
                             atlas, guard_atlas)
            info = {}
            for side, orig, mark in zip(("lh", "rh"), result, parts):
                fill = mark == 3.0
                excl = (mark == 1.0) & (marked > 1.0)
                # Where the original run filled tissue that was not pure WM, the
                # marked run must have filled it too -- the check that the
                # 1e-4 shift left the partition alone.
                ref = (orig == 3.0) & (p0 < 3.0)
                info[side] = {
                    "fill_voxels": int(fill.sum()),
                    "excluded_voxels": int(excl.sum()),
                    "fill_agreement": float((fill & ref).sum() / max(ref.sum(), 1)),
                }
                _STASH[side] = {"fill": fill.astype(np.float32),
                                "excl": excl.astype(np.float32)}
            _note("gt_hook.json", info)
        except Exception as exc:
            _note("gt_hook_error.json", {"error": repr(exc)})
        return result

    module.get_partition = get_partition


def _patch_utils(module):
    original = module.resample_and_save_nifti

    def resample_and_save_nifti(nifti_obj, grid, affine, header, out_name, *args, **kw):
        out = original(nifti_obj, grid, affine, header, out_name, *args, **kw)
        base = os.path.basename(str(out_name))
        match = _HEMI.search(base)
        if match:
            side = "lh" if (match.group(1) == "lh" or match.group(2) == "L") else "rh"
            import nibabel as nib

            for kind, data in _STASH.pop(side, {}).items():
                name = (base.replace(".seg.", "." + kind + ".", 1) if match.group(1)
                        else base.replace("_seg.", "_" + kind + ".", 1))
                img = nib.Nifti1Image(data, nifti_obj.affine, nifti_obj.header)
                target = os.path.join(_OUT, name)
                original(img, grid, affine, header, target, *args, **kw)
        return out

    module.resample_and_save_nifti = resample_and_save_nifti


_TARGETS = {"t1prep._partition": _patch_partition, "t1prep.utils": _patch_utils}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module, _run=run, _patch=_TARGETS[name]):
            _run(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if _OUT:
    sys.meta_path.insert(0, _Finder())
"""


def build_gt_hemispheres(work: Path, gt_path: Path) -> dict:
    """Ground-truth hemispheres on the grid of the pipeline's ones.

    The pipeline's ``?h.seg`` volumes are world-aligned with the native image,
    so the ground truth resamples onto them directly.  Inside the fills the
    value is WM, where the partition cleared tissue it is CSF, and elsewhere
    it is the ground-truth label (WMH as WM, background as CSF).
    """
    import nibabel as nib
    from scipy import ndimage as ndi

    gt = nib.load(str(gt_path))
    src, dst = work / "t1prep" / "mri", work / "gt_arm" / "mri"
    info = {}
    for hemi in ("lh", "rh"):
        seg = _one(f"{src}/{hemi}.seg.*.nii*")
        fill = _one(f"{dst}/{hemi}.fill.*.nii*")
        excl = _one(f"{dst}/{hemi}.excl.*.nii*")
        if not (seg and fill and excl):
            raise SystemExit(f"partition masks for {hemi} missing under {dst}")
        h_img = nib.load(seg)
        H = np.asarray(h_img.dataobj, dtype=np.float32)
        G = np.clip(_on_grid(gt, h_img), 1, 3)
        F = _on_grid(nib.load(fill), h_img) > 0.5
        X = _on_grid(nib.load(excl), h_img) > 0.5
        zooms = h_img.header.get_zooms()[:3]
        # Confine to this hemisphere's tissue plus a 2 mm margin, so that
        # ground-truth GM across the midline cannot leak in.
        region = ndi.binary_dilation(H > 1.05, iterations=round(2.0 / min(zooms)))
        out = np.where(region & ~X, G, 1.0)
        out[F] = 3.0
        tissue = (H > 1.5) & ~F
        info[hemi] = {
            "corr_with_pipeline": float(np.corrcoef(H[tissue], G[tissue])[0, 1]),
            "fill_frac": float(F.mean()),
            "excluded_frac": float(X.mean()),
        }
        nib.save(
            nib.Nifti1Image(out.astype(np.float32), h_img.affine, h_img.header),
            str(dst / Path(seg).name),
        )
    (dst / "gt_hemispheres.json").write_text(json.dumps(info, indent=1))
    return info


def _defaults() -> dict[str, str]:
    """``T1Prep_defaults.txt`` as a dict, so both arms use the same settings."""
    out = {}
    for line in (REPO / "T1Prep_defaults.txt").read_text().splitlines():
        m = re.match(r"^(\w+)=([^\s#]+)", line)
        if m:
            out[m.group(1)] = m.group(2).strip("'\"")
    return out


def _surface_arm(work: Path, bname: str, env: dict) -> None:
    """Surface pipeline on the ground-truth hemispheres, both sides at once."""
    sys.path.insert(0, str(REPO / "src"))
    from t1prep.utils import DATA_PATH_T1PREP

    d = _defaults()
    gt = work / "gt_arm"
    (gt / "surf").mkdir(parents=True, exist_ok=True)
    (gt / "report").mkdir(parents=True, exist_ok=True)
    cmds = []
    for side in ("left", "right"):
        cmd = [
            sys.executable,
            "-m",
            "t1prep.surface_estimation",
            "--bname",
            bname,
            "--side",
            side,
            "--mri-dir",
            str(gt / "mri"),
            "--surf-dir",
            str(gt / "surf"),
            "--estimate-spherereg",
            d.get("estimate_spherereg", "1"),
            "--thickness-method",
            d.get("thickness_method", "3"),
            "--save-pial-white",
            d.get("save_pial_white", "1"),
            "--pre-fwhm",
            d.get("pre_fwhm", "2"),
            "--median-filter",
            d.get("median_filter", "2"),
            "--vessel",
            d.get("vessel", "1"),
            "--amap",
            d.get("use_amap", "0"),
            "--correct-folding",
            d.get("correct_folding", "1"),
            "--multi",
            "0",
            "--nii-ext",
            d.get("nii_ext", "nii"),
            "--names-tsv",
            str(DATA_PATH_T1PREP / "Names.tsv"),
            "--bids-naming",
            "0",
            "--report-log",
            str(gt / "report" / f"surface_{side}.log"),
            "--surf-templates-dir",
            str(DATA_PATH_T1PREP / "templates_surfaces_32k"),
            "--atlas-templates-dir",
            str(DATA_PATH_T1PREP / "atlases_surfaces_32k"),
            "--atlas-surf",
            "'aparc_DK40.freesurfer'",
        ]
        cmds.append((side, cmd))
    with contextlib.ExitStack() as stack:
        procs = []
        for side, cmd in cmds:
            log = stack.enter_context(open(gt / "report" / f"surface_{side}.out", "w"))
            procs.append(
                (
                    side,
                    subprocess.Popen(
                        cmd, stdout=log, stderr=subprocess.STDOUT, env=env
                    ),
                )
            )
        codes = [(side, proc.wait()) for side, proc in procs]
    for side, code in codes:
        if code:
            raise SystemExit(
                f"ground-truth surface arm failed ({side}), see {gt / 'report'}"
            )


def run(
    work: Path,
    t1_path: Path,
    gt_path: Path,
    t1prep: str,
    device: str | None,
    extra: list[str],
) -> None:
    """T1Prep on the phantom, with the ground-truth arm, into ``work``.

    Only what an earlier run left in ``work`` is cleared first -- the
    directories and files this function writes -- so a stale output can never
    be scored, and nothing else in ``work`` is touched.
    """
    work.mkdir(parents=True, exist_ok=True)
    for sub in ("t1prep", "gt_arm", "hook"):
        shutil.rmtree(work / sub, ignore_errors=True)
    for name in (RESULTS_NAME, "run_info.json", "t1prep.log"):
        (work / name).unlink(missing_ok=True)
    hook_dir = work / "hook"
    hook_dir.mkdir(exist_ok=True)
    (hook_dir / "sitecustomize.py").write_text(HOOK)
    gt_mri = work / "gt_arm" / "mri"
    gt_mri.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(hook_dir), str(REPO / "src"), env.get("PYTHONPATH")) if p
    )
    env["T1PREP_EVAL_GT_OUT"] = str(gt_mri)
    if device:
        env["T1PREP_DEVICE"] = device

    cmd = [t1prep, "--out-dir", str(work / "t1prep"), "--lesions", *extra, str(t1_path)]
    print("running:", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    with open(work / "t1prep.log", "w") as log:
        code = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False
        ).returncode
    t_pipeline = time.perf_counter() - t0
    if code:
        raise SystemExit(f"T1Prep exited with {code}, see {work / 't1prep.log'}")
    for err in gt_mri.glob("gt_hook_error.json"):
        raise SystemExit(f"partition hook failed: {err.read_text()}")
    if not list(gt_mri.glob("*h.fill.*")):
        raise SystemExit(
            "the partition masks were not written: was surface "
            "estimation switched off, or the hook not loaded?"
        )
    build_gt_hemispheres(work, gt_path)

    # The ground-truth arm must not see the hook; it has nothing to patch.
    env_arm = {k: v for k, v in env.items() if not k.startswith("T1PREP_EVAL_")}
    env_arm["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH")) if p
    )
    t1 = time.perf_counter()
    bname = re.sub(r"\.nii(\.gz)?$", "", t1_path.name)
    _surface_arm(work, bname, env_arm)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from t1prep._device import resolve_device; "
                "print(resolve_device()[0].type)"
            ),
        ],
        capture_output=True,
        text=True,
        env=env_arm,
        check=False,
    )
    info = {
        "t1prep": t1prep,
        "extra_args": extra,
        "device": probe.stdout.strip() or device or "unknown",
        "pipeline_seconds": round(t_pipeline, 1),
        "gt_arm_seconds": round(time.perf_counter() - t1, 1),
        "started": (
            datetime.datetime.now().astimezone()
            - datetime.timedelta(seconds=t_pipeline)
        ).isoformat(timespec="seconds"),
    }
    (work / "run_info.json").write_text(json.dumps(info, indent=1))


# ---------------------------------------------------------------------------
# WMH over a set of simulations
# ---------------------------------------------------------------------------


def discover_sims(sims: Path) -> list[dict]:
    """Simulated T1w images and the ground truth each was rendered from.

    mri_simulate names the label after the WMH grade only, so
    ``…desc-snr50Rf90T4Wmh4_T1w`` pairs with ``…desc-Wmh4Clean_dseg`` and an
    image without ``Wmh`` in its description with ``…desc-Clean_dseg``.
    """
    cases = []
    for t1 in sorted(sims.glob("*_T1w.nii*")):
        m = re.match(r"(.*desc-)([^_]+)_T1w\.nii(\.gz)?$", t1.name)
        if not m:
            continue
        grade = re.search(r"Wmh\d+", m.group(2))
        gt = sims / f"{m.group(1)}{grade.group(0) if grade else ''}Clean_dseg.nii.gz"
        if gt.exists():
            bname = re.sub(r"_T1w\.nii(\.gz)?$", "", t1.name)
            cases.append({"desc": m.group(2), "t1": t1, "gt": gt, "bname": bname})
    return cases


def run_wmh(work: Path, cases: list[dict], t1prep: str, extra: list[str]) -> None:
    """T1Prep (volumes and lesions only) on every simulation in one call."""
    work.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(work / "t1prep", ignore_errors=True)
    cmd = [t1prep, "--out-dir", str(work / "t1prep"), "--no-surf", "--lesions", *extra]
    cmd += [str(c["t1"]) for c in cases]
    print("running:", " ".join(cmd[:6]), f"... ({len(cases)} images)", flush=True)
    with open(work / "t1prep.log", "w") as log:
        code = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, check=False
        ).returncode
    if code:
        raise SystemExit(f"T1Prep exited with {code}, see {work / 't1prep.log'}")


def score_wmh(work: Path, cases: list[dict]) -> dict:
    """Per-image WMH scores and their summary; writes phantom_results.json."""
    import nibabel as nib

    mri, report = work / "t1prep" / "mri", work / "t1prep" / "report"
    rows, clean_fp, clean_flagged, inv = [], [], [], {}
    for c in cases:
        p4 = _one(f"{mri}/p4{c['bname']}.nii*")
        if not p4:
            inv[f"{c['desc']}: p4 written"] = False
            continue
        gt = nib.load(str(c["gt"]))
        label = np.asarray(gt.dataobj, dtype=np.float32)
        img = _on_grid(nib.load(p4), gt)
        inv[f"{c['desc']}: p4 within [0, 1]"] = bool(
            np.isfinite(img).all() and img.min() >= -1e-4 and img.max() <= 1 + 1e-4
        )
        w = wmh_scores(label, img, float(np.prod(gt.header.get_zooms()[:3])))
        # exact name: "desc-snr25" is also a prefix of "desc-snr25Rf45T4Wmh2"
        rep = _one(f"{report}/log_{c['bname']}.json")
        if rep:
            sm = json.loads(Path(rep).read_text()).get("subjectmeasures", {})
            w["report_ml"] = float(sm.get("vol_WMH", {}).get("value", float("nan")))
        w["desc"] = c["desc"]
        if w["lesions"]:
            rows.append(w)
        else:
            clean_fp.append(w["pred_ml"])
            clean_flagged.append(w["flagged_ml"])
    found = sum(r["found"] for r in rows)
    n = sum(r["lesions"] for r in rows)
    vol_key = "report_ml" if all("report_ml" in r for r in rows) else "pred_ml"
    metrics = {
        "f1": float(np.mean([r["f1"] for r in rows])) if rows else float("nan"),
        "found_frac": found / n if n else float("nan"),
        "false_clusters": float(sum(r["false_clusters"] for r in rows)),
        "dice_t050": float(np.mean([r["dice_t050"] for r in rows]))
        if rows
        else float("nan"),
        "soft_dice": float(np.mean([r["soft_dice"] for r in rows]))
        if rows
        else float("nan"),
        "vol_abs_err_ml": (
            float(np.mean([abs(r[vol_key] - r["gt_ml"]) for r in rows]))
            if rows
            else float("nan")
        ),
        "clean_fp_ml": float(np.mean(clean_fp)) if clean_fp else float("nan"),
        "precision": (
            float(np.mean([r["precision"] for r in rows])) if rows else float("nan")
        ),
        "clean_flagged_ml": (
            float(np.mean(clean_flagged)) if clean_flagged else float("nan")
        ),
    }
    results = {
        "kind": "wmh",
        "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": _versions(),
        "git": _git_state(),
        "model_hash": _model_hash(),
        "run": {"images": [c["desc"] for c in cases]},
        "metrics": metrics,
        "details": {
            r["desc"]: (
                f"found {r['found']}/{r['lesions']}, false {r['false_clusters']},"
                f" Dice@0.5 {r['dice_t050']:.3f}, soft {r['soft_dice']:.3f},"
                f" vol {r.get(vol_key, r['pred_ml']):.2f} (true {r['gt_ml']:.2f}) ml,"
                f" flagged {r['flagged_ml']:.2f} ml, precision {r['precision']:.2f}"
            )
            for r in rows
        },
        "invariants": inv,
    }
    if clean_fp:
        results["details"]["WMH-free images, mean p4 volume"] = (
            f"{np.mean(clean_fp):.3f} ml"
        )
        results["details"]["WMH-free images, mean flagged volume"] = (
            f"{np.mean(clean_flagged):.3f} ml"
        )
    (work / RESULTS_NAME).write_text(json.dumps(results, indent=1, default=float))
    return results


# ---------------------------------------------------------------------------
# Reporting, pinning, checking
# ---------------------------------------------------------------------------


def _load_results(work: Path) -> dict:
    path = work / RESULTS_NAME
    if not path.exists():
        raise SystemExit(f"{path} not found: run 'score' first")
    return json.loads(path.read_text())


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def print_results(res: dict) -> None:
    """The metrics and details of one run as a table."""
    versions, git = res["versions"], res["git"]
    dirty = " (dirty)" if git.get("dirty") else ""
    print(
        f"\nT1Prep {versions.get('t1prep')}  cat-surf {versions.get('cat_surf')}"
        f"  git {git.get('commit')}{dirty}  models {res.get('model_hash')}"
    )
    for name, value in res["metrics"].items():
        print(f"  {name:28s} {_fmt(value)}")
    print("details:")
    for name, value in res["details"].items():
        print(f"  {name:28s} {_fmt(value)}")
    print("invariants:")
    for name, ok in res["invariants"].items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")


def append_history(res: dict, path: Path) -> None:
    """One CSV row per scored run, for the long view."""
    import csv

    names = list(METRICS)
    row = {
        "scored": res["scored"],
        "t1prep": res["versions"].get("t1prep"),
        "cat_surf": res["versions"].get("cat_surf"),
        "commit": res["git"].get("commit"),
        "dirty": res["git"].get("dirty"),
        "models": res.get("model_hash"),
        "device": res.get("run", {}).get("device"),
    }
    row.update({k: res["metrics"].get(k) for k in names})
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def cmd_pin(works: list[Path], pinned: Path) -> int:
    runs = [_load_results(w) for w in works]
    bad = [w for w, r in zip(works, runs) if not all(r["invariants"].values())]
    if bad:
        raise SystemExit(f"refusing to pin runs that break an invariant: {bad}")
    hashes = {r.get("model_hash") for r in runs}
    if len(hashes) > 1:
        raise SystemExit("the runs used different model weights")
    doc = {
        "pinned": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "comment": "Characterization of T1Prep on the HR075 phantom; see "
        "evaluation/tools/eval_phantom.py. Re-pin after an intended change.",
        "model_hash": runs[0].get("model_hash"),
        "versions": runs[0]["versions"],
        "git": runs[0]["git"],
        "devices": [r.get("run", {}).get("device") for r in runs],
        "metrics": make_pins(runs),
    }
    pinned.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"pinned {len(doc['metrics'])} metrics from {len(runs)} run(s) -> {pinned}")
    for name, pin in doc["metrics"].items():
        print(
            f"  {name:28s} {pin['value']:>10.4g}  tol {pin['tol']:.3g}"
            f"  ({pin['direction']}, spread {pin['spread']:.3g})"
        )
    return 0


def cmd_check(work: Path, pinned: Path, strict: bool, ignore_models: bool) -> int:
    res = _load_results(work)
    if not pinned.exists():
        print(f"no pins at {pinned}: run 'pin' on a trusted run first")
        return 2
    doc = json.loads(pinned.read_text())
    status = 0
    print_results(res)
    failed_inv = [k for k, ok in res["invariants"].items() if not ok]
    if (
        doc.get("model_hash")
        and res.get("model_hash") != doc["model_hash"]
        and not ignore_models
    ):
        print(
            f"\nmodel weights differ from the pinned ones ({res.get('model_hash')}"
            f" vs {doc['model_hash']}): the pins do not apply -- re-pin, or pass"
            " --ignore-models"
        )
        return 3
    print(
        f"\nagainst {pinned.name} (pinned {doc['pinned']}, T1Prep "
        f"{doc['versions'].get('t1prep')}, git {doc['git'].get('commit')}):"
    )
    counts = {"ok": 0, "improved": 0, "REGRESSED": 0}
    for name, pin in doc["metrics"].items():
        value = res["metrics"].get(name)
        verdict = compare(value, pin, strict)
        counts[verdict] += 1
        if verdict != "ok":
            print(
                f"  {verdict:9s} {name:28s} {_fmt(value):>10s}"
                f"  pinned {pin['value']:.4g} +/- {pin['tol']:.3g} ({pin['direction']})"
            )
    missing = [n for n in doc["metrics"] if n not in res["metrics"]]
    print(
        f"  {counts['ok']} ok, {counts['improved']} improved,"
        f" {counts['REGRESSED']} regressed"
        + (f", missing: {missing}" if missing else "")
    )
    if failed_inv:
        print(f"  invariants failed: {failed_inv}")
    if counts["REGRESSED"] or failed_inv:
        status = 1
    if counts["improved"]:
        print("  improvements: re-pin to lock them in")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="See evaluation/PHANTOM.md for what each metric means.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, work_many=False):
        if work_many:
            p.add_argument(
                "--work",
                type=Path,
                nargs="+",
                required=True,
                help="scored run directories",
            )
        else:
            p.add_argument("--work", type=Path, required=True, help="run directory")
        p.add_argument("--t1", type=Path, default=T1_DEFAULT, help="simulated T1w")
        p.add_argument("--gt", type=Path, default=GT_DEFAULT, help="ground-truth label")
        p.add_argument(
            "--pinned", type=Path, default=PINNED_DEFAULT, help="pinned scalars"
        )

    p = sub.add_parser("run", help="T1Prep + ground-truth arm + score")
    common(p)
    p.add_argument(
        "--t1prep",
        default=str(REPO / "scripts" / "T1Prep"),
        help="T1Prep launcher (default: this checkout's scripts/T1Prep)",
    )
    p.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default=None,
        help="set T1PREP_DEVICE for the run",
    )
    p.add_argument(
        "--history",
        action="store_true",
        help=f"append the scores to {HISTORY_DEFAULT.relative_to(REPO)}",
    )
    p.add_argument(
        "--check", action="store_true", help="check against the pins afterwards"
    )
    p.add_argument(
        "extra", nargs=argparse.REMAINDER, help="after '--': further T1Prep options"
    )

    p = sub.add_parser("score", help="score a finished run")
    common(p)
    p.add_argument("--history", action="store_true", help="append to the history CSV")

    p = sub.add_parser("pin", help="pin the scalars of one or more scored runs")
    common(p, work_many=True)

    p = sub.add_parser("wmh", help="WMH over a directory of simulations (+ score)")
    p.add_argument("--work", type=Path, required=True, help="run directory")
    p.add_argument(
        "--sims",
        type=Path,
        default=os.environ.get("T1PREP_PHANTOM_SIMS"),
        help="mri_simulate derivatives folder (default: $T1PREP_PHANTOM_SIMS)",
    )
    p.add_argument("--t1prep", default=str(REPO / "scripts" / "T1Prep"))
    p.add_argument("--no-run", action="store_true", help="only score an earlier run")
    p.add_argument("--check", action="store_true", help="check against the WMH pins")
    p.add_argument("--pinned", type=Path, default=WMH_PINNED_DEFAULT)
    p.add_argument("extra", nargs=argparse.REMAINDER, help="after '--': T1Prep options")

    p = sub.add_parser("check", help="compare a scored run with the pins")
    common(p)
    p.add_argument(
        "--strict",
        action="store_true",
        help="fail on improvements too (pure characterization)",
    )
    p.add_argument(
        "--ignore-models",
        action="store_true",
        help="compare even if the model weights changed",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        extra = [a for a in args.extra if a != "--"]
        run(args.work, args.t1, args.gt, args.t1prep, args.device, extra)
        res = score(args.work, args.t1, args.gt)
        print_results(res)
        if args.history:
            append_history(res, HISTORY_DEFAULT)
        if args.check:
            return cmd_check(args.work, args.pinned, False, False)
        return 0
    if args.command == "score":
        res = score(args.work, args.t1, args.gt)
        print_results(res)
        if args.history:
            append_history(res, HISTORY_DEFAULT)
        return 0
    if args.command == "wmh":
        if not args.sims or not Path(args.sims).is_dir():
            raise SystemExit(
                "--sims (or $T1PREP_PHANTOM_SIMS) must name the simulations"
            )
        cases = discover_sims(Path(args.sims))
        if not cases:
            raise SystemExit(f"no simulated T1w with ground truth under {args.sims}")
        if not args.no_run:
            run_wmh(args.work, cases, args.t1prep, [a for a in args.extra if a != "--"])
        res = score_wmh(args.work, cases)
        print_results(res)
        if args.check:
            return cmd_check(args.work, args.pinned, False, False)
        return 0
    if args.command == "pin":
        return cmd_pin(args.work, _pinned_for(args.work[0], args.pinned))
    return cmd_check(
        args.work, _pinned_for(args.work, args.pinned), args.strict, args.ignore_models
    )


def _pinned_for(work: Path, pinned: Path) -> Path:
    """The WMH pins for a ``wmh`` run unless another file was named."""
    path = work / RESULTS_NAME
    is_wmh = path.exists() and json.loads(path.read_text()).get("kind") == "wmh"
    return WMH_PINNED_DEFAULT if pinned == PINNED_DEFAULT and is_wmh else pinned


if __name__ == "__main__":
    sys.exit(main())
