#!/usr/bin/env python3
"""Refit the WMH probability of ``t1prep._lesions`` on simulated brains.

``_lesions.WMH_CALIBRATION`` and the lesion operating point
(``WMH_THRESHOLD``, ``WMH_MIN_LESION_MM3``) were fitted with this script on
mri_simulate renderings of one anatomy.  Rerun it when simulations of other
anatomies exist, and paste what it prints.

``dump``  Segment every simulation (``t1prep.segment --lesions``) and write,
          through the same reslice as ``p4``, the evidence the probability is
          computed from: the lesion signal, the WMH atlas, the candidate mask
          and the deep grey nuclei.
``fit``   Logistic regression of the true WMH fraction on the high-passed
          signal and the log prior, validated on held-out WMH grades and noise
          levels, then a table of operating points (threshold x minimum size):
          lesions found, false clusters, voxel precision, false volume on
          WMH-free images, and the bias of the lesion volume.

    python evaluation/tools/fit_wmh_calibration.py dump --sims DIR --work DIR
    python evaluation/tools/fit_wmh_calibration.py fit --sims DIR --work DIR
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_phantom", Path(__file__).with_name("eval_phantom.py")
)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)

#: sitecustomize for ``dump``: stashes the evidence when ``lesion_signal``
#: runs and writes it next to ``p4`` through the identical reslice.
HOOK = r"""
import importlib.abc
import importlib.machinery
import os
import sys

_OUT = os.environ.get("T1PREP_WMH_DUMP")
_STASH = {}


def _patch_lesions(m):
    orig = m.lesion_signal

    def lesion_signal(*a, **k):
        ev = orig(*a, **k)
        import numpy as np

        _STASH["diff"] = np.asarray(ev["diff"].dataobj, np.float32).copy()
        _STASH["prior"] = np.asarray(ev["prior"], np.float32)
        _STASH["cand"] = ev["candidates"].astype(np.float32)
        _STASH["deepgm"] = m._deep_gm(a[0], a[1], a[3], a[9]).astype(np.float32)
        return ev

    m.lesion_signal = lesion_signal


def _patch_utils(m):
    orig = m.resample_and_save_nifti

    def rs(nifti_obj, grid, affine, header, out_name, *a, **k):
        out = orig(nifti_obj, grid, affine, header, out_name, *a, **k)
        if os.path.basename(str(out_name)).startswith("p4") and _STASH:
            import nibabel as nib

            os.makedirs(_OUT, exist_ok=True)
            for key, data in list(_STASH.items()):
                img = nib.Nifti1Image(data, nifti_obj.affine, nifti_obj.header)
                target = os.path.join(_OUT, "feat_" + key + ".nii")
                orig(img, grid, affine, header, target, *a, **k)
            _STASH.clear()
        return out

    m.resample_and_save_nifti = rs


_TARGETS = {"t1prep._lesions": _patch_lesions, "t1prep.utils": _patch_utils}


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


def dump(sims: Path, work: Path, jobs: int) -> None:
    """Segment every simulation and write its WMH evidence."""
    cases = ep.discover_sims(sims)
    (work / "hook").mkdir(parents=True, exist_ok=True)
    (work / "hook" / "sitecustomize.py").write_text(HOOK)

    def one(c):
        out = work / c["desc"]
        for sub in ("mri", "report", "label"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(work / "hook"), str(REPO / "src")])
        env["T1PREP_WMH_DUMP"] = str(out / "feat")
        cmd = [
            sys.executable,
            "-m",
            "t1prep.segment",
            "--lesions",
            "--seed",
            "0",
            "--vessel",
            "1",
            "--input",
            str(c["t1"]),
            "--mri-dir",
            str(out / "mri"),
            "--report-dir",
            str(out / "report"),
            "--label-dir",
            str(out / "label"),
        ]
        with open(out / "segment.log", "w") as log:
            code = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False
            ).returncode
        return c["desc"], code

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for desc, code in pool.map(one, cases):
            print(f"{desc}: {'ok' if code == 0 else f'exit {code}'}", flush=True)


def load(sims: Path, work: Path) -> list[dict]:
    """Features and truth of every dumped case, on the truth's grid (cropped)."""
    import nibabel as nib
    from scipy import ndimage as ndi

    from t1prep import _lesions

    cases = []
    for c in ep.discover_sims(sims):
        feat = work / c["desc"] / "feat"
        if not (feat / "feat_diff.nii").exists():
            continue
        gt = nib.load(str(c["gt"]))
        zooms = gt.header.get_zooms()[:3]

        def grid(name, feat=feat, gt=gt):
            return ep._on_grid(nib.load(str(feat / f"feat_{name}.nii")), gt)

        cand0 = grid("cand") > 0.5
        box = ndi.find_objects(cand0.astype(np.uint8))[0]
        box = tuple(slice(max(b.start - 8, 0), b.stop + 8) for b in box)
        label = np.asarray(gt.dataobj, np.float32)[box]
        cand0 = cand0[box]
        hp = _lesions._highpass(
            grid("diff")[box], cand0, zooms, ref_max=_lesions.WMH_HP_REF_MAX
        )
        cases.append(
            {
                "desc": c["desc"],
                "vx": float(np.prod(zooms)),
                "label": label,
                "frac": np.clip(label - 3, 0, 1),
                "hp": hp,
                "prior": grid("prior")[box],
                "cand": cand0 & ~(grid("deepgm")[box] > 0.5),
            }
        )
    return cases


def design(c):
    m = c["cand"]
    from t1prep._lesions import WMH_PRIOR_EPS

    return np.stack(
        [np.ones(m.sum()), c["hp"][m], np.log(c["prior"][m] + WMH_PRIOR_EPS)], 1
    )


def fit(cases: list[dict], n_neg: int = 300000) -> np.ndarray:
    """Logistic regression, soft targets, negatives subsampled and reweighted."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    X, Y, W = [], [], []
    for c in cases:
        f, y = design(c), c["frac"][c["cand"]]
        pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y == 0)
        take = rng.choice(neg, min(n_neg, neg.size), replace=False)
        X += [f[pos], f[take]]
        Y += [y[pos], y[take]]
        W += [np.ones(pos.size), np.full(take.size, neg.size / max(take.size, 1))]
    X, Y, W = np.concatenate(X), np.concatenate(Y), np.concatenate(W)

    def loss(beta):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        value = (W * (np.logaddexp(0, z) - Y * z)).sum() / W.sum()
        return value, X.T @ (W * (p - Y)) / W.sum()

    return minimize(loss, np.array([-4.0, 20.0, 1.0]), jac=True, method="L-BFGS-B").x


def operating_point(cases, beta, threshold, min_mm3):
    """Lesions found, false clusters, precision, false and total volume."""
    from scipy import ndimage as ndi

    from t1prep._lesions import lesion_probability, select_lesions

    found = n = false = 0
    prec, clean, bias = [], [], []
    for c in cases:
        prob = lesion_probability(c["hp"], c["prior"], c["cand"], calibration=beta)
        _, mask = select_lesions(
            prob, c["vx"], threshold=threshold, min_lesion_mm3=min_mm3
        )
        comp, k = ndi.label(c["label"] > 3.5)
        lab, j = ndi.label(mask, structure=np.ones((3, 3, 3)))
        if j:
            false += int(
                j - (ndi.maximum(c["frac"] > 0, lab, np.arange(1, j + 1)) > 0).sum()
            )
        if k:
            n += k
            found += int((ndi.maximum(mask, comp, np.arange(1, k + 1)) > 0).sum())
            prec.append(((mask & (c["frac"] > 0)).sum()) / max(mask.sum(), 1))
            bias.append((mask.sum() - c["frac"].sum()) * c["vx"] / 1000)
        else:
            clean.append(mask.sum() * c["vx"] / 1000)
    return (
        found,
        n,
        false,
        np.mean(prec),
        np.mean(clean) if clean else np.nan,
        np.mean(bias),
    )


def _grade(desc: str) -> str:
    m = re.search(r"Wmh\d+", desc)
    return m.group(0) if m else ""


def _snr(desc: str) -> str:
    m = re.search(r"snr\d+", desc)
    return m.group(0) if m else ""


def run_fit(sims: Path, work: Path) -> None:
    cases = load(sims, work)
    print(f"{len(cases)} cases: {[c['desc'] for c in cases]}")
    for name, key in (("WMH grade", _grade), ("SNR", _snr)):
        for held in sorted({key(c["desc"]) for c in cases} - {""}):
            train = [c for c in cases if key(c["desc"]) != held]
            print(f"  held out {name} {held}: beta {np.round(fit(train), 3)}")
    beta = fit(cases)
    print(f"\nall cases: WMH_CALIBRATION = {tuple(round(float(b), 3) for b in beta)}")
    print(
        "threshold  min mm3   found     false  precision  WMH-free ml  volume bias ml"
    )
    for threshold in (0.05, 0.1, 0.2, 0.3):
        for min_mm3 in (10, 30, 62.5):
            f, n, fp, pr, cl, vb = operating_point(cases, beta, threshold, min_mm3)
            print(
                f"   {threshold:4.2f}    {min_mm3:6.1f}  {f:4d}/{n:<4d} {fp:5d}"
                f"    {pr:5.2f}      {cl:6.2f}       {vb:+6.2f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dump", "fit"):
        p = sub.add_parser(name)
        p.add_argument(
            "--sims",
            type=Path,
            default=ep.DATA,
            help="mri_simulate folder (default: the repo's)",
        )
        p.add_argument("--work", type=Path, required=True, help="dump directory")
        if name == "dump":
            p.add_argument("--jobs", type=int, default=2, help="parallel segmentations")
    args = parser.parse_args(argv)
    if args.command == "dump":
        dump(args.sims, args.work, args.jobs)
    else:
        run_fit(args.sims, args.work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
