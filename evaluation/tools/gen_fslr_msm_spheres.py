#!/usr/bin/env python3
"""Add the fsLR and msmsulc spheres to an existing T1Prep run.

Both are normally written only under ``--fmriprep``, which would also switch
the run to BIDS naming and AMAP segmentation.  Everything they need is in a
plain run, so they can be derived afterwards instead of reprocessing:

* ``?h.sphere.reg.fsLR`` -- the fsaverage-registered sphere carried into the
  fsLR frame by a fixed project-unproject.  No new registration; cheap
  (~1 s/hemisphere).  This is what ``run_newmsm.py`` regresses its affine
  against, mirroring sMRIPrep's ``sphere_reg_fsLR`` input.
* ``?h.sphere.reg.msm`` -- a second, independent Spherical Demons
  registration, onto the fsLR average.  The ``msm`` in the name is the BIDS
  ``desc-msmsulc`` entity, written so fMRIPrep finds this sphere and skips its
  own MSMSulc step; the algorithm is Spherical Demons, not MSM.
  ~40 s/hemisphere, so ``--fslr-only`` skips it when only preparing the inputs
  for an actual MSM run.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _one(job):
    subject, hemi, t1prep, data_dir, fslr_only = job
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from t1prep import fslr

    surf = Path(t1prep) / subject / "surf"
    regs = sorted(p for p in surf.glob(f"{hemi}.sphere.reg.*.gii")
                  if ".fsLR." not in p.name and ".msm." not in p.name)
    if not regs:
        return f"{subject} {hemi}: no sphere.reg"
    reg = regs[0]
    bname = reg.name[len(f"{hemi}.sphere.reg."):-len(".gii")]
    data = Path(data_dir)
    fslr_dir = data / "templates_surfaces_fsLR"
    fsavg_sphere = data / "templates_surfaces_32k" / f"{hemi}.sphere.freesurfer.gii"

    out_fslr = surf / f"{hemi}.sphere.reg.fsLR.{bname}.gii"
    try:
        if not out_fslr.exists():
            fslr.write_reg_sphere(
                sphere_reg_file=str(reg), out_file=str(out_fslr),
                fslr_templates_dir=str(fslr_dir),
                fsaverage_sphere_file=str(fsavg_sphere), fshemi=hemi)
        if not fslr_only:
            out_msm = surf / f"{hemi}.sphere.reg.msm.{bname}.gii"
            if not out_msm.exists():
                fslr.write_msm_sphere(
                    mid_surface_file=str(surf / f"{hemi}.central.{bname}.gii"),
                    reg_sphere_file=str(out_fslr), out_file=str(out_msm),
                    fslr_templates_dir=str(fslr_dir), fshemi=hemi,
                    sphere_file=str(surf / f"{hemi}.sphere.{bname}.gii"))
    except Exception as exc:
        return f"{subject} {hemi}: FAILED {type(exc).__name__}: {exc}"
    return None


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t1prep", required=True,
                    help="root holding one T1Prep output directory per subject")
    ap.add_argument("--data-dir", default=str(here / "src" / "t1prep" / "data"),
                    help="t1prep/data holding the surface templates")
    ap.add_argument("--subjects", nargs="+")
    ap.add_argument("--subject-file")
    ap.add_argument("--fslr-only", action="store_true",
                    help="skip the expensive msmsulc sphere")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)

    root = Path(a.t1prep)
    subjects = a.subjects or (
        [s.strip() for s in open(a.subject_file) if s.strip()]
        if a.subject_file else
        sorted(p.name for p in root.iterdir() if (p / "surf").is_dir()))

    jobs = [(s, h, a.t1prep, a.data_dir, a.fslr_only)
            for s in subjects for h in ("lh", "rh")]
    t0 = time.time()
    bad = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, r in enumerate(ex.map(_one, jobs), 1):
            if r:
                bad.append(r)
            if i % 10 == 0:
                print(f"{i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(jobs)} hemispheres in {time.time() - t0:.0f}s, "
          f"{len(bad)} failures")
    for b in bad[:10]:
        print(" ", b)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
