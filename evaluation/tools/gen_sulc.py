#!/usr/bin/env python3
"""Derive sulcal-depth and curvature maps from an existing T1Prep run.

MSM is driven by a feature map on the sphere, conventionally sulcal depth.
T1Prep writes ``?h.sulc`` only under ``--fmriprep``, which also switches the
run to BIDS naming and to AMAP segmentation -- so re-running with that flag
would produce differently named *and* differently segmented surfaces than the
ones already evaluated.  The features themselves need neither: they are a
property of the central surface, so they can be added afterwards.

This mirrors what ``surface_estimation`` does in its fMRIPrep branch:
``sulc`` is CAT-Surface curvature type 11 (inverted), ``curv`` is type 4
smoothed 5 mm (inverted), matching FreeSurfer's ``?h.sulc`` and ``?h.curv``.
"""
import argparse
import glob
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t1prep", required=True,
                    help="root holding one T1Prep output directory per subject")
    ap.add_argument("--subjects", nargs="+",
                    help="restrict to these subjects")
    ap.add_argument("--curv", action="store_true",
                    help="also write ?h.curv (mean curvature, 5 mm FWHM)")
    a = ap.parse_args()

    from cat_surf import cli as cs_cli

    subjects = sorted(p for p in os.listdir(a.t1prep)
                      if os.path.isdir(os.path.join(a.t1prep, p)))
    if a.subjects:
        subjects = [s for s in subjects if s in set(a.subjects)]

    n = 0
    for subject in subjects:
        surf = os.path.join(a.t1prep, subject, "surf")
        for hemi in ("lh", "rh"):
            mids = sorted(glob.glob(os.path.join(surf, f"{hemi}.central.*.gii")))
            if not mids:
                print(f"{subject} {hemi}: no central surface", file=sys.stderr)
                continue
            mid = mids[0]
            bname = os.path.basename(mid)[len(f"{hemi}.central."):-len(".gii")]
            cs_cli.surf_curvature(
                surface_file=mid,
                output_values_file=os.path.join(surf, f"{hemi}.sulc.{bname}"),
                curvtype=11, fwhm=0.0, use_abs_values=False, invert_values=True)
            if a.curv:
                cs_cli.surf_curvature(
                    surface_file=mid,
                    output_values_file=os.path.join(surf, f"{hemi}.curv.{bname}"),
                    curvtype=4, fwhm=5.0, use_abs_values=False,
                    invert_values=True)
            n += 1
        print(f"{subject}: sulc written", flush=True)
    print(f"\n{n} hemispheres processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
