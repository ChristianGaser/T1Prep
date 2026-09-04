#!/usr/bin/env python3
"""Resample Mindboggle's own affine-MNI labels onto the evaluation grid.

Mindboggle ships ``labels.DKT31.manual.MNI152.nii.gz`` -- the same manual
labels carried into MNI by an affine alone.  That is the baseline every
published registration study reports, and having it on identical data with an
identical metric is what makes the cross-study comparison meaningful.  It is
resampled onto the grid T1Prep's ``y_`` writes so no resolution difference
enters the comparison (it changes Dice by ~0.003 either way).
"""
import argparse, glob, os
import numpy as np
import nibabel as nib


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mindboggle", required=True)
    ap.add_argument("--reference", required=True,
                    help="any subject's y_*.nii, defining the target grid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref = nib.load(a.reference)
    shape, aff = ref.shape[:3], ref.affine
    world = nib.affines.apply_affine(
        aff, np.indices(shape).reshape(3, -1).T)
    n = 0
    for src in sorted(glob.glob(
            f"{a.mindboggle}/*/labels.DKT31.manual.MNI152.nii.gz")):
        subject = src.split(os.sep)[-2]
        img = nib.load(src)
        arr = np.asanyarray(img.dataobj).astype(np.int32)
        vox = np.rint(nib.affines.apply_affine(
            np.linalg.inv(img.affine), world)).astype(int)
        ok = np.all((vox >= 0) & (vox < np.array(arr.shape)), axis=1)
        out = np.zeros(len(vox), np.int32)
        out[ok] = arr[vox[ok, 0], vox[ok, 1], vox[ok, 2]]
        os.makedirs(f"{a.out}/{subject}", exist_ok=True)
        nib.save(nib.Nifti1Image(out.reshape(shape).astype(np.int16), aff),
                 f"{a.out}/{subject}/labels.DKT31.manual.MNI152.nii.gz")
        n += 1
    print(f"resampled {n} subjects onto {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
