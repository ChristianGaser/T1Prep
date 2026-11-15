#!/usr/bin/env python3
"""Register the subset to MNI with ANTs and warp the manual labels.

    run_ants_antspy.py --work <work> --mindboggle <mindboggle-101/data> <config> [workers]

``--work`` holds ``mni_fixed.nii.gz`` (the fixed image) and gets ``moving/``
and ``ants20_<config>/``; the subjects are read from ``--subject-file``
(default ``<work>/subset20.txt``, else ``evaluation/data/subset20.txt``).
Both folders can also be given as ``SP`` and ``MB`` in the environment, as
for ``run_ants_batch.sh``.

Fixed image: T1Prep's own MNI152NLin2009cAsym brain template on the grid its
``y_`` writes, so both pipelines land in the same space on the same grid.
Moving image: the bias-corrected native T1 masked by T1Prep's own brain
segmentation, so ANTs gets the same input quality and only the registration
differs.

Two configurations are run because ANTsPy's default ``SyN`` sets
``reg_iterations=(40, 20, 0)`` -- no iterations at the finest level -- which
would understate ANTs.  ``SyNCC`` is the cross-correlation configuration ANTs
is known for and the one behind its Klein-2009 result.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

# Set by main() from the command line (they used to be two paths of the
# machine the evaluation was run on, one of them a scratch folder long gone)
# and handed to the worker processes with each job.
SP = os.environ.get("SP", "")
MB = os.environ.get("MB", "")

CONFIGS = {
    "syncc":  dict(type_of_transform="SyNCC"),
    "syn100": dict(type_of_transform="SyN", reg_iterations=(100, 70, 50, 20)),
    # fMRIPrep/sMRIPrep's own antsRegistration settings, from niworkflows'
    # t1w-mni_registration_precise_000.json: Rigid+Affine (Mattes, 56 bins,
    # 25% regular sampling, 100x100) then SyN with CC radius 4, 100x70x50x20,
    # transform parameters [0.1, 3.0, 0.0].
    "fmriprep": dict(
        type_of_transform="SyN",
        syn_metric="CC", syn_sampling=4,
        reg_iterations=(100, 70, 50, 20),
        grad_step=0.1, flow_sigma=3.0, total_sigma=0.0,
        aff_metric="mattes", aff_sampling=56, aff_random_sampling_rate=0.25,
        aff_iterations=(100, 100), aff_shrink_factors=(2, 1),
        aff_smoothing_sigmas=(2, 1),
    ),
}


def one(args):
    subject, cfg, SP, MB = args
    import ants
    out = f"{SP}/ants20_{cfg}"
    os.makedirs(f"{out}/{subject}", exist_ok=True)
    dst = f"{out}/{subject}/labels.DKT31.manual.MNI152.nii.gz"
    if os.path.exists(dst):
        return f"{subject} [{cfg}]: cached"
    t0 = time.time()
    mri = f"{MB}/{subject}/mri"
    mov_path = f"{SP}/moving/{subject}.nii.gz"
    if not os.path.exists(mov_path):
        os.makedirs(f"{SP}/moving", exist_ok=True)
        m = nib.load(f"{mri}/mt1weighted.nii")
        p0 = np.asanyarray(nib.load(f"{mri}/p0t1weighted.nii").dataobj)
        brain = (np.asanyarray(m.dataobj) * (p0 > 0.5)).astype(np.float32)
        nib.save(nib.Nifti1Image(brain, m.affine), mov_path)

    fixed = ants.image_read(f"{SP}/mni_fixed.nii.gz")
    moving = ants.image_read(mov_path)
    reg = ants.registration(fixed=fixed, moving=moving, **CONFIGS[cfg])
    lab = ants.image_read(f"{MB}/{subject}/labels.DKT31.manual.nii.gz")
    warped = ants.apply_transforms(fixed=fixed, moving=lab,
                                   transformlist=reg["fwdtransforms"],
                                   interpolator="nearestNeighbor")
    ants.image_write(warped, dst)
    return f"{subject} [{cfg}]: {time.time() - t0:.0f}s"


def main(argv=None):
    """Run one configuration over the subjects of the subset."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", choices=sorted(CONFIGS))
    parser.add_argument("workers", nargs="?", type=int, default=4)
    parser.add_argument("--work", default=SP,
                        help="work folder with mni_fixed.nii.gz (default: $SP)")
    parser.add_argument("--mindboggle", default=MB,
                        help="Mindboggle-101 data folder (default: $MB)")
    parser.add_argument("--subject-file", default=None,
                        help="subjects, one per line (default: <work>/subset20.txt, "
                             "else evaluation/data/subset20.txt)")
    args = parser.parse_args(argv)
    if not args.work or not args.mindboggle:
        parser.error("give --work and --mindboggle (or SP and MB in the environment)")
    if not os.path.isfile(f"{args.work}/mni_fixed.nii.gz"):
        parser.error(f"{args.work}/mni_fixed.nii.gz not found")
    subject_file = args.subject_file or f"{args.work}/subset20.txt"
    if not os.path.isfile(subject_file):
        subject_file = str(Path(__file__).resolve().parents[1] / "data" / "subset20.txt")
    subs = [s.strip() for s in open(subject_file) if s.strip()]
    t0 = time.time()
    jobs = [(s, args.config, args.work, args.mindboggle) for s in subs]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(one, jobs):
            print(r, flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
