"""Register the subset to MNI with ANTs and warp the manual labels.

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
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, nibabel as nib

SP = "/private/tmp/claude-501/-Users-gaser-GitHub-T1Prep/d60c79ac-f20a-41db-974b-3e197cf0a946/scratchpad"
MB = "/Users/gaser/Downloads/mindboggle-101/data"

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
    subject, cfg = args
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


if __name__ == "__main__":
    cfg = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    subs = [s.strip() for s in open(f"{SP}/subset20.txt") if s.strip()]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, [(s, cfg) for s in subs]):
            print(r, flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)
