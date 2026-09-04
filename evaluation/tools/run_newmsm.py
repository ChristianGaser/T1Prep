#!/usr/bin/env python3
"""Run newMSM as sMRIPrep runs MSMSulc, on T1Prep surfaces.

sMRIPrep's ``init_msm_sulc_wf`` does four things before it ever calls MSM,
and skipping any of them changes the result:

1. regress an affine from the subject's *native* sphere onto its
   fsLR-registered sphere (``wb_command -surface-affine-regression``);
2. apply that affine to the native sphere, giving ``sphere_rot``;
3. re-inflate the result to an exact radius-100 sphere
   (``-surface-modify-sphere``), because the affine leaves it oblong;
4. **invert** sulc (``-metric-math "-1 * var"``) -- MSMSulc drives on the
   negated map, so feeding FreeSurfer-convention sulc straight in registers
   gyri onto sulci.

Then::

    msm --conf=MSMSulcStrainFinalconf \\
        --inmesh=<sphere_rot> \\
        --refmesh=fsaverage.<L|R>_LR.spherical_std.164k_fs_LR.surf.gii \\
        --indata=<inverted sulc> \\
        --refdata=<L|R>.refsulc.164k_fs_LR.shape.gii \\
        --out=<lh.|rh.> --verbose

The config and reference files in ``msm_data/`` are copied verbatim from
sMRIPrep, so the only deliberate difference here is the binary: ``newmsm``
instead of ``msm``.  newMSM takes the same options (see the FSL newMSM guide);
its config additions -- HOCR, ``--triclique``, the strain moduli -- are
already what ``MSMSulcStrainFinalconf`` asks for.

Inputs come from a plain T1Prep run plus two helpers in this directory:
``gen_sulc.py`` (writes ``?h.sulc``) and ``gen_fslr_msm_spheres.py`` (writes
``?h.sphere.reg.fsLR``).  Neither needs ``--fmriprep``, which would also
switch the run to BIDS naming and AMAP segmentation.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MSM_DATA = HERE / "msm_data"


def regress_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares affine taking ``source`` vertices onto ``target``.

    The workbench equivalent is ``-surface-affine-regression``; both solve an
    unconstrained 4x4 fit over corresponding vertices, which is well posed
    here because the two spheres share a mesh.
    """
    src = np.hstack([source, np.ones((len(source), 1))])
    sol, *_ = np.linalg.lstsq(src, target, rcond=None)
    affine = np.eye(4)
    affine[:3, :3] = sol[:3].T
    affine[:3, 3] = sol[3]
    return affine


def prepare_sphere(sphere_file: Path, reg_file: Path, out_file: Path,
                   radius: float = 100.0) -> None:
    """Steps 0-2: regress the affine, apply it, re-sphere to a fixed radius."""
    import cat_surf

    sphere, faces = cat_surf.read_surface(str(sphere_file))
    target, _ = cat_surf.read_surface(str(reg_file))
    if len(sphere) != len(target):
        raise ValueError(
            f"{sphere_file.name} has {len(sphere)} vertices but "
            f"{reg_file.name} has {len(target)} - not the same subject")
    affine = regress_affine(np.asarray(sphere), np.asarray(target))
    moved = np.asarray(sphere) @ affine[:3, :3].T + affine[:3, 3]
    # -surface-modify-sphere: project back onto an exact sphere of `radius`
    moved *= radius / np.linalg.norm(moved, axis=1, keepdims=True)
    cat_surf.write_surface(str(out_file), moved.astype(np.float32), faces)


def invert_sulc(sulc_file: Path, out_file: Path) -> None:
    """Step 3: MSMSulc drives on the negated sulcal-depth map."""
    import cat_surf
    import nibabel as nib

    values = np.asarray(cat_surf.read_values(str(sulc_file)), dtype=np.float32)
    darray = nib.gifti.GiftiDataArray(
        -values, intent="NIFTI_INTENT_SHAPE", datatype="NIFTI_TYPE_FLOAT32")
    nib.save(nib.gifti.GiftiImage(darrays=[darray]), str(out_file))


def msm_env(threads: int) -> dict:
    """Environment for the MSM child process.

    FSL's shared libraries have to be on the loader path for ``newmsm``, but
    exporting that into *this* process breaks numpy: its compiled extensions
    pick up FSL's BLAS/libstdc++ instead of their own and fail to import with
    a misleading "do not import numpy from its source directory".  So the
    loader path is set for the child only.
    """
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    libs = [p for p in (os.environ.get("FSLDEVDIR"), os.environ.get("FSLDIR"))
            if p]
    if libs:
        paths = [str(Path(p) / "lib") for p in libs]
        for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
            existing = env.get(var, "")
            env[var] = os.pathsep.join(paths + ([existing] if existing else []))
    return env


def one(job) -> str:
    """Register one hemisphere.  Hemispheres are independent, so they are the
    unit of parallelism -- 2x the jobs of a per-subject split, which keeps the
    pool busy at the tail end of a batch."""
    subject, hemi, side, t1prep, work, binary, config, threads, verbose = job
    surf = Path(t1prep) / subject / "surf"
    out_dir = Path(work) / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    spheres = [p for p in sorted(surf.glob(f"{hemi}.sphere.*.gii"))
               if ".reg." not in p.name]
    regs = sorted(surf.glob(f"{hemi}.sphere.reg.fsLR.*.gii"))
    sulcs = sorted(surf.glob(f"{hemi}.sulc.*"))
    missing = [n for n, v in (("sphere", spheres), ("sphere.reg.fsLR", regs),
                              ("sulc", sulcs)) if not v]
    if missing:
        return f"{subject} {hemi}: missing {', '.join(missing)}"

    final = out_dir / f"{hemi}.sphere.reg.surf.gii"
    if final.exists():
        return f"{subject} {hemi}: cached"

    t0 = time.time()
    rot = out_dir / f"{hemi}.sphere_rot.surf.gii"
    prepare_sphere(spheres[0], regs[0], rot)
    inv = out_dir / f"{hemi}.sulc_inv.shape.gii"
    invert_sulc(sulcs[0], inv)

    cmd = [
        binary,
        f"--conf={config}",
        f"--inmesh={rot}",
        "--refmesh=" + str(
            MSM_DATA / f"fsaverage.{side}_LR.spherical_std.164k_fs_LR.surf.gii"),
        f"--indata={inv}",
        "--refdata=" + str(MSM_DATA / f"{side}.refsulc.164k_fs_LR.shape.gii"),
        f"--out={out_dir}/{hemi}.",
    ]
    if verbose:
        cmd.append("--verbose")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env=msm_env(threads))
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return f"{subject} {hemi}: FAILED\n    " + "\n    ".join(tail)
    return f"{subject} {hemi}: ok ({time.time() - t0:.0f}s)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t1prep", required=True,
                    help="root holding one T1Prep output directory per subject")
    ap.add_argument("--work", required=True, help="where to write the results")
    ap.add_argument("--subjects", nargs="+",
                    help="subjects to run (default: every directory found)")
    ap.add_argument("--subject-file",
                    help="file listing subjects, one per line")
    ap.add_argument("--binary", default="newmsm",
                    help="MSM executable (default: newmsm; use 'msm' for the "
                         "original)")
    ap.add_argument("--sloppy", action="store_true",
                    help="use MSMSulcStrainSloppyconf, as fMRIPrep's --sloppy")
    ap.add_argument("--jobs", type=int, default=4,
                    help="hemispheres to register concurrently")
    ap.add_argument("--threads", type=int, default=2,
                    help="OpenMP threads per hemisphere")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if shutil.which(a.binary) is None:
        raise SystemExit(
            f"'{a.binary}' not on PATH.  newMSM builds inside FSL's build "
            "system: install FSL, then set FSLDIR and FSLCONFDIR "
            "(=$FSLDIR/config) before running make in the newMSM tree.")

    config = MSM_DATA / (
        "MSMSulcStrainSloppyconf" if a.sloppy else "MSMSulcStrainFinalconf")
    if not config.exists():
        raise SystemExit(f"missing {config}")

    t1prep, work = Path(a.t1prep), Path(a.work)
    subjects = a.subjects or (
        [s.strip() for s in open(a.subject_file) if s.strip()]
        if a.subject_file else
        sorted(p.name for p in t1prep.iterdir() if p.is_dir()))

    jobs = [(subject, hemi, side, a.t1prep, a.work, a.binary, str(config),
             a.threads, a.verbose)
            for subject in subjects
            for hemi, side in (("lh", "L"), ("rh", "R"))]

    failures = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for n, msg in enumerate(ex.map(one, jobs), 1):
            failures += ": ok" not in msg and ": cached" not in msg
            print(f"[{n}/{len(jobs)}] {msg}", flush=True)
    print(f"\n{len(jobs) - failures}/{len(jobs)} hemispheres in "
          f"{time.time() - t0:.0f}s -> {a.work}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
