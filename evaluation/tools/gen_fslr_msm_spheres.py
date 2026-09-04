"""Derive the fsLR and msmsulc spheres from an existing T1Prep run.

Both are normally written only under --fmriprep; everything they need is in a
plain run, so they can be added afterwards instead of reprocessing.
"""
import sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "/Users/gaser/GitHub/T1Prep/src")
D = Path("/Users/gaser/GitHub/T1Prep/src/t1prep/data")
MB = Path("/Users/gaser/Downloads/mindboggle-101/data")


def one(args):
    subject, h = args
    from t1prep import fslr
    surf = MB / subject / "surf"
    reg = surf / f"{h}.sphere.reg.t1weighted.gii"
    if not reg.exists():
        return f"{subject} {h}: no sphere.reg"
    fslr_out = surf / f"{h}.sphere.reg.fsLR.t1weighted.gii"
    msm_out = surf / f"{h}.sphere.reg.msm.t1weighted.gii"
    try:
        if not fslr_out.exists():
            fslr.write_reg_sphere(
                sphere_reg_file=str(reg), out_file=str(fslr_out),
                fslr_templates_dir=str(D / "templates_surfaces_fsLR"),
                fsaverage_sphere_file=str(
                    D / "templates_surfaces_32k" / f"{h}.sphere.freesurfer.gii"),
                fshemi=h)
        if not msm_out.exists():
            fslr.write_msm_sphere(
                mid_surface_file=str(surf / f"{h}.central.t1weighted.gii"),
                reg_sphere_file=str(fslr_out), out_file=str(msm_out),
                fslr_templates_dir=str(D / "templates_surfaces_fsLR"),
                fshemi=h,
                sphere_file=str(surf / f"{h}.sphere.t1weighted.gii"))
    except Exception as exc:                       # keep going, report at end
        return f"{subject} {h}: FAILED {type(exc).__name__}: {exc}"
    return None


if __name__ == "__main__":
    jobs = [(s.name, h) for s in sorted(MB.iterdir()) if (s / "surf").is_dir()
            for h in ("lh", "rh")]
    t0 = time.time()
    bad = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            if r:
                bad.append(r)
            if i % 20 == 0:
                print(f"{i}/{len(jobs)}  {time.time() - t0:.0f}s",
                      flush=True)
    print(f"done {len(jobs)} hemispheres in {time.time() - t0:.0f}s, "
          f"{len(bad)} failures", flush=True)
    for b in bad[:20]:
        print(" ", b, flush=True)
