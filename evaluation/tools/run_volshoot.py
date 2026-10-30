#!/usr/bin/env python3
"""Other settings of T1Prep's geodesic shooting, on T1Prep's own tissue maps.

The settings table of ``results/RESULTS.md``.  T1Prep registers its
affine-registered GM and WM maps by geodesic shooting
(``t1prep._shoot.shoot_register``, CAT-Surface's ``cat_surf.vol_shoot``) with
its default setting, CAT12's own; this tool repeats the shooting with others
-- ``--fmriprep``'s, every scheduled iteration with half the regularisation,
for instance -- on exactly the maps the pipeline registered, and composes
each field into a ``y_`` the way T1Prep composes its own.

``prepare``  T1Prep's segmentation for every subject, keeping the shooting's
             inputs: ``rp1``/``rp2`` on the 1.5 mm template grid, the affine
             in normalised coordinates and the native brain mask.  T1Prep's
             own ``y_`` (the default setting) is written alongside.
``shoot``    The shooting with the settings given as ``--set key=value``
             (keywords of ``cat_surf.vol_shoot``), written as
             ``<work>/<arm>/<subject>/y_<subject>.nii`` together with the
             Jacobian determinant of the deformation inside the brain.

    python evaluation/tools/run_volshoot.py prepare --mindboggle <mb>/data \\
        --work <work> --subject-file evaluation/data/subset20.txt
    python evaluation/tools/run_volshoot.py shoot --work <work> --arm fmriprep \\
        --set regstr=1 --set reg_scale=0.5
    python evaluation/tools/eval_mindboggle.py project-volume \\
        --mindboggle <mb>/data --t1prep <work>/fmriprep --work <work>/eval \\
        --out-space fmriprep --subject-file evaluation/data/subset20.txt
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def subjects(args: argparse.Namespace, available: list[str]) -> list[str]:
    """The requested subjects (``--subjects``/``--subject-file``), else all."""
    names = list(args.subjects or [])
    if args.subject_file:
        with open(args.subject_file) as fh:
            names += [s.strip() for s in fh if s.strip()]
    return names or sorted(available)


def find_t1(root: Path, subject: str) -> Path:
    """The subject's T1w: Mindboggle's ``t1weighted.nii.gz`` or ``<subject>.nii.gz``."""
    for name in ("t1weighted.nii.gz", f"{subject}.nii.gz", f"{subject}.nii"):
        if (root / subject / name).exists():
            return root / subject / name
    raise FileNotFoundError(f"no T1w for {subject} in {root / subject}")


def segment_with_dump(dump: Path, argv: list[str]) -> None:
    """``t1prep.segment`` with the shooting's inputs written to *dump*.

    Runs in its own process (``prepare`` starts one per subject): the first
    two maps ``_to_template_space`` produces are GM and WM on the 1.5 mm
    template grid, and ``_save_deformations`` receives the affine and the
    native mask the ``y_`` is composed with.
    """
    import nibabel as nib
    import numpy as np

    sys.path.insert(0, str(REPO / "src"))
    from t1prep import segment

    to_template, save_deformations = segment._to_template_space, segment._save_deformations
    maps = []

    def _to_template_space(img, warp_template):
        out = to_template(img, warp_template)
        maps.append(out)
        if len(maps) <= 2:
            data = np.asarray(out.dataobj, dtype=np.float32)
            nib.save(nib.Nifti1Image(data, out.affine), str(dump / f"rp{len(maps)}.nii"))
        return out

    def _save_deformations(opts, names, warp_xy, warp_yx, affine, mask):
        values = affine.values if hasattr(affine, "values") else affine
        np.save(dump / "affine.npy", np.asarray(values, dtype=np.float64))
        nib.save(mask, str(dump / "mask.nii.gz"))
        return save_deformations(opts, names, warp_xy, warp_yx, affine, mask)

    segment._to_template_space = _to_template_space
    segment._save_deformations = _save_deformations
    dump.mkdir(parents=True, exist_ok=True)
    sys.argv = ["t1prep.segment", *argv]
    segment.run_segment()


def cmd_prepare(args: argparse.Namespace) -> int:
    """Segment every subject and keep the inputs of its shooting."""
    root = Path(args.mindboggle)
    todo = subjects(args, [p.name for p in root.iterdir() if p.is_dir()])
    work = Path(args.work) / "t1prep"

    def one(subject):
        out = work / subject
        if (out / "shoot" / "mask.nii.gz").exists():
            return subject, "done before"
        for sub in ("mri", "report", "label"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__, "_segment", str(out / "shoot"), "--",
               "--mwp", "--vessel", "1", "--atlas", "",
               "--input", str(find_t1(root, subject)), "--mri-dir", str(out / "mri"),
               "--report-dir", str(out / "report"), "--label-dir", str(out / "label")]
        with open(out / "segment.log", "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  check=False).returncode
        return subject, "ok" if code == 0 else f"exit {code}, see {out / 'segment.log'}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for subject, status in pool.map(one, todo):
            print(f"{subject}: {status}", flush=True)
    return 0


def jacobian_summary(jac, brain) -> dict:
    """1st/99th percentile, median |log det J| and folded fraction in *brain*."""
    import numpy as np

    d = np.asarray(jac.dataobj, dtype=np.float64)[brain]
    return {"p01": float(np.percentile(d, 1)), "p99": float(np.percentile(d, 99)),
            "median_abs_log": float(np.median(np.abs(np.log(np.clip(d, 1e-6, None))))),
            "folded": float((d <= 0).mean())}


def parse_settings(pairs: list[str]) -> dict:
    """``key=value`` pairs -> keywords of ``cat_surf.vol_shoot``."""
    settings = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            settings[key] = int(value)
        except ValueError:
            settings[key] = float(value)
    return settings


def cmd_shoot(args: argparse.Namespace) -> int:
    """The shooting with other settings for every prepared subject, as ``y_``."""
    import nibabel as nib
    import numpy as np

    sys.path.insert(0, str(REPO / "src"))
    from t1prep._shoot import shoot_register, shooting_templates
    from t1prep.transforms import save_deformation_spm

    settings = parse_settings(args.set)
    prepared = Path(args.work) / "t1prep"
    todo = subjects(args, [p.name for p in prepared.iterdir()
                           if (p / "shoot" / "mask.nii.gz").exists()])
    templates, _ = shooting_templates()
    brain = templates[-1][..., 0] + templates[-1][..., 1] > 0.5   # stored order
    del templates
    arm = Path(args.work) / args.arm
    rows = []
    for subject in todo:
        dump = prepared / subject / "shoot"
        out = arm / subject
        if (out / f"y_{subject}.nii").exists():
            rows.append(json.loads((out / "jacobian.json").read_text()))
            print(f"{subject}: done before", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        rp = [nib.load(str(dump / f"rp{k}.nii")) for k in (1, 2)]
        reg = shoot_register(rp[0], rp[1], 1.0, **settings)
        save_deformation_spm(reg["warp_xy"], np.load(dump / "affine.npy"),
                             nib.load(str(dump / "mask.nii.gz")), str(out / f"y_{subject}.nii"))
        stats = jacobian_summary(reg["jac"], brain)   # both in the templates' order
        (out / "jacobian.json").write_text(json.dumps(stats, indent=1))
        rows.append(stats)
        print(f"{subject}: ok ({reg['info']['iterations']} iterations)", flush=True)
    if rows:
        med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        print(f"\nmedian over {len(rows)} subjects: det J 1st-99th percentile "
              f"{med['p01']:.2f}-{med['p99']:.2f}, median |log det J| "
              f"{med['median_abs_log']:.3f}, folded {med['folded']:.1e}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_segment"]:          # internal: one subject, own process
        segment_with_dump(Path(argv[1]), argv[3:])
        return 0

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--work", required=True, help="working directory")
        p.add_argument("--subjects", nargs="+", help="restrict to these subjects")
        p.add_argument("--subject-file", help="file with one subject per line")

    p = sub.add_parser("prepare", help="segment and keep the shooting's inputs")
    common(p)
    p.add_argument("--mindboggle", required=True,
                   help="the data root, one directory per subject")
    p.add_argument("--jobs", type=int, default=1, help="subjects at a time (1)")
    p.set_defaults(func=cmd_prepare)

    s = sub.add_parser("shoot", help="the shooting with other settings, as y_")
    common(s)
    s.add_argument("--arm", required=True, help="output subdirectory")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="a cat_surf.vol_shoot keyword (regstr, reg_scale, reg_vx, "
                        "n_threads, ...); repeatable")
    s.set_defaults(func=cmd_shoot)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
