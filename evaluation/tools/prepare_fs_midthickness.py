#!/usr/bin/env python3
"""Build midthickness surfaces from a FreeSurfer run, for label attachment.

Mindboggle's labelled surfaces sit on the **pial**, so attaching labels to a
pial mesh gives the smallest vertex-to-label distance -- and the worst
transfer.  At the pial the two banks of a sulcus are in contact, so a
nearest-neighbour lookup jumps across the fundus.  Measured on
OASIS-TRT-20-2 lh, the fraction of vertices whose label matches none of their
1-ring neighbours:

    pial   0.44 mm from the labels,  0.4 % unlabelled,  0.502 % speckle
    mid    1.12 mm,                  0.5 % unlabelled,  0.175 % speckle
    white  2.14 mm,                 13.6 % unlabelled,  0.117 % speckle

Midthickness is the only one that is good on both counts, and it is what
T1Prep's central surface already is -- so using it here is what makes the two
arms comparable rather than a choice tuned to the outcome.  Over 19 subjects
the difference is not small: Dice 0.8516 at mid against 0.8197 at pial and
0.7999 at white (leave-one-out).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freesurfer", required=True,
                    help="SUBJECTS_DIR holding one directory per subject")
    ap.add_argument("--out", required=True,
                    help="where to write <subject>/<hemi>.mid.gii")
    ap.add_argument("--subjects", nargs="+")
    ap.add_argument("--subject-file",
                    help="file listing subjects, one per line")
    a = ap.parse_args(argv)

    import cat_surf

    root = Path(a.freesurfer)
    names: list[str] = list(a.subjects or [])
    if a.subject_file:
        names += [s.strip() for s in open(a.subject_file) if s.strip()]
    subjects = sorted(set(names)) or sorted(
        p.name for p in root.iterdir() if (p / "surf").is_dir())

    written = 0
    for subject in subjects:
        for hemi in ("lh", "rh"):
            white = root / subject / "surf" / f"{hemi}.white"
            pial = root / subject / "surf" / f"{hemi}.pial"
            if not (white.is_file() and pial.is_file()):
                print(f"{subject} {hemi}: missing white or pial",
                      file=sys.stderr)
                continue
            w, faces = cat_surf.read_surface(str(white))
            p, _ = cat_surf.read_surface(str(pial))
            out = Path(a.out) / subject
            out.mkdir(parents=True, exist_ok=True)
            mid = (np.asarray(w) + np.asarray(p)) / 2.0
            cat_surf.write_surface(str(out / f"{hemi}.mid.gii"),
                                   mid.astype(np.float32), faces)
            written += 1
        print(f"{subject}: ok", flush=True)
    print(f"\n{written} hemispheres -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
