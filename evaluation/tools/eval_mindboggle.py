#!/usr/bin/env python3
"""Evaluate T1Prep's spherical registration on the Mindboggle-101 labels.

Mindboggle-101 (Klein & Tourville 2012) is the public stand-in for the
internal Buckner40/DK40 benchmark: 101 subjects whose cortex was manually
labelled under the DKT protocol, released CC-BY.  This script reproduces the
usual cross-subject label-transfer evaluation on it, so T1Prep's numbers land
on the same axis as the published FreeSurfer / Spherical Demons / MSM ones.

What is actually measured
-------------------------
Registration accuracy only.  Each subject's *manual* labels are carried into
the group template by **T1Prep's own spherical registration** and compared
there; nothing else in the chain differs between subjects.  Two protocols:

``loo``
    Leave-one-out majority-vote atlas: build a per-vertex majority label from
    the other N-1 subjects, apply it to the held-out subject, and Dice it
    against that subject's manual labels.  This is the FreeSurfer-style
    protocol and measures registration *and* the atlas the group forms.
``pairs``
    Pairwise Dice between every two subjects' labels in template space.  No
    atlas is involved, so this isolates registration alone.  This is what the
    SphereMorph / S3Reg / SUGAR comparisons report.

The ground truth never passes through a second registration.  The manual
labels are attached to T1Prep's own native mesh geometrically -- Mindboggle's
labelled surfaces and volumes share the world space of the ``t1weighted.nii.gz``
that T1Prep was run on, so this is a sub-millimetre transfer, not an
alignment.  Only then does the sphere under test move them.  Carrying the
labels in through FreeSurfer's ``sphere.reg`` instead would measure the two
registrations mixed together.

``--labels surface`` (default) takes the ground truth from Mindboggle's
labelled surfaces (``?h.labels.DKT31.manual.vtk``), by nearest vertex.
``--labels volume`` takes it from the label volume
(``labels.DKT31.manual.nii.gz``), by nearest labelled voxel.  Prefer the
surface: it is the primary manual product, and the volumes are a ribbon-filled
rasterisation of it.  Measured on OASIS-TRT-20-1, recovering surface labels
from the volume costs ~4.6 % Dice (96.9 % of vertices agree), which is the
same order as the differences between registration methods; the surface route
costs 0.5-0.9 % at a realistic 1-1.5 mm offset between T1Prep's central
surface and Mindboggle's.

Data
----
Only cohorts whose ``*_volumes.tar.gz`` you have can be evaluated -- T1Prep
needs the ``t1weighted.nii.gz``; the ``SurfaceLabels_*.tar.gz`` archives hold
labelled ``.vtk`` surfaces and nothing else.  Extract them side by side and
pass every root, e.g.::

    --mindboggle /data/mb/OASIS-TRT-20_volumes /data/mb/OASIS-TRT-20_surfaces

Usage
-----
Step 1 -- process the T1w volumes, one output directory per subject, named
after the subject::

    for s in /data/mb/OASIS-TRT-20_volumes/*/; do
        T1Prep --out-dir /data/mb_t1prep/$(basename "$s") "$s/t1weighted.nii.gz"
    done

Step 2 -- project every subject's manual labels into the template::

    python scripts/eval_mindboggle.py project \
        --mindboggle /data/mb/OASIS-TRT-20_surfaces \
        --t1prep     /data/mb_t1prep \
        --work       /data/mb_eval \
        --space fsaverage

Step 3 -- score::

    python scripts/eval_mindboggle.py dice --work /data/mb_eval \
        --protocol both --csv /data/mb_eval/dice.csv

``--space`` selects which registration is under test:

=============  ==========================================  ====================
``--space``    sphere evaluated                            target template
=============  ==========================================  ====================
``fsaverage``  ``Spherereg_surface`` (Spherical Demons)    fsaverage 32k
``fsLR``       ``SphereregFsLR_surface`` (project-unproj.) fsLR 32k
``msm``        ``SphereregMSM_surface``                    fsLR 32k
=============  ==========================================  ====================

``fsaverage`` is T1Prep's default registration.  The other two are the same
algorithm, not different ones: ``fsLR`` is that result carried into the fsLR
frame by a fixed project-unproject (no new registration), and ``msm`` is a
second Spherical Demons run onto the fsLR average.  The ``msm`` name is the
BIDS ``desc-msmsulc`` entity, written so fMRIPrep skips its own MSMSulc step
(see :mod:`t1prep.fslr`) -- nothing in it derives from MSM.  To score real
MSM, run it and point ``--sphere-file`` and ``--template-sphere`` at its
output.
"""

from __future__ import annotations

import argparse
import json
import sys
from fnmatch import fnmatch
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

# Prefer the checkout over any installed copy: this tool tracks the source
# tree, and an older site-packages t1prep silently lacks newer helpers.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

import cat_surf
from t1prep.utils import DATA_PATH_T1PREP, NameTable, name_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Which sphere is evaluated, and which template sphere it was registered to.
SPACES = {
    "fsaverage": ("Spherereg_surface", "templates_surfaces_32k",
                  "{fshemi}.sphere.freesurfer.gii"),
    "fsLR": ("SphereregFsLR_surface", "templates_surfaces_fsLR",
             "{fshemi}.sphere.fsLR.gii"),
    "msm": ("SphereregMSM_surface", "templates_surfaces_fsLR",
            "{fshemi}.sphere.fsLR.gii"),
}

#: Default ground-truth file per route, relative to a Mindboggle subject dir.
LABEL_PATTERNS = {
    "surface": "{fshemi}.labels.DKT31.manual.vtk",
    "volume": "labels.DKT31.manual.nii.gz",
}

#: aparc code block each hemisphere uses in Mindboggle's label *volumes*.
#: The labelled *surfaces* carry the bare offset instead, with the hemisphere
#: in the file name -- both routes are reduced to offsets here.
HEMI_CODE_BASE = {"lh": 1000, "rh": 2000}

#: The 31 DKT cortical regions, as aparc offsets.  DKT drops DK's banks of the
#: superior temporal sulcus (1), corpus callosum (4), frontal pole (32) and
#: temporal pole (33); ``unknown`` (0) is not a region.
DKT_OFFSETS = tuple(o for o in range(1, 36) if o not in (1, 4, 32, 33))


@lru_cache(maxsize=1)
def region_names() -> dict[int, str]:
    """Map aparc offsets to region names, read from the shipped DK40 annot.

    The Desikan-Killiany annot's label index *is* the aparc offset, so this
    gives DKT names without needing a FreeSurfer colour LUT on the machine.
    """
    annot = (DATA_PATH_T1PREP / "atlases_surfaces_32k"
             / "lh.aparc_DK40.freesurfer.annot")
    _, _, names = nib.freesurfer.read_annot(str(annot))
    return {i: (n.decode() if isinstance(n, bytes) else n).strip()
            for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# Locating files
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _name_table(data_dir: str | None = None) -> NameTable:
    """Names.tsv, read once — the filename patterns come from the pipeline.

    ``--data-dir`` has to reach here too: an installed copy older than the
    checkout lacks the newer codes (the fsLR and msmsulc spheres among them),
    and looking them up would fail with a bare KeyError.
    """
    path = Path(data_dir) / "Names.tsv" if data_dir else name_file
    return NameTable(path)


def find_t1prep_file(subject_dir: Path, code: str, fshemi: str,
                     data_dir: str | None = None) -> Path:
    """Return the T1Prep output ``code`` for one hemisphere.

    Both the CAT12 (``surf/lh.central.<bname>.gii``) and the BIDS
    (``<bname>_hemi-L_..._midthickness.surf.gii``) layouts are searched, so
    the caller does not have to know whether ``--bids`` was used.
    """
    table = _name_table(data_dir)

    def _glob(code_: str, column: int, hemi: str) -> str:
        # bname varies per subject; glob over it rather than guessing.
        return (table.pattern(code_, column)
                .replace("{bname}", "*")
                .replace("{side}", hemi)
                .replace("{desc}", "")
                .replace("{space}", "")
                .replace("{nii_ext}", "nii.gz")
                .replace("..", "."))

    candidates: list[Path] = []
    for column, hemi, subdir in ((1, fshemi, "surf"),
                                 (2, "L" if fshemi == "lh" else "R", "")):
        own = _glob(code, column, hemi)
        # Codes nest: Spherereg_surface's "lh.sphere.reg.*.gii" also matches
        # "lh.sphere.reg.fsLR.<bname>.gii" once that file exists, because
        # bname is a wildcard.  A file belongs to this code only if no other
        # code matches it with more literal text before its wildcard.
        others = [g for g in (_glob(c, column, hemi)
                              for c in table.codes() if c != code)
                  if len(g.split("*")[0]) > len(own.split("*")[0])]
        for hit in sorted((subject_dir / subdir).glob(own)):
            if not any(fnmatch(hit.name, o) for o in others):
                candidates.append(hit)
    hits = [p for p in candidates if p.is_file()]
    if not hits:
        raise FileNotFoundError(
            f"no {code} for {fshemi} under {subject_dir} — was T1Prep run "
            "with surface reconstruction and without --no-sphere-reg?"
        )
    if len(hits) > 1:
        raise FileNotFoundError(
            f"{code} for {fshemi} is ambiguous under {subject_dir}: "
            + ", ".join(p.name for p in hits)
        )
    return hits[0]


def find_labels(roots: list[str], subject: str, pattern: str) -> Path:
    """Return one subject's ground-truth file, searched across the roots.

    The cohorts ship as separate archives, and the labelled surfaces and the
    volumes as separate archives again, so more than one extracted tree may
    have to be searched for any given subject.
    """
    hits: list[Path] = []
    for root in roots:
        hits += sorted((Path(root) / subject).glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"no {pattern} for {subject} under " + ", ".join(roots))
    # Every subject ships its labels in native *and* MNI152 space, and the
    # MNI152 name sorts first.  Taking it would silently measure an affine
    # misalignment instead of the registration, so drop it unless the caller
    # asked for it by name.  Then prefer cortex-only over the +aseg variant.
    for drop in ("MNI152", "aseg"):
        if drop not in pattern:
            hits = [p for p in hits if drop not in p.name] or hits
    return hits[0]


def find_ground_truth(roots: list[str], subject: str, fshemi: str,
                      routes: list[str],
                      override: str | None) -> tuple[Path, str]:
    """Locate one subject's ground truth, trying each route in turn.

    Returns the file and the route that found it, so a run mixing the two
    (``--labels auto``, where a cohort ships no labelled surfaces) can say
    per subject which was used rather than hiding it.
    """
    last: Exception | None = None
    for route in routes:
        pattern = (override or LABEL_PATTERNS[route]).format(fshemi=fshemi)
        try:
            return find_labels(roots, subject, pattern), route
        except FileNotFoundError as exc:
            last = exc
    raise last  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Reading the ground truth
# ---------------------------------------------------------------------------

def read_vtk_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read points and per-vertex scalars from a legacy ASCII VTK PolyData.

    Mindboggle's labelled surfaces are ASCII ``POLYDATA`` with the labels as
    ``POINT_DATA`` scalars.  Only the points and the scalars are needed here,
    so the polygons are skipped.

    Returns:
        Tuple of the ``(V, 3)`` point array and the ``(V,)`` label array.
    """
    lines = path.read_text().splitlines()
    try:
        p_at = next(i for i, ln in enumerate(lines)
                    if ln.startswith("POINTS"))
        s_at = next(i for i, ln in enumerate(lines)
                    if ln.startswith("LOOKUP_TABLE"))
    except StopIteration:
        raise ValueError(f"{path} is not a VTK PolyData with POINT_DATA")
    n = int(lines[p_at].split()[1])

    def _take(start: int, count: int) -> np.ndarray:
        """Collect ``count`` floats from ``start``, however they are wrapped."""
        out: list[float] = []
        for ln in lines[start:]:
            out.extend(float(t) for t in ln.split())
            if len(out) >= count:
                break
        return np.asarray(out[:count], dtype=np.float64)

    points = _take(p_at + 1, 3 * n).reshape(n, 3)
    labels = _take(s_at + 1, n)
    return points, np.rint(labels).astype(np.int32)


def tkr_offset(reference_file: Path) -> np.ndarray:
    """FreeSurfer surface RAS -> the reference volume's scanner RAS.

    Mindboggle's labelled surfaces carry FreeSurfer "tkrRAS" coordinates,
    whose origin is the volume centre rather than the scanner origin, so they
    must be shifted by ``c_ras`` before they sit on the anatomy T1Prep saw.
    OASIS-TRT-20 happens to have ``c_ras == 0``, which makes that cohort work
    without the shift and every other one land 3-170 mm away.
    """
    img = nib.load(str(reference_file))
    return nib.affines.apply_affine(img.affine, np.array(img.shape) / 2.0)


def labels_from_surface(label_file: Path, vertices: np.ndarray,
                        reference_file: Path,
                        max_dist: float) -> tuple[np.ndarray, dict]:
    """Attach Mindboggle's labelled surface to a native mesh by proximity.

    Once the labelled surface is shifted into the reference volume's scanner
    space, both surfaces describe the same physical cortex, so each T1Prep
    vertex takes the label of the nearest Mindboggle vertex.  They are not the
    same surface -- T1Prep's is central, Mindboggle's is a FreeSurfer surface
    -- but that offset is along the normal, and a displacement of 1-1.5 mm
    changes only ~0.5-0.9 % of the labels.
    """
    points, labels = read_vtk_labels(label_file)
    points = points + tkr_offset(reference_file)
    dist, idx = cKDTree(points).query(vertices)
    return _assemble(labels, dist, idx, len(vertices), max_dist)


def labels_from_volume(label_file: Path, vertices: np.ndarray, fshemi: str,
                       max_dist: float) -> tuple[np.ndarray, dict]:
    """Attach manual label-volume codes to a native mesh by proximity.

    Each vertex takes the code of the nearest labelled voxel in world
    coordinates, reduced to a bare aparc offset so the two routes agree.
    """
    img = nib.load(str(label_file))
    data = np.asanyarray(img.dataobj)
    base = HEMI_CODE_BASE[fshemi]
    mask = (data >= base + 1) & (data < base + 1000)
    if not mask.any():
        raise ValueError(
            f"{label_file} holds no codes in {base}..{base + 999}; this route "
            "expects Mindboggle's aparc-coded DKT volumes (1002..2035)"
        )
    codes = (data[mask] - base).astype(np.int32)
    xyz = nib.affines.apply_affine(img.affine, np.argwhere(mask))
    dist, idx = cKDTree(xyz).query(vertices)
    return _assemble(codes, dist, idx, len(vertices), max_dist)


def _assemble(source: np.ndarray, dist: np.ndarray, idx: np.ndarray,
              n_vertices: int, max_dist: float) -> tuple[np.ndarray, dict]:
    """Gather a nearest-neighbour query into labels plus a QC record.

    The distances are summarised over *every* vertex, not only the ones that
    found a label: a ground truth in the wrong space still puts some vertices
    close to something, and a median taken over those alone looks reassuring
    while most of the surface is nowhere near a label.
    """
    hit = dist <= max_dist
    native = np.zeros(n_vertices, dtype=np.int32)
    native[hit] = source[idx[hit]]
    qc = {
        "n_vertices": int(n_vertices),
        "frac_unlabelled": float(1.0 - hit.mean()),
        "median_dist_mm": float(np.median(dist)),
        "p95_dist_mm": float(np.percentile(dist, 95)),
    }
    return native, qc


def warp_labels_to_mni(label_file: Path, def_file: Path) -> tuple:
    """Resample a native label volume into MNI with T1Prep's deformation.

    ``y_*.nii`` is SPM's pull convention: defined on the MNI grid, holding for
    each output voxel the native millimetres it samples from.  Labels are
    categorical, so the lookup is nearest neighbour.

    Returns:
        Tuple of the MNI-grid label array and the grid's affine.
    """
    y = nib.load(str(def_file))
    if y.ndim != 5 or y.shape[3] != 1 or y.shape[4] != 3:
        raise ValueError(
            f"{def_file} is {y.shape}; expected a 5-D [X,Y,Z,1,3] SPM "
            "deformation.  Runs before the y_ fix wrote a 4-D field holding "
            "normalised coordinates and only the non-linear half of the "
            "transform -- those subjects have to be reprocessed."
        )
    mm = np.asanyarray(y.dataobj)[:, :, :, 0, :].reshape(-1, 3)
    img = nib.load(str(label_file))
    data = np.asanyarray(img.dataobj).astype(np.int32)
    # SPM leaves the deformation undefined outside the field of view and
    # writes NaN there -- CAT12's fields are ~5 % NaN.  Rounding those to int
    # yields arbitrary indices, so they have to be dropped before the lookup.
    finite = np.isfinite(mm).all(axis=1)
    safe = np.where(finite[:, None], mm, 0.0)
    vox = np.rint(
        nib.affines.apply_affine(np.linalg.inv(img.affine), safe)
    ).astype(int)
    inside = finite & np.all((vox >= 0) & (vox < np.array(data.shape)), axis=1)
    out = np.zeros(len(vox), dtype=np.int32)
    out[inside] = data[vox[inside, 0], vox[inside, 1], vox[inside, 2]]
    return out.reshape(y.shape[:3]), y.affine


def cmd_project_volume(args: argparse.Namespace) -> int:
    """Warp every subject's manual labels into MNI with the volume warp.

    The output is the same per-subject label vector the surface path caches,
    so ``dice`` scores both with identical code -- but note that voxel Dice on
    a filled ribbon and vertex Dice on a surface are different measurements,
    comparable across volume methods rather than against the surface numbers.
    """
    subjects = sorted(p.name for p in Path(args.t1prep).iterdir() if p.is_dir())
    if args.subjects:
        subjects = [s for s in subjects if s in set(args.subjects)]
    out_dir = Path(args.work) / args.out_space
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.work) / f"manifest_{args.out_space}.json"
    manifest = {"space": args.out_space,
                "sphere": "identity" if args.preresampled else args.def_glob,
                "labels": "volume",
                "max_dist_mm": 0.0, "subjects": {}}
    prior = {}
    if manifest_path.exists() and not args.fresh:
        prior = json.loads(manifest_path.read_text()).get("subjects", {})

    shape = None
    for n, subject in enumerate(subjects, 1):
        try:
            labels = find_labels(args.mindboggle, subject, args.label_glob
                                 or LABEL_PATTERNS["volume"])
            if args.preresampled:
                # Labels already in the target space -- an affine baseline, or
                # any other tool's normalisation scored through this same
                # protocol.  Every subject must share one grid.
                img = nib.load(str(labels))
                mni = np.asanyarray(img.dataobj).astype(np.int32)
            else:
                defs = sorted((Path(args.t1prep) / subject).glob(
                    f"*/{args.def_glob}")) or sorted(
                    (Path(args.t1prep) / subject).glob(args.def_glob))
                if not defs:
                    raise FileNotFoundError(
                        f"no {args.def_glob} for {subject}")
                mni, _ = warp_labels_to_mni(labels, defs[0])
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{n}/{len(subjects)}] {subject}: skipped — {exc}",
                  file=sys.stderr, flush=True)
            continue
        if shape is None:
            shape = mni.shape
        elif mni.shape != shape:
            print(f"[{n}/{len(subjects)}] {subject}: skipped — grid "
                  f"{mni.shape} differs from {shape}", file=sys.stderr)
            continue
        per_hemi = {}
        for fshemi in ("lh", "rh"):
            base = HEMI_CODE_BASE[fshemi]
            off = np.where((mni >= base + 1) & (mni < base + 1000),
                           mni - base, 0).astype(np.int32)
            np.save(out_dir / f"{subject}_{fshemi}.npy", off.ravel())
            per_hemi[fshemi] = {
                "n_vertices": int(off.size),
                "frac_unlabelled": float((off == 0).mean()),
                "median_dist_mm": 0.0, "p95_dist_mm": 0.0, "route": "volume",
            }
        manifest["subjects"][subject] = per_hemi
        print(f"[{n}/{len(subjects)}] {subject}: "
              f"{int((mni > 0).sum())} labelled voxels", flush=True)

    fresh_count = len(manifest["subjects"])
    manifest["subjects"] = {**prior, **manifest["subjects"]}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nprojected {fresh_count} subjects -> {out_dir}")
    return 0


def native_to_template(native: np.ndarray, mid: Path, sphere: Path,
                       template_sphere: Path) -> np.ndarray:
    """Resample native-mesh labels onto the template mesh.

    The subject's registered sphere shares its topology with the central
    surface, so the pair ``(mid, sphere)`` defines the mapping T1Prep's
    registration produced; nearest-neighbour interpolation keeps the labels
    categorical.
    """
    verts, faces = cat_surf.read_surface(str(mid))
    sv, sf = cat_surf.read_surface(str(sphere))
    tv, tf = cat_surf.read_surface(str(template_sphere))
    if len(sv) != len(verts):
        raise ValueError(
            f"{sphere.name} has {len(sv)} vertices but {mid.name} has "
            f"{len(verts)} — these are not the same subject's surfaces"
        )
    _, _, out = cat_surf.resample_to_sphere(
        verts, faces, sv, sf, tv, tf,
        values=native.astype(np.float64), label_interpolation=True,
    )
    return np.rint(out).astype(np.int32)


def cmd_project(args: argparse.Namespace) -> int:
    sphere_code, template_dir, template_pat = SPACES[args.space]
    tmpl_root = Path(args.data_dir) if args.data_dir else DATA_PATH_T1PREP
    routes = (["surface", "volume"] if args.labels == "auto"
              else [args.labels])

    subjects = sorted(p.name for p in Path(args.t1prep).iterdir() if p.is_dir())
    if args.subjects:
        wanted = set(args.subjects)
        subjects = [s for s in subjects if s in wanted]
    if not subjects:
        raise SystemExit(f"no subject directories under {args.t1prep}")

    space_name = args.out_space or args.space
    out_dir = Path(args.work) / space_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.work) / f"manifest_{space_name}.json"
    manifest = {"space": space_name, "sphere": args.sphere_file or sphere_code,
                "labels": args.labels, "max_dist_mm": args.max_dist,
                "subjects": {}}
    # Projecting 101 subjects takes minutes, so a re-run that adds a cohort or
    # repairs a few subjects tops the cache up instead of starting over.  The
    # cached .npy files stay valid; only the subjects named now are redone.
    prior: dict = {}
    rejected: set[str] = set()
    if manifest_path.exists() and not args.fresh:
        prior = json.loads(manifest_path.read_text()).get("subjects", {})
    for n, subject in enumerate(subjects, 1):
        subject_dir = Path(args.t1prep) / subject
        per_hemi: dict[str, dict] = {}
        for fshemi in ("lh", "rh"):
            try:
                label_file, route = find_ground_truth(
                    args.mindboggle, subject, fshemi, routes, args.label_glob)
                mid = find_t1prep_file(subject_dir, "Mid_surface", fshemi,
                                       args.data_dir)
                if args.sphere_file:
                    sphere = Path(args.sphere_file.format(subject=subject,
                                                          hemi=fshemi))
                    if not sphere.is_file():
                        raise FileNotFoundError(f"no sphere at {sphere}")
                else:
                    sphere = find_t1prep_file(subject_dir, sphere_code,
                                              fshemi, args.data_dir)
            except FileNotFoundError as exc:
                print(f"[{n}/{len(subjects)}] {subject} {fshemi}: skipped — "
                      f"{exc}", file=sys.stderr, flush=True)
                per_hemi = {}
                break
            template_sphere = (
                Path(args.template_sphere.format(fshemi=fshemi))
                if args.template_sphere else
                tmpl_root / template_dir / template_pat.format(fshemi=fshemi))
            verts, _ = cat_surf.read_surface(str(mid))
            if route == "surface":
                reference = find_labels(args.mindboggle, subject,
                                        args.reference_glob)
                native, qc = labels_from_surface(label_file, verts, reference,
                                                 args.max_dist)
            else:
                native, qc = labels_from_volume(label_file, verts, fshemi,
                                                args.max_dist)
            qc["route"] = route
            tmpl = native_to_template(native, mid, sphere, template_sphere)
            np.save(out_dir / f"{subject}_{fshemi}.npy", tmpl)
            per_hemi[fshemi] = qc
        if per_hemi:
            worst = max(h["median_dist_mm"] for h in per_hemi.values())
            lost = max(h["frac_unlabelled"] for h in per_hemi.values())
            if lost > args.max_unlabelled:
                # Mindboggle ships the odd damaged surface (NKI-RS-22-16's rh
                # sits ~40 mm from its own anatomy).  Such a hemisphere would
                # still produce labels, just wrong ones, so drop the subject
                # rather than let it into the averages.
                rejected.add(subject)
                for fs in ("lh", "rh"):
                    (out_dir / f"{subject}_{fs}.npy").unlink(missing_ok=True)
                print(f"[{n}/{len(subjects)}] {subject}: REJECTED — "
                      f"{100 * lost:.1f} % unlabelled at {worst:.2f} mm "
                      f"(over --max-unlabelled {100 * args.max_unlabelled:.0f} %)",
                      file=sys.stderr, flush=True)
                continue
            manifest["subjects"][subject] = per_hemi
            flag = "  <-- CHECK SPACE" if worst > 3.0 else ""
            print(f"[{n}/{len(subjects)}] {subject}: median vertex-to-label "
                  f"distance {worst:.2f} mm, {100 * lost:.1f} % unlabelled"
                  f"{flag}", flush=True)

    fresh_count = len(manifest["subjects"])
    manifest["subjects"] = {**prior, **manifest["subjects"]}
    for subject in rejected:
        manifest["subjects"].pop(subject, None)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    carried = len(manifest["subjects"]) - fresh_count
    note = f" (+{carried} carried over)" if carried else ""
    print(f"\nprojected {fresh_count} subjects{note} -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compact(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack per-subject offsets as compact indices, 0 meaning unlabelled."""
    lookup = np.zeros(max(DKT_OFFSETS) + 1, dtype=np.int16)
    for i, off in enumerate(DKT_OFFSETS, 1):
        lookup[off] = i
    out = np.zeros((len(arrays), arrays[0].size), dtype=np.int16)
    for i, a in enumerate(arrays):
        inside = (a > 0) & (a <= max(DKT_OFFSETS))
        out[i, inside] = lookup[a[inside]]
    return out


def _dice_from_confusion(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Per-label Dice between two compact label vectors.

    Labels absent from both vectors get NaN so they can be dropped rather
    than counted as perfect or as zero.
    """
    conf = np.bincount(a.astype(np.int64) * k + b, minlength=k * k)
    conf = conf.reshape(k, k)
    inter = np.diag(conf).astype(np.float64)
    size = conf.sum(1) + conf.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2.0 * inter / size
    dice[size == 0] = np.nan
    return dice


def cortex_mask(space: str, fshemi: str, labels: np.ndarray,
                tmpl_root: Path) -> np.ndarray:
    """Vertices to score, i.e. cortex without the medial wall.

    ``fsaverage`` has a canonical mask shipped with the templates.  The fsLR
    meshes are a different parameterisation of the same 32k vertex count, so
    that mask does not apply to them and the wall is taken instead as the
    vertices no more than half the group manages to label.
    """
    if space == "fsaverage":
        mask_file = tmpl_root / "templates_surfaces_32k" / f"{fshemi}.mask"
        if mask_file.exists():
            return np.asarray(cat_surf.read_values(str(mask_file))) != 0
        print(f"note: {mask_file} missing, deriving the {fshemi} cortex mask "
              "from the labels", file=sys.stderr)
    return (labels != 0).mean(0) > 0.5


def cmd_dice(args: argparse.Namespace) -> int:
    manifest_path = Path(args.work) / f"manifest_{args.space}.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found — run `project` first")
    manifest = json.loads(manifest_path.read_text())
    subjects = sorted(manifest["subjects"])
    if args.subjects:
        # Restricting every arm to one subject list is how a method compared
        # on a subset stays comparable with one run over everything.
        wanted = set(args.subjects)
        missing = wanted - set(subjects)
        if missing:
            raise SystemExit(
                f"not in {manifest_path.name}: " + ", ".join(sorted(missing)))
        subjects = [s for s in subjects if s in wanted]
    in_dir = Path(args.work) / args.space

    tmpl_root = Path(args.data_dir) if args.data_dir else DATA_PATH_T1PREP
    names = region_names()
    k = len(DKT_OFFSETS) + 1
    rows: list[tuple] = []

    for fshemi in ("lh", "rh"):
        labels = _compact([np.load(in_dir / f"{subject}_{fshemi}.npy")
                           for subject in subjects])
        if args.cortex_mask:
            labels = labels[:, cortex_mask(args.space, fshemi, labels,
                                           tmpl_root)]

        n, v = labels.shape
        if args.protocol in ("loo", "both"):
            counts = np.zeros((k, v), dtype=np.int32)
            cols = np.arange(v)
            for i in range(n):
                counts[labels[i], cols] += 1
            for i, subject in enumerate(subjects):
                left = counts.copy()
                left[labels[i], cols] -= 1
                left[0] = -1          # never let "unlabelled" win the vote
                pred = left.argmax(0).astype(np.int16)
                dice = _dice_from_confusion(pred, labels[i], k)
                for j, off in enumerate(DKT_OFFSETS, 1):
                    if np.isfinite(dice[j]):
                        rows.append(("loo", subject, "", fshemi,
                                     names[off], float(dice[j])))

        if args.protocol in ("pairs", "both"):
            pairs = list(combinations(range(n), 2))
            if args.max_pairs and len(pairs) > args.max_pairs:
                rng = np.random.default_rng(args.seed)
                pairs = [pairs[i] for i in
                         rng.choice(len(pairs), args.max_pairs, replace=False)]
            for ia, ib in pairs:
                dice = _dice_from_confusion(labels[ia], labels[ib], k)
                for j, off in enumerate(DKT_OFFSETS, 1):
                    if np.isfinite(dice[j]):
                        rows.append(("pairs", subjects[ia], subjects[ib],
                                     fshemi, names[off], float(dice[j])))

    if not rows:
        raise SystemExit("nothing scored")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write("protocol,subject_a,subject_b,hemi,region,dice\n")
            for r in rows:
                fh.write("%s,%s,%s,%s,%s,%.6f\n" % r)
        print(f"wrote {len(rows)} rows -> {args.csv}\n")

    report(rows, args.space, manifest)
    return 0


def report(rows: list[tuple], space: str, manifest: dict) -> None:
    """Print the headline means, the per-region table and the LOO outliers."""
    routes: dict[str, int] = {}
    for hemis in manifest.get("subjects", {}).values():
        for qc in hemis.values():
            routes[qc.get("route", "?")] = routes.get(qc.get("route", "?"), 0) + 1
    split = ", ".join(f"{k} {v // 2}" for k, v in sorted(routes.items()))

    for protocol in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == protocol]
        n = len({(r[1], r[2]) for r in sub})
        print(f"=== {protocol}  (space={space}, ground truth: {split}, "
              f"n={n} comparisons) ===")
        print(f"\n  mean Dice, both hemispheres: "
              f"{np.mean([r[5] for r in sub]):.4f}")
        for fshemi in ("lh", "rh"):
            per_region: dict[str, list[float]] = {}
            for _, _, _, h, region, d in sub:
                if h == fshemi:
                    per_region.setdefault(region, []).append(d)
            if not per_region:
                continue
            means = {r: float(np.mean(d)) for r, d in per_region.items()}
            print(f"\n  {fshemi}   mean Dice over regions: "
                  f"{np.mean(list(means.values())):.4f}")
            for region, m in sorted(means.items(), key=lambda kv: kv[1]):
                print(f"    {region:<28s} {m:.4f}"
                      f"  (sd {np.std(per_region[region]):.4f})")

        if protocol == "loo":
            # A subject well below the rest is usually a failed surface, not
            # a hard anatomy, so name the worst few for inspection.
            per_subject: dict[str, list[float]] = {}
            for _, subject, _, _, _, d in sub:
                per_subject.setdefault(subject, []).append(d)
            worst = sorted(((np.mean(v), k) for k, v in per_subject.items()))
            print("\n  lowest-scoring subjects:")
            for m, subject in worst[:5]:
                print(f"    {subject:<28s} {m:.4f}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", help="override t1prep/data (use the "
                        "source tree when the installed copy is older)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("project", help="manual labels -> template space")
    p.add_argument("--mindboggle", required=True, nargs="+",
                   help="extracted Mindboggle roots, each holding one "
                        "directory per subject; pass the surfaces and the "
                        "volumes trees together")
    p.add_argument("--t1prep", required=True,
                   help="root holding one T1Prep output directory per subject")
    p.add_argument("--work", required=True, help="where to cache the result")
    p.add_argument("--space", default="fsaverage", choices=sorted(SPACES),
                   help="which registration to evaluate (default: fsaverage)")
    p.add_argument("--labels", default="surface",
                   choices=sorted(LABEL_PATTERNS) + ["auto"],
                   help="ground truth: the labelled surfaces (default, the "
                        "primary manual product), the label volumes, or "
                        "'auto' to prefer the surfaces and fall back to the "
                        "volumes per subject")
    p.add_argument("--label-glob",
                   help="override the ground-truth file name; '{fshemi}' is "
                        "substituted")
    p.add_argument("--out-space",
                   help="name for this cache/manifest (default: --space). "
                        "Lets several registrations onto the same template "
                        "share one work directory")
    p.add_argument("--template-sphere",
                   help="target sphere overriding the one --space implies; "
                        "'{fshemi}' is substituted.  Needed when an external "
                        "registration targets a different mesh, e.g. MSMSulc "
                        "onto the 164k fs_LR sphere")
    p.add_argument("--sphere-file",
                   help="template path overriding the T1Prep sphere, e.g. "
                        "'/msm/{subject}/{hemi}.sphere.reg.gii'")
    p.add_argument("--reference-glob", default="t1weighted.nii.gz",
                   help="volume defining the scanner space of the subject's "
                        "surfaces; the labelled surfaces are shifted from "
                        "FreeSurfer tkrRAS into it (surface route only)")
    p.add_argument("--max-unlabelled", type=float, default=0.25,
                   help="drop a subject whose ground truth reaches less than "
                        "this fraction of its surface (default 0.25)")
    p.add_argument("--max-dist", type=float, default=3.0,
                   help="mm beyond which a vertex stays unlabelled")
    p.add_argument("--subjects", nargs="+", help="restrict to these subjects")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any cached manifest instead of topping it up")
    p.set_defaults(func=cmd_project)

    v = sub.add_parser("project-volume",
                       help="manual labels -> MNI via the volume warp")
    v.add_argument("--mindboggle", required=True, nargs="+")
    v.add_argument("--t1prep", required=True,
                   help="root holding one T1Prep output directory per subject")
    v.add_argument("--work", required=True)
    v.add_argument("--def-glob", default="y_*.nii",
                   help="the SPM deformation inside the subject's output "
                        "(default y_*.nii, searched one level down too)")
    v.add_argument("--label-glob",
                   help="override the manual label volume name")
    v.add_argument("--subjects", nargs="+")
    v.add_argument("--fresh", action="store_true")
    v.add_argument("--preresampled", action="store_true",
                   help="the label volumes are already in the target space "
                        "(no warp applied) -- for an affine baseline, or for "
                        "scoring another tool's normalisation")
    v.add_argument("--out-space", default="mni",
                   help="name for this cache/manifest (default: mni)")
    v.set_defaults(func=cmd_project_volume)

    d = sub.add_parser("dice", help="score the projected labels")
    d.add_argument("--work", required=True)
    d.add_argument("--space", default="fsaverage",
                   help="fsaverage/fsLR/msm for the surface spaces, or the "
                        "--out-space name a volume projection used")
    d.add_argument("--protocol", default="loo",
                   choices=("loo", "pairs", "both"))
    d.add_argument("--csv", help="write the long-format per-region rows here")
    d.add_argument("--max-pairs", type=int, default=0,
                   help="subsample the pairwise protocol (0 = all pairs)")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--subjects", nargs="+",
                   help="score only these subjects (use the same list across "
                        "arms when comparing methods on a subset)")
    d.add_argument("--no-cortex-mask", dest="cortex_mask",
                   action="store_false",
                   help="score the medial wall too (default: exclude it)")
    d.set_defaults(func=cmd_dice)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
