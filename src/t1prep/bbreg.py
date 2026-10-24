"""Boundary-based registration of BOLD to T1w, as fMRIPrep consumes it.

fMRIPrep coregisters each BOLD run to the anatomical with FreeSurfer's
``bbregister`` and stores the result as ``from-boldref_to-T1w_mode-image
_desc-coreg_xfm.txt``.  It skips that step entirely when the file is already
there, so writing one is all it takes to substitute CAT-Surface's
``CAT_SurfBBReg`` for FreeSurfer's.

The transform is defined against a BOLD reference, and it is only right for
fMRIPrep if fMRIPrep uses that same reference.  Given the 4-D series, this
tool therefore makes the reference itself -- the series realigned with
CAT-Surface and averaged -- and with ``--fmriprep-dir`` writes it under the
names fMRIPrep reuses (``desc-hmc_boldref``, ``desc-coreg_boldref``), together
with the transform and, with ``--hmc``, the head-motion transforms.  fMRIPrep
then skips its reference, head-motion and coregistration stages.  Without
``--hmc`` it still estimates head motion itself, against that reference.

Exposed as the ``t1prep-bbreg`` console script.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

# Command-line behaviour shared by every T1Prep tool: no argument prints the
# synopsis, --help the full description, and only "--" options are advertised.
from .cli_help import ArgumentParser
from .transforms import save_affine_itk_txt, save_affines_itk_txt

__all__ = [
    "bbregister",
    "fmriprep_paths",
    "realign",
    "save_boldref_to_t1w_xfm",
    "save_hmc_xfm",
    "main",
]

#: Entities of a raw BOLD file name, in BIDS order; the derivatives are named
#: after these, so ``desc``, ``space`` and the suffix of a reference are dropped
_RUN_ENTITIES = ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "echo", "part")


def _read_surface(path: str):
    """Load a surface as the ``(vertices, faces)`` pair CAT-Surface expects."""
    import cat_surf

    vertices, faces = cat_surf.read_surface(path)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _read_values(path: Optional[str]):
    if not path:
        return None
    import cat_surf

    return np.asarray(cat_surf.read_values(path), dtype=np.float32)


def bbregister(
    bold: str,
    lh_white: Optional[str] = None,
    rh_white: Optional[str] = None,
    t1w: Optional[str] = None,
    lh_cortex_mask: Optional[str] = None,
    rh_cortex_mask: Optional[str] = None,
    lh_thickness: Optional[str] = None,
    rh_thickness: Optional[str] = None,
    gm_proj_frac: float = 0.0,
    verbose: bool = False,
    **kwargs,
):
    """Register a BOLD reference to the white surfaces with the BBR cost.

    Args:
        bold: BOLD reference volume; a 4-D series is reduced to its middle
            frame.  The transform is defined against this reference, so use
            the one fMRIPrep will use: its own ``desc-coreg_boldref``, or the
            reference :func:`realign` makes, written where fMRIPrep finds it.
        lh_white: Left white-matter surface, in the same RAS space as ``t1w``.
        rh_white: Right white-matter surface.
        t1w: Anatomical reference.  When given, a normalised-mutual-information
            volume registration initialises the transform before BBR, which is
            what makes the search robust to a poor starting position.
        lh_cortex_mask: Per-vertex left cortex mask; vertices at or below 0.5
            are dropped from the cost, as FreeSurfer's ``?h.cortex.label`` does.
        rh_cortex_mask: Per-vertex right cortex mask.
        lh_thickness: Left cortical thickness, enabling ``gm_proj_frac``.
        rh_thickness: Right cortical thickness.
        gm_proj_frac: Sample the grey-matter side at this fraction of the local
            thickness instead of a fixed distance.  Needs the thickness files.
        verbose: Forwarded to CAT-Surface.
        **kwargs: Passed through to ``cat_surf._bbreg.bbreg`` (``wm_dist``,
            ``slope``, ``grid_range_mm``, ``invert_contrast``, ...).

    Returns:
        Tuple ``(matrix, cost)``: the 4x4 RAS transform mapping **BOLD points
        to T1w points**, and the final BBR cost (lower is better).

    Raises:
        ValueError: If neither hemisphere surface is given.
    """
    if not lh_white and not rh_white:
        raise ValueError("at least one of lh_white / rh_white is required")

    from cat_surf import _bbreg

    matrix, cost = _bbreg.bbreg(
        volume_file=bold,
        lh_surface=_read_surface(lh_white) if lh_white else None,
        rh_surface=_read_surface(rh_white) if rh_white else None,
        ref_file=t1w,
        lh_mask=_read_values(lh_cortex_mask),
        rh_mask=_read_values(rh_cortex_mask),
        lh_thickness=_read_values(lh_thickness),
        rh_thickness=_read_values(rh_thickness),
        gm_proj_frac=gm_proj_frac,
        verbose=verbose,
        **kwargs,
    )
    return np.asarray(matrix, dtype=np.float64), float(cost)


def realign(bold: str):
    """Realign a 4-D BOLD series with CAT-Surface and average it.

    Every volume is registered rigidly to the middle one, as mcflirt does
    (``cat_surf.vol_realign``), and the realigned series is averaged into the
    reference.

    Args:
        bold: The 4-D BOLD series.

    Returns:
        Tuple ``(reference, transforms)``: the reference as a 3-D NIfTI image
        in the series' grid, and the ``(volumes, 4, 4)`` RAS transforms that
        map each volume's points onto the reference's.
    """
    import cat_surf
    import nibabel as nib

    img = nib.load(bold)
    result = cat_surf.vol_realign(np.asanyarray(img.dataobj, dtype=np.float32), img.affine)
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    mean = result["realigned"].mean(axis=-1).astype(np.float32)
    return nib.Nifti1Image(mean, img.affine, header), np.asarray(result["ras"], dtype=np.float64)


def fmriprep_paths(bold: str, out_dir: str) -> dict:
    """The files fMRIPrep reuses for the run ``bold`` belongs to.

    fMRIPrep 25 and its development version look for these names in
    ``--derivatives`` and skip the stages that would compute them.

    Args:
        bold: The BOLD series or a reference of it, named after the BIDS run
            (``sub-01_task-rest_run-1_bold.nii.gz``, or fMRIPrep's
            ``..._desc-coreg_boldref.nii.gz``).
        out_dir: Root of the derivatives dataset, e.g. T1Prep's output folder.

    Returns:
        Paths keyed ``hmc_boldref``, ``coreg_boldref``, ``hmc_xfm`` and
        ``coreg_xfm``.

    Raises:
        ValueError: If the file name carries no ``sub-`` entity.
    """
    entities = {}
    for token in os.path.basename(bold).split("_"):
        key, sep, value = token.partition("-")
        if sep and key in _RUN_ENTITIES:
            entities[key] = value
    if "sub" not in entities:
        raise ValueError(f"cannot tell the subject from {bold!r}; name it as in BIDS")
    prefix = "_".join(f"{key}-{entities[key]}" for key in _RUN_ENTITIES if key in entities)
    session = [f"ses-{entities['ses']}"] if "ses" in entities else []
    stem = os.path.join(out_dir, f"sub-{entities['sub']}", *session, "func", prefix)
    return {
        "hmc_boldref": f"{stem}_desc-hmc_boldref.nii.gz",
        "coreg_boldref": f"{stem}_desc-coreg_boldref.nii.gz",
        "hmc_xfm": f"{stem}_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt",
        "coreg_xfm": f"{stem}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt",
    }


def save_boldref_to_t1w_xfm(matrix: np.ndarray, out_path: str) -> None:
    """Write the ``from-boldref_to-T1w`` transform fMRIPrep reads.

    ITK stores the mapping from the *output* image's space back to the
    *input*'s, which is the reverse of what the BIDS ``from``/``to`` entities
    name -- so the file that resamples BOLD *images* into T1w space holds the
    T1w-to-BOLD *point* mapping, and :func:`bbregister`'s matrix is inverted
    before writing.

    Args:
        matrix: 4x4 RAS transform from BOLD points to T1w points.
        out_path: Destination ``..._mode-image_desc-coreg_xfm.txt``.
    """
    save_affine_itk_txt(np.linalg.inv(np.asarray(matrix, dtype=np.float64)), out_path)


def save_hmc_xfm(transforms: np.ndarray, out_path: str) -> None:
    """Write head-motion transforms as the ``from-orig_to-boldref`` file fMRIPrep reads.

    One ITK affine per volume, inverted for the reason given in
    :func:`save_boldref_to_t1w_xfm`: each maps reference points back to that
    volume's points.

    Args:
        transforms: ``(volumes, 4, 4)`` RAS transforms from volume points to
            reference points, as :func:`realign` returns them.
        out_path: Destination ``..._from-orig_to-boldref_mode-image_desc-hmc_xfm.txt``.
    """
    save_affines_itk_txt([np.linalg.inv(t) for t in np.asarray(transforms)], out_path)


def _parse_args(argv):
    parser = ArgumentParser(
        prog="t1prep-bbreg",
        description="Boundary-based registration of a BOLD reference to T1Prep "
        "surfaces, written as the ITK transform fMRIPrep expects.  Given the 4-D "
        "BOLD series and --fmriprep-dir, the reference is made here and written "
        "where fMRIPrep reuses it.",
    )
    parser.add_argument(
        "--bold",
        required=True,
        help="BOLD reference volume, or the 4-D series: realigned with CAT-Surface and "
        "averaged into the reference (needs --fmriprep-dir)",
    )
    parser.add_argument("--lh-white", help="left white-matter surface")
    parser.add_argument("--rh-white", help="right white-matter surface")
    parser.add_argument("--t1w", help="anatomical reference for the NMI initialisation")
    parser.add_argument("--lh-cortex-mask")
    parser.add_argument("--rh-cortex-mask")
    parser.add_argument("--lh-thickness")
    parser.add_argument("--rh-thickness")
    parser.add_argument("--gm-proj-frac", type=float, default=0.0)
    parser.add_argument("--wm-dist", type=float, default=0.5)
    parser.add_argument(
        "--contrast",
        choices=("auto", "t1", "t2"),
        default="auto",
        help="tissue contrast of the moving volume; BOLD is t2 (default: auto-detect)",
    )
    parser.add_argument(
        "--fmriprep-dir",
        help="derivatives dataset to write the reference and the transforms to, named "
        "as fMRIPrep reuses them (e.g. the folder passed to its --derivatives)",
    )
    parser.add_argument(
        "--hmc",
        action="store_true",
        help="with the 4-D series, also write the head-motion transforms, so that "
        "fMRIPrep does not estimate them itself",
    )
    parser.add_argument("--out", "-o", help="output ITK transform (.txt)")
    parser.add_argument("--verbose", action="store_true")
    # Called with nothing at all: the synopsis says what the tool takes,
    # '--help' then gives the full description
    parser.exit_without_arguments(argv)
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """Console-script entry point for ``t1prep-bbreg``."""
    import nibabel as nib

    from .bids_derivatives import write_dataset_description

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    bold_img = nib.load(args.bold)
    series = bold_img.ndim == 4 and bold_img.shape[3] > 1
    # The transform is only right with the reference it was estimated on, so a
    # reference made here has to be written where fMRIPrep finds it
    problem = None
    if not (args.out or args.fmriprep_dir):
        problem = "give --out or --fmriprep-dir"
    elif series and not args.fmriprep_dir:
        problem = "a 4-D --bold needs --fmriprep-dir, which receives the reference"
    elif args.hmc and not series:
        problem = "--hmc needs the 4-D BOLD series as --bold"
    if problem:
        print(f"t1prep-bbreg: {problem}", file=sys.stderr)
        return 2

    reference = args.bold
    paths = None
    if args.fmriprep_dir:
        try:
            paths = fmriprep_paths(args.bold, args.fmriprep_dir)
        except ValueError as exc:
            print(f"t1prep-bbreg: {exc}", file=sys.stderr)
            return 2
        os.makedirs(os.path.dirname(paths["coreg_boldref"]), exist_ok=True)
        write_dataset_description(os.path.dirname(paths["coreg_boldref"]))
        if series:
            ref_img, hmc = realign(args.bold)
            if args.hmc:
                save_hmc_xfm(hmc, paths["hmc_xfm"])
        else:
            ref_img = nib.squeeze_image(bold_img)
        # fMRIPrep realigns to the head-motion reference and coregisters the
        # other one, so both are this reference
        for key in ("hmc_boldref", "coreg_boldref"):
            ref_img.to_filename(paths[key])
        reference = paths["coreg_boldref"]

    try:
        matrix, cost = bbregister(
            bold=reference,
            lh_white=args.lh_white,
            rh_white=args.rh_white,
            t1w=args.t1w,
            lh_cortex_mask=args.lh_cortex_mask,
            rh_cortex_mask=args.rh_cortex_mask,
            lh_thickness=args.lh_thickness,
            rh_thickness=args.rh_thickness,
            gm_proj_frac=args.gm_proj_frac,
            wm_dist=args.wm_dist,
            invert_contrast={"auto": -1, "t1": 0, "t2": 1}[args.contrast],
            verbose=args.verbose,
        )
    except ValueError as exc:
        print(f"t1prep-bbreg: {exc}", file=sys.stderr)
        return 2

    outputs = [path for path in (args.out, paths and paths["coreg_xfm"]) if path]
    for out in outputs:
        out_dir = os.path.dirname(os.path.abspath(out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_boldref_to_t1w_xfm(matrix, out)
    print(f"BBR cost {cost:.4f} -> {', '.join(outputs)}")
    if paths:
        written = ["hmc_boldref", "coreg_boldref"] + (["hmc_xfm"] if args.hmc else [])
        for key in written:
            print(f"  {paths[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
