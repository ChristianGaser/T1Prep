"""Boundary-based registration of BOLD to T1w, as fMRIPrep consumes it.

fMRIPrep coregisters each BOLD run to the anatomical with FreeSurfer's
``bbregister`` and stores the result as ``from-boldref_to-T1w_mode-image
_desc-coreg_xfm.txt``.  It skips that step entirely when the file is already
there, so writing one is all it takes to substitute CAT-Surface's
``CAT_SurfBBReg`` for FreeSurfer's.

Exposed as the ``t1prep-bbreg`` console script; the BOLD reference has to come
from fMRIPrep (or another BOLD pipeline), so this runs between T1Prep's
anatomical pass and fMRIPrep's functional one rather than inside T1Prep.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

from .itk_transforms import save_affine_itk_txt

__all__ = ["bbregister", "save_boldref_to_t1w_xfm", "main"]


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
            frame.  Use fMRIPrep's ``desc-coreg_boldref``, which is what the
            transform it expects is defined against.
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


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="t1prep-bbreg",
        description="Boundary-based registration of a BOLD reference to T1Prep "
        "surfaces, written as the ITK transform fMRIPrep expects.",
    )
    parser.add_argument("--bold", required=True, help="BOLD reference volume")
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
    parser.add_argument("-o", "--out", required=True, help="output ITK transform (.txt)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """Console-script entry point for ``t1prep-bbreg``."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        matrix, cost = bbregister(
            bold=args.bold,
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

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_boldref_to_t1w_xfm(matrix, args.out)
    print(f"BBR cost {cost:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
