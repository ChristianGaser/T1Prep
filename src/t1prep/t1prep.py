"""T1Prep – pure-Python single-subject pipeline.

Replaces the former bash-subprocess wrapper with a direct Python implementation.
Accepts the same flags as ``scripts/T1Prep`` (minus the bash-only parallelisation
options ``--multi`` and ``--min-memory``) and drives the two pipeline stages
entirely in-process::

    python -m t1prep.t1prep [options] file.nii.gz

API::

    from t1prep import run_t1prep
    run_t1prep("sub-01_T1w.nii.gz", out_dir="/results", bids=True)

The pipeline executes two stages in sequence:

1. ``t1prep.segment``            – skull-stripping, segmentation, atlas ROI export
2. ``t1prep.surface_estimation`` – cortical surface extraction (both hemispheres)

Multi-subject parallelisation is left to the caller (e.g. via
``concurrent.futures`` or the original ``scripts/T1Prep`` bash pipeline).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Atlas argument normalisation
# ---------------------------------------------------------------------------

def _as_atlas_str(arg: Optional[Union[str, Sequence[str]]]) -> str:
    """Normalise an atlas argument to a comma-separated string.

    Accepts either a plain string (passed through) or a sequence of atlas
    names.  Returns an empty string when *arg* is ``None`` or empty.
    """

    if arg is None:
        return ""
    if isinstance(arg, str):
        return arg
    items = [str(a) for a in arg if a is not None and str(a) != ""]
    return ", ".join(f"'{a}'" for a in items)


# ---------------------------------------------------------------------------
# Output-directory resolution  (Python port of t1prep_output_folder_from_input)
# ---------------------------------------------------------------------------

def _resolve_output_dirs(
    input_file: str,
    out_dir: Optional[str] = None,
    version: str = "",
    use_amap: bool = False,
) -> tuple:
    """Return *(outdir0, use_subfolder, bname)*.

    Mirrors ``t1prep_output_folder_from_input`` in ``scripts/utils.sh``.

    Parameters
    ----------
    input_file:
        Path to the input NIfTI file.
    out_dir:
        Optional override for the base output directory.
    version:
        T1Prep version string (a leading ``v`` is stripped).
    use_amap:
        When ``True``, inserts ``"Amap"`` into the BIDS derivatives folder name.

    Returns
    -------
    outdir0 : Path
        Resolved output directory for this subject.
    use_subfolder : bool
        ``True`` when non-BIDS CAT12-style subdirectories (``mri/``, ``surf/``,
        ``report/``, ``label/``) should be created inside *outdir0*.
    bname : str
        Stem of the input filename (no extension).
    """
    p = Path(input_file).resolve()
    bname = p.name.removesuffix(".nii.gz").removesuffix(".nii")

    add_str = "Amap" if use_amap else ""
    version_clean = version.lstrip("v")

    dname = p.parent
    if dname.name == "anat":
        # BIDS dataset – place output in derivatives/
        use_subfolder = False
        sess_folder0 = dname.parent
        if sess_folder0.name.startswith("ses-"):
            sess_folder = sess_folder0.name
            subj_dir = sess_folder0.parent
        else:
            sess_folder = ""
            subj_dir = dname.parent
        subj_base = subj_dir.name
        dataset_root = subj_dir.parent

        base_dir = Path(out_dir).resolve() if out_dir else dataset_root
        deriv = f"T1Prep{add_str}-v{version_clean}"
        if sess_folder:
            outdir0 = base_dir / "derivatives" / deriv / subj_base / sess_folder / "anat"
        else:
            outdir0 = base_dir / "derivatives" / deriv / subj_base / "anat"
    else:
        # Non-BIDS – CAT12-style subfolders under outdir0
        use_subfolder = True
        outdir0 = Path(out_dir).resolve() if out_dir else dname

    return outdir0, use_subfolder, bname


# ---------------------------------------------------------------------------
# Segmentation subprocess helper
# ---------------------------------------------------------------------------

def _build_segment_cmd(
    input_file: str,
    mri_dir: str,
    report_dir: str,
    label_dir: str,
    *,
    atlas: str = "",
    need_surf_outputs: bool = True,
    save_mwp: bool = True,
    save_wp: bool = False,
    save_rp: bool = False,
    save_p: bool = False,
    save_csf: bool = False,
    save_lesions: bool = False,
    save_fmriprep: bool = False,
    use_amap: bool = False,
    use_bids: bool = False,
    gz: bool = False,
    verbose: bool = True,
    debug: bool = False,
    vessel: float = 1.0,
    skullstrip_only: bool = False,
    skip_skullstrip: bool = False,
    seed: int = 0,
    end_count: int = 0,
) -> List[str]:
    """Return the argument list for ``python -m t1prep.segment``."""
    cmd = [sys.executable, "-m", "t1prep.segment"]

    def flag(f: str, cond: bool) -> None:
        if cond:
            cmd.append(f)

    flag("--amap",            use_amap)
    flag("--mwp",             save_mwp)
    flag("--wp",              save_wp)
    flag("--rp",              save_rp)
    flag("--p",               save_p)
    flag("--gz",              gz)
    flag("--save-fmriprep",   save_fmriprep)
    flag("--lesions",         save_lesions)
    flag("--bids",            use_bids)
    flag("--verbose",         verbose)
    flag("--debug",           debug)
    flag("--csf",             save_csf)
    flag("--skullstrip-only", skullstrip_only)
    flag("--skip-skullstrip", skip_skullstrip)
    # --surf tells segment.py to emit hemisphere partition maps consumed by
    # surface_estimation; also required when any warped/atlas output is requested.
    flag("--surf",            need_surf_outputs)

    cmd += ["--seed",       str(seed)]
    cmd += ["--vessel",     str(vessel)]
    cmd += ["--label_dir",  label_dir]
    cmd += ["--input",      input_file]
    cmd += ["--mri_dir",    mri_dir]
    cmd += ["--report_dir", report_dir]
    cmd += ["--atlas",      atlas]
    cmd += ["--count",      str(end_count)]

    return cmd


# ---------------------------------------------------------------------------
# Surface-estimation helper
# ---------------------------------------------------------------------------

def _run_surface_estimation(
    bname: str,
    side: str,
    mri_dir: str,
    surf_dir: str,
    report_log: str,
    *,
    estimate_spherereg: bool = True,
    thickness_method: int = 3,
    save_pial_white: bool = True,
    pre_fwhm: float = 2.0,
    median_filter: int = 1,
    downsample: float = 0.0,
    vessel: float = 1.0,
    correct_folding: bool = True,
    debug: bool = False,
    gz: bool = False,
    use_bids: bool = False,
    atlas_surf: str = "",
    initial_surf: str = "",
    fmriprep: bool = False,
) -> int:
    """Call :func:`surface_estimation.surface_estimation` directly (in-process).

    Returns 0 on success, non-zero on failure.
    """
    from . import surface_estimation as _se
    from .utils import DATA_PATH_T1PREP, _resolve_names_tsv

    return _se.surface_estimation(
        bname=bname,
        side=side,
        mri_dir=mri_dir,
        surf_dir=surf_dir,
        estimate_spherereg=int(estimate_spherereg),
        thickness_method=thickness_method,
        save_pial_white=int(save_pial_white),
        pre_fwhm=pre_fwhm,
        median_filter=int(median_filter),
        downsample=downsample,
        vessel=int(vessel > 0),
        correct_folding=int(correct_folding),
        debug=int(debug),
        multi=0,
        nii_ext="nii.gz" if gz else "nii",
        names_tsv=str(_resolve_names_tsv()),
        bids_naming=int(use_bids),
        report_log=report_log,
        surf_templates_dir=str(DATA_PATH_T1PREP / "templates_surfaces_32k"),
        atlas_templates_dir=str(DATA_PATH_T1PREP / "atlases_surfaces_32k"),
        atlas_surf=atlas_surf,
        initial_surface=initial_surf or "",
        fmriprep=int(fmriprep),
        # No bash progress-bar in the pure-Python path.
        progress_bar_script=None,
        progress_count_file=None,
        progress_end_count=0,
        progress_start_count=0,
    )


# ---------------------------------------------------------------------------
# Single-subject pipeline
# ---------------------------------------------------------------------------

def _process_single(
    input_file: str,
    *,
    out_dir: Optional[str] = None,
    long_data: Optional[str] = None,
    no_overwrite: Optional[str] = None,
    pre_fwhm: float = 2.0,
    downsample: float = 0.0,
    median_filter: int = 1,
    vessel: float = 1.0,
    thickness_method: int = 3,
    seed: int = 0,
    atlas: str = "",
    atlas_surf: str = "",
    no_atlas: bool = False,
    gz: bool = False,
    hemisphere: bool = False,
    estimate_surf: bool = True,
    skullstrip_only: bool = False,
    skip_skullstrip: bool = False,
    estimate_seg: bool = True,
    estimate_spherereg: bool = True,
    pial_white: bool = True,
    lesions: bool = False,
    save_mwp: bool = True,
    wp: bool = False,
    rp: bool = False,
    p: bool = False,
    csf: bool = False,
    amap: bool = False,
    bids: bool = False,
    correct_folding: bool = True,
    fmriprep: bool = False,
    debug: bool = False,
    verbose: bool = True,
    initial_surf: str = "",
    retry: bool = True,
) -> int:
    """Process one NIfTI file through the T1Prep pipeline.

    Returns 0 on success, 1 on failure.
    """
    from . import __version__

    # ------------------------------------------------------------------ dirs
    outdir0, use_subfolder, bname = _resolve_output_dirs(
        input_file, out_dir, __version__, amap
    )

    if bids or not use_subfolder:
        mri_dir = surf_dir = report_dir = label_dir = outdir0
    else:
        mri_dir    = outdir0 / "mri"
        surf_dir   = outdir0 / "surf"
        report_dir = outdir0 / "report"
        label_dir  = outdir0 / "label"

    os.makedirs(mri_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    if estimate_surf:
        os.makedirs(surf_dir, exist_ok=True)

    # ---------------------------------------------------- --no-overwrite skip
    if no_overwrite:
        import glob as _glob
        found = any(
            _glob.glob(os.path.join(str(d), f"*{no_overwrite}*"))
            for d in {str(mri_dir), str(surf_dir)}
        )
        if found:
            if verbose:
                print(f"Skip processing of {input_file}")
            return 0

    # ------------------------------------------- --long-data input override
    actual_input = input_file
    if long_data:
        ld = Path(long_data)
        if not ld.is_absolute():
            ld = outdir0 / ld
        stem = Path(input_file).name.removesuffix(".nii.gz").removesuffix(".nii")
        for candidate in (
            ld / f"{stem}_desc-realigned.nii.gz",
            ld / f"{stem}_desc-realigned.nii",
            ld / f"r{stem}.nii.gz",
            ld / f"r{stem}.nii",
            ld / Path(input_file).name,
        ):
            if candidate.exists():
                actual_input = str(candidate)
                break

    report_log = str(report_dir / f"{bname}.log")

    start = time.perf_counter()

    # ------------------------------------------------------------------ segmentation
    if estimate_seg:
        atlas_arg = "" if no_atlas else atlas
        # --surf flag: segment.py must emit hemisphere partition maps when
        # surface estimation, hemispheric outputs, or warped outputs are needed.
        need_surf_outputs = (
            estimate_surf or hemisphere or save_mwp or wp or bool(atlas_arg)
        )
        seg_cmd = _build_segment_cmd(
            actual_input,
            str(mri_dir), str(report_dir), str(label_dir),
            atlas=atlas_arg,
            need_surf_outputs=need_surf_outputs,
            save_mwp=save_mwp,
            save_wp=wp,
            save_rp=rp,
            save_p=p,
            save_csf=csf,
            save_lesions=lesions,
            save_fmriprep=fmriprep,
            use_amap=amap,
            use_bids=bids,
            gz=gz,
            verbose=verbose,
            debug=debug,
            vessel=vessel,
            skullstrip_only=skullstrip_only,
            skip_skullstrip=skip_skullstrip,
            seed=seed,
        )
        # Ensure the package is importable inside the subprocess when running
        # from the repository in editable / non-installed mode.
        seg_env = os.environ.copy()
        src_path = str(Path(__file__).resolve().parents[2] / "src")
        existing_pp = seg_env.get("PYTHONPATH", "")
        seg_env["PYTHONPATH"] = (
            src_path + os.pathsep + existing_pp if existing_pp else src_path
        )

        seg_status = subprocess.call(seg_cmd, env=seg_env)
        if seg_status != 0 and retry:
            if verbose:
                print(f"Segmentation failed \u2014 retrying {actual_input}\u2026")
            seg_status = subprocess.call(seg_cmd, env=seg_env)
        if seg_status != 0:
            print(f"ERROR: Segmentation failed for {input_file}", file=sys.stderr)
            return 1

        if skullstrip_only:
            return 0

    # ------------------------------------------------------------------ surfaces
    if estimate_surf:
        atlas_surf_arg = "" if no_atlas else atlas_surf

        for side in ("left", "right"):
            surf_status = _run_surface_estimation(
                bname, side, str(mri_dir), str(surf_dir), report_log,
                estimate_spherereg=estimate_spherereg,
                thickness_method=thickness_method,
                save_pial_white=pial_white,
                pre_fwhm=pre_fwhm,
                median_filter=median_filter,
                downsample=downsample,
                vessel=vessel,
                correct_folding=correct_folding,
                debug=debug,
                gz=gz,
                use_bids=bids,
                atlas_surf=atlas_surf_arg,
                initial_surf=initial_surf,
                fmriprep=fmriprep,
            )
            if surf_status != 0 and retry:
                if verbose:
                    print(
                        f"Surface estimation failed for {side} hemisphere"
                        f" \u2014 retrying\u2026"
                    )
                surf_status = _run_surface_estimation(
                    bname, side, str(mri_dir), str(surf_dir), report_log,
                    estimate_spherereg=estimate_spherereg,
                    thickness_method=thickness_method,
                    save_pial_white=pial_white,
                    pre_fwhm=pre_fwhm,
                    median_filter=median_filter,
                    downsample=downsample,
                    vessel=vessel,
                    correct_folding=correct_folding,
                    debug=debug,
                    gz=gz,
                    use_bids=bids,
                    atlas_surf=atlas_surf_arg,
                    initial_surf=initial_surf,
                    fmriprep=fmriprep,
                )
            if surf_status != 0:
                print(
                    f"ERROR: Surface estimation failed for {side} hemisphere"
                    f" of {bname}",
                    file=sys.stderr,
                )
                return 1

    # ------------------------------------------------------------------ timing
    elapsed = time.perf_counter() - start
    if verbose:
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        parts: List[str] = []
        if h:
            parts.append(f"{h}hrs")
        if m:
            parts.append(f"{m}min")
        parts.append(f"{s}s")
        print(f"Finished after {' '.join(parts)}")

    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_t1prep(
    inputs: Union[str, Sequence[str]],
    *,
    out_dir: Optional[Union[str, os.PathLike]] = None,
    long_data: Optional[Union[str, os.PathLike]] = None,
    initial_surf: Optional[Union[str, os.PathLike]] = None,
    no_overwrite: Optional[str] = None,
    # numeric / tuning options
    pre_fwhm: float = 2.0,
    downsample: float = 0.0,
    median_filter: int = 1,
    vessel: float = 1.0,
    thickness_method: int = 3,
    seed: int = 0,
    # atlas options
    atlas: Optional[Union[str, Sequence[str]]] = None,
    atlas_surf: Optional[Union[str, Sequence[str]]] = None,
    no_atlas: bool = False,
    # output format
    gz: bool = False,
    bids: bool = False,
    # pipeline stages
    no_surf: bool = False,
    no_seg: bool = False,
    skullstrip_only: bool = False,
    skip_skullstrip: bool = False,
    no_sphere_reg: bool = False,
    # surface options
    pial_white: bool = True,
    no_correct_folding: bool = False,
    # tissue maps
    no_mwp: bool = False,
    wp: bool = False,
    rp: bool = False,
    p: bool = False,
    csf: bool = False,
    hemisphere: bool = False,
    # extra outputs
    lesions: bool = False,
    amap: bool = False,
    fmriprep: bool = False,
    # fast mode
    fast: bool = False,
    # misc
    debug: bool = False,
    verbose: bool = True,
    no_retry: bool = False,
) -> int:
    """Run the T1Prep pipeline for one or more input files.

    This is a pure-Python implementation that calls ``t1prep.segment``
    (via subprocess) and ``t1prep.surface_estimation`` (in-process) for
    each input file sequentially.  For batch parallelisation use
    ``scripts/T1Prep --multi N`` or dispatch :func:`run_t1prep` calls
    from multiple threads / processes at the caller's level.

    Parameters
    ----------
    inputs:
        One path or a sequence of paths to ``.nii`` / ``.nii.gz`` files.
    out_dir:
        Base output directory.  When ``None`` the output is placed next to
        the input (non-BIDS) or inside a BIDS derivatives folder (BIDS).
    long_data:
        Longitudinal data path.  Use the realigned volume at this location
        as the actual input while deriving output folders from *inputs*.
    initial_surf:
        Path to an initial surface (skips Marching Cubes, longitudinal use).
    no_overwrite:
        Filename pattern string; skip a subject when matching files already
        exist in the output directory.
    pre_fwhm:
        Pre-smoothing FWHM (mm) applied before Marching Cubes (default 2).
    downsample:
        Downsample input to this resolution (mm) before processing; 0 = off.
    median_filter:
        Number of median-filter passes to reduce topology artefacts (default 1).
    vessel:
        Blood-vessel-correction weight; set to 0 to disable (default 1).
    thickness_method:
        Cortical thickness algorithm (1 = Tfs, 2 = pial/white, 3 = PBT; default 3).
    seed:
        Random seed for reproducibility (default 0).
    atlas:
        Volumetric atlas or list of atlases for ROI estimation.
        Default: ``"'neuromorphometrics', 'cobra'"``.
    atlas_surf:
        Surface atlas or list of surface atlases.
        Default: ``"'aparc_DK40.freesurfer','aparc_a2009s.freesurfer'"``.
    no_atlas:
        Disable atlas labelling entirely.
    gz:
        Write compressed (``.nii.gz``) outputs.
    bids:
        Use BIDS derivatives naming conventions.
    no_surf:
        Skip cortical surface and thickness estimation.
    no_seg:
        Skip tissue segmentation.
    skullstrip_only:
        Run skull-stripping only and stop.
    skip_skullstrip:
        Assume input is already skull-stripped.
    no_sphere_reg:
        Skip spherical surface registration.
    pial_white:
        Estimate pial and white matter surfaces (default ``True``).
    no_correct_folding:
        Disable folding-based thickness correction.
    no_mwp:
        Do not save modulated warped tissue maps.
    wp:
        Save warped (non-modulated) tissue maps.
    rp:
        Save affine-registered tissue maps.
    p:
        Save native-space tissue maps.
    csf:
        Save CSF tissue maps in addition to GM/WM.
    hemisphere:
        Save hemispheric (lh/rh) partitions of the label map.
    lesions:
        Save WMH lesion maps.
    amap:
        Use AMAP instead of DeepMRIPrep for segmentation.
    fmriprep:
        Save fMRIPrep-compatible outputs (deformation fields, dseg, etc.).
    fast:
        Fast mode: skip spherical registration, pial/white surfaces,
        modulated warped maps, and atlas labelling.
    debug:
        Retain temporary files and save additional diagnostic outputs.
    verbose:
        Print progress messages (default ``True``).
    no_retry:
        Disable automatic retry of failed processing steps.

    Returns
    -------
    int
        Number of files that failed to process (0 = all succeeded).
    """
    if isinstance(inputs, (str, os.PathLike)):
        input_list = [str(inputs)]
    else:
        input_list = [str(i) for i in inputs]
    if not input_list:
        raise ValueError("No input files provided.")

    # --fast shortcut
    if fast:
        no_sphere_reg = True
        pial_white = False
        no_mwp = True
        hemisphere = False
        if atlas is None:
            atlas = ""
        if atlas_surf is None:
            atlas_surf = ""

    # Resolve atlas strings
    if atlas is None:
        atlas_str = "'neuromorphometrics', 'cobra'"
    else:
        atlas_str = _as_atlas_str(atlas)

    if atlas_surf is None:
        atlas_surf_str = "'aparc_DK40.freesurfer','aparc_a2009s.freesurfer'"
    else:
        atlas_surf_str = _as_atlas_str(atlas_surf)

    errors = 0
    total = len(input_list)
    for idx, input_file in enumerate(input_list, 1):
        if not os.path.isfile(input_file):
            print(f"ERROR: {input_file} not found — skipping.", file=sys.stderr)
            errors += 1
            continue

        if verbose and total > 1:
            print("-" * 58)
            print(f"{idx}/{total} Processing {input_file}")

        rc = _process_single(
            input_file,
            out_dir=str(out_dir) if out_dir else None,
            long_data=str(long_data) if long_data else None,
            no_overwrite=no_overwrite,
            pre_fwhm=float(pre_fwhm),
            downsample=float(downsample),
            median_filter=int(median_filter),
            vessel=float(vessel),
            thickness_method=int(thickness_method),
            seed=int(seed),
            atlas=atlas_str,
            atlas_surf=atlas_surf_str,
            no_atlas=no_atlas,
            gz=bool(gz),
            hemisphere=bool(hemisphere),
            estimate_surf=not no_surf,
            skullstrip_only=bool(skullstrip_only),
            skip_skullstrip=bool(skip_skullstrip),
            estimate_seg=not no_seg,
            estimate_spherereg=not no_sphere_reg,
            pial_white=bool(pial_white),
            lesions=bool(lesions),
            save_mwp=not no_mwp,
            wp=bool(wp),
            rp=bool(rp),
            p=bool(p),
            csf=bool(csf),
            amap=bool(amap),
            bids=bool(bids),
            correct_folding=not no_correct_folding,
            fmriprep=bool(fmriprep),
            debug=bool(debug),
            verbose=bool(verbose),
            initial_surf=str(initial_surf) if initial_surf else "",
            retry=not no_retry,
        )
        errors += rc

    return errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="t1prep",
        description=(
            "T1Prep – pure-Python single-subject pipeline.\n\n"
            "Preprocesses T1-weighted MRI data: skull-stripping, segmentation,\n"
            "cortical surface reconstruction, and atlas labelling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # positional
    p.add_argument("inputs", nargs="+", metavar="FILE",
                   help="Input NIfTI file(s) (.nii or .nii.gz)")

    # output options
    g = p.add_argument_group("Save options")
    g.add_argument("--out-dir", metavar="DIR",
                   help="Base output directory")
    g.add_argument("--bids", action="store_true",
                   help="Use BIDS derivatives naming conventions")
    g.add_argument("--gz", action="store_true",
                   help="Write compressed .nii.gz outputs")
    g.add_argument("--no-overwrite", metavar="PATTERN",
                   help="Skip subject if files matching PATTERN already exist")
    g.add_argument("--no-surf", action="store_true",
                   help="Skip cortical surface estimation")
    g.add_argument("--no-seg", action="store_true",
                   help="Skip tissue segmentation")
    g.add_argument("--no-sphere-reg", action="store_true",
                   help="Skip spherical surface registration")
    g.add_argument("--no-mwp", action="store_true",
                   help="Do not save modulated warped tissue maps")
    g.add_argument("--wp", action="store_true",
                   help="Save warped (non-modulated) tissue maps")
    g.add_argument("--rp", action="store_true",
                   help="Save affine-registered tissue maps")
    g.add_argument("--p", action="store_true",
                   help="Save native-space tissue maps")
    g.add_argument("--csf", action="store_true",
                   help="Include CSF in tissue map outputs")
    g.add_argument("--hemi", "--hemisphere", dest="hemisphere",
                   action="store_true",
                   help="Save hemispheric (lh/rh) partitions of the label map")
    g.add_argument("--lesions", action="store_true",
                   help="Save WMH lesion maps")
    g.add_argument("--fmriprep", action="store_true",
                   help="Save fMRIPrep-compatible outputs")
    g.add_argument("--fast", action="store_true",
                   help="Fast mode: skip sphere reg, pial/white, MWP and atlases")

    # skull-strip (mutually exclusive)
    ss = p.add_mutually_exclusive_group()
    ss.add_argument("--skullstrip-only", action="store_true",
                    help="Run skull-stripping only and stop")
    ss.add_argument("--skip-skullstrip", "--no-skullstrip",
                    dest="skip_skullstrip", action="store_true",
                    help="Skip skull-stripping (input already skull-stripped)")

    # atlas
    g2 = p.add_argument_group("Atlas options")
    g2.add_argument("--atlas", metavar="LIST",
                    default="'neuromorphometrics', 'cobra'",
                    help="Comma-separated volumetric atlas list "
                         "(default: %(default)r)")
    g2.add_argument("--atlas-surf", metavar="LIST",
                    default="'aparc_DK40.freesurfer','aparc_a2009s.freesurfer'",
                    help="Comma-separated surface atlas list "
                         "(default: %(default)r)")
    g2.add_argument("--no-atlas", action="store_true",
                    help="Disable atlas labelling")

    # expert / tuning
    g3 = p.add_argument_group("Expert options")
    g3.add_argument("--pre-fwhm", type=float, default=2.0, metavar="MM",
                    help="Pre-smoothing FWHM before Marching Cubes (default 2)")
    g3.add_argument("--downsample", type=float, default=0.0, metavar="MM",
                    help="Downsample to this resolution (0 = off)")
    g3.add_argument("--median-filter", type=int, default=1, metavar="N",
                    help="Median-filter passes (default 1)")
    g3.add_argument("--vessel", type=float, default=1.0,
                    help="Blood-vessel correction weight; 0 = off (default 1)")
    g3.add_argument("--thickness-method", type=int, default=3, metavar="N",
                    choices=[1, 2, 3],
                    help="Thickness algorithm: 1=Tfs, 2=pial/white, 3=PBT "
                         "(default 3)")
    g3.add_argument("--no-correct-folding", action="store_true",
                    help="Disable folding-based thickness correction")
    g3.add_argument("--no-pial-white", dest="pial_white",
                    action="store_false", default=True,
                    help="Skip pial and white matter surface estimation")
    g3.add_argument("--amap", action="store_true",
                    help="Use AMAP instead of DeepMRIPrep for segmentation")
    g3.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility (default 0)")
    g3.add_argument("--long-data", metavar="PATH",
                    help="Use realigned volume at PATH as the actual input "
                         "while deriving output folders from the original file")
    g3.add_argument("--initial-surf", metavar="FILE", default="",
                    help="Initial surface file for longitudinal processing")
    g3.add_argument("--no-retry", action="store_true",
                    help="Disable automatic retry of failed processing steps")
    g3.add_argument("--debug", action="store_true",
                    help="Retain temporary files and save extra diagnostic outputs")
    g3.add_argument("--verbose", action="store_true", default=True,
                    help="Print progress messages (default: on)")
    g3.add_argument("--quiet", dest="verbose", action="store_false",
                    help="Suppress progress messages")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: ``python -m t1prep.t1prep`` or ``t1prep`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    return run_t1prep(
        args.inputs,
        out_dir=args.out_dir,
        long_data=args.long_data,
        initial_surf=args.initial_surf,
        no_overwrite=args.no_overwrite,
        pre_fwhm=args.pre_fwhm,
        downsample=args.downsample,
        median_filter=args.median_filter,
        vessel=args.vessel,
        thickness_method=args.thickness_method,
        seed=args.seed,
        atlas=args.atlas,
        atlas_surf=args.atlas_surf,
        no_atlas=args.no_atlas,
        gz=args.gz,
        bids=args.bids,
        no_surf=args.no_surf,
        no_seg=args.no_seg,
        skullstrip_only=args.skullstrip_only,
        skip_skullstrip=args.skip_skullstrip,
        no_sphere_reg=args.no_sphere_reg,
        pial_white=args.pial_white,
        no_correct_folding=args.no_correct_folding,
        no_mwp=args.no_mwp,
        wp=args.wp,
        rp=args.rp,
        p=args.p,
        csf=args.csf,
        hemisphere=args.hemisphere,
        lesions=args.lesions,
        amap=args.amap,
        fmriprep=args.fmriprep,
        fast=args.fast,
        debug=args.debug,
        verbose=args.verbose,
        no_retry=args.no_retry,
    )


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["run_t1prep"]

