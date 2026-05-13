"""
t1prep.surface_estimation — Python port of the bash surface_estimation()
function from scripts/T1Prep, using the cat_surf in-process wrappers
instead of the CAT_* command-line binaries.

The function mirrors the bash original step-for-step.  Each former
``cmd="CAT_Foo …"; run_cmd_log "$cmd"`` block becomes a call to the
matching :mod:`cat_surf.cli` shim, wrapped in :func:`_run_step` for
logging and timing parity with the bash side.

Invocation
----------
This module is normally driven from the T1Prep bash script::

    python -m t1prep.surface_estimation \
        --bname SUB \
        --side left \
        --mri-dir mri \
        --surf-dir surf \
        --estimate-spherereg 1 \
        --thickness-method 3 \
        --save-pial-white 1 \
        --pre-fwhm 1.0 \
        --median-filter 2 \
        --downsample 0 \
        --vessel 1 \
        --correct-folding 0 \
        --debug 0 \
        --multi 1 \
        --nii-ext nii.gz \
        --names-tsv /path/to/Names.tsv \
        --bids-naming 0 \
        --report-log /path/to/report.log \
        --surf-templates-dir /path/to/templates_surfaces_32k \
        --atlas-templates-dir /path/to/atlases_surfaces_32k \
        --atlas-surf "aparc_DK40,aparc_a2009s" \
        --initial-surface "" \
        --progress-bar /path/to/progress_bar_multi.sh \
        --progress-count-file /path/to/.count \
        --progress-end-count 12

The bash side updates a count file in lock-step with the progress
bar; the Python port increments the same file so the multi-process
bar across hemispheres still works without code changes in T1Prep.

What this port *does not* replace yet
-------------------------------------
* The fmriprep branch (``CAT_SurfCurvature``) — no wrapper exists yet
  in cat_surf.  Pass ``--fmriprep 0``; otherwise it falls back to the
  binary.
* The atlas annot resampling (``CAT_SurfResample -label … .annot``).
  Annot read/write requires libCAT's ``read_annotation_table`` path
  which isn't surfaced in cat_surf yet.  This step also falls back
  to the binary.

Both fallbacks happen transparently — set the env var
``T1PREP_DISABLE_FALLBACK=1`` to make them raise instead.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# cat_surf imports
# ---------------------------------------------------------------------------
import cat_surf
from cat_surf import cli as cs_cli


# ===========================================================================
# Names.tsv: pattern lookup and substitution
# ===========================================================================

class NameTable:
    """In-memory representation of ``Names.tsv``.

    The file is whitespace-separated: column 0 is a symbolic code,
    column 1 is the CAT12-style filename pattern, column 2 (if present)
    is the BIDS-style filename pattern.  Patterns contain placeholders
    ``{bname}``, ``{side}``, ``{desc}``, ``{space}``, ``{atlas}``,
    ``{nii_ext}``.
    """

    def __init__(self, path: str | os.PathLike):
        self._rows: dict[str, list[str]] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                self._rows[parts[0]] = parts

    def pattern(self, code: str, column: int) -> str:
        row = self._rows.get(code)
        if row is None or column >= len(row):
            raise KeyError(
                f"Names.tsv: no pattern for code={code!r} column={column}")
        return row[column]

    def substitute(self, code: str, column: int, *,
                   bname: str, hemi: str, desc: str = "",
                   space: str = "", atlas: str = "",
                   nii_ext: str = "") -> str:
        pat = self.pattern(code, column)
        # Match utils.sh:substitute_pattern — strip _T1w suffix from bname.
        bname_clean = bname[:-4] if bname.endswith("_T1w") else bname
        out = (pat
               .replace("{bname}", bname_clean)
               .replace("{side}", hemi)
               .replace("{desc}", desc)
               .replace("{space}", space)
               .replace("{atlas}", atlas)
               .replace("{nii_ext}", nii_ext)
               .replace("..", "."))
        return out


# ===========================================================================
# Logging / step timing (mirrors run_cmd_log in utils.sh)
# ===========================================================================

def _setup_logger(report_log: Optional[str]) -> logging.Logger:
    log = logging.getLogger("t1prep.surface_estimation")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    if report_log:
        Path(report_log).parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(report_log, mode="a", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    # Mirror to stderr for live monitoring.
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(message)s"))
    sh.setLevel(logging.WARNING)
    log.addHandler(sh)
    return log


@contextmanager
def _run_step(log: logging.Logger, description: str):
    """Mimic ``run_cmd_log`` — log description + runtime."""
    log.info(description)
    t0 = time.monotonic()
    try:
        yield
    except Exception:
        log.exception("FAILED: %s", description)
        raise
    finally:
        runtime = int(time.monotonic() - t0)
        log.info("Execution time: %ds", runtime)


# ===========================================================================
# Progress-bar bridge
# ===========================================================================

class ProgressBar:
    """Drives the same ``progress_bar_multi.sh`` script T1Prep uses.

    A persistent count file is shared between hemispheres (T1Prep runs
    one ``surface_estimation`` per hemisphere in parallel).  We
    increment it atomically and invoke the bash bar with the new value.
    """

    def __init__(self, bar_script: Optional[str], end_count: int,
                 count_file: Optional[str], show: bool):
        self.bar_script = bar_script
        self.end_count = end_count
        self.count_file = count_file
        self.show = bool(show and bar_script and count_file)

    def step(self, label: str) -> None:
        if not self.show:
            return
        try:
            with open(self.count_file, "r+", encoding="utf-8") as fh:
                cur = int(fh.read().strip() or "0") + 1
                fh.seek(0)
                fh.truncate()
                fh.write(str(cur))
        except FileNotFoundError:
            cur = 1
            Path(self.count_file).write_text(str(cur))
        subprocess.run(
            [self.bar_script, "1", "", str(cur), str(self.end_count),
             f"{label:<31}"],
            check=False,
        )


# ===========================================================================
# Optional binary fallbacks
# ===========================================================================

_FALLBACK_DISABLED = os.environ.get("T1PREP_DISABLE_FALLBACK") == "1"


def _bin_fallback(argv: list[str], log: logging.Logger) -> None:
    """Run a CAT_* binary as subprocess (used for steps not yet wrapped)."""
    if _FALLBACK_DISABLED:
        raise RuntimeError(
            f"binary fallback disabled but step requires {argv[0]}")
    log.info("[fallback to binary] " + " ".join(argv))
    subprocess.run(argv, check=True)


# ===========================================================================
# The main port
# ===========================================================================

def surface_estimation(
    *,
    bname: str,
    side: str,
    mri_dir: str,
    surf_dir: str,
    estimate_spherereg: int = 1,
    thickness_method: int = 3,
    save_pial_white: int = 1,
    pre_fwhm: float = 1.0,
    median_filter: int = 2,
    downsample: float = 0.0,
    vessel: int = 1,
    correct_folding: int = 0,
    debug: int = 0,
    multi: int = -1,
    nii_ext: str = "nii.gz",
    names_tsv: str,
    bids_naming: int = 0,
    report_log: Optional[str] = None,
    surf_templates_dir: str,
    atlas_templates_dir: str,
    atlas_surf: str = "",
    initial_surface: str = "",
    fmriprep: int = 0,
    progress_bar_script: Optional[str] = None,
    progress_count_file: Optional[str] = None,
    progress_end_count: int = 0,
) -> int:
    """Port of ``surface_estimation()`` from T1Prep's bash script.

    Returns 0 on success, non-zero on failure (matching the bash
    function's ``return`` behaviour).
    """
    log = _setup_logger(report_log)

    # If mri_dir == surf_dir, the bash code uses "." for both; we cd into
    # that directory.  Otherwise both are relative paths under base_dir.
    if os.path.realpath(mri_dir) == os.path.realpath(surf_dir):
        base_dir = mri_dir
        mri_rel = "."
        surf_rel = "."
    else:
        base_dir = os.path.dirname(mri_dir)
        mri_rel = "mri"
        surf_rel = "surf"
    cwd_prev = os.getcwd()
    os.chdir(base_dir or ".")
    try:
        return _run(
            log=log,
            bname=bname, side=side,
            mri=mri_rel, surf=surf_rel,
            estimate_spherereg=estimate_spherereg,
            thickness_method=thickness_method,
            save_pial_white=save_pial_white,
            pre_fwhm=pre_fwhm, median_filter=median_filter,
            downsample=downsample, vessel=vessel,
            correct_folding=correct_folding, debug=debug, multi=multi,
            nii_ext=nii_ext, names_tsv=names_tsv,
            bids_naming=bids_naming,
            surf_templates_dir=surf_templates_dir,
            atlas_templates_dir=atlas_templates_dir,
            atlas_surf=atlas_surf,
            initial_surface=initial_surface,
            fmriprep=fmriprep,
            bar=ProgressBar(progress_bar_script, progress_end_count,
                            progress_count_file,
                            show=(multi != -2 and side == "left")),
        )
    finally:
        os.chdir(cwd_prev)


def _run(*, log, bname, side, mri, surf, estimate_spherereg,
         thickness_method, save_pial_white, pre_fwhm, median_filter,
         downsample, vessel, correct_folding, debug, multi, nii_ext,
         names_tsv, bids_naming, surf_templates_dir, atlas_templates_dir,
         atlas_surf, initial_surface, fmriprep, bar) -> int:

    # Hemisphere coding
    fshemi = "lh" if side == "left" else "rh"
    hemi = ("L" if side == "left" else "R") if bids_naming else fshemi
    name_columns = 2 if bids_naming else 1   # 0-indexed: row[1] CAT12, row[2] BIDS
    if bids_naming:
        name_columns = 2  # column index 2 == BIDS
    else:
        name_columns = 1  # column index 1 == CAT12

    names = NameTable(names_tsv)

    def f(code: str) -> str:
        return names.substitute(code, name_columns,
                                bname=bname, hemi=hemi,
                                nii_ext=nii_ext)

    # Symbolic filename codes used by the bash function
    codes = ("PBT_shape Area_shape Sulc_shape GMT_shape Mask_label "
             "Hemi_volume mT1_volume PPM_volume GMT_volume Mid_surface "
             "Pial_surface WM_surface Sphere_surface Spherereg_surface "
             "Inflated_surface Intensity_Mid Intensity_Pial Intensity_WM "
             "Topochange_volume").split()
    paths = {code: f(code) for code in codes}

    p = lambda d, c: os.path.join(d, paths[c])
    hemi_vol = p(mri, "Hemi_volume")
    if not os.path.exists(hemi_vol):
        log.warning("No hemisphere volume for %s — skipping (%s)",
                    side, hemi_vol)
        return 1

    # Freesurfer templates
    Fsavg = os.path.join(surf_templates_dir, f"{fshemi}.central.freesurfer.gii")
    Fsavgsphere = os.path.join(surf_templates_dir,
                               f"{fshemi}.sphere.freesurfer.gii")
    Fsavgmask = os.path.join(surf_templates_dir, f"{fshemi}.mask")

    verbose = bool(debug)

    # ---- Initial-surface hemisphere adaptation ----------------------------
    init_surf = initial_surface or ""
    if init_surf:
        if "lh" in init_surf:
            adapted = names.substitute_path(init_surf, fshemi) \
                if hasattr(names, "substitute_path") \
                else init_surf.replace("lh", fshemi)
        elif "-L" in init_surf:
            adapted = init_surf.replace("-L", f"-{hemi}")
        else:
            adapted = init_surf
        if not os.path.isfile(adapted):
            log.info("Surface %s adapted for %s hemisphere not found "
                     "(tried: %s). Skip use of initial surface.",
                     init_surf, side, adapted)
            init_surf = ""
        else:
            init_surf = adapted
          
    # =====================================================================
    # 1) Thickness estimation (CAT_VolThicknessPbt)
    # =====================================================================
    bar.step("Calculate thickness")
    with _run_step(log, f"CAT_VolThicknessPbt -> {p(mri, 'GMT_volume')}, "
                        f"{p(mri, 'PPM_volume')}"):
        import nibabel as nib
        img = nib.load(hemi_vol)
        vol = img.get_fdata().astype(np.float32)
        if vessel:
            vol = cat_surf.vol_blood_vessel_correction(
                vol, voxelsize=img.header.get_zooms()[:3])
        gmt, ppm, dcsf, dwm = cat_surf.vol_thickness_pbt(
            vol,
            voxelsize=img.header.get_zooms()[:3],
            n_avgs=2,
            n_median_filter=median_filter,
            median_subsample=2,
            range_val=0.45,
            correct_voxelsize=-0.75,
            sulcal_width=5.0,
            verbose=verbose,
        )

        def _save_like(arr, out_path, dtype=np.float32):
            nib.save(nib.Nifti1Image(arr.astype(dtype), img.affine,
                                     img.header),
                     out_path)
        _save_like(gmt, p(mri, "GMT_volume"))
        _save_like(ppm, p(mri, "PPM_volume"))

        # Optional downsampling (CLI does this post-PBT for storage).
        if downsample and downsample > 0:
            log.info("downsample > 0 requested but not yet implemented "
                     "in this port — output kept at native resolution")

    if not os.path.exists(p(mri, "PPM_volume")):
        log.error("Surface estimation for %s hemisphere failed: "
                  "PPM volume not produced", side)
        return 1

    # =====================================================================
    # 2) Marching cubes or copy of initial surface
    # =====================================================================
    if init_surf:
        bar.step("Use initial surface")
        log.info("Use %s as initial surface", init_surf)
        shutil.copy(init_surf, p(surf, "Mid_surface"))
    else:
        bar.step("Extract surface")
        topochange = p(mri, "Topochange_volume") if debug else None
        # Note: change-map output (3rd positional arg of CAT_VolMarchingCubes)
        # is debug-only and currently not surfaced through cat_surf — skip.
        with _run_step(log, f"CAT_VolMarchingCubes -> "
                            f"{p(surf, 'Mid_surface')}"):
            v, fcs = cat_surf.vol_marching_cubes(
                p(mri, "PPM_volume"),
                label=p(mri, "Hemi_volume"),
                threshold=0.5,
                pre_fwhm=pre_fwhm,
                iter_laplacian=50,
                n_median_filter=median_filter,
                strength_gyri_mask=0.1,
                verbose=verbose,
            )
            cat_surf.write_surface(p(surf, "Mid_surface"), v, fcs)
        if topochange:
            log.info("Topochange volume output skipped (not surfaced)")

        if downsample == 0:
            bar_label = "Reduce mesh"
            with _run_step(log, "CAT_SurfReduce ratio=0.25 aggr=7"):
                v, fcs = cat_surf.read_surface(p(surf, "Mid_surface"))
                target = int(round(0.25 * fcs.shape[0]))
                v, fcs = cat_surf.reduce_mesh(
                    v, fcs, target_faces=target,
                    aggressiveness=7.0, preserve_sharp=True,
                    verbose=verbose)
                cat_surf.write_surface(p(surf, "Mid_surface"), v, fcs)

    # =====================================================================
    # 3) Refine central surface with SurfDeform
    # =====================================================================
    bar.step("Refine central surface")
    with _run_step(log, "CAT_SurfDeform"):
        v, fcs = cat_surf.read_surface(p(surf, "Mid_surface"))
        v, fcs = cat_surf.surf_deform(
            v, fcs, p(mri, "PPM_volume"),
            w1=0.1, w2=0.1, w3=1.0, sigma=0.2,
            isovalue=0.5, iterations=75,
            remove_intersect=True, verbose=verbose,
        )
        cat_surf.write_surface(p(surf, "Mid_surface"), v, fcs)

    # Self-intersection cleanup (not supported on Windows)
    if platform.system() not in ("Windows",) and not sys.platform.startswith(
            ("cygwin", "msys")):
        with _run_step(log, "CAT_SurfRemoveIntersections"):
            v, fcs = cat_surf.read_surface(p(surf, "Mid_surface"))
            v, fcs = cat_surf.remove_intersections(v, fcs, verbose=verbose)
            cat_surf.write_surface(p(surf, "Mid_surface"), v, fcs)

    # =====================================================================
    # 4) Map thickness values onto the surface (CAT_Vol2Surf)
    # =====================================================================
    bar.step("Map thickness values")
    with _run_step(log, "CAT_Vol2Surf -weighted-avg -start -0.4 -end 0.4 -steps 5"):
        v, fcs = cat_surf.read_surface(p(surf, "Mid_surface"))
        values, _grid = cat_surf.vol2surf(
            p(mri, "GMT_volume"), v, fcs,
            grid_start=-0.4, grid_end=0.4, grid_steps=5,
            map_func="waverage",
        )
        cat_surf.write_values(p(surf, "PBT_shape"), values)

    if thickness_method == 3:
        shutil.copy(p(surf, "PBT_shape"), p(surf, "GMT_shape"))
    else:
        with _run_step(log, "CAT_SurfDistance -thickness -mean -max 6.0"):
            cs_cli.surf_distance(
                p(surf, "Mid_surface"), None,
                p(surf, "GMT_shape"),
                thickness_file=p(surf, "PBT_shape"),
                mode="mean", max_dist=6.0,
                check_intersect=True, verbose=verbose,
            )

    # =====================================================================
    # 5) Pial + white surfaces (CAT_Surf2PialWhite) + central by averaging
    # =====================================================================
    if save_pial_white or thickness_method == 2:
        bar.step("Estimate pial and white surface")
        with _run_step(log, "CAT_Surf2PialWhite method=2"):
            v, fcs = cat_surf.read_surface(p(surf, "Mid_surface"))
            t = cat_surf.read_values(p(surf, "GMT_shape"))
            pv, pf, wv, wf = cat_surf.surf_to_pial_white(
                v, fcs, t, p(mri, "Hemi_volume"),
                w1=0.05, w2=0.05, w3=0.05, sigma=0.2,
                iterations=100, gradient_iterations=0,
                method=2, verbose=verbose,
            )
            cat_surf.write_surface(p(surf, "Pial_surface"), pv, pf)
            cat_surf.write_surface(p(surf, "WM_surface"), wv, wf)
        with _run_step(log, "CAT_SurfAverage (pial+white -> central)"):
            cs_cli.surf_average(
                p(surf, "Mid_surface"),
                p(surf, "Pial_surface"), p(surf, "WM_surface"),
            )

    # =====================================================================
    # 6) Optional refined thickness via Tfs distance between pial and white
    # =====================================================================
    if thickness_method == 2:
        bar.step("Refine thickness")
        with _run_step(log, "CAT_SurfDistance pial vs white -max 6.0 -mean"):
            v1, f1 = cat_surf.read_surface(p(surf, "Pial_surface"))
            v2, f2 = cat_surf.read_surface(p(surf, "WM_surface"))
            d, _ = cat_surf.point_distance_mean(v1, f1, v2, f2,
                                                symmetric=False,
                                                max_dist=6.0)
            cat_surf.write_values(p(surf, "GMT_shape"), d)

    # =====================================================================
    # 7) Folding-based thickness correction
    # =====================================================================
    if correct_folding:
        with _run_step(log, "CAT_SurfCorrectThicknessFolding -slope 1 -max 6"):
            v, fcs = cat_surf.read_surface(p(surf, "Mid_surface"))
            t = cat_surf.read_values(p(surf, "GMT_shape"))
            t = cat_surf.correct_thickness_folding(
                v, fcs, t, slope=1.0, max_dist=6.0)
            cat_surf.write_values(p(surf, "GMT_shape"), t)

    # =====================================================================
    # 8) Debug intensity mapping
    # =====================================================================
    if debug and os.path.exists(p(mri, "mT1_volume")):
        with _run_step(log, "CAT_Vol2Surf debug intensity mapping"):
            for surf_name, out_key in [
                ("Mid_surface", "Intensity_Mid"),
                ("Pial_surface", "Intensity_Pial"),
                ("WM_surface", "Intensity_WM"),
            ]:
                if out_key != "Intensity_Mid" and not save_pial_white:
                    continue
                v, fcs = cat_surf.read_surface(p(surf, surf_name))
                values, _ = cat_surf.vol2surf(
                    p(mri, "mT1_volume"), v, fcs,
                    grid_start=0.0, grid_end=0.0, grid_steps=1,
                    map_func="mean",
                )
                cat_surf.write_values(p(surf, out_key), values)

    # =====================================================================
    # 9) Surface area
    # =====================================================================
    with _run_step(log, "CAT_SurfArea"):
        v, fcs = cat_surf.read_surface(p(surf, "Mid_surface"))
        area, _total = cat_surf.get_area(v, fcs)
        cat_surf.write_values(p(surf, "Area_shape"), area)

    # =====================================================================
    # 10) Spherical inflation + DARTEL registration + atlas resampling
    # =====================================================================
    if estimate_spherereg:
        bar.step("Spherical inflation")
        with _run_step(log, "CAT_Surf2Sphere stop_at=6"):
            v, fcs = cat_surf.read_surface(p(surf, "Mid_surface"))
            sv, sf = cat_surf.surf_to_sphere(v, fcs, stop_at=6,
                                             verbose=verbose)
            cat_surf.write_surface(p(surf, "Sphere_surface"), sv, sf)

        bar.step("Spherical registration")
        with _run_step(log, "CAT_SurfWarp -steps 2 -avg"):
            cs_cli.surf_warp(
                source_file=p(surf, "Mid_surface"),
                source_sphere_file=p(surf, "Sphere_surface"),
                target_file=Fsavg,
                target_sphere_file=Fsavgsphere,
                output_sphere_file=p(surf, "Spherereg_surface"),
                n_steps=2,
                avg=True,
                verbose=verbose,
            )

        # Atlas annot resampling — in-process via cat_surf.cli.
        for atl in [a.strip().strip("'") for a in atlas_surf.split(",") if a.strip()]:
            value = names.substitute("ATLAS_label", name_columns,
                                     bname=bname, hemi=hemi,
                                     atlas=atl, nii_ext=nii_ext)
            annot_in = os.path.join(atlas_templates_dir,
                                    f"{hemi}.{atl}.annot")
            with _run_step(log, f"CAT_SurfResample -label {atl}"):
                cs_cli.surf_resample_annot(
                    source_surface_file=Fsavg,
                    source_sphere_file=Fsavgsphere,
                    target_sphere_file=p(surf, "Spherereg_surface"),
                    annot_in_file=annot_in,
                    annot_out_file=os.path.join(surf, value),
                )

    # =====================================================================
    # 11) fmriprep additions (CAT_SurfCurvature + extra Surf2Sphere + mask
    #     resampling) — fully in-process via cat_surf.cli.
    # =====================================================================
    if fmriprep:
        with _run_step(log, "CAT_SurfCurvature (curvtype=11, invert)"):
            # CAT_SurfCurvature <surf> <out> 11 0 0 1 in the legacy CLI is:
            #   curvtype=11, fwhm=0, use_abs_values=0, invert_values=1
            cs_cli.surf_curvature(
                surface_file=p(surf, "Mid_surface"),
                output_values_file=p(surf, "Sulc_shape"),
                curvtype=11,
                fwhm=0.0,
                use_abs_values=False,
                invert_values=True,
            )
        with _run_step(log, "CAT_Surf2Sphere stop_at=2"):
            cs_cli.surf2sphere(
                surface_file=p(surf, "Mid_surface"),
                output_file=p(surf, "Mid_surface"),
                stop_at=2,
                verbose=verbose,
            )
        with _run_step(log, "CAT_SurfResample -label Fsavgmask"):
            # Fsavgmask is a per-vertex label file (not .annot) -- routed
            # through the values resampler with label_interpolation=True.
            cs_cli.surf_resample(
                surface_file_or_None=Fsavg,
                sphere_file_or_None=Fsavgsphere,
                target_sphere_file=p(surf, "Spherereg_surface"),
                output_surface_file_or_None="NULL",
                input_values_file=Fsavgmask,
                output_values_file=p(surf, "Mask_label"),
                label=True,
            )

    # =====================================================================
    # Clean up
    # =====================================================================
    if not debug:
        for f_ in (p(mri, "GMT_volume"), p(mri, "PPM_volume")):
            if os.path.exists(f_):
                try:
                    os.remove(f_)
                except OSError:
                    pass

    return 0


# ===========================================================================
# CLI entry
# ===========================================================================

def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--bname", required=True)
    ap.add_argument("--side", required=True, choices=("left", "right"))
    ap.add_argument("--mri-dir", required=True)
    ap.add_argument("--surf-dir", required=True)
    ap.add_argument("--estimate-spherereg", type=int, default=1)
    ap.add_argument("--thickness-method", type=int, default=3)
    ap.add_argument("--save-pial-white", type=int, default=1)
    ap.add_argument("--pre-fwhm", type=float, default=1.0)
    ap.add_argument("--median-filter", type=int, default=2)
    ap.add_argument("--downsample", type=float, default=0.0)
    ap.add_argument("--vessel", type=int, default=1)
    ap.add_argument("--correct-folding", type=int, default=0)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--multi", type=int, default=-1)
    ap.add_argument("--nii-ext", default="nii.gz")
    ap.add_argument("--names-tsv", required=True)
    ap.add_argument("--bids-naming", type=int, default=0)
    ap.add_argument("--report-log", default=None)
    ap.add_argument("--surf-templates-dir", required=True)
    ap.add_argument("--atlas-templates-dir", required=True)
    ap.add_argument("--atlas-surf", default="")
    ap.add_argument("--initial-surface", default="")
    ap.add_argument("--fmriprep", type=int, default=0)
    ap.add_argument("--progress-bar", default=None,
                    dest="progress_bar_script")
    ap.add_argument("--progress-count-file", default=None)
    ap.add_argument("--progress-end-count", type=int, default=0)
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    return surface_estimation(**vars(args))


if __name__ == "__main__":
    sys.exit(main())
