"""T1Prep JSON report writer.

This module provides :func:`write_t1prep_report`, which assembles a
structured JSON report for a processed subject.  The report mirrors the
CAT12 XML-report layout (``filedata``, ``software``, ``subjectmeasures``,
``qualitymeasures``) and can be refreshed after each pipeline stage so that
quality measures are progressively filled in.
"""

import functools
import glob
import hashlib
import json
import os
import platform
import re
from datetime import datetime
from pathlib import Path

from .utils import get_filenames

#: Distributions whose versions determine the numerical result of a run.
#: ``pyproject.toml`` only sets lower bounds, so two installs can differ here.
PROVENANCE_PACKAGES = (
    "numpy",
    "scipy",
    "nibabel",
    "torch",
    "tinygrad",
    "torchreg",
    "fill-voids",
    "spline-resize",
    "SplineSmooth3D",
    "nxbc",
    "cat-surf",
)


def _package_versions() -> dict:
    """Installed versions of :data:`PROVENANCE_PACKAGES` (``None`` if missing)."""
    import importlib.metadata as md

    versions = {}
    for name in PROVENANCE_PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def _sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def model_provenance() -> dict:
    """Which network weights ran, where they were loaded from, on which backend.

    The weights are checked for presence by file name only, so the SHA-256 of
    every file and a fingerprint over all of them record the weights in use
    in each subject's report, together with the backend
    (``T1PREP_NN_BACKEND``), whose results differ slightly.  Hashed once per
    process (about 120 MB).
    """
    try:
        from .nn.models import MODEL_DIR, MODEL_FILES
    except Exception as exc:  # report must never fail because of provenance
        return {"error": f"model directory unavailable: {exc}"}

    files = {}
    for name in sorted(MODEL_FILES):
        path = Path(MODEL_DIR) / name
        files[name] = _sha256(path) if path.is_file() else None
    joined = "\n".join(f"{n}:{h}" for n, h in files.items())
    try:
        from .nn.backends import backend_name

        backend = backend_name()
    except Exception as exc:  # report must never fail because of provenance
        backend = f"unavailable: {exc}"
    return {
        "backend": backend,
        "dir": str(MODEL_DIR),
        "fingerprint": hashlib.sha256(joined.encode()).hexdigest(),
        "complete": all(files.values()),
        "sha256": files,
    }


def write_t1prep_report(
    report_dir: str,
    out_name: str,
    use_bids: bool,
    input_fname: str,
    segmentation: dict = None,
) -> None:
    """Write a comprehensive T1Prep JSON report to the report directory.

    Reads the existing volume-summary JSON (Report_file), parses the
    T1Preplog file for version info and topology changed-voxel counts, and
    writes ``t1prep_{bname}.json`` (or its BIDS equivalent).  The function
    may be called both immediately after the volume pipeline (changed-voxel
    count = 0 at that point) and again after the surface pipeline to capture
    the final topology quality measure.

    The output JSON mirrors the CAT12 XML-report structure with four
    top-level sections: ``filedata``, ``software``, ``subjectmeasures``,
    and ``qualitymeasures``.

    Args:
        report_dir: Directory where report files are written.
        out_name: Base name of the subject (no extension).
        use_bids: Whether to use BIDS-style output filenames.
        input_fname: Full path to the original input T1 file.
        segmentation: What segmented the image when deepmriprep's networks
            did not: ``--unified-seg``'s entry
            (:func:`t1prep._unified_seg.report_entry`).  The report then
            records it instead of the provenance of the network weights.
    """
    code_vars = get_filenames(use_bids, out_name, "", "", "", "")

    report_json_name = code_vars.get("Report_file", "")
    log_name = code_vars.get("Log_file", "")

    report_json_path = Path(report_dir) / report_json_name
    log_path = Path(report_dir) / log_name

    # --- Parse T1Preplog for version and changed-voxel quality measure ---
    # Only the bash orchestrator writes that log; t1prep-run, the Python API
    # and Docker do not, so start from the running package and platform
    from . import __version__

    uname = platform.uname()
    t1prep_version = __version__
    system_str = f"{uname.system} {uname.release} {uname.machine}"

    if log_path.exists():
        with open(log_path, "r", errors="replace") as fh:
            log_text = fh.read()
        m = re.search(r"T1Prep version\s+(\S+)", log_text)
        if m:
            t1prep_version = m.group(1)
        m = re.search(r"\n(Darwin[^\n]*|Linux[^\n]*|Windows[^\n]*)\n", log_text)
        if m:
            system_str = m.group(1).strip()

    # --- Read existing volume summary JSON (written by save_results) ---
    vol_data: dict = {}
    if report_json_path.exists():
        with open(report_json_path, "r") as fh:
            vol_data = json.load(fh)

    # --- Build report (handle both flat and already-structured formats) ---
    if "subjectmeasures" in vol_data:
        # Already in report format (from a previous write_t1prep_report call)
        subjectmeasures = vol_data["subjectmeasures"]
        qualitymeasures = vol_data.get("qualitymeasures", {})
    else:
        # Flat format written by save_results()
        subjectmeasures = {
            "vol_TIV": {
                "value": vol_data.get("vol_tiv", {}).get("value"),
                "desc": "Total intracranial volume in mL (CSF+GM+WM incl. WMH)",
            },
            "vol_abs_CGW": {
                "value": vol_data.get("vol_CGW", {}).get("value"),
                "desc": "Absolute tissue volumes in mL [CSF, GM, WM incl. WMH]",
            },
            "vol_rel_CGW": {
                "value": vol_data.get("vol_rel_CGW", {}).get("value"),
                "desc": "Relative tissue volumes (fraction of TIV) "
                "[CSF, GM, WM incl. WMH]",
            },
            "vol_WMH": {
                "value": vol_data.get("vol_WMH", {}).get("value", 0.0),
                "desc": "WMH volume in mL",
            },
            "vol_rel_WMH": {
                "value": vol_data.get("WMH_rel_WM", {}).get("value", 0.0),
                "desc": "WMH volume relative to WM (incl. WMH)",
            },
        }
        qualitymeasures = vol_data.get("qualitymeasures", {})

    report = {
        "filedata": {
            "path": str(Path(input_fname).parent),
            "bname": out_name,
            "fname": str(input_fname),
        },
        "software": {
            "version_t1prep": t1prep_version,
            "version_python": platform.python_version(),
            "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "system": system_str,
            "versions": _package_versions(),
            "segmentation": segmentation or {"method": "deepmriprep"},
            "models": model_provenance() if segmentation is None else None,
        },
        "subjectmeasures": subjectmeasures,
        "qualitymeasures": qualitymeasures,
    }

    with open(report_json_path, "w") as fh:
        json.dump(report, fh, indent=2)


def merge_surface_measures(report_dir: str, bname: str) -> None:
    """Fold the per-hemisphere surface QA sidecars into the report JSON.

    The sidecars carry the glued-sulcus measure, the shape of the thickness
    distribution (see :mod:`t1prep.thickness_qa`) and the reference the
    sulcal-barrier gate of PBT was derived from.
    ``surface_estimation`` runs the two hemispheres as concurrent subprocesses
    that both resolve the same ``Report_file``, so each writes its own sidecar
    and the merge happens here, once both have finished.  Purely additive: a
    missing or unreadable sidecar, or a measure missing from it, leaves the
    report untouched.

    Both orchestrators call it once the surfaces of a subject are done:
    ``t1prep-run`` directly, ``scripts/T1Prep`` through
    ``merge_surface_report`` in ``T1Prep_utils.sh``.  Up to 0.9.0 only
    ``t1prep-run`` did, so a report written by ``T1Prep`` lacked these
    measures and the sidecars stayed behind.

    Args:
        report_dir: Folder of the report and the sidecars.
        bname: Name of the subject as the sidecars carry it.
    """
    from .thickness_qa import HIGH_FACTOR, LOWER_CUT

    sidecars = sorted(
        glob.glob(os.path.join(report_dir, f"{bname}_glued-*.json"))
    )
    if not sidecars:
        return
    entries = {}
    report_file = ""
    for path in sidecars:
        try:
            with open(path) as fh:
                data = json.load(fh)
            entries[data["hemi"]] = data
            report_file = data.get("report_file") or report_file
        except (OSError, ValueError, KeyError):
            continue
    report_path = os.path.join(report_dir, report_file) if report_file else ""
    if entries and report_path and os.path.exists(report_path):
        try:
            with open(report_path) as fh:
                report = json.load(fh)
            qa = report.setdefault("qualitymeasures", {})
            for hemi, data in entries.items():
                if data.get("glued_fraction") is not None:
                    qa[f"glued_{hemi}"] = {
                        "value": round(100.0 * data["glued_fraction"], 4),
                        "desc": (
                            "Percentage of central-surface vertices touching a "
                            "facing patch of the same surface (glued/buried "
                            "sulci; ideal = 0; lower is better)"
                        ),
                    }
                if data.get("thickness_upper_skewness") is not None:
                    qa[f"thickness_skew_{hemi}"] = {
                        "value": round(data["thickness_upper_skewness"], 4),
                        "desc": (
                            "Skewness of the cortical thickness over vertices "
                            f">= {LOWER_CUT:g} mm (medial wall excluded); a "
                            "long upper tail (higher value) points at "
                            "residual overestimation such as unopened glued "
                            "sulci"
                        ),
                    }
                if data.get("thickness_high_fraction") is not None:
                    qa[f"thickness_high_{hemi}"] = {
                        "value": round(100.0 * data["thickness_high_fraction"], 4),
                        "desc": (
                            f"Percentage of vertices thicker than {HIGH_FACTOR:g}x "
                            "the median thickness of this hemisphere"
                        ),
                    }
                if data.get("sulci_sigma_factor") is not None:
                    qa[f"glued_{hemi}_sigma"] = {
                        "value": data["sulci_sigma_factor"],
                        "desc": (
                            "sulci_sigma_factor used by CAT_VolMarchingCubes "
                            "for this hemisphere"
                        ),
                    }
                if data.get("barrier_reference") is not None:
                    qa[f"barrier_ref_{hemi}"] = {
                        "value": round(data["barrier_reference"], 4),
                        "desc": (
                            "Reference thickness in mm the sulcal-barrier "
                            "gate of this hemisphere is derived from (trimmed "
                            "mean of dist_WM + dist_CSF in the cortical band)"
                        ),
                    }
                if data.get("barrier_reference_shared") is not None:
                    qa["barrier_ref_shared"] = {
                        "value": round(data["barrier_reference_shared"], 4),
                        "desc": (
                            "Mean of both hemisphere references; the barrier "
                            "gate of both hemispheres is barrier_gmtfactor "
                            "times this value"
                        ),
                    }
            skew = {h: d.get("thickness_upper_skewness")
                    for h, d in entries.items()}
            if skew.get("lh") is not None and skew.get("rh") is not None:
                # The two hemispheres of a subject agree closely (rank
                # correlation 0.95 on 38 test hemispheres), so their
                # difference is the most sensitive use of the measure.
                qa["thickness_skew_asym"] = {
                    "value": round(skew["lh"] - skew["rh"], 4),
                    "desc": (
                        "Left minus right thickness skewness; a large "
                        "difference points at residual overestimation in "
                        "one hemisphere"
                    ),
                }
            with open(report_path, "w") as fh:
                json.dump(report, fh, indent=2)
        except (OSError, ValueError):
            pass
    for path in sidecars:
        try:
            os.remove(path)
        except OSError:
            pass
