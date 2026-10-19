"""T1Prep JSON report writer.

This module provides :func:`write_t1prep_report`, which assembles a
structured JSON report for a processed subject.  The report mirrors the
CAT12 XML-report layout (``filedata``, ``software``, ``subjectmeasures``,
``qualitymeasures``) and can be refreshed after each pipeline stage so that
quality measures are progressively filled in.
"""

import functools
import hashlib
import json
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
    "deepmriprep",
    "deepbet",
    "torchreg",
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
    """Where the model weights were loaded from and what they were.

    The weights are only checked for presence by file name, and
    ``_resolve_model_dir`` prefers deepmriprep's own model directory whenever
    it holds a complete set, so the weights in use can change with a
    deepmriprep update.  The SHA-256 of every file and a fingerprint over all
    of them make that visible in each subject's report.  Hashed once per
    process (about 210 MB).
    """
    try:
        from ._models import MODEL_DIR, MODEL_FILES
    except Exception as exc:  # report must never fail because of provenance
        return {"error": f"model directory unavailable: {exc}"}

    files = {}
    for name in sorted(MODEL_FILES):
        path = Path(MODEL_DIR) / name
        files[name] = _sha256(path) if path.is_file() else None
    joined = "\n".join(f"{n}:{h}" for n, h in files.items())
    return {
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
    """
    code_vars = get_filenames(use_bids, out_name, "", "", "", "")

    report_json_name = code_vars.get("Report_file", "")
    log_name = code_vars.get("Log_file", "")

    report_json_path = Path(report_dir) / report_json_name
    log_path = Path(report_dir) / log_name

    # --- Parse T1Preplog for version and changed-voxel quality measure ---
    t1prep_version = "unknown"
    system_str = "unknown"
    changed_voxels_total = 0

    if log_path.exists():
        with open(log_path, "r", errors="replace") as fh:
            log_text = fh.read()
        m = re.search(r"T1Prep version\s+(\S+)", log_text)
        if m:
            t1prep_version = m.group(1)
        m = re.search(r"\n(Darwin[^\n]*|Linux[^\n]*|Windows[^\n]*)\n", log_text)
        if m:
            system_str = m.group(1).strip()
        # Sum all occurrences of "(N voxel changed)" from topology-fixing steps
        changed_voxels_total = sum(
            int(x) for x in re.findall(r"\((\d+)\s+voxel changed\)", log_text)
        )

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
            "models": model_provenance(),
        },
        "subjectmeasures": subjectmeasures,
        "qualitymeasures": qualitymeasures,
    }

    with open(report_json_path, "w") as fh:
        json.dump(report, fh, indent=2)
