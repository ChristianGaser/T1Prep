"""What a T1Prep run leaves on disk for one image, and how to look at it.

A :class:`Subject` knows where the results of one image are and what they are
called — from ``Names.tsv`` and the folder rules of ``scripts/T1Prep``, the
same sources the pipeline writes them by.  From that follow the three things a
results browser needs: which phases are complete, the measures of the report,
and the calls of ``CAT_VolView`` and ``CAT_SurfView`` that show the images.

The viewers are named by their module and started as processes of their own.
Nothing here imports them, so a front end stays free of VTK and OpenGL.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..cli_help import version as t1prep_version
from ..t1prep import _resolve_output_dirs
from .._names import NameTable, name_file

VOLUME, SURFACE = "volume", "surface"
SIDES = ("left", "right")

#: Modules holding the ``main()`` of the viewers
VOLUME_VIEWER = "t1prep.gui.cat_vol_view"
SURFACE_VIEWER = "t1prep.gui.cat_surf_view"

_NII_EXTENSIONS = ("nii", "nii.gz")
_RESULT_FOLDERS = ("mri", "surf", "report", "label")


@lru_cache(maxsize=1)
def _names() -> NameTable:
    return NameTable(name_file)


def _folder_of(code: str) -> str:
    """The CAT12 subfolder a file of the name table is written to."""
    if code in ("Report_file", "Log_file"):
        return "report"
    if code == "Atlas_ROI":
        return "label"
    if code.endswith("_volume"):
        return "mri"
    return "surf"


@dataclass(frozen=True)
class Subject:
    """The results of one image.

    Attributes:
        name: The image's name without folder and ``.nii(.gz)``.
        folder: The output folder T1Prep derives for it.
        subfolders: Whether the results sit in ``mri/``, ``surf/``, ``report/``
            and ``label/`` below *folder* (CAT12 layout) or in it.
        bids: Whether the files carry BIDS names.
        input: The image itself; None for results found on disk, where only
            what T1Prep wrote is known.
    """

    name: str
    folder: Path
    subfolders: bool = True
    bids: bool = False
    input: Optional[Path] = None

    def directory(self, kind: str) -> Path:
        """Folder of the ``mri``, ``surf``, ``report`` or ``label`` files."""
        if self.bids or not self.subfolders:
            return self.folder
        return self.folder / kind

    def expected(self, code: str, side: str = "", nii_ext: str = "nii",
                 atlas: str = "") -> Path:
        """Where T1Prep writes the file *code* of ``Names.tsv``.

        Args:
            code: Symbolic name of the output, e.g. ``Label_volume``.
            side: ``"left"`` or ``"right"`` for a file per hemisphere.
            nii_ext: ``"nii"`` or ``"nii.gz"``.
            atlas: Name of the atlas, for ``ATLAS_label``.
        """
        if side:
            hemi = side[0].upper() if self.bids else f"{side[0]}h"
        else:
            hemi = ""
        name = _names().substitute(
            code, 2 if self.bids else 1, bname=self.name, hemi=hemi,
            atlas=atlas, nii_ext=nii_ext)
        return self.directory(_folder_of(code)) / name

    def find(self, code: str, side: str = "", atlas: str = "") -> Optional[Path]:
        """The file *code* if it exists, compressed or not."""
        for nii_ext in _NII_EXTENSIONS:
            try:
                path = self.expected(code, side, nii_ext, atlas)
            except KeyError:
                return None
            if path.is_file():
                return path
        return None

    def stage_done(self, stage: str, since: Optional[float] = None,
                   skullstrip_only: bool = False) -> bool:
        """Whether the results of *stage* are on disk.

        The volume phase ends by writing the report, the surfaces by writing
        the thickness of both hemispheres.

        Args:
            stage: ``"volume"`` or ``"surface"``.
            since: Only count files written at or after this time (seconds
                since the epoch), so the results of an earlier run do not
                pass for those of the one that just failed.
            skullstrip_only: The run stops after skull-stripping, which
                writes the skull-stripped image and its mask and no report.
        """
        if stage == VOLUME:
            files = [self.find("skullstripped_volume" if skullstrip_only
                               else "Report_file")]
        else:
            files = [self.find("GMT_shape", side) for side in SIDES]
        for path in files:
            if path is None:
                return False
            if since is not None:
                try:
                    if path.stat().st_mtime < since:
                        return False
                except OSError:
                    return False
        return True

    def report(self) -> Optional[dict]:
        """The parsed report, or None when there is none to read."""
        path = self.find("Report_file")
        if path is None:
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def log_file(self) -> Optional[Path]:
        """The log T1Prep kept for this image."""
        return self.find("Log_file")

    def underlay(self) -> Optional[Path]:
        """The image to show results on: the input, else the corrected one."""
        if self.input is not None and self.input.is_file():
            return self.input
        return self.find("mT1_volume")


def subject_for_input(path: os.PathLike, out_dir: Optional[os.PathLike] = None,
                      bids: bool = False, amap: bool = False) -> Subject:
    """The :class:`Subject` T1Prep's folder rules give an input image.

    Args:
        path: The image.
        out_dir: The ``--out-dir`` of the run; None puts the results next to
            the image, or into ``derivatives/`` of a BIDS dataset.
        bids: Whether the run uses ``--bids`` names.
        amap: Whether the run uses ``--amap``, which is part of the name of
            the derivatives folder.
    """
    folder, subfolders, name = _resolve_output_dirs(
        str(path), str(out_dir) if out_dir else None, t1prep_version(), amap)
    return Subject(name=name, folder=Path(folder), subfolders=bool(subfolders),
                   bids=bool(bids), input=Path(os.path.abspath(str(path))))


def _report_patterns() -> List[Tuple[re.Pattern, bool]]:
    """Regular expressions of a report's file name: (pattern, is BIDS)."""
    patterns = []
    for column, bids in ((1, False), (2, True)):
        try:
            template = _names().pattern("Report_file", column)
        except KeyError:
            continue
        expression = re.escape(template).replace(re.escape("{bname}"), "(.+)")
        patterns.append((re.compile(f"^{expression}$"), bids))
    return patterns


def find_results(folder: os.PathLike, depth: int = 6) -> List[Subject]:
    """The subjects whose reports lie in *folder* or below it.

    A report in a folder called ``report`` belongs to a CAT12 layout, whose
    other files are in ``mri/``, ``surf/`` and ``label/`` next to it; anywhere
    else the results sit in the folder of the report.

    Args:
        folder: Where to look; one of the result folders themselves
            (``report``, ``mri``, …) stands for the folder above it.
        depth: How many levels below *folder* are searched — enough for
            ``derivatives/T1Prep-v…/sub-…/ses-…/anat``.
    """
    root = Path(folder)
    if root.name in _RESULT_FOLDERS and (root.parent / "report").is_dir():
        root = root.parent
    patterns = _report_patterns()
    found: Dict[Tuple[str, str, bool], Subject] = {}
    base_depth = len(root.parts)
    for current, folders, files in os.walk(root):
        here = Path(current)
        folders[:] = sorted(name for name in folders if not name.startswith("."))
        if len(here.parts) - base_depth >= depth:
            folders[:] = []
        for name in sorted(files):
            for pattern, bids in patterns:
                match = pattern.match(name)
                if not match:
                    continue
                subfolders = not bids and here.name == "report"
                subject = Subject(
                    name=match.group(1),
                    folder=here.parent if subfolders else here,
                    subfolders=subfolders, bids=bids)
                found.setdefault((str(subject.folder), subject.name, bids), subject)
                break
    return list(found.values())


# ---------------------------------------------------------------------------
# The measures of the report
# ---------------------------------------------------------------------------

def _value(report: dict, section: str, key: str):
    entry = (report.get(section) or {}).get(key)
    return entry.get("value") if isinstance(entry, dict) else entry


def quality(report: Optional[dict]) -> Optional[Tuple[str, float]]:
    """The image quality rating of a report: ``(grade, IQR)``."""
    if not report:
        return None
    entry = (report.get("qualitymeasures") or {}).get("IQR")
    if not isinstance(entry, dict) or entry.get("value") is None:
        return None
    try:
        return str(entry.get("grade") or ""), float(entry["value"])
    except (TypeError, ValueError):
        return None


def describe(report: Optional[dict]) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """The measures of a report as sections of ``(label, text)`` rows.

    Only what the report holds is listed, so the result of a run without
    surfaces, or of an older T1Prep, simply has fewer rows.
    """
    if not report:
        return []
    sections: List[Tuple[str, List[Tuple[str, str]]]] = []

    rows: List[Tuple[str, str]] = []
    for key, label in (("IQR", "Image quality (IQR)"),
                       ("SIQR", "Structural quality (SIQR)")):
        entry = (report.get("qualitymeasures") or {}).get(key)
        if isinstance(entry, dict) and entry.get("value") is not None:
            grade = str(entry.get("grade") or "").strip()
            rows.append((label, f"{grade}  {float(entry['value']):.2f}".strip()))
    resolution = _value(report, "qualitymeasures", "res_RMS")
    if resolution is not None:
        rows.append(("Voxel size (RMS)", f"{float(resolution):.2f} mm"))
    if rows:
        sections.append(("Quality", rows))

    rows = []
    tiv = _value(report, "subjectmeasures", "vol_TIV")
    if tiv is not None:
        rows.append(("Total intracranial", f"{float(tiv):.0f} mL"))
    absolute = _value(report, "subjectmeasures", "vol_abs_CGW") or []
    relative = _value(report, "subjectmeasures", "vol_rel_CGW") or []
    # Stored as CSF, GM, WM; shown in the order they are usually read
    for index, label in ((1, "Grey matter"), (2, "White matter"), (0, "CSF")):
        if index < len(absolute):
            text = f"{float(absolute[index]):.0f} mL"
            if index < len(relative):
                text += f"  ·  {100 * float(relative[index]):.1f} %"
            rows.append((label, text))
    wmh = _value(report, "subjectmeasures", "vol_WMH")
    if wmh:
        rows.append(("WM hyperintensities", f"{float(wmh):.1f} mL"))
    if rows:
        sections.append(("Volumes", rows))

    rows = []
    euler = [_value(report, "qualitymeasures", f"euler_{hemi}")
             for hemi in ("lh", "rh")]
    if all(value is not None for value in euler):
        rows.append(("Euler number, left | right",
                     f"{int(euler[0])} | {int(euler[1])}"))
    if rows:
        sections.append(("Surfaces", rows))

    rows = []
    software = report.get("software") or {}
    if software.get("version_t1prep"):
        rows.append(("T1Prep", str(software["version_t1prep"])))
    try:
        stamp = datetime.strptime(str(software.get("date")), "%Y%m%d-%H%M%S")
        rows.append(("Processed", stamp.strftime("%d %b %Y, %H:%M")))
    except ValueError:
        pass
    if rows:
        sections.append(("Run", rows))
    return sections


# ---------------------------------------------------------------------------
# The viewers
# ---------------------------------------------------------------------------

def viewer_environment(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The environment a viewer process starts in.

    The viewer has to import the same ``t1prep`` as its caller.  In a source
    checkout that one is found through ``sys.path`` and not installed, so its
    folder is put in front of ``PYTHONPATH``.
    """
    env = dict(os.environ if environ is None else environ)
    package_root = str(Path(__file__).resolve().parents[2])
    paths = [package_root] + [
        part for part in env.get("PYTHONPATH", "").split(os.pathsep)
        if part and part != package_root]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _viewer(module: str, arguments: Sequence[object]) -> List[str]:
    return [sys.executable, "-m", module] + [str(item) for item in arguments]


def central_surfaces(subject: Subject) -> List[Path]:
    """The central surfaces that exist, left before right."""
    surfaces = [subject.find("Mid_surface", side) for side in SIDES]
    return [surface for surface in surfaces if surface is not None]


def segmentation_view(subject: Subject) -> Optional[List[str]]:
    """``CAT_VolView``: the image with the tissue label on top."""
    image, label = subject.underlay(), subject.find("Label_volume")
    if image is None or label is None:
        return None
    # The background of the label stays clear, so the skull remains visible
    return _viewer(VOLUME_VIEWER, [image, "--overlay", label, "--clip", "0", "0.5"])


def surfaces_view(subject: Subject) -> Optional[List[str]]:
    """``CAT_VolView``: the central surfaces outlined on the image."""
    image, surfaces = subject.underlay(), central_surfaces(subject)
    if image is None or not surfaces:
        return None
    return _viewer(VOLUME_VIEWER, [image] + surfaces)


def thickness_view(subject: Subject) -> Optional[List[str]]:
    """``CAT_SurfView``: the cortical thickness on the surface."""
    thickness = subject.find("GMT_shape", "left") or subject.find("GMT_shape", "right")
    if thickness is None:
        return None
    return _viewer(SURFACE_VIEWER, [thickness])


def preview(subject: Subject, stage: str, png: os.PathLike,
            size: int = 300) -> Optional[List[str]]:
    """``CAT_VolView --headless``: a picture of a phase's result, no window.

    The volume phase is shown as the outline of the tissue label on the
    image, the surfaces as the central surfaces on it.

    Args:
        subject: Whose results.
        stage: ``"volume"`` or ``"surface"``.
        png: The file the picture is written to.
        size: Approximate half-width of the picture in pixels.

    Returns:
        The command, or None while the results it needs are not there.
    """
    image = subject.underlay()
    if image is None:
        return None
    arguments: List[object] = [image]
    if stage == SURFACE:
        surfaces = central_surfaces(subject)
        if not surfaces:
            return None
        arguments += surfaces
    else:
        label = subject.find("Label_volume")
        if label is None:
            return None
        arguments += ["--contour", label]
    arguments += ["--headless", "--no-info", "--size", size, "--screenshot", png]
    return _viewer(VOLUME_VIEWER, arguments)


def preview_sources(subject: Subject, stage: str) -> List[Path]:
    """The files a :func:`preview` is drawn from; it is stale when they change."""
    image = subject.underlay()
    if image is None:
        return []
    if stage == SURFACE:
        return [image] + central_surfaces(subject)
    label = subject.find("Label_volume")
    return [image, label] if label is not None else []
