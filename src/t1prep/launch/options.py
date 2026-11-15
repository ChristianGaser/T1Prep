"""The options a front end offers, and the T1Prep arguments they become.

One table, :data:`OPTIONS`, says for every option of ``scripts/T1Prep`` that a
graphical front end exposes how it is labelled, which variable of
``T1Prep_defaults.txt`` holds its default and which flag switches it.  A front
end builds its form from the table and hands the values to
:func:`build_arguments`; it never spells a flag itself.
``tests/test_launch_options.py`` holds every flag against the option parser of
``scripts/T1Prep``, so an option that is renamed or removed there fails in the
test suite rather than at the user's desk.

The helpers around the table — locating the orchestrator and its defaults
file, reading that file, listing the shipped atlases — are what the web UI
used to keep to itself; ``t1prep.webui.app`` imports them from here.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..nn.backends import BACKEND_ENV, backend_names

_PKG_DIR = Path(__file__).resolve().parents[1]              # .../t1prep
_repo_root_candidate = _PKG_DIR.parents[1]                   # .../<repo> in a checkout
_REPO_ROOT = (
    _repo_root_candidate
    if (_repo_root_candidate / "scripts" / "T1Prep").exists()
    else None
)


# ---------------------------------------------------------------------------
# The orchestrator, its defaults file and the shipped atlases
# ---------------------------------------------------------------------------

def data_dir() -> Path:
    """The packaged ``t1prep/data`` folder (installed) or the source tree's."""
    try:
        from importlib.resources import files as _res_files

        path = Path(str(_res_files("t1prep.data")))
        if path.is_dir():
            return path
    except Exception:
        pass
    return _PKG_DIR / "data"


def t1prep_command() -> str:
    """Locate the bash ``T1Prep`` orchestrator (needs --multi/--min-memory).

    Prefer one on PATH (the ``bin/`` of the environment once installed); fall
    back to ``scripts/T1Prep`` of a source checkout.
    """
    found = shutil.which("T1Prep")
    if found:
        return found
    if _REPO_ROOT is not None:
        return str(_REPO_ROOT / "scripts" / "T1Prep")
    return "T1Prep"


def defaults_file(command: Optional[str] = None) -> Path:
    """Return the defaults file of the T1Prep at *command*.

    The lookup of ``scripts/T1Prep``: beside the script's folder (a source
    checkout keeps the file at the repository root), else in that folder
    (``bin/`` of an installed environment).
    """
    script_dir = Path(command or t1prep_command()).parent
    beside = script_dir.parent / "T1Prep_defaults.txt"
    return beside if beside.is_file() else script_dir / "T1Prep_defaults.txt"


def strip_quotes(value: str) -> str:
    """Drop one pair of matching quotes around *value*."""
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def load_defaults(path: Path) -> dict:
    """Read the ``name=value`` lines of a T1Prep defaults file.

    The file is a bash fragment; only plain assignments are understood, which
    is all the shipped file holds.  A missing file gives an empty mapping, so
    every option falls back to the value T1Prep ships with.
    """
    defaults: Dict[str, str] = {}
    if not path.exists():
        return defaults

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        defaults[key.strip()] = strip_quotes(value.strip())
    return defaults


def format_command(cmd: Sequence[str]) -> str:
    """Spell an argument list as one line for a log, quoting where needed."""
    formatted = []
    for arg in cmd:
        if arg == "" or any(ch in arg for ch in [" ", ",", "\t"]):
            escaped = arg.replace('"', '\\"')
            formatted.append(f'"{escaped}"')
        else:
            formatted.append(arg)
    return " ".join(formatted)


def list_atlas_names(directory: Optional[Path] = None) -> List[str]:
    """Names of the volume atlases T1Prep ships: an image with a ``.csv``."""
    directory = Path(directory) if directory else (
        data_dir() / "templates_MNI152NLin2009cAsym")
    if not directory.exists():
        return []
    names: List[str] = []
    for path in sorted(directory.glob("*.nii*")):
        name = path.name
        if name.endswith(".nii.gz"):
            name = name[:-7]
        elif name.endswith(".nii"):
            name = name[:-4]
        if not (directory / f"{name}.csv").exists():
            continue
        if name and name not in names:
            names.append(name)
    return names


def list_surface_atlas_names(directory: Optional[Path] = None) -> List[str]:
    """Names of the surface atlases T1Prep ships (``lh.<name>.annot``)."""
    directory = Path(directory) if directory else (
        data_dir() / "atlases_surfaces_32k")
    if not directory.exists():
        return []
    names: List[str] = []
    for path in sorted(directory.glob("lh.*.annot")):
        name = path.name[3:-6]
        if name and name not in names:
            names.append(name)
    return names


def parse_atlas_string(value: str) -> List[str]:
    """Split an atlas list such as ``"'suit', 'cobra'"`` into its names."""
    if not value:
        return []
    parts = [item.strip().strip("'\"") for item in value.split(",")]
    return [item for item in parts if item]


def selected_atlases(names: Iterable[str], value: str) -> Set[str]:
    """Return the *names* that the atlas list *value* asks for.

    Case-insensitive, as the pipeline looks atlases up (the defaults file
    says ``neuromorphometrics``, the file is ``Neuromorphometrics.nii.gz``).
    """
    wanted = {item.lower() for item in parse_atlas_string(value)}
    return {name for name in names if name.lower() in wanted}


def build_atlas_arg(atlas_items: Sequence[str]) -> Optional[str]:
    """Spell names as T1Prep's atlas list, ``'a','b'``; None for no names."""
    if not atlas_items:
        return None
    return ",".join(f"'{item}'" for item in atlas_items)


# ---------------------------------------------------------------------------
# The option table
# ---------------------------------------------------------------------------

#: Sections of the form, in the order they are shown: (key, title, open at
#: the first start).  What most runs touch is open, the rest is one click away.
GROUPS: Tuple[Tuple[str, str, bool], ...] = (
    ("steps", "Steps", True),
    ("volume", "Volume outputs", True),
    ("surface", "Surface outputs", True),
    ("files", "Files", False),
    ("processing", "Processing", False),
    ("expert", "Expert", False),
)


@dataclass(frozen=True)
class Option:
    """One option of T1Prep as a front end shows it.

    Attributes:
        key: Name of the value in the dictionaries this module exchanges.
        label: What the control is called.
        group: Section of the form, a key of :data:`GROUPS`.
        kind: ``"bool"`` (check box), ``"int"`` (number), ``"choice"`` (one of
            several) or ``"atlas"`` (any of the shipped atlases).
        variable: Variable of ``T1Prep_defaults.txt`` holding the default.
        fallback: The default T1Prep ships with, for a file that leaves the
            variable out.
        on, off: Flags switching a ``bool`` on and off.  Many options have
            only one of them: T1Prep can switch them one way only.
        flag: Option taking the value of an ``int`` or a ``choice``.
        choices: ``(value, label)`` pairs of a ``choice``.
        flags: ``(value, flag)`` pairs of a ``choice`` that is spelled as one
            flag per value instead of ``flag value``.
        minimum, maximum, suffix: Range and unit of an ``int``.
        auto: Value of an ``int`` that T1Prep takes for "work it out"; the
            form shows it as *Auto* at the lower end of the range.
        needs: ``"seg"`` or ``"surf"`` when the option only matters while
            that step runs; the form disables it otherwise.
        help: One or two sentences for the tool tip.
    """

    key: str
    label: str
    group: str
    kind: str = "bool"
    variable: str = ""
    fallback: object = False
    on: str = ""
    off: str = ""
    flag: str = ""
    choices: Tuple[Tuple[str, str], ...] = ()
    flags: Tuple[Tuple[str, str], ...] = ()
    minimum: int = 0
    maximum: int = 99
    suffix: str = ""
    auto: Optional[int] = None
    needs: str = ""
    help: str = ""


OPTIONS: Tuple[Option, ...] = (
    # -- Steps ------------------------------------------------------------
    Option("seg", "Segmentation", "steps", variable="estimate_seg",
           fallback=True, off="--no-seg",
           help="Skull-stripping, tissue segmentation, spatial registration "
                "and ROI volumes. Switch it off to add surfaces to an "
                "earlier run that kept its hemisphere partitions."),
    Option("surf", "Surfaces and thickness", "steps", variable="estimate_surf",
           fallback=True, off="--no-surf",
           help="Cortical surface reconstruction and thickness estimation."),

    # -- Volume outputs ---------------------------------------------------
    Option("mwp", "Modulated, warped (mwp)", "volume", variable="save_mwp",
           fallback=True, off="--no-mwp", needs="seg",
           help="Grey and white matter in template space, modulated by the "
                "volume change: the maps voxel-based morphometry uses."),
    Option("wp", "Warped (wp)", "volume", variable="save_wp", on="--wp",
           needs="seg",
           help="Grey and white matter in template space, not modulated."),
    Option("rp", "Affine registered (rp)", "volume", variable="save_rp",
           on="--rp", needs="seg",
           help="Grey and white matter after the affine registration only."),
    Option("p", "Native space (p)", "volume", variable="save_p", on="--p",
           needs="seg", help="Grey and white matter in the space of the scan."),
    Option("csf", "Include CSF maps", "volume", variable="save_csf",
           on="--csf", needs="seg",
           help="Save CSF next to grey and white matter. Applies to the "
                "native-space and the affine-registered maps."),
    Option("lesions", "WMH lesion maps (p4)", "volume", variable="save_lesions",
           on="--lesions", needs="seg",
           help="Probability of a white matter hyperintensity inside the "
                "detected lesions. Applies to the native-space and the "
                "affine-registered maps."),
    Option("atlas", "Atlases", "volume", kind="atlas", variable="atlas_vol",
           fallback="'neuromorphometrics', 'cobra'", flag="--atlas",
           off="--no-atlas", needs="seg",
           help="Atlases whose regional volumes are written to the label "
                "folder."),

    # -- Surface outputs --------------------------------------------------
    Option("pial_white", "Pial and white surfaces", "surface",
           variable="save_pial_white", fallback=True, on="--pial-white",
           off="--no-pial-white", needs="surf",
           help="Estimate the pial and the white matter surface in addition "
                "to the central surface."),
    Option("sphere_reg", "Spherical registration", "surface",
           variable="estimate_spherereg", fallback=True, off="--no-sphere-reg",
           needs="surf",
           help="Register the surface to the template sphere. Surface "
                "atlases and resampling to the template mesh depend on it."),
    Option("atlas_surf", "Atlases", "surface", kind="atlas",
           variable="atlas_surf",
           fallback="'aparc_DK40.freesurfer','aparc_a2009s.freesurfer'",
           flag="--atlas-surf", needs="surf",
           help="Parcellations mapped onto the individual surface."),

    # -- Files ------------------------------------------------------------
    Option("bids", "BIDS names", "files", variable="use_bids_naming",
           on="--bids",
           help="Name files and folders by the BIDS derivatives standard "
                "instead of CAT12's. Volumes are then compressed."),
    Option("gz", "Compress volumes (.nii.gz)", "files", variable="nii_ext",
           on="--gz", help="Write volumes as .nii.gz instead of .nii."),
    Option("fmriprep", "fMRIPrep-compatible results", "files",
           variable="fmriprep", on="--fmriprep",
           help="Write what fMRIPrep expects from its anatomical workflow: "
                "BIDS names, native-space maps with CSF, ITK transforms, "
                "fsLR spheres and sidecars. No modulated maps and no ROI "
                "volumes are saved; not meant for VBM."),
    Option("save_h5", "Deformations as ANTs/ITK .h5", "files",
           variable="save_h5", on="--save-h5", needs="seg",
           help="Save the deformations to and from template space as "
                "composite HDF5 transforms as well."),

    # -- Processing -------------------------------------------------------
    Option("multi", "Parallel jobs", "processing", kind="int", variable="multi",
           fallback=-1, flag="--multi", minimum=0, maximum=256, auto=-1,
           help="Images processed at the same time. Auto uses every core "
                "for the surfaces and as many segmentations as the memory "
                "per job allows."),
    Option("min_memory", "Memory per job", "processing", kind="int",
           variable="min_memory", fallback=10, flag="--min-memory",
           minimum=1, maximum=1024, suffix=" GB",
           help="Memory reserved for each segmentation. Raise it if the "
                "computer runs short of memory during a batch."),
    # No flag: the front end leaves finished images out of the run itself,
    # which is exact per subject where --no-overwrite matches one pattern
    Option("skip_existing", "Skip finished images", "processing",
           help="Leave out the images whose results are complete already."),
    Option("retry", "Retry a failed step once", "processing",
           variable="retry", on="--retry", off="--no-retry",
           help="Give the segmentation or the surfaces of an image a second "
                "attempt before reporting the failure."),

    # -- Expert -----------------------------------------------------------
    Option("skullstrip", "Skull-stripping", "expert", kind="choice",
           fallback="run",
           choices=(("run", "Run"),
                    ("skip", "Input is skull-stripped"),
                    ("only", "Skull-strip only")),
           flags=(("skip", "--no-skullstrip"), ("only", "--skullstrip-only")),
           needs="seg",
           help="Skip skull-stripping for an image that has no skull left, "
                "or stop after it and save the brain and its mask."),
    Option("amap", "AMAP segmentation", "expert", variable="use_amap",
           on="--amap", needs="seg",
           help="Use the network's segmentation as the starting estimate of "
                "CAT12's AMAP segmentation instead of as the result."),
    Option("vessel", "Vessel removal", "expert", variable="vessel",
           fallback=True, off="--no-vessel",
           help="Remove blood vessels from the tissue maps and the surface "
                "input."),
    Option("nn_backend", "Network backend", "expert", kind="choice",
           variable="nn_backend", fallback="torch", flag="--nn-backend",
           needs="seg",
           help="Framework that runs the networks. tinygrad is experimental "
                "and needs pip install 'T1Prep[tinygrad]'."),
    Option("thickness_method", "Thickness method", "expert", kind="choice",
           variable="thickness_method", fallback="3", flag="--thickness-method",
           choices=(("3", "PBT"),
                    ("1", "Tfs distance, PBT-based"),
                    ("2", "Tfs distance, pial to white")),
           needs="surf",
           help="Measure thickness by projection (PBT) or as FreeSurfer's "
                "Tfs distance."),
    Option("correct_folding", "Correct thickness for folding", "expert",
           variable="correct_folding", fallback=True,
           off="--no-correct-folding", needs="surf",
           help="Correct the thickness for the effect of cortical folding."),
    Option("median_filter", "Median filter passes", "expert", kind="int",
           variable="median_filter", fallback=1, flag="--median-filter",
           minimum=0, maximum=10, needs="surf",
           help="Median filter passes that reduce topology artefacts before "
                "the surface is extracted."),
    Option("pre_fwhm", "Pre-smoothing FWHM", "expert", kind="int",
           variable="pre_fwhm", fallback=2, flag="--pre-fwhm",
           minimum=0, maximum=10, suffix=" mm", needs="surf",
           help="Smoothing applied before Marching Cubes."),
    Option("hemi", "Keep hemisphere partitions", "expert",
           variable="save_hemi", on="--hemi",
           help="Keep the left and right partitions of the segmentation, "
                "which the surfaces are built from. Needed to run the "
                "surfaces later without the segmentation."),
    Option("debug", "Keep temporary files (debug)", "expert",
           variable="debug", on="--debug",
           help="Keep temporary files and save additional maps for "
                "inspection."),
    Option("track", "Send anonymous usage statistics", "expert",
           variable="track", fallback=True, off="--no-track",
           help="Report the number of processed and failed images with the "
                "T1Prep version and the operating system, and the type and "
                "code location of a crash. Never file names, paths or "
                "images."),
)

_BY_KEY: Dict[str, Option] = {option.key: option for option in OPTIONS}


def option(key: str) -> Option:
    """The :class:`Option` named *key*."""
    return _BY_KEY[key]


def choices(item: Option) -> Tuple[Tuple[str, str], ...]:
    """The ``(value, label)`` pairs of a choice, the backends looked up live."""
    if item.key == "nn_backend":
        return tuple((name, name) for name in backend_names())
    return item.choices


def _flag_is_on(raw: Optional[str], fallback: bool) -> bool:
    """Read a 0/1 variable of the defaults file."""
    if raw is None:
        return bool(fallback)
    try:
        return int(str(raw).strip()) != 0
    except ValueError:
        return bool(fallback)


def _number(raw: Optional[str], fallback: int) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(fallback)


def default_values(
    defaults: Optional[Mapping[str, str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    atlas_names: Optional[Sequence[str]] = None,
    surface_atlas_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """What every option starts with: what T1Prep itself would do.

    Args:
        defaults: The parsed defaults file (:func:`load_defaults`); a variable
            it leaves out takes the value T1Prep ships with.
        environ: The environment; ``T1PREP_NN_BACKEND`` in it overrides the
            defaults file, as in ``scripts/T1Prep``.
        atlas_names, surface_atlas_names: The atlases to choose from; the
            shipped ones when not given.

    Returns:
        A value per option key: bool, int, str or a tuple of atlas names.
    """
    defaults = defaults or {}
    environ = os.environ if environ is None else environ
    available = {
        "atlas": list_atlas_names() if atlas_names is None else list(atlas_names),
        "atlas_surf": (list_surface_atlas_names() if surface_atlas_names is None
                       else list(surface_atlas_names)),
    }
    values: Dict[str, object] = {}
    for item in OPTIONS:
        raw = defaults.get(item.variable) if item.variable else None
        if item.key == "gz":
            values["gz"] = str(raw or "nii").lower() == "nii.gz"
        elif item.key == "skullstrip":
            if _flag_is_on(defaults.get("skullstrip_only"), False):
                values["skullstrip"] = "only"
            elif _flag_is_on(defaults.get("skip_skullstrip"), False):
                values["skullstrip"] = "skip"
            else:
                values["skullstrip"] = "run"
        elif item.key == "nn_backend":
            names = backend_names()
            name = (environ.get(BACKEND_ENV, "").strip().lower()
                    or str(raw or "").strip().lower() or names[0])
            values["nn_backend"] = name if name in names else names[0]
        elif item.kind == "bool":
            values[item.key] = _flag_is_on(raw, bool(item.fallback))
        elif item.kind == "int":
            number = _number(raw, int(item.fallback))
            if item.auto is not None:
                # 0 (one image after the other in T1Prep's own loop) has no
                # place in the form: one job at a time is the nearest thing
                number = item.minimum if number == item.auto else max(1, number)
            values[item.key] = min(max(number, item.minimum), item.maximum)
        elif item.kind == "choice":
            known = [value for value, _ in choices(item)]
            text = str(raw).strip() if raw is not None else str(item.fallback)
            values[item.key] = text if text in known else str(item.fallback)
        elif item.kind == "atlas":
            names = available[item.key]
            wanted = selected_atlases(
                names, str(item.fallback) if raw is None else str(raw))
            values[item.key] = tuple(name for name in names if name in wanted)
    return values


def implied(values: Mapping[str, object]) -> Dict[str, object]:
    """The values other choices force, as ``scripts/T1Prep`` sets them.

    ``--fmriprep`` and ``--bids`` switch further options, and skull-stripping
    alone leaves nothing for the surfaces.  A form shows these as they will
    be and keeps the user from changing them.
    """
    forced: Dict[str, object] = {}
    if values.get("seg", True) and values.get("skullstrip") == "only":
        forced["surf"] = False
    if values.get("fmriprep"):
        forced.update(bids=True, csf=True, p=True, mwp=False, hemi=False,
                      sphere_reg=True, atlas=())
    if values.get("bids") or forced.get("bids"):
        forced["gz"] = True
    return forced


def effective(values: Mapping[str, object]) -> Dict[str, object]:
    """*values* with everything :func:`implied` forces filled in."""
    result = dict(values)
    result.update(implied(values))
    return result


def irrelevant(values: Mapping[str, object]) -> Set[str]:
    """Keys of the options that have no effect with these *values*.

    An option that belongs to a step (:attr:`Option.needs`) does nothing
    while that step is switched off.
    """
    current = effective(values)
    return {item.key for item in OPTIONS
            if item.needs and not current.get(item.needs, True)}


def can_change(item: Option, default: object) -> bool:
    """Whether T1Prep has a flag that moves *item* away from *default*.

    Most switches work one way only (there is ``--wp`` but no ``--no-wp``),
    which is no restriction with the shipped defaults.  A defaults file that
    turns such an option on leaves the form nothing to turn it off with.
    """
    if item.kind != "bool" or not (item.on or item.off):
        return True
    return bool(item.off if default else item.on)


def problems(values: Mapping[str, object]) -> List[str]:
    """Why T1Prep would refuse these *values*; empty when it would not."""
    current = effective(values)
    found = []
    if not current.get("seg", True) and not current.get("surf", True):
        found.append("Segmentation and surfaces are both switched off.")
    return found


def build_arguments(
    values: Mapping[str, object],
    defaults: Optional[Mapping[str, str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """The T1Prep options that turn its defaults into *values*.

    Only what differs from the defaults file is spelled out, so the command
    line stays as short as one typed by hand; the atlases are always named,
    because an empty selection has to say "none" rather than "the default".

    Args:
        values: A value per option key, as :func:`default_values` returns
            them and a form changes them.  Missing keys keep their default.
        defaults: The parsed defaults file of the T1Prep that will run.
        environ: The environment the job will run in.

    Returns:
        The arguments, without the command, the output folder and the files.
    """
    start = default_values(defaults, environ)
    current = dict(start)
    current.update(values)
    forced = implied(current)
    current.update(forced)
    # What another flag sets anyway is left to that flag (--fmriprep implies
    # --bids, which implies --gz), and so is what cannot matter
    skip = irrelevant(current) | set(forced)

    args: List[str] = []
    for item in OPTIONS:
        if item.key in skip:
            continue
        value, default = current[item.key], start[item.key]
        if item.kind == "bool":
            if bool(value) != bool(default):
                flag = item.on if value else item.off
                if flag:
                    args.append(flag)
        elif item.kind == "int":
            if int(value) != int(default):
                number = int(value)
                if item.auto is not None and number == item.minimum:
                    number = item.auto
                args.extend([item.flag, str(number)])
        elif item.kind == "choice":
            if str(value) == str(default):
                continue
            spelled = dict(item.flags).get(str(value))
            if spelled:
                args.append(spelled)
            elif item.flag:
                args.extend([item.flag, str(value)])
        elif item.kind == "atlas":
            names = build_atlas_arg(list(value or ()))
            if names:
                args.extend([item.flag, names])
            elif item.off:
                args.append(item.off)
            else:
                # T1Prep has no flag for "no surface atlas", and an empty
                # value is refused.  A lone comma is a list without a name:
                # surface_estimation drops the empty items and maps nothing.
                args.extend([item.flag, ","])
    return args
