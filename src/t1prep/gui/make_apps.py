"""macOS application bundles for the T1Prep viewers.

The bundles are thin: each one launches the installed ``CAT_SurfView`` /
``CAT_VolView`` console script of the environment it was built from, so they
follow every update of that installation instead of freezing a copy of VTK and
Qt.  They exist so the viewers can be started from the Dock, Spotlight or
Finder, and so Finder can route ``.nii``/``.gii`` documents to them.

Used in two ways:

* ``t1prep-make-apps`` — build (or rebuild) them explicitly.
* :func:`ensure_apps_exist` — called when a viewer starts, so the apps appear
  once after ``pip install`` without a separate step.  Set ``T1PREP_NO_APPS=1``
  to keep that from happening.

There is deliberately no hook in the installation itself: wheels have no
post-install step, and writing outside the environment during ``pip install``
would leave files that ``pip uninstall`` could not remove.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: Bundle identifier prefix; also used for the types the apps declare
BUNDLE_PREFIX = "de.uni-jena.t1prep"

#: Set to skip the automatic creation on first start
DISABLE_ENV = "T1PREP_NO_APPS"

_LSREGISTER = ("/System/Library/Frameworks/CoreServices.framework/Frameworks"
               "/LaunchServices.framework/Support/lsregister")


def _document_type(name: str, rank: str, content_types: Sequence[str],
                   extensions: Sequence[str]) -> dict:
    return {
        "CFBundleTypeName": name,
        "CFBundleTypeRole": "Viewer",
        "LSHandlerRank": rank,
        "LSItemContentTypes": list(content_types),
        "CFBundleTypeExtensions": list(extensions),
    }


def _type_declaration(identifier: str, description: str, extension: str,
                      conforms_to: str) -> dict:
    return {
        "UTTypeIdentifier": identifier,
        "UTTypeDescription": description,
        "UTTypeConformsTo": [conforms_to],
        "UTTypeTagSpecification": {"public.filename-extension": [extension]},
    }


# Finder routes documents by Uniform Type Identifier, not by extension.
# ".nii" already has one on macOS (gov.nih.nifti-1), which the volume viewer
# imports; ".gii" and ".annot" have none, so the surface viewer exports its
# own.  ".nii.gz" counts as plain gzip — only the last extension is looked at —
# so the volume viewer takes that type as an *alternate* handler: it shows up
# under "Open With" without claiming every .gz file on the system.
VIEWERS: Dict[str, dict] = {
    "CAT_SurfView": {
        "description": "T1Prep surface viewer",
        "document_types": [
            _document_type("GIFTI surface", "Owner",
                           [f"{BUNDLE_PREFIX}.gifti"], ["gii"]),
            _document_type("FreeSurfer annotation", "Owner",
                           [f"{BUNDLE_PREFIX}.annot"], ["annot"]),
            _document_type("Surface overlay", "Alternate",
                           ["public.plain-text"], ["txt"]),
        ],
        "exported_types": [
            _type_declaration(f"{BUNDLE_PREFIX}.gifti", "GIFTI surface",
                              "gii", "public.xml"),
            _type_declaration(f"{BUNDLE_PREFIX}.annot",
                              "FreeSurfer annotation", "annot", "public.data"),
        ],
        "imported_types": [],
        "default_types": [f"{BUNDLE_PREFIX}.gifti", "gii",
                          f"{BUNDLE_PREFIX}.annot"],
    },
    "CAT_VolView": {
        "description": "T1Prep volume viewer",
        "document_types": [
            _document_type("NIfTI volume", "Default",
                           ["gov.nih.nifti-1"], ["nii"]),
            _document_type("Compressed volume", "Alternate",
                           ["org.gnu.gnu-zip-archive"], ["gz"]),
            _document_type("Volume", "Alternate",
                           ["public.data"], ["mnc", "nrrd", "mha", "mhd"]),
        ],
        "exported_types": [],
        "imported_types": [
            _type_declaration("gov.nih.nifti-1", "NIfTI volume",
                              "nii", "public.data"),
        ],
        "default_types": ["gov.nih.nifti-1", "nii"],
    },
}


def default_output_dir() -> Path:
    """Where the bundles go: /Applications when writable, else ~/Applications."""
    system = Path("/Applications")
    if os.access(system, os.W_OK):
        return system
    return Path.home() / "Applications"


def find_bin_dir() -> Optional[Path]:
    """Directory holding the viewer console scripts.

    The interpreter running this code sits next to them in the same
    environment, which is also the installation the bundles should launch.
    """
    candidates = [Path(sys.executable).parent]
    for name in VIEWERS:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found).parent)
    for candidate in candidates:
        if all((candidate / name).exists() for name in VIEWERS):
            return candidate
    return None


def _package_version() -> str:
    try:
        from t1prep import __version__
        return str(__version__)
    except Exception:
        return "0.0"


def _logo() -> Optional[Path]:
    """The T1Prep logo, from the package or from a source checkout."""
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent / "data" / "T1Prep_logo.svg",
                      here.parents[3] / "T1Prep_logo.svg"):
        if candidate.exists():
            return candidate
    return None


def _make_icon(work_dir: Path) -> Optional[Path]:
    """Render the logo to an .icns, or None when macOS cannot do it here."""
    logo = _logo()
    if logo is None or not shutil.which("iconutil") or not shutil.which("sips"):
        return None
    png = work_dir / "logo.png"
    try:
        if shutil.which("qlmanage"):
            subprocess.run(["qlmanage", "-t", "-s", "1024", "-o", str(work_dir),
                            str(logo)], capture_output=True, check=False)
            rendered = next(iter(work_dir.glob("*.png")), None)
            if rendered is None:
                return None
            png = rendered
        else:
            return None
        iconset = work_dir / "T1Prep.iconset"
        iconset.mkdir(exist_ok=True)
        for size in (16, 32, 64, 128, 256, 512):
            for scale, suffix in ((1, ""), (2, "@2x")):
                subprocess.run(
                    ["sips", "-z", str(size * scale), str(size * scale), str(png),
                     "--out", str(iconset / f"icon_{size}x{size}{suffix}.png")],
                    capture_output=True, check=False)
        icns = work_dir / "T1Prep.icns"
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
                       capture_output=True, check=False)
        return icns if icns.exists() else None
    except Exception:
        return None


def build_app(name: str, bin_dir: Path, out_dir: Path,
              icon: Optional[Path] = None) -> Path:
    """Write one application bundle and return its path."""
    spec = VIEWERS[name]
    app = out_dir / f"{name}.app"
    contents = app / "Contents"
    if app.exists():
        shutil.rmtree(app)
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir(parents=True)

    # Finder discards stdout/stderr, so a crash would be silent; keep the last
    # run in a log the user can be pointed at.
    launcher = contents / "MacOS" / name
    launcher.write_text(
        "#!/bin/bash\n"
        f"# Launcher for {name}; regenerate with t1prep-make-apps\n"
        "export T1PREP_APP=1\n"
        'log_dir="$HOME/Library/Logs/T1Prep"\n'
        'mkdir -p "$log_dir"\n'
        f'exec "{bin_dir / name}" "$@" > "$log_dir/{name}.log" 2>&1\n'
    )
    launcher.chmod(0o755)

    info = {
        "CFBundleName": name,
        "CFBundleDisplayName": name,
        "CFBundleGetInfoString": spec["description"],
        "CFBundleExecutable": name,
        "CFBundleIdentifier": f"{BUNDLE_PREFIX}.{name.lower()}",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": _package_version(),
        "CFBundleVersion": _package_version(),
        "LSApplicationCategoryType": "public.app-category.medical",
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
        "CFBundleDocumentTypes": spec["document_types"],
    }
    if spec["exported_types"]:
        info["UTExportedTypeDeclarations"] = spec["exported_types"]
    if spec["imported_types"]:
        info["UTImportedTypeDeclarations"] = spec["imported_types"]
    if icon is not None and icon.exists():
        shutil.copy(icon, contents / "Resources" / f"{name}.icns")
        info["CFBundleIconFile"] = name

    with open(contents / "Info.plist", "wb") as handle:
        plistlib.dump(info, handle)
    return app


def register(apps: Sequence[Path]) -> None:
    """Tell Launch Services about the bundles, so Finder picks them up."""
    if not Path(_LSREGISTER).exists():
        return
    subprocess.run([_LSREGISTER, "-f", *[str(a) for a in apps]],
                   capture_output=True, check=False)


def set_as_default(quiet: bool = False) -> bool:
    """Make the apps the default for the types they declare (needs duti)."""
    if not shutil.which("duti"):
        if not quiet:
            print("duti is not installed, so the defaults were not changed.\n"
                  "  Install it with 'brew install duti' and try again, or set it\n"
                  "  once in Finder: select a file, press ⌘I, choose the app under\n"
                  "  \"Open with\" and click \"Change All…\".", file=sys.stderr)
        return False
    for name, spec in VIEWERS.items():
        identifier = f"{BUNDLE_PREFIX}.{name.lower()}"
        for uti in spec["default_types"]:
            subprocess.run(["duti", "-s", identifier, uti, "all"],
                           capture_output=True, check=False)
    return True


def build_apps(out_dir: Optional[Path] = None, bin_dir: Optional[Path] = None,
               quiet: bool = False) -> List[Path]:
    """Build both bundles.

    Raises:
        FileNotFoundError: when the viewer console scripts cannot be found.
    """
    bin_dir = Path(bin_dir) if bin_dir else find_bin_dir()
    if bin_dir is None or not all((bin_dir / n).exists() for n in VIEWERS):
        raise FileNotFoundError(
            "Could not find CAT_SurfView and CAT_VolView; install T1Prep and "
            "use the interpreter of that environment, or pass --bin-dir.")
    out_dir = Path(out_dir) if out_dir else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    work = out_dir / ".t1prep-icon"
    work.mkdir(exist_ok=True)
    try:
        icon = _make_icon(work)
        apps = [build_app(name, bin_dir, out_dir, icon) for name in VIEWERS]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    register(apps)
    if not quiet:
        for app in apps:
            print(f"✅ {app}")
    return apps


def ensure_apps_exist(quiet: bool = False) -> List[Path]:
    """Create the bundles once, unless they are there or unwanted.

    Called when a viewer starts, so ``pip install`` followed by one run leaves
    working apps behind.  Never raises and never gets in the way: anything
    unexpected simply means no bundles.
    """
    if sys.platform != "darwin" or os.environ.get(DISABLE_ENV):
        return []
    if os.environ.get("T1PREP_APP"):        # started from a bundle already
        return []
    try:
        out_dir = default_output_dir()
        if any((out_dir / f"{name}.app").exists() for name in VIEWERS):
            return []
        apps = build_apps(out_dir=out_dir, quiet=True)
        if not quiet and apps:
            print(f"Created {out_dir}/CAT_SurfView.app and CAT_VolView.app "
                  f"({DISABLE_ENV}=1 disables this).", file=sys.stderr)
        return apps
    except Exception:
        return []


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry-point (``t1prep-make-apps``)."""
    parser = argparse.ArgumentParser(
        prog="t1prep-make-apps",
        description="Create macOS application bundles for the T1Prep viewers.")
    parser.add_argument("-o", "--out-dir", type=Path, default=None,
                        help="where to write them (default: /Applications, "
                             "or ~/Applications when that is not writable)")
    parser.add_argument("-b", "-p", "--bin-dir", type=Path, default=None,
                        help="directory holding CAT_SurfView and CAT_VolView "
                             "(default: the environment this runs in)")
    parser.add_argument("-d", "--set-default", action="store_true",
                        help="also make them the default for the file types "
                             "they declare (needs duti)")
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        print("Application bundles are a macOS feature; nothing to do here.",
              file=sys.stderr)
        return 1
    try:
        apps = build_apps(out_dir=args.out_dir, bin_dir=args.bin_dir)
    except FileNotFoundError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    if args.set_default:
        set_as_default()
    print("\nDouble-click one to pick a file, or drop files onto its icon.")
    print("To always open a file type with them, select a file in Finder, press")
    print("⌘I, choose the app under \"Open with\" and click \"Change All…\".")
    print("Do this once for .nii and once for .nii.gz: macOS only looks at the")
    print("last extension, so .nii.gz counts as a gzip archive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
