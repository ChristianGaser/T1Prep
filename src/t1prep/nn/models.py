"""The networks T1Prep ships: names, where their files live, how they are made.

Each network is one safetensors file, ``<name>.safetensors`` in
:data:`MODEL_DIR`.  The files are made once, from the TorchScript archives
deepmriprep ships: those already on disk (deepmriprep's own ``data/models``,
a source checkout's ``data/models``) or those in the GitHub release asset,
which is downloaded when there are none.  The weights are not bundled in the
PyPI wheel (they exceed the 100 MB limit).

The preparation can be triggered:
- automatically at first use (called by ``segment.py``)
- manually by the user via the ``t1prep-download-models`` console script

Console script entry point:  ``t1prep.nn.models:main``
"""

from __future__ import annotations

import os as _os
import shutil
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "MODEL_DIR",
    "MODEL_FILES",
    "NETWORKS",
    "all_models_present",
    "download_models",
    "load_network",
    "model_path",
    "prepare_model_files",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The networks T1Prep runs: deepbet's two skull-stripping models and
#: deepmriprep's coarse and 18 patch segmentation models.  deepmriprep's nogm
#: and warp networks are not among them: T1Prep removes non-cortical GM by
#: rule (``t1prep.nogm``) and registers by geodesic shooting
#: (``t1prep._shoot``).
NETWORKS: list = [
    "brain_extraction_bbox_model",
    "brain_extraction_model",
    "segmentation_model",
] + [f"segmentation_patch_{i}_model" for i in range(18)]

#: All model filenames that must be present for the pipeline to run.
MODEL_FILES: list = [f"{name}.safetensors" for name in NETWORKS]


def _deepmriprep_models() -> Path | None:
    """deepmriprep's model directory, where existing installs hold its archives.

    Found without importing deepmriprep, whose package import loads its whole
    pipeline (and deepbet).
    """
    import importlib.util

    spec = importlib.util.find_spec("deepmriprep")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))) / "data" / "models"


_DEEPMRIPREP_MODELS: Path | None = _deepmriprep_models()


def _user_cache_dir() -> Path:
    """Platform-appropriate user cache base for T1Prep."""
    if sys.platform == "win32":
        base = _os.environ.get("LOCALAPPDATA")
        return (
            (Path(base) if base else Path.home() / "AppData" / "Local")
            / "t1prep"
            / "Cache"
        )
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "t1prep"
    xdg = _os.environ.get("XDG_CACHE_HOME")
    return Path(xdg) / "t1prep" if xdg else Path.home() / ".cache" / "t1prep"


def _resolve_model_dir() -> Path:
    """Decide where T1Prep stores its model weights.

    Resolution order:

    1. ``T1PREP_MODEL_DIR`` environment variable (explicit override).
    2. ``deepmriprep``'s ``data/models``, when it already contains the full
       set of weights — existing installs keep working with no migration.
    3. ``deepmriprep``'s ``data/models``, when writable, so the network files
       sit next to the archives they are converted from.
    4. Platform user cache (``~/Library/Caches/t1prep/models`` on macOS,
       ``$XDG_CACHE_HOME/t1prep/models`` or ``~/.cache/t1prep/models`` on
       Linux, ``%LOCALAPPDATA%\\t1prep\\Cache\\models`` on Windows).
    """
    override = _os.environ.get("T1PREP_MODEL_DIR")
    if override:
        return Path(override)

    if _DEEPMRIPREP_MODELS is not None:
        if all((_DEEPMRIPREP_MODELS / f).is_file() for f in MODEL_FILES):
            return _DEEPMRIPREP_MODELS
        try:
            _DEEPMRIPREP_MODELS.mkdir(parents=True, exist_ok=True)
            probe = _DEEPMRIPREP_MODELS / ".t1prep_writable"
            probe.touch()
            probe.unlink()
            return _DEEPMRIPREP_MODELS
        except (OSError, PermissionError):
            pass

    return _user_cache_dir() / "models"


#: Directory where model weight files are stored at runtime.
MODEL_DIR: Path = _resolve_model_dir()

#: URL of the GitHub release archive that contains the model weights.
#: Override with the ``T1PREP_MODEL_ZIP_URL`` environment variable if needed.
_DEFAULT_MODEL_ZIP_URL = (
    "https://github.com/ChristianGaser/T1Prep/releases/download/"
    "v0.2.0-beta/T1Prep_Models.zip"
)
MODEL_ZIP_URL: str = _os.environ.get("T1PREP_MODEL_ZIP_URL", _DEFAULT_MODEL_ZIP_URL)

#: Dev-mode fallback: models bundled inside the source tree (not in the wheel).
_DEV_MODEL_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def model_path(name: str) -> Path:
    """Path of network *name*'s weights, e.g. ``model_path("segmentation_model")``."""
    return MODEL_DIR / f"{name}.safetensors"


def load_network(name: str, backend: str | None = None, device="cpu"):
    """Network *name* of :data:`NETWORKS`, loaded on *backend* and *device*."""
    from .network import Network

    return Network(model_path(name), backend, device)


def all_models_present() -> bool:
    """Return ``True`` if every required model file exists in :data:`MODEL_DIR`."""
    return all((MODEL_DIR / f).exists() for f in MODEL_FILES)


def _write_atomically(dest: Path, source) -> None:
    """Copy the file object *source* to *dest* through a temporary file beside it."""
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    try:
        with _os.fdopen(fd, "wb") as out:
            if hasattr(_os, "fchmod"):
                _os.fchmod(out.fileno(), 0o644)
            shutil.copyfileobj(source, out)
        _os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def _preparation_lock():
    """Hold :data:`MODEL_DIR`'s lock while the weights are made.

    ``T1Prep --multi`` starts several runs at once; on a fresh install each
    would otherwise download and convert the same files.  A no-op where
    ``fcntl`` is missing (Windows).
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(MODEL_DIR / ".t1prep_prepare.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _convert_missing(verbose: bool) -> None:
    """Make every missing network file from what is already on disk.

    A ready network file in the source tree is copied; otherwise the first
    TorchScript archive found -- in :data:`MODEL_DIR`, deepmriprep's
    ``data/models`` or the source tree -- is converted.
    """
    from .convert import convert_model

    folders = [
        d for d in (MODEL_DIR, _DEEPMRIPREP_MODELS, _DEV_MODEL_DIR) if d is not None
    ]
    made = 0
    for name in NETWORKS:
        dst = model_path(name)
        if dst.exists():
            continue
        ready = _DEV_MODEL_DIR / dst.name
        if ready.is_file():
            with open(ready, "rb") as source:
                _write_atomically(dst, source)
            made += 1
            continue
        archive = next(
            (d / f"{name}.pt" for d in folders if (d / f"{name}.pt").is_file()), None
        )
        if archive is not None:
            convert_model(archive, dst)
            made += 1
    if made and verbose:
        print(f"Prepared {made} network file(s) in {MODEL_DIR}.")


def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest*, printing a simple progress indicator.

    Uses requests, which verifies against certifi's CA bundle: python.org's
    macOS Python ships an OpenSSL without one until its "Install
    Certificates.command" has run, so urllib fails there with
    CERTIFICATE_VERIFY_FAILED.
    """
    import requests

    def _report(downloaded: int, total_size: int) -> None:
        if total_size <= 0:
            print(
                f"\r  Downloaded {downloaded // 1_048_576} MB ...",
                end="",
                flush=True,
            )
        else:
            pct = min(100, int(100 * downloaded / total_size))
            bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
            mb_done = downloaded // 1_048_576
            mb_total = total_size // 1_048_576
            print(
                f"\r  [{bar}] {pct:3d}% ({mb_done}/{mb_total} MB)", end="", flush=True
            )

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as out:
            for chunk in response.iter_content(chunk_size=1_048_576):
                out.write(chunk)
                downloaded += len(chunk)
                _report(downloaded, total_size)
    print()  # newline after progress bar


def download_models(verbose: bool = True) -> None:
    """Download the release archive and extract its weights to :data:`MODEL_DIR`.

    Both network files and the TorchScript archives they are converted from
    are extracted; :func:`prepare_model_files` does the conversion.

    Parameters
    ----------
    verbose:
        When ``True`` (default) print progress to stdout.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(MODEL_FILES) | {f"{name}.pt" for name in NETWORKS}

    if verbose:
        print(f"Downloading model weights from:\n  {MODEL_ZIP_URL}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "T1Prep_Models.zip"
        try:
            _download_with_progress(MODEL_ZIP_URL, zip_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download model weights from {MODEL_ZIP_URL}: {exc}\n"
                "Check your internet connection or set the T1PREP_MODEL_ZIP_URL "
                "environment variable to an alternative URL."
            ) from exc

        if verbose:
            print(f"Extracting to {MODEL_DIR} ...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                filename = Path(member).name
                if not filename or filename not in wanted:
                    continue
                with zf.open(member) as source:
                    _write_atomically(MODEL_DIR / filename, source)

    if verbose:
        print("Model weights downloaded successfully.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_model_files(verbose: bool = True) -> None:
    """Ensure all model weight files are present, converting or downloading if necessary.

    Strategy (in order):
    1. If all files already exist in :data:`MODEL_DIR` → nothing to do.
    2. Convert the TorchScript archives already on disk (deepmriprep's
       ``data/models``, a source checkout's ``data/models``) — no network
       round-trip.
    3. Download the model archive from GitHub, extract it to
       :data:`MODEL_DIR` and convert what it brought.

    Parameters
    ----------
    verbose:
        When ``True`` (default) print progress to stdout.
    """
    if all_models_present():
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with _preparation_lock():
        _convert_missing(verbose)  # a run that held the lock may have made them
        if all_models_present():
            return

        # --- download from GitHub ---
        if verbose:
            missing_count = sum(1 for f in MODEL_FILES if not (MODEL_DIR / f).exists())
            print(
                f"{missing_count} model file(s) missing — downloading from GitHub ..."
            )
        download_models(verbose=verbose)
        _convert_missing(verbose)

    if not all_models_present():
        still_missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).exists()]
        raise RuntimeError(
            "Model download completed but the following files are still missing:\n"
            + "\n".join(f"  {MODEL_DIR / f}" for f in still_missing)
        )


def main() -> None:
    """Console script entry point for ``t1prep-download-models``.

    Prepares the model weights in :data:`MODEL_DIR` if not already present.
    Pass ``--force`` to re-download them even if they are present.
    """
    from ..cli_help import ArgumentParser

    # Called without an argument this tool does its job — downloading the
    # weights — so it keeps that as the default action; the rest of the
    # unified behaviour (--help, "--" options) is the same as everywhere.
    parser = ArgumentParser(
        prog="t1prep-download-models",
        description="Download T1Prep model weights from GitHub.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download model weights even if they are already present.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Override the target directory for model weights.",
    )
    args = parser.parse_args()

    if args.model_dir is not None:
        global MODEL_DIR
        MODEL_DIR = Path(args.model_dir)

    if all_models_present() and not args.force:
        print(f"All model weights already present in {MODEL_DIR}")
        sys.exit(0)

    try:
        if args.force:
            print(f"--force: replacing the models in {MODEL_DIR}")
            for f in MODEL_FILES:
                (MODEL_DIR / f).unlink(missing_ok=True)
            download_models(verbose=True)
        prepare_model_files(verbose=True)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
