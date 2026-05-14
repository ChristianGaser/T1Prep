"""Model file management: location constants, presence checks, and downloading.

Model weights are not bundled in the PyPI wheel (they exceed the 100 MB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Directory where model weight files are stored at runtime.
MODEL_DIR: Path = Path(DATA_PATH) / "models"

#: All model filenames that must be present for the pipeline to run.
MODEL_FILES: list[str] = (
    [
        "brain_extraction_bbox_model.pt",
        "brain_extraction_model.pt",
        "segmentation_nogm_model.pt",
    ]
    + [f"segmentation_patch_{i}_model.pt" for i in range(18)]
    + ["segmentation_model.pt", "warp_model.pt"]
)

#: URL of the GitHub release archive that contains the model weights.
#: Override with the ``T1PREP_MODEL_ZIP_URL`` environment variable if needed.
import os as _os
_DEFAULT_MODEL_ZIP_URL = (
    "https://github.com/ChristianGaser/T1Prep/releases/download/"
    "v0.2.0-beta/T1Prep_Models.zip"
)
MODEL_ZIP_URL: str = _os.environ.get("T1PREP_MODEL_ZIP_URL", _DEFAULT_MODEL_ZIP_URL)

#: Dev-mode fallback: models bundled inside the source tree (not in the wheel).
_DEV_MODEL_DIR: Path = Path(__file__).resolve().parent / "data" / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def all_models_present() -> bool:
    """Return ``True`` if every required model file exists in :data:`MODEL_DIR`."""
    return all((MODEL_DIR / f).exists() for f in MODEL_FILES)


def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest*, printing a simple progress indicator."""

    def _reporthook(count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            print(f"\r  Downloaded {count * block_size // 1_048_576} MB ...", end="", flush=True)
        else:
            downloaded = count * block_size
            pct = min(100, int(100 * downloaded / total_size))
            bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
            mb_done = downloaded // 1_048_576
            mb_total = total_size // 1_048_576
            print(f"\r  [{bar}] {pct:3d}% ({mb_done}/{mb_total} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
    print()  # newline after progress bar


def download_models(verbose: bool = True) -> None:
    """Download model weights from GitHub and extract them to :data:`MODEL_DIR`.

    Parameters
    ----------
    verbose:
        When ``True`` (default) print progress to stdout.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
                if not filename or filename not in MODEL_FILES:
                    continue
                dest = MODEL_DIR / filename
                with zf.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)

    if verbose:
        print("Model weights downloaded successfully.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_model_files(verbose: bool = True) -> None:
    """Ensure all model weight files are present, downloading if necessary.

    Strategy (in order):
    1. If all files already exist in :data:`MODEL_DIR` → nothing to do.
    2. Dev-mode: if the source-tree ``data/models/`` directory is populated
       (e.g. a git clone with model files present), copy missing files from
       there — avoids a network round-trip during development.
    3. Download the model archive from GitHub and extract to :data:`MODEL_DIR`.

    Parameters
    ----------
    verbose:
        When ``True`` (default) print progress to stdout.
    """
    if all_models_present():
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --- dev-mode: copy from source tree if available ---
    if _DEV_MODEL_DIR.is_dir():
        missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).exists()]
        copied = []
        for filename in missing:
            src = _DEV_MODEL_DIR / filename
            if src.exists():
                shutil.copy2(str(src), str(MODEL_DIR / filename))
                copied.append(filename)
        if copied and verbose:
            print(f"Copied {len(copied)} model file(s) from local data directory.")
        if all_models_present():
            return

    # --- download from GitHub ---
    if verbose:
        missing_count = sum(1 for f in MODEL_FILES if not (MODEL_DIR / f).exists())
        print(f"{missing_count} model file(s) missing — downloading from GitHub ...")
    download_models(verbose=verbose)

    if not all_models_present():
        still_missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).exists()]
        raise RuntimeError(
            "Model download completed but the following files are still missing:\n"
            + "\n".join(f"  {MODEL_DIR / f}" for f in still_missing)
        )


def main() -> None:
    """Console script entry point for ``t1prep-download-models``.

    Downloads model weights to :data:`MODEL_DIR` if not already present.
    Pass ``--force`` to re-download even if models are present.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Download T1Prep model weights from GitHub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    if args.force and MODEL_DIR.exists():
        print(f"--force: removing existing models in {MODEL_DIR}")
        for f in MODEL_FILES:
            p = MODEL_DIR / f
            if p.exists():
                p.unlink()

    if all_models_present() and not args.force:
        print(f"All model weights already present in {MODEL_DIR}")
        sys.exit(0)

    try:
        prepare_model_files(verbose=True)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
