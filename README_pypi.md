[![Python 3.9 | 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?logo=apache&logoColor=white)](https://github.com/ChristianGaser/T1Prep/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/ChristianGaser/T1Prep?display_name=tag&include_prereleases)](https://github.com/ChristianGaser/T1Prep/releases)

> **Note:** This project is still in development and may contain bugs.
> Please [report issues](https://github.com/ChristianGaser/T1Prep/issues) if you encounter problems.

# T1Prep: T1 PREProcessing Pipeline (aka PyCAT)

T1Prep is a Python pipeline for preprocessing and segmenting T1-weighted MRI data. It supports:

- Bias-field correction and denoising
- Brain extraction (skull stripping)
- Tissue segmentation (GM, WM, CSF)
- Cortical surface reconstruction and thickness estimation
- Non-linear spatial registration to MNI152 space
- Atlas-based ROI extraction
- White matter hyperintensity (WMH/lesion) detection
- BIDS derivatives output naming

Cortical surface reconstruction uses the [`cat-surf`](https://pypi.org/project/cat-surf/) Python package
(pure Python bindings to the CAT-Surface C library — no compiled binaries required).

For full documentation, CLI usage, Docker instructions, and helper scripts see the
[GitHub repository](https://github.com/ChristianGaser/T1Prep).

---

## Installation

```bash
pip install T1Prep
```

`pip install` places every entry point into the active environment's `bin/`
directory. With that directory on your `PATH` (e.g. an activated venv) the
following commands are available:

| Command | Role |
| --- | --- |
| `T1Prep` | main CLI — batch + parallel processing (`--multi`) |
| `t1prep-ui` | browser-based web UI |
| `t1prep-run` | single-subject Python entry |
| `cat-viewsurf` | surface viewer |
| `t1prep-download-models` | fetch model weights |

### Download model weights

Model weights are not bundled in the wheel (they are ~500 MB). Download them after installation:

```bash
t1prep-download-models
```

Models are stored alongside the `deepmriprep` package data and are downloaded automatically
on first pipeline use if this step is skipped.

---

## Requirements

- Python 3.9–3.12
- ~2 GB disk space for model weights (downloaded separately, see above)
- For GPU acceleration: CUDA-capable GPU or Apple Silicon (MPS)

---

## Python API

```python
from t1prep import run_t1prep

# Single file — results saved next to input
run_t1prep("/data/sub-01_T1w.nii.gz")

# Single file, BIDS-compatible output
run_t1prep("/data/sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz", bids=True)

# Batch processing with options
run_t1prep(
    ["/data/T1/sub-01.nii.gz", "/data/T1/sub-02.nii.gz"],
    out_dir="/results",
    atlas=["neuromorphometrics", "suit"],
    multi=-1,          # auto-detect parallelism
    wp=True,           # save warped segmentations
    p=True,            # save native segmentations
    csf=True,          # save CSF segmentation
    lesions=True,      # save WMH lesion map
    gz=True,           # compress outputs (.nii.gz)
    stream_output=True,
    log_file="/results/T1Prep_run.log",
)
```

### Key parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `files` | `str` or `list[str]` | Input NIfTI file(s) |
| `out_dir` | `str` | Output directory (default: same as input) |
| `atlas` | `list[str]` | Atlas names for ROI extraction |
| `surf` | `bool` | Run cortical surface estimation (default: `True`) |
| `multi` | `int` | Parallel workers; `-1` = auto (default: `1`) |
| `bids` | `bool` | Use BIDS derivatives naming |
| `gz` | `bool` | Save compressed NIfTI (`.nii.gz`) |
| `wp` | `bool` | Save warped (MNI space) segmentations |
| `p` | `bool` | Save native space segmentations |
| `csf` | `bool` | Save CSF segmentation |
| `lesions` | `bool` | Save WMH/lesion map |
| `amap` | `bool` | Use AMAP segmentation (CAT12-style) |
| `skullstrip_only` | `bool` | Only run skull stripping then exit |
| `skip_skullstrip` | `bool` | Skip skull stripping (pre-stripped input) |

---

## Output structure

**Non-BIDS** (default): subfolders `mri/`, `surf/`, `report/`, `label/` in the output directory,
with CAT12-compatible filenames (e.g., `mwp1sub-01.nii`, `lh.thickness.sub-01`).

**BIDS** (with `bids=True`): BIDS derivatives layout
`<out_dir>/derivatives/T1Prep-v<version>/sub-XX/ses-YY/anat/`.

---

## License

Distributed under the [Apache License 2.0](https://github.com/ChristianGaser/T1Prep/blob/main/LICENSE).
