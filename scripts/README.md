# Scripts

This directory contains the shell scripts that make up the T1Prep command-line interface, environment helpers, processing utilities, and CAT12 surface/volume tools. All scripts automatically activate the project virtual environment (`env/`) when needed.

---

## Table of Contents

- [Main Pipeline](#main-pipeline)
- [Environment & Installation](#environment--installation)
- [Longitudinal Processing](#longitudinal-processing)
- [Evaluation](#evaluation)
- [CAT12 Surface & Volume Tools](#cat12-surface--volume-tools)
- [Internal Utilities](#internal-utilities)

---

## Main Pipeline

### `T1Prep`

The primary CLI entry point for the entire T1Prep pipeline. It performs preprocessing, segmentation, and cortical surface reconstruction on T1-weighted MRI images.

```bash
# Show all available options
./scripts/T1Prep --help

# Process a single file
./scripts/T1Prep --out-dir /tmp/out sub-01_T1w.nii.gz

# Batch process with automatic parallelization
./scripts/T1Prep --multi -1 --out-dir /tmp/out *.nii.gz

# Segmentation only (no surface estimation)
./scripts/T1Prep --no-surf sub-01_T1w.nii.gz

# Fast mode (skip spherical registration, atlas, pial/white surfaces)
./scripts/T1Prep --fast sub-01_T1w.nii.gz
```

**Key features:**
- Bias-field correction, skull-stripping, tissue segmentation, lesion detection
- Cortical surface reconstruction and thickness estimation
- Automatic parallelization across multiple files (`--multi`)
- Supports both CAT12-style and BIDS derivatives naming (`--bids`)
- Configurable via `T1Prep_defaults.txt`

### `T1Prep_ui`

Launches the Flask-based Web UI for T1Prep in a browser.

```bash
# Start on default port (5050)
./scripts/T1Prep_ui

# Start on a custom port
./scripts/T1Prep_ui 5500

# Start without opening a browser
./scripts/T1Prep_ui --no-browser
```

---

## Environment & Installation

### `install.sh`

Downloads and installs the latest T1Prep release from GitHub. Handles release selection, download, extraction, and initial setup.

```bash
# Interactive installation
bash scripts/install.sh

# Non-interactive (via environment variables)
T1PREP_VERSION=latest T1PREP_INSTALL_DIR=/opt/T1Prep bash scripts/install.sh
```

**Environment overrides:**
- `REPO_OWNER` — GitHub owner (default: `ChristianGaser`)
- `REPO_NAME` — Repository name (default: `T1Prep`)
- `T1PREP_INSTALL_DIR` — Skip directory prompt
- `T1PREP_VERSION` — Skip version prompt (e.g., `v1.0.0` or `latest`)

### `activate_env.sh`

Sources the T1Prep virtual environment. Intended to be **sourced** (not executed) in your current shell.

```bash
source scripts/activate_env.sh
```

### `run_with_env.sh`

Runs any Python script with the T1Prep virtual environment automatically activated. Useful for running individual modules without manually activating the environment.

```bash
# Run a specific module
./scripts/run_with_env.sh src/t1prep/segment.py --help

# Run the surface viewer
./scripts/run_with_env.sh src/t1prep/gui/cat_viewsurf.py --help
```

---

## Longitudinal Processing

### `process_longitudinal.sh`

Batch helper for longitudinal studies. Groups time-point scans by subject, runs inverse-consistent rigid realignment, then invokes T1Prep on each time point.

```bash
# Process time points for a single subject
./scripts/process_longitudinal.sh \
    --out-dir /path/to/output \
    /path/to/tp1.nii.gz /path/to/tp2.nii.gz

# Pass extra T1Prep options
./scripts/process_longitudinal.sh \
    --out-dir /path/to/output \
    --t1prep-arg "--no-surf" \
    /path/to/tp1.nii.gz /path/to/tp2.nii.gz

# Dry run (show what would be executed)
./scripts/process_longitudinal.sh --dry-run /path/to/tp1.nii.gz /path/to/tp2.nii.gz
```

**Input modes:**
- **NIfTI files:** Treated as time points for a single subject
- **Text files:** Each file is a time-point list; each line is a subject

### `realign_longitudinal.sh`

Wrapper around the Python module `t1prep.realign_longitudinal`. Performs inverse-consistent rigid realignment of longitudinal scans.

```bash
./scripts/realign_longitudinal.sh \
    --inputs scan1.nii.gz scan2.nii.gz \
    --out-dir /path/to/output

# With gradient-based sampling strategy
./scripts/realign_longitudinal.sh \
    --inputs scan1.nii.gz scan2.nii.gz \
    --out-dir /path/to/output \
    --sample-strategy gradient
```

---

## Evaluation

### `dice.sh`

Computes Dice-based similarity metrics between a ground truth and a predicted segmentation. Wraps the Python module `t1prep.dice`.

```bash
# Basic usage
./scripts/dice.sh --gt GT.nii.gz --pred PRED.nii.gz

# Soft Dice (for probability/partial-volume maps)
./scripts/dice.sh --gt GT.nii.gz --pred PRED.nii.gz --soft

# Verbose output (one line per label)
./scripts/dice.sh --gt GT.nii.gz --pred PRED.nii.gz --verbose

# Save confusion matrix
./scripts/dice.sh --gt GT.nii.gz --pred PRED.nii.gz --save-conf conf.csv
```

**Output (default):** `[dice_label_1, dice_label_2, ...] generalized_dice, dice_weighted`

---

## CAT12 Surface & Volume Tools

These scripts provide user-friendly wrappers around the compiled CAT-Surface binaries in `src/t1prep/bin/`. They support batch processing with built-in parallelization via `parallelize`.

### `cat_viewsurf.sh`

Interactive 3D surface viewer (PySide6/VTK). Displays cortical meshes and overlays.

```bash
# View a surface mesh
./scripts/cat_viewsurf.sh /path/to/lh.central.gii

# View a surface overlay (e.g., thickness)
./scripts/cat_viewsurf.sh /path/to/lh.thickness
```

### `CAT_SurfParameters_ui`

Computes curvature-based surface parameters (curvature, fractal dimension, surface area, sulcus depth) from surface mesh files. Wraps `CAT_SurfCurvature`, `CAT_SurfFractalDimension`, `CAT_SurfArea`, `CAT_SurfRatio`, and `CAT_SurfSulcusDepth`.

```bash
./scripts/CAT_SurfParameters_ui [options] <surface_file> [<surface_file> ...]
```

- Automatically processes both hemispheres when given a `lh.*` file
- Supports GIfTI (`.gii`) and OBJ (`.obj`) input formats
- Output: text files or GIfTI (with `-gifti` flag)

### `CAT_SurfResampleMulti_ui`

Resamples surface data to a target sphere (default: 32k), with optional smoothing. Wraps `CAT_SurfResampleMulti`.

```bash
# Resample and smooth (default FWHM=12)
./scripts/CAT_SurfResampleMulti_ui lh.thickness.gii

# Custom smoothing kernel
./scripts/CAT_SurfResampleMulti_ui --fwhm 20 lh.thickness.gii
```

- Input: left hemisphere value files (right hemisphere auto-derived)
- Output: combined GIfTI with resampled LH (+RH) values

### `CAT_Surf2ROIMulti_ui`

Extracts ROI-wise values from surface data using atlas parcellations. Wraps `CAT_Surf2ROIMulti`.

```bash
# Default atlases (aparc_DK40, aparc_a2009s)
./scripts/CAT_Surf2ROIMulti_ui lh.thickness.gii

# Custom atlas
./scripts/CAT_Surf2ROIMulti_ui --annot "'aparc_DK40.freesurfer'" lh.thickness.gii
```

- Input: left hemisphere value files (sphere.reg and right hemisphere auto-derived)
- Output: one ROI table (JSON) per input file

### `CAT_VolSmooth_ui`

Smooths volumetric NIfTI data with a Gaussian kernel. Wraps `CAT_VolSmooth`.

```bash
# Default FWHM=6mm
./scripts/CAT_VolSmooth_ui volume.nii.gz

# Custom smoothing kernel
./scripts/CAT_VolSmooth_ui --fwhm 8 volume.nii.gz
```

### `CAT_GrepJson`

Extracts specific fields from T1Prep JSON report files (generated during processing).

```bash
# Extract total intracranial volume
./scripts/CAT_GrepJson --field subjectmeasures.vol_TIV report*.json

# Save output to file
./scripts/CAT_GrepJson --field subjectmeasures.vol_abs_CGW --out volumes.txt report*.json
```

**Common field names:**
| Field | Description |
|-------|-------------|
| `subjectmeasures.vol_TIV` | Total intracranial volume |
| `subjectmeasures.vol_abs_CGW` | Absolute tissue volumes [CSF, GM, WM+WMH] |
| `subjectmeasures.vol_rel_CGW` | Relative volumes [CSF, GM, WM+WMH] / TIV |
| `subjectmeasures.vol_WMH` | WMH volume in mL |
| `subjectmeasures.vol_rel_WMH` | WMH volume relative to WM |
| `qualitymeasures` | All quality measures |

---

## Internal Utilities

These scripts are used internally by the pipeline and typically not called directly by users.

### `utils.sh`

Shared bash utility functions sourced by most other scripts. Provides:
- `exit_if_empty` — Argument validation
- `check_python_cmd` / `check_python_module` / `check_python_libraries` — Python environment checks
- `get_OS` — OS and binary directory detection
- `check_files` — Input file validation
- `run_cmd_log` — Command execution with logging
- `filter_arguments` — Argument filtering
- Text formatting constants (`BOLD`, `RED`, `GREEN`, etc.)

### `parallelize`

Generic job parallelization engine. Distributes a list of input files across multiple worker processes, monitors progress, and handles cleanup.

```bash
# Used internally by T1Prep and CAT_ scripts
./scripts/parallelize -p 4 -c "command_to_run" file1.nii file2.nii ...
```

**Options:**
| Flag | Description |
|------|-------------|
| `-p N` | Number of parallel jobs |
| `-m N` | Memory limit per job (GB) |
| `-l DIR` | Log directory |
| `-d N` | Delay (seconds) between job starts |
| `-c CMD` | Command template to execute per file |
| `-b` | Run in background |

### `progress_bar_multi.sh`

Displays real-time progress bars with ETA for parallel jobs. Supports single-job and multi-job modes with per-job and overall progress tracking.

```bash
# Used internally by parallelize
./scripts/progress_bar_multi.sh <n_jobs> <progress_dir> [width] [label]
```
