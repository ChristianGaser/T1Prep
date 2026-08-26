# Scripts

This directory contains the shell scripts behind the T1Prep command-line interface, environment helpers, processing utilities, and CAT surface/volume tools. All scripts automatically activate the project virtual environment (`env/`) when needed.

> **Installed usage:** after `pip install` (or the bash bootstrapper), these
> entry points are placed into the environment's `bin/` directory — put that on
> your `PATH` and call them directly (`T1Prep`, `PyCAT`, `t1prep-ui`, `t1prep-run`,
> `CAT_SurfView`, `CAT_VolView`, `t1prep-download-models`, the `CAT_*_ui` helpers). The
> `./scripts/<name>` form shown below is the source-tree/dev fallback; the
> `scripts/` folder itself does not need to be on `PATH`.

---

## Table of Contents

- [Main Pipeline](#main-pipeline)
- [Environment & Installation](#environment--installation)
- [Longitudinal Processing](#longitudinal-processing)
- [Evaluation](#evaluation)
- [CAT Surface & Volume Tools](#cat-surface--volume-tools)
- [Internal Utilities](#internal-utilities)

---

## Main Pipeline

### `T1Prep`

The primary CLI entry point for the entire T1Prep pipeline. It performs preprocessing, segmentation, and cortical surface reconstruction on T1-weighted MRI images.

```bash
# Show all available options
T1Prep --help

# Process a single file
T1Prep --out-dir /tmp/out sub-01_T1w.nii.gz

# Batch process with automatic parallelization
T1Prep --multi -1 --out-dir /tmp/out *.nii.gz

# Segmentation only (no surface estimation)
T1Prep --no-surf sub-01_T1w.nii.gz

# Fast mode (skip spherical registration, atlas, pial/white surfaces)
T1Prep --fast sub-01_T1w.nii.gz
```

**Key features:**
- Bias-field correction, skull-stripping, tissue segmentation, lesion detection
- Cortical surface reconstruction and thickness estimation
- Automatic parallelization across multiple files (`--multi`)
- Supports both CAT-style and BIDS derivatives naming (`--bids`)
- Configurable via `T1Prep_defaults.txt`

### `PyCAT`

A symbolic link to `T1Prep` — same script, same options, same behaviour. The only
difference is the startup banner: `logo()` in `T1Prep_utils.sh` checks
`basename "$0"` and prints the PyCAT wordmark instead of the T1Prep one.

```bash
PyCAT --out-dir /tmp/out sub-01_T1w.nii.gz   # identical to the T1Prep call above
```

`setuptools` dereferences the symlink at build time, so an installed environment
gets `bin/PyCAT` as a second copy of the orchestrator; the `$0` dispatch keeps
the banner correct either way.

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

Bash bootstrapper that downloads a T1Prep release tarball from GitHub,
extracts it to a chosen directory, and runs `scripts/T1Prep --install`
to create a managed virtual environment.  Useful for users who want the
full source tree (e.g. to run the bash orchestrator with multi-job
parallelism) without needing pip themselves.

> **For most Python users, `pip install T1Prep` is simpler** — it pulls
> the same package from PyPI without a source checkout and handles
> versioning via standard pip specifiers.  Use `install.sh` only if you
> specifically need the bundled bash scripts on disk.

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

# Run the viewers from a source checkout
./scripts/run_with_env.sh src/t1prep/gui/cat_surf_view.py --help
./scripts/run_with_env.sh src/t1prep/gui/cat_vol_view.py --help
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

# Compare voxel-to-voxel, ignoring the NIfTI affines (same shape required)
./scripts/dice.sh --gt GT.nii.gz --pred PRED.nii.gz --no-resample

# Select the interpreter (same flag/env var as T1Prep)
./scripts/dice.sh --python python3.12 --gt GT.nii.gz --pred PRED.nii.gz
```

**Python discovery:** identical to `T1Prep` — `T1Prep_utils.sh` resolves the source-tree vs. installed layout, `--python <FILE>` / `$T1PREP_PYTHON` override the auto-detected interpreter, the project venv (`env/`) is activated in a source checkout, and the module form `python -m t1prep.dice` is used in both layouts.

**Output (default):** `[dice_label_1, dice_label_2, ...] generalized_dice, dice_weighted`

**Geometry:** the NIfTI affine (`sform`/`qform`) of both images is honoured. If `--pred` differs from `--gt` in shape, voxel size, orientation or rotation, it is resampled onto the grid of `--gt` — nearest neighbour for label maps, trilinear with `--soft` — and a note is written to stderr. Pass `--no-resample` to disable this and compare voxel-to-voxel.

### `qa_calibrate.py`

Derives the rating bounds of the image quality measures (`_RATING_BOUNDS` in `src/t1prep/qa.py`) from a processed BrainWeb Phantom (BWP) set. This is the Python counterpart of CAT12's `calc_limits_QA.m`: a robust line is fitted through each measure as a function of the simulated degradation level and evaluated at level 1 (mark 1, "best") and level 6 (mark 6, "worst").

```bash
# Process the BWP volumes first, then calibrate from their report JSONs
T1Prep --out-dir /data/BWP /data/BWP/BWPC_HC_T1_pn*_rf*.nii.gz
python scripts/qa_calibrate.py /data/BWP/report

# Only a single measure
python scripts/qa_calibrate.py /data/BWP/report --measure NCR
```

**Input:** report JSON files whose names contain the BWP noise and inhomogeneity levels (e.g. `log_BWPC_HC_T1_pn3_rf040pA_vx100x100x100.json`). `pn1..pn9` and `rf020..rf100` are mapped linearly onto a 1–5 degradation scale.

**Output:** per measure the fitted line, its `R²`, the leakage of the *other* degradation factor (how much inhomogeneity contaminates the noise measure and vice versa), and a ready-to-paste `_RATING_BOUNDS` entry.

---

## CAT Surface & Volume Tools

These scripts provide user-friendly wrappers around the compiled CAT-Surface binaries in `src/t1prep/bin/` for post-processing tasks (surface parameters, resampling, ROI extraction, volume smoothing). The main T1Prep pipeline uses the `cat-surf` Python package instead of these binaries. They support batch processing with built-in parallelization via `parallelize`.

### `make_macos_apps.sh`

Builds macOS application bundles for the two viewers, so they can be started from the Dock,
Spotlight or Finder instead of a terminal. The script is a thin wrapper around the
installed `t1prep-make-apps` command (`src/t1prep/gui/make_apps.py`); it uses that entry
point when it is on `PATH`, otherwise the project venv, otherwise plain `python3` — the
implementation needs nothing but the standard library.

```bash
t1prep-make-apps                          # installed: /Applications or ~/Applications
./scripts/make_macos_apps.sh              # same, from a source checkout
t1prep-make-apps -o ~/Desktop -p /path/to/env/bin -d
```

**They also appear on their own:** the first interactive start of `CAT_SurfView` or
`CAT_VolView` on macOS creates the bundles if none exist yet, so `pip install` plus one run
is enough. `T1PREP_NO_APPS=1` switches that off, and batch runs (`--screenshot`,
`-output`) never do it. There is no hook in `pip install` itself — wheels have no
post-install step, and files placed outside the environment could not be removed by
`pip uninstall`.

The bundles are thin: each one only launches the installed `CAT_SurfView` / `CAT_VolView`
entry point, so they follow every update of that installation (and stop working if it is
removed). Double-clicking an app asks for a file; dropping files on its icon, or *Open
With* in Finder, opens them directly. The T1Prep logo is used as icon when macOS can
render it.

**File types.** Finder routes documents by Uniform Type Identifier, not by extension, so
the bundles declare them: `CAT_VolView` imports the system type `gov.nih.nifti-1` (`.nii`)
and `CAT_SurfView` exports its own for `.gii` and `.annot`, which nothing else on macOS
declares. `.mnc`, `.nrrd` and `.mha/.mhd` are registered by extension only.

**`.nii.gz` is a special case:** macOS looks at the last extension only, so such a file is
a gzip archive (`org.gnu.gnu-zip-archive`). `CAT_VolView` registers for that type as an
*alternate* handler — it appears under *Open With* for `.nii.gz` without claiming every
`.gz` file on the system.

**Always open with the viewer.** Which app owns a type is a user setting. Either select a
file in Finder, press ⌘I, pick the app under *Open with* and click *Change All…* (once for
`.nii`, once for `.nii.gz`), or install [duti](https://github.com/moretension/duti)
(`brew install duti`) and re-run the script with `-d`, which sets the defaults for you.
The apps also have to live where Launch Services looks — `/Applications` or
`~/Applications`; the script registers them with `lsregister` either way.

**When an app does not start.** Finder throws away whatever the program prints, so the
bundle puts the reason in a dialog and the whole report in `~/Library/Logs/T1Prep/<name>.log`
— check there first. The bundles point at the interpreter and the package of the
environment they were built from, so re-run the script after moving or reinstalling that
environment.

### `CAT_SurfView` / `CAT_VolView`

The two viewers (PySide6/VTK) are installed as console scripts, not as wrappers in this
folder. From a source checkout they run through the environment wrapper:

```bash
scripts/run_with_env.sh src/t1prep/gui/cat_surf_view.py lh.central.gii
scripts/run_with_env.sh src/t1prep/gui/cat_vol_view.py T1.nii.gz
```

What they can do — overlays, atlases, cluster tables, montages, batch figures — is
documented in **[docs/viewers.md](../docs/viewers.md)**.

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

### `T1Prep_utils.sh`

Shared bash utility functions sourced by most other scripts. Provides:
- `exit_if_empty` — Argument validation
- `check_python_cmd` / `check_python_module` / `check_python_libraries` — Python environment checks
- `check_files` — Input file validation
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
