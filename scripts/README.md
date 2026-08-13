# Scripts

This directory contains the shell scripts behind the T1Prep command-line interface, environment helpers, processing utilities, and CAT surface/volume tools. All scripts automatically activate the project virtual environment (`env/`) when needed.

> **Installed usage:** after `pip install` (or the bash bootstrapper), these
> entry points are placed into the environment's `bin/` directory — put that on
> your `PATH` and call them directly (`T1Prep`, `t1prep-ui`, `t1prep-run`,
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

---

## CAT Surface & Volume Tools

These scripts provide user-friendly wrappers around the compiled CAT-Surface binaries in `src/t1prep/bin/` for post-processing tasks (surface parameters, resampling, ROI extraction, volume smoothing). The main T1Prep pipeline uses the `cat-surf` Python package instead of these binaries. They support batch processing with built-in parallelization via `parallelize`.

### `make_macos_apps.sh`

Builds macOS application bundles for the two viewers, so they can be started from the Dock,
Spotlight or Finder instead of a terminal.

```bash
# into /Applications (or ~/Applications when that is not writable)
./scripts/make_macos_apps.sh

# somewhere else, or against another installation
./scripts/make_macos_apps.sh -o ~/Desktop -p /path/to/env/bin
```

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

### `CAT_SurfView` / `CAT_VolView`

The two viewers (PySide6/VTK) are installed as console scripts, not as wrappers in this
folder: `CAT_SurfView` shows cortical meshes and overlays in a six-view montage,
`CAT_VolView` shows a volume as three orthogonal slices. Run either without arguments (or
with `-h`) for the full help. From a source checkout without installing, use
`./scripts/run_with_env.sh src/t1prep/gui/cat_surf_view.py …`.

```bash
# View a surface mesh
CAT_SurfView /path/to/lh.central.gii

# View a surface overlay (e.g., thickness) – the mesh is found automatically
CAT_SurfView /path/to/lh.thickness

# Several overlays, stepped through with the ←/→ keys
CAT_SurfView sub-*/lh.thickness.*

# CAT12/SPM statistic results with a fixed range and everything up to 6 hidden
CAT_SurfView -range 6 16 -clip -100 6 -colorbar stat/logP_*.gii

# Predefined settings (-preset 1: C3 colormap with 16 discrete levels)
CAT_SurfView -preset 1 -colorbar lh.thickness.subj

# Volume with surface outlines drawn onto the slices
CAT_VolView T1.nii.gz lh.central.gii rh.central.gii

# Up to three volumes: one window each, cursors linked
CAT_VolView T1.nii.gz p1T1.nii.gz p2T1.nii.gz

# Statistic map in colour on top of a T1 (same voxel grid)
CAT_VolView T1.nii.gz --overlay TFCE_log_pFWE.nii.gz
```

**Linked volume view.** `-volume <image>` (or the *Open NIfTI…* button) opens the three
orthogonal slices of a volume next to the surface, sharing one millimetre space: clicking
the surface moves the slices, and clicking or scrolling a slice marks the closest surface
point in every montage view. It is the same window `CAT_VolView` opens on its own, so both
offer the identical slices and right-click menu. Slices are shown in neurological
orientation (left is left) in the millimetre space of the NIfTI sform/qform, and
`--screenshot` writes a PNG without opening a window.

Up to three volumes can be given at once: each opens its own window, the windows are tiled
side by side, and their cursors are linked — clicking in one moves the others to the same
millimetre position, so the same anatomical point is shown even when the volumes differ in
grid, voxel size or orientation.

The free quadrant carries an information panel: file name, dimensions, voxel size,
orientation code, data type and intensity range, plus the voxel index, mm coordinates and
value under the cursor. The right-click menu holds:

- **Zoom** — full volume or a 160/80/40/20/10 mm bounding box, as in the SPM ortho viewer;
  a zoomed view follows the cursor, keeping the picked point in the middle of the pane.
  *Re-centre on cursor* in the same submenu turns that off (`--no-recenter`), so the view
  stays where it is and only picking a zoom level moves it. The cursor itself is not
  rounded to the voxel grid, so it sits exactly where it was placed — only the displayed
  slices and the intensity readout are voxel-wise.
- **Atlas** — name the region under the cursor from any atlas shipped with T1Prep (or one
  of your own via *Other…*, or `--atlas` on the command line). The atlas is sampled at the
  mm position of the cursor, so pick one only when the displayed image is registered to its
  space; *None* switches the lookup off.
- **Raw voxels (nearest neighbour)** — draw the slices unsmoothed to see the data as
  stored, useful for segmentation edges and resampling artefacts (`--nearest`). The
  reported intensity follows: trilinear at the exact cursor position while smoothing
  is on, the untouched voxel value with raw voxels selected.
- **Overlay** — draw a second volume in colour on top (`--overlay`, or *Open…*). It has to
  be on the same voxel grid (same dimensions and voxel size); anything else is refused
  rather than silently resampled. The reported intensity is then the overlay's, with the
  image value kept on a `background` line. The overlay is always drawn with nearest
  neighbour, so a thresholded map keeps its edges; clipped values and voxels the map has
  no value for (NaN outside a statistic mask) are left unpainted, so the image shows
  through them.
- **Controls** — the control panel of `CAT_SurfView`, wired to the overlay: value range,
  clip window, p-value thresholds for `log`-named overlays, image intensity range,
  opacity, colormap, discrete levels and inversion.
- **Image information** — hide or show the panel (`--no-info` starts without it).

**Presets.** `-preset N` applies a predefined set of options — currently `1` for the C3
colormap with 16 discrete levels. Anything given explicitly on the command line takes
precedence, so `-preset 1 -fire` keeps the fire colormap and the 16 levels. Further presets
are added in the `PRESETS` table of `src/t1prep/gui/cat_surf_view.py`.

**Statistic results.** When the overlay name contains `log` (CAT12/SPM `logP_*` files), the
colorbar is labelled with p-values instead of the raw -log10(p) values, as in
`cat_surf_results`: `1.3` → `0.05`, `2` → `0.01`, `3` → `0.001`, and `1e-08` beyond `p<1e-7`.
Thresholded maps (`-clip -1.3 1.3`) get their first tick at exactly ±log10(0.05). Use `-log`
to force the p-value labels for files that do not follow the naming convention.

For such overlays the control panel also shows a **Threshold** entry with the three common
levels — `p<0.05`, `p<0.01`, `p<0.001` (and `none`) — which set the clip window to
±log10(p), hiding everything below it and greying that band on the colorbar, exactly like
the threshold menu of `cat_surf_results`. It stays in sync with the clip spin boxes, so
`-clip -2 2` on the command line starts on `p<0.01`.

**How the surface is found.** An overlay file does not reference the surface it belongs to,
so it is resolved in this order:

1. geometry stored inside the overlay file itself (CAT12 `mesh.*` files and statistic
   results usually carry it),
2. the mesh matching the overlay name (`lh.thickness.subj` → `lh.central.subj.gii`) or a
   `central`/`midthickness` surface in the same folder,
3. the number of values, matched against the shipped 4k/32k/164k templates.

Step 3 is what makes free-form names work — most notably CAT12/SPM statistic results
(`logP_age_(...)_pFWE0.1_k0.gii`, `TFCE_log_pFWE_0001.gii`). Any `.gii` that holds values but
no surface is treated as an overlay, so such results also load when they were copied away
from their `SPM.mat`. Because the lookup runs for every overlay, files from different
folders, subjects and mesh resolutions can be mixed in a single call. The right hemisphere is added when an `rh.`/`right`/`_hemi-R_` file
sits next to the left one, or when a mesh (`mesh.central.*`) or overlay holds both
hemispheres back to back — those are split so all six views are shown.

**Batch use.** `-output` renders the view, writes the PNG and exits, so the viewer can be
called from a loop without any interaction:

```bash
for f in sub-*/lh.thickness.*; do
  CAT_SurfView -range 1 5 -colorbar -output "$(dirname "$f")/thickness.png" "$f"
done
```

**Keys.** `←/→` previous/next overlay (or mesh), `u/d/l/r` rotate, `o` reset view,
`b` flip dorsal views, `w/s` wireframe/shaded, `g` screenshot, `h` key help, `q` quit.

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
