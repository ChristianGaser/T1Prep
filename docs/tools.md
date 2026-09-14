# Additional tools

T1Prep installs more than the pipeline itself: a web UI, four small GUIs for
surface post-processing, and the two viewers (documented separately in
[viewers.md](viewers.md)).

| Command | What it is |
|---------|------------|
| `t1prep-ui` | Web UI for the pipeline |
| `CAT_SurfView` / `CAT_VolView` | Surface and volume viewers — see [viewers.md](viewers.md) |
| `CAT_SurfResampleMulti_ui` | Resample and smooth surface data |
| `CAT_SurfParameters_ui` | Extract surface parameters |
| `CAT_Surf2ROIMulti_ui` | Map surface values to atlas ROIs |
| `CAT_VolDiff` | Voxel-wise differences between volumes |
| `t1prep-make-apps` | macOS: build the viewer `.app` bundles |
| `t1prep-download-models` | Fetch the model weights ahead of time |

Every one of them follows the same command-line convention as `T1Prep`: no
argument prints the synopsis, `--help` the full description, `--version` the
release, and options are spelled with two dashes. See
[usage.md](usage.md#command-line-conventions) for the details, including the
single-dash spellings that are still accepted.

Back to the [README](../README.md).

---

## Web UI (Flask)

A minimal browser-based UI is available for local use. It uploads selected NIfTI
files, lets you configure General and Save options, and can schedule jobs to
start at a specific time.

```bash
t1prep-ui
```

By default the Web UI runs on port 5050. To use a different port:

```bash
t1prep-ui 5500
```

When started, the UI will try to open an app-style window (Chrome if available,
otherwise your default browser). You can also open the URL manually in any
browser.

Then open http://127.0.0.1:5050 (or the port you selected) in your browser.

To prevent auto-opening a browser window:

```bash
t1prep-ui --no-browser
```

Uploaded files are stored under `webui_uploads/` (in the current working
directory) and per-job logs under `webui_jobs/`.

In addition to `T1Prep`, the following commands — all installed into the
environment's `bin/` — provide convenient entry points for the Web UI and
CAT-Surface post-processing.

### `t1prep-ui`

Launches the Flask Web UI (same tool described in the [Web UI (Flask)](#web-ui-flask) section).

```bash
t1prep-ui
t1prep-ui 5500
t1prep-ui --no-browser
```

- Default port: `5050`
- Optional positional port argument (e.g., `5500`)
- `--no-browser` disables auto-launching a browser/app window

### `CAT_SurfResampleMulti_ui`

Resamples LH/RH surface values to target spheres and writes a combined output
per LH input using `CAT_SurfResampleMulti`.

```bash
CAT_SurfResampleMulti_ui [options] lh.thickness.subject.gii
```

Common options:
- `--out <DIR>` output directory
- `--res <STR>` output surface resolution (`32k` or `4k`)
- `--fwhm <FLOAT>` smoothing FWHM
- `--trg-sphere <FILE>` target LH sphere
- `--mask <FILE>` target LH mask
- `--jobs <N>` parallel worker count

Input expectations:
- Supports `lh.*` naming and auto-derives RH counterparts
- BIDS-style `*_left*` naming is currently not implemented

### `CAT_SurfParameters_ui`

Computes surface parameters from mesh files using CAT-Surface binaries
(`CAT_SurfCurvature`, `CAT_SurfFractalDimension`, `CAT_SurfArea`,
`CAT_SurfRatio`, `CAT_SurfSulcusDepth`) bundled in `src/t1prep/bin/`.

```bash
CAT_SurfParameters_ui [options] lh.central.gii
```

Common options:
- `--gy`, `--mc`, `--gc`, `--cv`, `--si`, `--sh`, `--fi`, `--area`, `--fd`, `--sr`, `--sra`
- `--depth`, `--sqrt-depth`, `--min-curv`, `--max-curv`, `--dp`
- `--gifti` write GIfTI output
- `--noclobber` do not overwrite existing files
- `--jobs <N>` / `--no-parallel` parallel control

Input expectations:
- Accepts `.obj` and `.gii`
- For `lh.*` files, matching `rh.*` is processed automatically when available

### `CAT_Surf2ROIMulti_ui`

Extracts ROI-wise values from surface value files using `CAT_Surf2ROIMulti`.
For each LH input, RH files are derived automatically.

```bash
CAT_Surf2ROIMulti_ui [options] lh.thickness.subject.gii
```

Common options:
- `--out <DIR>` output directory
- `--res <STR>` surface/atlas resolution (default `32k`)
- `--trg-sphere <FILE>` target LH sphere
- `--annot <NAMES>` one or multiple atlas names
- `--jobs <N>` / `--no-parallel` parallel control

Atlas names for `--annot` are resolved as:
- `src/t1prep/data/atlases_surfaces_<res>/lh.<name>.annot`
- `src/t1prep/data/atlases_surfaces_<res>/rh.<name>.annot`

Multi-atlas examples:

```bash
CAT_Surf2ROIMulti_ui --annot "'aparc_DK40.freesurfer' 'aparc_a2009s.freesurfer'" lh.thickness.subject.gii
CAT_Surf2ROIMulti_ui --annot "aparc_DK40.freesurfer,aparc_a2009s.freesurfer" lh.thickness.subject.gii
```

### `CAT_VolDiff`

Computes voxel-wise differences between volumes with `CAT_VolCalc`, following
CAT12's `cat_stat_diff.m`. Within a subject the first image is the reference,
and every further image `j` gives `image_j - image_1`, written next to
`image_j` as `diff_<name>`.

```bash
CAT_VolDiff tp1.nii tp2.nii tp3.nii        # diff_tp2.nii, diff_tp3.nii
CAT_VolDiff --subject s1_tp1.nii s1_tp2.nii --subject s2_tp1.nii s2_tp2.nii
```

Options:
- `--subject <FILES>` images of one subject (reference first); repeat per subject
- `--rel` relative difference in percent, `200*(i2-i1)/(i1+i2)`, written as `diffrel_<name>`
- `--glob` scale the images of a subject to their common global mean first
  (as `spm_global` computes it), so a global intensity factor cancels out
- `--quiet` no progress output

Input expectations:
- The images of a subject must share one grid; nothing is resliced
- Output is float32 with the header of the reference image
