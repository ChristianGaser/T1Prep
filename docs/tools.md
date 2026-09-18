# Additional tools

T1Prep installs more than the pipeline itself: a web UI, four small GUIs for
surface post-processing, a histogram plotter, and the two viewers (documented
separately in [viewers.md](viewers.md)).

| Command | What it is |
|---------|------------|
| `t1prep-ui` | Web UI for the pipeline |
| `CAT_SurfView` / `CAT_VolView` | Surface and volume viewers — see [viewers.md](viewers.md) |
| `CAT_SurfResampleMulti_ui` | Resample and smooth surface data |
| `CAT_SurfParameters_ui` | Extract surface parameters |
| `CAT_Surf2ROIMulti_ui` | Map surface values to atlas ROIs |
| `CAT_VolDiff` | Voxel-wise differences between volumes |
| `CAT_PlotHistogram` | Histograms of volumes, surfaces or text data |
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

---

## `CAT_PlotHistogram`

Draws the histogram of one or more data sets, the Python counterpart of
CAT12's `cat_plot_histogram.m`. Every input is reduced to a one-dimensional
sample, binned on a grid shared by all inputs, and drawn as one line per
input, so distributions can be compared directly.

```bash
CAT_PlotHistogram mwp1*.nii.gz                  # one line per input
CAT_PlotHistogram --mean p1*.nii.gz             # + average with standard error
CAT_PlotHistogram --dist none --xrange 0 6 \
    --save /tmp/hist lh.thickness.* rh.thickness.*
```

Reads NIfTI volumes (`.nii`, `.nii.gz`, `.img`), GIFTI surface overlays,
FreeSurfer morphometry data (`lh.thickness`, `lh.curv`, ...) and plain text.
Volumes, GIFTI and text files lose their zero background — zeros become NaN
once they make up more than 1% of the data — while FreeSurfer data keeps its
zeros, where zero is a value like any other.

Options:
- `--dist <NAME>` the curve fitted through each sample. `kernel` (the default)
  is a Gaussian kernel density estimate, `none` draws the plain histogram and
  adds a second figure with all inputs pooled, and the parametric families are
  `normal`, `gamma`, `rician`, `rayleigh`, `weibull`, `lognormal`,
  `exponential`, `beta`, `logistic` and `tlocationscale`. The families with
  positive support are fitted with the location pinned to zero, as MATLAB's
  `fitdist` does, and refuse data that reaches below it.
- `--rawline {0,1,2}` the raw histogram as a dotted line next to a fitted
  curve — never, always, or (the default) only for fewer than six inputs
- `--bins <N>` upper limit on the number of bins (default 500; without
  `--xrange` about one bin per 100 values of the first input is used)
- `--xrange MIN MAX` the range to bin over, `--xlim` / `--ylim` the drawn axes
- `--no-norm-frequency` plot counts instead of normalizing each histogram by
  its own total, which is what makes inputs of different size comparable
- `--mean` the average of all histograms with its standard error as a shaded
  band, in its own figure (number 11)
- `--color <NAME>` a categorical palette from CAT12's `cat_io_colormaps`:
  `nejm` (default), `jco`, `jama`, `d3`, `set1`–`set3`, `accent`, `dark2`,
  `paired`. `--alpha` sets the line opacity; outside `[0, 1]` it follows the
  number of inputs.
- `--winsize W H` figure size in pixels, `--fig N` the figure to draw into
- `--save <PREFIX>` write `PREFIX.png`, `PREFIX_mean.png`, ... instead of
  opening windows; `--quiet` suppresses the printed table

Printed per input: mean, median, standard deviation, effect size and the peak
of the histogram. For `spmT*` maps and effect size maps (names starting with
`D`) the table shows the upper 5% tail cutoff `TH5` instead, and each legend
entry is labelled with it.

Two inputs of identical shape additionally get a density scatter plot, and for
volumes the three orthogonal projections of their difference in CAT12's
diverging colormap (blue where the first is larger, red where the second is).
`--no-scatter` turns both off. The surface rendering of a difference between
two GIFTI files is not reproduced here — use
[`CAT_SurfView`](viewers.md) for that.

It is also importable, and then returns what it drew instead of only drawing
it:

```python
from t1prep.plot_histogram import plot_histogram

result = plot_histogram(["p1a.nii.gz", "p1b.nii.gz"], dist="kernel")
print([(s.name, s.mean, s.std, s.max_freq) for s in result.stats])
result.figures["histogram"].savefig("hist.png")
```

`plot_histogram()` takes file names, a single array, a list of arrays or a
matrix whose smaller dimension counts the data sets, and returns the bin
centers, the histograms, the fitted curves, the per-input statistics and the
figures. Repeated calls with `mean=True` overlay their averages in figure 11,
each in the next colour, so a caller can label them with one `legend()` call.
