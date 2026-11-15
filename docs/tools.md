# Additional tools

T1Prep installs more than the pipeline itself: a desktop launcher, command-line
helpers for post-processing (three for surfaces, two for volumes), a tool that
prints fields of the JSON reports, the job runner `parallelize`, a histogram
plotter, a box and violin plotter, and the two viewers (documented separately
in [viewers.md](viewers.md)).

| Command | What it is |
|---------|------------|
| `t1prep-gui` | Desktop launcher: choose images, run the pipeline, browse the results |
| `CAT_SurfView` / `CAT_VolView` | Surface and volume viewers — see [viewers.md](viewers.md) |
| `CAT_SurfResampleMulti_ui` | Resample and smooth surface data |
| `CAT_SurfParameters_ui` | Extract surface parameters |
| `CAT_Surf2ROIMulti_ui` | Map surface values to atlas ROIs |
| `CAT_VolDiff` | Voxel-wise differences between volumes |
| `CAT_VolSmooth_ui` | Smooth volumes with a Gaussian kernel |
| `CAT_GrepJson` | Print a field of the JSON reports |
| `parallelize` | Run a command on many files in parallel |
| `CAT_PlotHistogram` | Histograms of volumes, surfaces or text data |
| `CAT_PlotBoxplot` | Box and violin plots of volumes, surfaces or text data |
| `t1prep-make-apps` | macOS: build the viewer `.app` bundles |
| `t1prep-download-models` | Fetch the model weights ahead of time |
| `t1prep-bbreg`, `process_longitudinal.sh` and its helpers | fMRIPrep registration and the longitudinal pipeline — see [usage.md](usage.md) |

They follow the same command-line convention as `T1Prep`: no argument prints
the synopsis, `--help` the full description, `--version` the release, and
options are spelled with two dashes. Three of them do their job when called
without an argument instead: `t1prep-gui` opens its window,
`t1prep-download-models` fetches the weights and `t1prep-make-apps` builds the
app bundles. See
[usage.md](usage.md#command-line-conventions) for the details, including the
single-dash spellings that are still accepted.

Back to the [README](../README.md).

---

## Launcher (Qt)

`t1prep-gui` is the desktop front end of the pipeline: add images, choose the
options, run, and look at the results, all in one window.

```bash
t1prep-gui                                        # open the window
t1prep-gui sub-01_T1w.nii.gz sub-02_T1w.nii.gz    # with these images in the table
t1prep-gui --out-dir /data/results /data/study/*.nii
t1prep-gui /data/results                          # browse the results of an earlier run
```

From a source checkout, without installing: `./scripts/T1Prep_gui`, which takes the same arguments.

- **Images.** *Add images…*, *Add folder…* (a name pattern, optionally with
  subfolders; folders that hold T1Prep results are left out) or drag and drop.
  The images are processed where they are; nothing is copied.
- **Options.** The form starts from the defaults file of the `T1Prep` it runs
  (`T1Prep_defaults.txt` next to it), so it shows what that `T1Prep` would do
  on its own. What another choice forces is shown as it will be used (`--bids`
  compresses the volumes, `--fmriprep` implies BIDS names), and the options of
  a step that is switched off are disabled. The choices are remembered between
  sessions; *Restore defaults* returns to the defaults file.
- **Output.** Next to each image, as `T1Prep` does by default, or one folder
  for all results.
- **Run.** The table shows for each image whether the volume and the surface
  phase are waiting, running, finished or failed, and the image quality rating
  of its report. *Stop* ends the run and every process it started. *Skip
  finished images* (under Processing) leaves out what is complete, so a batch
  that was stopped or partly failed can be continued.
- **Results.** Selecting an image shows a picture of the segmentation outline
  or of the central surfaces on the image, the measures of its report, and
  buttons that open the segmentation and the surfaces in `CAT_VolView` and the
  thickness in `CAT_SurfView`. The picture appears as soon as a phase has
  written its results, so the segmentation can be checked while the surfaces
  are still being estimated. *Open results…* shows the results of an earlier
  run without processing anything.

The launcher itself uses neither VTK nor OpenGL. The pictures are rendered by
`CAT_VolView --headless`, and the viewers are started as processes of their
own. Where the viewers cannot start (no working OpenGL), the launcher still
runs the pipeline; only the pictures and the viewer buttons are missing.

## Post-processing helpers

In addition to `T1Prep`, the following commands — all installed into the
environment's `bin/` — provide convenient entry points for post-processing
with CAT-Surface. They are command-line tools, not graphical ones. They call
the CAT-Surface functions of the `cat-surf` package (`cat_surf.cli`), the same
package the pipeline uses; no separate binaries are involved. Several input
files are processed in parallel through `parallelize`.

### `CAT_SurfResampleMulti_ui`

Resamples LH/RH surface values to target spheres and writes a combined output
per LH input (`cat_surf.cli.surf_resample_multi`).

```bash
CAT_SurfResampleMulti_ui [options] surf/lh.thickness.subject
```

Common options:
- `--out <DIR>` output directory, created if it does not exist (default: next
  to the input)
- `--res <STR>` mesh to resample to, `32k` (default) or `4k`: it selects the
  template sphere and mask of that mesh and is part of the output name
- `--fwhm <FLOAT>` smoothing FWHM
- `--trg-sphere <FILE>`, `--mask <FILE>` another target LH sphere and mask
  than the template's
- `--jobs <N>` parallel worker count

Input expectations:
- A value file named as T1Prep writes it, `lh.<measure>.<name>` (no `.gii`);
  the central surface `lh.central.<name>.gii`, the registered sphere and the
  RH counterparts are derived from that name and have to lie next to it
- BIDS-style `*_left*` naming is currently not implemented

### `CAT_SurfParameters_ui`

Computes surface parameters from mesh files with the CAT-Surface functions of
the `cat-surf` package (`cat_surf.cli.surf_curvature`, `surf_sulcus_depth`,
`surf_fractal_dimension`, `surf_ratio`, `surf_area`).

```bash
CAT_SurfParameters_ui [options] lh.central.gii
```

Common options:
- `--gy` (the default: gyrification, the absolute mean curvature), `--mc`, `--gc`, `--cv`, `--si`, `--sh`, `--fi`, `--area`, `--fd`, `--sr`, `--sra`
- `--depth`, `--sqrt-depth`, `--min-curv`, `--max-curv`, `--dp`
- `--gifti` write GIfTI output
- `--noclobber` do not overwrite existing files
- `--jobs <N>` / `--no-parallel` parallel control

Input expectations:
- Accepts `.obj` and `.gii`
- For `lh.*` files, matching `rh.*` is processed automatically when available

### `CAT_Surf2ROIMulti_ui`

Extracts ROI-wise values from surface value files
(`cat_surf.cli.surf2roi_multi`). For each LH input, RH files are derived
automatically; the input is named as for `CAT_SurfResampleMulti_ui`.

```bash
CAT_Surf2ROIMulti_ui [options] surf/lh.thickness.subject
```

Common options:
- `--out <DIR>` output directory (default: for an input in a `surf` folder
  the `label` folder next to it, where T1Prep writes its own tables;
  otherwise the folder of the input)
- `--res <STR>` surface/atlas resolution (default `32k`)
- `--trg-sphere <FILE>` target LH sphere
- `--annot <NAMES>` one or multiple atlas names
- `--jobs <N>` / `--no-parallel` parallel control

Atlas names for `--annot` are those of the shipped surface atlases, the files
`lh.<name>.annot` and `rh.<name>.annot` in `t1prep/data/atlases_surfaces_32k`.

Multi-atlas examples:

```bash
CAT_Surf2ROIMulti_ui --annot "'aparc_DK40.freesurfer' 'aparc_a2009s.freesurfer'" surf/lh.thickness.subject
CAT_Surf2ROIMulti_ui --annot "aparc_DK40.freesurfer,aparc_a2009s.freesurfer" surf/lh.thickness.subject
```

Like the other `CAT_*_ui` helpers it exits with status 1 when a file could not
be processed (a missing input or atlas, a failed computation) and with 0 only
when every output was written.

### `CAT_VolDiff`

Computes voxel-wise differences between volumes (`cat_surf.cli.vol_calc`),
following CAT12's `cat_stat_diff.m`. Within a subject the first image is the reference,
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

### `CAT_VolSmooth_ui`

Smooths volumes with a Gaussian kernel (`cat_surf.cli.vol_smooth`). Each
result is written next to its input with `s<fwhm>` in front of the name.

```bash
CAT_VolSmooth_ui mwp1*.nii                 # s6mwp1*.nii, 6 mm FWHM
CAT_VolSmooth_ui --fwhm 8 mwp1*.nii        # s8mwp1*.nii
```

Options:
- `--fwhm <FLOAT>` kernel FWHM in mm (default 6)
- `--jobs <N>` / `--no-parallel` parallel control

### `CAT_GrepJson`

Prints one field of T1Prep's JSON reports for a list of files, e.g. to
collect the total intracranial volume of every subject. It needs `jq`.

```bash
CAT_GrepJson --field subjectmeasures.vol_TIV report/log_*.json
CAT_GrepJson --field qualitymeasures.IQR --out iqr.txt report/log_*.json
```

Options:
- `--field <NAME>` the field, nested ones in dot notation
- `--out <FILE>` write the output to a file instead of the terminal

### `parallelize`

Runs a command on a list of files in several processes; `T1Prep --multi` and
the helpers above use it for their batches.

```bash
parallelize --processes 4 --command "CAT_VolSmooth_ui --no-parallel --fwhm 8" mwp1*.nii
```

It exits with status 0 when the command succeeded for every file and with 1
when it failed for at least one, in which case the failed files are listed.
`parallelize --help` describes the options.

---

## `CAT_PlotHistogram`

Draws the histogram of one or more data sets, the Python counterpart of
CAT12's `cat_plot_histogram.m`. Every input is reduced to a one-dimensional
sample, binned on a grid shared by all inputs, and drawn as one line per
input, so distributions can be compared directly.

```bash
CAT_PlotHistogram mwp1*.nii.gz                  # one line per input
CAT_PlotHistogram --mean p1*.nii.gz             # + average with standard error
CAT_PlotHistogram --ridgeline lh.thickness.*    # one stacked ridge per input
CAT_PlotHistogram --dist none --xrange 0 6 \
    --save /tmp/hist lh.thickness.* rh.thickness.*
```

Reads NIfTI volumes (`.nii`, `.nii.gz`, `.img`), GIFTI surface overlays,
FreeSurfer morphometry data (`lh.thickness`, `lh.curv`, ...) and plain text
(`.txt`). A file with any other name has to be FreeSurfer morphometry data by
its header; anything else, such as a text file named `.csv`, is refused
rather than read as numbers it does not hold.
Volumes, GIFTI and text files lose their zero background — zeros become NaN
once they make up more than 1% of the data — while FreeSurfer data keeps its
zeros, where zero is a value like any other.

Options:
- `--dist <NAME>` the curve fitted through each sample. `kernel` (the default)
  is a Gaussian kernel density estimate (see below), `gaussian_kde` the
  estimate of scipy's `gaussian_kde` with Scott's rule, which seaborn uses as
  well, `none` draws the plain histogram and adds a second figure with all inputs pooled, and the parametric families are
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
- `--ridgeline` stacks the curves as a ridgeline plot instead of overlaying
  them: one filled ridge per input, the first input at the top, labelled on
  the y axis instead of in a legend. Good for many inputs, where overlaid
  lines become unreadable. By default all ridges share one height scale, so
  their heights stay comparable, as in joypy, seaborn's ridge plot and R's
  ggridges; `--ridge-height <H>` sets the tallest ridge to `H` baseline
  spacings (default 1.5; at 1 it just reaches the baseline above). The scale
  is taken within `--xlim`, so cutting off a dominant peak — the saturated
  end of a p1 map — brings out the rest. `--ridge-scale own` instead scales
  every ridge to its own maximum, so that all reach `H` and only their
  shapes compare (joypy's `ylim='own'`, ggridges' `ndensity`). `--alpha`
  sets the fill opacity here (0.6 when automatic), and `--ylim` does not
  apply.
- `--winsize W H` figure size in pixels, `--fig N` the figure to draw into
- `--save <PREFIX>` write `PREFIX.png`, `PREFIX_mean.png`, ... instead of
  opening windows; `--quiet` suppresses the printed table

Printed per input: mean, median, standard deviation, effect size and the peak
of the histogram. For `spmT*` maps and effect size maps (names starting with
`D`) the table shows the upper 5% tail cutoff `TH5` instead, and each legend
entry is labelled with it.

The kernel density estimate uses a Gaussian kernel. Its bandwidth comes from
the Improved Sheather–Jones selector (Botev et al., 2010), which does not
assume a normal distribution. It therefore resolves multimodal and skewed
distributions, such as the sharp GM peak of a p1 map, which rules of thumb
like Silverman's or Scott's flatten and widen. On data that are not a smooth
density, the selector would collapse into a spike at every value: values
stored at fixed steps (a uint8 p1 map), many tied values, or a pile-up of
near-zero probabilities. So the bandwidth never goes below a quarter of
Silverman's rule of thumb, a bound that smooth distributions do not reach.
`--dist gaussian_kde` uses the bandwidth of scipy's `gaussian_kde` instead,
Scott's rule `σ · n^(-1/5)`: smoother, but it flattens sharp peaks like the
GM peak of a p1 map. Either density is computed from every value, binned onto
a fine grid, so whole-brain maps are not thinned out. That gives scipy's
curve to within 0.03% of its peak, 50 to 80 times faster than scipy's own
evaluation (65 ms instead of 3.4 s for 2 million voxels). This differs from MATLAB's
`fitdist(y, 'kernel')`, which uses a normal-reference bandwidth, so curves
are not identical to `cat_plot_histogram.m`. The density scatter below uses
the same Gaussian kernel with the two-dimensional rule of thumb.

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

---

## `CAT_PlotBoxplot`

Draws box and violin plots of one or more data sets, the Python counterpart of
CAT12's `cat_plot_boxplot.m`. It reads the same files as
[`CAT_PlotHistogram`](#cat_plothistogram) and draws one box or violin per
file, so that the distributions of several images, or one measure over a group
of subjects, can be compared at a glance.

```bash
CAT_PlotBoxplot mwp1*.nii                                  # one box per volume
CAT_PlotBoxplot --style violinbox --sort ascending lh.thickness.*
CAT_GrepJson --field subjectmeasures.vol_TIV --out tiv.txt report/log_*.json
CAT_PlotBoxplot --style jitter --list-outliers tiv.txt     # which subjects stand out
CAT_GrepJson --field subjectmeasures.vol_abs_CGW --out cgw.txt report/log_*.json
CAT_PlotBoxplot --columns --names CSF,GM,WM --connect cgw.txt
```

Volumes and GIFTI overlays lose their zero background and FreeSurfer data keep
their zeros, as in `CAT_PlotHistogram`. Text files keep their zeros too, unlike
in `CAT_PlotHistogram`: a list of values has no background, and dropping its
zeros would shift the quartiles. In a text file the values are separated by
white space; words such as a header or a subject name are skipped, lines
starting with `#` or `%` are comments, and `N/A`, `NA`, `null` or `none` —
what `CAT_GrepJson` writes for a field a report lacks or leaves empty — is a
missing value that keeps its place. `--columns` makes every column of a text
file a box of its own, the counterpart of the matrix input of
`cat_plot_boxplot`; a header line above the values names the columns, and a
row holding only a missing value — `CAT_GrepJson`'s `N/A` for a report without
`vol_abs_CGW` — stands for a whole row of them.

`--style` picks one of the predefined styles of `cat_plot_boxplot`:

| Style | `opt.style` | Draws |
|-------|-------------|-------|
| `box` (default) | 0 | box plot |
| `violin` | 1 | violin plot, the values as a barcode |
| `violinbox` | 2 | violin plot with a thin box plot inside (without the barcode) |
| `density` | 3 | horizontal half violins, the values along the baseline |
| `jitter` | 4 | box plot with the values as points, spread by their density |

A style only sets defaults; each option below changes one thing of it.

Options:
- `--horizontal` / `--vertical` the orientation. Horizontal boxes run from the
  first input at the top downwards; `density` is horizontal by default.
- `--notch` notches the boxes of `box` and `jitter` at the confidence
  interval of the median
- `--whisker <K>` the reach of the whiskers in interquartile ranges (IQR) of
  the box (default 1.5); `0` makes every value outside the box an outlier
- `--outliers` / `--no-outliers` mark the mild outliers with `+` and the
  extreme ones with `o`. On by default for `box` and `jitter`, off for the
  violin styles: a violin shows the tails itself, and the thousands of
  outliers of a volume or a surface would bury it.
- `--list-outliers` prints every outlier with its 1-based position among the
  values of its box. For a one-column text file that is the n-th value — with
  the output of `CAT_GrepJson` the n-th report it read; a header or a comment
  line does not count. A text file of several columns read without `--columns`
  counts row by row. For a volume it prints the voxel coordinates `i,j,k`,
  counted from 1 as SPM shows them.
- `--show-data {none,strip,jitter,barcode}` draws the values themselves: as
  points on the centre line, as points spread sideways by their density (a
  sina plot), or as one short line each. At most 5000 random values per box
  are drawn; the statistics always use all of them.
- `--sort {none,ascending,descending}` orders the boxes by their median, the
  first on the left or at the top; every box keeps its colour
- `--names A,B,...` comma-separated labels; `--groupnum` adds the number of
  values to each
- `--connect` draws a line through the n-th value of every box, such as one
  subject over several time points. It needs data sets of equal size and
  draws the values as points unless the style already shows them; a missing
  value leaves a gap.
- `--width <W>` the width of a box, or of a violin at its widest, in box
  spacings. `--varwidth` scales it with the square root of the number of
  values, as R's `boxplot(varwidth = TRUE)`; `cat_plot_boxplot` scales its
  boxes with the number itself, by default.
- `--no-fill`, `--alpha <A>` the opacity of the fill between 0 and 1 (default
  0.5), `--no-grid`
- `--ylim MIN MAX` the range of the value axis — the horizontal one when the
  boxes are horizontal — as `opt.ylim`. An end given as `nan` is left to the
  data, as `-inf`/`inf` are in MATLAB (`--ylim nan 5`). Outliers beyond it are
  drawn at its edge, and every violin is scaled to its widest point within
  it. That matters for p1 maps: their many near-zero values form a spike that
  squeezes a violin into a sliver, and `--ylim 0.05 1.5` brings out the rest.
  All values still enter the density and the statistics.
- `--ylabel`, `--title`, `--color <NAME>` (the palettes of
  `CAT_PlotHistogram`), `--winsize W H`, `--fig N`
- `--save <FILE>` writes the figure instead of opening a window; the extension
  picks the format, `.png` when there is none or it is no image format (as the
  `.2` of `qc_v1.2`). `--quiet` suppresses the table.

Options are not abbreviated: `--out`, the option of `CAT_GrepJson`, is an
error here rather than `--outliers`.

Printed per box: the number of finite values, mean, standard deviation,
minimum, quartiles, maximum and the number of outliers.

The statistics are those of `cat_plot_boxplot`. The quartiles are computed as
`cat_stat_prctile` does (MATLAB's `prctile`, numpy's `hazen` method), not with
numpy's default, which matplotlib's `boxplot` uses and which gives other
quartiles for small samples. The whiskers reach the last value within `K` IQR
of the box; mild outliers lie up to `2K` IQR from the box, extreme ones
beyond — Tukey's inner and outer fences for `K = 1.5`. The notch spans the
median ± 1.57 IQR/√n, kept inside the box. The drawing is matplotlib's own
(`Axes.bxp` and `Axes.violin`), so the figures do not look like the MATLAB
ones. The violins show the Gaussian kernel density estimate of
`CAT_PlotHistogram`, whose Improved Sheather–Jones bandwidth is what
`cat_plot_boxplot` uses as well. It is computed from every value, so
whole-brain maps are not thinned out (19 mwp1 volumes take about 3 s), and cut
at the smallest and the largest value, as matplotlib cuts its violins. Each
violin is scaled to its own widest point, as in both.

Not carried over from `cat_plot_boxplot`: the shading of subsets, the dark
mode, the flip of horizontal plots, the brighter colours for repeated groups,
the median shown by two fill colours, the barcode of style 2 (`--show-data
barcode` adds it), and the design matrix `opt.I`, whose common case — one
data set per time point — `--connect` covers. Its batch mode over CAT12 XML
fields is replaced by `CAT_GrepJson` and a text file.

It is also importable, and then returns the statistics with the figure:

```python
from t1prep.plot_boxplot import plot_boxplot

result = plot_boxplot(["tiv_controls.txt", "tiv_patients.txt"], notch=True)
print(result.matrix())            # the s of cat_plot_boxplot, 7 x boxes
print(result.stats[1].outliers)   # 0-based indices of the patients' outliers
result.figure.savefig("tiv.png")
```

`plot_boxplot()` takes file names; a vector or a list of numbers, which is one
data set; a matrix or a pandas DataFrame with one column per data set — as
`cat_plot_boxplot` does, not the smaller dimension as `plot_histogram()` does;
a list of arrays; or a dict of names and arrays. A DataFrame, a named Series
and a dict bring their labels. Every argument after the figure has to be
named (`style="violin"`). `result.stats` holds one entry per data set in the
order of the input, with its outliers as 0-based indices into the flattened
data set. Missing values count, so the indices match the input positions; the
MATLAB original counts without them. `result.order` gives the order the boxes
were drawn in.
