# Using T1Prep

How to run the pipeline, what it writes and what the options mean. For
installation see [installation.md](installation.md); for the viewers and the
other tools see [viewers.md](viewers.md) and [tools.md](tools.md).

- [Usage](#usage)
- [Command-line conventions](#command-line-conventions)
- [Options](#options)
- [Python API](#python-api)
- [Output folder structure and naming](#output-folder-structure-and-naming-conventions)
- [Examples](#examples)
- [Longitudinal realignment](#longitudinal-realignment-experimental)
- [Longitudinal models](#longitudinal-models)
- [Input](#input)

Back to the [README](../README.md).

---

## Usage
```bash
T1Prep [options] file1.nii.[.gz] file2.nii[.gz] ...
```

(`T1Prep` resolves from the environment's `bin/` once it is on your `PATH`; from
a source checkout without an install you can still run `./scripts/T1Prep`.)

## Command-line conventions

Every T1Prep tool — `T1Prep`, `PyCAT`, the viewers `CAT_VolView` and
`CAT_SurfView`, `CAT_VolDiff`, the `CAT_*_ui` wrappers, `parallelize` and the
longitudinal scripts — answers the same way:

| You type | You get |
|----------|---------|
| nothing at all | the synopsis: the command line and an overview of the options |
| `--help` | the full description of every option, with examples |
| `--version` | the T1Prep version the tool belongs to |

Options are spelled with two dashes throughout. The single-dash spellings some
tools used before (`CAT_SurfView -overlay`, `CAT_VolDiff -s`, `parallelize -p`)
are still accepted so existing command lines and scripts keep working; they are
no longer listed in the help.

Two tools do a job rather than take a file, so calling them without an argument
runs that job instead of printing the synopsis: `t1prep-ui` starts the web
interface, and `t1prep-download-models` fetches the model weights. Both still
answer `--help`.

## Options
Call T1Prep without an argument for the overview, and with `--help` for the
full description of each option:
```bash
T1Prep            # synopsis
T1Prep --help     # every option, in full
```

Skull-stripping modes:
- `--skullstrip-only`: run skull-stripping only and exit after writing a skull-stripped image and brain mask.
- `--no-skullstrip` / `--skip-skullstrip`: skip skull-stripping (assumes input is already skull-stripped).

Longitudinal / advanced flags:
- `--initial-surf <FILE>`: use an initial surface estimate for longitudinal processing.
- `--long-data <PATH>`: process the volume at `<PATH>` while keeping output naming/folders based on the provided input file.
- `--no-atlas`: disable atlas labeling (overrides any defaults file atlas selection).

Segmentation refinement:
- `--no-vessel`: disable the blood-vessel correction.
- `--nogm-model`: remove non-cortical grey matter with the DeepMRIPrep `nogm`
  model instead of the default atlas-and-geometry rule.

  This step fixes a partial-volume artefact: a voxel that is half white
  matter and half CSF has grey-matter intensity, so the ventricle rims, the
  brain stem and the periventricular white matter come out of an
  intensity-driven segmentation labelled as grey matter. That grey matter is
  deleted and split evenly between white matter and CSF. By default T1Prep
  makes this decision from anatomy, following the strategy CAT12 uses in
  `cat_vol_partvol.m`: a Neuromorphometrics admission region where cortical
  grey matter cannot exist, and inside it a test for voxels whose
  neighbourhood holds both white matter and CSF but little grey matter. The
  DeepMRIPrep `nogm` model makes it with a UNet instead.

  Measured on one 0.5 mm subject the model takes 43.8 s and 5.6 GB against
  5.5 s and 2.0 GB for the default rule, and the two masks agree at Dice 0.74
  (13.7 vs 12.3 cm3 corrected). The flag has no effect together with `--amap`,
  which replaces the whole segmentation.

  This step cannot move the surfaces. Reassigning grey matter evenly to white
  matter and CSF leaves `p0 = csf + 2*gm + 3*wm` unchanged by construction, so
  the label map that drives surface reconstruction and cortical thickness is
  identical either way -- two full runs on the same subject produced
  bit-identical native `p0`. What does change is `p1`/`p2`/`p3`, and with them
  the modulated warped maps and the reported tissue volumes (0.05% on that
  subject, with TIV unchanged).

Deformation fields:
- `--save-h5`: additionally save the T1w↔MNI152NLin2009cAsym deformations as
  ANTs/ITK composite HDF5 files (`y_*.h5`, `iy_*.h5`), next to the NIfTI `y_`
  field. The composites are what `antsApplyTransforms` and `nitransforms`
  consume, and unlike `--fmriprep` this does not switch the run to fMRIPrep
  output mode. Requires the `nitransforms` package.

  The NIfTI `y_*.nii` written by default is an SPM12-compatible deformation
  (5-D `[X, Y, Z, 1, 3]`, native millimetres, affine and non-linear stages
  composed), so it can be passed straight to SPM's *Normalise: Write*.

Robustness:
- `--retry`: retry a failed processing step once. By default, if segmentation or surface
  estimation fails for a subject it is reported as an error straight away.

## Python API
You can also call the full pipeline from Python without shelling out manually:

```python
from t1prep import run_t1prep

# Single file, BIDS naming
run_t1prep("/data/sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz", bids=True)

# Multiple files with options and logging
run_t1prep([
  "/data/T1/sub-01.nii.gz",
  "/data/T1/sub-02.nii.gz",
], out_dir="/results", atlas=["neuromorphometrics", "suit"], multi=-1,
   wp=True, p=True, csf=True, lesions=True, gz=True, stream_output=True,
   log_file="/results/T1Prep_run.log")

# Use the DeepMRIPrep nogm model instead of the atlas-and-geometry rule
run_t1prep("/data/T1/sub-01.nii.gz", nogm_model=True)
```

## Output Folder Structure and Naming Conventions

T1Prep automatically determines output locations based on the input data structure:

1. **BIDS datasets**  
   If the input NIfTI is located in an `anat` folder:

`<dataset-root>/derivatives/T1Prep-v<version>/<sub-XXX>/<ses-YYY>/anat/`
   
- Subject (`sub-XXX`) and session (`ses-YYY`) are extracted from the path.
- If `--out-dir <DIR>` is specified, the BIDS substructure will still be created inside `<DIR>`.

2. **Non-BIDS datasets**  
Results are written to **CAT12-style subfolders** (`mri/`, `surf/`, etc.) in:
   
`<input-folder>/<subfolder>/`

or in `<DIR>` if `--out-dir <DIR>` is specified.

3. **Naming Conventions**  
- **Default (CAT12)**: Uses classic names like `mri/brainmask.nii` and `surf/lh.thickness`.
- **With `--bids`**: Uses BIDS derivatives naming, e.g.:
  ```
  sub-01_ses-1_space-T1w_desc-brain_mask.nii.gz
  sub-01_ses-1_hemi-L_thickness.shape.gii
  ```
- All filename mappings for both modes are defined in `Names.tsv` and can be customized.   
   

## Output folders structure
Output folder structure depends on the input dataset type:
* BIDS datasets (if the upper-level folder of the input files is 'anat'):
    Results are placed in a BIDS-compatible derivatives folder:
    inside &lt;DIR&gt;
    Subject ('sub-XXX') and session ('ses-YYY') are auto-detected.
* Non-BIDS datasets:
    Results are placed in subfolders similar to CAT12 output
    (e.g., 'mri/', 'surf/', 'report/', 'label') inside the specified 
    output directory.

If '--bids' is set, the BIDS derivatives substructure will always be used
inside &lt;DIR&gt;.

## Quality measures

The JSON report in `report/` carries a `qualitymeasures` block. Alongside the
Euler numbers (`euler_lh`, `euler_rh`, `EC_abs`) it reports glued sulci:

| Measure | Meaning |
|---------|---------|
| `glued_lh`, `glued_rh` | Percentage of central-surface vertices touching a facing patch of the same surface. Ideal 0; lower is better. |
| `glued_lh_sigma`, `glued_rh_sigma` | Present only when the surface was re-extracted at a reduced `sulci_sigma_factor` (see below). |

A glued (buried) sulcus is one whose two banks were never separated, so the
surface touches itself. This is *contact*, not a self-intersection — the
triangles need not cross — which is why it needs its own measure: surface
area barely moves even when the defect count changes many-fold.

Glued sulci originate upstream, in the distance map, so they are already
present in the raw marching-cubes output. When the measure exceeds its
threshold, `CAT_VolMarchingCubes` is re-run with a lower `sulci_sigma_factor`
and the least-glued result is kept; the value used is recorded in the report.
The parameter responds as a step rather than a slope (measured: glued
vertices roughly triple between 0.60 and 0.75), and lowering it does not help
every hemisphere, which is why the choice is made per hemisphere from the
measurement rather than by changing the default.

## Naming behaviour
* CAT12 style (default): Uses legacy folder and file names
  (e.g., 'mri/mwp1sub-01.nii', 'surf/lh.thickness.sub-01').
* BIDS style: Uses standardized derivatives names, including 
  subject/session identifiers, modality, and processing steps.

The complete mapping between internal outputs and both naming conventions
is stored in 'Names.tsv' and can be customized.

Examples:
Input: /data/study/sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz
Default output (no --out-dir):
    /data/study/derivatives/T1Prep-v${version}/sub-01/ses-1/anat/
With --out-dir /results:
    /results/derivatives/T1Prep-v${version}/sub-01/ses-1/anat/

Input: /data/T1_images/subject01.nii.gz
Default output (no --out-dir):
    /data/T1_images/mri/
With --out-dir /results:
    /results/mri/

## Examples
```bash
  T1Prep --out-dir test_folder sTRIO*.nii
```
Process all files matching the pattern 'sTRIO*.nii'. Generate segmentation 
and surface maps, saving the results in the 'test_folder' directory.

```bash
  T1Prep --no-surf sTRIO*.nii
```
Process all files matching the pattern 'sTRIO*.nii', but skip surface 
creation. Only segmentation maps are generated and saved in the same 
directory as the input files.

```bash
  T1Prep --python python3.11 --no-overwrite "surf/lh.thickness." sTRIO*.nii
```
Process all files matching the pattern `'sTRIO*.nii'` and use python3.11. 
Skip processing for files where 'surf/lh.thickness.*' already exists, and 
save new results in the same directory as the input files.

```bash
  T1Prep --lesion --no-sphere sTRIO*.nii
```
Process all files matching the pattern `'sTRIO*.nii'`. Skip processing of 
spherical registration, but additionally save lesion map (named p7sTRIO*.nii) 
in native space.

```bash
  T1Prep --amap sTRIO*.nii
```
Process all files matching the pattern `'sTRIO*.nii'` and enable AMAP segmentation.

```bash
  T1Prep --nogm-model sTRIO*.nii
```
Process all files matching the pattern `'sTRIO*.nii'`, removing non-cortical
grey matter with the DeepMRIPrep `nogm` model instead of the default
atlas-and-geometry rule.
  
```bash
  T1Prep --multi 8 --p --csf sTRIO*.nii
```

```bash
  T1Prep --skullstrip-only --out-dir test_folder sTRIO*.nii
```
Only run skull-stripping and write the skull-stripped image and brain mask.

```bash
  T1Prep --skip-skullstrip --out-dir test_folder sTRIO*_brain.nii
```
Skip skull-stripping for already skull-stripped inputs.
Process all files matching the pattern 'sTRIO*.nii'. Additionally save 
segmentations in native space, including CSF segmentation. The processing 
pipeline involves two stages of parallelization:

1. Segmentation (Python-based): Runs best with about 10-16 GB of memory per 
   process. The number of processes is automatically estimated based on 
   available memory to optimize resource usage.

2. Surface Extraction: This stage does not require significant memory and is
   fully distributed across all available processorsor limited to the 
   defined number of processes using the "--multi" flag.

If "--multi" is set to a specific number (e.g., 8), the system still 
estimates memory-based constraints for segmentation parallelization. However,
the specified number of processes (e.g., 8) will be used for surface 
extraction, ensuring efficient parallelization across the two stages. The 
default setting is -1, which automatically estimates the number of
available processors.

## Longitudinal realignment (experimental)

For rigid realignment of a series of NIfTI volumes, use the realignment helper:

```bash
./scripts/realign_longitudinal.sh --help
```

New tuning flags in the Python realigner:
- `--max-fwhm-mm <FLOAT>`: maximum smoothing (FWHM, mm) for coarse alignment.
- `--no-intensity-scale`: disable SPM-like global intensity scaling.
- `--overlap-penalty-weight <FLOAT>`: penalize samples that fall outside the moving FOV.
- `--sample-strategy {grid,gradient}`: choose deterministic grid or edge-biased gradient sampling.
- `--grad-quantile <FLOAT>`: threshold for selecting high-gradient samples.

### Longitudinal models

`scripts/process_longitudinal.sh --long-model` selects how much of the
between-scan difference is modelled, following CAT12's naming:

- `plasticity` (default): rigid realignment only. The anatomy itself is assumed
  unchanged between scans, which is appropriate for short intervals.
- `ageing`: rigid realignment, then one small low-dimensional diffeomorphic
  deformation per time point towards an unbiased subject average.
- `both`: saves both models, as CAT12's "detect both models" option does —
  `mwmwp1r<name>` from the ageing model and `mwp1r<name>` from the plasticity
  one, using CAT12's own names. The `r` marks the realigned input CAT12
  processes, and borrowing it keeps both distinct from T1Prep's cross-sectional
  `mwp1<name>` in the same folder — which is a different map, normalised with
  each time point's *own* warp rather than the shared one.

The ageing model is a stationary velocity field stored on a coarse control
lattice (12 mm by default) and integrated by scaling and squaring, re-centred
on its mean across time points so no scan is the reference -- the non-linear
analogue of the SE(3) barycentre the rigid stage already uses. It follows
Ashburner & Ridgway (2013), which is what SPM's serial longitudinal
registration and CAT12's ageing model use; over the small deformations between
serial scans a velocity field and a true geodesic agree to high order, so
geodesic shooting would add cost without adding accuracy here.

Per time point it writes `<stem>_desc-longLogJacobian.nii[.gz]`: the per-voxel
log volume ratio against the subject average, which is the map longitudinal VBM
runs statistics on. Note that the membrane prior shrinks the estimate towards
zero, so the map is better read as a spatial pattern than as a calibrated
absolute rate.

The deformations are estimated but **not** applied to what T1Prep then
processes: warping a time point onto the average would make its segmentation
and surfaces describe the average anatomy rather than that time point's own.
Pass `--warp-arg --apply` to also write the warped volumes.

Run the step on its own with:

```bash
./scripts/warp_longitudinal.sh --help
```

At the 1.5 mm default working resolution this costs a few seconds per subject
(about 4.5 s for two time points, 14 s for five, on CPU); `--resolution 1.0`
raises that to roughly three minutes for a pair.

The pipeline passes `--use-skullstrip`, so the deformations are estimated from
brain-extracted copies and applied to the originals, as the rigid stage already
does. It matters: on a real ADNI pair, estimating from full heads let scalp,
skull and neck drive 44 % of the volume instead of 13 %, left a 4.3x larger
residual, and moved the resulting log-Jacobian enough that the two estimates
correlate only r = 0.87 inside the brain.

One gap against SPM's serial longitudinal registration remains: it estimates a
bias field jointly with the deformation, and this does not. A strong intensity
non-uniformity difference between scans is therefore not modelled, and the
robust intensity normalisation is all that stands against it.

#### Jacobian modulation

After every time point has been processed, `--long-model ageing` (and `both`)
modulates each tissue map by its longitudinal Jacobian and carries it to MNI --
the map a longitudinal VBM analysis is run on. Names are CAT12's own:
`mwmwp1r<name>.nii` for the ageing model and `mwp1r<name>.nii` for plasticity in
the legacy scheme (in BIDS, `_desc-long` with `-modulated` doubled for ageing).
The doubled marker records that the ageing maps really are modulated twice --
by the longitudinal Jacobian, then by the spatial normalisation -- and the `r`
is CAT12's marker for the realigned input, which keeps both distinct from the
*cross-sectional* `mwp1<name>.nii` T1Prep writes into the same folder.

Following CAT12's ageing model, each time point keeps its **own segmentation**
and has its longitudinal Jacobian applied on top, so individual anatomy stays
visible. What is shared across the series is the **spatial normalisation** — the
mean of the per-time-point `y_` fields, standing in for the average image's warp
to MNI, since the average is not segmented here. That sharing matters on its
own: on an ADNI pair the two independently estimated cross-sectional warps
differed by a median of 1.08 mm inside the brain, more than the atrophy between
the scans.

Validated against CAT12's own `mwmwp1r` output on that pair: the default
reproduces CAT12's GM change to 0.02 percentage points (−8.74 % vs −8.72 %) and
correlates with CAT12's difference map at r = 0.42.

`--modulate-arg --tissue-source --modulate-arg shared` instead reuses one tissue
map for every time point, so the contrast carries *only* the Jacobian. It is far
quieter — on a synthetic series a 0.7 % segmentation perturbation corrupted the
contrast by 3.3 % with per-time-point maps against 0.1 % with the shared one —
but it is a different estimator from CAT12's, the two time points then differ
only by a smooth multiplier so individual anatomy is no longer visible, and on
the ADNI pair it gave a tenth of CAT12's amplitude with no spatial agreement
(r = −0.02).

Modulation is volume preserving: the integral of a modulated map reproduces that
time point's native tissue volume. `--modulate-arg --modulation --modulate-arg
nonlinear` divides the affine out to control for head size, as CAT12's
non-linear-only modulation does.

`--long-model ageing` and `both` add `--p` to the T1Prep calls automatically,
since the modulation needs the native segmentations and they are off by default.

**Input layout.** The time points must sit in a plain folder. A BIDS `anat/`
layout is not yet supported by the longitudinal pipeline and is refused up
front: the realigned copies would land on top of the inputs, and T1Prep would be
handed a directory as `--long-data`.

Run the stage on its own with:

```bash
./scripts/modulate_longitudinal.sh --help
```

## Input
T1-weighted MRI images in NIfTI format (extension nii/nii.gz).
