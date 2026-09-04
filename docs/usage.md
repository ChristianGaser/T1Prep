# Using T1Prep

How to run the pipeline, what it writes and what the options mean. For
installation see [installation.md](installation.md); for the viewers and the
other tools see [viewers.md](viewers.md) and [tools.md](tools.md).

- [Usage](#usage)
- [Options](#options)
- [Python API](#python-api)
- [Output folder structure and naming](#output-folder-structure-and-naming-conventions)
- [Examples](#examples)
- [Longitudinal realignment](#longitudinal-realignment-experimental)
- [Input](#input)

Back to the [README](../README.md).

---

## Usage
```bash
T1Prep [options] file1.nii.[.gz] file2.nii[.gz] ...
```

(`T1Prep` resolves from the environment's `bin/` once it is on your `PATH`; from
a source checkout without an install you can still run `./scripts/T1Prep`.)

## Options
Simply call T1Prep to see available options
```bash
T1Prep
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

## Input
T1-weighted MRI images in NIfTI format (extension nii/nii.gz).
