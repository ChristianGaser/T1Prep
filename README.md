[![Python 3.9 | 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?logo=apache&logoColor=white)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ChristianGaser/T1Prep?display_name=tag&include_prereleases)](https://github.com/ChristianGaser/T1Prep/releases)
<!--
[![Tag](https://img.shields.io/github/v/tag/ChristianGaser/T1Prep?sort=semver)](https://github.com/ChristianGaser/T1Prep/tags)
-->
> [!WARNING]
> This project is **still in development** and might contain bugs. **If you experience any issues, please [let me know](https://github.com/ChristianGaser/T1Prep/issues)!**

<img src="T1Prep.png" alt="T1Prep logo" width="340"> 

# T1Prep: T1 PREProcessing Pipeline (aka PyCAT) 

T1Prep is a pipeline that preprocesses T1-weighted MRI data and supports segmentation and cortical surface reconstruction. It provides a complete set of tools for efficiently processing structural MRI scans.

T1Prep partially integrates [DeepMriPrep](https://github.com/wwu-mmll/deepmriprep), which uses deep learning (DL) techniques to mimic CAT12's functionality for processing structural MRIs. For details, see:
Lukas Fisch et al., "deepmriprep: Voxel-based Morphometry (VBM) Preprocessing via Deep Neural Networks," available on arXiv at https://doi.org/10.48550/arXiv.2408.10656.

An alternative approach uses DeepMriPrep for bias field correction, lesion detection, and also serves as an initial estimate for the subsequent AMAP segmentation from CAT12. 

Cortical surface reconstruction and thickness estimation are performed using [Cortex Analysis Tools for Surface](https://github.com/ChristianGaser/CAT-Surface), a core component of the [CAT12 toolbox](https://github.com/ChristianGaser/cat12).

It is designed for both single-subject and batch processing, with optional parallelization and flexible output naming conventions. The naming patterns are compatible with both 
CAT12 folder structures and the BIDS derivatives standard.

## Requirements
 [Python 3.9-3.12](https://www.python.org/downloads/) is required, and all necessary libraries are automatically installed the first time T1Prep is run or is called with the flag "--install".

## Main Differences to CAT12
- Implemented entirely in Python and C, eliminating the need for a Matlab license.
- Newly developed pipeline to estimate cortical surface and thickness.
- Skull-stripping, segmentation and non-linear spatial registration uses DeepMriPrep
- Does not yet support longitudinal pipelines.
- No quality assessment implemented yet.
- Only T1 MRI data supported.

## Installation

### Quick Install (Recommended)
Install T1Prep directly with a single command:
```bash
curl -fsSL https://raw.githubusercontent.com/ChristianGaser/T1Prep/refs/heads/main/scripts/install.sh | bash
```

The installer will interactively prompt you to:
1. **Select a version**: Latest release, development (main branch), or choose from available releases
2. **Choose installation directory**: Current folder, temporary folder, or custom path

#### Non-Interactive Installation
Use environment variables to skip the interactive prompts:
```bash
# Install latest release to current directory
T1PREP_VERSION=latest T1PREP_INSTALL_DIR="$PWD/T1Prep" \
  curl -fsSL https://raw.githubusercontent.com/ChristianGaser/T1Prep/refs/heads/main/scripts/install.sh | bash

# Install specific version to custom directory
T1PREP_VERSION=v1.0.0 T1PREP_INSTALL_DIR=/opt/T1Prep \
  curl -fsSL https://raw.githubusercontent.com/ChristianGaser/T1Prep/refs/heads/main/scripts/install.sh | bash
```

| Environment Variable | Description |
|---------------------|-------------|
| `T1PREP_VERSION` | Release tag (e.g., `v1.0.0`) or `latest` |
| `T1PREP_INSTALL_DIR` | Absolute path for installation |

### Manual Installation
Download T1Prep_$version.zip from Github and unzip:
```bash
  unzip T1Prep_$version.zip -d your_installation_folder
```
Install required Python packages (check that the correct Python version 3.9-3.12
is being used):
```bash
./scripts/T1Prep --python python3.12 --install
```
Alternatively, install the dependencies manually:
```bash
python3.12 -m pip install -r requirements.txt
```

### Docker

A Dockerfile is provided to build an image with all required dependencies.

#### Build

**Default (release ZIP):**
```bash
docker build -t t1prep:latest .
```

**Latest GitHub source (e.g., main):**

```bash
docker build \
  --build-arg T1PREP_SOURCE=git \
  --build-arg T1PREP_REF=main \
  -t t1prep:git-main .
```

**Specific release:**

```bash
docker build \
  --build-arg T1PREP_VERSION=v0.2.0-beta \
  -t t1prep:release .
```

#### Run

Mount your data directory into the container (replace /path/to/data with your folder):

```bash
docker run --rm -it \
  -v /path/to/data:/data \
  t1prep:latest \
  --out-dir /data/out /data/file.nii.gz
```
Append `--gpus all` to `docker run` to enable GPU acceleration when available.

#### Memory & performance

Make sure that the container has at least 16-24 GB of RAM available. If you are using Docker Desktop/WSL2, increase the VM memory in the settings. If you receive an error message stating that there is no space left on the device: /tmp/, you can try the following:
If you obtain an error that no space is left on device: /tmp/ you can try that:
```bash
docker run --rm -it \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=16g \
  -v /path/to/data:/data \
  t1prep:latest \
  --out-dir /data/out /data/file.nii.gz
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
   
## Usage
```bash
./scripts/T1Prep [options] file1.nii.[.gz] file2.nii[.gz] ...
```

### Python API
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

## Options
Simply call T1Prep to see available options
```bash
./scripts/T1Prep
```

Skull-stripping modes:
- `--skullstrip-only`: run skull-stripping only and exit after writing a skull-stripped image and brain mask.
- `--no-skullstrip` / `--skip-skullstrip`: skip skull-stripping (assumes input is already skull-stripped).

Longitudinal / advanced flags:
- `--initial-surf <FILE>`: use an initial surface estimate for longitudinal processing.
- `--long-data <PATH>`: process the volume at `<PATH>` while keeping output naming/folders based on the provided input file.
- `--no-atlas`: disable atlas labeling (overrides any defaults file atlas selection).

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
  ./scripts/T1Prep --out-dir test_folder sTRIO*.nii
```
Process all files matching the pattern 'sTRIO*.nii'. Generate segmentation 
and surface maps, saving the results in the 'test_folder' directory.

```bash
  ./scripts/T1Prep --no-surf sTRIO*.nii
```
Process all files matching the pattern 'sTRIO*.nii', but skip surface 
creation. Only segmentation maps are generated and saved in the same 
directory as the input files.

```bash
  ./scripts/T1Prep --python python3.9 --no-overwrite "surf/lh.thickness." sTRIO*.nii
```
Process all files matching the pattern `'sTRIO*.nii'` and use python3.9. 
Skip processing for files where 'surf/lh.thickness.*' already exists, and 
save new results in the same directory as the input files.

```bash
  ./scripts/T1Prep --lesion --no-sphere sTRIO*.nii
```
Process all files matching the pattern `'sTRIO*.nii'`. Skip processing of 
spherical registration, but additionally save lesion map (named p7sTRIO*.nii) 
in native space.

```bash
  ./scripts/T1Prep --amap sTRIO*.nii
```
Process all files matching the pattern `'sTRIO*.nii'` and enable AMAP segmentation.
  
```bash
  ./scripts/T1Prep --multi 8 --p --csf sTRIO*.nii
```

```bash
  ./scripts/T1Prep --skullstrip-only --out-dir test_folder sTRIO*.nii
```
Only run skull-stripping and write the skull-stripped image and brain mask.

```bash
  ./scripts/T1Prep --skip-skullstrip --out-dir test_folder sTRIO*_brain.nii
```
Skip skull-stripping for already skull-stripped inputs.
Process all files matching the pattern 'sTRIO*.nii'. Additionally save 
segmentations in native space, including CSF segmentation. The processing 
pipeline involves two stages of parallelization:

1. Segmentation (Python-based): Runs best with about 16-24GB of memory per 
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

## Support
For issues and inquiries, contact [me](mailto:christian.gaser@uni-jena.de).

## License
T1Prep is distributed under the terms of the [Apache License](https://www.apache.org/licenses/LICENSE-2.0) 
as published by the Apache Software Foundation.

