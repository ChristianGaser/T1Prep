<!--
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/T1Prep)
![PyPI - License](https://img.shields.io/pypi/l/T1Prep)
![PyPI - Version](https://img.shields.io/pypi/v/T1Prep)
-->

![Alt-Text](T1Prep.png)

> [!WARNING]
> This project is **currently under construction** and might contain bugs. **If you experience any issues, please [let me know](https://github.com/ChristianGaser/T1Prep/issues)!**

# T1Prep: T1 PREProcessing Pipeline (aka PyCAT)

T1Prep is a pipeline that preprocesses T1-weighted MRI data and supports segmentation and cortical surface reconstruction. It provides a complete set of tools for efficiently processing structural MRI scans.

T1Prep partially integrates [DeepMriPrep](https://github.com/wwu-mmll/deepmriprep), which uses deep learning (DL) techniques to mimic CAT12's functionality for processing structural MRIs. For details, see:
Lukas Fisch et al., "deepmriprep: Voxel-based Morphometry (VBM) Preprocessing via Deep Neural Networks," available on arXiv at https://doi.org/10.48550/arXiv.2408.10656.

An alternative approach uses DeepMriPrep for bias field correction, lesion detection, and also serves as an initial estimate for the subsequent AMAP segmentation from CAT12. 

Cortical surface reconstruction and thickness estimation are performed using [Cortex Analysis Tools for Surface](https://github.com/ChristianGaser/CAT-Surface), a core component of the [CAT12 toolbox](https://github.com/ChristianGaser/cat12).

It is designed for both single-subject and batch processing, with optional parallelization and flexible output naming conventions. The naming patterns are compatible with both 
CAT12 folder structures and the BIDS derivatives standard.

## Requirements
 Python 3.9 is required, and all necessary libraries are automatically installed the first time T1Prep is run or is called with the flag "--install".

## Main Differences to CAT12
- Implemented entirely in Python and C, eliminating the need for a Matlab license.
- New developed pipeline to estimate cortical surface and thickness.
- Does not yet support longitudinal pipelines.
- No quality assessment implemented yet.
- Only T1 MRI data supported.

## Output Folder Structure and Naming Conventions

T1Prep automatically determines output locations based on the input data structure:

1. **BIDS datasets**  
   If the input NIfTI is located in an `anat` folder:

<dataset-root>/derivatives/T1Prep-v<version>/<sub-XXX>/<ses-YYY>/anat/
   
- Subject (`sub-XXX`) and session (`ses-YYY`) are extracted from the path.
- If `--out-dir <DIR>` is specified, the BIDS substructure will still be created inside `<DIR>`.

2. **Non-BIDS datasets**  
Results are written to **CAT12-style subfolders** (`mri/`, `surf/`, etc.) in:
   
<input-folder>/<subfolder>/

or in `<DIR>` if `--out-dir <DIR>` is specified.

3. **Naming Conventions**  
- **Default (CAT12)**: Uses classic names like `mri/brainmask.nii` and `surf/lh.thickness`.
- **With `--bids`**: Uses BIDS derivatives naming, e.g.:
  ```
  sub-01_ses-1_space-T1w_desc-brain_mask.nii.gz
  sub-01_ses-1_hemi-L_thickness.shape.gii
  ```
- All filename mappings for both modes are defined in `Names.tsv` and can be customized.

**Examples:**

_BIDS input_

Input: /data/study/sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz
Default: /data/study/derivatives/T1Prep-v1.0/sub-01/ses-1/anat/
With --out-dir /results:
/results/derivatives/T1Prep-v1.0/sub-01/ses-1/anat/
   
   
## Usage
```bash
./scripts/T1Prep [options] file1.nii file2.nii ...
```

## Options
**General Options** ||
:-------- | --------
`--install` |Install the required Python libraries.
`--re-install` |Remove the existing installation and reinstall the required Python libraries.
`--python <FILE>` |Set the Python interpreter to use.
`--multi <NUMBER>` |Set the number of processes for parallelization. Use '-1' to automatically estimate the number of available processors 
`--debug` | Enable verbose output, retain temporary files, and save additional debugging information.
**Save Options** ||
`--out-dir <DIR>` |Set the relative output directory (default: current working directory).
`--no-overwrite <STRING>` |Avoid overwriting existing results by checking for the specified filename pattern.
`--gz' |Save images as nii.gz instead of nii.
`--no-surf` |Skip surface and thickness estimation.
`--no-seg` |Skip segmentation processing.
`--no-sphere-reg` |Skip spherical surface registration.
`--no-mwp` |Skip the estimation of modulated and warped segmentations.
`--hemisphere` |Additionally save hemispheric partitions of the segmentation.
`--wp` |Additionally save warped segmentations.
`--rp` |Additionally save affine-registered segmentations.
`--p` |Additionally save native space segmentations.
`--csf` |Additionally save CSF segmentations (default: only GM/WM are saved).
`--lesions` |Additionally save WMH lesions.
`--bids` |Use BIDS (Brain Imaging Data Structure) standard for output file naming conventions.
**Expert Options** ||
`--no-amap` | Use DeepMRIPrep instead of AMAP for segmentation.
`--thickness-method` <NUMBER> |Set the thickness method (default: $thickness_method). Use 1 for PBT-based method and 2 for approach based on distance between pial and white matter surface.
`--no-correct-folding` |Do not correct for cortical thickness by folding effects.
`--pre-fwhm <NUMBER>` |Set the pre-smoothing FWHM size in CAT_VolMarchingCubes 
`--post-fwhm <NUMBER>` |Set the post-smoothing FWHM size in CAT_VolMarchingCubes 
`--thickness-fwhm <NUMBER>` |Set the FWHM size for volumetric thickness smoothing in CAT_VolThicknessPbt
`--sharpening <NUMBER>` |Set the sharpening level applied to the PPM map by enhancing differences between the unsmoothed and smoothed PPM maps 
`--thresh <NUMBER>` |Set the isovalue threshold for surface creation in CAT_VolMarchingCubes
`--vessel <NUMBER>` |Set the initial white matter (WM) threshold for vessel removal. Use 0.2 for mild cleanup, 0.5 for strong cleanup, or 0 to disable vessel removal.
`--median-filter <NUMBER>` |Set the number of median filter applications to reduce topology artifacts.

## Examples
```bash
  ./scripts/T1Prep --out-dir test_folder sTRIO*.nii
```
    Process all files matching the pattern 'sTRIO*.nii'. Generate segmentation and 
    surface maps, saving the results in the 'test_folder' directory.

```bash
  ./scripts/T1Prep --no-surf sTRIO*.nii
```
    Process all files matching the pattern 'sTRIO*.nii', but skip surface creation. 
    Only segmentation maps are generated and saved in the same directory as the input files.

```bash
  ./scripts/T1Prep --python python3.9 --no-overwrite "surf/lh.thickness." sTRIO*.nii
```
    Process all files matching the pattern 'sTRIO*.nii' and use python3.9. Skip processing
    for files where 'surf/lh.thickness.*' already exists, and save new results in the same
    directory as the input files.

```bash
  ./scripts/T1Prep --lesion --no-sphere sTRIO*.nii
```
   Process all files matching the pattern 'sTRIO*.nii'. Skip processing of spherical
   registration, but additionally save lesion map (named p7sTRIO*.nii) in native space.

```bash
  ./scripts/T1Prep --no-amap sTRIO*.nii
```
   Process all files matching the pattern 'sTRIO*.nii' and use DeppMriPrep instead of AMAP
   segmentation.

  
```bash
  ./scripts/T1Prep --multi 8 --p --csf sTRIO*.nii
```
    Process all files matching the pattern 'sTRIO*.nii'. Additionally save segmentations 
    in native space, including CSF segmentation. The processing pipeline involves two stages 
    of parallelization:
    
    1. Segmentation (Python-based): Runs best with about 24GB of memory per process. 
       The number of processes is automatically estimated based on available memory to 
       optimize resource usage.
  
    2. Surface Extraction: This stage does not require significant memory and is fully 
       distributed across all available processors.
  
    If "--multi" is set to a specific number (e.g., 8), the system still estimates memory-based 
    constraints for segmentation parallelization. However, the specified number of processes 
    (e.g., 8) will be used for surface extraction, ensuring efficient parallelization across 
    the two stages. The default setting is -1, which automatically estimates the number of
    available processors.


## Input
Files: T1-weighted MRI images in NIfTI format.

## Installation
Download T1Prep_$version.zip from Github and unzip:
```bash
  unzip T1Prep_$version.zip -d your_installation_folder
```
Download T1Prep_Models.zip from Github and unzip:
```bash
  unzip T1Prep_Models.zip -d your_installation_folder
```
Install required Python packages:
```bash
./scripts/T1Prep --install
```
Alternatively, install the dependencies manually:
```bash
pip install -r requirements.txt
```

## Support
For issues and inquiries, contact christian.gaser@uni-jena.de.

## License
T1Prep is distributed under the terms of the [Apache License](https://www.apache.org/licenses/LICENSE-2.0) as published by the Apache Software Foundation.

