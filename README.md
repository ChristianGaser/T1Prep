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

As with other DL-based methods, DeepMriPrep slightly underestimates gray matter in cases of significant atrophy. Therefore, it is primarily used for bias field correction, lesion detection, and as an initial estimate for the subsequent AMAP segmentation from CAT12. The skull-stripping and nonlinear spatial registration steps provided by DeepMriPrep are unaffected by this bias and are fully utilized in T1Prep.

Cortical surface reconstruction and thickness estimation are performed using [Cortex Analysis Tools for Surface](https://github.com/ChristianGaser/CAT-Surface), a core component of the [CAT12 toolbox](https://github.com/ChristianGaser/cat12).

## Requirements
Python 3.8 (or higher) is required, and all necessary libraries are automatically installed the first time T1Prep is run.

## Main Differences to CAT12
- Implemented entirely in Python and C, eliminating the need for a Matlab license.
- New pipeline to estimate cortical surface and thickness.
- Does not yet support longitudinal pipelines.
- No quality assessment implemented yet.
- Only T1 MRI data supported.

## Usage
```bash
./scripts/T1Prep [options] file1.nii file2.nii ...
```

## Options
**General Options** ||
:-------- | --------
`--re-install` |Remove the old installation and reinstall the required Python libraries.
`--python <FILE>` |Specify the Python interpreter to use (default: $python).
`--multi <NUMBER>` |Specify the number of processes for parallelization. Use '-1' to automatically estimate the number of available processors 
`--debug` | Enable verbose output, retain temporary files, and save additional debugging information.
**Save Options** ||
`--out-dir <DIR>` |Specify the output directory (default: current working directory).
`--no-overwrite <STRING>` |Prevent overwriting existing results by checking for the specified filename pattern.
`--no-surf` |Skip surface and thickness estimation.
`--no-seg` |Skip segmentation processing.
`--no-sphere` |Skip spherical surface registration.
`--no-mwp` |Skip the estimation of modulated and warped segmentations.
`--hemisphere` |Additionally save hemispheric partitions of the segmentation.
`--pial-white` |Additionally extract the pial and white surface.
`--wp` |Additionally save warped segmentations.
`--rp` |Additionally save affine-registered segmentations.
`--p` |Additionally save native space segmentations.
`--csf` |Additionally save CSF segmentations (default: only GM/WM are saved).
`--lesions` |Additionally save WMH lesions.
`--bids` |Use BIDS (Brain Imaging Data Structure) standard for output file naming conventions.
**Expert Options** ||
`--amap` | Use AMAP segmentation instead of DeepMRIPrep.
`--pre-fwhm <NUMBER>` |Specify the pre-smoothing FWHM size in CAT_VolMarchingCubes 
`--post-fwhm <NUMBER>` |Specify the post-smoothing FWHM size in CAT_VolMarchingCubes 
`--thickness-fwhm <NUMBER>` |Specify the FWHM size for volumetric thickness smoothing in CAT_VolThicknessPbt
`--sharpening <NUMBER>` |Specify the amount of sharpening applied to the PPM map by adding the difference between the unsmoothed and smoothed PPM map 
`--thresh <NUMBER>` |Specify the isovalue threshold for surface creation in CAT_VolMarchingCubes
`--vessel <NUMBER>` |Set the initial white matter (WM) threshold for vessel removal. Use 0.2 for mild cleanup, 0.5 for strong cleanup, or 0 to disable vessel removal.
`--median-filter <NUMBER>` |Specify the number of median filter applications to reduce topology artifacts.

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
  ./scripts/T1Prep --python python3.8 --no-overwrite "surf/lh.thickness." sTRIO*.nii
```
    Process all files matching the pattern 'sTRIO*.nii' and use python3.8. Skip processing 
    for files where 'surf/lh.thickness.*' already exists, and save new results in the same 
    directory as the input files.

```bash
  ./scripts/T1Prep --pial-white --lesion --no-sphere sTRIO*.nii
```
    Process all files matching the pattern 'sTRIO*.nii'. Skip processing of spherical
    registration, but additionally save lesion map (named p7sTRIO*.nii) and pial
    and white surface.
  
```bash
  ./scripts/T1Prep --multi -1 --p --csf sTRIO*.nii
```
    Process all files matching the pattern 'sTRIO*.nii'. Additionally save segmentations 
    in native space, including CSF segmentation. The processing pipeline involves two stages 
    of parallelization:
    
    1. **Segmentation (Python-based)**: Runs best with about 24GB of memory per process. 
       The number of processes is automatically estimated based on available memory to 
       optimize resource usage.
  
    2. **Surface Extraction**: This stage does not require significant memory and is fully 
       distributed across all available processors.
  
    If "--multi" is set to a specific number (e.g., 8), the system still estimates memory-based 
    constraints for segmentation parallelization. However, the specified number of processes 
    (e.g., 8) will be used for surface extraction, ensuring efficient parallelization across 
    the two stages.

## Input
Files: T1-weighted MRI images in NIfTI format.

## Support
For issues and inquiries, contact christian.gaser@uni-jena.de.

## License
T1Prep is distributed under the terms of the [Apache License](https://www.apache.org/licenses/LICENSE-2.0) as published by the Apache Software Foundation.

