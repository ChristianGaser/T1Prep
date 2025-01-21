# T1Prep: T1 PREProcessing Pipeline (aka PyCAT)
T1Prep is a pipeline designed for the preprocessing of T1-weighted MRI images, facilitating segmentation and cortical surface extraction. This tool provides a comprehensive suite of utilities to efficiently manage and process MRI data.

The segmentation leverages [deepmriprep](https://github.com/wwu-mmll/deepmriprep), which employs deep-learning techniques to mimic the capabilities of CAT12 in processing MRI data. For more details, see: Lukas Fisch et al., "deepmriprep: Voxel-based Morphometry (VBM) Preprocessing via Deep Neural Networks," available on arXiv at https://doi.org/10.48550/arXiv.2408.10656.

Surface creation and thickness estimation are performed using the [Cortex Analysis Tools for Surface](https://github.com/ChristianGaser/CAT-Surface), also integral to the [CAT12 toolbox](https://github.com/ChristianGaser/cat12).

## Requirements
Python 3.8 (or higher) is required, and all necessary libraries are automatically installed the first time T1Prep is run.

## Main Differences to CAT12
- Implemented entirely in Python and C, eliminating the need for a Matlab license.
- New pipeline to estimate cortical surface and thickness.
- Does not currently support longitudinal pipelines.
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
**Processing Parameters** ||
`--out-dir <DIR>` |Specify the output directory (default: current working directory).
`--amap` | Use AMAP segmentation instead of DeepMRIPrep.
`--pre-fwhm <NUMBER>` |Specify the pre-smoothing FWHM size in CAT_VolMarchingCubes 
`--post-fwhm <NUMBER>` |Specify the post-smoothing FWHM size in CAT_VolMarchingCubes 
`--thickness-fwhm <NUMBER>` |Specify the FWHM size for volumetric thickness smoothing in CAT_VolThicknessPbt
`--sharpening <NUMBER>` |Specify the amount of sharpening applied to the PPM map by adding the difference between the unsmoothed and smoothed PPM map 
`--thresh <NUMBER>` |Specify the isovalue threshold for surface creation in CAT_VolMarchingCubes
`--downsample <NUMBER>` |Specify the downsampling factor for PPM and GMT maps to reduce surface intersections (default: $downsample).
`--min-thickness <NUMBER>` |Specify the minimum thickness value (values below this are set to zero) for the vbdist method (default: $min_thickness).
`--median-filter <NUMBER>` |Specify the number of median filter applications to reduce topology artifacts.
**Save Options** ||
`--no-overwrite <STRING>` |Prevent overwriting existing results by checking for the specified filename pattern.
`--no-surf` |Skip surface and thickness estimation.
`--no-seg` |Skip segmentation processing.
`--no-sphere` |Skip spherical surface registration.
`--no-mwp` |Skip the estimation of modulated and warped segmentations.
`--hemisphere` |Additionally save hemispheric partitions of the segmentation.
`--wp` |Additionally save warped segmentations.
`--rp` |Additionally save affine-registered segmentations.
`--p` |Additionally save native space segmentations.
`--csf` |Additionally save CSF segmentations (default: only GM/WM are saved).
`--bids` |Use BIDS (Brain Imaging Data Structure) standard for output file naming conventions.

## Examples
```bash
  ./scripts/T1Prep --out-dir test_folder sTRIO*.nii
```
    Process all files matching the pattern 'sTRIO*.nii' to extract segmentation 
    and surface maps, saving the results in the 'test_folder' directory.

```bash
  ./scripts/T1Prep --no-surf sTRIO*.nii
```
    Extract only the segmentation maps for all files matching the pattern 'sTRIO*.nii',
    skipping surface creation and saving the results in the same directory as the input files.

```bash
  ./scripts/T1Prep --no-overwrite "surf/lh.thickness." sTRIO*.nii
```
    Process all files matching the pattern 'sTRIO*.nii', skipping all files, where the files
    'surf/lh.thickness.*' already exist and save the results in the same directory as the
    input files.

```bash
  ./scripts/T1Prep --out-dir test_folder --no-surf --hemisphere sTRIO*.nii
  ./scripts/parallelize -p 8 -c "./scripts/T1prep --out-dir test_folder --no-seg" sTRIO*.nii
```
    1. Runs 'T1prep' on all files matching the pattern 'sTRIO*.nii', skipping surface creation 
       and saving results in 'test_folder'.
    2. Parallelizes the processing using 8 processors for improved performance. The first 
       command is not parallelized because it already uses multi-threading and is memory-
       intensive, while the second command (surface creation) is single-threaded and can 
       benefit from parallelization.
       
    NOTE: You can automatically enable parallelization by using the '--multi' flag.

## Input
Files: T1-weighted MRI images in NIfTI format.

## Support
For issues and inquiries, contact christian.gaser@uni-jena.de.

## License
T1Prep is distributed under the terms of the [Apache License](https://www.apache.org/licenses/LICENSE-2.0) as published by the Apache Software Foundation.

