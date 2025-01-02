# T1Prep: T1 PREProcessing pipeline (aka DeepCAT)

T1Prep is a pipeline designed to preprocess T1-weighted MRI images for segmentation and cortical surface extraction. It provides a comprehensive suite of tools to handle and process MRI data efficiently.

The segmentation is based on [deepmriprep](https://github.com/wwu-mmll/deepmriprep), which mimics CAT12 using deep-learning based processing of MRI data: Lukas Fisch et al. deepmriprep: Voxel-based Morphometry (VBM) Preprocessing via Deep Neural Networks. arXiv. https://doi.org/10.48550/arXiv.2408.10656

The surface creation and thickness estimation is based on the [Cortex Analysis Tools for Surface](https://github.com/ChristianGaser/CAT-Surface). which are also part of the [CAT12 toolbox](https://github.com/ChristianGaser/cat12).

## Main differences to CAT12
- **Code is purely written in python and c and no Matlab license is necessary anymore
- **No support yet for longitduinal pipelines

## Features

- **Segmentation and Surface Extraction:** Automates the processing of MRI images to extract segmentation maps and cortical surfaces.
- **Multiple Input Formats:** Supports `.nii` and `.nii.gz` formats.

## Usage

./T1Prep.sh [options] file1.nii file2.nii ...

## Options
  --python <FILE>            python command
  --out-dir <DIR>            output folder (default same folder)
  --pre-fwhm  <NUMBER>       FWHM size of pre-smoothing in CAT_VolMarchingCubes. 
  --post-fwhm <NUMBER>       FWHM size of post-smoothing in CAT_VolMarchingCubes. 
  --thickness-fwhm <NUMBER>  FWHM size of volumetric thickness smoothing in CAT_VolThicknessPbt. 
  --thresh    <NUMBER>       threshold (isovalue) for creating surface in CAT_VolMarchingCubes. 
  --min-thickness <NUMBER>   values below minimum thickness are set to zero and will be approximated
                             using the replace option in the vbdist method. 
  --median-filter <NUMBER>   specify how many times to apply a median filter to areas with
                             topology artifacts to reduce these artifacts.
  --no-surf                  skip surface and thickness estimation
  --no-mwp                   skip estimation of modulated and warped segmentations
  --rp                       additionally estimate affine registered segmentations
  --sanlm                    apply denoising with SANLM-filter
  --bids                     use BIDS naming of output files
  --debug                    keep temporary files for debugging

## Example
```bash
./T1Prep.sh --out-dir output_folder --no-surf image1.nii

## Input
Files: T1-weighted MRI images in NIfTI format.

## Support
For issues and inquiries, contact christian.gaser@uni-jena.de.

## License
T1Prep is distributed under the terms of the [Apache License](https://www.apache.org/licenses/LICENSE-2.0) as published by the Apache Software Foundation.

