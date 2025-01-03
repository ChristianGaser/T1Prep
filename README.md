# T1Prep: T1 PREProcessing Pipeline (aka DeepCAT)
T1Prep is a pipeline designed for the preprocessing of T1-weighted MRI images, facilitating segmentation and cortical surface extraction. This tool provides a comprehensive suite of utilities to efficiently manage and process MRI data.

The segmentation leverages [deepmriprep](https://github.com/wwu-mmll/deepmriprep), which employs deep-learning techniques to mimic the capabilities of CAT12 in processing MRI data. For more details, see: Lukas Fisch et al., "deepmriprep: Voxel-based Morphometry (VBM) Preprocessing via Deep Neural Networks," available on arXiv at https://doi.org/10.48550/arXiv.2408.10656.

Surface creation and thickness estimation are performed using the [Cortex Analysis Tools for Surface](https://github.com/ChristianGaser/CAT-Surface), also integral to the [CAT12 toolbox](https://github.com/ChristianGaser/cat12).

## Options
Python 3.8 (or higher) is required, and all necessary libraries are automatically installed the first time T1Prep is run.

## Main Differences to CAT12
- Implemented entirely in Python and C, eliminating the need for a Matlab license.
- Does not currently support longitudinal pipelines.
- No quality assessment implemented yet.
- Only T1 MRI data supported.

## Usage
```bash
./scripts/T1Prep.sh [options] file1.nii file2.nii ...
```

## Options
- `--python <FILE>`: Specify the Python command.
- `--out-dir <DIR>`: Specify the output directory (default is the same directory as the input files).
- `--pre-fwhm <NUMBER>`: Set the FWHM size for pre-smoothing in CAT_VolMarchingCubes.
- `--post-fwhm <NUMBER>`: Set the FWHM size for post-smoothing in CAT_VolMarchingCubes.
- `--thickness-fwhm <NUMBER>`: Set the FWHM size for volumetric thickness smoothing in CAT_VolThicknessPbt.
- `--thresh <NUMBER>`: Set the threshold (isovalue) for surface generation in CAT_VolMarchingCubes.
- `--min-thickness <NUMBER>`: Values below the minimum thickness are set to zero and approximated using the replace option in the vbdist method.
- `--median-filter <NUMBER>`: Specify the number of times a median filter should be applied to regions with topology artifacts to reduce these artifacts.
- `--no-surf`: Skip surface and thickness estimation.
- `--no-mwp`: Skip modulated and warped segmentation estimation.
- `--rp`: Additionally estimate affine registered segmentations.
- `--sanlm`: Apply denoising with the SANLM filter.
- `--amap`: Use segmentation from AMAP instead of deepmriprep
- `--bids`: Use BIDS naming for output files.
- `--debug`: Keep temporary files for debugging.

## Example
```bash
./scripts/T1Prep.sh --out-dir output_folder --no-surf image1.nii
```

## Input
Files: T1-weighted MRI images in NIfTI format.

## Support
For issues and inquiries, contact christian.gaser@uni-jena.de.

## License
T1Prep is distributed under the terms of the [Apache License](https://www.apache.org/licenses/LICENSE-2.0) as published by the Apache Software Foundation.

