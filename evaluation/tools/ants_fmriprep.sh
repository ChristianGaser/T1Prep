#!/usr/bin/env bash
# antsRegistration invoked exactly as fMRIPrep/sMRIPrep does.  The stages are
# niworkflows/data/t1w-mni_registration_precise_000.json, which Nipype renders
# into this command line field for field; niworkflows' SpatialNormalization adds
# --float 1 and starts from antsAffineInitializer instead of the centres of mass.
#
#   ants_fmriprep.sh <fixed> <moving> <prefix>
#
# Both images come masked, as SpatialNormalization's explicit masking leaves
# them: fMRIPrep's fixed image is tpl-MNI152NLin2009cAsym_res-01_T1w (1 mm) set
# to 0 outside res-01_desc-brain_mask, the moving image the T1w set to 0 outside
# its brain mask.  sMRIPrep truncates the moving image before masking it
# (ImageMath 3 out.nii.gz TruncateImageIntensity in.nii.gz 0.01 0.999 256);
# that is left to the caller.
set -euo pipefail
FIXED="$1"; MOVING="$2"; PREFIX="$3"
"${ANTSPATH}/antsAffineInitializer" 3 "$FIXED" "$MOVING" "${PREFIX}init.mat" \
  15.000000 0.100000 0 10
"${ANTSPATH}/antsRegistration" \
  --dimensionality 3 \
  --float 1 \
  --collapse-output-transforms 1 \
  --interpolation LanczosWindowedSinc \
  --output "[${PREFIX},${PREFIX}Warped.nii.gz]" \
  --winsorize-image-intensities "[0.005,0.995]" \
  --use-histogram-matching 1 \
  --initial-moving-transform "[${PREFIX}init.mat,0]" \
  --transform "Rigid[0.05]" \
  --metric "Mattes[${FIXED},${MOVING},1,56,Regular,0.25]" \
  --convergence "[100x100,1e-06,20]" \
  --smoothing-sigmas 2.0x1.0vox \
  --shrink-factors 2x1 \
  --use-histogram-matching 1 \
  --transform "Affine[0.08]" \
  --metric "Mattes[${FIXED},${MOVING},1,56,Regular,0.25]" \
  --convergence "[100x100,1e-06,20]" \
  --smoothing-sigmas 1.0x0.0vox \
  --shrink-factors 2x1 \
  --use-histogram-matching 1 \
  --transform "SyN[0.1,3.0,0.0]" \
  --metric "CC[${FIXED},${MOVING},1,4,None,1.0]" \
  --convergence "[100x70x50x20,1e-06,10]" \
  --smoothing-sigmas 3.0x2.0x1.0x0.0vox \
  --shrink-factors 8x4x2x1 \
  --use-histogram-matching 1 \
  --write-composite-transform 1 \
  --verbose 0
