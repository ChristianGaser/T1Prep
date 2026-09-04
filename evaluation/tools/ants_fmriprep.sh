#!/usr/bin/env bash
# antsRegistration invoked exactly as fMRIPrep/sMRIPrep does, from
# niworkflows/data/t1w-mni_registration_precise_000.json.  Nipype renders that
# JSON into this command line; the values below are copied field for field.
set -euo pipefail
FIXED="$1"; MOVING="$2"; PREFIX="$3"
"${ANTSPATH}/antsRegistration" \
  --dimensionality 3 \
  --float 0 \
  --collapse-output-transforms 1 \
  --interpolation LanczosWindowedSinc \
  --output "[${PREFIX},${PREFIX}Warped.nii.gz]" \
  --winsorize-image-intensities "[0.005,0.995]" \
  --use-histogram-matching 1 \
  --initial-moving-transform "[${FIXED},${MOVING},1]" \
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
