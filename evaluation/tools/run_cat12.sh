#!/usr/bin/env bash
# CAT12 (geodesic shooting) on the 20-subject subset.
#
# CAT12 needs one uncompressed .nii per subject and writes mri/y_<name>.nii
# next to it, so each subject gets its own directory named after the subject
# (every Mindboggle file is called t1weighted.nii.gz and would otherwise
# collide).  -ns skips surface reconstruction, which the volume evaluation
# does not need; -p sets the number of parallel jobs.
set -euo pipefail
MB="${1:?usage: run_cat12.sh <mindboggle-data-dir> <work-dir> [subset.txt]}"
WORK="${2:?}"
SUBSET="${3:-$(dirname "$0")/../data/subset20.txt}"
CAT="${CAT_BATCH:-$HOME/spm/spm12/toolbox/CAT/cat_batch_cat.sh}"

mkdir -p "$WORK"
while read -r s; do
  [ -z "$s" ] && continue
  mkdir -p "$WORK/$s"
  python -c "
import nibabel as nib
img = nib.load('$MB/$s/t1weighted.nii.gz')
nib.save(nib.Nifti1Image(img.dataobj, img.affine, img.header), '$WORK/$s/$s.nii')"
done < "$SUBSET"

cd "$WORK"
"$CAT" -ns -p 4 */*.nii
