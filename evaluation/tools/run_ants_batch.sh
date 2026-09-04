#!/usr/bin/env bash
# antsRegistration with fMRIPrep's own parameters, then warp the manual labels.
SP=/private/tmp/claude-501/-Users-gaser-GitHub-T1Prep/d60c79ac-f20a-41db-974b-3e197cf0a946/scratchpad
MB=/Users/gaser/Downloads/mindboggle-101/data
export ANTSPATH=/Users/gaser/ants/ants-2.6.5/bin
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
run_one() {
  s="$1"
  out="$SP/antsbin_fmriprep/$s"; mkdir -p "$out"
  [ -f "$out/labels.DKT31.manual.MNI152.nii.gz" ] && { echo "$s cached"; return; }
  t0=$(date +%s)
  "$SP/ants_fmriprep.sh" "$SP/mni_fixed.nii.gz" "$SP/moving/$s.nii.gz" "$out/x" >/dev/null 2>&1 \
    || { echo "$s FAILED"; return; }
  "$ANTSPATH/antsApplyTransforms" -d 3 \
    -i "$MB/$s/labels.DKT31.manual.nii.gz" -r "$SP/mni_fixed.nii.gz" \
    -o "$out/labels.DKT31.manual.MNI152.nii.gz" \
    -n NearestNeighbor -t "$out/xComposite.h5" >/dev/null 2>&1 \
    || { echo "$s APPLY FAILED"; return; }
  rm -f "$out/xComposite.h5" "$out/xInverseComposite.h5" "$out/xWarped.nii.gz"
  echo "$s $(( $(date +%s) - t0 ))s"
}
export -f run_one; export SP MB ANTSPATH
xargs -P 4 -I{} bash -c 'run_one {}' < "$SP/subset20.txt"
echo BATCH_DONE
