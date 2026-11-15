#!/usr/bin/env bash
# antsRegistration with fMRIPrep's own parameters, then warp the manual labels.
#
#   SP=<work> MB=<mindboggle-101/data> ANTSPATH=<ants/bin> run_ants_batch.sh
#
# SP  work folder: mni_fixed.nii.gz (the fixed image), moving/<subject>.nii.gz
#     (run_ants_antspy.py writes them) and, optionally, subset20.txt; the
#     results go to SP/antsbin_fmriprep/<subject>
# MB  the Mindboggle-101 data folder (<subject>/labels.DKT31.manual.nii.gz)
# ANTSPATH  the bin folder of ANTs
# SUBJECTS  list of subjects, one per line (default: SP/subset20.txt, else
#     evaluation/data/subset20.txt)
#
# (The three paths used to be set here to folders of the machine the
# evaluation was run on, one of them a scratch folder that is long gone.)
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SP:?set SP to the work folder (mni_fixed.nii.gz, moving/)}"
: "${MB:?set MB to the Mindboggle-101 data folder}"
: "${ANTSPATH:?set ANTSPATH to the bin folder of ANTs}"
if [ -z "${SUBJECTS:-}" ]; then
  SUBJECTS="$SP/subset20.txt"
  [ -f "$SUBJECTS" ] || SUBJECTS="$TOOLS/../data/subset20.txt"
fi
for needed in "$SP/mni_fixed.nii.gz" "$SUBJECTS" "$ANTSPATH/antsRegistration"; do
  [ -e "$needed" ] || { echo "ERROR: $needed not found" >&2; exit 1; }
done
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-2}"
run_one() {
  s="$1"
  out="$SP/antsbin_fmriprep/$s"; mkdir -p "$out"
  [ -f "$out/labels.DKT31.manual.MNI152.nii.gz" ] && { echo "$s cached"; return; }
  t0=$(date +%s)
  "$TOOLS/ants_fmriprep.sh" "$SP/mni_fixed.nii.gz" "$SP/moving/$s.nii.gz" "$out/x" >/dev/null 2>&1 \
    || { echo "$s FAILED"; return; }
  "$ANTSPATH/antsApplyTransforms" -d 3 \
    -i "$MB/$s/labels.DKT31.manual.nii.gz" -r "$SP/mni_fixed.nii.gz" \
    -o "$out/labels.DKT31.manual.MNI152.nii.gz" \
    -n NearestNeighbor -t "$out/xComposite.h5" >/dev/null 2>&1 \
    || { echo "$s APPLY FAILED"; return; }
  rm -f "$out/xComposite.h5" "$out/xInverseComposite.h5" "$out/xWarped.nii.gz"
  echo "$s $(( $(date +%s) - t0 ))s"
}
export -f run_one; export SP MB ANTSPATH TOOLS
xargs -P "${JOBS:-4}" -I{} bash -c 'run_one {}' < "$SUBJECTS"
echo BATCH_DONE
