#!/usr/bin/env bash
# Build newMSM against an FSL installation on macOS.
set -euo pipefail
export FSLDIR="${FSLDIR:-$HOME/fsl}"
export FSLCONFDIR="$FSLDIR/config"
export FSLDEVDIR="${FSLDEVDIR:-$HOME/fsl-dev}"
OMP="${OMP_PREFIX:-/opt/homebrew/opt/libomp}"
SRC="${1:-$HOME/Downloads/newmsm_sources/newMSM}"

# Apple ships GNU Make 3.81, which cannot parse the "define VAR =" syntax in
# FSL's rules.mk -- the compile rules silently never get generated and make
# reports "No rule to make target '<file>.o'".  FSL ships make 4.x itself.
MAKE="$FSLDIR/bin/make"
"$MAKE" --version | head -1

# Apple clang rejects a bare -fopenmp; it needs libomp via -Xpreprocessor.
# Setting USRCXXFLAGS on the command line overrides the "+= -fopenmp" in each
# project Makefile, which is why CXXFLAGS itself must not be touched (doing so
# drops FSL's own include and library paths).
FLAGS=(USRCXXFLAGS="-Xpreprocessor -fopenmp -I$OMP/include"
       USRLDFLAGS="-L$OMP/lib -lomp")

mkdir -p "$FSLDEVDIR"
for d in libraries/msm-newresampler/src libraries/msm-newmeshreg/src src; do
  echo "=== building $d"
  cd "$SRC/$d"
  "$MAKE" "${FLAGS[@]}"
  "$MAKE" "${FLAGS[@]}" install
done
echo "=== done; binaries in $FSLDEVDIR/bin"
ls "$FSLDEVDIR/bin" 2>/dev/null
