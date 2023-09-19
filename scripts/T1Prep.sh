#! /bin/sh
#
# PURPOSE:
#
# USAGE:
#
# INPUT:
#
# OUTPUT:
#
# FUNCTIONS:
#

########################################################
# global parameters
########################################################

NUMBER_OF_JOBS=-1
target_res=0.5
nu_strength=2
estimate_surf=1
sub=64

# check whether python or python3 is in your path
found=`which python 2>/dev/null`
if [ ! -n "$found" ]; then
  found=`which python3 2>/dev/null`
  if [ ! -n "$found" ]; then
    echo "python or python3 not found. Please use '--python' flag to define python command"
    exit 1
  else
    python=python3
  fi
else
  python=python
fi

echo $PYTHON

########################################################
# run main
########################################################

main ()
{
  parse_args ${1+"$@"}
  
  check_python
  check_files
  get_no_of_cpus
  process

  exit 0
}


########################################################
# check arguments and files
########################################################

parse_args ()
{
  local optname optarg

  if [ $# -lt 1 ]; then
    help
    exit 1
  fi

  while [ $# -gt 0 ]; do
    optname="`echo $1 | sed 's,=.*,,'`"
    optarg="`echo $2`"
    paras="$paras $optname $optarg"

    case "$1" in
        --python*)
          exit_if_empty "$optname" "$optarg"
          python=$optarg
          shift
          ;;
        --outdir*)
          exit_if_empty "$optname" "$optarg"
          outdir=$optarg
          if [ ! -d $LOGDIR ]; then
            mkdir -p $outdir
          fi
          shift
          ;;
        --target-res*)
          exit_if_empty "$optname" "$optarg"
          target_res=$optarg
          shift
          ;;
        --nu-strength*)
          exit_if_empty "$optname" "$optarg"
          nu_strength=$optarg
          shift
          ;;
        --nproc*)
          exit_if_empty "$optname" "$optarg"
          NUMBER_OF_JOBS="-$optarg"
          shift
          ;; 
        --no-surf)
            estimate_surf=0
            ;;
        --fast)
            fast=" --fast "
            ;;
        --robust)
            robust=" --robust "
            ;;
        --quiet | -q)
            GLOBAL_show_progress=0
            ;;
        -h | --help | -v | --version | -V)
            help
            exit 1
            ;;
        -*)
            echo "`basename $0`: ERROR: Unrecognized option \"$1\"" >&2
            ;;
        *)
            ARRAY[$count]=$1
            ((count++))
            ;;
    esac
    shift
  done

}

########################################################
# check arguments
########################################################

exit_if_empty ()
{
  local desc val

  desc="$1"
  shift
  val="$*"

  if [ ! -n "$val" ]; then
    echo ERROR: No argument given with \"$desc\" command line argument! >&2
    exit 1
  fi
}

########################################################
# check files
########################################################

check_files ()
{
  
  SIZE_OF_ARRAY="${#ARRAY[@]}"
  if [ "$SIZE_OF_ARRAY" -eq 0 ]; then
    echo 'ERROR: No files given!' >&2
    help
    exit 1
  fi

  i=0
  while [ "$i" -lt "$SIZE_OF_ARRAY" ]; do
    if [ ! -f "${ARRAY[$i]}" ]; then
      if [ ! -L "${ARRAY[$i]}" ]; then
      echo ERROR: File ${ARRAY[$i]} not found
      help
      exit 1
      fi
    fi
    ((i++))
  done

}

########################################################
# check for python
########################################################

check_python ()
{
  found=`which "${python}" 2>/dev/null`
  if [ ! -n "$found" ]; then
    echo $python not found.
    exit 1
  fi
}

########################################################
# get # of cpus
########################################################
# modified code from
# PPSS, the Parallel Processing Shell Script
# 
# Copyright (c) 2009, Louwrentius
# All rights reserved.

get_no_of_cpus () {

  CPUINFO=/proc/cpuinfo
  ARCH=`uname`

  if [ ! -n "$NUMBER_OF_JOBS" ] | [ $NUMBER_OF_JOBS -le -1 ]; then
    if [ "$ARCH" == "Linux" ]; then
      NUMBER_OF_PROC=`grep ^processor $CPUINFO | wc -l`
    elif [ "$ARCH" == "Darwin" ]; then
      NUMBER_OF_PROC=`sysctl -a hw | grep -w hw.logicalcpu | awk '{ print $2 }'`
    elif [ "$ARCH" == "FreeBSD" ]; then
      NUMBER_OF_PROC=`sysctl hw.ncpu | awk '{ print $2 }'`
    else
      NUMBER_OF_PROC=`grep ^processor $CPUINFO | wc -l`
    fi
  
    if [ ! -n "$NUMBER_OF_PROC" ]; then
      echo "$FUNCNAME ERROR - number of CPUs not obtained. Use --nproc to define number of processes."
      exit 1
    fi
  
    # use all processors if not defined otherwise
    if [ ! -n "$NUMBER_OF_JOBS" ]; then
      NUMBER_OF_JOBS=$NUMBER_OF_PROC
    fi

    if [ $NUMBER_OF_JOBS -le -1 ]; then
      NUMBER_OF_JOBS=$(echo "$NUMBER_OF_PROC + $NUMBER_OF_JOBS" | bc)
      
      if [ "$NUMBER_OF_JOBS" -lt 1 ]; then
        NUMBER_OF_JOBS=1
      fi
    fi
    if [ "$NUMBER_OF_JOBS" -gt "$NUMBER_OF_PROC" ]; then
      NUMBER_OF_JOBS=$NUMBER_OF_PROC
    fi
  fi
}

########################################################
# process data
########################################################

process ()
{
  
  cmd_dir=`dirname $0`
  
  SIZE_OF_ARRAY="${#ARRAY[@]}"

  i=0
  while [ "$i" -lt "$SIZE_OF_ARRAY" ]; do

    # check whether absolute or relative names were given
    if [ ! -f "${ARRAY[$i]}" ]; then
      if [ -f "${pwd}/${ARRAY[$i]}" ]; then
        FILE="${pwd}/${ARRAY[$i]}"
      fi
    else
      FILE=${ARRAY[$i]}
    fi

    # replace white spaces
    FILE=$(echo "$FILE" | sed -e 's/ /\\ /g')

    dn=$(dirname "$FILE")
    bn=$(basename "$FILE")
    
    # if defined use outputdir otherwise use the folder of input files
    if [ ! -n "$outdir" ]; then
      outdir=${dn}
    fi

    # get names
    resampled=$(echo $bn | sed 's/.nii/_res-high_desc-corr.nii/g')
    label=$(echo $bn     | sed 's/.nii/_res-low_label.nii/g')
    atlas=$(echo $bn     | sed 's/.nii/_res-low_atlas.nii/g')
    hemi=$(echo $bn      | sed 's/.nii/_res-high_hemi.nii/g') # -[L|R]_seg will be added internally
    seg=$(echo $bn       | sed 's/.nii/_res-high_desc-corr_seg.nii/g')
    
    # size of sub is dependent on voxel size
    # supress floting number by using scale = 0
    sub=`echo "scale = 0; ${sub} / $target_res" | bc` 

    # call SynthSeg segmentation
    ${python} ${cmd_dir}/SynthSeg_predict.py --i ${FILE} --o ${outdir}/${atlas} ${fast} ${robust} \
        --target-res ${target_res} --threads $NUMBER_OF_PROC --nu-strength ${nu_strength}\
        --label ${outdir}/${label} --resamp ${outdir}/${resampled}
    
    if [ ! -f "${outdir}/${resampled}" ] ||  [ ! -f "${outdir}/${label}" ]; then
      echo
      echo ERROR: ${cmd_dir}/SynthSeg_predict.py failed
      exit 1
    fi
    
    # use output from SynthSeg segmentation to estimate Amap segmentation
    CAT_VolAmap -write_seg 1 1 1 -mrf 0 -sub ${sub} -label ${outdir}/${label} ${outdir}/${resampled}
    
    if [ ! -f "${outdir}/${seg}" ]; then
      echo
      echo ERROR: CAT_VolAmap failed
      exit 1
    fi

    # create hemispheric label maps for cortical surface extraction
    if [ "$estimate_surf" -eq 1 ]; then
        ${python} ${cmd_dir}/partition_hemispheres.py \
            --label ${outdir}/${seg} --atlas ${outdir}/${atlas}
    fi
    
    ((i++))
  done

}

########################################################
# help
########################################################

help ()
{
cat <<__EOM__

USAGE:

PURPOSE:

INPUT:

OUTPUT:

USED FUNCTIONS:

This script was written by Christian Gaser (christian.gaser@uni-jena.de).

__EOM__
}

########################################################
# call main program
########################################################

main ${1+"$@"}
