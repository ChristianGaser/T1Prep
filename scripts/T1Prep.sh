#! /bin/bash
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

# output colors
BOLD='\033[1;30m'
CYAN=$(tput setaf 6)
PINK=$(tput setaf 5)
BLUE=$(tput setaf 4)
YELLOW=$(tput setaf 3)
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)


NUMBER_OF_JOBS=-1
estimate_surf=1
target_res=0.5
nu_strength=2
sub=64
use_sanlm=1
debug=0
vessel_strength=-1

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
        if [ ! -d $outdir ]; then
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
      --vessel-strength*)
        exit_if_empty "$optname" "$optarg"
        vessel_strength=$optarg
        shift
        ;;
      --nproc*)
        exit_if_empty "$optname" "$optarg"
        NUMBER_OF_JOBS="$optarg"
        shift
        ;; 
      --no-surf)
          estimate_surf=0
          ;;
      --no-sanlm)
          use_sanlm=0
          ;;
      --fast)
          fast=" --fast "
          ;;
      --debug)
          debug=1
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
    echo "${RED}ERROR: No argument given with \"$desc\" command line argument!${NC}" >&2
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
    echo "${RED}ERROR: No files given!${NC}" >&2
    help
    exit 1
  fi

  i=0
  while [ "$i" -lt "$SIZE_OF_ARRAY" ]; do
    if [ ! -f "${ARRAY[$i]}" ]; then
      if [ ! -L "${ARRAY[$i]}" ]; then
      echo "${RED}ERROR: File ${ARRAY[$i]} not found${NC}"
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
    echo "${RED}ERROR: $python not found${NC}"
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
      echo "${RED}${FUNCNAME} ERROR - number of CPUs not obtained. Use --nproc to define number of processes.${NC}"
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
  
  # if target-res is set add a field to the name
  if [ "${target_res}" == "-1" ]; then
    res_str=''
  else
    res_str='_res-high'
  fi
  
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
    FILE=$(echo "$FILE" | sed -e "s/ /\\ /g")

    # get directory and basename and also consider ancient Analyze img files
    dn=$(dirname "$FILE")
    bn=$(basename "$FILE" | sed -e "s/.img/.nii/g")
    
    # if defined use outputdir otherwise use the folder of input files
    if [ ! -n "$outdir" ]; then
      outdir=${dn}
    fi

    # get names
    resampled=$(echo $bn | sed -e "s/.nii/${res_str}_desc-corr.nii/g")
    sanlm=$(echo $bn     | sed -e "s/.nii/_desc-sanlm.nii/g")
    label=$(echo $bn     | sed -e "s/.nii/_res-low_label.nii/g")
    atlas=$(echo $bn     | sed -e "s/.nii/_res-low_atlas.nii/g")
    hemi=$(echo $bn      | sed -e "s/.nii/${res_str}_hemi.nii/g") # -[L|R]_seg will be added internally
    seg=$(echo $bn       | sed -e "s/.nii/${res_str}_desc-corr_seg.nii/g")
    
    # size of sub is dependent on voxel size
    # supress floating number by using scale = 0
    sub=`echo "scale = 0; ${sub} / $target_res" | bc` 
    
    echo
    echo -e "${BOLD}Processing ${FILE} ${NC}"
    
    # apply SANLM denoising filter
    if [ "${use_sanlm}" -eq 1 ]; then
      echo SANLM denoising
      CAT_VolSanlm ${FILE} ${outdir}/${sanlm}
      input=${outdir}/${sanlm}
    else
      input=${FILE}
    fi
    
    # call SynthSeg segmentation
    ${python} ${cmd_dir}/SynthSeg_predict.py --i ${input} --o ${outdir}/${atlas} ${fast} ${robust} \
        --target-res ${target_res} --threads $NUMBER_OF_JOBS --nu-strength ${nu_strength}\
        --vessel-strength ${vessel_strength} --label ${outdir}/${label} --resamp ${outdir}/${resampled}
    
    if [ ! -f "${outdir}/${resampled}" ] ||  [ ! -f "${outdir}/${label}" ]; then
      echo -e "${RED}ERROR: ${cmd_dir}/SynthSeg_predict.py failed ${NC}"
    fi
    
    # use output from SynthSeg segmentation to estimate Amap segmentation
    CAT_VolAmap -write_seg 1 1 1 -mrf 0 -sub ${sub} -label ${outdir}/${label} ${outdir}/${resampled}
    
    # create hemispheric label maps for cortical surface extraction
    if [ -f "${outdir}/${seg}" ]; then
      if [ "${estimate_surf}" -eq 1 ]; then
          ${python} ${cmd_dir}/partition_hemispheres.py \
              --label ${outdir}/${seg} --atlas ${outdir}/${atlas}
      fi      
    else
      echo -e "${RED}ERROR: CAT_VolAmap failed ${NC}"
    fi

    
    if [ "${debug}" -eq 0 ]; then
      if [ "${use_sanlm}" -eq 1 ]; then
        rm ${outdir}/${sanlm} 
      fi
      rm ${outdir}/${atlas} ${outdir}/${seg} ${outdir}/${label}
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
