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
CYAN=$(tput setaf 6)
PINK=$(tput setaf 5)
BLUE=$(tput setaf 4)
YELLOW=$(tput setaf 3)
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
BLACK=$(tput setaf 0)
NC=$(tput sgr0)
BOLD=$(tput bold)

# defaults
vessel_strength=-1
NUMBER_OF_JOBS=-1
estimate_surf=1
target_res=0.5
nu_strength=2
use_sanlm=1
debug=0
sub=64


########################################################
# run main
########################################################

main ()
{
    check_python_cmd
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
    cmd_dir=`dirname $0`
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
            --outdir* | --out-dir)
                exit_if_empty "$optname" "$optarg"
                outdir=$optarg
                shift
                ;;
            --target-res*)
                exit_if_empty "$optname" "$optarg"
                target_res=$optarg
                shift
                ;;
            --nu-str*)
                exit_if_empty "$optname" "$optarg"
                nu_strength=$optarg
                shift
                ;;
            --vessel-str*)
                exit_if_empty "$optname" "$optarg"
                vessel_strength=$optarg
                shift
                ;;
            --nproc*)
                exit_if_empty "$optname" "$optarg"
                NUMBER_OF_JOBS="$optarg"
                shift
                ;; 
            --sub*)
                exit_if_empty "$optname" "$optarg"
                sub="$optarg"
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
            --robust)
                robust=" --robust "
                    ;;
            --debug)
                debug=1
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
# check for python version
########################################################

check_python_cmd ()
{
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
            NUMBER_OF_JOBS=$NUMBER_OF_PROC
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
        
    # check that sub is large enough
    if [ $sub -lt 20 ]; then
        echo -e "${RED}ERROR: sub has to be >= 20${NC}"
        exit 1
    fi

    # if target-res is set add a field to the name
    if [ "${target_res}" == "-1" ]; then
        res_str=''
    else
        res_str='_res-high'
    fi
    
    SIZE_OF_ARRAY="${#ARRAY[@]}"

    # set overall starting time
    start0=`date +%s`

    i=0
    while [ "$i" -lt "$SIZE_OF_ARRAY" ]; do
      
        # set starting time
        start=`date +%s`

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
        
        # create outdir if not exists
        if [ ! -d $outdir ]; then
            mkdir -p $outdir
        fi
        
        # get output names
        resampled=$(echo $bn | sed -e "s/.nii/${res_str}_desc-corr.nii/g")
        sanlm=$(echo $bn     | sed -e "s/.nii/_desc-sanlm.nii/g")
        
        # remove T1w|T2w from basename
        bn0=$(echo $bn     | sed -e "s/_T1w.nii/.nii/g" -e "s/_T2w.nii/.nii/g")        
        label=$(echo $bn0  | sed -e "s/.nii/${res_str}_label.nii/g")
        atlas=$(echo $bn0  | sed -e "s/.nii/${res_str}_atlas.nii/g")
        seg=$(echo $bn0    | sed -e "s/.nii/${res_str}_desc-corr_seg.nii/g")
        hemi_L=$(echo $bn0 | sed -e "s/.nii/${res_str}_desc-corr_hemi-L_seg.nii/g") 
        hemi_R=$(echo $bn0 | sed -e "s/.nii/${res_str}_desc-corr_hemi-R_seg.nii/g")
        gmt_L=$(echo $bn0  | sed -e "s/.nii/${res_str}_desc-corr_hemi-L_thickness.nii/g") 
        gmt_R=$(echo $bn0  | sed -e "s/.nii/${res_str}_desc-corr_hemi-R_thickness.nii/g")
        ppm_L=$(echo $bn0  | sed -e "s/.nii/${res_str}_desc-corr_hemi-L_ppm.nii/g") 
        ppm_R=$(echo $bn0  | sed -e "s/.nii/${res_str}_desc-corr_hemi-R_ppm.nii/g")
        mid_L=$(echo $bn0  | sed -e "s/.nii/_hemi-L_midthickness.surf.gii/g") 
        mid_R=$(echo $bn0  | sed -e "s/.nii/_hemi-R_midthickness.surf.gii/g")
        
        # print progress and filename
        j=`expr $i + 1`
        echo -e "${BOLD}${BLACK}######################################################${NC}"
        echo -e "${GREEN}${j}/${SIZE_OF_ARRAY} ${BOLD}${BLACK}Processing ${FILE} ${NC}"

        # 1. Call SANLM denoising filter
        # ----------------------------------------------------------------------
        if [ "${use_sanlm}" -eq 1 ]; then
            echo SANLM denoising
            echo "---------------------------------------------"
            cmd="CAT_VolSanlm ${FILE} ${outdir}/${sanlm}"
            eval ${cmd}
            input=${outdir}/${sanlm}
        else
            input=${FILE}
        fi
        
        # 2. Call SynthSeg segmentation 
        # ----------------------------------------------------------------------
        # check for outputs from previous step
        if [ -f "${input}" ]; then
            echo SynthSeg segmentation
            echo "---------------------------------------------"
            cmd="${python} ${cmd_dir}/SynthSeg_predict.py --i ${input} --o ${outdir}/${atlas} ${fast} ${robust} \
                    --target-res ${target_res} --threads $NUMBER_OF_JOBS --nu-strength ${nu_strength}\
                    --vessel-strength ${vessel_strength} --label ${outdir}/${label} --resamp ${outdir}/${resampled}"
            eval ${cmd}
        else
            echo -e "${RED}ERROR: CAT_VolSanlm failed ${NC}"
            echo -e "${CYAN}${cmd}${NC}"
            ((i++))
            continue
        fi
        
        # remove denoised image
        if [ "${use_sanlm}" -eq 1 ]; then
            rm ${outdir}/${sanlm} 
        fi
        
        # 3. Amap segmentation using output from SynthSeg label segmentation
        # ----------------------------------------------------------------------
        # check for outputs from previous step
        if [ -f "${outdir}/${resampled}" ] &&  [ -f "${outdir}/${label}" ]; then
            echo Amap segmentation
            echo "---------------------------------------------"
            cmd="CAT_VolAmap -write_seg 1 1 1 -mrf 0 -sub ${sub} -label ${outdir}/${label} ${outdir}/${resampled}"
            eval ${cmd}
        else
            echo -e "${RED}ERROR: ${cmd_dir}/SynthSeg_predict.py failed ${NC}"
            echo -e "${CYAN}${cmd}${NC}"
            ((i++))
            continue
        fi

        # optionally extract surface
        if [ "${estimate_surf}" -eq 1 ]; then
                
            # 4. Create hemispheric label maps for cortical surface extraction
            # ----------------------------------------------------------------------
            # check for outputs from previous step
            if [ -f "${outdir}/${seg}" ]; then
                echo Hemispheric partitioning
                echo "---------------------------------------------"
                cmd="${python} ${cmd_dir}/partition_hemispheres.py \
                    --label ${outdir}/${seg} --atlas ${outdir}/${atlas}"
                eval ${cmd}
            else
                echo -e "${RED}ERROR: CAT_VolAmap failed ${NC}"
                ((i++))
                continue
            fi
            
            # 5. Estimate thickness and percentage position maps for each hemisphere
            #    and extract cortical surface
            # ----------------------------------------------------------------------
            # check for outputs from previous step
            if [ -f "${outdir}/${hemi_L}" ]; then
                echo Extracting left hemisphere
                echo "---------------------------------------------"
                cmd="CAT_VolThicknessPbt ${outdir}/${hemi_L} ${outdir}/${gmt_L} ${outdir}/${ppm_L}"
                eval ${cmd}
                cmd="CAT_MarchingCubesGenus0 -thresh 0.5 -dist 0.9 ${outdir}/${ppm_L} ${outdir}/${mid_L}"
                eval ${cmd}
            fi
            if [ -f "${outdir}/${hemi_R}" ]; then
                echo Extracting right hemisphere
                echo "---------------------------------------------"
                cmd="CAT_VolThicknessPbt ${outdir}/${hemi_R} ${outdir}/${gmt_R} ${outdir}/${ppm_R}"
                eval ${cmd}
                cmd="CAT_MarchingCubesGenus0 -thresh 0.5 -dist 0.9 ${outdir}/${ppm_R} ${outdir}/${mid_R}"
                eval ${cmd}
            fi
            if [ ! -f "${outdir}/${hemi_L}" ] && [ ! -f "${outdir}/${hemi_R}" ]; then
                echo -e "${RED}ERROR: partition_hemispheres.py failed ${NC}"
                ((i++))
                continue
            fi
            
        fi # estimate_surf

        # remove temporary files if not debugging
        if [ "${debug}" -eq 0 ]; then
            rm ${outdir}/${atlas} ${outdir}/${seg} ${outdir}/${label}
            
            # only remove temporary files if surfaces exist
            if [ -f "${outdir}/${mid_L}" ] && [ -f "${outdir}/${mid_R}" ]; then
                rm ${outdir}/${hemi_L} ${outdir}/${hemi_R}
                rm ${outdir}/${ppm_L} ${outdir}/${ppm_R}
                #rm ${outdir}/${gmt_L} ${outdir}/${gmt_R} 
            fi
        fi

        # print execution time per data set
        end=`date +%s`
        runtime=$((end-start))
        echo -e "${GREEN}Finished after ${runtime}s${NC}"
            
        ((i++))
    done
    
    # print overall execution time for more than one data set
    if [ "$SIZE_OF_ARRAY" -gt 1 ]; then
        end0=`date +%s`
        runtime=$((end0-start0))
        runtime="T1Prep finished after: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
        echo -e "${GREEN}${runtime}${NC}"  
    fi

}

########################################################
# help
########################################################

help ()
{
cat <<__EOM__

USAGE:
  T1Prep.sh [--python python_command] [--outdir out_folder] [--target-res voxel_size] 
                    [--nu-strength nu_strength] [--vessel-strength vessel_strength]
                    [--nproc number_of_processes] [--sub subsampling] [--no-surf]
                    [--no-sanlm] [--fast] [--robust] [--debug] filenames 
 
  --python <FILE>            python command (default $python)
  --outdir <DIR>             output folder (default same folder)
  --target-res <NUMBER>      target voxel size in mm for resampled and hemispheric label data  
                             that will be used for cortical surface extraction. Use a negative
                             value to save outputs with original voxel size (default $target_res).
  --nu-strength <NUMBER>     strength of nu-correction (0 - none, 1 - light, 2 - medium, 3 - strong
                             4 - heavy). (default $nu_strength). 
  --vessel-strength <NUMBER> strength of vessel-correction (-1 - automatic, 0 - none, 1 - medium
                             2 - strong). (default $vessel_strength). 
  --nproc <NUMBER>           number of parallel jobs (=number of processors)
  --sub <NUMBER>             subsampling for Amap segmentation (default $sub)
  --no-surf                  skip surface and thickness estimation
  --no-sanlm                 skip denoising with SANLM-filter
  --fast                     bypass some processing for faster prediction
  --robust                   use robust predictions (slower)
  --debug                    keep temporary files for debugging
 
PURPOSE:
  Computational Anatomy Pipeline for structural MRI data 

EXAMPLE
  T1Prep.sh --fast --outdir test_folder single_subj_T1.nii.
  This command will extract segmentation and surface maps for single_subj_T1.nii
  and bypasses some processing steps for faster (but less accurate) processing. 
  Resuts will be saved in test_folder.

  T1Prep.sh --target-res -1 --no-surf single_subj_T1.nii.
  This command will extract segmentation maps for single_subj_T1.nii with
  the original voxel size in the same folder.

INPUT:
  analyze or nifti files

OUTPUT:
  segmented images
  surface extractions

USED FUNCTIONS:
  CAT_VolAmap
  CAT_Sanlm
  ${cmd_dir}/SynthSeg_predict.py
  
This script was written by Christian Gaser (christian.gaser@uni-jena.de).

__EOM__
}

########################################################
# call main program
########################################################

main ${1+"$@"}
