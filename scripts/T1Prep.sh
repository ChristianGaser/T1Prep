#! /bin/bash
#
# PURPOSE: This script performs preprocessing steps on T1-weighted MRI images
#          to create segmentations and extract cortical surface. 
#
# USAGE: T1Prep.sh [options] file1.nii file2.nii ...
#
# INPUT: T1-weighted MRI images in NIfTI format (.nii or .nii.gz).
#
# OUTPUT: Processed images and segmentation results in the specified output directory.
#
# FUNCTIONS: 
# - main: The main function that executes the preprocessing steps.
# - parse_args: Parses the command line arguments.
# - exit_if_empty: Checks if a command line argument is empty and exits with an error message if it is.
# - check_python_cmd: Checks if the Python command is available.
# - check_files: Checks if the input files exist.
# - check_python: Checks if the specified Python command is available.
# - get_no_of_cpus: Determines the number of available CPUs.
# - process: Performs the preprocessing steps on each input file.

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
T1prep_dir=$(dirname $(dirname "$0"))
surf_templates_dir=${T1prep_dir}/templates_surfaces_32k
vessel_strength=-1
NUMBER_OF_JOBS=-1
use_bids_naming=1
estimate_surf=1
target_res=0.5
bias_fwhm=15
pre_fwhm=-1
use_sanlm=1
use_amap=1
bin_dir="/usr/local/bin"
debug=0
sub=64
las=0.5

# if we use post-smoothing we have to correct the isovalue/threshold:
# post_fwhm=1 thresh=0.495
# post_fwhm=2 thresh=0.490
# post_fwhm=3 thresh=0.475
post_fwhm=2
thresh=0.490

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
    cmd_dir=$(dirname "$0")
    local optname optarg

    if [ $# -lt 1 ]; then
        help
        exit 1
    fi

    while [ $# -gt 0 ]; do
        optname=$(echo "$1" | sed 's,=.*,,' )
        optarg=$(echo "$2")
        paras="$paras $optname $optarg"

        case "$1" in
            --python)
                exit_if_empty "$optname" "$optarg"
                python=$optarg
                shift
                ;;
            --out-dir | --outdir)
                exit_if_empty "$optname" "$optarg"
                outdir=$optarg
                shift
                ;;
            --target-res)
                exit_if_empty "$optname" "$optarg"
                target_res=$optarg
                shift
                ;;
            --pre-fwhm)
                exit_if_empty "$optname" "$optarg"
                pre_fwhm=$optarg
                shift
                ;;
            --post-fwhm)
                exit_if_empty "$optname" "$optarg"
                post_fwhm=$optarg
                shift
                ;;
            --thresh*)
                exit_if_empty "$optname" "$optarg"
                thresh=$optarg
                shift
                ;;
            --bias-fwhm)
                exit_if_empty "$optname" "$optarg"
                bias_fwhm=$optarg
                shift
                ;;
            --vessel-str*)
                exit_if_empty "$optname" "$optarg"
                vessel_strength=$optarg
                shift
                ;;
            --bin-dir| --bindir)
                exit_if_empty "$optname" "$optarg"
                bin_dir=$optarg
                shift
                ;;
            --nproc)
                exit_if_empty "$optname" "$optarg"
                NUMBER_OF_JOBS="$optarg"
                shift
                ;; 
            --sub)
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
            --no-amap)
                use_amap=0
                ;;
            --no-bids)
                use_bids_naming=0
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
        
    # if target-res is set add a field to the name
    if [ "${target_res}" == "-1" ]; then
        res_str=''
    else
        res_str='_res-high'
    fi
    
    SIZE_OF_ARRAY="${#ARRAY[@]}"

    # set overall starting time
    start0=$(date +%s)

    i=0
    while [ "$i" -lt "$SIZE_OF_ARRAY" ]; do

        # set starting time
        start=$(date +%s)

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
        bn_gz=$(basename "$FILE" | sed -e "s/.img/.nii/g" -e "s/.gz//g")
        
        # if defined use output dir otherwise use the folder of input files
        if [ ! -n "$outdir" ]; then
            outdir=${dn}
        fi
        
        # create outdir if not exists
        if [ ! -d "$outdir" ]; then
            mkdir -p "$outdir"
        fi
        
        if [ "${use_bids_naming}" -eq 1 ]; then
        
            # get output names
            resampled=$(echo "$bn" | sed -e "s/.nii/${res_str}.nii/g")
            sanlm=$(echo "$bn"     | sed -e "s/.nii/_desc-sanlm.nii/g")
            
            # remove T1w|T2w from basename
            label=$(echo "$bn"  | sed -e "s/.nii/${res_str}_label.nii/g")
            atlas=$(echo "$bn"  | sed -e "s/.nii/${res_str}_atlas.nii/g")
            
            # use label from Synthseg if we don't use Amap segmentation
            if [ "${use_amap}" -eq 1 ]; then
                seg=$(echo "$bn"    | sed -e "s/.nii/${res_str}_seg.nii/g")
            else
                seg=${label}
            fi
            
            hemi_left=$(echo "$seg"  | sed -e "s/.nii/_hemi-L.nii/g")
            hemi_right=$(echo "$seg" | sed -e "s/.nii/_hemi-R.nii/g")
            gmt_left=$(echo "$bn"    | sed -e "s/.nii/${res_str}_hemi-L_thickness.nii/g")
            gmt_right=$(echo "$bn"   | sed -e "s/.nii/${res_str}_hemi-R_thickness.nii/g")
            
            # for the following filenames we have to remove the potential .gz from name
            ppm_left=$(echo "$bn_gz"    | sed -e "s/.nii/${res_str}_hemi-L_ppm.nii/g")
            ppm_right=$(echo "$bn_gz"   | sed -e "s/.nii/${res_str}_hemi-R_ppm.nii/g")
            mid_left=$(echo "$bn_gz"    | sed -e "s/.nii/_hemi-L_midthickness.surf.gii/g")
            mid_right=$(echo "$bn_gz"   | sed -e "s/.nii/_hemi-R_midthickness.surf.gii/g")
            thick_left=$(echo "$bn_gz"  | sed -e "s/.nii/_hemi-L_thickness.txt/g")
            thick_right=$(echo "$bn_gz" | sed -e "s/.nii/_hemi-R_thickness.txt/g")
            pbt_left=$(echo "$bn_gz"    | sed -e "s/.nii/_hemi-L_pbt.txt/g")
            pbt_right=$(echo "$bn_gz"   | sed -e "s/.nii/_hemi-R_pbt.txt/g")
            sphere_left=$(echo "$bn_gz"     | sed -e "s/.nii/_hemi-L_sphere.surf.gii/g")
            sphere_right=$(echo "$bn_gz"    | sed -e "s/.nii/_hemi-R_sphere.surf.gii/g")
            spherereg_left=$(echo "$bn_gz"  | sed -e "s/.nii/_hemi-L_sphere.reg.surf.gii/g")
            spherereg_right=$(echo "$bn_gz" | sed -e "s/.nii/_hemi-R_sphere.reg.surf.gii/g")
            resamp_thick_left=$(echo "$bn_gz"  | sed -e "s/.nii/_hemi-L_thickness.resampled_32k.12mm.gii/g")
            resamp_thick_right=$(echo "$bn_gz" | sed -e "s/.nii/_hemi-R_thickness.resampled_32k.12mm.gii/g")
        else
            echo "Not yet prepared for other naming scheme"
            exit
        fi
        
        # print progress and filename
        j=$(expr $i + 1)
        echo -e "${BOLD}${BLACK}######################################################${NC}"
        echo -e "${GREEN}${j}/${SIZE_OF_ARRAY} ${BOLD}${BLACK}Processing ${FILE}${NC}"

        # 1. Call SANLM denoising filter
        # ----------------------------------------------------------------------
        if [ "${use_sanlm}" -eq 1 ]; then
            echo -e "${BLUE}SANLM denoising${NC}"
            echo -e "${BLUE}---------------------------------------------${NC}"
            ${bin_dir}/CAT_VolSanlm "${FILE}" "${outdir}/${sanlm}"
            input="${outdir}/${sanlm}"
        else
            input="${FILE}"
        fi
        
        # 2. Call SynthSeg segmentation 
        # ----------------------------------------------------------------------
        # check for outputs from previous step
        if [ -f "${input}" ]; then
            echo -e "${BLUE}SynthSeg segmentation${NC}"
            echo -e "${BLUE}---------------------------------------------${NC}"
                "${python}" "${cmd_dir}/SynthSeg_predict.py" --i "${input}" --o "${outdir}/${atlas}" ${fast} ${robust} \
                    --target-res "${target_res}" --threads "$NUMBER_OF_JOBS" --vessel-strength "${vessel_strength}" \
                    --label "${outdir}/${label}" --resamp "${outdir}/${resampled}"
        else
            echo -e "${RED}ERROR: CAT_VolSanlm failed${NC}"
            ((i++))
            continue
        fi
        
        # remove denoised image
        if [ "${use_sanlm}" -eq 1 ]; then
            rm "${outdir}/${sanlm}"
        fi
        
        # 3. Amap segmentation using output from SynthSeg label segmentation
        # ----------------------------------------------------------------------
        # check for outputs from previous step
        if [ -f "${outdir}/${resampled}" ] && [ -f "${outdir}/${label}" ]; then
            if [ "${use_amap}" -eq 1 ]; then
                echo -e "${BLUE}Amap segmentation${NC}"
                echo -e "${BLUE}---------------------------------------------${NC}"
                ${bin_dir}/CAT_VolAmap -cleanup 2 -las ${las} -mrf 0 -bias-fwhm "${bias_fwhm}" -write-seg 1 1 1 -sub "${sub}" -label "${outdir}/${label}" "${outdir}/${resampled}"
            fi
        else
            echo -e "${RED}ERROR: ${cmd_dir}/SynthSeg_predict.py failed${NC}"
            ((i++))
            continue
        fi

        # optionally extract surface
        if [ "${estimate_surf}" -eq 1 ]; then
                
            # 4. Create hemispheric label maps for cortical surface extraction
            # ----------------------------------------------------------------------
            # check for outputs from previous step
            if [ -f "${outdir}/${seg}" ]; then
                echo -e "${BLUE}Hemispheric partitioning${NC}"
                echo -e "${BLUE}---------------------------------------------${NC}"
                    "${python}" "${cmd_dir}/partition_hemispheres.py" \
                        --label "${outdir}/${seg}" --atlas "${outdir}/${atlas}"
            else
                echo -e "${RED}ERROR: CAT_VolAmap failed${NC}"
                ((i++))
                continue
            fi
            echo ${T1prep_dir}
            # 5. Estimate thickness and percentage position maps for each hemisphere
            #    and extract cortical surface
            # ----------------------------------------------------------------------
            # check for outputs from previous step
            for side in left right; do
              
                # create dynamic variables
                pbt=pbt_$side
                thick=thick_$side
                hemi=hemi_$side
                ppm=ppm_$side
                gmt=gmt_$side
                mid=mid_$side
                sphere=sphere_$side
                spherereg=spherereg_$side
                fshemi=$(echo "$side" | sed -e "s/left/lh/g" -e "s/right/rh/g")
                Fsavg=${surf_templates_dir}/${fshemi}.central.freesurfer.gii
                Fsavgsphere=${surf_templates_dir}/${fshemi}.sphere.freesurfer.gii
                resamp_thick=resamp_thick_$side
                
                if [ -f "${outdir}/${!hemi}" ]; then
                    echo -e "${BLUE}Extracting $side hemisphere${NC}"
                    echo -e "${BLUE}---------------------------------------------${NC}"
                    echo Calculate thickness
                    ${bin_dir}/CAT_VolThicknessPbt ${outdir}/${!hemi} ${outdir}/${!gmt} ${outdir}/${!ppm}
                    
                    # The pre-smoothing helps in preserving gyri and sulci by creating a weighted 
                    # average between original and smoothed images based on their distance to 
                    # the threshold (isovalue). A negative value will force masked smoothing, which may preserves gyri and sulci even better.
                    # In contrast, post-smoothing aids in correcting the mesh in folded areas like gyri and sulci and removes the voxel-steps
                    # that are visible after marching cubes. However, this also slightly changes the curve in we have to correct the isovalue:
                    # -post-fwhm 1 -thresh 0.495
                    # -post-fwhm 2 -thresh 0.490
                    # -post-fwhm 3 -thresh 0.475
                    echo Extract surface
                    ${bin_dir}/CAT_VolMarchingCubes -pre-fwhm ${pre_fwhm} -post-fwhm ${post_fwhm} -thresh ${thresh} -scl-opening 0.9 ${outdir}/${!ppm} ${outdir}/${!mid}
                    echo Map thickness values
                    ${bin_dir}/CAT_3dVol2Surf -start -0.5 -steps 7 -end 0.5 ${outdir}/${!mid} ${outdir}/${!gmt} ${outdir}/${!pbt}
                    ${bin_dir}/CAT_SurfDistance -mean -thickness ${outdir}/${!pbt} ${outdir}/${!mid} ${outdir}/${!thick}
                    echo Spherical inflation
                    ${bin_dir}/CAT_Surf2Sphere ${outdir}/${!mid} ${outdir}/${!sphere} 6
                    echo Spherical registration
                    ${bin_dir}/CAT_WarpSurf -steps 2 -avg -i ${outdir}/${!mid} -is ${outdir}/${!sphere} -t ${Fsavg} -ts ${Fsavgsphere} -ws ${outdir}/${!spherereg}
                    echo Resample and smooth
                    ${bin_dir}CAT_ResampleSurf ${outdir}/${!mid} ${outdir}/${!spherereg} ${Fsavgsphere} ${outdir}/${!resamp_thick} ${outdir}/${!thick}
                else
                    echo -e "${RED}ERROR: ${python} ${cmd_dir}/partition_hemispheres.py failed${NC}"
                    ((i++))
                    continue
                fi
            done
            
        fi # estimate_surf

        # remove temporary files if not debugging
        if [ "${debug}" -eq 0 ]; then
            #rm ${outdir}/${atlas} ${outdir}/${seg} ${outdir}/${label}
            
            # only remove temporary files if surfaces exist
            if [ -f "${outdir}/${mid_left}" ] && [ -f "${outdir}/${mid_right}" ]; then
                rm "${outdir}/${hemi_left}" "${outdir}/${hemi_right}"
                rm "${outdir}/${ppm_left}" "${outdir}/${ppm_right}"
                #rm ${outdir}/${gmt_left} ${outdir}/${gmt_right} 
            fi
        fi

        # print execution time per data set
        end=$(date +%s)
        runtime=$((end - start))
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
  T1Prep.sh [--python python_command] [--out-dir out_folder] [--bin-dir bin_folder]
                    [--target-res voxel_size] [--vessel-strength vessel_strength] 
                    [--no-sanlm] [--no-amap] [--no-surf] [--nproc number_of_processes] 
                    [--sub subsampling] [--fast] [--robust] [--debug] filenames 
 
  --python <FILE>            python command (default $python)
  --out-dir <DIR>            output folder (default same folder)
  --bin-dir <DIR>            folder of CAT binaries (default $bin_dir)
  --target-res <NUMBER>      target voxel size in mm for resampled and hemispheric label data  
                             that will be used for cortical surface extraction. Use a negative
                             value to save outputs with original voxel size (default $target_res).
  --bias-fwhm <NUMBER>       FWHM size of nu-correction in CAT_VolAmap (default $bias_fwhm). 
  --pre-fwhm  <NUMBER>       FWHM size of pre-smoothing in CAT_VolMarchingCubes (default $pre_fwhm). 
  --post-fwhm <NUMBER>       FWHM size of post-smoothing in CAT_VolMarchingCubes (default $post_fwhm). 
  --thresh    <NUMBER>       Threshold (isovalue) for creating surface in CAT_VolMarchingCubes (default $thresh). 
  --vessel-strength <NUMBER> strength of vessel-correction (-1 - automatic, 0 - none, 1 - medium
                             2 - strong). (default $vessel_strength). 
  --nproc <NUMBER>           number of parallel jobs (=number of processors)
  --sub <NUMBER>             subsampling for Amap segmentation (default $sub)
  --no-surf                  skip surface and thickness estimation
  --no-sanlm                 skip denoising with SANLM-filter
  --no-amap                  use segmentation from mri_synthseg instead of Amap
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
  nifti files

OUTPUT:
  segmented images
  surface extractions

USED FUNCTIONS:
  CAT_VolAmap
  CAT_Sanlm
  CAT_VolThicknessPbt
  CAT_VolMarchingCubes
  CAT_3dVol2Surf
  CAT_SurfDistance
  ${cmd_dir}/SynthSeg_predict.py
  
This script was written by Christian Gaser (christian.gaser@uni-jena.de).

__EOM__
}

########################################################
# call main program
########################################################

main "${@}"
    