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
UNDERLINE=$(tput smul)

# defaults
T1prep_dir=$(dirname $(dirname "$0"))
surf_templates_dir=${T1prep_dir}/templates_surfaces_32k
NUMBER_OF_JOBS=-1
use_bids_naming=0
median_filter=2
estimate_surf=1
min_thickness=1
registration=0 # currently skip spherical registration to save time
post_fwhm=2
pre_fwhm=2
use_sanlm=0
bin_dir="/usr/local/bin"
thresh=0.5
debug=0
sub=64
las=0


########################################################
# run main
########################################################

main ()
{
    logo
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
            --min-thickness)
                exit_if_empty "$optname" "$optarg"
                min_thickness=$optarg
                shift
                ;;
            --median-filter)
                exit_if_empty "$optname" "$optarg"
                median_filter=$optarg
                shift
                ;;
            --thresh*)
                exit_if_empty "$optname" "$optarg"
                thresh=$optarg
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
            --bids)
                use_bids_naming=1
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
# logo
########################################################

logo ()
{
echo $BLUE
cat <<__EOM__

████████╗ ██╗ ██████╗ ██████╗ ███████╗██████╗
╚══██╔══╝███║ ██╔══██╗██╔══██╗██╔════╝██╔══██╗
   ██║    ██║ █████╔╝║██████╔╝█████╗  ██████╔╝
   ██║    ██║ ██╔═══╝ ██╔══██╗██╔══╝  ██╔═══╝ 
   ██║    ██║ ██║     ██║  ██║███████╗██║     
   ╚═╝    ╚═╝ ╚═╝     ╚═╝  ╚═ ╚══════╝╚═╝     

__EOM__
echo $NC
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
# progress bar
########################################################
bar() {
    # Usage: bar 1 100
    #            ^----- Elapsed Percentage (0-100).
    #               ^-- Total length in chars.
    ((elapsed=$1*$2/100))

    # Create the bar with spaces.
    printf -v prog  "%${elapsed}s"
    printf -v total "%$(($2-elapsed))s"

    printf '%s\r' "${prog// /■}${total} ${elapsed}%"
}

########################################################
# surface estimation
########################################################
surface_estimation() {
  
    side=$1
    outmridir=$2
    outsurfdir=$3
    registration=$4

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
    
    if [ -f "${outmridir}/${!hemi}" ]; then
        echo Calculate $side thickness
        #${bin_dir}/CAT_VolThicknessPbt -weight-thickness -no-thin-cortex -min-thickness ${min_thickness} -fwhm 4 ${outmridir}/${!hemi} ${outmridir}/${!gmt} ${outmridir}/${!ppm}
        ${bin_dir}/CAT_VolThicknessPbt -min-thickness ${min_thickness} -fwhm 4 ${outmridir}/${!hemi} ${outmridir}/${!gmt} ${outmridir}/${!ppm}
        
        # The pre-smoothing helps in preserving gyri and sulci by creating a weighted 
        # average between original and smoothed images based on their distance to 
        # the threshold (isovalue). A negative value will force masked smoothing, which may preserves gyri and sulci even better.
        # In contrast, post-smoothing aids in correcting the mesh in folded areas like gyri and sulci and removes the voxel-steps
        # that are visible after marching cubes. However, this also slightly changes the curve in we have to correct the isovalue:
        # -post-fwhm 1 -thresh 0.495
        # -post-fwhm 2 -thresh 0.490
        # -post-fwhm 3 -thresh 0.475
        echo Extract $side surface
        ${bin_dir}/CAT_VolMarchingCubes -median-filter ${median_filter} -pre-fwhm ${pre_fwhm} -post-fwhm ${post_fwhm} -thresh ${thresh} -no-distopen ${outmridir}/${!ppm} ${outsurfdir}/${!mid}
        echo Map $side thickness values
        ${bin_dir}/CAT_3dVol2Surf -weighted_avg -start -0.4 -steps 5 -end 0.4 ${outsurfdir}/${!mid} ${outmridir}/${!gmt} ${outsurfdir}/${!pbt}
        ${bin_dir}/CAT_SurfDistance -mean -thickness ${outsurfdir}/${!pbt} ${outsurfdir}/${!mid} ${outsurfdir}/${!thick}
        if [ "${registration}" -eq 1 ]; then
            echo Spherical inflation $side hemisphere
            ${bin_dir}/CAT_Surf2Sphere ${outsurfdir}/${!mid} ${outsurfdir}/${!sphere} 6
            echo Spherical registration $side hemisphere
            ${bin_dir}/CAT_WarpSurf -steps 2 -avg -i ${outsurfdir}/${!mid} -is ${outsurfdir}/${!sphere} -t ${Fsavg} -ts ${Fsavgsphere} -ws ${outsurfdir}/${!spherereg}
        fi
    else
        echo -e "${RED}ERROR: ${python} ${cmd_dir}/deepmriprep_predict.py failed${NC}"
        ((i++))
        continue
    fi

}

########################################################
# process data
########################################################

process ()
{
        
    # if target-res is set add a field to the name
    res_str='_res-high'
    
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
            sanlm=$(echo "$bn"     | sed -e "s/.nii/_desc-sanlm.nii/g")
            
            # remove T1w|T2w from basename
            seg=$(echo "$bn"  | sed -e "s/.nii/_seg.nii/g")            
            
            hemi_left=$(echo "$seg"  | sed -e "s/.nii/_hemi-L.nii/g")
            hemi_right=$(echo "$seg" | sed -e "s/.nii/_hemi-R.nii/g")
            gmt_left=$(echo "$bn"    | sed -e "s/.nii/_hemi-L_thickness.nii/g")
            gmt_right=$(echo "$bn"   | sed -e "s/.nii/_hemi-R_thickness.nii/g")
            
            # for the following filenames we have to remove the potential .gz from name
            ppm_left=$(echo "$bn_gz"    | sed -e "s/.nii/_hemi-L_ppm.nii/g")
            ppm_right=$(echo "$bn_gz"   | sed -e "s/.nii/_hemi-R_ppm.nii/g")
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
            
            outmridir=${outdir}
            outsurfdir=${outdir}
        else
            # get output names
            sanlm=$(echo "$bn"     | sed -e "s/.nii/_desc-sanlm.nii/g")
            
            # remove T1w|T2w from basename
            seg=$(echo "$bn"  | sed -e "s/.nii/_seg.nii/g")
                        
            hemi_left=$(echo "$seg"  | sed -e "s/.nii/_hemi-L.nii/g")
            hemi_right=$(echo "$seg" | sed -e "s/.nii/_hemi-R.nii/g")
            gmt_left=$(echo "$bn"    | sed -e "s/.nii/_hemi-L_thickness.nii/g")
            gmt_right=$(echo "$bn"   | sed -e "s/.nii/_hemi-R_thickness.nii/g")
            
            # for the following filenames we have to remove the potential .gz from name
            ppm_left=$(echo "$bn_gz"    | sed -e "s/.nii/_hemi-L_ppm.nii/g")
            ppm_right=$(echo "$bn_gz"   | sed -e "s/.nii/_hemi-R_ppm.nii/g")
            
            mid_left=$(echo "lh.central.${bn_gz}" | sed -e "s/.nii/.gii/g")
            mid_right=$(echo "rh.central.${bn_gz}" | sed -e "s/.nii/.gii/g")
            thick_left=$(echo "lh.thickness.${bn_gz}" | sed -e "s/.nii//g")
            thick_right=$(echo "rh.thickness.${bn_gz}" | sed -e "s/.nii//g")
            pbt_left=$(echo "lh.pbt.${bn_gz}" | sed -e "s/.nii//g")
            pbt_right=$(echo "rh.pbt.${bn_gz}" | sed -e "s/.nii//g")
            sphere_left=$(echo "lh.sphere.${bn_gz}" | sed -e "s/.nii/.gii/g")
            sphere_right=$(echo "rh.sphere.${bn_gz}" | sed -e "s/.nii/.gii/g")
            spherereg_left=$(echo "lh.sphere.reg.${bn_gz}" | sed -e "s/.nii/.gii/g")
            spherereg_right=$(echo "rh.sphere.reg.${bn_gz}" | sed -e "s/.nii/.gii/g")
            
            outmridir=${outdir}/mri
            outsurfdir=${outdir}/surf
            
        fi
        
        # create output folders if needed
        if [ ! -d "$outmridir" ]; then
            mkdir -p "$outmridir"
        fi
        if [ ! -d "$outsurfdir" ]; then
            mkdir -p "$outsurfdir"
        fi

        # print progress and filename
        j=$(expr $i + 1)
        echo -e "${BOLD}${BLACK}######################################################${NC}"
        echo -e "${GREEN}${j}/${SIZE_OF_ARRAY} ${BOLD}${BLACK}Processing ${FILE}${NC}"

        # 1. Call SANLM denoising filter
        # ----------------------------------------------------------------------
        if [ "${use_sanlm}" -eq 1 ]; then
            echo -e "${BLUE}---------------------------------------------${NC}"
            echo -e "${BLUE}SANLM denoising${NC}"
            echo -e "${BLUE}---------------------------------------------${NC}"
            ${bin_dir}/CAT_VolSanlm "${FILE}" "${outmridir}/${sanlm}"
            input="${outmridir}/${sanlm}"
        else
            input="${FILE}"
        fi
        
        # 2. Call deepmriprep segmentation 
        # ----------------------------------------------------------------------
        # check for outputs from previous step
        if [ -f "${input}" ]; then
            echo -e "${BLUE}---------------------------------------------${NC}"
            echo -e "${BLUE}Deepmriprep segmentation${NC}"
            echo -e "${BLUE}---------------------------------------------${NC}"
                "${python}" "${cmd_dir}/deepmriprep_predict.py" --input "${input}" \
                    --outdir "${outmridir}"
        else
            echo -e "${RED}ERROR: CAT_VolSanlm failed${NC}"
            ((i++))
            continue
        fi
        
        # remove denoised image
        if [ "${use_sanlm}" -eq 1 ]; then
            [ -f "${outmridir}/${sanlm}" ] && rm "${outmridir}/${sanlm}"
        fi
        
        # optionally extract surface
        if [ "${estimate_surf}" -eq 1 ]; then
                
            # 3. Create hemispheric label maps for cortical surface extraction
            # ----------------------------------------------------------------------
            # check for outputs from previous step
            if [ ! -f "${outmridir}/${seg}" ]; then
                echo -e "${RED}ERROR: ${cmd_dir}/deepmriprep_predict.py failed${NC}"
                ((i++))
                continue
            fi
            
            # 4. Estimate thickness and percentage position maps for each hemisphere
            #    and extract cortical surface and call it as background process to
            # allow parallelization
            # ----------------------------------------------------------------------
            # check for outputs from previous step
            echo -e "${BLUE}---------------------------------------------${NC}"
            echo -e "${BLUE}Extracting surfaces${NC}"
            echo -e "${BLUE}---------------------------------------------${NC}"
            for side in left right; do
                surface_estimation $side $outmridir $outsurfdir $registration &
            done
            
            # use wait to check finishing the background processes
            wait
            
        fi # estimate_surf

        # remove temporary files if not debugging
        if [ "${debug}" -eq 0 ]; then
            [ -f "${outmridir}/${seg}" ] && rm "${outmridir}/${seg}"
            
            # only remove temporary files if surfaces exist
            if [ -f "${outsurfdir}/${mid_left}" ] && [ -f "${outsurfdir}/${mid_right}" ]; then
                [ -f "${outsurfdir}/${pbt_left}" ] && rm "${outsurfdir}/${pbt_left}"
                [ -f "${outmridir}/${hemi_left}" ] && rm "${outmridir}/${hemi_left}"
                [ -f "${outmridir}/${ppm_left}" ] && rm "${outmridir}/${ppm_left}"
                [ -f "${outmridir}/${gmt_left}" ] && rm "${outmridir}/${gmt_left}"
            fi
        fi

        # print execution time per data set
        end=$(date +%s)
        runtime=$((end - start))
        echo -e "${GREEN}---------------------------------------------${NC}"
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
                    [--no-sanlm] [--no-surf] [--nproc number_of_processes] 
                    [--bids] [--sub subsampling] [--debug] filenames 
 
  --python <FILE>            python command (default $python)
  --out-dir <DIR>            output folder (default same folder)
  --bin-dir <DIR>            folder of CAT binaries (default $bin_dir)
  --pre-fwhm  <NUMBER>       FWHM size of pre-smoothing in CAT_VolMarchingCubes (default $pre_fwhm). 
  --post-fwhm <NUMBER>       FWHM size of post-smoothing in CAT_VolMarchingCubes (default $post_fwhm). 
  --thresh    <NUMBER>       Threshold (isovalue) for creating surface in CAT_VolMarchingCubes (default $thresh). 
  --min-thickness <NUMBER>   Values below minimum thickness are set to zero and will be approximated
                             using the replace option in the vbdist method (default $min_thickness). 
  --median-filter <NUMBER>   Specify how many times to apply a median filter to areas with
                             topology artifacts to reduce these artifacts.
  --nproc <NUMBER>           number of parallel jobs (=number of processors)
  --sub <NUMBER>             subsampling for Amap segmentation (default $sub)
  --no-surf                  skip surface and thickness estimation
  --no-sanlm                 skip denoising with SANLM-filter
  --bids                     use BIDS naming of output files
  --debug                    keep temporary files for debugging
 
PURPOSE:
  Computational Anatomy Pipeline for structural MRI data 

EXAMPLE
  T1Prep.sh --outdir test_folder single_subj_T1.nii.
  This command will extract segmentation and surface maps for single_subj_T1.nii. 
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
  ${cmd_dir}/deepmriprep_predict.py
  
This script was written by Christian Gaser (christian.gaser@uni-jena.de).

__EOM__
}

########################################################
# call main program
########################################################

main "${@}"
    