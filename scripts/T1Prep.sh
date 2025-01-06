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
# - check_install: Checks python installation.
# - get_OS: Identifies operations system and folder of binaries.
# - bar: Displays progress bar.
# - get_no_of_cpus: Determines the number of available CPUs.
# - process: Performs the preprocessing steps on each input file.

########################################################
# global parameters
########################################################

# output colors
UNDERLINE=$(tput smul)
YELLOW=$(tput setaf 3)
GREEN=$(tput setaf 2)
BLACK=$(tput setaf 0)
CYAN=$(tput setaf 6)
PINK=$(tput setaf 5)
BLUE=$(tput setaf 4)
RED=$(tput setaf 1)
BOLD=$(tput bold)
NC=$(tput sgr0)

# defaults
T1prep_dir=$(dirname $(dirname "$0"))
surf_templates_dir=${T1prep_dir}/templates_surfaces_32k
use_bids_naming=0
thickness_fwhm=8
median_filter=4
estimate_surf=1
estimate_mwp=1
estimate_wp=0
estimate_rp=0
estimate_p=0
min_thickness=1
registration=1 # currently skip spherical registration to save time
post_fwhm=2
pre_fwhm=4
do_install=0
use_sanlm=0
use_amap=0
thresh=0.5
debug=0

########################################################
# run main
########################################################

main ()
{
    logo
    check_python_cmd
    parse_args ${1+"$@"}
    
    check_python
    check_install
    check_files
    get_OS
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
            --install)
                do_install=1
                ;;
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
            --thickness-fwhm)
                exit_if_empty "$optname" "$optarg"
                thickness_fwhm=$optarg
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
            --no-overwrite* | -no*)
                exit_if_empty "$optname" "$optarg"
                no_overwrite=$optarg
                shift
                ;;
            --no-surf)
                estimate_surf=0
                ;;
            --no-mwp*)
                estimate_mwp=0
                ;;
            --wp*)
                estimate_wp=1
                ;;
            --rp*)
                estimate_rp=1
                ;;
            --p*)
                estimate_p=1
                ;;
            --amap)
                use_amap=1
                ;;
            --sanlm)
                use_sanlm=1
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
    if command -v python3 &>/dev/null; then
        python="python3"
    elif command -v python &>/dev/null; then
        python="python"
    else
        echo "python or python3 not found. Please use '--python' flag to define python command and/or install python"
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
    if ! command -v "${python}" &>/dev/null; then
        echo "${RED}ERROR: $python not found${NC}"
        exit 1
    fi
}

########################################################
# check for installation
########################################################

check_install ()
{
    if [ -d ${T1prep_dir}/T1prep-env ]; then
        $python -m venv ${T1prep_dir}/T1prep-env
        source ${T1prep_dir}/T1prep-env/bin/activate
        $python -m pip install -U pip &>/dev/null
        
        $python -c "import deepmriprep" &>/dev/null
        if [ $? -gt 0 ]; then
            install_deepmriprep
        fi
    else
        install_deepmriprep
    fi        
}

########################################################
# install deepmriprep
########################################################

install_deepmriprep ()
{
    echo "Install deepmriprep"
    $python -m venv ${T1prep_dir}/T1prep-env
    source ${T1prep_dir}/T1prep-env/bin/activate
    $python -m pip install -U pip
    $python -m pip install scipy==1.13.1 torch deepbet torchreg requests SplineSmooth3D nxbc deepmriprep
    
    $python -c "import deepmriprep" &>/dev/null
    if [ $? -gt 0 ]; then
        echo "${RED}ERROR: Installation of deepmriprep not successful. 
            Please install it manually${NC}"
        exit 1
    fi
}

########################################################
# get OS
########################################################

get_OS () {

    # Determine OS type
    os_type=$(uname -s)
    
    # Determine CPU architecture
    cpu_arch=$(uname -m)

    case "$os_type" in
        Linux*)     
            bin_dir="${T1prep_dir}/Linux"
            ;;
        Darwin*)    
            if [[ "$cpu_arch" == arm64 ]]; then 
                bin_dir="${T1prep_dir}/MacOS"
            else 
                echo "MacOS Intel not supported anymore"
                exit 1
            fi
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*) 
            bin_dir="${T1prep_dir}/Windows"
            ;;
        *)
            echo "Unknown OS system"
            exit 1
            ;;
    esac
    
}

########################################################
# progress bar
########################################################
bar() {
    # Usage: bar 1 100 Name
    #            ^--------- Elapsed Percentage (0-100).
    #               ^------ Total length in chars.
    #                   ^-- Name of process.
    ((it=$1*100/$2))
    ((elapsed=$1))

    # Create the bar with spaces.
    printf -v prog  "%${elapsed}s"
    printf -v total "%$(($2-elapsed))s"

    # Pad the name to 20 characters (using printf)
    printf -v padded_name "%-50s" "$3"

    #printf '%s %s\r' "${prog// /■}${total} ${it}%" "${3}"
    printf '%s %s\r' "${prog// /■}${total} ${elapsed}/${2}" "${padded_name}"
    
    # if end is reached print extra line
    if [ "${1}" -eq "${2}" ]; then
        printf -v padded_name "%-100s" " "
        #printf '%s\r' "${padded_name}"
    fi
}

########################################################
# surface estimation
########################################################
surface_estimation() {
  
    side=$1
    outmridir=$2
    outsurfdir=$3
    registration=$4
    ((side_count=$side))
    if [ "${registration}" -eq 1 ]; then
        ((end_count=10))
    else
        ((end_count=6))
    fi

    # create dynamic variables
    pbt=pbt_$side
    thick=thick_$side
    hemi=hemi_$side
    ppm=ppm_$side
    gmt=gmt_$side
    mid=mid_$side
    pial=pial_$side
    wm=wm_$side
    sphere=sphere_$side
    spherereg=spherereg_$side
    fshemi=$(echo "$side" | sed -e "s/left/lh/g" -e "s/right/rh/g")
    Fsavg=${surf_templates_dir}/${fshemi}.central.freesurfer.gii
    Fsavgsphere=${surf_templates_dir}/${fshemi}.sphere.freesurfer.gii
    
    if [ "${debug}" -eq 0 ]; then
        verbose=''
    else 
        verbose=' -verbose '
    fi

    
    if [ -f "${outmridir}/${!hemi}" ]; then
        bar 2 $end_count "Calculate $side thickness"
        ${bin_dir}/CAT_VolThicknessPbt ${verbose} -min-thickness ${min_thickness} -fwhm ${thickness_fwhm} ${outmridir}/${!hemi} ${outmridir}/${!gmt} ${outmridir}/${!ppm}
        
        # The pre-smoothing helps in preserving gyri and sulci by creating a weighted 
        # average between original and smoothed images based on their distance to 
        # the threshold (isovalue). A negative value will force masked smoothing, which may preserves gyri and sulci even better.
        # In contrast, post-smoothing aids in correcting the mesh in folded areas like gyri and sulci and removes the voxel-steps
        # that are visible after marching cubes. However, this also slightly changes the curve in we have to correct the isovalue:
        # -post-fwhm 1 -thresh 0.495
        # -post-fwhm 2 -thresh 0.490
        # -post-fwhm 3 -thresh 0.475
        bar 4 $end_count "Extract $side surface"
        ${bin_dir}/CAT_VolMarchingCubes ${verbose} -median-filter ${median_filter} -pre-fwhm ${pre_fwhm} -post-fwhm ${post_fwhm} -thresh ${thresh} -no-distopen ${outmridir}/${!ppm} ${outsurfdir}/${!mid}

        bar 6 $end_count "Map $side thickness values"
        ${bin_dir}/CAT_3dVol2Surf -weighted_avg -start -0.4 -steps 5 -end 0.4 ${outsurfdir}/${!mid} ${outmridir}/${!gmt} ${outsurfdir}/${!pbt}
        ${bin_dir}/CAT_SurfDistance -mean -position ${outmridir}/${!ppm} -thickness ${outsurfdir}/${!pbt} ${outsurfdir}/${!mid} ${outsurfdir}/${!thick}
        
        ${bin_dir}/CAT_Central2Pial -position ${outmridir}/${!ppm} ${outsurfdir}/${!mid} ${outsurfdir}/${!thick} ${outsurfdir}/${!pial} 0.5 &
        ${bin_dir}/CAT_Central2Pial -position ${outmridir}/${!ppm} ${outsurfdir}/${!mid} ${outsurfdir}/${!thick} ${outsurfdir}/${!wm}  -0.5 &
        wait
        if [ "${registration}" -eq 1 ]; then
            bar 8 $end_count "Spherical inflation $side hemisphere"
            ${bin_dir}/CAT_Surf2Sphere ${outsurfdir}/${!mid} ${outsurfdir}/${!sphere} 6
            bar 10 $end_count "Spherical registration $side hemisphere       "
            ${bin_dir}/CAT_WarpSurf CAT_WarpSurf ${verbose} -steps 2 -avg -i ${outsurfdir}/${!mid} -is ${outsurfdir}/${!sphere} -t ${Fsavg} -ts ${Fsavgsphere} -ws ${outsurfdir}/${!spherereg}
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

    # use defined environment
    $python -m venv ${T1prep_dir}/T1prep-env
    source ${T1prep_dir}/T1prep-env/bin/activate
    python="${T1prep_dir}/T1prep-env/bin/python"

    ((i=0))
    ((j=0))
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

        # check whether processed files exist if no-overwrite flag is used
        if [ -n "${no_overwrite}" ]; then
            dn=$(dirname "$FILE")
            bn=$(basename "$FILE" |cut -f1 -d'.')
            if [ ! -n "$outdir" ]; then
                outdir0=${dn}
            else
                outdir0=${outdir}
            fi
            processed=$(ls "${outdir0}/${no_overwrite}${bn}"* 2>/dev/null)
        fi

        if [ ! -n "${processed}" ]; then
            ARRAY2[$j]="$FILE"
            ((j++))
        else
            echo Skip processing of ${FILE}
        fi
        ((i++))
    done

    ((i=0))
    SIZE_OF_ARRAY="${#ARRAY2[@]}"

    while [ "$i" -lt "$SIZE_OF_ARRAY" ]; do
        
        # set starting time
        start=$(date +%s)

        FILE="${ARRAY2[$i]}"

        # get directory and basename and also consider ancient Analyze img files
        dn=$(dirname "$FILE")
        bn=$(basename "$FILE" | sed -e "s/.img/.nii/g")
        bn_gz=$(basename "$FILE" | sed -e "s/.img/.nii/g" -e "s/.gz//g")     
        
        # if defined use output dir otherwise use the folder of input files
        if [ ! -n "$outdir" ]; then
            outdir=${dn}
        fi

        # check again whether processed files exist if no-overwrite flag is used
        if [ -n "${no_overwrite}" ]; then
            bn0=$(basename "$FILE" |cut -f1 -d'.')
            processed=$(ls "${outdir}/${no_overwrite}${bn0}"* 2>/dev/null)
            echo $processed
            
            # Check if $processed is empty
            if [ ! -n $processed ]; then
                echo Skip processing of ${FILE}
                continue  # Skip to the next iteration of the loop
            fi
        fi
        
        # create outdir if not exists
        if [ ! -d "$outdir" ]; then
            mkdir -p "$outdir"
        fi

        if [ "${use_bids_naming}" -eq 1 ]; then
        
            echo -e "${RED}BIDS names for volumes not yet supported.${NC}"
            
            # get output names
            sanlm=$(echo "$bn"     | sed -e "s/.nii/_desc-sanlm.nii/g")
            
            # remove T1w|T2w from basename
            seg=$(echo "$bn_gz"  | sed -e "s/.nii/_seg.nii/g")            
            p0=$(echo p0"$bn_gz")
            
            hemi_left=$(echo "$seg"  | sed -e "s/.nii/_hemi-L.nii/g")
            hemi_right=$(echo "$seg" | sed -e "s/.nii/_hemi-R.nii/g")
            gmt_left=$(echo "$bn"    | sed -e "s/.nii/_hemi-L_thickness.nii/g")
            gmt_right=$(echo "$bn"   | sed -e "s/.nii/_hemi-R_thickness.nii/g")
            
            # for the following filenames we have to remove the potential .gz from name
            ppm_left=$(echo "$bn_gz"    | sed -e "s/.nii/_hemi-L_ppm.nii/g")
            ppm_right=$(echo "$bn_gz"   | sed -e "s/.nii/_hemi-R_ppm.nii/g")
            mid_left=$(echo "$bn_gz"    | sed -e "s/.nii/_hemi-L_midthickness.surf.gii/g")
            mid_right=$(echo "$bn_gz"   | sed -e "s/.nii/_hemi-R_midthickness.surf.gii/g")
            pial_left=$(echo "$bn_gz"    | sed -e "s/.nii/_hemi-L_pial.surf.gii/g")
            pial_right=$(echo "$bn_gz"   | sed -e "s/.nii/_hemi-R_pial.surf.gii/g")
            wm_left=$(echo "$bn_gz"    | sed -e "s/.nii/_hemi-L_wm.surf.gii/g")
            wm_right=$(echo "$bn_gz"   | sed -e "s/.nii/_hemi-R_wm.surf.gii/g")
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
            seg=$(echo "$bn_gz"  | sed -e "s/.nii/_seg.nii/g")
            p0=$(echo p0"$bn_gz")
                        
            hemi_left=$(echo "$seg"  | sed -e "s/.nii/_hemi-L.nii/g")
            hemi_right=$(echo "$seg" | sed -e "s/.nii/_hemi-R.nii/g")
            gmt_left=$(echo "$bn"    | sed -e "s/.nii/_hemi-L_thickness.nii/g")
            gmt_right=$(echo "$bn"   | sed -e "s/.nii/_hemi-R_thickness.nii/g")
            
            # for the following filenames we have to remove the potential .gz from name
            ppm_left=$(echo "$bn_gz"    | sed -e "s/.nii/_hemi-L_ppm.nii/g")
            ppm_right=$(echo "$bn_gz"   | sed -e "s/.nii/_hemi-R_ppm.nii/g")
            
            mid_left=$(echo "lh.central.${bn_gz}" | sed -e "s/.nii/.gii/g")
            mid_right=$(echo "rh.central.${bn_gz}" | sed -e "s/.nii/.gii/g")
            pial_left=$(echo "lh.pial.${bn_gz}" | sed -e "s/.nii/.gii/g")
            pial_right=$(echo "rh.pial.${bn_gz}" | sed -e "s/.nii/.gii/g")
            wm_left=$(echo "lh.white.${bn_gz}" | sed -e "s/.nii/.gii/g")
            wm_right=$(echo "rh.white.${bn_gz}" | sed -e "s/.nii/.gii/g")
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
        echo -e "${BOLD}${BLACK}-------------------------------------------------------${NC}"
        echo -e "${GREEN}${j}/${SIZE_OF_ARRAY} ${BOLD}${BLACK}Processing ${FILE}${NC}"

        # 1. Call SANLM denoising filter
        # ----------------------------------------------------------------------
        if [ "${use_sanlm}" -eq 1 ]; then
            #echo -e "${BLUE}---------------------------------------------${NC}"
            #echo -e "${BLUE}SANLM denoising${NC}"
            ${bin_dir}/CAT_VolSanlm "${FILE}" "${outmridir}/${sanlm}"
            input="${outmridir}/${sanlm}"
        else
            input="${FILE}"
        fi
        
        # 2. Call deepmriprep segmentation 
        # ----------------------------------------------------------------------
        # check for outputs from previous step
        if [ -f "${input}" ]; then
            #echo -e "${BLUE}---------------------------------------------${NC}"
            #echo -e "${BLUE}Segmentation${NC}"
            if [ "${use_amap}" -eq 1 ]; then
                amap=" --amap --amapdir ${bin_dir}"
            else amap=""
            fi
            if [ "${estimate_mwp}" -eq 1 ]; then
                mwp=" --mwp "
            else mwp=""
            fi
            if [ "${estimate_wp}" -eq 1 ]; then
                wp=" --wp "
            else wp=""
            fi
            if [ "${estimate_rp}" -eq 1 ]; then
                rp=" --rp "
            else rp=""
            fi
            if [ "${estimate_p}" -eq 1 ]; then
                p=" --p "
            else p=""
            fi
            if [ "${use_bids_naming}" -eq 1 ]; then
                bids=" --bids "
            else bids=""
            fi
            if [ "${estimate_surf}" -eq 1 ]; then
                surf=" --surf "
            else surf=""
            fi
            "${python}" "${cmd_dir}/deepmriprep_predict.py" ${amap} ${mwp} ${rp} \
                ${wp} ${p} ${surf} ${bids} --input "${input}" --outdir "${outmridir}"
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
            if [ ! -f "${outmridir}/${p0}" ]; then
                echo -e "${RED}ERROR: ${cmd_dir}/deepmriprep_predict.py failed${NC}"
                ((i++))
                continue
            fi
            
            # 4. Estimate thickness and percentage position maps for each hemisphere
            #    and extract cortical surface and call it as background process to
            # allow parallelization
            # ----------------------------------------------------------------------
            # check for outputs from previous step
            #echo -e "${BLUE}---------------------------------------------${NC}"
            #echo -e "${BLUE}Extracting surfaces${NC}"
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
            if  [ -f "${outsurfdir}/${mid_left}" ] && [ -f "${outsurfdir}/${mid_right}" ]; then
                for files in hemi ppm gmt; do
                    for side in left right; do
                        rm_file=${files}_${side}
                        [ -f "${outmridir}/${!rm_file}" ] && rm "${outmridir}/${!rm_file}"
                    done
                done
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
  T1Prep.sh [--python python_command] [--out-dir out_folder] [--bin-dir bin_folder] [--amap]
                    [--thickness-fwhm thickness_fwm] [--sanlm] [--no-surf] [--no-mwp] [--rp]
                    [--wp] [--p] [--pre-fwhm pre_fwhm] [--post-fwhm post_fwhm]  
                    [--bids] [--debug] filenames 
 
  --python <FILE>            python command (default $python)
  --out-dir <DIR>            output folder (default same folder)
  --pre-fwhm  <NUMBER>       FWHM size of pre-smoothing in CAT_VolMarchingCubes 
                             (default $pre_fwhm). 
  --post-fwhm <NUMBER>       FWHM size of post-smoothing in CAT_VolMarchingCubes 
                             (default $post_fwhm). 
  --thickness-fwhm <NUMBER>  FWHM size of volumetric thickness smoothing in CAT_VolThicknessPbt 
                             (default $thickness_fwhm). 
  --thresh    <NUMBER>       threshold (isovalue) for creating surface in CAT_VolMarchingCubes 
                             (default $thresh). 
  --min-thickness <NUMBER>   values below minimum thickness are set to zero and will be approximated
                             using the replace option in the vbdist method (default $min_thickness). 
  --median-filter <NUMBER>   specify how many times to apply a median filter to areas with
                             topology artifacts to reduce these artifacts.
  --no-overwrite <STRING>    do not overwrite existing results
  --no-surf                  skip surface and thickness estimation
  --no-mwp                   skip estimation of modulated and warped segmentations
  --wp                       additionally save warped segmentations
  --rp                       additionally save affine registered segmentations
  --p                        additionally save native segmentations
  --sanlm                    apply denoising with SANLM-filter
  --amap                     use segmentation from AMAP instead of deepmriprep
  --bids                     use BIDS naming of output files
  --debug                    keep temporary files for debugging
 
PURPOSE:
  Computational Anatomy Pipeline for structural MRI data 

EXAMPLE
  T1Prep.sh --out-dir test_folder single_subj_T1.nii.
  This command will extract segmentation and surface maps for single_subj_T1.nii. 
  Resuts will be saved in test_folder.

  T1Prep.sh --no-surf single_subj_T1.nii.
  This command will extract segmentation maps for single_subj_T1.nii in the same 
  folder.

INPUT:
  nifti files

OUTPUT:
  segmented images
  surface extractions

USED FUNCTIONS:
  CAT_VolAmap
  CAT_VolSanlm
  CAT_VolThicknessPbt
  CAT_VolMarchingCubes
  CAT_3dVol2Surf
  CAT_SurfDistance
  CAT_Central2Pial
  ${cmd_dir}/deepmriprep_predict.py
  
This script was written by Christian Gaser (christian.gaser@uni-jena.de).

__EOM__
}

########################################################
# call main program
########################################################

main "${@}"
    