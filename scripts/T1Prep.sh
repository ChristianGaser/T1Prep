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
# - check_python_libraries: Checks python installation.
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
T1prep_env=${T1prep_dir}/T1prep-env
os_type=$(uname -s) # Determine OS type
outsurfdir=''
outmridir=''
use_bids_naming=0
thickness_fwhm=1
median_filter=2
save_hemi=0
save_surf=1
save_csf=0
save_mwp=1
save_wp=0
save_rp=0
save_p=0
min_thickness=1
registration=1
post_fwhm=1
pre_fwhm=0
re_install=0
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
    check_python_libraries    
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
            --re-install)
                re_install=1
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
            --hemi*)
                save_hemi=1
                ;;
            --no-surf)
                save_surf=0
                ;;
            --no-mwp*)
                save_mwp=0
                ;;
            --wp*)
                save_wp=1
                ;;
            --rp*)
                save_rp=1
                ;;
            --p*)
                save_p=1
                ;;
            --csf*)
                save_csf=1
                ;;
            --amap)
                use_amap=1
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
        echo "${RED}python or python3 not found. Please use '--python' flag to define Python command and/or install Python${NC}"
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
# check for python libraries
########################################################

check_python_libraries ()
{
    # Remove T1pre-env if reinstallation is selected
    if [[ -d "${T1prep_env}" && "${re_install}" == "1" ]]; then
        rm -r "${T1prep_env}"
    fi

    if [ -d ${T1prep_env} ]; then
        $python -m venv ${T1prep_env}
        source ${T1prep_env}/bin/activate
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
    $python -m venv ${T1prep_env}
    source ${T1prep_env}/bin/activate
    $python -m pip install -U pip
    $python -m pip install scipy==1.13.1 torch deepbet torchreg requests SplineSmooth3D nxbc deepmriprep
    
    $python -c "import deepmriprep" &>/dev/null
    if [ $? -gt 0 ]; then
        echo "${RED}ERROR: Installation of deepmriprep not successful. 
            Please install it manually${NC}"
        exit 1
    fi
    
    # Allow executable on MacOS
    case "$os_type" in
        Darwin*)    
            find MacOS -name "${bin_dir}/CAT*" -exec xattr -d com.apple.quarantine {} \;
            ;;
    esac
}

########################################################
# get OS
########################################################

get_OS () {

    
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

    # Pad the name to 50 characters (using printf)
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

    # This section of the script checks if a specific hemisphere file exists within a given directory.
    if [ -f "${outmridir}/${!hemi}" ]; then
    
        # Progress indication
        bar 2 $end_count "Calculate $side thickness"
    
        # Executes the 'CAT_VolThicknessPbt' tool with various options:
        # - ${verbose} toggles verbose output.
        # - '-n-avgs 4' sets the number of averages for distance estimation.
        # - '-min-thickness' sets a minimum thickness threshold.
        # - '-fwhm' sets the FWHM for thickness smoothing.
        ${bin_dir}/CAT_VolThicknessPbt ${verbose} -n-avgs 4 -min-thickness ${min_thickness} -fwhm ${thickness_fwhm} ${outmridir}/${!hemi} ${outmridir}/${!gmt} ${outmridir}/${!ppm}
        
        # Updates progress to 'Extract $side surface'.
        bar 4 $end_count "Extract $side surface"
    
        # Executes the 'CAT_VolMarchingCubes' tool to generate a surface mesh from volumetric data:
        # - '-median-filter' applies a median filter a specified number of times to reduce artifacts.
        # - '-pre-fwhm' and '-post-fwhm' control pre and post smoothing.
        # - '-thresh' sets the isovalue for mesh generation.
        # - '-no-distopen' disables distance-based opening, a form of morphological operation.
        # - '-local-smoothing' applies additional local smoothing to the mesh.
        ${bin_dir}/CAT_VolMarchingCubes ${verbose} -median-filter ${median_filter} -pre-fwhm ${pre_fwhm} -post-fwhm ${post_fwhm} -thresh ${thresh} -no-distopen -local-smoothing 10 ${outmridir}/${!ppm} ${outsurfdir}/${!mid}
    
        # Updates progress to 'Map $side thickness values'.
        bar 6 $end_count "Map $side thickness values"
    
        # Executes 'CAT_3dVol2Surf' to map volumetric data to a surface representation.
        # It uses a weighted average approach for mapping, ranging from -0.4..0.4 of the relative thickness using 5 steps
        ${bin_dir}/CAT_3dVol2Surf -weighted_avg -start -0.4 -steps 5 -end 0.4 ${outsurfdir}/${!mid} ${outmridir}/${!gmt} ${outsurfdir}/${!pbt}
    
        # Executes 'CAT_SurfDistance' to correct and map thickness values from a volumetric dataset to a surface.
        # It uses the mean of the closest distance between both surfaces and vice versa (Tfs from Freesurfer).
        ${bin_dir}/CAT_SurfDistance -mean -position ${outmridir}/${!ppm} -thickness ${outsurfdir}/${!pbt} ${outsurfdir}/${!mid} ${outsurfdir}/${!thick}
    
        # Executes 'CAT_Central2Pial' twice to estimate both pial and white matter surfaces.
        ${bin_dir}/CAT_Central2Pial ${outsurfdir}/${!mid} ${outsurfdir}/${!thick} ${outsurfdir}/${!pial} 0.5
        ${bin_dir}/CAT_Central2Pial ${outsurfdir}/${!mid} ${outsurfdir}/${!thick} ${outsurfdir}/${!wm}  -0.5
    
        # If registration is enabled, additional steps for spherical inflation and registration are performed.
        if [ "${registration}" -eq 1 ]; then
            # Updates progress to 'Spherical inflation $side hemisphere'.
            bar 8 $end_count "Spherical inflation $side hemisphere"
            # Inflates the surface to a sphere with additional areal smoothing.
            ${bin_dir}/CAT_Surf2Sphere ${outsurfdir}/${!mid} ${outsurfdir}/${!sphere} 6
    
            # Updates progress to 'Spherical registration $side hemisphere'.
            bar 10 $end_count "Spherical registration $side hemisphere"
            # Warps the surface to align with a standard sphere template, using specific mapping steps and averaging options.
            ${bin_dir}/CAT_WarpSurf ${verbose} -steps 2 -avg -i ${outsurfdir}/${!mid} -is ${outsurfdir}/${!sphere} -t ${Fsavg} -ts ${Fsavgsphere} -ws ${outsurfdir}/${!spherereg}
        fi
    else
        echo -e "${RED}ERROR: ${python} ${cmd_dir}/segment.py failed${NC}"
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
    $python -m venv ${T1prep_env}
    source ${T1prep_env}/bin/activate
    python="${T1prep_env}/bin/python"

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
            bn=$(basename "$FILE" | cut -f1 -d'.')
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
            bn0=$(basename "$FILE" | cut -f1 -d'.')
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

            # Initialize command string with the Python script call
            cmd="${python} ${cmd_dir}/segment.py"
            
            # Append options conditionally
            [ "${use_amap}" -eq 1 ] && cmd+=" --amap --amapdir ${bin_dir}"
            [ "${save_mwp}" -eq 1 ] && cmd+=" --mwp"
            [ "${save_wp}" -eq 1 ] && cmd+=" --wp"
            [ "${save_rp}" -eq 1 ] && cmd+=" --rp"
            [ "${save_p}" -eq 1 ] && cmd+=" --p"
            [ "${use_bids_naming}" -eq 1 ] && cmd+=" --bids"
            [ "${save_surf}" -eq 1 ] || [ "${save_hemi}" -eq 1 ] && cmd+=" --surf"
            [ "${save_csf}" -eq 1 ] && cmd+=" --csf"     
            
            cmd+=" --input \"${input}\" --outdir \"${outmridir}\""
            
            # Execute the command
            [ "${save_mwp}" -eq 1 ] || [ "${save_hemi}" -eq 1 ] || [ "${save_wp}" -eq 1 ] || [ "${save_p}" -eq 1 ] || [ "${save_rp}" -eq 1 ] && eval $cmd
  
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
        if [ "${save_surf}" -eq 1 ]; then
                
            # 3. Create hemispheric label maps for cortical surface extraction
            # ----------------------------------------------------------------------
            # check for outputs from previous step
            if [ ! -f "${outmridir}/${p0}" ]; then
                echo -e "${RED}ERROR: ${cmd_dir}/segment.py failed${NC}"
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
            
        fi # save_surf

        # remove temporary files if not debugging
        if [ "${debug}" -eq 0 ]; then
            [ -f "${outmridir}/${seg}" ] && rm "${outmridir}/${seg}"
            
            # only remove temporary files if surfaces exist
            if  [ -f "${outsurfdir}/${mid_left}" ] && [ -f "${outsurfdir}/${mid_right}" ]; then
                if [ "${save_hemi}" -eq 1 ]; then
                    file_list='ppm gmt'
                else
                    file_list='hemi ppm gmt'
                fi
                for files in ${file_list}t; do
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

help() {
cat <<__EOM__
${BOLD}${BLUE}T1Prep Computational Anatomy Pipeline (PyCAT)
---------------------------------------------${NC}

${BOLD}USAGE:${NC}
  ${GREEN}T1Prep.sh [options] <filenames>${NC}

${BOLD}DESCRIPTION:${NC}
  Processes T1-weighted brain data for computational anatomy analysis, including segmentation and surface extraction.

${BOLD}OPTIONS:${NC}
  ${YELLOW}System:${NC}
    --re-install              Remove old installation and reinstall required Python libraries.
    --python <FILE>           Specify Python command to use (default: $python).
    --debug                   Enable verbose output and do not delete temporary files.

  ${YELLOW}Parameters:${NC}
    --out-dir <DIR>           Specify output directory (default: current folder).
    --amap                    Use AMAP segmentation instead of deepmriprep.
    --pre-fwhm <NUMBER>       Pre-smoothing FWHM size in CAT_VolMarchingCubes (default: $pre_fwhm).
    --post-fwhm <NUMBER>      Post-smoothing FWHM size in CAT_VolMarchingCubes (default: $post_fwhm).
    --thickness-fwhm <NUMBER> Volumetric thickness smoothing FWHM size in CAT_VolThicknessPbt (default: $thickness_fwhm).
    --thresh <NUMBER>         Isovalue threshold for surface creation in CAT_VolMarchingCubes (default: $thresh).
    --min-thickness <NUMBER>  Minimum thickness values (below this are set to zero) for vbdist method (default: $min_thickness).
    --median-filter <NUMBER>  Number of median filter applications to reduce topology artifacts.

  ${YELLOW}Save Options:${NC}
    --no-overwrite <STRING>   Prevent overwriting of existing results by defining filename pattern that will be checked.
    --no-surf                 Skip surface and thickness estimation.
    --no-mwp                  Skip estimation of modulated and warped segmentations.
    --hemisphere              Additionally save hemispheric partitions.
    --wp                      Additionally save warped segmentations.
    --rp                      Additionally save affine registered segmentations.
    --p                       Additionally save native segmentations.
    --csf                     Additionally save CSF segmentations (default: GM/WM only).
    --bids                    Adopt BIDS standard for output file naming.

${BOLD}EXAMPLES:${NC}
  ${BLUE}T1Prep.sh --out-dir test_folder single_subj_T1.nii${NC}
    Extract segmentation and surface maps for 'single_subj_T1.nii', saving results in 'test_folder'.

  ${BLUE}T1Prep.sh --no-surf single_subj_T1.nii${NC}
    Extract segmentation maps for 'single_subj_T1.nii', saving in the same folder.

${BOLD}PURPOSE:${NC}
  This script facilitates the analysis of T1-weighted brain images by providing tools for segmentation, surface mapping, and more.

${BOLD}INPUT:${NC}
  Accepts NIfTI files as input.

${BOLD}OUTPUT:${NC}
  Produces segmented images and surface extractions.

${BOLD}USED FUNCTIONS:${NC}
  - CAT_VolAmap
  - CAT_VolSanlm
  - CAT_VolThicknessPbt
  - CAT_VolMarchingCubes
  - CAT_3dVol2Surf
  - CAT_SurfDistance
  - CAT_Central2Pial
  - ${cmd_dir}/segment.py

${BOLD}Author:${NC}
  Christian Gaser (christian.gaser@uni-jena.de)

__EOM__

check_python
check_python_cmd

# Check hether local Python environment exists
if [ ! -d "${T1prep_env}" ]; then

    # Prompt the user with a Y/N question
    echo "${RED}${BOLD}Local Python environment "${T1prep_env}" not found.${NC}"
    echo "Do you want to install required Python libraries? (Y/N)"
    read -r response
    
    # Check if the user's answer is 'Y'
    case "$response" in
        [Yy]*)
            check_python_libraries
            ;;
    esac
fi

}

########################################################
# call main program
########################################################

main "${@}"
    