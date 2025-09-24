#!/usr/bin/env bash
#
# PURPOSE: Move utility functions from T1Prep for better code structure
#
# FUNCTIONS:
# - exit_if_empty: Checks if a command line argument is empty and exits with an error message if it is.
# - check_python: Checks if the specified Python command is available.
# - check_python_module: Checks for python modules.
# - check_python_libraries: Checks for python libraries.
# - get_OS: Identifies operations system and folder of binaries.
# - get_pattern: Get the pattern from the desired column in namefile
# - substitute_pattern: Substitute variables in the pattern
# - check_files: Checks if the input files exist.
# - run_cmd_log: Run command and print output and execution time to report file
#
# ______________________________________________________________________
#
# Christian Gaser
# Structural Brain Mapping Group (https://neuro-jena.github.io)
# Departments of Neurology and Psychiatry
# Jena University Hospital
# ______________________________________________________________________

# defaults
os_type=$(uname -s) # Determine OS type
script_dir=$(dirname "$0")
root_dir=$(dirname $script_dir)
name_file=${root_dir}/Names.tsv
surf_templates_dir=${root_dir}/data/templates_surfaces_32k
atlas_templates_dir=${root_dir}/data/atlases_surfaces_32k
T1prep_env=${root_dir}/env
src_dir=${root_dir}/src

# Text formatting and colors with fallback
if [ -t 1 ] && [ -n "$TERM" ] && command -v tput >/dev/null 2>&1; then
  # tput available
  BOLD=$(tput bold)
  UNDERLINE=$(tput smul)
  NC=$(tput sgr0)

  BLACK=$(tput setaf 0)
  RED=$(tput setaf 1)
  GREEN=$(tput setaf 2)
  YELLOW=$(tput setaf 3)
  BLUE=$(tput setaf 4)
  PINK=$(tput setaf 5)
  CYAN=$(tput setaf 6)
  WHITE=$(tput setaf 7)
  GRAY=$(tput setaf 8)
  MAGENTA=$(tput setaf 13)
fi

progress_pid= # This will hold the PID of the progress bar monitor
pids=()

# indent, x spaces
function indent02() { sed 's/^/  /'; }
function indent04() { sed 's/^/    /'; }
function indent10() { sed 's/^/          /'; }
function indent15() { sed 's/^/               /'; }
function indent18() { sed 's/^/                  /'; }
function indent20() { sed 's/^/                    /'; }
function indent25() { sed 's/^/                         /'; }

# ----------------------------------------------------------------------
# check arguments
# ----------------------------------------------------------------------

exit_if_empty()
{
  local desc val

  desc="$1"
  shift
  val="$*"

  if [ -z "${val}" ]; then
    echo "${RED}ERROR: No argument given with \"$desc\" command line argument!${NC}" >&2
    exit 1
  fi
}

# ----------------------------------------------------------------------
# check files
# ----------------------------------------------------------------------

check_files()
{
  SIZE_OF_ARRAY="${1}"
  if [ "$SIZE_OF_ARRAY" -eq 0 ]; then
    echo "${RED}ERROR: No files given!${NC}" >&2
    help
    exit 1
  fi

  i=0
  while [ "$i" -lt "$SIZE_OF_ARRAY" ]; do
    if [[ ! "${ARRAY[$i]}" =~ \.nii(\.gz)?$ ]]; then
      echo "${RED}ERROR: File ${ARRAY[$i]} is not a valid NIfTI file${NC}" >&2
      help
      exit 1
    fi
    ((i++))
  done

}

# ----------------------------------------------------------------------
# get operation system
# ----------------------------------------------------------------------

get_OS() {
  local os_type cpu_arch
  os_type="$(uname -s)"
  cpu_arch="$(uname -m)"

  case "$os_type" in
    Linux*)
      case "$cpu_arch" in
        x86_64) bin_dir="${root_dir}/bin/Linux" ;;
        aarch64|arm64) bin_dir="${root_dir}/bin/LinuxARM64" ;;
        *) echo "Unsupported Linux arch: ${cpu_arch}" >&2; exit 1 ;;
      esac
      ;;
    Darwin*)
      case "$cpu_arch" in
        arm64) bin_dir="${root_dir}/bin/MacOS" ;;
        *) echo "macOS Intel not supported anymore" >&2; exit 1 ;;
      esac
      ;;
    CYGWIN*|MINGW*|MSYS*) bin_dir="${root_dir}/bin/Windows" ;;
    *) echo "Unknown OS: ${os_type}" >&2; exit 1 ;;
  esac

  export bin_dir
  export PATH="${bin_dir}:$PATH"
}

# ----------------------------------------------------------------------
# get # of processes w.r.t. available memory
# ----------------------------------------------------------------------

get_no_processes () {

  ARCH=`uname`
  MEM_LIMIT=$1
  
  if [ "$ARCH" == "Linux" ]; then
    # Get total installed memory in MB
    mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')

    # Convert KB to GB
    mem_gb=$(echo "scale=2; $mem_total / 1024 / 1024" | bc)

  elif [ "$ARCH" == "Darwin" ]; then
    # Get total installed memory in bytes
    mem_total=$(sysctl hw.memsize | awk '{print $2}')

    # Convert bytes to GB
    mem_gb=$(echo "scale=2; $mem_total / 1024 / 1024 / 1024" | bc)

  elif [ "$ARCH" == "CYGWIN_NT" ] || [ "$ARCH" == "MSYS_NT" ] || [ "$ARCH" == "MINGW32_NT" ] || [ "$ARCH" == "MINGW64_NT" ]; then
    # Get total installed memory in KB
    mem_total=$(wmic ComputerSystem get TotalPhysicalMemory | grep -Eo '[0-9]+')

    # Convert bytes to GB
    mem_gb=$(echo "$mem_total / 1024 / 1024 / 1024" | bc)
  else
    echo "${RED}System $ARCH not recognized${NC}"
    exit 1
  fi
    
  # Calculate number of processes (at least 1) w.r.t. to defined memory limit for each process
  NUM_JOBS=$(echo "$mem_gb / $MEM_LIMIT" | bc)
  if [ "$NUM_JOBS" -lt 1 ]; then
    NUM_JOBS=1
  fi
}

# ----------------------------------------------------------------------
# Run command and log function
# ----------------------------------------------------------------------

run_cmd_log() {
    local report_log="$1"
    shift
    local start_cmd end_cmd runtime

    start_cmd=$(date +%s)

    for cmd in "$@"; do
        echo "${cmd}" >> "${report_log}"
        eval "${cmd}" >> "${report_log}" 2>&1
    done

    end_cmd=$(date +%s)
    runtime=$((end_cmd - start_cmd))
    echo "Execution time: ${runtime}s" >> "${report_log}"
}

# ----------------------------------------------------------------------
# Cleanup function
# ----------------------------------------------------------------------

cleanup() {
  echo " Caught interrupt. Cleaning up..."
  echo ""
  echo "Please note that only new processes can be killed, but already started child processes hast to be maybe manually interrupted."
  if [ -n "$progress_pid" ]; then
    kill "$progress_pid" 2>/dev/null
  fi

  # Kill all background jobs started with nohup
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
    fi
  done

  wait $progress_pid
  rm -rf "$PROGRESS_DIR"
  exit 1
}

# ----------------------------------------------------------------------
# filter arguments so that filenames are removed 
# ----------------------------------------------------------------------

filter_arguments() {
  local args=("$@")   # All input arguments to filter
  local filtered=()   # Array to store filtered arguments
  local filepatterns=("${ARRAY[@]}")  # File patterns to exclude

  for arg in "${args[@]}"; do
    # Check if the argument is a file pattern by comparing it to ${ARRAY[@]}
    local is_filepattern=0
    for pattern in "${filepatterns[@]}"; do
      if [[ "$arg" == "$pattern" ]]; then
        is_filepattern=1
        break
      fi
    done

    # If it's not a file pattern, add it to the filtered list
    if [[ $is_filepattern -eq 0 ]]; then
      filtered+=("$arg")
    fi
  done

  echo "${filtered[@]}"  # Return the filtered arguments
}

# ----------------------------------------------------------------------
# Check for python version
# ----------------------------------------------------------------------

check_python_cmd()
{
  if [ -z "$python" ]; then
    if command -v python3.12 &>/dev/null; then
      python="python3.12"
    elif command -v python3.11 &>/dev/null; then
      python="python3.11"
    elif command -v python3.10 &>/dev/null; then
      python="python3.10"
    elif command -v python3.9 &>/dev/null; then
      python="python3.9"
    elif command -v python3 &>/dev/null; then
      python="python3"
    elif command -v python &>/dev/null; then
      python="python"
    else
      echo "${RED}Correct python version 3.9-3.12 was not found. Please use '--python' flag to define Python command and/or install Python${NC}" 2>&1
      exit 1
    fi
  fi
  
  python_version=$($python -V 2>&1)
  if ! echo "$python_version" | grep -qE '^Python 3\.(9|10|11|12)\.'; then
    echo "${RED}Only Python version 3.9-3.12 is supported. Please use '--python' flag to define Python command and/or install Python${NC}" 2>&1
    exit 1
  fi  
}

# ----------------------------------------------------------------------
# Check for python modules (e.g. pip)
# ----------------------------------------------------------------------

check_python_module() {
    ${python} -c "import $1" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Python module '$1' is not installed."
        echo "On Linux use 'apt install $(basename "$python")-"$1"'"
        exit 1
    fi
}

# ----------------------------------------------------------------------
# Logo
# ----------------------------------------------------------------------

logo() {
    local BLOCK_COLOR="$BLUE"    # colour for the █ glyphs
    local TEXT_COLOR="$GRAY"    # colour for every other character

    # ASCII art in one variable (can be here-doc or external file)
    local art='
████████╗ ██╗ ██████╗ ██████╗ ███████╗██████╗ 
╚══██╔══╝███║ ██╔══██╗██╔══██╗██╔════╝██╔══██╗
   ██║    ██║ ██████╔╝██████╔╝█████╗  ██████╔╝
   ██║    ██║ ██╔═══╝ ██╔══██╗██╔══╝  ██╔═══╝ 
   ██║    ██║ ██║     ██║  ██║███████╗██║     
   ╚═╝    ╚═╝ ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝   
'

    # Parameter substitution: colourise each █, keep rest in TEXT_COLOR
    art=${art//█/${BLOCK_COLOR}█${TEXT_COLOR}}

    # trim one trailing newline if present (so last-line extraction works)
    [[ ${art: -1} == $'\n' ]] && art="${art%$'\n'}"

    # split into head (everything up to last newline) and last line
    local last="${art##*$'\n'}"
    local head="${art%$last}"

    # append version to the last line (stay in TEXT_COLOR)
    last+="${TEXT_COLOR} version ${version}"

    # print
    printf "%b%s%b\n" "$TEXT_COLOR" "$head$last" "$NC"
}

# ----------------------------------------------------------------------
# Get the pattern from the desired column in name_file
# ----------------------------------------------------------------------

get_pattern() {
  local code="$1"
  local colnum="$2"
  awk -v c="$code" -v n="$colnum" '$1 == c {print $n}' ${name_file}
}

# ----------------------------------------------------------------------
# Substitute variables in the pattern
# ----------------------------------------------------------------------

substitute_pattern() {
  local pattern="$1"
  local hemi="$2"
  local desc="$3"
  local space="$4"
  local atlas_surf="$5"
  pattern="${pattern//\{bname\}/$bname}"
  pattern="${pattern//\{side\}/$hemi}"
  pattern="${pattern//\{space\}/$space}"
  pattern="${pattern//\{desc\}/$desc}"
  pattern="${pattern//\{atlas\}/$atlas_surf}"
  pattern="${pattern//\{nii_ext\}/$nii_ext}"
  echo $pattern
}

# ----------------------------------------------------------------------
# Check for python libraries
# ----------------------------------------------------------------------

check_python_libraries()
{  
  # Remove T1pre-env if reinstallation is selected
  [[ -d "${T1prep_env}" && "${re_install}" -eq 1 ]] &&  rm -r "${T1prep_env}"

  if [ ! -d ${T1prep_env} ]; then
    $python -m venv ${T1prep_env}
    install_deepmriprep
  fi

  source ${T1prep_env}/bin/activate
  
  # Check that installation was successful and try it a 2nd time otherwise
  $python -c "import deepmriprep" &>/dev/null
  if [ $? -gt 0 ]; then
    install_deepmriprep
  fi
}

# ----------------------------------------------------------------------
# Install deepmriprep
# ----------------------------------------------------------------------

install_deepmriprep()
{
  echo "Install deepmriprep"
  $python -m venv ${T1prep_env}
  source ${T1prep_env}/bin/activate

  $python -m pip install -U pip
  $python -m pip install -r ${root_dir}/requirements.txt
  
  $python -c "import deepmriprep" &>/dev/null
  if [ $? -gt 0 ]; then
    echo "${RED}ERROR: Installation of deepmriprep not successful. 
      Please install it manually${NC}" >&2
    exit 1
  fi
  
  # Allow executable on MacOS
  case "$os_type" in
    Darwin*)  
      # Remove quarantine for all files in folder
      if command -v xattr >/dev/null 2>&1; then
        xattr -dr com.apple.quarantine "$root_dir" 2>/dev/null || true
      else
        echo "xattr not found; skipping quarantine removal."
      fi
      ;;
  esac
}

# ----------------------------------------------------------------------
# Get output folder depending on BIDS structure
# ----------------------------------------------------------------------

get_output_folder()
{
  local FILE=$1
  bname=$(basename "$FILE")
  bname="${bname%.nii.gz}"
  bname="${bname%.nii}"

  local dname=$(dirname "$FILE")
  dname=$(cd "$dname" && pwd) # get absolute path
  
  # check for BIDS folder structure, where the upper folder is "anat"
  local upper_dname=$(basename "$dname") # get upper directory
  if [ "${upper_dname}" == "anat" ]; then
    use_subfolder=0
    sess_folder0=$(dirname "$dname")

    if [ ! $(echo $(basename "$sess_folder0") | grep "ses-") == "" ]; then
      sess_folder=$(basename "$sess_folder0")
      updir="../"
      subj_folder=$(dirname "$sess_folder0")
    else
      sess_folder=""
      updir=""
      subj_folder=$(dirname "$dname")
    fi

    subj_folder=$(basename "$subj_folder")
    bids_folder="${updir}../../../derivatives/T1Prep-v${version}/${subj_folder}/${sess_folder}/anat/"
  else
    use_subfolder=1
    bids_folder=""
  fi
  
  if [ -z "${outdir}" ]; then
    outdir0=${dname}${bids_folder}
  else
    outdir0=${outdir}${bids_folder}
  fi
} 


