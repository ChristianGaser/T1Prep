#!/usr/bin/env bash
#
# PURPOSE: Move utility functions from T1Prep for better code structure
#
# FUNCTIONS:
# - exit_if_empty: Checks if a command line argument is empty and exits with an error message if it is.
# - check_python: Checks if the specified Python command is available.
# - check_python_module: Checks for python modules.
# - check_python_libraries: Checks for python libraries.
# - get_pattern: Get the pattern from the desired column in namefile
# - substitute_pattern: Substitute variables in the pattern
# - check_files: Checks if the input files exist.
# - filter_arguments: Filter arguments so that filenames are removed
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

# Directory of this T1Prep_utils.sh file (robust when sourced)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------
# Dual-mode path resolution
# ----------------------------------------------------------------------
# T1Prep_utils.sh works in two layouts:
#
#   1. Source-tree mode  — repo checkout; this script lives in <repo>/scripts/
#      and the package sits at <repo>/src/t1prep/.  A project-managed venv is
#      expected at <repo>/env/.
#
#   2. Installed mode    — `pip install T1Prep` placed this script in
#      <venv>/bin/ via setuptools `script-files`.  The package sits inside
#      site-packages; locate its data dir via importlib.resources using the
#      venv's python (sibling of this script).  The venv is already active
#      (or at least on PATH), so the venv-management functions are no-ops.
# ----------------------------------------------------------------------

if [ -d "${script_dir}/../src/t1prep" ]; then
  T1PREP_INSTALLED=0
  root_dir="$(cd "${script_dir}/.." && pwd)"
  src_dir=${root_dir}/src/t1prep
  data_dir=${src_dir}/data
  T1prep_env=${root_dir}/env
else
  T1PREP_INSTALLED=1
  # Prefer the venv's python sitting next to this script
  if [ -x "${script_dir}/python" ]; then
    python="${python:-${script_dir}/python}"
  elif [ -x "${script_dir}/python3" ]; then
    python="${python:-${script_dir}/python3}"
  else
    python="${python:-$(command -v python3 || command -v python)}"
  fi
  data_dir="$("${python}" -c 'from importlib.resources import files; print(files("t1prep").joinpath("data"))' 2>/dev/null)"
  if [ -z "${data_dir}" ] || [ ! -d "${data_dir}" ]; then
    echo "ERROR: cannot locate t1prep package data — is T1Prep installed in this Python environment?" >&2
    exit 1
  fi
  src_dir="$(dirname "${data_dir}")"   # site-packages/t1prep
  root_dir="${src_dir}"                # used only by a few legacy code paths
  T1prep_env=""                        # already inside a managed venv
fi

name_file=${data_dir}/Names.tsv
surf_templates_dir=${data_dir}/templates_surfaces_32k
atlas_templates_dir=${data_dir}/atlases_surfaces_32k

# Read T1Prep version from the Python package's __init__.py — the single
# source of truth (pyproject.toml already derives its version from
# t1prep.__version__ via setuptools attr).  Bump __version__ there and
# every shell consumer of T1PREP_VERSION picks it up automatically.
T1PREP_VERSION="$(awk -F'"' '/^__version__[[:space:]]*=/ {print $2; exit}' "${src_dir}/__init__.py" 2>/dev/null)"
T1PREP_VERSION=${T1PREP_VERSION:-unknown}
export T1PREP_VERSION

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
# get # of processes w.r.t. available memory
# ----------------------------------------------------------------------

get_no_processes () {

  ARCH=`uname`
  MEM_LIMIT=$1
  # Guard: ensure MEM_LIMIT is a positive integer to avoid division by zero
  if ! [[ "$MEM_LIMIT" =~ ^[0-9]+$ ]]; then
    MEM_LIMIT=12
  fi
  if [ "$MEM_LIMIT" -le 0 ]; then
    MEM_LIMIT=12
  fi
  
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
  # Use bc for robust division and guard against any unexpected zero values
  if [ "$MEM_LIMIT" -le 0 ]; then
    NUM_JOBS=1
  else
    NUM_JOBS=$(echo "$mem_gb / $MEM_LIMIT" | bc)
  fi
  if [ "$NUM_JOBS" -lt 1 ]; then
    NUM_JOBS=1
  fi
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
  # In installed mode the dual-mode block above has already set ``python`` to
  # the venv's interpreter — nothing to discover or validate.
  if [ "${T1PREP_INSTALLED:-0}" -eq 1 ]; then
    return 0
  fi

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
    last+="${TEXT_COLOR} version ${T1PREP_VERSION}"

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
  local bname_clean="${bname%_T1w}"
  pattern="${pattern//\{bname\}/$bname_clean}"
  pattern="${pattern//\{side\}/$hemi}"
  pattern="${pattern//\{space\}/$space}"
  pattern="${pattern//\{desc\}/$desc}"
  pattern="${pattern//\{atlas\}/$atlas_surf}"
  pattern="${pattern//\{nii_ext\}/$nii_ext}"
  pattern="${pattern//\../.}"
  echo $pattern
}

# ----------------------------------------------------------------------
# Verify that the selected Python can create a venv (ensurepip available).
# Prints a precise install hint and exits if not.
# ----------------------------------------------------------------------

check_venv_prerequisites()
{
  if $python -m ensurepip --version &>/dev/null; then
    return 0
  fi

  # Determine the major.minor version for the precise package name
  local py_ver
  py_ver="$($python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'  2>/dev/null || echo "")"

  echo "${RED}Python venv/ensurepip is not available for ${python:-python}.${NC}" >&2
  if [ -n "$py_ver" ]; then
    echo "On Debian/Ubuntu run:  sudo apt-get install python${py_ver}-venv" >&2
  else
    echo "On Debian/Ubuntu run:  sudo apt-get install python3-venv" >&2
  fi
  echo "Then re-run the installer." >&2
  exit 1
}

# ----------------------------------------------------------------------
# Check for python libraries
# ----------------------------------------------------------------------

check_python_libraries()
{
  # In installed mode the user already has a managed venv with T1Prep on it
  # (otherwise we wouldn't be running).  Recreating it from here would clobber
  # their environment.  Quietly accept --install / --re-install as no-ops.
  if [ "${T1PREP_INSTALLED:-0}" -eq 1 ]; then
    if [ "${re_install:-0}" -eq 1 ]; then
      echo "${YELLOW}--install/--re-install ignored: T1Prep was installed via pip; reinstall with 'pip install --force-reinstall T1Prep' if needed.${NC}"
    fi
    return 0
  fi

  # Remove T1pre-env if reinstallation is selected
  [[ -d "${T1prep_env}" && "${re_install}" -eq 1 ]] &&  rm -r "${T1prep_env}"

  # Repair broken venv python symlinks (e.g. env synced via Dropbox across systems)
  if [[ -d "${T1prep_env}" ]] && ! "${T1prep_env}/bin/python" -c "pass" &>/dev/null; then
    repair_venv
  fi

  if [ ! -d ${T1prep_env} ]; then
    check_venv_prerequisites
    if ! $python -m venv ${T1prep_env}; then
      echo "${RED}Failed to create Python virtual environment at ${T1prep_env}.${NC}" >&2
      exit 1
    fi
    install_deepmriprep
  fi

  if [ ! -f "${T1prep_env}/bin/activate" ]; then
    echo "${RED}Virtual environment at ${T1prep_env} is missing the activate script.${NC}" >&2
    exit 1
  fi

  source ${T1prep_env}/bin/activate
  
  # Check that installation was successful and try it a 2nd time otherwise
  $python -c "import deepmriprep" &>/dev/null
  if [ $? -gt 0 ]; then
    install_deepmriprep
  fi
}

# ----------------------------------------------------------------------
# Repair a broken venv by re-linking python to the current system interpreter.
# If the Python major.minor version changed, recreate the venv instead.
# ----------------------------------------------------------------------

repair_venv()
{
  local real_python
  real_python="$(command -v "$python")"
  if [[ -z "$real_python" ]]; then
    echo "${RED}Cannot find system python '${python}' to repair venv.${NC}" >&2
    return 1
  fi

  # Resolve to absolute path
  real_python="$(cd "$(dirname "$real_python")" && pwd)/$(basename "$real_python")"

  # Determine the version of the system python
  local sys_ver
  sys_ver="$("$real_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

  # Determine the version the venv was built with (from pyvenv.cfg)
  local venv_ver
  venv_ver="$(sed -n 's/^version = \([0-9]*\.[0-9]*\).*/\1/p' "${T1prep_env}/pyvenv.cfg" 2>/dev/null || true)"

  if [[ "$sys_ver" != "$venv_ver" ]]; then
    echo "${YELLOW}Python version changed (${venv_ver} -> ${sys_ver}). Recreating environment...${NC}"
    rm -rf "${T1prep_env}"
    return
  fi

  echo "${YELLOW}Repairing environment symlinks for this system...${NC}"
  local bin_dir="${T1prep_env}/bin"
  local real_dir
  real_dir="$(dirname "$real_python")"

  # Fix the versioned symlink (e.g. python3.9 -> system python)
  ln -sf "$real_python" "${bin_dir}/python${sys_ver}"
  # Fix python3 and python -> the versioned symlink
  ln -sf "python${sys_ver}" "${bin_dir}/python3"
  ln -sf "python${sys_ver}" "${bin_dir}/python"

  # Update pyvenv.cfg home to the new interpreter directory
  if [[ -f "${T1prep_env}/pyvenv.cfg" ]]; then
    sed -i.bak "s|^home = .*|home = ${real_dir}|" "${T1prep_env}/pyvenv.cfg"
    rm -f "${T1prep_env}/pyvenv.cfg.bak"
  fi

  # Verify the repair worked
  if ! "${bin_dir}/python" -c "pass" &>/dev/null; then
    echo "${YELLOW}Repair failed. Recreating environment...${NC}"
    rm -rf "${T1prep_env}"
  fi
}

# ----------------------------------------------------------------------
# Install deepmriprep
# ----------------------------------------------------------------------

install_deepmriprep()
{
  echo "Install deepmriprep"
  check_venv_prerequisites
  if ! $python -m venv ${T1prep_env}; then
    echo "${RED}Failed to create Python virtual environment at ${T1prep_env}.${NC}" >&2
    exit 1
  fi
  if [ ! -f "${T1prep_env}/bin/activate" ]; then
    echo "${RED}Virtual environment at ${T1prep_env} is missing the activate script.${NC}" >&2
    exit 1
  fi
  source ${T1prep_env}/bin/activate

  # PYTHONIOENCODING=utf-8: prevents a Python 3.9 crash when sys.stdout.encoding
  # is None (no locale set) inside pip's error handler.
  # --no-compile: skips bytecode compilation during install, avoiding pip crashes
  # on template files (e.g. PySide6's __init__.tmpl.py) that contain Jinja2
  # syntax and are not valid Python.
  PYTHONIOENCODING=utf-8 $python -m pip install -U pip
  PYTHONIOENCODING=utf-8 $python -m pip install --no-compile -r ${root_dir}/requirements.txt

  $python -c "import deepmriprep" &>/dev/null
  if [ $? -gt 0 ]; then
    echo "${RED}ERROR: Installation of deepmriprep not successful.
      Please install it manually${NC}" >&2
    exit 1
  fi

  # Install the T1Prep package itself so all entry points land in <venv>/bin:
  # the bash 'T1Prep' orchestrator (via setuptools script-files) plus the
  # t1prep-run / t1prep-ui / cat-viewsurf / t1prep-download-models console
  # scripts.  --no-deps because requirements.txt above already installed (and
  # pinned) every dependency.  This makes the environment's bin/ the single
  # directory to put on PATH.
  PYTHONIOENCODING=utf-8 $python -m pip install --no-compile --no-deps "${root_dir}"

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

t1prep_output_folder_from_input()
{
  # Compute output folder (outdir0), use_subfolder, and bname from an input file.
  # This is the shared logic used by both scripts/T1Prep and scripts/process_longitudinal.sh.
  #
  # Arguments:
  #   $1: input NIfTI file path
  #   $2: override out-dir ("" means: use dataset-root for BIDS, or input folder for non-BIDS)
  #   $3: T1Prep version string (e.g. "0.2.4")
  #   $4: use_amap flag (0/1)
  #
  # Outputs (globals, for backward compatibility with existing scripts):
  #   outdir0, use_subfolder, bname
  local FILE="$1"
  local outdir_override="${2:-}"
  local version_arg="${3:-${T1PREP_VERSION:-}}"
  local use_amap_arg="${4:-0}"

  # Keep folder naming stable: T1Prep-v<version> and avoid accidental "vv".
  version_arg="${version_arg#v}"

  # Normalize outdir override to an absolute path so that later `cd` calls
  # (e.g. in surface estimation) do not break relative paths.
  if [ -n "${outdir_override}" ] && [[ "${outdir_override}" != /* ]]; then
    outdir_override="$(pwd)/${outdir_override}"
  fi

  bname=$(basename "$FILE")
  bname="${bname%.nii.gz}"
  bname="${bname%.nii}"

  local add_str=""
  if [ "${use_amap_arg}" -eq 1 ]; then
    add_str="Amap"
  fi

  local dname
  dname=$(dirname "$FILE")
  dname=$(cd "$dname" && pwd) # absolute directory of input file

  # Detect BIDS structure if parent is 'anat'
  local upper_dname
  upper_dname=$(basename "$dname")
  if [ "${upper_dname}" = "anat" ]; then
    use_subfolder=0
    local sess_folder=""
    local subj_dir=""
    local dataset_root=""

    local sess_folder0
    sess_folder0=$(dirname "$dname")
    local sess_base
    sess_base=$(basename "$sess_folder0")
    if [[ "$sess_base" == ses-* ]]; then
      sess_folder="$sess_base"
      subj_dir=$(dirname "$sess_folder0")
    else
      subj_dir=$(dirname "$dname")
      sess_folder=""
    fi
    local subj_base
    subj_base=$(basename "$subj_dir")
    dataset_root=$(dirname "$subj_dir")

    local base_dir
    if [ -z "${outdir_override}" ]; then
      base_dir="${dataset_root}"
    else
      base_dir="${outdir_override}"
    fi

    if [ -n "$sess_folder" ]; then
      outdir0="${base_dir}/derivatives/T1Prep${add_str}-v${version_arg}/${subj_base}/${sess_folder}/anat"
    else
      outdir0="${base_dir}/derivatives/T1Prep${add_str}-v${version_arg}/${subj_base}/anat"
    fi
  else
    use_subfolder=1
    if [ -z "${outdir_override}" ]; then
      outdir0="${dname}"
    else
      outdir0="${outdir_override}"
    fi
  fi
}

get_output_folder()
{
  local FILE=$1
  t1prep_output_folder_from_input "$FILE" "${outdir:-}" "${T1PREP_VERSION}" "${use_amap}"
} 

_send_sentry_event()
{
  # Usage: _send_sentry_event <host> <project_id> <dsn> <level> <message> <n_success> <n_errors>
  # Sends a single Sentry event envelope (feeds issues-timeseries and release tracking).
  local host="$1" project_id="$2" dsn="$3" level="$4" message="$5"
  local n_success="$6" n_errors="$7"

  local EVENT_ID
  EVENT_ID=$(uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '-')
  local SENT_AT
  SENT_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  local event
  event=$(printf \
    '{"event_id":"%s","timestamp":"%s","platform":"other","release":"%s","level":"%s","logger":"t1prep","message":"%s","tags":[["version","%s"],["system","%s"]],"extra":{"n_success":%d,"n_errors":%d}}' \
    "${EVENT_ID}" "${SENT_AT}" "${T1PREP_VERSION}" "${level}" "${message}" \
    "${T1PREP_VERSION}" "${cpu_arch}" "${n_success}" "${n_errors}")

  local event_len=${#event}

  printf '{"event_id":"%s","dsn":"%s","sent_at":"%s"}\n{"type":"event","length":%d}\n%s' \
    "${EVENT_ID}" "${dsn}" "${SENT_AT}" "${event_len}" "${event}" \
    | curl -sS -o /dev/null -X POST "https://${host}/api/${project_id}/envelope/" \
        -H "Content-Type: application/x-sentry-envelope" \
        --data-binary @- 2>/dev/null &
}

send_sentry()
{
  # Usage: send_sentry <n_success> <n_errors>
  # Sends Sentry events so they appear in the Sentry issues-timeseries dashboard
  # with per-release/version breakdowns:
  #   info event  – one per invocation (total calls + processed count)
  #   error event – only when n_errors > 0 (feeds "Errors Over Time" widget)
  local n_success="${1:-0}"
  local n_errors="${2:-0}"

  local DSN="https://ca6089ed5dc4326c6c69afc3684c5fc1@o4511309449068544.ingest.de.sentry.io/4511309454311504"
  local HOST
  HOST=$(echo "$DSN" | sed -E 's#https://[^@]+@([^/]+)/.*#\1#')
  local PROJECT_ID
  PROJECT_ID=$(echo "$DSN" | sed -E 's#.*/([0-9]+)$#\1#')

  # Always send a summary info event (counts as a "Total Event" in the dashboard)
  _send_sentry_event "${HOST}" "${PROJECT_ID}" "${DSN}" \
    "info" "T1Prep run: ${n_success} succeeded, ${n_errors} failed" \
    "${n_success}" "${n_errors}"

  # Send an error-level event only when there were failures
  if [ "${n_errors}" -gt 0 ]; then
    _send_sentry_event "${HOST}" "${PROJECT_ID}" "${DSN}" \
      "error" "T1Prep: ${n_errors} file(s) failed to process" \
      "${n_success}" "${n_errors}"
  fi
}
