#!/usr/bin/env bash
#
# PURPOSE: Define text colors
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

# Text formatting
BOLD=$(tput bold)
UNDERLINE=$(tput smul)
NC=$(tput sgr0) # Reset all attributes

# Colors
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
  SIZE_OF_ARRAY="${#ARRAY[@]}"
  if [ "$SIZE_OF_ARRAY" -eq 0 ]; then
    echo "${RED}ERROR: No files given!${NC}" >&2
    help
    exit 1
  fi
}

# ----------------------------------------------------------------------
# get operation system
# ----------------------------------------------------------------------

get_OS() 
{
  # Determine CPU architecture
  cpu_arch=$(uname -m)

  case "$os_type" in
    Linux*)   
      bin_dir="${root_dir}/bin/Linux"
      ;;
    Darwin*)  
      if [[ "$cpu_arch" == arm64 ]]; then 
        bin_dir="${root_dir}/bin/MacOS"
      else 
        echo "MacOS Intel not supported anymore"
        exit 1
      fi
      ;;
    CYGWIN*|MINGW32*|MSYS*|MINGW*) 
      bin_dir="${root_dir}/bin/Windows"
      ;;
    *)
      echo "Unknown OS system"
      exit 1
      ;;
  esac
  
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
# Prepare binary folder for MacOS function
# ----------------------------------------------------------------------

prepare_MacOS_bin_folder() {
  if [ "$(uname)" != "Darwin" ]; then
    return 0
  fi

  local bin_path="$1"
  if [ -z "$bin_path" ] || [ ! -d "$bin_path" ]; then
    echo "Usage: prepare_bin_folder <folder-with-binaries>"
    return 2
  fi
  
  # Remove qurantine for all files in folder
  if command -v xattr >/dev/null 2>&1; then
    xattr -dr com.apple.quarantine "$bin_path" 2>/dev/null || true
  else
    echo "xattr not found; skipping quarantine removal."
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

    # If it's not a file pattern, add it to the filtered list
    if [[ $is_filepattern -eq 0 ]]; then
      filtered+=("$arg")
    fi
  done

  echo "${filtered[@]}"  # Return the filtered arguments
}

