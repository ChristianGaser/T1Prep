#! /bin/bash
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

########################################################
# check arguments
########################################################

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

########################################################
# check files
########################################################

check_files()
{
  SIZE_OF_ARRAY="${#ARRAY[@]}"
  if [ "$SIZE_OF_ARRAY" -eq 0 ]; then
    echo "${RED}ERROR: No files given!${NC}" >&2
    help
    exit 1
  fi
}

########################################################
# get operation system
########################################################

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

########################################################
# filter arguments so that filenames are removed 
########################################################

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

