#!/usr/bin/env bash
#
# PURPOSE: This script displays a progress bar.
#
# USAGE: progress_bar Current Total [Name Width Color]
# Current length.
# Total length.
# Name of process (default Progress).
# Width of bar (default 40).
# Color code (default 0-black).
# (0-black, 1-red, 2-green, 3-yellow, 4-blue, 5-pink, 6-cyan, 7-white)
# ______________________________________________________________________
#
# Christian Gaser
# Structural Brain Mapping Group (https://neuro-jena.github.io)
# Departments of Neurology and Psychiatry
# Jena University Hospital
# ______________________________________________________________________

progress_bar() 
{
  # Usage: progress_bar Current Total Name Width Color
  if [ "$#" -lt 2 ]; then
    echo "Usage: $0 Current Total Name Width Color" >&2
    exit 1
  fi
  local current=$1 total=$2 label=$3 width=$4 failed=$5
  if [ -z "$label" ]; then label="Progress"; fi
  if [ -z "$width" ]; then width=40; fi
  if [ -z "$failed" ]; then failed=0; fi

  # Check both variables for being integers (positive or negative)
  if [[ ! "$current" == ^-?[0-9]+$ ]] || [[ ! "$total" =~ ^-?[0-9]+$ ]]; then
    current=10
    total=10
    failed=1
  fi
  
  # Text formatting & colors with fallback
  if [ -t 1 ] && [ -n "$TERM" ] && command -v tput >/dev/null 2>&1; then  
    # Colors
    NC=$(tput sgr0) # Reset all attributes
    DEFAULT_COLOR=$(tput setaf 2)   # green
    FAIL_COLOR=$(tput setaf 1)      # red
  else  
    # ANSI fallback
    NC='\033[0m'
    DEFAULT_COLOR='\033[0;32m'  # green
    FAIL_COLOR='\033[0;31m'     # red
  fi

  local percent=$(( current * 100 / total ))
  local percent=$(( current * 100 / total ))
  local filled=$(( width * current / total ))
  local empty=$(( width - filled ))
  local size=${#label} 

  if [ "$failed" -eq 1 ]; then
    COLOR=${DEFAULT_COLOR}
  else
    COLOR=${FAIL_COLOR}
  fi

  local bar=$(printf "%${filled}s" "" | awk '{gsub(/ /,"█"); print}')
  bar+=$(printf "%${empty}s")

  printf "[${COLOR}%-${width}s${NC}] %3d%% %s\r" "$bar" "$percent" "$label"

  if [ "$current" -eq "$total" ]; then
    printf -v padded_name "%-$(( width + size + 6))s" " "
    printf '%s\r' "${padded_name}"
  fi
}
