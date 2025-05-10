#! /bin/bash
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

########################################################
# progress bar
########################################################
progress_bar() 
{
  # Usage: progress_bar Current Total Name Width Color
  if [ "$#" -lt 2 ]; then
    echo "Usage: $0 Current Total Name Width Color" >&2
    exit 1
  fi
  local current=$1 total=$2 label=$3 width=$4 color=$5
  if [ -z "$label" ]; then label="Progress"; fi
  if [ -z "$color" ]; then color=0; fi
  if [ -z "$width" ]; then width=40; fi
  local percent=$(( current * 100 / total ))
  local filled=$(( width * current / total ))
  local empty=$(( width - filled ))
  local NC=$(tput sgr0) # Reset all attributes
  local COLOR=$(tput setaf $color)
  local size=${#label} 

  local bar=$(printf "%${filled}s" | tr ' ' '█')
  bar+=$(printf "%${empty}s")

  printf "${COLOR}%-${width}s %3d%%${NC} %s\r" "$bar" "$percent" "$label"

  if [ "$current" -eq "$total" ]; then
    printf -v padded_name "%-$(( width + size + 6))s" " "
    printf '%s\r' "${padded_name}"
  fi
}
