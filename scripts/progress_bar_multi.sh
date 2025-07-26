#! /bin/bash
#
# PURPOSE: This script displays progress bars for each job.
#
# USAGE: progress_bar_multi.sh  n_Jobs [Width Color]
# Number of jobs.
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

# ----------------------------------------------------------------------
# global parameters
# ----------------------------------------------------------------------
n_jobs=$1 
PROGRESS_DIR=$2
width=$3
color=$4
BOLD=$(tput bold)
NC=$(tput sgr0) # Reset all attributes
REFRESH_INTERVAL=2.0
JOB_ID="jobrun_$(date +%s%N)"

# ----------------------------------------------------------------------
# run main
# ----------------------------------------------------------------------

main ()
{
  if [ -z "$n_jobs" ]; then
    help
  fi
  if [ -z "$PROGRESS_DIR" ]; then
    PROGRESS_DIR="/tmp/progress_bars/$JOB_ID"; 
  fi
  if [ -z "$color" ]; then color=2; fi
  if [ -z "$width" ]; then width=40; fi

  DEFAULT_COLOR=$(tput setaf $color)
  FAIL_COLOR=$(tput setaf 1)
  
  # Print empty lines for each job
  for ((i=0; i<n_jobs; i++)); do
    echo ""
  done
  
  while true; do
    tput cuu "$n_jobs"  # Move cursor up N lines
    all_done=true
    for ((i=0; i<n_jobs; i++)); do
      file="$PROGRESS_DIR/job${i}.progress"
      if [[ -f "$file" ]]; then
        progress=$(cat "$file")
        done_items=$(echo "$progress" | cut -d/ -f1)
        total_items=$(echo "$progress" | cut -d/ -f2)
        percent=$((100 * done_items / total_items))
        filled=$((width * done_items / total_items))
        unfilled=$((width - filled))
        bar=$(printf "%${filled}s" | tr ' ' '█')
        bar+=$(printf "%${unfilled}s")
        jobnumber=$((i+1))
        printf -v jobstr "%2d" "$jobnumber"

        # If process failed, display bar in red and decrease
        # refresh interval
        status_file="$PROGRESS_DIR/job${i}.status"
        status=0
        [[ -f "$status_file" ]] && status=$(cat "$status_file")
        if [[ "$status" -ne 0 ]]; then
          BAR_COLOR=$FAIL_COLOR
          REFRESH_INTERVAL=0.1
        else
          BAR_COLOR=$DEFAULT_COLOR
          REFRESH_INTERVAL=2.0
        fi

        echo -ne "Job ${jobstr}: [${BAR_COLOR}${bar}${NC}] (${percent}%)\n"
        [[ "$done_items" -lt "$total_items" ]] && all_done=false
      else
        echo -ne "Job $((i+1)): [waiting...]\n"
        all_done=false
      fi
    done
    $all_done && break
    sleep "$REFRESH_INTERVAL"
  done
}

# ----------------------------------------------------------------------
# help
# ----------------------------------------------------------------------
help ()
{
cat <<__EOM__

${BOLD}USAGE:${NC}
  $(basename "$0") n_jobs [job_dir width color]

${BOLD}OPTIONS:${NC}
  n_jobs  Specify the number of parallel jobs.
  job_dir Specify the temporary folder where the jobs are saved.
  width   Width of progress bar (default $width).
  color   Color code (default $color).
          (0-black, 1-red, 2-green, 3-yellow, 4-blue, 5-pink, 6-cyan, 7-white)
  Failed jobs are displayed in red regardless of the chosen color.

${BOLD}EXAMPLE:${NC}
  #! /bin/bash
  n_jobs=3
  PROGRESS_DIR="/tmp/progress_bars"
  mkdir -p "$PROGRESS_DIR"
  rm -f "$PROGRESS_DIR"/*.progress
  
  # Dummy work for each job
  COMMAND="sleep"  # Replace with your actual command
  
  # Arguments per job (simulate multiple tasks per job)
  ARG_LIST=(
    "1 2 3 4 5"
    "a b c"
    "x y z w"
  )
  
  # Start progress monitor
  $(dirname "$0")/progress_bar_multi.sh "$n_jobs" "$PROGRESS_DIR" &
  progress_pid=$!
  
  # Launch jobs
  for ((i=0; i<n_jobs; i++)); do
    (
      args=(${ARG_LIST[$i]})
      total=${#args[@]}
      for ((k=0; k<total; k++)); do
        echo "$((k+1))/$total" > "$PROGRESS_DIR/job${i}.progress"
        $COMMAND 0.5  # Simulate work
      done
    ) &
  done
  
  wait  # Wait for all jobs
  
  echo -e "\n✅ All jobs completed!"
  rm -r "$PROGRESS_DIR"

${BOLD}Author:${NC}
  Christian Gaser (christian.gaser@uni-jena.de)

__EOM__

}

# ----------------------------------------------------------------------
# call main program
# ----------------------------------------------------------------------

main ${1+"$@"}