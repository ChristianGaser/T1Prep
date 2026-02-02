#!/usr/bin/env bash
#
# install.sh — Download the latest T1Prep release and run installation
#
# This script:
# 1) Detects curl or wget
# 2) Queries GitHub for available releases of ChristianGaser/T1Prep
# 3) Prompts user to choose a release version
# 4) Prompts user to choose installation directory
# 5) Downloads the tarball, extracts to chosen directory
# 6) Locates the "T1Prep" launcher and runs `T1Prep --install`
#
# Env overrides:
#   REPO_OWNER  (default: ChristianGaser)
#   REPO_NAME   (default: T1prep)
#   T1PREP_INSTALL_DIR (skip interactive prompt, use this directory)
#   T1PREP_VERSION (skip interactive prompt, use this release version, e.g., "v1.0.0" or "latest")
#
# Requirements: bash, tar, curl or wget. jq is optional.

set -euo pipefail

REPO_OWNER="${REPO_OWNER:-ChristianGaser}"
REPO_NAME="${REPO_NAME:-T1Prep}"

API_URL_LATEST="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/latest"
API_URL_RELEASES="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases"
FALLBACK_TARBALL="https://github.com/${REPO_OWNER}/${REPO_NAME}/archive/refs/heads/main.tar.gz"

# Track whether we created a temporary directory (for cleanup)
CREATED_TEMP_DIR=""
TMP_DOWNLOAD_DIR=""
SELECTED_VERSION=""
SELECTED_TARBALL_URL=""

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
info() { printf "\033[32m[info]\033[0m %s\n" "$*"; }
warn() { printf "[warn] %s\n" "$*"; }
err()  { printf "\033[31m[error]\033[0m %s\n" "$*" 1>&2; }

# Read user input - works even when script is piped
read_input() {
  local prompt="$1"
  printf "%s" "$prompt"
  # When piped (stdin not a tty), read from /dev/tty instead
  if [ -t 0 ]; then
    read -r REPLY
  elif [ -e /dev/tty ]; then
    read -r REPLY < /dev/tty
  else
    err "Cannot read user input (no tty available). Use environment variables for non-interactive mode."
    exit 1
  fi
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_to() {
  local url="$1" out="$2"
  if have_cmd curl; then
    curl -fsSL "$url" -o "$out"
  elif have_cmd wget; then
    wget -qO "$out" "$url"
  else
    err "Neither curl nor wget is available. Please install one of them."
    exit 1
  fi
}

fetch_url_to_stdout() {
  local url="$1"
  if have_cmd curl; then
    curl -fsSL "$url"
  elif have_cmd wget; then
    wget -qO- "$url"
  else
    err "Neither curl nor wget is available. Please install one of them."
    exit 1
  fi
}

get_latest_tarball_url() {
  # Prints the latest release tarball_url or empty string on failure
  local json
  if ! json="$(fetch_url_to_stdout "$API_URL_LATEST" 2>/dev/null || true)"; then
    json=""
  fi

  if [ -z "$json" ]; then
    printf "\n"
    return 0
  fi

  if have_cmd jq; then
    # Use jq if available
    printf "%s" "$json" | jq -r '.tarball_url // empty'
  else
    # Fallback: crude extraction via sed
    printf "%s" "$json" | sed -n 's/.*"tarball_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1
  fi
}

get_available_releases() {
  # Prints available releases (tag names) one per line
  local json
  if ! json="$(fetch_url_to_stdout "$API_URL_RELEASES" 2>/dev/null || true)"; then
    json=""
  fi

  if [ -z "$json" ]; then
    return 0
  fi

  if have_cmd jq; then
    printf "%s" "$json" | jq -r '.[].tag_name // empty'
  else
    # Fallback: crude extraction via grep/sed
    printf "%s" "$json" | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/'
  fi
}

get_tarball_url_for_version() {
  local version="$1"
  local json
  
  if ! json="$(fetch_url_to_stdout "$API_URL_RELEASES" 2>/dev/null || true)"; then
    json=""
  fi

  if [ -z "$json" ]; then
    return 0
  fi

  if have_cmd jq; then
    printf "%s" "$json" | jq -r --arg v "$version" '.[] | select(.tag_name == $v) | .tarball_url // empty'
  else
    # Fallback: This is trickier without jq, construct the URL directly
    printf "https://github.com/%s/%s/archive/refs/tags/%s.tar.gz" "$REPO_OWNER" "$REPO_NAME" "$version"
  fi
}

prompt_release_version() {
  # If T1PREP_VERSION is set, use it directly (non-interactive mode)
  if [ -n "${T1PREP_VERSION:-}" ]; then
    if [ "$T1PREP_VERSION" = "latest" ]; then
      info "Using latest release (from environment)"
      SELECTED_TARBALL_URL="$(get_latest_tarball_url || true)"
      SELECTED_VERSION="latest"
    else
      info "Using release version from environment: $T1PREP_VERSION"
      SELECTED_VERSION="$T1PREP_VERSION"
      SELECTED_TARBALL_URL="$(get_tarball_url_for_version "$T1PREP_VERSION")"
    fi
    return
  fi

  info "Fetching available releases from GitHub…"
  local releases
  releases="$(get_available_releases)"
  
  # Convert to array
  local release_array=()
  while IFS= read -r line; do
    [ -n "$line" ] && release_array+=("$line")
  done <<< "$releases"

  if [ ${#release_array[@]} -eq 0 ]; then
    warn "No releases found. Will use latest from main branch."
    SELECTED_VERSION="main"
    SELECTED_TARBALL_URL="$FALLBACK_TARBALL"
    return
  fi

  echo ""
  bold "Which version would you like to install?"
  echo ""
  echo "  1) Latest release (${release_array[0]:-unknown})"
  echo "  2) Development version (main branch)"
  echo "  3) Select from available releases"
  echo ""

  local choice
  while true; do
    read_input "Enter your choice [1-3]: " choice
    case "$choice" in
      1)
        SELECTED_VERSION="${release_array[0]}"
        SELECTED_TARBALL_URL="$(get_latest_tarball_url || true)"
        break
        ;;
      2)
        SELECTED_VERSION="main"
        SELECTED_TARBALL_URL="$FALLBACK_TARBALL"
        break
        ;;
      3)
        echo ""
        bold "Available releases:"
        local i=1
        for rel in "${release_array[@]}"; do
          echo "  $i) $rel"
          ((i++))
        done
        echo ""
        
        local rel_choice
        while true; do
          read_input "Enter release number [1-${#release_array[@]}]: "
          rel_choice="$REPLY"
          if [[ "$rel_choice" =~ ^[0-9]+$ ]] && [ "$rel_choice" -ge 1 ] && [ "$rel_choice" -le "${#release_array[@]}" ]; then
            local idx=$((rel_choice - 1))
            SELECTED_VERSION="${release_array[$idx]}"
            SELECTED_TARBALL_URL="$(get_tarball_url_for_version "$SELECTED_VERSION")"
            break 2
          else
            warn "Invalid choice. Please enter a number between 1 and ${#release_array[@]}."
          fi
        done
        ;;
      *)
        warn "Invalid choice. Please enter 1, 2, or 3."
        ;;
    esac
  done

  info "Selected version: $SELECTED_VERSION"
}

validate_tar_gz() {
  local file="$1"
  if ! tar -tzf "$file" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

cleanup() {
  # Only remove the temporary download directory, never the installation directory
  if [ -n "${TMP_DOWNLOAD_DIR:-}" ] && [ -d "$TMP_DOWNLOAD_DIR" ]; then
    rm -rf "$TMP_DOWNLOAD_DIR"
  fi
  # Only remove install dir if it was a temp dir AND installation failed
  if [ -n "${CREATED_TEMP_DIR:-}" ] && [ -d "$CREATED_TEMP_DIR" ] && [ "${INSTALL_SUCCESS:-0}" != "1" ]; then
    rm -rf "$CREATED_TEMP_DIR"
  fi
}
trap cleanup EXIT

prompt_install_location() {
  # If T1PREP_INSTALL_DIR is set, use it directly (non-interactive mode)
  if [ -n "${T1PREP_INSTALL_DIR:-}" ]; then
    INSTALL_DIR="$T1PREP_INSTALL_DIR"
    info "Using installation directory from environment: $INSTALL_DIR"
    return
  fi

  local current_dir
  current_dir="$(pwd)"
  local temp_dir="${TMPDIR:-/tmp}/T1Prep"

  echo ""
  bold "Where would you like to install T1Prep?"
  echo ""
  echo "  1) Current directory: $current_dir/T1Prep"
  echo "  2) Temporary directory: $temp_dir"
  echo "  3) Custom directory (you will be prompted)"
  echo ""
  
  local choice
  while true; do
    read_input "Enter your choice [1-3]: "
    choice="$REPLY"
    case "$choice" in
      1)
        INSTALL_DIR="$current_dir/T1Prep"
        break
        ;;
      2)
        INSTALL_DIR="$temp_dir"
        CREATED_TEMP_DIR="$INSTALL_DIR"
        break
        ;;
      3)
        read_input "Enter installation path: "
        custom_path="$REPLY"
        if [ -z "$custom_path" ]; then
          warn "Path cannot be empty. Please try again."
          continue
        fi
        # Expand ~ to home directory
        custom_path="${custom_path/#\~/$HOME}"
        # Handle relative paths
        if [[ "$custom_path" != /* ]]; then
          custom_path="$current_dir/$custom_path"
        fi
        INSTALL_DIR="$custom_path"
        break
        ;;
      *)
        warn "Invalid choice. Please enter 1, 2, or 3."
        ;;
    esac
  done

  info "Installation directory: $INSTALL_DIR"
}

prepare_install_dir() {
  if [ -d "$INSTALL_DIR" ]; then
    # Check if it looks like an existing T1Prep installation
    if [ -f "$INSTALL_DIR/scripts/T1Prep" ] || [ -f "$INSTALL_DIR/T1Prep" ]; then
      warn "Existing T1Prep installation found at: $INSTALL_DIR"
      read_input "Do you want to overwrite it? [y/N]: "
      confirm="$REPLY"
      case "$confirm" in
        [yY]|[yY][eE][sS])
          info "Removing existing installation..."
          rm -rf "$INSTALL_DIR"
          ;;
        *)
          err "Installation cancelled."
          exit 1
          ;;
      esac
    fi
  fi
  
  # Create the installation directory
  if ! mkdir -p "$INSTALL_DIR"; then
    err "Failed to create installation directory: $INSTALL_DIR"
    exit 1
  fi
}

main() {
  bold "T1Prep installer"

  # Prompt for release version
  prompt_release_version

  # Prompt for installation location
  prompt_install_location

  # Create temporary directory for download only
  local base_tmp
  base_tmp="${TMPDIR:-/tmp}"
  TMP_DOWNLOAD_DIR="$(mktemp -d "${base_tmp%/}/t1prep-download-XXXXXX")"
  info "Download directory: $TMP_DOWNLOAD_DIR"

  # Resolve tarball URL (use selected or fallback)
  local tarball_url
  tarball_url="${SELECTED_TARBALL_URL:-}"

  if [ -z "$tarball_url" ]; then
    warn "Could not determine tarball URL. Falling back to main branch tarball."
    tarball_url="$FALLBACK_TARBALL"
  fi
  info "Using tarball: $tarball_url"

  local tarball_path
  tarball_path="$TMP_DOWNLOAD_DIR/t1prep.tar.gz"
  info "Downloading…"
  if ! download_to "$tarball_url" "$tarball_path"; then
    err "Failed to download: $tarball_url"
    exit 1
  fi

  if ! validate_tar_gz "$tarball_path"; then
    warn "Downloaded file is not a valid tar.gz. Falling back to main branch tarball."
    download_to "$FALLBACK_TARBALL" "$tarball_path"
    if ! validate_tar_gz "$tarball_path"; then
      err "Fallback tarball is also invalid. Aborting."
      exit 1
    fi
  fi

  # Prepare and verify installation directory
  prepare_install_dir

  info "Extracting to: $INSTALL_DIR"
  # Extract directly to a temp location first, then move contents
  local extract_tmp
  extract_tmp="$TMP_DOWNLOAD_DIR/extract"
  mkdir -p "$extract_tmp"
  tar -xzf "$tarball_path" -C "$extract_tmp"

  # Determine extracted root directory (GitHub tarballs create one top-level dir)
  local extract_root
  extract_root="$(find "$extract_tmp" -mindepth 1 -maxdepth 1 -type d | head -n1)"
  if [ -z "$extract_root" ] || [ ! -d "$extract_root" ]; then
    err "Could not locate extracted directory."
    exit 1
  fi

  # Move contents to installation directory
  mv "$extract_root"/* "$INSTALL_DIR/" 2>/dev/null || cp -R "$extract_root"/* "$INSTALL_DIR/"
  info "Installed to: $INSTALL_DIR"

  # Find the T1Prep launcher
  local t1prep_bin
  t1prep_bin="$(find "$INSTALL_DIR" -type f -name 'T1Prep' | head -n1)"
  if [ -z "$t1prep_bin" ]; then
    # Some setups may use lowercase or extension; try a wider search
    t1prep_bin="$(find "$INSTALL_DIR" -type f \( -name 'T1Prep' -o -name 'T1Prep.sh' -o -name 't1prep' -o -name 't1prep.sh' \) | head -n1)"
  fi

  if [ -z "$t1prep_bin" ]; then
    err "Could not find the 'T1Prep' launcher in the extracted archive."
    err "Please check the repository layout or run installation manually."
    exit 1
  fi

  chmod +x "$t1prep_bin" || true

  # For safety, run from repo root
  local repo_root
  repo_root="$(cd "$(dirname "$t1prep_bin")/.." >/dev/null 2>&1 && pwd || true)"
  if [ -z "$repo_root" ] || [ ! -d "$repo_root" ]; then
    repo_root="$(dirname "$t1prep_bin")"
  fi

  info "Starting T1Prep installation…"
  ( cd "$repo_root" && "$t1prep_bin" --install )

  # Mark installation as successful (prevents cleanup of temp install dir)
  INSTALL_SUCCESS=1

  bold "T1Prep installation finished."
  echo ""
  info "T1Prep is installed at: $INSTALL_DIR"
  info "To run T1Prep, use: $t1prep_bin"
  echo ""
  info "You may want to add T1Prep to your PATH:"
  echo "  export PATH=\"\$PATH:$(dirname "$t1prep_bin")\""
  echo ""
  if [ -n "${CREATED_TEMP_DIR:-}" ]; then
    warn "Note: T1Prep was installed to a temporary directory."
    warn "It may be removed on system reboot. Consider reinstalling to a permanent location."
  fi
}

main "$@"
