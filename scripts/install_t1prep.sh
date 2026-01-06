#!/usr/bin/env bash
#
# install_t1prep.sh — Download the latest T1Prep release and run installation
#
# This script:
# 1) Detects curl or wget
# 2) Queries GitHub for the latest release of ChristianGaser/T1Prep
# 3) Downloads the tarball, extracts into a temporary directory
# 4) Locates the "T1Prep" launcher and runs `T1Prep --install`
#
# Env overrides:
#   REPO_OWNER  (default: ChristianGaser)
#   REPO_NAME   (default: T1prep)
#   KEEP_TEMP=1 (keep temporary directory for debugging)
#
# Requirements: bash, tar, curl or wget. jq is optional.

set -euo pipefail

REPO_OWNER="${REPO_OWNER:-ChristianGaser}"
REPO_NAME="${REPO_NAME:-T1Prep}"

API_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/latest"
FALLBACK_TARBALL="https://github.com/${REPO_OWNER}/${REPO_NAME}/archive/refs/heads/main.tar.gz"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
info() { printf "[info] %s\n" "$*"; }
warn() { printf "[warn] %s\n" "$*"; }
err()  { printf "[error] %s\n" "$*" 1>&2; }

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
  if ! json="$(fetch_url_to_stdout "$API_URL" 2>/dev/null || true)"; then
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

validate_tar_gz() {
  local file="$1"
  if ! tar -tzf "$file" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

cleanup() {
  if [ "${KEEP_TEMP:-0}" != "1" ] && [ -n "${TMP_WORKDIR:-}" ] && [ -d "$TMP_WORKDIR" ]; then
    rm -rf "$TMP_WORKDIR"
  fi
}
trap cleanup EXIT

main() {
  bold "T1Prep installer"

  # Create temp workspace
  local base_tmp
  base_tmp="${TMPDIR:-/tmp}"
  TMP_WORKDIR="$(mktemp -d "${base_tmp%/}/t1prep-install-XXXXXX")"
  info "Working directory: $TMP_WORKDIR"

  # Resolve tarball URL
  info "Querying latest release from GitHub…"
  local tarball_url
  tarball_url="$(get_latest_tarball_url || true)"

  if [ -z "$tarball_url" ]; then
    warn "Could not determine latest release tarball_url (API rate limit or no releases?). Falling back to main branch tarball."
    tarball_url="$FALLBACK_TARBALL"
  fi
  info "Using tarball: $tarball_url"

  local tarball_path
  tarball_path="$TMP_WORKDIR/t1prep.tar.gz"
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

  info "Extracting…"
  tar -xzf "$tarball_path" -C "$TMP_WORKDIR"

  # Determine extracted root directory (GitHub tarballs create one top-level dir)
  local extract_root
  extract_root="$(find "$TMP_WORKDIR" -mindepth 1 -maxdepth 1 -type d | head -n1)"
  if [ -z "$extract_root" ] || [ ! -d "$extract_root" ]; then
    err "Could not locate extracted directory."
    exit 1
  fi
  info "Extracted to: $extract_root"

  # Find the T1Prep launcher
  local t1prep_bin
  t1prep_bin="$(find "$extract_root" -type f -name 'T1Prep' | head -n1)"
  if [ -z "$t1prep_bin" ]; then
    # Some setups may use lowercase or extension; try a wider search
    t1prep_bin="$(find "$extract_root" -type f \( -name 'T1Prep' -o -name 'T1Prep.sh' -o -name 't1prep' -o -name 't1prep.sh' \) | head -n1)"
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

  bold "T1Prep installation finished."
  info "You can re-run installation later with: $t1prep_bin --install"
  if [ "${KEEP_TEMP:-0}" != "1" ]; then
    info "Temporary files were cleaned up. Set KEEP_TEMP=1 to keep them next time."
  else
    info "Temporary files kept at: $TMP_WORKDIR"
  fi
}

main "$@"
