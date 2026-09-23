#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

profile="core"
state_dir="$PROJECT_ROOT/.cache/cloud-gpu"
skip_system_packages=false
gpu_target=""
dry_run=false

usage() {
    cat <<'EOF'
Usage: bash tool_scripts/setup_cloud_gpu.sh [options]

Prepare the repository's locked environment on an Ubuntu x86_64 RTX 5080/5090
cloud host. The script does not install NVIDIA drivers, the CUDA Toolkit,
datasets, Codex, or editor extensions.

Options:
  --profile PROFILE          core (default), examples, or full
  --state-dir PATH           Persistent uv/Python cache directory
  --skip-system-packages     Do not install missing curl/git/rsync/tmux
  --gpu MODEL                Require 5080 or 5090 (default: detect either)
  --dry-run                  Validate inputs and print planned actions
  -h, --help                 Show this help

Profiles:
  core       Base runtime dependencies only
  examples   Core plus the optional examples dependencies
  full       Every optional dependency and development/test tools
EOF
}

log() {
    printf '[cloud-gpu-setup] %s\n' "$*"
}

die() {
    printf '[cloud-gpu-setup] ERROR: %s\n' "$*" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
        --profile)
            (($# >= 2)) || die "--profile requires a value"
            profile="$2"
            shift 2
            ;;
        --state-dir)
            (($# >= 2)) || die "--state-dir requires a value"
            state_dir="$2"
            shift 2
            ;;
        --skip-system-packages)
            skip_system_packages=true
            shift
            ;;
        --gpu)
            (($# >= 2)) || die "--gpu requires a value"
            gpu_target="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$profile" in
    core|examples|full) ;;
    *) die "profile must be one of: core, examples, full" ;;
esac
case "$gpu_target" in
    ""|5080|5090) ;;
    *) die "cloud GPU must be 5080 or 5090" ;;
esac
[[ "$state_dir" == /* ]] || state_dir="$PROJECT_ROOT/$state_dir"
[[ "$(uname -s)" == "Linux" ]] || die "Linux is required"
[[ "$(uname -m)" == "x86_64" ]] || die "x86_64 is required"
[[ -f "$PROJECT_ROOT/pyproject.toml" ]] || die "pyproject.toml not found"
[[ -f "$PROJECT_ROOT/uv.lock" ]] || die "uv.lock not found"
[[ -f "$PROJECT_ROOT/.python-version" ]] || die ".python-version not found"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "${gpu_name^^}" in
    *"RTX 5080") detected_gpu=5080 ;;
    *"RTX 5090") detected_gpu=5090 ;;
    *) die "expected cloud RTX 5080 or RTX 5090, found: $gpu_name" ;;
esac
[[ -z "$gpu_target" || "$gpu_target" == "$detected_gpu" ]] || \
    die "expected RTX $gpu_target, found: $gpu_name"
log "GPU preflight passed: $gpu_name"

missing_packages=()
for command_package in curl:curl git:git rsync:rsync tmux:tmux; do
    command_name="${command_package%%:*}"
    package_name="${command_package##*:}"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        missing_packages+=("$package_name")
    fi
done

if ((${#missing_packages[@]} > 0)); then
    [[ "$skip_system_packages" == false ]] || \
        die "missing commands: ${missing_packages[*]}"
    command -v apt-get >/dev/null 2>&1 || \
        die "apt-get is required to install: ${missing_packages[*]}"
    if ((EUID == 0)); then
        privilege=()
    else
        command -v sudo >/dev/null 2>&1 || die "sudo is required"
        privilege=(sudo)
    fi
    if [[ "$dry_run" == true ]]; then
        log "would install: ca-certificates ${missing_packages[*]}"
    else
        "${privilege[@]}" apt-get update
        "${privilege[@]}" env DEBIAN_FRONTEND=noninteractive \
            apt-get install -y ca-certificates "${missing_packages[@]}"
    fi
else
    log "required system commands are already available"
fi

IFS= read -r python_version < "$PROJECT_ROOT/.python-version"
python_version="${python_version//[[:space:]]/}"
[[ -n "$python_version" ]] || die ".python-version is empty"
log "project: $PROJECT_ROOT"
log "profile: $profile"
log "persistent state: $state_dir"
log "Python: $python_version"

if [[ "$dry_run" == true ]]; then
    log "would install uv if absent, install Python, and sync the lock"
    exit 0
fi

mkdir -p "$state_dir/uv-cache" "$state_dir/uv-python"
export UV_CACHE_DIR="$state_dir/uv-cache"
export UV_PYTHON_INSTALL_DIR="$state_dir/uv-python"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation failed"

cd "$PROJECT_ROOT"
uv python install "$python_version"
sync_args=(sync --locked)
case "$profile" in
    core) sync_args+=(--no-dev) ;;
    examples) sync_args+=(--extra examples) ;;
    full) sync_args+=(--all-extras) ;;
esac
uv "${sync_args[@]}"

uv run --locked --no-sync python tool_scripts/pytorch_test.py --gpu "$detected_gpu"

log "setup completed"
log "next: run a Python lesson or tool from the repository root"
