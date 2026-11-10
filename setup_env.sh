#!/usr/bin/env bash
#
# setup_env.sh - one-shot CANN 9.0.0 environment bootstrap for a disposable
# aarch64 notebook container.
#
# SCOPE / ASSUMPTIONS
#   * This script is intended for a fresh, disposable container that does not
#     already have a conflicting CANN installation.
#   * If another CANN version (for example 8.5) is present, this script only
#     installs CANN 9.0.0 and selects the 9.0.0 environment. It does NOT
#     uninstall, clean, or modify the old installation.
#   * Run it as:  bash setup_env.sh
#     Do not use `source setup_env.sh`: the installer and `yes | ...` pipeline
#     must run in child processes, and a source made inside this script cannot
#     modify the shell that launched it.
#   * pip3 is a hard prerequisite for the CANN ge-compiler subpackage. If it is
#     missing, this script tries to install python3-pip and aborts with manual
#     instructions if that is not possible.
#
set -euo pipefail

CANN_VERSION="9.0.0"
TOOLKIT_PKG_NAME="Ascend-cann_9.0.0_linux-aarch64.run"
OPS_PKG_NAME="Ascend-cann-910b-ops_9.0.0_linux-aarch64.run"
TOOLKIT_URL="https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann_9.0.0_linux-aarch64.run"
OPS_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-910b-ops_9.0.0_linux-aarch64.run"

JSON_URL="https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp"
JSON_ROOT="${NLOHMANN_JSON_ROOT:-${HOME:?HOME is not set}/nlohmann_json}"
JSON_INCLUDE_DIR="${JSON_ROOT}/include"
JSON_FILE="${JSON_INCLUDE_DIR}/nlohmann/json.hpp"
GTEST_SRC_DIR="${GTEST_SOURCE_DIR:-/usr/src/googletest}"

PKG_DIR="${CANN_PKG_DIR:-${HOME}/cann_installers}"
INSTALL_LOG_DIR="${PKG_DIR}/logs"
MIN_PKG_BYTES=$((100 * 1024 * 1024))
CANN_INSTALL_PATH="${CANN_INSTALL_PATH:-}"

log()  { printf '\n[INFO] %s\n' "$*"; }
warn() { printf '\n[WARN] %s\n' "$*" >&2; }
die()  { printf '\n[ERROR] %s\n' "$*" >&2; exit 1; }

as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo -n "$@"
    else
        return 1
    fi
}

run_installer_cmd() {
    if [ "$ROOT_INSTALL" -eq 1 ] && [ "$(id -u)" -ne 0 ]; then
        sudo -n bash "$@"
    else
        bash "$@"
    fi
}

file_size() {
    if stat -c%s "$1" >/dev/null 2>&1; then
        stat -c%s "$1"
    else
        wc -c < "$1" | tr -d ' '
    fi
}

download_file() {
    local url="$1"
    local output="$2"
    local partial="${output}.part"
    local downloaded=0

    mkdir -p "$(dirname "$output")"
    rm -f "$partial"

    if command -v curl >/dev/null 2>&1; then
        if curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 \
            --progress-bar -o "$partial" "$url"; then
            downloaded=1
        else
            rm -f "$partial"
        fi
    fi

    if [ "$downloaded" -eq 0 ] && command -v wget >/dev/null 2>&1; then
        if wget -O "$partial" "$url"; then
            downloaded=1
        else
            rm -f "$partial"
        fi
    fi

    if [ "$downloaded" -ne 1 ]; then
        warn "Download failed: ${url}"
        return 1
    fi

    mv -f "$partial" "$output"
    return 0
}

install_cann_package() {
    local pkg="$1"
    local label="$2"
    local log_file="${INSTALL_LOG_DIR}/$(basename "$pkg").log"

    log "Installing ${label}: $(basename "$pkg")"
    log "Installer output: ${log_file}"

    # `yes | installer` answers the licence prompts automatically. A successful
    # `yes` process receives SIGPIPE when the installer stops reading, so
    # pipefail must be disabled around this pipeline.
    set +o pipefail
    if ! yes | run_installer_cmd "$pkg" "${INSTALL_ARGS[@]}" >"$log_file" 2>&1; then
        set -o pipefail
        warn "${label} installer returned a non-zero status. Last log lines:"
        tail -n 80 "$log_file" >&2 || true
        return 1
    fi
    set -o pipefail
    log "${label} installation finished."
}

probe_set_env() {
    local candidate="$1"
    (
        set +eu
        # shellcheck disable=SC1090
        source "$candidate" >/dev/null 2>&1 || true
        printf '%s\n' "${ASCEND_HOME_PATH:-}"
    )
}

json_headers_look_valid() {
    local file="$1"
    [ -s "$file" ] || return 1
    [ "$(file_size "$file")" -gt 100000 ] || return 1
    grep -q 'nlohmann' "$file"
}

setup_json_header() {
    JSON_READY=0
    JSON_PATH=""
    JSON_INCLUDE_RESULT=""

    log "Installing nlohmann/json.hpp (v3.11.3) under ${JSON_INCLUDE_DIR}"
    mkdir -p "$(dirname "$JSON_FILE")"

    if json_headers_look_valid "$JSON_FILE"; then
        log "Existing json.hpp is present and looks valid; reusing it."
        JSON_READY=1
    else
        rm -f "$JSON_FILE"
        if download_file "$JSON_URL" "$JSON_FILE" && json_headers_look_valid "$JSON_FILE"; then
            log "Downloaded json.hpp successfully."
            JSON_READY=1
        else
            rm -f "$JSON_FILE"
            warn "Could not download nlohmann/json.hpp from raw.githubusercontent.com."
            warn "This usually means the domain is not on the container network allowlist."
            warn "Falling back to: apt-get install -y nlohmann-json3-dev"
        fi
    fi

    if [ "$JSON_READY" -eq 0 ]; then
        local apt_log="${INSTALL_LOG_DIR}/apt_nlohmann_json.log"
        mkdir -p "$INSTALL_LOG_DIR"
        as_root env DEBIAN_FRONTEND=noninteractive apt-get update -y >"$apt_log" 2>&1 \
            || warn "apt-get update failed; trying the install anyway."
        if as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y nlohmann-json3-dev >>"$apt_log" 2>&1 \
            && [ -f /usr/include/nlohmann/json.hpp ]; then
            JSON_INCLUDE_DIR="/usr/include"
            JSON_READY=1
            log "Installed nlohmann-json3-dev from apt; using /usr/include."
        else
            warn "apt fallback also failed. See ${apt_log}"
            tail -n 40 "$apt_log" >&2 || true
        fi
    fi

    if [ "$JSON_READY" -eq 1 ]; then
        JSON_INCLUDE_RESULT="$JSON_INCLUDE_DIR"
        if [ -f "${JSON_INCLUDE_DIR}/nlohmann/json.hpp" ]; then
            JSON_PATH="${JSON_INCLUDE_DIR}/nlohmann/json.hpp"
        else
            JSON_PATH="$JSON_FILE"
        fi
        log "json.hpp: ${JSON_PATH}"
    else
        warn "json.hpp is unavailable; build.sh -u will not work until it is provided."
    fi
}

setup_googletest() {
    GTEST_READY=0

    log "Checking GoogleTest sources at ${GTEST_SRC_DIR}"
    if [ -f "${GTEST_SRC_DIR}/CMakeLists.txt" ]; then
        GTEST_READY=1
        log "Found GoogleTest sources: ${GTEST_SRC_DIR}"
        return 0
    fi

    warn "GoogleTest sources not found; trying: apt-get install -y libgtest-dev"
    local apt_log="${INSTALL_LOG_DIR}/apt_googletest.log"
    mkdir -p "$INSTALL_LOG_DIR"
    as_root env DEBIAN_FRONTEND=noninteractive apt-get update -y >"$apt_log" 2>&1 \
        || warn "apt-get update failed; trying the install anyway."

    if as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y libgtest-dev >>"$apt_log" 2>&1 \
        && [ -f "${GTEST_SRC_DIR}/CMakeLists.txt" ]; then
        GTEST_READY=1
        log "Found GoogleTest sources after apt install: ${GTEST_SRC_DIR}"
    else
        warn "Could not install/find libgtest-dev sources. See ${apt_log}"
        tail -n 40 "$apt_log" >&2 || true
    fi

    if [ "$GTEST_READY" -ne 1 ]; then
        warn "GoogleTest is unavailable. This is non-fatal: build.sh -e can still run,"
        warn "but build.sh -u will need GTEST_SOURCE_DIR or a system gtest installation."
    fi
    return 0
}

ensure_pip3() {
    if command -v pip3 >/dev/null 2>&1; then
        if pip3 --version >/dev/null 2>&1; then
            log "pip3 found: $(command -v pip3)"
            return 0
        fi
        die "pip3 exists at $(command -v pip3) but cannot run. Repair the Python 3 pip installation, then rerun: bash setup_env.sh"
    fi

    warn "pip3 is required by the CANN ge-compiler subpackage, but it is not installed."
    warn "Attempting: apt-get update && apt-get install -y python3-pip"

    local apt_log="${INSTALL_LOG_DIR}/apt_python3_pip.log"
    mkdir -p "$INSTALL_LOG_DIR"
    as_root env DEBIAN_FRONTEND=noninteractive apt-get update -y >"$apt_log" 2>&1 \
        || warn "apt-get update failed; trying apt-get install anyway."

    if ! as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip >>"$apt_log" 2>&1; then
        tail -n 60 "$apt_log" >&2 || true
        die "Could not install pip3 automatically. pip3 is required by the CANN ge-compiler subpackage. Install it manually (as root: apt-get update && apt-get install -y python3-pip, or with sudo), then rerun: bash setup_env.sh"
    fi

    hash -r 2>/dev/null || true
    if ! command -v pip3 >/dev/null 2>&1 || ! pip3 --version >/dev/null 2>&1; then
        tail -n 60 "$apt_log" >&2 || true
        die "python3-pip was installed but pip3 is still not usable. Check PATH and the apt log, then rerun: bash setup_env.sh"
    fi
    log "pip3 installed: $(command -v pip3)"
    return 0
}

# ---------------------------------------------------------------------------
# 0. Basic environment checks
# ---------------------------------------------------------------------------
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    die "Unsupported architecture: ${ARCH}. The CANN 9.0.0 packages in this script require aarch64/ARM64."
fi
log "Architecture: ${ARCH}"

ROOT_INSTALL=0
if [ "$(id -u)" -eq 0 ]; then
    ROOT_INSTALL=1
elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    ROOT_INSTALL=1
else
    warn "Not running as root and passwordless sudo is unavailable."
    if [ -z "$CANN_INSTALL_PATH" ]; then
        CANN_INSTALL_PATH="${HOME}/Ascend"
        warn "Installing CANN under the user-writable path ${CANN_INSTALL_PATH} instead of /usr/local/Ascend."
    fi
    warn "apt-get fallbacks may still fail without root; CANN itself will be installed as the current user."
fi

INSTALL_ARGS=(--install)
if [ -n "$CANN_INSTALL_PATH" ]; then
    INSTALL_ARGS+=(--install-path "$CANN_INSTALL_PATH")
    log "CANN install path: ${CANN_INSTALL_PATH}"
else
    log "CANN install path: installer default (usually /usr/local/Ascend)"
fi

mkdir -p "$PKG_DIR" "$INSTALL_LOG_DIR"
export DEBIAN_FRONTEND=noninteractive

ensure_pip3

# ---------------------------------------------------------------------------
# 1. Download and install the two CANN 9.0.0 packages
# ---------------------------------------------------------------------------
download_and_check_package() {
    local url="$1"
    local pkg="$2"
    local name="$3"

    if [ -s "$pkg" ]; then
        local existing_size
        existing_size="$(file_size "$pkg" || echo 0)"
        if [ "$existing_size" -ge "$MIN_PKG_BYTES" ]; then
            log "Reusing existing package: ${pkg}"
            ls -lh "$pkg" || true
            return 0
        fi
        warn "Existing ${pkg} is too small (${existing_size} bytes); deleting and re-downloading."
        rm -f "$pkg"
    fi

    log "Downloading ${name}"
    log "URL: ${url}"
    if ! download_file "$url" "$pkg"; then
        die "Download failed for ${name}. Check network access, proxy settings, and the URL."
    fi

    ls -lh "$pkg" || true
    local size
    size="$(file_size "$pkg" || echo 0)"
    if [ "$size" -lt "$MIN_PKG_BYTES" ]; then
        rm -f "$pkg"
        die "Downloaded ${name} is only ${size} bytes (< 100 MiB). This is probably a 404/HTML error page, not an installer. Aborting."
    fi
    set +o pipefail
    if head -c 512 "$pkg" 2>/dev/null | grep -qiE '<html|<error'; then
        set -o pipefail
        rm -f "$pkg"
        die "Downloaded ${name} looks like an HTML error page. Aborting."
    fi
    set -o pipefail
    log "${name} download OK (${size} bytes)."
}

TOOLKIT_PKG="${PKG_DIR}/${TOOLKIT_PKG_NAME}"
OPS_PKG="${PKG_DIR}/${OPS_PKG_NAME}"
download_and_check_package "$TOOLKIT_URL" "$TOOLKIT_PKG" "$TOOLKIT_PKG_NAME"
download_and_check_package "$OPS_URL" "$OPS_PKG" "$OPS_PKG_NAME"

install_cann_package "$TOOLKIT_PKG" "CANN toolkit" \
    || die "CANN toolkit installation failed. Inspect the log, free disk space, and check for a conflicting CANN installation."
install_cann_package "$OPS_PKG" "CANN 910B ops" \
    || die "CANN 910B ops installation failed. Inspect the log and verify that the toolkit was installed successfully."

# ---------------------------------------------------------------------------
# 2. Locate, source, and validate the real set_env.sh
# ---------------------------------------------------------------------------
log "Searching for set_env.sh under / (max depth 6). This can take a moment..."
CANDIDATES="$(
    find / -maxdepth 6 \
        \( -path /proc -o -path /sys -o -path /dev -o -path /run \) -prune -o \
        \( -type f -o -type l \) -name set_env.sh -print 2>/dev/null | sort -u
)"

if [ -z "$CANDIDATES" ]; then
    die "No set_env.sh was found after installation. The install may have failed or used an unexpected layout."
fi

SELECTED_SET_ENV=""
SELECTED_HOME=""

for pass in 1 2; do
    while IFS= read -r candidate; do
        [ -n "$candidate" ] || continue
        case "$candidate" in
            *9.0.0*) [ "$pass" -eq 1 ] || continue ;;
            *)       [ "$pass" -eq 2 ] || continue ;;
        esac

        home="$(probe_set_env "$candidate" || true)"
        [ -n "$home" ] || continue
        resolved_home="$(readlink -f "$home" 2>/dev/null || printf '%s' "$home")"
        case "${home} ${resolved_home}" in
            *9.0.0*)
                SELECTED_SET_ENV="$candidate"
                SELECTED_HOME="$home"
                break 2
                ;;
        esac
    done <<< "$CANDIDATES"
done

if [ -z "$SELECTED_SET_ENV" ]; then
    warn "Found set_env.sh candidates, but none resolved ASCEND_HOME_PATH to a 9.0.0 path:"
    printf '%s\n' "$CANDIDATES" >&2
    die "Could not select a CANN 9.0.0 environment. Refusing to continue with an unknown CANN version."
fi

log "Selected set_env.sh: ${SELECTED_SET_ENV}"
set +u
# shellcheck disable=SC1090
source "$SELECTED_SET_ENV" >/dev/null 2>&1 || true
set -euo pipefail

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    die "Sourcing ${SELECTED_SET_ENV} did not set ASCEND_HOME_PATH."
fi
export ASCEND_HOME_PATH

if [ ! -d "$ASCEND_HOME_PATH" ]; then
    die "ASCEND_HOME_PATH=${ASCEND_HOME_PATH} is not an existing directory."
fi

RESOLVED_ASCEND_HOME="$(readlink -f "$ASCEND_HOME_PATH" 2>/dev/null || printf '%s' "$ASCEND_HOME_PATH")"
HAS_900=0
case "${ASCEND_HOME_PATH} ${RESOLVED_ASCEND_HOME}" in
    *9.0.0*) HAS_900=1 ;;
esac
if [ "$HAS_900" -eq 0 ] && [ -f "${ASCEND_HOME_PATH}/version.cfg" ] \
    && grep -q '9\.0\.0' "${ASCEND_HOME_PATH}/version.cfg"; then
    HAS_900=1
fi
if [ "$HAS_900" -ne 1 ]; then
    die "ASCEND_HOME_PATH=${ASCEND_HOME_PATH} does not point to CANN 9.0.0 (resolved: ${RESOLVED_ASCEND_HOME})."
fi

if [ -f "${ASCEND_HOME_PATH}/version.cfg" ] \
    && ! grep -q '9\.0\.0' "${ASCEND_HOME_PATH}/version.cfg"; then
    die "Version file ${ASCEND_HOME_PATH}/version.cfg does not report 9.0.0."
fi

log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
log "CANN 9.0.0 environment validated."

# ---------------------------------------------------------------------------
# 3. nlohmann/json.hpp (required by build.sh -u)
# ---------------------------------------------------------------------------
setup_json_header

# ---------------------------------------------------------------------------
# 4. GoogleTest (non-fatal)
# ---------------------------------------------------------------------------
setup_googletest || true

# ---------------------------------------------------------------------------
# 5. Summary and next steps
# ---------------------------------------------------------------------------
SETUP_FAILED=0
if [ "$JSON_READY" -ne 1 ]; then
    SETUP_FAILED=1
fi

printf '\n'
printf '================ SETUP SUMMARY ================\n'
printf 'CANN version          : %s\n' "$CANN_VERSION"
printf 'CANN install path     : %s\n' "${CANN_INSTALL_PATH:-<installer default>}"
printf 'ASCEND_HOME_PATH      : %s\n' "${ASCEND_HOME_PATH:-<unset>}"
printf 'set_env.sh            : %s\n' "$SELECTED_SET_ENV"
printf 'nlohmann json.hpp     : %s\n' "${JSON_PATH:-<missing>}"
printf 'NLOHMANN_JSON_INCLUDE : %s\n' "${JSON_INCLUDE_RESULT:-<missing>}"
if [ "$GTEST_READY" -eq 1 ]; then
    printf 'googletest source     : %s\n' "$GTEST_SRC_DIR"
else
    printf 'googletest source     : <missing>\n'
fi
printf '-----------------------------------------------\n'
printf 'One-line export for build.sh -u:\n'
printf '  export NLOHMANN_JSON_INCLUDE_DIR=%q GTEST_SOURCE_DIR=%q\n' \
    "${JSON_INCLUDE_RESULT:-}" "$([ "$GTEST_READY" -eq 1 ] && printf '%s' "$GTEST_SRC_DIR")"
printf '-----------------------------------------------\n'
printf 'The source above only affected this script process. In YOUR current shell, run:\n'
printf '  source %q\n' "$SELECTED_SET_ENV"
printf 'Then, for example:\n'
printf '  cd code7   # or code8\n'
printf '  bash build.sh -u\n'
printf '  DEVICE_ID=0 bash build.sh -e\n'
printf '===============================================\n'

if [ "$SETUP_FAILED" -ne 0 ]; then
    warn "Setup finished, but nlohmann/json.hpp is missing. build.sh -e may work; build.sh -u will not."
    exit 1
fi

log "Setup completed successfully."
