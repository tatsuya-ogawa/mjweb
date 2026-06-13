#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MJLAB_DIR="${MJWEB_PREPARE_MJLAB_DIR:-mjlab}"
MJLAB_REPO="${MJWEB_PREPARE_MJLAB_REPO:-https://github.com/tatsuya-ogawa/mjlab.git}"
MJLAB_REF="${MJWEB_PREPARE_MJLAB_REF:-a0ba05890a2ea4111b33c9cbb85f690bf19ca434}"
EXPORT_SCENES="${MJWEB_PREPARE_EXPORT_SCENES:-auto}"
PRUNE_STL="${MJWEB_PREPARE_PRUNE_STL:-0}"

GENERATED_ENVS=(
  "g1_flat"
  "g1_rough"
  "g1_backflip"
  "go1_flat"
  "go1_rough"
)

log() {
  printf '[prepare-assets] %s\n' "$*"
}

fail() {
  printf '[prepare-assets] ERROR: %s\n' "$*" >&2
  exit 1
}

truthy() {
  case "${1}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

scene_xml_missing() {
  for env_id in "${GENERATED_ENVS[@]}"; do
    if [[ ! -f "public/envs/${env_id}/scene.xml" ]]; then
      return 0
    fi
  done
  return 1
}

source_assets_missing() {
  for env_id in "${GENERATED_ENVS[@]}"; do
    if ! find "public/envs/${env_id}/assets" -type f \( -iname '*.stl' -o -iname '*.obj' -o -iname '*.dae' -o -iname '*.png' \) -print -quit 2>/dev/null | grep -q .; then
      return 0
    fi
  done
  return 1
}

scene_prepare_mode() {
  case "${EXPORT_SCENES}" in
    1|true|TRUE|yes|YES|on|ON)
      printf 'full\n'
      ;;
    0|false|FALSE|no|NO|off|OFF)
      printf 'skip\n'
      ;;
    assets|assets-only)
      printf 'assets\n'
      ;;
    auto)
      if scene_xml_missing; then
        printf 'full\n'
      elif source_assets_missing; then
        printf 'assets\n'
      else
        printf 'skip\n'
      fi
      ;;
    *)
      fail "Invalid MJWEB_PREPARE_EXPORT_SCENES=${EXPORT_SCENES}; expected auto, assets, 1, or 0"
      ;;
  esac
}

ensure_mjlab_checkout() {
  if [[ -d "${MJLAB_DIR}" ]]; then
    log "Using existing mjlab checkout at ${MJLAB_DIR}"
    return
  fi

  command -v git >/dev/null 2>&1 || fail "git is required to clone mjlab"

  local clone_url="${MJLAB_REPO}"
  if [[ -n "${MJWEB_PREPARE_MJLAB_TOKEN:-}" && "${MJLAB_REPO}" == https://github.com/* ]]; then
    clone_url="https://x-access-token:${MJWEB_PREPARE_MJLAB_TOKEN}@${MJLAB_REPO#https://}"
  fi

  log "Cloning mjlab dependency into ${MJLAB_DIR}"
  git clone --depth 1 "${clone_url}" "${MJLAB_DIR}"
  if [[ -n "${MJLAB_REF}" ]]; then
    git -C "${MJLAB_DIR}" fetch --depth 1 origin "${MJLAB_REF}" >/dev/null 2>&1 || true
    git -C "${MJLAB_DIR}" checkout --detach "${MJLAB_REF}"
  fi
}

prepare_scene_assets() {
  local mode="$1"
  command -v uv >/dev/null 2>&1 || fail "uv is required for scene export. Install uv, then rerun npm run prepare:assets."
  ensure_mjlab_checkout

  export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.cache/matplotlib}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.cache/uv}"
  mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}/fontconfig" "${UV_CACHE_DIR}"

  log "Syncing scene export environment"
  (cd scripts/export_scenes && uv sync --locked)

  if [[ "${mode}" == "full" ]]; then
    log "Exporting MuJoCo scenes and source mesh assets"
    (cd scripts/export_scenes && uv run python export.py)
  else
    log "Copying source mesh assets for existing scene XML files"
    (cd scripts/export_scenes && uv run python export.py --assets-only)
  fi
}

prune_source_mesh_assets() {
  log "Pruning source mesh assets from public envs"
  for env_id in "${GENERATED_ENVS[@]}"; do
    local optimized_xml="public/envs/${env_id}/scene_optimized.xml"
    if [[ -f "${optimized_xml}" ]] && grep -Eiq 'file="[^"]+\.(stl|obj|dae|png)"' "${optimized_xml}"; then
      log "Keeping public/envs/${env_id}/assets because ${optimized_xml} still references file assets"
      continue
    fi
    rm -rf "public/envs/${env_id}/assets"
  done
}

SCENE_MODE="$(scene_prepare_mode)"
case "${SCENE_MODE}" in
  full|assets)
    prepare_scene_assets "${SCENE_MODE}"
    ;;
  skip)
    log "Scene source assets already present; skipping scene export"
    ;;
esac

log "Generating optimized render assets"
npm run optimize:render-assets

log "Generating Gaussian source manifest"
npm run generate:gaussian-sources

if truthy "${PRUNE_STL}"; then
  prune_source_mesh_assets
fi

log "Done"
