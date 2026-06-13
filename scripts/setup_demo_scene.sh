#!/bin/bash

# scripts/setup_demo_scene.sh
# Automates the setup of the premium Garden 3D Gaussian Splatting demo scene.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REL_TARGET_DIR="public/envs/go1_gaussian/splats/garden"
TARGET_DIR="${ROOT_DIR}/${REL_TARGET_DIR}"
SPLAT_PATH="${TARGET_DIR}/garden.splat"
JSON_PATH="${TARGET_DIR}/transform.json"
URL="https://huggingface.co/cakewalk/splat-data/resolve/main/garden.splat"

echo "=========================================================="
echo "🌸 Premium Garden Demo Scene Setup"
echo "=========================================================="

# 1. Create target directory
echo "1. Creating directory: ${REL_TARGET_DIR}..."
mkdir -p "${TARGET_DIR}"

# 2. Download dataset
echo "2. Downloading garden.splat (~135 MB from Hugging Face Model)..."
echo "   (This may take a few seconds depending on your connection)"

# Clean up old PLY file if exists to save space
if [ -f "${TARGET_DIR}/garden.ply" ]; then
    echo "   Removing old garden.ply..."
    rm -f "${TARGET_DIR}/garden.ply"
fi

# Delete existing corrupted splat file if it's too small or invalid
if [ -f "${SPLAT_PATH}" ] && [ "$(wc -c < "${SPLAT_PATH}")" -lt 1000 ]; then
    echo "   Removing corrupted existing splat file..."
    rm -f "${SPLAT_PATH}"
fi

if [ -f "${SPLAT_PATH}" ]; then
    echo "   File already exists. Skipping download."
else
    if command -v curl >/dev/null 2>&1; then
        curl -f -L -# -o "${SPLAT_PATH}" "${URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget --show-progress -O "${SPLAT_PATH}" "${URL}"
    else
        echo "❌ Error: Neither curl nor wget was found on your system."
        echo "   Please download the file manually from:"
        echo "   ${URL}"
        echo "   and place it at: ${SPLAT_PATH}"
        exit 1
    fi
fi

# Verify downloaded SPLAT file validity (should be non-empty and reasonably sized)
if [ ! -f "${SPLAT_PATH}" ] || [ "$(wc -c < "${SPLAT_PATH}")" -lt 1000000 ]; then
    echo "❌ Error: The downloaded file is not a valid SPLAT file."
    if [ -f "${SPLAT_PATH}" ]; then
        echo "   File content begins with:"
        head -n 5 "${SPLAT_PATH}"
        rm -f "${SPLAT_PATH}"
    fi
    exit 1
fi

# 3. Create transformation matrix
echo "3. Creating alignment config: ${JSON_PATH}..."
cat << 'EOF' > "${JSON_PATH}"
{
  "sourceUrl": "https://huggingface.co/cakewalk/splat-data/resolve/main/garden.splat",
  "matrix": {
    "garden.splat": [
      0.9999911218190988, -0.004161033313721418, -0.00066489479653502, 0.16752801296150288,
      -0.004161033313721418, -0.9501965641975403, -0.3116234540939331, 3.1704580485186478,
      0.00066489479653502, 0.3116234540939331, -0.9502054423784415, -0.1134703657170073,
      0.0, 0.0, 0.0, 1.0
    ]
  },
  "scale": 0.5,
  "spawn": [0.0, 0.0, 0.0]
}
EOF

echo "=========================================================="
echo "🎉 Success! The Garden demo scene is ready."
echo "   - Dataset: ${REL_TARGET_DIR}/garden.splat"
echo "   - Alignment: ${REL_TARGET_DIR}/transform.json"
echo "=========================================================="
echo "👉 Start the simulation:"
echo "   npm run dev:local-gaussian"
echo "   Then select the Garden preset in the Gaussian Terrain controls."
echo "=========================================================="
