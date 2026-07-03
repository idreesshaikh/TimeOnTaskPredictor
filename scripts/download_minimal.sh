#!/bin/bash
# Download AND extract the raw WebChain trajectory JSON (gated — `hf login`
# or `uv run hf auth login` first). Idempotent: skips whatever already
# exists, so it is safe as stage 0 of scripts/run_all.sbatch.
# Ends with the JSON at data/raw/all_json_files/ — the default input of
# scripts/build_dataset.py.

set -euo pipefail
cd "$(dirname "$0")/.."

RAW_DIR="data/raw"
JSON_DIR="$RAW_DIR/all_json_files"
ZIP="$RAW_DIR/all_json_files.zip"

if [ -d "$JSON_DIR" ] && [ -n "$(ls -A "$JSON_DIR" 2>/dev/null)" ]; then
    echo "Raw JSON already extracted: $(ls "$JSON_DIR" | wc -l) files in $JSON_DIR — nothing to do."
    exit 0
fi

if [ ! -f "$ZIP" ]; then
    echo "Downloading JSON trajectories from webagentlab/WebChain (gated) ..."
    TMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TMP_DIR"' EXIT
    uv run hf download webagentlab/WebChain \
        --repo-type dataset \
        --include "raw/json/*" \
        --local-dir "$TMP_DIR"
    mkdir -p "$RAW_DIR"
    mv "$TMP_DIR/raw/json/all_json_files.zip" "$ZIP"
fi

# The archive carries a top-level all_json_files/ folder → extract into raw/.
echo "Extracting $ZIP → $JSON_DIR ..."
unzip -q -o "$ZIP" -d "$RAW_DIR"
echo "Done: $(ls "$JSON_DIR" | wc -l) trajectory files in $JSON_DIR"
