#!/bin/bash
# JSON trajectories with timestamps

set -e

OUTPUT_DIR="./webchain_raw"
mkdir -p "$OUTPUT_DIR"

echo "Downloading JSON trajectories ..."
HF_HUB_ENABLE_HF_TRANSFER=1 hf download webagentlab/WebChain \
    --repo-type dataset \
    --include "raw/json/*" \
    --local-dir "$OUTPUT_DIR"

echo ""
echo "Download complete!"
echo ""
echo "Extract: unzip $OUTPUT_DIR/raw/json/all_json_files.zip -d $OUTPUT_DIR/raw/json/"
