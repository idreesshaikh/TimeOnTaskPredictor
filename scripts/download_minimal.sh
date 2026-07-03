#!/bin/bash
# Download the raw WebChain trajectory JSON (gated — `hf login` first),
# then resolve row screenshots if the label rows already exist.

set -e

OUTPUT_DIR="./webchain_raw"
mkdir -p "$OUTPUT_DIR"

echo "Downloading JSON trajectories ..."
hf download webagentlab/WebChain \
    --repo-type dataset \
    --include "raw/json/*" \
    --local-dir "$OUTPUT_DIR"

echo ""
echo "Download complete!"
echo ""
echo "Extract: unzip $OUTPUT_DIR/raw/json/all_json_files.zip -d $OUTPUT_DIR/raw/json/"

# Screenshot cache: fetches every https img_ref from
# data/processed/rows.parquet into data/images_cache/ (resumable — cached
# files are skipped; failures are logged, not fatal). Opaque-hash refs are
# looked up under data/images_root/ if you have it.
if [ -f data/processed/rows.parquet ]; then
    echo ""
    echo "Resolving row screenshots into data/images_cache/ ..."
    uv run python scripts/build_dataset.py --resolve-images
else
    echo "Skipping image resolution: data/processed/rows.parquet not found."
    echo "Run: uv run python scripts/build_dataset.py <json_dir> --labels"
fi
