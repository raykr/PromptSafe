#!/usr/bin/env bash
set -euo pipefail

# Train the FLUX embedding adapter on one safety category.
# Usage: bash scripts/train_flux.sh [GPU_ID] [CATEGORY]
# Example: bash scripts/train_flux.sh 0 sexual

GPU_ID="${1:-0}"
CATEGORY="${2:-sexual}"
MODEL_PATH="${MODEL_PATH:-/home/raykr/models/black-forest-labs/FLUX.1-dev}"
TRAIN_CSV="${TRAIN_CSV:-datasets/train/train_${CATEGORY}_toxic.csv}"
OUT_DIR="${OUT_DIR:-outputs/flux/flux_${CATEGORY}}"

if [[ "$GPU_ID" == *","* ]]; then
    echo "flux/train_flux.py is single-process; pass exactly one GPU ID." >&2
    exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python flux/train_flux.py \
    --model_path "$MODEL_PATH" \
    --train_csv "$TRAIN_CSV" \
    --category "$CATEGORY" \
    --out_dir "$OUT_DIR" \
    --dtype bf16 \
    --train_mode all
