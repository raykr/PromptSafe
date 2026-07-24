#!/usr/bin/env bash
set -euo pipefail

# Train the SD3.5 multi-encoder embedding adapter on one safety category.
# Usage: bash scripts/train_sd35.sh [GPU_ID] [CATEGORY]
# Example: bash scripts/train_sd35.sh 0 sexual

GPU_ID="${1:-0}"
CATEGORY="${2:-sexual}"
MODEL_PATH="${MODEL_PATH:-/home/raykr/models/stabilityai/stable-diffusion-3.5-large}"
TRAIN_CSV="${TRAIN_CSV:-datasets/train/train_${CATEGORY}_toxic.csv}"
OUT_DIR="${OUT_DIR:-outputs/sd3/sd35_${CATEGORY}}"

if [[ "$GPU_ID" == *","* ]]; then
    echo "sd3/train_sd35.py is single-process; pass exactly one GPU ID." >&2
    exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python sd3/train_sd35.py \
    --model_path "$MODEL_PATH" \
    --train_csv "$TRAIN_CSV" \
    --category "$CATEGORY" \
    --out_dir "$OUT_DIR" \
    --dtype bf16 \
    --train_mode all
