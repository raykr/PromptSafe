#!/usr/bin/env bash
set -euo pipefail

# Run SD3.5 inference with the trained adapter. The Python entry point writes
# matching baseline/ and adapter_all/ subdirectories in one pass.
# Usage: bash scripts/infer_sd35.sh [GPU_ID] [CATEGORY]

GPU_ID="${1:-0}"
CATEGORY="${2:-sexual}"
MODEL_PATH="${MODEL_PATH:-/home/raykr/models/stabilityai/stable-diffusion-3.5-large}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/sd3/sd35_${CATEGORY}/multi_emb_adapter_all_final.safetensors}"
CSV_PATH="${CSV_PATH:-datasets/test/VEIL/${CATEGORY}.csv}"
OUT_DIR="${OUT_DIR:-results/sd3/VEIL/${CATEGORY}}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python sd3/infer_sd35.py \
    --model_path "$MODEL_PATH" \
    --adapter_path "$ADAPTER_PATH" \
    --csv_path "$CSV_PATH" \
    --out_dir "$OUT_DIR" \
    --dtype bf16 \
    --steps 28 \
    --cfg 7.0 \
    --scale_t5 10.0 \
    --scale_clip_g 5.0 \
    --scale_clip_l 5.0 \
    --max_len_t5 256 \
    --freeze_negative
