#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CATEGORY="${2:-sexual}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python safe_adapter/infer_sd35_safe_adapter.py \
  --model_path "${MODEL_PATH:-/home/raykr/models/stabilityai/stable-diffusion-3.5-large}" \
  --adapter_ckpt "outputs/sd35_adapter_${CATEGORY}/safe_adapter_final.pt" \
  --test_csv "datasets/train/train_${CATEGORY}_toxic.csv" \
  --label_filter toxic \
  --out_dir "results/sd35_safeadapter_${CATEGORY}" \
  --dtype bf16 \
  --scale 1.0
