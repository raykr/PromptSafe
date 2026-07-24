#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CATEGORY="${2:-sexual}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python safe_adapter/train_sd35_safe_adapter.py \
  --model_path "${MODEL_PATH:-/home/raykr/models/stabilityai/stable-diffusion-3.5-large}" \
  --train_csv "datasets/train/train_${CATEGORY}_toxic.csv" \
  --out_dir "outputs/sd35_adapter_${CATEGORY}" \
  --max_length 256 \
  --batch_size 8 \
  --num_epochs 500 \
  --lr 5e-4 \
  --rank 256 \
  --dtype bf16 \
  --lambda_tri 1.0 \
  --lambda_align 0.5 \
  --lambda_benign 0.1 \
  --margin 0.2
