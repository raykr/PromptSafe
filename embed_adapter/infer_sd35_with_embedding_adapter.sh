#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CATEGORY="${2:-sexual}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python embed_adapter/infer_sd35_with_embedding_adapter.py \
  --model_path "${MODEL_PATH:-/home/raykr/models/stabilityai/stable-diffusion-3.5-large}" \
  --adapter_path "outputs/sd35_embedding_adapter_${CATEGORY}/embedding_adapter_final.safetensors" \
  --test_csv "datasets/train/train_${CATEGORY}_toxic.csv" \
  --label_filter toxic \
  --limit 50 \
  --out_dir "results/sd35_embedding_adapter_${CATEGORY}" \
  --dtype bf16 \
  --scale 1.0
