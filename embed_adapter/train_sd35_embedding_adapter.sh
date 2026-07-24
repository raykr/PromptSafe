#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CATEGORY="${2:-sexual}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python embed_adapter/train_sd35_embedding_adapter.py \
  --model_path "${MODEL_PATH:-/home/raykr/models/stabilityai/stable-diffusion-3.5-large}" \
  --train_csv "datasets/train/train_${CATEGORY}_toxic.csv" \
  --out_dir "outputs/sd35_embedding_adapter_${CATEGORY}" \
  --dtype bf16 \
  --max_length 256 \
  --batch_size 8 \
  --num_epochs 20 \
  --lr 5e-4 \
  --rank 64 \
  --benign_mode rewritten
