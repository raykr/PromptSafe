#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CATEGORY="${2:-sexual}"

CUDA_VISIBLE_DEVICES="$GPU_ID" accelerate launch soft_token/train_sd35_t5_soft_token_textonly.py \
  --model_path "${MODEL_PATH:-/home/raykr/models/stabilityai/stable-diffusion-3.5-large}" \
  --train_csv "datasets/train/train_${CATEGORY}_toxic.csv" \
  --output_dir "outputs/sd35_t5_${CATEGORY}" \
  --mixed_precision bf16 \
  --batch_size 6 \
  --max_steps 20000 \
  --position prefix \
  --placeholder_token "<SAFE>" \
  --num_vectors 8 \
  --lambda_align 0.1 \
  --lambda_preserve 0.2 \
  --lambda_triplet 0.9 \
  --margin_coef 0.1 \
  --checkpoint_steps 1000
