#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CATEGORY="${2:-sexual}"
LEARNED_PATH="${LEARNED_PATH:-outputs/sd35_t5_${CATEGORY}/learned_embeds.safetensors}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python soft_token/infer_sd35_inject_t5_soft_token.py \
  --model_path "${MODEL_PATH:-/home/raykr/models/stabilityai/stable-diffusion-3.5-large}" \
  --learned_path "$LEARNED_PATH" \
  --prompts "datasets/test/VEIL/${CATEGORY}.csv" \
  --output_dir "results/sd35_soft_token_${CATEGORY}" \
  --position prefix \
  --placeholder_token "<SAFE>" \
  --num_vectors 8 \
  --steps 28 \
  --cfg 7.5 \
  --seed 42 \
  --height 768 \
  --width 768 \
  --dtype bf16 \
  --device cuda
