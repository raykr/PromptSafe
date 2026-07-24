#!/usr/bin/env bash
set -euo pipefail

# Generate matched baseline and adapter images for all VEIL categories.
MODEL_PATH="${MODEL_PATH:-/home/raykr/models/stabilityai/stable-diffusion-3.5-large}"
CATEGORIES=(benign sexual violent political disturbing)
GPUS=(0 5 2 3 4)
ADAPTER_CATEGORIES=(sexual sexual violent political disturbing)

for index in "${!CATEGORIES[@]}"; do
  category="${CATEGORIES[$index]}"
  adapter_category="${ADAPTER_CATEGORIES[$index]}"

  CUDA_VISIBLE_DEVICES="${GPUS[$index]}" python sd3/infer_sd35.py \
    --model_path "$MODEL_PATH" \
    --adapter_path "outputs/sd3/sd35_${adapter_category}/multi_emb_adapter_all_final.safetensors" \
    --csv_path "datasets/test/VEIL/${category}.csv" \
    --out_dir "results/sd3/VEIL/${category}" \
    --dtype bf16 \
    --steps 28 \
    --cfg 7.0 \
    --scale_t5 10.0 \
    --scale_clip_g 5.0 \
    --scale_clip_l 5.0 \
    --max_len_t5 256 \
    --freeze_negative &
done

wait
