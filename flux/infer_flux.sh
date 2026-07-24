#!/usr/bin/env bash
set -euo pipefail

# Generate matched baseline and adapter images for all VEIL categories.
MODEL_PATH="${MODEL_PATH:-/home/raykr/models/black-forest-labs/FLUX.1-dev}"
CATEGORIES=(benign sexual violent political disturbing)
GPUS=(5 1 2 3 4)
ADAPTER_CATEGORIES=(sexual sexual violent political disturbing)

for index in "${!CATEGORIES[@]}"; do
  category="${CATEGORIES[$index]}"
  adapter_category="${ADAPTER_CATEGORIES[$index]}"

  CUDA_VISIBLE_DEVICES="${GPUS[$index]}" python flux/infer_flux.py \
    --model_path "$MODEL_PATH" \
    --adapter_path "outputs/flux/flux_${adapter_category}/flux_emb_adapter_all_final.safetensors" \
    --csv_path "datasets/test/VEIL/${category}.csv" \
    --out_dir "results/flux/VEIL/${category}" \
    --dtype bf16 \
    --steps 28 \
    --cfg 7.0 \
    --scale_clip 2.0 \
    --scale_t5 2.0 \
    --max_len_t5 256 &
done

wait
