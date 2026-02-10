# CUDA_VISIBLE_DEVICES=1 python multi_encoder/infer_flux_schemeB_all_csv.py \
#   --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
#   --adapter_path outputs/flux/flux_emb_all_sexual/flux_emb_adapter_all_final.safetensors \
#   --csv_path datasets/test/pg_sexual_toxic_demo.csv \
#   --out_dir outputs/flux/infer_flux_schemeB_all_sexual_demo \
#   --dtype bf16 --steps 50 --guidance 3.5 \
#   --height 768 --width 1360 \
#   --scale_clip 0.2 --scale_t5 0.2 --max_delta_ratio 0.3 \
#   --max_sequence_length 256 --clip_seq_len 77 \
#   --freeze_negative --save_rewritten_ref


# CUDA_VISIBLE_DEVICES=0 python multi_encoder/infer_flux_schemeB_all_csv_with_pooled.py \
#   --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
#   --adapter_path outputs/flux/flux_all_triplet_pooled_sexual/flux_emb_adapter_all_triplet_pooled_final.safetensors \
#   --csv_path datasets/test/pg_sexual_toxic_demo.csv \
#   --out_dir results/flux/infer_flux_all_triplet_pooled_sexual_demo \
#   --dtype bf16 --steps 50 --guidance 3.5 \
#   --scale_clip 0.5 --scale_t5 0.5 --scale_pooled 1.0 \
#   --max_delta_ratio_pe 0.3 --max_delta_ratio_pooled 0.5 \
#   --save_rewritten_ref


# python multi_encoder/infer_flux_defense.py \
#   --model_path "/home/raykr/models/black-forest-labs/FLUX.1-dev" \
#   --adapter_path "./outputs/flux/flux_defense_sexual/flux_emb_adapter_all_final.safetensors" \
#   --csv_path "./datasets/test/pg_sexual_toxic_demo.csv" \
#   --out_dir "./results/flux/infer_flux_defense_sexual" \
#   --dtype bf16 \
#   --scale_t5 2.0 \
#   --scale_clip 1.5 \
#   --save_baseline \
#   --save_adapter \


#!/bin/bash

# MODEL=/home/raykr/models/black-forest-labs/FLUX.1-dev
# ADAPTER=outputs/flux/flux_v2_sexual/flux_emb_adapter_all_final.safetensors

# 单条 prompt 推理（生成一张图）
# CUDA_VISIBLE_DEVICES=1 python multi_encoder/infer_flux_adapter.py \
#   --model_path $MODEL \
#   --adapter_path $ADAPTER \
#   --prompt "nude man" \
#   --negative_prompt "" \
#   --scale_clip 2.0 \
#   --scale_t5 2.0 \
#   --steps 28 \
#   --cfg 7.0 \
#   --out_dir ./results/flux/infer_flux_v2_sexual

# CSV 推理（仿照 sd3/infer_sd35.py）：每行生成 baseline 与 adapter 两张图，保存到 baseline/ 与 adapter_all/
# CUDA_VISIBLE_DEVICES=1 python multi_encoder/infer_flux.py \
#   --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
#   --adapter_path outputs/flux/flux_sexual/flux_emb_adapter_all_final.safetensors \
#   --csv_path datasets/test/pg_sexual_toxic.csv \
#   --out_dir results/flux/flux_sexual \
#   --dtype bf16 \
#   --steps 28 \
#   --cfg 7.0 \
#   --scale_clip 2.0 \
#   --scale_t5 2.0 \
#   --max_len_t5 256

# CUDA_VISIBLE_DEVICES=3 python multi_encoder/infer_flux.py \
#   --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
#   --adapter_path outputs/flux/flux_violent/flux_emb_adapter_all_final.safetensors \
#   --csv_path datasets/test/pg_violent_toxic.csv \
#   --out_dir results/flux/flux_violent \
#   --dtype bf16 \
#   --steps 28 \
#   --cfg 7.0 \
#   --scale_clip 2.0 \
#   --scale_t5 2.0 \
#   --max_len_t5 256 &

# CUDA_VISIBLE_DEVICES=5 python multi_encoder/infer_flux.py \
#   --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
#   --adapter_path outputs/flux/flux_political/flux_emb_adapter_all_final.safetensors \
#   --csv_path datasets/test/pg_political_toxic.csv \
#   --out_dir results/flux/flux_political \
#   --dtype bf16 \
#   --steps 28 \
#   --cfg 7.0 \
#   --scale_clip 2.0 \
#   --scale_t5 2.0 \
#   --max_len_t5 256 &


# CUDA_VISIBLE_DEVICES=6 python multi_encoder/infer_flux.py \
#   --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
#   --adapter_path outputs/flux/flux_disturbing/flux_emb_adapter_all_final.safetensors \
#   --csv_path datasets/test/pg_disturbing_toxic.csv \
#   --out_dir results/flux/flux_disturbing \
#   --dtype bf16 \
#   --steps 28 \
#   --cfg 7.0 \
#   --scale_clip 2.0 \
#   --scale_t5 2.0 \
#   --max_len_t5 256 &

# wait

CUDA_VISIBLE_DEVICES=1 python multi_encoder/infer_flux.py \
  --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
  --adapter_path outputs/flux/flux_sexual/flux_emb_adapter_all_final.safetensors \
  --csv_path datasets/test/test_benign_200.csv \
  --out_dir results/flux/flux_benign \
  --dtype bf16 \
  --steps 28 \
  --cfg 7.0 \
  --scale_clip 2.0 \
  --scale_t5 2.0 \
  --max_len_t5 256