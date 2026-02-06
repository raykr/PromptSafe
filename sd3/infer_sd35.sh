


# CUDA_VISIBLE_DEVICES=6 python sd3/infer_sd35.py \
#   --model_path /home/raykr/models/stabilityai/stable-diffusion-3.5-large \
#   --adapter_path outputs/sd3/sd35_seuxal/multi_emb_adapter_all_final.safetensors \
#   --csv_path datasets/test/pg_sexual_toxic.csv \
#   --out_dir results/sd3/sd35_seuxal \
#   --dtype bf16 --steps 28 --cfg 7.0 \
#   --scale_t5 10.0 --scale_clip_g 5.0 --scale_clip_l 5.0 \
#   --max_len_t5 256 \
#   --freeze_negative


# CUDA_VISIBLE_DEVICES=3 python sd3/infer_sd35.py \
#   --model_path /home/raykr/models/stabilityai/stable-diffusion-3.5-large \
#   --adapter_path outputs/sd3/sd35_violent/multi_emb_adapter_all_final.safetensors \
#   --csv_path datasets/test/pg_violent_toxic.csv \
#   --out_dir results/sd3/sd35_violent \
#   --dtype bf16 --steps 28 --cfg 7.0 \
#   --scale_t5 10.0 --scale_clip_g 5.0 --scale_clip_l 5.0 \
#   --max_len_t5 256 \
#   --freeze_negative &

# CUDA_VISIBLE_DEVICES=4 python sd3/infer_sd35.py \
#   --model_path /home/raykr/models/stabilityai/stable-diffusion-3.5-large \
#   --adapter_path outputs/sd3/sd35_political/multi_emb_adapter_all_final.safetensors \
#   --csv_path datasets/test/pg_political_toxic.csv \
#   --out_dir results/sd3/sd35_political \
#   --dtype bf16 --steps 28 --cfg 7.0 \
#   --scale_t5 10.0 --scale_clip_g 5.0 --scale_clip_l 5.0 \
#   --max_len_t5 256 \
#   --freeze_negative &

# CUDA_VISIBLE_DEVICES=6 python sd3/infer_sd35.py \
#   --model_path /home/raykr/models/stabilityai/stable-diffusion-3.5-large \
#   --adapter_path outputs/sd3/sd35_disturbing/multi_emb_adapter_all_final.safetensors \
#   --csv_path datasets/test/pg_disturbing_toxic.csv \
#   --out_dir results/sd3/sd35_disturbing \
#   --dtype bf16 --steps 28 --cfg 7.0 \
#   --scale_t5 10.0 --scale_clip_g 5.0 --scale_clip_l 5.0 \
#   --max_len_t5 256 \
#   --freeze_negative &

# wait



CUDA_VISIBLE_DEVICES=0 python sd3/infer_sd35.py \
  --model_path /home/raykr/models/stabilityai/stable-diffusion-3.5-large \
  --adapter_path outputs/sd3/sd35_seuxal/multi_emb_adapter_all_final.safetensors \
  --csv_path datasets/test/test_benign_200.csv \
  --out_dir results/sd3/sd35_benign \
  --dtype bf16 --steps 28 --cfg 7.0 \
  --scale_t5 10.0 --scale_clip_g 5.0 --scale_clip_l 5.0 \
  --max_len_t5 256 \
  --freeze_negative