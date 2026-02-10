

# CUDA_VISIBLE_DEVICES=1 python multi_encoder/train_flux.py \
#   --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
#   --train_csv datasets/train/train_sexual_toxic.csv \
#   --out_dir outputs/flux/flux_sexual \
#   --dtype bf16

CUDA_VISIBLE_DEVICES=3 python multi_encoder/train_flux.py \
  --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
  --train_csv datasets/train/train_violent_toxic.csv \
  --out_dir outputs/flux/flux_violent \
  --dtype bf16 &

CUDA_VISIBLE_DEVICES=5 python multi_encoder/train_flux.py \
  --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
  --train_csv datasets/train/train_political_toxic.csv \
  --out_dir outputs/flux/flux_political \
  --dtype bf16 &

CUDA_VISIBLE_DEVICES=6 python multi_encoder/train_flux.py \
  --model_path /home/raykr/models/black-forest-labs/FLUX.1-dev \
  --train_csv datasets/train/train_disturbing_toxic.csv \
  --out_dir outputs/flux/flux_disturbing \
  --dtype bf16 &

wait