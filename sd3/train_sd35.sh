
# 全训
# CUDA_VISIBLE_DEVICES=0 python sd3/train_sd35.py \
#   --train_csv datasets/train/train_sexual_toxic.csv \
#   --category sexual \
#   --out_dir outputs/sd3/sd35_sexual \
#   --train_mode all \
#   --margin 0.3 --lambda_tri 1.0 --lambda_align 0.5 --lambda_benign 0.1 \
#   --dtype bf16 --batch_size 8 --num_epochs 20 --lr 5e-4 \
#   --rank_t5 64 --rank_clip_g 32 --rank_clip_l 32


CUDA_VISIBLE_DEVICES=1 python sd3/train_sd35.py \
  --train_csv datasets/train/train_violent_toxic.csv \
  --category violent \
  --out_dir outputs/sd3/sd35_violent \
  --train_mode all \
  --margin 0.3 --lambda_tri 1.0 --lambda_align 0.5 --lambda_benign 0.1 \
  --dtype bf16 --batch_size 8 --num_epochs 20 --lr 5e-4 \
  --rank_t5 64 --rank_clip_g 32 --rank_clip_l 32 &


CUDA_VISIBLE_DEVICES=2 python sd3/train_sd35.py \
  --train_csv datasets/train/train_political_toxic.csv \
  --category political \
  --out_dir outputs/sd3/sd35_political \
  --train_mode all \
  --margin 0.3 --lambda_tri 1.0 --lambda_align 0.5 --lambda_benign 0.1 \
  --dtype bf16 --batch_size 8 --num_epochs 20 --lr 5e-4 \
  --rank_t5 64 --rank_clip_g 32 --rank_clip_l 32 &

CUDA_VISIBLE_DEVICES=3 python sd3/train_sd35.py \
  --train_csv datasets/train/train_disturbing_toxic.csv \
  --category disturbing \
  --out_dir outputs/sd3/sd35_disturbing \
  --train_mode all \
  --margin 0.3 --lambda_tri 1.0 --lambda_align 0.5 --lambda_benign 0.1 \
  --dtype bf16 --batch_size 8 --num_epochs 20 --lr 5e-4 \
  --rank_t5 64 --rank_clip_g 32 --rank_clip_l 32 &

wait