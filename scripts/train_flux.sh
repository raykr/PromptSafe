#!/bin/bash

# Flux模型训练脚本示例
# 使用方法: bash scripts/train_flux.sh

python main_flux.py \
    --pretrained_model_name_or_path "/home/raykr/models/black-forest-labs/FLUX.1-dev" \
    --clip_model_path "/home/raykr/models/openai/clip-vit-large-patch14" \
    --train_data_csv datasets/train/train_sexual_toxic.csv \
    --placeholder_token "<safety>" \
    --initializer_token "safe" \
    --output_dir outputs/flux_sexual_toxic \
    --trainer two_loss \
    --train_batch_size 2 \
    --learning_rate 1e-4 \
    --max_train_steps 5000 \
    --save_steps 500 \
    --validation_steps 500 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --lambda_triplet 0.7 \
    --margin_coef 0.1 \
    --num_vectors 1 \
    --position end \
    --repeats 100
