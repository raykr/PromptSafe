#!/bin/bash

# Flux模型训练脚本示例（支持多 GPU）
# 使用方法: 
#   bash scripts/train_flux.sh [GPU_IDS]
# 
# 示例:
#   bash scripts/train_flux.sh 5,6      # 使用 GPU 5 和 6（2个GPU）
#   bash scripts/train_flux.sh 5,6,7,8  # 使用 GPU 5, 6, 7, 8（4个GPU）
#   bash scripts/train_flux.sh 5         # 只使用 GPU 5（单GPU）

# 默认使用 GPU 5，如果提供了参数则使用指定的 GPU
if [ -z "$1" ]; then
    GPU_IDS="5"
    echo "未指定 GPU，默认使用 GPU 5"
else
    GPU_IDS="$1"
fi

# 计算 GPU 数量
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "=========================================="
echo "多 GPU 训练配置"
echo "=========================================="
echo "使用 GPU: $GPU_IDS"
echo "GPU 数量: $NUM_GPUS"
echo "每个 GPU 的 batch size: 1"
echo "总 batch size: $((1 * NUM_GPUS))"
echo "=========================================="

# 使用 accelerate launch 进行多 GPU 训练
CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port=29500 \
    main_flux.py \
    --pretrained_model_name_or_path "/home/beihang/jzl/models/black-forest-labs/FLUX.1-dev" \
    --clip_model_path "/home/beihang/jzl/models/openai/clip-vit-large-patch14" \
    --train_data_csv datasets/train/train_sexual_toxic.csv \
    --placeholder_token "<safety>" \
    --initializer_token "safe" \
    --output_dir outputs/flux_sexual_toxic_1 \
    --trainer two_loss \
    --train_batch_size 1 \
    --learning_rate 1e-4 \
    --max_train_steps 5000 \
    --save_steps 500 \
    --validation_steps 500 \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 5 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --lambda_triplet 0.9 \
    --margin_coef 0.1 \
    --num_vectors 4 \
    --position end \
    --repeats 10
    # 如果需要从检查点恢复训练，添加以下参数：
    # --resume_from_checkpoint "latest"  # 自动使用最新的检查点
    # --resume_from_checkpoint "checkpoint-1000"  # 使用指定的检查点
    # --resume_from_checkpoint "/path/to/checkpoint-1000"  # 使用绝对路径

# 如果仍然 OOM，可以尝试以下优化：
# 1. 减少 batch size: 将 --train_batch_size 1 改为更小的值（但已经是 1 了）
# 2. 增加 gradient accumulation: 添加 --gradient_accumulation_steps 2
# 3. 使用更多 GPU: bash scripts/train_flux.sh 5,6,7,8
