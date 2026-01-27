# Flux模型训练指南

本文档说明如何使用Flux模型进行训练，以及Flux版本与SD1.4版本的主要差异。

## 主要差异

### 1. Text Encoder差异

**SD1.4版本：**
- 使用 `CLIPTextModel` 作为text encoder
- 使用 `CLIPTokenizer`
- Tokenizer的max_length为77
- Text encoder输出有`pooler_output`属性

**Flux版本：**
- 使用 `T5EncoderModel` 作为text encoder（T5-XXL）
- 使用 `T5Tokenizer`
- Tokenizer的max_length为512（T5默认）
- Text encoder只输出`last_hidden_state`，没有`pooler_output`
- Flux可能同时使用CLIP和T5两个text encoder（dual encoder模式）

### 2. UNet/Transformer差异

**SD1.4版本：**
- 使用 `UNet2DConditionModel`
- Latent shape: (B, 4, 64, 64)

**Flux版本：**
- 使用 `Transformer2DModel`（Flux.1-dev）
- Latent shape: (B, 16, 64, 64) 或其他（取决于具体模型）

### 3. Embedding层差异

**SD1.4版本：**
- Embedding层在 `text_encoder.text_model.embeddings.token_embedding`
- 冻结策略：冻结encoder层，只训练embedding

**Flux版本：**
- Embedding层在 `text_encoder.encoder.embed_tokens`
- 冻结策略：冻结encoder层和final_layer_norm，只训练embedding

## 使用方法

### 基本训练命令

```bash
python main_flux.py \
    --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --clip_model_path "openai/clip-vit-large-patch14" \
    --train_data_csv datasets/train/train_sexual_toxic.csv \
    --placeholder_token "<safety>" \
    --initializer_token "safe" \
    --output_dir outputs/flux_training \
    --trainer two_loss \
    --train_batch_size 4 \
    --learning_rate 1e-4 \
    --max_train_steps 5000 \
    --save_steps 500 \
    --validation_steps 500 \
    --mixed_precision bf16
```

### 使用Dual Encoder（CLIP + T5）

如果Flux模型同时使用CLIP和T5两个text encoder：

```bash
python main_flux.py \
    --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --use_dual_encoder \
    --clip_model_path "openai/clip-vit-large-patch14" \
    --train_data_csv datasets/train/train_sexual_toxic.csv \
    --placeholder_token "<safety>" \
    --initializer_token "safe" \
    --output_dir outputs/flux_training \
    --trainer two_loss \
    --train_batch_size 4 \
    --learning_rate 1e-4 \
    --max_train_steps 5000
```

### 参数说明

#### Flux特有参数

- `--use_dual_encoder`: 是否使用dual encoder模式（CLIP + T5），默认False
- `--tokenizer_name`: T5 tokenizer路径（如果与模型路径不同）

#### 其他参数

大部分参数与SD1.4版本相同，包括：
- `--trainer`: 选择训练器（`two_loss` 或 `three_loss`）
- `--lambda_triplet`, `--lambda_align`, `--lambda_benign`: 损失函数权重
- `--margin_coef`: Triplet loss的margin系数
- `--train_batch_size`: 批次大小（Flux模型较大，建议使用较小的batch size）
- `--mixed_precision`: 混合精度（建议使用`bf16`）

## 代码结构

```
PromptSafe/
├── main_flux.py          # Flux训练入口
├── trainer_flux.py       # Flux训练器实现
├── main.py              # SD1.4训练入口（原版）
├── trainer.py           # SD1.4训练器（原版）
└── README_FLUX.md       # 本文档
```

## 训练器类型

### TwoLossFluxTrainer

使用triplet loss和benign loss：

```python
loss = lambda_triplet * triplet_loss + (1 - lambda_triplet) * benign_loss
```

### ThreeLossFluxTrainer

使用align loss、triplet loss和benign loss：

```python
loss = lambda_align * align_loss + lambda_triplet * triplet_loss + lambda_benign * benign_loss
```

## 注意事项

1. **内存需求**: Flux模型（特别是T5-XXL）比SD1.4大得多，需要更多GPU内存。建议：
   - 使用较小的batch size（如2-4）
   - 启用gradient checkpointing (`--gradient_checkpointing`)
   - 使用混合精度训练 (`--mixed_precision bf16`)

2. **Tokenizer差异**: T5 tokenizer的max_length是512，而CLIP是77。代码会自动处理这个差异。

3. **模型路径**: 确保Flux模型路径正确。Flux模型结构可能因版本而异：
   - `text_encoder_2`: T5 encoder
   - `text_encoder`: CLIP encoder（如果使用dual encoder）
   - `transformer`: Transformer2DModel（不是unet）

4. **验证**: 验证功能可能需要适配Flux pipeline。当前代码会尝试使用原有的pipeline，如果失败会记录警告。

## 故障排除

### 问题1: 找不到text_encoder_2

**错误**: `OSError: Can't load tokenizer for 'xxx/text_encoder_2'`

**解决**: 检查模型路径，某些Flux变体可能将T5放在`text_encoder`子目录。代码会自动尝试多个路径。

### 问题2: Transformer API不匹配

**错误**: `TypeError: forward() got an unexpected keyword argument`

**解决**: Flux的Transformer API可能与代码中的调用方式不同。检查`trainer_flux.py`中的`predict_noise`方法，根据实际模型API调整。

### 问题3: 内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
- 减小batch size
- 启用gradient checkpointing
- 使用更小的模型变体（如量化版本）

## 示例脚本

创建一个训练脚本 `scripts/train_flux.sh`:

```bash
#!/bin/bash

python main_flux.py \
    --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --clip_model_path "openai/clip-vit-large-patch14" \
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
    --margin_coef 0.1
```

## 参考

- [Flux模型文档](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Diffusers Flux Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/flux)
- [T5模型文档](https://huggingface.co/docs/transformers/model_doc/t5)
