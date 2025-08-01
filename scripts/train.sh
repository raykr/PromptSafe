train() {
  local TYPE=$1
  local STEP=$2
  local DATETIME=$3
  local TRAINER=$4
  local EXTRA=$5
  local MODEL_PATH=$6
  local CLIP_MODEL_PATH=$7

  DATASET_NAME="train_${TYPE}_toxic"
  OUR_DIR="./out/$DATASET_NAME-$DATETIME-$TRAINER$EXTRA"
  CUDA_DEVICE=0,1

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --main_process_port=29501 \
    ./main.py \
    --pretrained_model_name_or_path=$MODEL_PATH \
    --clip_model_path=$CLIP_MODEL_PATH \
    --train_data_csv="./datasets/train/$DATASET_NAME.csv" \
    --placeholder_token="<${TYPE}>" \
    --initializer_token="safe" \
    --position="start" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=$STEP \
    --learning_rate=5.0e-04 \
    --lambda_lr=1e-3 \
    --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="$OUR_DIR" \
    --num_vectors=1 \
    --seed=42 \
    --lambda_align=0.5 \
    --lambda_triplet=0.9 \
    --lambda_benign=0.1 \
    --margin_coef=0.1 \
    --trainer="$TRAINER" \
    --resume_from_checkpoint "latest" \
    --repeats=2 \
    --enable_xformers_memory_efficient_attention \
    # --gradient_checkpointing
}

# ------------------------ sdv14 twoloss --------------------------------
STEP=500
DATETIME=20250725
TRAINER="two_loss"
EXTRA="-0.7"
MODEL_PATH="/home/raykr/models/CompVis/stable-diffusion-v1-4"
CLIP_MODEL_PATH="/home/raykr/models/openai/clip-vit-large-patch14"

train sexual $STEP $DATETIME $TRAINER $EXTRA $MODEL_PATH $CLIP_MODEL_PATH
train violent $STEP $DATETIME $TRAINER $EXTRA $MODEL_PATH $CLIP_MODEL_PATH
train political $STEP $DATETIME $TRAINER $EXTRA $MODEL_PATH $CLIP_MODEL_PATH
train disturbing $STEP $DATETIME $TRAINER $EXTRA $MODEL_PATH $CLIP_MODEL_PATH
