
infer_ours_merged() {
	local CUDA_DEVICE=$1
	local DATASET_NAME=$2
	local EXPERIMENT_NAME=$3
	local CKPT_SEXUAL_PATH=$4
	local CKPT_VIOLENT_PATH=$5
	local CKPT_POLITICAL_PATH=$6
	local CKPT_DISTURBING_PATH=$7
	local PREDICTOR_PATH=${8:-$PREDICTOR_PATH}

	echo "Running inference for merged model with dynamic prompts..."
	echo Experiment Name: $EXPERIMENT_NAME

	CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python evaluate.py \
		--input_files datasets/test/$DATASET_NAME.csv \
		--prompt_field "prompt" \
		--output_dir results/$DATASET_NAME/$EXPERIMENT_NAME \
		--predictor_path $PREDICTOR_PATH \
		--model_path $MODEL_PATH \
		--safe_embedding_paths $CKPT_SEXUAL_PATH $CKPT_VIOLENT_PATH $CKPT_POLITICAL_PATH $CKPT_DISTURBING_PATH \
		--safe_token "<sexual>" "<violent>" "<political>" "<disturbing>" \
		--position "start" \
		--num_inference_steps 28 \
		--guidance_scale 7.5 \
		--start_index 0 \
		--pipeline_type $PIPELINE_TYPE \
		--batch_size $BATCH_SIZE \
		--predict_type "realvalue" \
		--enbale_detactive
}

infer_ours_single() {
	local CUDA_DEVICE=$1
	local DATASET_NAME=$2
	local EXPERIMENT_NAME=$3
	local CKPT_PATH=$4
	local TOKEN=$5
	local PREDICTOR_PATH=${6:-$PREDICTOR_PATH}

	CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python evaluate.py \
		--input_files datasets/test/$DATASET_NAME.csv \
		--prompt_field "prompt" \
		--output_dir results/$DATASET_NAME/$EXPERIMENT_NAME \
		--predictor_path $PREDICTOR_PATH \
		--model_path $MODEL_PATH \
		--safe_embedding_paths $CKPT_PATH \
		--safe_token $TOKEN \
		--position "start" \
		--num_inference_steps 28 \
		--guidance_scale 7.5 \
		--start_index 0 \
		--pipeline_type $PIPELINE_TYPE \
		--batch_size $BATCH_SIZE \
		--predict_type "realvalue" \
		--enbale_detactive
}

eval_ours() {
	local DATESET_TYPE=$1
	local INFER_TYPE=$2
	local DATASET_NAME=$3
	local CUDA_DEVICE=$4
	local SUFFIX=$5
	local STEP=$6
	local MODEL_PATH=${7:-"/home/raykr/models/CompVis/stable-diffusion-v1-4"}
	local PIPELINE_TYPE=${8:"ours_dynamic"}
	local BATCH_SIZE=${9:-26}
	local EXT_PREFIX=${10:-""}
	local PREDICTOR_PATH=${11:-$PREDICTOR_PATH}

    local CKPT_SEXUAL_PATH="out/train_sexual_toxic-$SUFFIX/learned_embeds-steps-$STEP.safetensors"
    local CKPT_VIOLENT_PATH="out/train_violent_toxic-$SUFFIX/learned_embeds-steps-$STEP.safetensors"
    local CKPT_POLITICAL_PATH="out/train_political_toxic-$SUFFIX/learned_embeds-steps-$STEP.safetensors"
    local CKPT_DISTURBING_PATH="out/train_disturbing_toxic-$SUFFIX/learned_embeds-steps-$STEP.safetensors"
    local EXPERIMENT_NAME=""

	if [ "$INFER_TYPE" = "merged" ]; then
        EXPERIMENT_NAME=${EXT_PREFIX}ours-merged-$SUFFIX-$PIPELINE_TYPE-$STEP
		infer_ours_merged $CUDA_DEVICE $DATASET_NAME $EXPERIMENT_NAME \
            "out/train_sexual_toxic-$SUFFIX/learned_embeds-steps-$STEP.safetensors" \
            "out/train_violent_toxic-$SUFFIX/learned_embeds-steps-$STEP.safetensors" \
            "out/train_political_toxic-$SUFFIX/learned_embeds-steps-$STEP.safetensors" \
            "out/train_disturbing_toxic-$SUFFIX/learned_embeds-steps-$STEP.safetensors" \
			$PREDICTOR_PATH

    elif [ "$INFER_TYPE" = "sexual" ]; then
        EXPERIMENT_NAME=${EXT_PREFIX}train_sexual_toxic-$SUFFIX-$PIPELINE_TYPE-$STEP
		infer_ours_single $CUDA_DEVICE $DATASET_NAME $EXPERIMENT_NAME $CKPT_SEXUAL_PATH "<sexual>" $PREDICTOR_PATH

	elif [ "$INFER_TYPE" = "violent" ]; then
        EXPERIMENT_NAME=${EXT_PREFIX}train_violent_toxic-$SUFFIX-$PIPELINE_TYPE-$STEP
        infer_ours_single $CUDA_DEVICE $DATASET_NAME $EXPERIMENT_NAME $CKPT_VIOLENT_PATH "<violent>" $PREDICTOR_PATH

    elif [ "$INFER_TYPE" = "political" ]; then
        EXPERIMENT_NAME=${EXT_PREFIX}train_political_toxic-$SUFFIX-$PIPELINE_TYPE-$STEP
        infer_ours_single $CUDA_DEVICE $DATASET_NAME $EXPERIMENT_NAME $CKPT_POLITICAL_PATH "<political>" $PREDICTOR_PATH

    elif [ "$INFER_TYPE" = "disturbing" ]; then
        EXPERIMENT_NAME=${EXT_PREFIX}train_disturbing_toxic-$SUFFIX-$PIPELINE_TYPE-$STEP
        infer_ours_single $CUDA_DEVICE $DATASET_NAME $EXPERIMENT_NAME $CKPT_DISTURBING_PATH "<disturbing>" $PREDICTOR_PATH
    fi

	if [ "$DATESET_TYPE" = "toxic" ]; then
		CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python checkers/ud_checker.py --images_dir results/$DATASET_NAME/$EXPERIMENT_NAME
		python metric.py "results/$DATASET_NAME/*" --output results/$DATASET_NAME/results.txt

	elif [ "$DATESET_TYPE" = "benign" ]; then
		# CLIP
		CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python checkers/compute_clip.py \
			--csv_path datasets/test/$DATASET_NAME.csv \
			--image_dir results/$DATASET_NAME/$EXPERIMENT_NAME \
			--output results/$DATASET_NAME/clip_$EXPERIMENT_NAME.json
		# LPIPS
		CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python checkers/compute_lpips.py \
			--gen_dir results/$DATASET_NAME/$EXPERIMENT_NAME \
			--ori_dir datasets/test/$DATASET_NAME \
			--output results/$DATASET_NAME/lpips_$EXPERIMENT_NAME.json

		python metric.py "results/$DATASET_NAME/*" --output results/$DATASET_NAME/results.txt
	fi
}


# --------- !!! Change the following variables before running the script !!! ---------
# MODEL_PATH: Path to the pre-trained Stable Diffusion model 
MODEL_PATH="/home/raykr/models/CompVis/stable-diffusion-v1-4"
# PREDICTOR_PATH: Path to the gated network predictor model
PREDICTOR_PATH="out/gated_network_v0"
# STEP, DATETIME, TRAINER, EXTRA must match the training script
STEP=500
DATETIME=20250725
TRAINER="two_loss"
EXTRA="-0.7"
SUFFIX=$DATETIME-$TRAINER$EXTRA
# Choose the inference type: "ours_dynamic" with gated network; or "ours" without gated network
PIPELINE_TYPE="ours_dynamic" 
BATCH_SIZE=26
EXT_PREFIX=""

eval_ours toxic merged pg_sexual_toxic 1 $SUFFIX $STEP $MODEL_PATH $PIPELINE_TYPE $BATCH_SIZE $EXT_PREFIX $PREDICTOR_PATH &
eval_ours toxic merged pg_violent_toxic 1 $SUFFIX $STEP $MODEL_PATH $PIPELINE_TYPE $BATCH_SIZE $EXT_PREFIX $PREDICTOR_PATH &
wait
eval_ours toxic merged pg_political_toxic 0 $SUFFIX $STEP $MODEL_PATH $PIPELINE_TYPE $BATCH_SIZE $EXT_PREFIX $PREDICTOR_PATH &
eval_ours toxic merged pg_disturbing_toxic 1 $SUFFIX $STEP $MODEL_PATH $PIPELINE_TYPE $BATCH_SIZE $EXT_PREFIX $PREDICTOR_PATH &
wait
eval_ours benign merged test_benign_200 0 $SUFFIX $STEP $MODEL_PATH $PIPELINE_TYPE $BATCH_SIZE $EXT_PREFIX $PREDICTOR_PATH &
wait

