CUDA_VISIBLE_DEVICES=6 python evaluate.py \
    --input_files datasets/test/pg_sexual_toxic.csv \
    --prompt_field "prompt" \
    --output_dir results/flux_sexual_toxic \
    --model_path "/home/beihang/jzl/models/black-forest-labs/FLUX.1-dev" \
    --safe_embedding_paths outputs/flux_sexual_toxic/learned_embeds-steps-3000.safetensors \
    --safe_token "<safety>" \
    --position "start" \
    --pipeline_type "ours_flux" \
    --batch_size 1 \
    --seed 42 \
    --num_inference_steps 10 \
    --guidance_scale 7.5


# baseline
# CUDA_VISIBLE_DEVICES=6 python evaluate.py \
#     --input_files datasets/test/pg_sexual_toxic.csv \
#     --prompt_field "prompt" \
#     --output_dir results/flux_sexual_toxic_baseline \
#     --model_path "/home/beihang/jzl/models/black-forest-labs/FLUX.1-dev" \
#     --pipeline_type "baseline_flux" \
#     --batch_size 1 \
#     --seed 42 \
#     --num_inference_steps 10 \
#     --guidance_scale 7.5