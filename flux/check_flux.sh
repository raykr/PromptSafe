CLASS=(benign sexual violent political disturbing)
OUT_DIR=results/flux/VEIL

for class in "${CLASS[@]}"; do
    CUDA_VISIBLE_DEVICES=7 python checkers/ud_checker.py --images_dir $OUT_DIR/$class/adapter_all

    CUDA_VISIBLE_DEVICES=7 python checkers/compute_clip.py \
        --csv_path datasets/test/VEIL/$class.csv \
        --image_dir $OUT_DIR/$class/adapter_all \
        --output $OUT_DIR/$class/clip_adapter_all.json
    CUDA_VISIBLE_DEVICES=7 python checkers/compute_lpips.py \
        --gen_dir $OUT_DIR/$class/adapter_all \
        --ori_dir $OUT_DIR/$class/baseline \
        --output $OUT_DIR/$class/lpips_adapter_all.json

    python metric.py "$OUT_DIR/$class/*" --output $OUT_DIR/$class/results.txt
done