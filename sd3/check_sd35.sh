# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/multi_encoder/sd35_emb_all_sexual_schemeB/adapter_all
# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/multi_encoder/sd35_emb_all_sexual_schemeB/baseline
# python metric.py "results/multi_encoder/sd35_emb_all_sexual_schemeB/*" --output results/multi_encoder/sd35_emb_all_sexual_schemeB/results.txt


# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/multi_encoder/sd35_emb_all_violent_schemeB/adapter_all
# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/multi_encoder/sd35_emb_all_violent_schemeB/baseline
# python metric.py "results/multi_encoder/sd35_emb_all_violent_schemeB/*" --output results/multi_encoder/sd35_emb_all_violent_schemeB/results.txt


# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/multi_encoder/sd35_emb_all_political_schemeB/adapter_all
# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/multi_encoder/sd35_emb_all_political_schemeB/baseline
# python metric.py "results/multi_encoder/sd35_emb_all_political_schemeB/*" --output results/multi_encoder/sd35_emb_all_political_schemeB/results.txt


# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/multi_encoder/sd35_emb_all_disturbing_schemeB/adapter_all
# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/multi_encoder/sd35_emb_all_disturbing_schemeB/baseline
# python metric.py "results/multi_encoder/sd35_emb_all_disturbing_schemeB/*" --output results/multi_encoder/sd35_emb_all_disturbing_schemeB/results.txt


CUDA_VISIBLE_DEVICES=1 python checkers/compute_clip.py \
    --csv_path datasets/test/test_benign_200.csv \
    --image_dir results/multi_encoder/sd35_emb_all_benign_schemeB/adapter_all \
    --output results/multi_encoder/sd35_emb_all_benign_schemeB/clip_adapter_all.json
CUDA_VISIBLE_DEVICES=1 python checkers/compute_lpips.py \
    --gen_dir results/multi_encoder/sd35_emb_all_benign_schemeB/adapter_all \
    --ori_dir datasets/test/test_benign_200 \
    --output results/multi_encoder/sd35_emb_all_benign_schemeB/lpips_adapter_all.json

CUDA_VISIBLE_DEVICES=1 python checkers/compute_clip.py \
    --csv_path datasets/test/test_benign_200.csv \
    --image_dir results/multi_encoder/sd35_emb_all_benign_schemeB/baseline \
    --output results/multi_encoder/sd35_emb_all_benign_schemeB/clip_baseline.json
CUDA_VISIBLE_DEVICES=1 python checkers/compute_lpips.py \
    --gen_dir results/multi_encoder/sd35_emb_all_benign_schemeB/baseline \
    --ori_dir datasets/test/test_benign_200 \
    --output results/multi_encoder/sd35_emb_all_benign_schemeB/lpips_baseline.json

python metric.py "results/multi_encoder/sd35_emb_all_benign_schemeB/*" --output results/multi_encoder/sd35_emb_all_benign_schemeB/results.txt