# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/sd3/sd35_sexual/adapter_all
# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/sd3/sd35_sexual/baseline
# python metric.py "results/sd3/sd35_sexual/*" --output results/sd3/sd35_sexual/results.txt


# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/sd3/sd35_violent/adapter_all
# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/sd3/sd35_violent/baseline
# python metric.py "results/sd3/sd35_violent/*" --output results/sd3/sd35_violent/results.txt


# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/sd3/sd35_political/adapter_all
# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/sd3/sd35_political/baseline
# python metric.py "results/sd3/sd35_political/*" --output results/sd3/sd35_political/results.txt


# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/sd3/sd35_disturbing/adapter_all
# CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/sd3/sd35_disturbing/baseline
# python metric.py "results/sd3/sd35_disturbing/*" --output results/sd3/sd35_disturbing/results.txt


CUDA_VISIBLE_DEVICES=1 python checkers/compute_clip.py \
    --csv_path datasets/test/test_benign_200.csv \
    --image_dir results/sd3/sd35_benign/adapter_all \
    --output results/sd3/sd35_benign/clip_adapter_all.json
CUDA_VISIBLE_DEVICES=1 python checkers/compute_lpips.py \
    --gen_dir results/sd3/sd35_benign/adapter_all \
    --ori_dir datasets/test/test_benign_200 \
    --output results/sd3/sd35_benign/lpips_adapter_all.json

CUDA_VISIBLE_DEVICES=1 python checkers/compute_clip.py \
    --csv_path datasets/test/test_benign_200.csv \
    --image_dir results/sd3/sd35_benign/baseline \
    --output results/sd3/sd35_benign/clip_baseline.json
CUDA_VISIBLE_DEVICES=1 python checkers/compute_lpips.py \
    --gen_dir results/sd3/sd35_benign/baseline \
    --ori_dir datasets/test/test_benign_200 \
    --output results/sd3/sd35_benign/lpips_baseline.json

python metric.py "results/sd3/sd35_benign/*" --output results/sd3/sd35_benign/results.txt