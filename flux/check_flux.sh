CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/flux/flux_sexual/adapter_all
CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/flux/flux_sexual/baseline
python metric.py "results/flux/flux_sexual/*" --output results/flux/flux_sexual/results.txt


CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/flux/flux_violent/adapter_all
CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/flux/flux_violent/baseline
python metric.py "results/flux/flux_violent/*" --output results/flux/flux_violent/results.txt


CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/flux/flux_political/adapter_all
CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/flux/flux_political/baseline
python metric.py "results/flux/flux_political/*" --output results/flux/flux_political/results.txt


CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/flux/flux_disturbing/adapter_all
CUDA_VISIBLE_DEVICES=1 python checkers/ud_checker.py --images_dir results/flux/flux_disturbing/baseline
python metric.py "results/flux/flux_disturbing/*" --output results/flux/flux_disturbing/results.txt


CUDA_VISIBLE_DEVICES=1 python checkers/compute_clip.py \
    --csv_path datasets/test/test_benign_200.csv \
    --image_dir results/flux/flux_benign/adapter_all \
    --output results/flux/flux_benign/clip_adapter_all.json
CUDA_VISIBLE_DEVICES=1 python checkers/compute_lpips.py \
    --gen_dir results/flux/flux_benign/adapter_all \
    --ori_dir datasets/test/test_benign_200 \
    --output results/flux/flux_benign/lpips_adapter_all.json

CUDA_VISIBLE_DEVICES=1 python checkers/compute_clip.py \
    --csv_path datasets/test/test_benign_200.csv \
    --image_dir results/flux/flux_benign/baseline \
    --output results/flux/flux_benign/clip_baseline.json
CUDA_VISIBLE_DEVICES=1 python checkers/compute_lpips.py \
    --gen_dir results/flux/flux_benign/baseline \
    --ori_dir datasets/test/test_benign_200 \
    --output results/flux/flux_benign/lpips_baseline.json

python metric.py "results/flux/flux_benign/*" --output results/flux/flux_benign/results.txt