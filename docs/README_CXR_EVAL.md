# CXR Multi-View Generation — Inference & Evaluation

## Overview

`test_omnigen_cxr.py` is a comprehensive script that:
1. **Generates** multi-view CXR images using OmniGen + LoRA, leveraging multi-GPU data parallelism.
2. **Evaluates** generation quality against Ground Truth using the same 4 metrics reported in the SV-DRR paper: **SSIM↑, PSNR↑, LPIPS↓, FID↓**.

## Dependencies

The script requires the following packages beyond the base OmniGen environment:

```bash
# Evaluation metrics
pip install scikit-image        # SSIM, PSNR
pip install lpips               # LPIPS (AlexNet backbone)
pip install torchmetrics[image] # FID (InceptionV3)
```

## Quick Start

```bash
cd /raid/home/CAMCA/hj880/wt/code/cxr_synth/gen_code

# Mini-test: 100 samples, 1 GPU (quick sanity check)
CUDA_VISIBLE_DEVICES=2 python3 test_omnigen_cxr.py \
    --model_path Shitao/OmniGen-v1 \
    --lora_path ./results/cxr_finetune_lora/checkpoints/0000500 \
    --jsonl_path /raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/anno/cxr_synth_anno_test.jsonl \
    --output_dir /raid/home/CAMCA/hj880/wt/output/cxr_mini_test \
    --max_samples 100

# Full evaluation: all ~24k samples, 2 GPUs
CUDA_VISIBLE_DEVICES=2,3 python3 test_omnigen_cxr.py \
    --model_path Shitao/OmniGen-v1 \
    --lora_path ./results/cxr_finetune_lora/checkpoints/0000500 \
    --jsonl_path /raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/anno/cxr_synth_anno_test.jsonl \
    --output_dir /raid/home/CAMCA/hj880/wt/output/cxr_synth_results \
    --num_gpus 2 --batch_size 4

# Or simply use the launch script:
bash lanuch/test.sh
bash lanuch/test.sh --max_samples 100  # override
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `Shitao/OmniGen-v1` | Base model (HF repo or local path) |
| `--lora_path` | `None` | LoRA adapter checkpoint folder |
| `--jsonl_path` | (required) | Test set JSONL annotation file |
| `--output_dir` | (required) | Where generated images & report are saved |
| `--batch_size` | `4` | Per-GPU batch size |
| `--num_gpus` | `1` | Number of GPUs for parallel generation |
| `--max_samples` | `None` | Limit to first N samples (mini-test) |
| `--skip_generation` | `False` | Only run evaluation (images already exist) |
| `--skip_eval` | `False` | Only generate, do not compute metrics |

## Output Structure

```
output_dir/
├── LIDC-IDRI-0030/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── LIDC-IDRI-0089/
│   └── ...
├── ... (16 patient directories)
├── eval_config.json          # Run configuration snapshot
├── gen_stats_gpu0.json       # Per-GPU generation statistics
├── gen_stats_gpu1.json
├── metrics_report.json       # ← Final evaluation results
└── run.log                   # Execution log
```

## Viewing Results

```bash
# Print the metrics report
cat output_dir/metrics_report.json | python3 -m json.tool

# Expected format:
# {
#   "SSIM": 0.8234,
#   "PSNR": 28.51,
#   "LPIPS": 0.0712,
#   "FID": 42.15,
#   "num_evaluated": 23984,
#   "per_patient": {
#     "LIDC-IDRI-0030": { "SSIM": ..., "PSNR": ..., "LPIPS": ..., "count": 1499 },
#     ...
#   }
# }
```

## Comparing Checkpoints

```bash
for STEP in 0000500 0001000 0001500 0002000; do
    CUDA_VISIBLE_DEVICES=2,3 python3 test_omnigen_cxr.py \
        --lora_path ./results/cxr_finetune_lora/checkpoints/${STEP} \
        --jsonl_path .../cxr_synth_anno_test.jsonl \
        --output_dir .../results_ckpt${STEP} \
        --num_gpus 2 --max_samples 200
done

# Then compare:
for d in .../results_ckpt*; do
    echo "=== $(basename $d) ==="
    python3 -c "import json; r=json.load(open('$d/metrics_report.json')); print(f'SSIM={r[\"SSIM\"]:.4f} PSNR={r[\"PSNR\"]:.2f} LPIPS={r[\"LPIPS\"]:.4f} FID={r[\"FID\"]:.2f}')"
done
```

## Metric Computation Notes

| Metric | Library | Input Format | Notes |
|---|---|---|---|
| **SSIM** | `skimage.metrics.structural_similarity` | Grayscale, uint8, data_range=255 | Standard in radiology papers |
| **PSNR** | `skimage.metrics.peak_signal_noise_ratio` | Grayscale, uint8, data_range=255 | |
| **LPIPS** | `lpips.LPIPS(net='alex')` | RGB float32 [-1, 1] | AlexNet backbone (default) |
| **FID** | `torchmetrics.FrechetInceptionDistance` | RGB uint8 [0, 255] | InceptionV3, 2048-d features |

Since the SV-DRR repository does not open-source its evaluation code, we use the standard libraries listed above (which are the de-facto choices in the medical image generation community) to ensure fair comparison with their reported numbers.

## Batch Inference Notes

- OmniGen's `use_input_image_size_as_output=True` only supports single-image mode (asserts `isinstance(prompt, str)`).
- Since all CXR images are 256×256, we pass `height=256, width=256` explicitly for batch mode — the result is identical.
- When `batch_size=1`, the script automatically uses the single-item code path with `use_input_image_size_as_output=True`.
