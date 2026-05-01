# Interpretability and Evaluation Guide

This guide explains how to run the post-generation analysis scripts for Joint Image-Mask Co-Generation. The key scientific constraint is that the test set does not have paired ground-truth segmentation masks, so these scripts do not compute Dice, IoU, or any segmentation-quality metric. The predicted mask is used as an interpretability signal and as a model-induced ROI for local image-fidelity analysis.

## 1. Workflow Overview

Recommended paper workflow:

1. Run ROI metrics on the whole dataset.
   This produces `sample_metrics.csv` and `dataset_summary.json`, giving global image fidelity and local fidelity inside/outside the predicted-mask ROI.

2. Pick representative sample IDs from `sample_metrics.csv`.
   Use the CSV to identify high-quality, median, challenging, or visually representative cases. The `sample_id` column is the safest value to pass into the visualization scripts.

3. Run the overlay visualizer on selected IDs.
   Use `--sample_ids ID1,ID2,...` to generate targeted, publication-quality figure panels instead of rendering the whole dataset.

4. Run the standalone attention heatmap extractor.
   This reruns inference only for selected samples, captures one selected self-attention layer, and saves image-token to mask-token attention overlays for architectural interpretability.

Typical data flow:

```text
test JSONL
  -> test_joint_mask.py inference output directory
      -> generated images
      -> masks/*.npz
  -> analyze_roi_metrics.py
      -> sample_metrics.csv
      -> dataset_summary.json
  -> analyze_visualization.py
      -> panorama and channel-breakdown figures
  -> launch/extract_attention_heatmap.sh
      -> selected-sample attention heatmaps only
```

## 2. Script 1: Local Error Correlation

Script: `analyze_roi_metrics.py`

### Purpose

This script measures image-generation fidelity globally and inside/outside the model-predicted ROI.

For each sample, it computes:

- `global_psnr`: PSNR between GT edited image and generated image over the whole image.
- `global_ssim`: SSIM between GT edited image and generated image over the whole image.
- `roi_inside_psnr`: PSNR only on pixels where the predicted 10-channel mask union is positive.
- `roi_inside_ssim`: Mean SSIM-map value only inside the predicted ROI.
- `roi_outside_psnr`: PSNR only on pixels outside the predicted ROI.
- `roi_outside_ssim`: Mean SSIM-map value outside the predicted ROI.

The ROI is defined as:

```python
roi = predicted_mask.sum(axis=0) > mask_threshold
```

This is a self-consistency analysis. It can support the claim that the generated image has strong local fidelity in model-induced anatomical regions, but it is not a segmentation evaluation and not causal proof.

### Arguments

Current implemented arguments:

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--jsonl_path` | Yes | None | Path to the test JSONL used for inference. Each record must contain `output_image`; `input_images` is not required for this metrics script. |
| `--inference_dir` | Yes | None | Directory containing generated images and `masks/` from `test_joint_mask.py`. Alias: `--results_dir`. |
| `--output_dir` | No | `analysis_roi_metrics` | Directory where `sample_metrics.csv` and `dataset_summary.json` will be saved. |
| `--sample_ids` | No | None | Optional comma-separated IDs to process. Matches explicit `sample_id`, explicit `id`, patient ID, view stem, or `patient_view`. |
| `--max_samples` | No | None | Process only the first N records when `--sample_ids` is not provided. |
| `--mask_threshold` | No | `0.0` | Threshold applied after summing the predicted mask channels to define the ROI. |

Path behavior:

- GT image path comes from `record["output_image"]` in the JSONL.
- Generated image path is derived as:

```text
{inference_dir}/{patient_id}/{view_file}.png
```

- Predicted mask path is derived as:

```text
{inference_dir}/masks/{patient_id}/{view_file}.npz
```

There are no `--gt_img_dir`, `--gen_img_dir`, or `--pred_mask_dir` arguments in the implemented script.

### Example Command

Run on the full test set:

```bash
python analyze_roi_metrics.py \
  --jsonl_path /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl \
  --inference_dir results/joint_mask_inference \
  --output_dir analysis_outputs/roi_metrics_full \
  --mask_threshold 0.0
```

Run on a quick subset:

```bash
python analyze_roi_metrics.py \
  --jsonl_path /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl \
  --inference_dir results/joint_mask_inference \
  --output_dir analysis_outputs/roi_metrics_smoke \
  --max_samples 20
```

Run on specific samples:

```bash
python analyze_roi_metrics.py \
  --jsonl_path /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl \
  --inference_dir results/joint_mask_inference \
  --output_dir analysis_outputs/roi_metrics_selected \
  --sample_ids LIDC-IDRI-0251_0001,LIDC-IDRI-0089_0217
```

### Expected Outputs

Output directory:

```text
analysis_outputs/roi_metrics_full/
  sample_metrics.csv
  dataset_summary.json
```

`sample_metrics.csv` contains one row per successfully processed sample. Important columns:

- `sample_id`: Stable ID to reuse with `--sample_ids`.
- `gt_path`: GT edited image path.
- `gen_path`: Generated image path.
- `mask_path`: Predicted mask `.npz` path.
- `roi_pixels`: Number of pixels inside the predicted ROI.
- `background_pixels`: Number of pixels outside the predicted ROI.
- `roi_fraction`: Fraction of image pixels inside the predicted ROI.
- `global_psnr`, `global_ssim`
- `roi_inside_psnr`, `roi_inside_ssim`
- `roi_outside_psnr`, `roi_outside_ssim`

`dataset_summary.json` contains mean, standard deviation, and sample count for each metric:

```json
{
  "num_samples": 1000,
  "metrics": {
    "global_psnr": {"mean": 0.0, "std": 0.0, "n": 1000},
    "global_ssim": {"mean": 0.0, "std": 0.0, "n": 1000},
    "roi_inside_psnr": {"mean": 0.0, "std": 0.0, "n": 1000}
  }
}
```

Use `sample_metrics.csv` for statistical testing, p-values, paired tests, and figure-case selection.

## 3. Script 2: Overlay Visualizer

Script: `analyze_visualization.py`

### Purpose

This script creates publication-quality qualitative figures from existing inference outputs.

For each selected sample, it writes:

- A panorama figure with 4 panels:
  1. Input image
  2. GT edited image
  3. Generated image
  4. Generated image with all 10 predicted mask channels overlaid

- A channel-breakdown figure with 10 panels:
  Each panel overlays only one organ channel on the generated image.

The 10 channel names are:

```text
0 Left Lung
1 Right Lung
2 Heart
3 Aorta
4 Liver
5 Stomach
6 Trachea
7 Ribs
8 Vertebrae
9 Upper Skeleton
```

### Arguments

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--jsonl_path` | Yes | None | Path to the test JSONL used for inference. Each record must contain `input_images` and `output_image`. |
| `--inference_dir` | Yes | None | Directory containing generated images and `masks/` from `test_joint_mask.py`. Alias: `--results_dir`. |
| `--output_dir` | No | `analysis_visualizations` | Directory where PNG figures will be saved. |
| `--sample_ids` | No | None | Comma-separated sample IDs to process, for example `ID1,ID2`. Matches explicit `sample_id`, explicit `id`, patient ID, view stem, or `patient_view`. |
| `--max_samples` | No | None | Process the first N samples only when `--sample_ids` is not provided. |
| `--alpha` | No | `0.4` | Overlay opacity for predicted masks. |
| `--mask_threshold` | No | `0.0` | Threshold used to binarize each predicted mask channel before overlay. |
| `--dpi` | No | `300` | Figure export DPI. Use `300` or higher for paper figures. |

### Using `--sample_ids`

The recommended source of IDs is the `sample_id` column from `sample_metrics.csv`.

Format:

```text
--sample_ids ID1,ID2,ID3
```

Example:

```text
--sample_ids LIDC-IDRI-0251_0001,LIDC-IDRI-0089_0217
```

If the JSONL does not contain an explicit `sample_id`, the scripts generate IDs using:

```text
{patient_id}_{view_stem}
```

For a GT path like:

```text
/.../img_complex_fb_256/LIDC-IDRI-0251/0001.png
```

the generated sample ID is:

```text
LIDC-IDRI-0251_0001
```

`--sample_ids` takes priority over `--max_samples`. If `--sample_ids` is provided, only matching samples are processed.

### Example Command

Target specific paper-figure cases:

```bash
python analyze_visualization.py \
  --jsonl_path /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl \
  --inference_dir results/joint_mask_inference \
  --output_dir analysis_outputs/paper_figures \
  --sample_ids LIDC-IDRI-0251_0001,LIDC-IDRI-0089_0217 \
  --alpha 0.4 \
  --dpi 300
```

Generate visualizations for the first 10 samples:

```bash
python analyze_visualization.py \
  --jsonl_path /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl \
  --inference_dir results/joint_mask_inference \
  --output_dir analysis_outputs/visualization_smoke \
  --max_samples 10
```

### Expected Outputs

Output files are saved directly under `--output_dir`:

```text
analysis_outputs/paper_figures/
  LIDC-IDRI-0251_0001_panorama.png
  LIDC-IDRI-0251_0001_channels.png
  LIDC-IDRI-0089_0217_panorama.png
  LIDC-IDRI-0089_0217_channels.png
```

File meanings:

- `*_panorama.png`: 4-panel Input / GT / Generated / Generated+Mask figure.
- `*_channels.png`: 10-panel generated-image overlay, one channel per subplot.

These PNGs are intended to be directly usable in paper figures or supplementary material.

## 4. Script 3: Attention Heatmap

Launcher: `launch/extract_attention_heatmap.sh`

Python script called by launcher: `extract_attention_heatmap.py`

### Purpose

This script is a surgical qualitative diagnostic. It reruns joint image-mask inference only for selected samples and captures one self-attention layer:

```text
model.llm.layers[layer_index].self_attn
```

It extracts the attention submatrix:

```text
Queries = generated image tokens
Keys    = generated mask tokens
```

Then it:

1. Averages attention across heads.
2. Averages across mask-token keys.
3. Reshapes image-token attention to a spatial grid.
4. Uses a 16x16 grid for the 256 image tokens.
5. Upscales the heatmap to 256x256.
6. Applies a colormap.
7. Overlays the heatmap on the generated image.

This provides architectural interpretability evidence that image tokens attend to mask tokens during joint co-generation.

This script does not compute PSNR, SSIM, FID, Dice, IoU, or any other metric. Full-dataset quantitative evaluation remains the responsibility of `test_joint_mask.py` and `analyze_roi_metrics.py`.

### Arguments

Important Python arguments exposed by the launcher:

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--jsonl_path` | Yes through launcher variable | Launcher `JSONL_PATH` | Test JSONL containing `instruction`, `input_images`, and `output_image`. |
| `--model_path` | No | `Shitao/OmniGen-v1` | Base OmniGen model path or Hugging Face repo. |
| `--lora_path` | No | Launcher `LORA_PATH` | LoRA checkpoint directory. |
| `--mask_modules_path` | Yes | Launcher `MASK_MODULES_PATH` | Path to `mask_modules.bin`. |
| `--output_dir` | No | `analysis_attention_heatmaps` | Directory where heatmaps are saved. |
| `--sample_ids` | No | None | Comma-separated selected samples or patient IDs. Examples: `LIDC-IDRI-0251,LIDC-IDRI-0657` or `LIDC-IDRI-0251_0001`. |
| `--max_samples` | No | `4` | Process first N records when `--sample_ids` is not provided. |
| `--layer_index` | No | `-1` | Which transformer layer to hook. Negative values count from the end. |
| `--inference_steps` | No | `50` | Diffusion denoising steps. |
| `--guidance_scale` | No | `2.5` | Text CFG guidance scale. |
| `--img_guidance_scale` | No | `2.0` | Image guidance scale. |
| `--seed` | No | `42` | Random seed. |
| `--device` | No | `cuda:0` | Torch device used for extraction. |
| `--heatmap_alpha` | No | `0.45` | Overlay opacity. |
| `--save_generated` | No | Off | Also save the generated image beside the heatmap. |

### Layer Selection: `--layer_index`

`--layer_index` controls which transformer self-attention layer is inspected.

Physical interpretation:

- Early layers usually emphasize local token interactions and low-level conditioning signals.
- Middle layers often show stronger semantic and spatial-layout alignment across modalities.
- Late layers can become more focused on detail rendering and final denoising refinements.

The default is:

```text
--layer_index -1
```

This means the final transformer layer. It is a reasonable first check, but it is not guaranteed to produce the cleanest cross-modality map.

For paper figures, experiment with middle-to-late layers:

```text
--layer_index 15
--layer_index 20
--layer_index -1
```

Then choose the layer that gives the clearest image-token to mask-token anatomical alignment for the selected cases. The goal is qualitative architectural evidence, so it is acceptable to inspect several layers and report the selected layer explicitly in the figure caption or appendix.

### Launcher Configuration

Open or edit:

```bash
launch/extract_attention_heatmap.sh
```

Important variables:

| Variable | Default | Description |
|---|---|---|
| `PYTHON_BIN` | `python` | Python executable. Override if needed, for example `PYTHON_BIN=python3`. |
| `MODEL_PATH` | `Shitao/OmniGen-v1` | Base OmniGen model path or Hugging Face repo. |
| `LORA_PATH` | `results/joint_mask_cogen/checkpoints/0010000/` | LoRA checkpoint directory. |
| `MASK_MODULES_PATH` | `results/joint_mask_cogen/checkpoints/0010000/mask_modules.bin` | Joint mask module checkpoint. |
| `JSONL_PATH` | `/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl` | Test JSONL. |
| `OUTPUT_DIR` | `analysis_attention_heatmaps` | Output directory for attention heatmaps. |
| `INFERENCE_STEPS` | `50` | Diffusion denoising steps. |
| `GUIDANCE_SCALE` | `2.5` | Text CFG guidance scale. |
| `IMG_GUIDANCE_SCALE` | `2.0` | Image CFG guidance scale. |
| `SEED` | `42` | Random seed. |
| `MASK_LATENT_CHANNELS` | `4` | Mask latent channel count. |
| `MASK_THRESHOLD` | `0.0` | Mask threshold passed to the pipeline. |
| `HEATMAP_ALPHA` | `0.45` | Heatmap overlay opacity. |
| `MAX_IMAGE_SIZE` | `1024` | Maximum image size used by OmniGen preprocessing. |
| `DEVICE` | `cuda:0` | Torch device used by the script. |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU visibility. Override externally for different GPUs. |

Every variable can be overridden from the command line without editing the file.

### Example Usage

Run attention extraction for specific samples:

```bash
OUTPUT_DIR=analysis_outputs/attention_layer20 \
CUDA_VISIBLE_DEVICES=0 \
bash launch/extract_attention_heatmap.sh \
  --sample_ids LIDC-IDRI-0251,LIDC-IDRI-0657 \
  --layer_index 20
```

Run a small smoke test:

```bash
OUTPUT_DIR=analysis_outputs/attention_smoke \
CUDA_VISIBLE_DEVICES=0 \
bash launch/extract_attention_heatmap.sh \
  --max_samples 2 \
  --layer_index -1
```

Use a different checkpoint:

```bash
LORA_PATH=results/joint_mask_cogen/checkpoints/0030000 \
MASK_MODULES_PATH=results/joint_mask_cogen/checkpoints/0030000/mask_modules.bin \
OUTPUT_DIR=analysis_outputs/attention_0030000 \
CUDA_VISIBLE_DEVICES=0 \
bash launch/extract_attention_heatmap.sh \
  --sample_ids LIDC-IDRI-0251_0001 \
  --layer_index 15
```

### Expected Outputs

The output directory contains only diagnostic attention artifacts:

```text
analysis_outputs/attention_layer20/
  extract_attention_config.json
  run.log
  LIDC-IDRI-0251_0001_layer20_attn.png
  LIDC-IDRI-0251_0001_layer20_attn.npy
  LIDC-IDRI-0657_0001_layer20_attn.png
  LIDC-IDRI-0657_0001_layer20_attn.npy
```

File meanings:

- `{sample_id}_layer{layer_index}_attn.png`: Heatmap overlay on the generated image.
- `{sample_id}_layer{layer_index}_attn.npy`: Raw normalized 256x256 attention heatmap array.
- `{sample_id}_generated.png`: Optional generated image, saved only when `--save_generated` is passed.
- `extract_attention_config.json`: Exact extraction configuration.
- `run.log`: Diagnostic log.

### Practical Notes

- Attention extraction is heavier than normal inference because attention weights are materialized. Use a few selected samples.
- The script disables KV cache for attention extraction so the full image-token and mask-token attention matrix is available.
- Use `--sample_ids` for paper figures. Running attention extraction on the full dataset is not recommended.
- If no attention is captured, check the installed `transformers` version and confirm that eager attention with `output_attentions=True` is supported for the Phi-3 attention module.

## Recommended End-to-End Commands

1. Full ROI metrics:

```bash
python analyze_roi_metrics.py \
  --jsonl_path /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl \
  --inference_dir results/joint_mask_inference \
  --output_dir analysis_outputs/roi_metrics_full
```

2. Inspect:

```bash
head -n 20 analysis_outputs/roi_metrics_full/sample_metrics.csv
```

3. Visualize selected samples:

```bash
python analyze_visualization.py \
  --jsonl_path /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl \
  --inference_dir results/joint_mask_inference \
  --output_dir analysis_outputs/paper_figures \
  --sample_ids LIDC-IDRI-0251_0001,LIDC-IDRI-0089_0217
```

4. Extract selected-sample attention heatmaps:

```bash
OUTPUT_DIR=analysis_outputs/attention_layer20 \
CUDA_VISIBLE_DEVICES=0 \
bash launch/extract_attention_heatmap.sh \
  --sample_ids LIDC-IDRI-0251_0001,LIDC-IDRI-0089_0217 \
  --layer_index 20
```
