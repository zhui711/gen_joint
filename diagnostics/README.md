# Diagnostic Tools for OmniGen Anatomy-Aware Fine-tuning

This directory contains diagnostic tools to analyze why the Anatomy Loss decreases while Diffusion Loss increases, and whether the model is creating "adversarial shortcuts" or suffering from gradient conflicts.

## Problem Statement

When fine-tuning OmniGen with the auxiliary Anatomy-Aware Loss (Logits MSE via frozen ResNet34-UNet):
- **Anatomy Loss** decreases (good)
- **Diffusion Loss** slightly increases (concerning)
- **Image fidelity metrics** (FID, LPIPS, SSIM) degrade (bad)

This could indicate:
1. **Adversarial shortcuts**: Model creates localized artifacts that "fool" the UNet
2. **Gradient conflicts**: Anatomy gradients overpower or oppose diffusion gradients
3. **Loss imbalance**: `lambda_anatomy` is too high

## Tools Overview

### Tool 1: Objective Anatomy Evaluation (`eval_anatomy_dice.py`)

Computes Macro-Dice scores to objectively measure whether anatomical structure improved.

**Key Question**: Did anatomy actually improve despite worse FID?

### Tool 2: Gradient Norm Monitoring (`gradient_monitor.py`)

Measures gradient norms from each loss component and detects conflicts.

**Key Question**: Are anatomy gradients overpowering or opposing diffusion gradients?

### Tool 3: Visual Difference Heatmap (`plot_diff_heatmap.py`)

Creates side-by-side visualizations highlighting pixel differences.

**Key Question**: Where are the visual differences located? Are they anatomically meaningful or adversarial artifacts?

---

## Usage Instructions

### Tool 1: Anatomy Dice Evaluation

```bash
# Navigate to gen_code directory
cd /home/wenting/zr/gen_code

# Run the evaluation
python diagnostics/eval_anatomy_dice.py \
    --baseline_dir outputs/cxr_finetune_lora_30000 \
    --seg_dir outputs/cxr_finetune_lora_30ksteps_SegMSE_lamda0.005_subbatch16_10500 \
    --gt_mask_dir /home/wenting/zr/Segmentation/data/lidc_TotalSeg \
    --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
    --output_report diagnostics/anatomy_comparison_report.json
```

**Output Example**:
```
================================================================================
ANATOMY DICE COMPARISON: Baseline vs. OmniGen+Seg
================================================================================

Class                Baseline        OmniGen+Seg     Delta
-----------------------------------------------------------------
Lung_Left            0.8521          0.8673          +0.0152 ++
Lung_Right           0.8489          0.8612          +0.0123 ++
Heart                0.7823          0.7901          +0.0078 +
Aorta                0.6234          0.6189          -0.0045 -
...
-----------------------------------------------------------------
MACRO DICE           0.7521          0.7598          +0.0077

DIAGNOSTIC INTERPRETATION:
[MILD] Slight anatomy improvement with segmentation loss.
       But gains may not justify FID degradation.
```

---

### Tool 2: Gradient Norm Monitoring

**Option A: Quick logging (add to training loop)**

```python
# In train_anatomy.py, add at the top:
from diagnostics.gradient_monitor import log_gradient_analysis

# In training loop, after loss computation but BEFORE backward:
if global_step % 100 == 0:
    log_gradient_analysis(
        model=model,
        loss_diffusion=loss_dict["loss_diffusion"],
        loss_anatomy=loss_dict["loss_anatomy"],
        lambda_anatomy=args.lambda_anatomy,
        step=global_step,
        retain_graph=True,  # Keep graph for actual backward
    )

loss_dict["loss_total"].backward()
```

**Option B: Full monitoring with summary**

```python
from diagnostics.gradient_monitor import GradientMonitor

# Before training loop:
gradient_monitor = GradientMonitor(model, log_every=100)

# In training loop:
for step, batch in enumerate(dataloader):
    loss_dict = training_losses_with_anatomy(...)

    # Log gradient analysis (does nothing if step % log_every != 0)
    gradient_monitor.log_step(
        step=step,
        loss_diffusion=loss_dict["loss_diffusion"],
        loss_anatomy=loss_dict["loss_anatomy"],
        lambda_anatomy=args.lambda_anatomy,
    )

    loss_dict["loss_total"].backward()
    optimizer.step()
    optimizer.zero_grad()

# After training:
print(gradient_monitor.diagnose())
```

**Output Example**:
```
[GradAnalysis] Step 100: ||g_diff||=0.0234, ||g_anat||=0.0891, ratio=3.8077, cos_sim=-0.1523
[GradAnalysis] Step 100: WARNING - Gradient conflict (cos_sim=-0.1523)
[GradAnalysis] Step 100: WARNING - Anatomy gradients dominating (ratio=3.8077)
```

**Diagnostic Report**:
```
============================================================
GRADIENT MONITORING DIAGNOSTIC REPORT
============================================================
Total logged steps: 500
Gradient conflict events: 127

Gradient Norm Ratio (anatomy/diffusion):
  Mean: 3.2145
  Std:  1.8923
  Range: [0.5234, 8.9012]

[WARNING] Anatomy gradients DOMINATE diffusion gradients!
  -> Reduce lambda_anatomy (try 0.5x current value)
  -> Consider gradient clipping for anatomy loss

Gradient Cosine Similarity:
  Mean: -0.0823
  Min:  -0.4521

[CRITICAL] Gradients are OPPOSED on average!
  -> The losses are conflicting - model is being pulled in opposite directions
  -> Consider: gradient projection, PCGrad, or removing anatomy loss

[WARNING] 127/500 (25.4%) steps had gradient conflicts (cos_sim < -0.1)
============================================================
```

---

### Tool 3: Visual Difference Heatmap

**Single image comparison**:
```bash
python diagnostics/plot_diff_heatmap.py \
    --baseline_img outputs/cxr_finetune_lora_30000/LIDC-IDRI-0030/0001.png \
    --seg_img outputs/cxr_finetune_lora_30ksteps_SegMSE_lamda0.005_subbatch16_10500/LIDC-IDRI-0030/0001.png \
    --output diagnostics/single_comparison.png
```

**Batch comparison** (recommended):
```bash
python diagnostics/plot_diff_heatmap.py \
    --baseline_dir outputs/cxr_finetune_lora_30000 \
    --seg_dir outputs/cxr_finetune_lora_30ksteps_SegMSE_lamda0.005_subbatch16_10500 \
    --output_dir diagnostics/diff_heatmaps \
    --n_samples 100 \
    --threshold 0.1 \
    --compute_stats \
    --create_grid
```

**Options**:
- `--colormap`: jet, hot, viridis, inferno, plasma, magma
- `--threshold`: Values below this are zeroed (noise suppression)
- `--compute_stats`: Print global difference statistics
- `--create_grid`: Create a summary grid image

**Output**:
- Individual comparison images: `{output_dir}/{patient_id}_{image_idx}_diff.png`
- Summary grid: `{output_dir}/_summary_grid.png`
- Console statistics showing mean/std pixel differences

---

## Interpretation Guide

### Scenario 1: Anatomy Improves + High Gradient Ratio + Localized Differences
**Diagnosis**: Adversarial shortcuts - the model is creating localized "patches" that fool the UNet without improving overall image quality.
**Solution**:
- Add total variation (TV) loss to penalize high-frequency artifacts
- Use a multi-scale discriminator
- Reduce `lambda_anatomy`

### Scenario 2: Anatomy Unchanged + Negative Cosine Similarity
**Diagnosis**: Severe gradient conflict - the losses are working against each other.
**Solution**:
- Try PCGrad (Projecting Conflicting Gradients)
- Alternate between loss terms
- Remove anatomy loss entirely

### Scenario 3: Anatomy Worse + Diffusion Dominates
**Diagnosis**: The anatomy loss is too weak to have effect.
**Solution**:
- Increase `lambda_anatomy` (carefully)
- Check that gradients are flowing through VAE decode

### Scenario 4: Anatomy Improves + Balanced Gradients + Dispersed Differences
**Diagnosis**: Healthy training - the model is learning anatomical structure.
**Solution**: Continue training, possibly reduce `lambda_anatomy` slightly to recover FID.

---

## Quick Diagnostic Workflow

1. **Run Dice evaluation** to confirm whether anatomy improved:
   ```bash
   python diagnostics/eval_anatomy_dice.py --baseline_dir ... --seg_dir ...
   ```

2. If anatomy **did not improve**, check gradients:
   - Add gradient monitoring to training
   - Look for conflicts (negative cosine similarity)
   - Look for domination (high ratio)

3. If anatomy **improved but FID worse**, check for artifacts:
   ```bash
   python diagnostics/plot_diff_heatmap.py --baseline_dir ... --seg_dir ... --compute_stats
   ```
   - Look at the difference heatmaps
   - If differences are localized to non-anatomical regions → adversarial shortcuts
   - If differences are in anatomical regions → expected structural improvement

---

## File Structure

```
diagnostics/
├── __init__.py              # Module initialization
├── README.md                # This file
├── eval_anatomy_dice.py     # Tool 1: Objective Dice evaluation
├── gradient_monitor.py      # Tool 2: Gradient norm monitoring
└── plot_diff_heatmap.py     # Tool 3: Visual difference heatmaps
```

---

## Requirements

These tools use standard PyTorch dependencies plus:
- `segmentation_models_pytorch` (from /home/wenting/zr/Segmentation)
- `opencv-python` (cv2)
- `matplotlib`
- `tqdm`
- `PIL` (Pillow)

All should already be installed in your training environment.
