# Perceptual Feature Matching Loss for Anatomy-Aware CXR Generation

## Overview

This document describes the implementation of **Perceptual Feature Matching Loss** for OmniGen fine-tuning on CXR (Chest X-Ray) datasets. This approach replaces the previous "hard pixel alignment" loss (MSE on final logits or BCE+Dice on masks) with a softer, spatially-tolerant loss computed on intermediate encoder features.

## Problem Statement

The previous approach computed MSE loss on the final segmentation logits:
```python
# OLD APPROACH
gt_logits = seg_model(gt_images)      # (B, 10, 256, 256)
gen_logits = seg_model(gen_images)    # (B, 10, 256, 256)
loss = F.mse_loss(gen_logits, gt_logits)
```

**Issues:**
1. Final logits enforce pixel-exact alignment, which is too strict
2. Small spatial shifts in generated images cause large losses
3. This degraded image quality metrics (SSIM, PSNR)
4. The decoder part of UNet adds complexity without benefit

## Solution: Perceptual Feature Matching

Following the **first principles of Perceptual Loss (Feature Matching)**, we now compute MSE on **intermediate feature maps** from the UNet's encoder:

```python
# NEW APPROACH
gt_features = seg_model.encoder(gt_images)    # List of 6 feature tensors
gen_features = seg_model.encoder(gen_images)  # List of 6 feature tensors
loss = MSE(gt_features[selected], gen_features[selected])
```

**Benefits:**
1. Provides spatial tolerance (features are at lower resolution)
2. Captures anatomical structures (ribs, heart contours) without pixel-exact matching
3. Preserves image quality metrics
4. Uses only the encoder (faster, no decoder computation)

---

## Layer Selection Logic

### ResNet34 Encoder Output Structure

The `segmentation_models_pytorch` (smp) library's ResNet34 encoder produces **6 feature maps**:

| Index | Shape (B=1, H=W=256) | Stride | Resolution | Description |
|-------|---------------------|--------|------------|-------------|
| 0 | (1, 3, 256, 256) | 1 | 1/1 | Original input (passthrough) |
| 1 | (1, 64, 128, 128) | 2 | 1/2 | After initial conv + pool |
| 2 | (1, 64, 64, 64) | 4 | 1/4 | Layer1 output (ResNet block 1) |
| 3 | (1, 128, 32, 32) | 8 | 1/8 | Layer2 output (ResNet block 2) |
| 4 | (1, 256, 16, 16) | 16 | 1/16 | Layer3 output (ResNet block 3) |
| 5 | (1, 512, 8, 8) | 32 | 1/32 | Layer4 output (ResNet block 4) |

### Selected Layers: [2, 3, 4]

We select **indices [2, 3, 4]** corresponding to:
- **1/4 resolution (stride 4)**: Mid-level edge features
- **1/8 resolution (stride 8)**: Structural shape features
- **1/16 resolution (stride 16)**: Semantic region features

**Rationale:**
1. **Indices 0, 1 skipped**: Too close to pixel level, would reintroduce hard alignment
2. **Index 5 skipped**: Too high-level/semantic, loses spatial structure
3. **Indices 2, 3, 4**: Balance between spatial tolerance and anatomical detail

### Verification Code

```python
import segmentation_models_pytorch as smp
import torch

model = smp.Unet(encoder_name="resnet34", ...)
x = torch.randn(1, 3, 256, 256)
features = model.encoder(x)

for i, f in enumerate(features):
    print(f"Feature {i}: shape={f.shape}, stride={256 // f.shape[-1]}")
```

Output:
```
Feature 0: shape=torch.Size([1, 3, 256, 256]), stride=1
Feature 1: shape=torch.Size([1, 64, 128, 128]), stride=2
Feature 2: shape=torch.Size([1, 64, 64, 64]), stride=4
Feature 3: shape=torch.Size([1, 128, 32, 32]), stride=8
Feature 4: shape=torch.Size([1, 256, 16, 16]), stride=16
Feature 5: shape=torch.Size([1, 512, 8, 8]), stride=32
```

---

## Implementation Details

### Loss Computation

```python
def compute_feature_matching_loss(seg_model, gen_images, gt_images):
    FEATURE_INDICES = [2, 3, 4]

    # GT features: no grad (frozen target)
    with torch.no_grad():
        gt_features = seg_model.encoder(gt_images)

    # Gen features: WITH grad (training signal flows back)
    gen_features = seg_model.encoder(gen_images)

    # MSE for each layer, normalized by element count
    total_loss = 0.0
    for idx in FEATURE_INDICES:
        mse = F.mse_loss(gen_features[idx], gt_features[idx], reduction='mean')
        total_loss = total_loss + mse

    # Average across layers (equal weight = 1.0)
    loss = total_loss / len(FEATURE_INDICES)
    return loss
```

### Gradient Flow

```
Diffusion Model Output (u_hat)
        ↓
    x1_hat = xt + (1-t) * u_hat   [predicted clean latent]
        ↓
    VAE Decode (WITH grad)         [NO torch.no_grad()]
        ↓
    gen_decoded (pixel space)
        ↓
    seg_model.encoder(gen_decoded)  [WITH grad, frozen weights]
        ↓
    MSE(gen_features, gt_features)
        ↓
    loss_anatomy → backprop → diffusion model
```

**Critical**: The VAE decode step retains the autograd graph. Gradients flow from `loss_anatomy` back through the encoder features, through the VAE decoder, to the predicted latent `x1_hat`, and ultimately to the diffusion model's output `u_hat`.

### Pseudo-Masks Not Required

Since we're matching features between `GT_img` and `Gen_img`, the `.npz` pseudo-masks are **no longer needed** for this specific loss. The dataloader can still load them (for debugging or other purposes), but they are not used in the loss computation.

---

## Files Modified

### 1. `loss_anatomy.py`
**Path:** `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py`

**Changes:**
- Updated `compute_feature_matching_loss()` to use `seg_model.encoder()` instead of `seg_model()`
- Added layer selection logic with `FEATURE_INDICES = [2, 3, 4]`
- Updated docstrings to reflect perceptual feature matching approach

### 2. `train_anatomy.py`
**Path:** `/home/wenting/zr/gen_code/train_anatomy.py`

**Changes:**
- Updated docstrings to reflect the new approach
- Updated log message to say "Perceptual Feature Matching mode"

### 3. `visualize_features.py` (New)
**Path:** `/home/wenting/zr/gen_code/visualize_features.py`

**New script for:**
- Visualizing encoder feature maps
- Comparing GT vs Generated features
- Generating heatmap overlays
- Computing per-layer MSE values

---

## Usage

### Training

No changes to the launch command. The loss function automatically uses the new perceptual feature matching approach:

```bash
accelerate launch train_anatomy.py \
    --seg_model_ckpt /path/to/best_anatomy_model.pth \
    --lambda_anatomy 0.005 \
    --anatomy_subbatch_size 16 \
    ...
```

### Visualization

```bash
python visualize_features.py \
    --gt_image /path/to/ground_truth.png \
    --gen_image /path/to/generated.png \
    --seg_model_ckpt /path/to/best_anatomy_model.pth \
    --output_dir ./feature_vis_output
```

**Outputs:**
- `Layer2_stride4_gt_vs_gen.png` - Mid-level feature comparison
- `Layer3_stride8_gt_vs_gen.png` - Structural feature comparison
- `Layer4_stride16_gt_vs_gen.png` - Semantic feature comparison
- `summary_all_layers.png` - Combined overview with difference maps

---

## Expected Behavior

### Loss Values

The perceptual feature MSE will typically be **larger** than the previous logits MSE because:
1. Features have more channels (64, 128, 256 vs 10)
2. Feature values have different magnitude than logits

You may need to adjust `lambda_anatomy` accordingly (e.g., start with 0.001-0.01 and tune).

### Training Dynamics

- The loss should decrease smoothly as the generated images become more anatomically similar to GT
- Small spatial shifts should NOT cause large loss spikes (unlike pixel-level matching)
- SSIM/PSNR metrics should improve compared to the previous logits-MSE approach

---

## References

1. **Perceptual Losses for Real-Time Style Transfer and Super-Resolution** (Johnson et al., 2016)
   - Introduced the concept of feature matching for image quality

2. **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric** (Zhang et al., 2018)
   - Demonstrated that intermediate CNN features correlate with human perception

3. **Segmentation Models PyTorch** (smp)
   - https://github.com/qubvel/segmentation_models.pytorch
