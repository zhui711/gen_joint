# Implementation Summary: Perceptual Feature Matching Loss

**Date:** 2026-03-27
**Author:** Claude
**Status:** ✅ Complete and Verified

---

## Executive Summary

Successfully implemented **Perceptual Feature Matching Loss** for OmniGen fine-tuning on CXR datasets. This replaces the previous approach of computing MSE on final segmentation logits with MSE on intermediate encoder features, providing spatial tolerance while preserving anatomical structures.

---

## Changes Made

### 1. `loss_anatomy.py` (Modified)

**Path:** `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py`

**Key Change:** Updated `compute_feature_matching_loss()` to use encoder features instead of full model logits.

```python
# OLD APPROACH (removed)
with torch.no_grad():
    gt_logits = seg_model(gt_images)      # Full UNet: (n, 10, 256, 256)
gen_logits = seg_model(gen_images)
loss = F.mse_loss(gen_logits, gt_logits)

# NEW APPROACH (implemented)
FEATURE_INDICES = [2, 3, 4]  # 1/4, 1/8, 1/16 resolution

with torch.no_grad():
    gt_features = seg_model.encoder(gt_images)   # Encoder only: 6 feature maps
gen_features = seg_model.encoder(gen_images)

total_loss = 0.0
for idx in FEATURE_INDICES:
    mse = F.mse_loss(gen_features[idx], gt_features[idx], reduction='mean')
    total_loss = total_loss + mse
loss = total_loss / len(FEATURE_INDICES)
```

### 2. `train_anatomy.py` (Modified)

**Path:** `/home/wenting/zr/gen_code/train_anatomy.py`

**Changes:**
- Updated module docstring to reflect "Perceptual Feature Matching" approach
- Changed log message from "Feature Matching mode" to "Perceptual Feature Matching mode"

### 3. `visualize_features.py` (Created)

**Path:** `/home/wenting/zr/gen_code/visualize_features.py`

**New standalone script for:**
- Loading GT and Generated images
- Extracting encoder features at selected layers
- Generating heatmap visualizations
- Computing per-layer MSE values
- Creating comparison figures

**Usage:**
```bash
python visualize_features.py \
    --gt_image /path/to/gt.png \
    --gen_image /path/to/gen.png \
    --seg_model_ckpt /path/to/best_anatomy_model.pth \
    --output_dir ./feature_vis_output
```

### 4. `perceptual_feature_matching_loss.md` (Created)

**Path:** `/home/wenting/zr/gen_code/docs/perceptual_feature_matching_loss.md`

**Comprehensive documentation covering:**
- Problem statement and motivation
- Solution architecture
- Layer selection rationale
- Implementation details
- Usage instructions

---

## Layer Selection Rationale

ResNet34 encoder outputs 6 feature maps:

| Index | Shape | Stride | Resolution | Selection |
|-------|-------|--------|------------|-----------|
| 0 | (B, 3, 256, 256) | 1 | 1/1 | ❌ Skip - original input |
| 1 | (B, 64, 128, 128) | 2 | 1/2 | ❌ Skip - too close to pixel level |
| 2 | (B, 64, 64, 64) | 4 | 1/4 | ✅ USE - mid-level edges |
| 3 | (B, 128, 32, 32) | 8 | 1/8 | ✅ USE - structural shapes |
| 4 | (B, 256, 16, 16) | 16 | 1/16 | ✅ USE - semantic regions |
| 5 | (B, 512, 8, 8) | 32 | 1/32 | ❌ Skip - too high-level |

**Selected: [2, 3, 4]** for balanced perceptual loss across mid-to-high level features.

---

## Dry Run Verification Results

### Test 1: Encoder Feature Extraction
```
Number of features: 6
Feature 0: shape=(2, 3, 256, 256), stride=1
Feature 1: shape=(2, 64, 128, 128), stride=2
Feature 2: shape=(2, 64, 64, 64), stride=4    ✅ Used
Feature 3: shape=(2, 128, 32, 32), stride=8   ✅ Used
Feature 4: shape=(2, 256, 16, 16), stride=16  ✅ Used
Feature 5: shape=(2, 512, 8, 8), stride=32
```

### Test 2: Loss Computation
```
Layer 2 MSE: 0.456726
Layer 3 MSE: 1.989871
Layer 4 MSE: 35.610207
Average MSE (loss_anatomy): 12.685601
```

### Test 3: Gradient Flow
```
Gradient exists: True
Gradient shape: torch.Size([2, 3, 256, 256])
Gradient norm: 0.526152
```

### Test 4: Module Import
```
All functions imported successfully!
Loss value: 11.079437
Has grad_fn: True
Gradient computed: True
```

### Test 5: Visualization Script
```
Saved: /tmp/feature_vis_test/Layer2_stride4_gt_vs_gen.png    ✅
Saved: /tmp/feature_vis_test/Layer3_stride8_gt_vs_gen.png    ✅
Saved: /tmp/feature_vis_test/Layer4_stride16_gt_vs_gen.png   ✅
Saved: /tmp/feature_vis_test/summary_all_layers.png          ✅

Feature MSE Summary:
  Layer2_stride4: MSE = 0.349754
  Layer3_stride8: MSE = 0.286707
  Layer4_stride16: MSE = 0.263443
  Average MSE: 0.299968
```

---

## Gradient Flow Verification

```
Diffusion Model (u_hat)
        │
        ▼
x1_hat = xt + (1-t) * u_hat     ← Computed WITH grad
        │
        ▼
VAE Decode (NO torch.no_grad)   ← Gradients preserved
        │
        ▼
gen_decoded (pixel space)
        │
        ▼
seg_model.encoder(gen_decoded)  ← WITH grad (frozen weights)
        │
        ▼
MSE(gen_features, gt_features)
        │
        ▼
loss_anatomy.backward()         ← Flows back to u_hat
```

✅ Verified: Gradients flow correctly from anatomy loss back to diffusion model.

---

## Notes on Pseudo-Masks

Since the new approach computes loss between `GT_img` and `Gen_img` features directly, the `.npz` pseudo-masks are **no longer required** for the anatomy loss computation. However:

- The dataloader still supports loading them (for debugging/analysis)
- No changes were made to the dataloader to avoid breaking existing pipelines
- The masks can be safely removed from the dataset if not needed elsewhere

---

## Recommended Next Steps

1. **Tune `lambda_anatomy`**: The new loss has different magnitude than the old logits MSE. Start with values like 0.001-0.01 and tune based on validation metrics.

2. **Monitor metrics**: Track SSIM, PSNR, and anatomical segmentation accuracy during training to verify improvements.

3. **Use visualization script**: Run `visualize_features.py` on validation samples to qualitatively assess feature alignment.

4. **Consider layer weights**: Currently all selected layers have equal weight (1.0). You may experiment with different weights to emphasize certain scales.

---

## Files Summary

| File | Status | Description |
|------|--------|-------------|
| `OmniGen/train_helper/loss_anatomy.py` | Modified | Perceptual feature matching loss |
| `train_anatomy.py` | Modified | Updated docstrings and log messages |
| `visualize_features.py` | Created | Feature visualization tool |
| `docs/perceptual_feature_matching_loss.md` | Created | Detailed documentation |
| `docs/implementation_summary.md` | Created | This summary |
