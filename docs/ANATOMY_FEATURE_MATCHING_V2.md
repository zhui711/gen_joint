# Anatomy-Aware Feature Matching v2: Implementation Guide

## Overview

This document describes the **Mask-Weighted Feature Matching** loss (v2), which replaces the original BCE+Dice pixel-level segmentation loss (v1). The key insight: instead of forcing pixel-perfect alignment of segmentation logits (which destroyed SSIM/PSNR), we match **intermediate encoder features** weighted by anatomical masks, providing spatial tolerance while maintaining anatomical fidelity.

## Architecture Decision: Why Feature Matching at 1/4 Resolution?

### The Problem with v1 (BCE+Dice on Logits)
- Operates at full resolution (256x256) on 10-channel segmentation logits
- Forces pixel-perfect alignment → unnatural artifacts
- Gradients are dominated by boundary pixels → instability
- No spatial tolerance → SSIM/PSNR degradation

### Why 1/4 Resolution (64x64)?
We analyzed the ResNet34 encoder layers in `smp.Unet`:

| Feature Index | Shape (B=1, 256 input) | Stride | Resolution | Character |
|---|---|---|---|---|
| 0 | (1, 3, 256, 256) | 1 | 1/1 | Raw pixels (too noisy) |
| 1 | (1, 64, 128, 128) | 2 | 1/2 | Low-level edges (too local) |
| **2** | **(1, 64, 64, 64)** | **4** | **1/4** | **Mid-level structure** ← TARGET |
| 3 | (1, 128, 32, 32) | 8 | 1/8 | Structural shapes |
| 4 | (1, 256, 16, 16) | 16 | 1/16 | Semantic regions |
| 5 | (1, 512, 8, 8) | 32 | 1/32 | Too abstract |

**Feature[2] (1/4 res, 64 channels)** is ideal because:
- **Spatial tolerance**: Each feature pixel covers a 4x4 patch → small misalignments don't penalize
- **Rich features**: 64 channels encode edges, textures, and local structure
- **Configurable**: `feature_layer_idx` parameter allows easy experimentation

### How We Identified This
The encoder features were traced through `smp.Unet` → `ResNet34Encoder`:
```
encoder(x) → [
  features[0]: identity(x),           # stride 1
  features[1]: relu(bn(conv1(x))),    # stride 2
  features[2]: layer1(maxpool(...)),   # stride 4  ← maxpool doubles stride
  features[3]: layer2(...),            # stride 8
  features[4]: layer3(...),            # stride 16
  features[5]: layer4(...),            # stride 32
]
```

## Mathematical Flow

```
Input: X_gt (GT image), X_gen (decoded generated image), Mask_gt (10ch GT mask)

1. Encoder Forward:
   F_gt  = seg_model.encoder(X_gt)[2]    → (B, 64, 64, 64)  [no_grad]
   F_gen = seg_model.encoder(X_gen)[2]   → (B, 64, 64, 64)  [WITH grad]

2. Mask Prediction (for gen image):
   Reuse encoder features: decoder(gen_features) → segmentation_head → sigmoid
   Mask_gen = σ(seg_head(decoder(gen_features_all)))  → (B, 10, 256, 256)

3. Downsample Masks:
   Mask_gt_down  = avg_pool2d(Mask_gt, kernel=4)   → (B, 10, 64, 64)
   Mask_gen_down = avg_pool2d(Mask_gen, kernel=4)   → (B, 10, 64, 64)

4. Per-Organ MSE (for each channel c ∈ [0, 9]):
   M_gt^c  = Mask_gt_down[:, c:c+1]     → (B, 1, 64, 64)
   M_gen^c = Mask_gen_down[:, c:c+1]     → (B, 1, 64, 64)

   OrganFeat_gt^c  = M_gt^c  ⊙ F_gt     → (B, 64, 64, 64)  [broadcasting]
   OrganFeat_gen^c = M_gen^c ⊙ F_gen     → (B, 64, 64, 64)

   MSE_c = mean((OrganFeat_gen^c - OrganFeat_gt^c)²)

5. Total: Loss_anatomy = Σ_{c=0}^{9} MSE_c
```

**Critical Design Choices:**
- **Sum, not average** across organs → each organ contributes independently
- **avg_pool2d** for downsampling → smooth soft masks at feature resolution
- **Reuse encoder features** for mask prediction → avoids redundant forward pass
- **GT features under no_grad** → only gen image gradients flow back

## Files Modified

### `OmniGen/train_helper/loss_anatomy_v2.py`
Core loss implementation. Key changes from previous version:
- **Fixed encoder reuse**: When `use_gen_mask=True`, the encoder features are computed once and reused for both feature extraction AND mask prediction (decoder+head only). Previously, `seg_model(gen_images)` ran the entire model again.
- **Removed `MaskWeightedFeatureLoss` class and backward-compatible wrapper**: Simplified to just the functional API `compute_mask_weighted_feature_loss()` and the training integration `training_losses_with_anatomy_v2()`.

### `train_anatomy.py`
Training script changes:
- **Import**: `loss_anatomy.py` → `loss_anatomy_v2.py`
- **GT pixel images preserved**: `output_images_pixel` saved before VAE encoding
- **New args**: `--feature_layer_idx` (default 2), `--use_gen_mask` (default True)
- **Loss call**: `training_losses_with_anatomy_v2(...)` with `output_images_pixel` parameter

### `visualize_features.py`
Standalone visualization tool. New capabilities:
- **Mask-weighted organ visualization**: Pass `--mask_gt /path/to/mask.npz` to see per-organ isolated features
- **Per-organ MSE table**: Replicates exact loss computation for diagnosis
- **Efficient**: Reuses encoder features for mask prediction (same optimization as loss)

## Usage

### Training
```bash
# Standard anatomy-aware training with feature matching v2
accelerate launch train_anatomy.py \
    --model_name_or_path /path/to/omnigen \
    --json_file /path/to/cxr_synth_anno_mask_train.jsonl \
    --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
    --lambda_anatomy 0.1 \
    --anatomy_subbatch_size 4 \
    --feature_layer_idx 2 \
    --use_gen_mask True \
    --use_lora \
    --lora_rank 8 \
    ...
```

### Visualization
```bash
# Raw features only
python visualize_features.py \
    --gt_image /path/to/gt.png \
    --gen_image /path/to/gen.png

# With mask-weighted organ visualization
python visualize_features.py \
    --gt_image /path/to/gt.png \
    --gen_image /path/to/gen.png \
    --mask_gt /path/to/mask.npz \
    --organ_channels 0 1 2 7 \
    --feature_layer_idx 2
```

### Dry Run Validation
```bash
python dry_run_v2.py
```

## Anatomical Channels

| Index | Organ | Notes |
|---|---|---|
| 0 | Lung_Left | Large area, high coverage |
| 1 | Lung_Right | Large area, high coverage |
| 2 | Heart | Critical for CXR quality |
| 3 | Aorta | Thin structure |
| 4 | Liver | Partially visible in CXR |
| 5 | Stomach | Small/variable |
| 6 | Trachea | Thin tubular structure |
| 7 | Ribs | Detailed bone alignment |
| 8 | Vertebrae | Central spine alignment |
| 9 | Upper_Skeleton | Clavicle, scapula |

## Hyperparameter Tuning Guide

| Parameter | Default | Notes |
|---|---|---|
| `lambda_anatomy` | 0.1 | Start low; increase if anatomy not improving |
| `feature_layer_idx` | 2 | Layer 2 (1/4 res) for balance of detail and tolerance |
| `use_gen_mask` | True | Set False for early training stability |
| `anatomy_subbatch_size` | 4 | Reduce for VRAM constraints |

## Dry Run Results

All 7 validation tests pass:
- Encoder feature shapes match expected ResNet34 architecture
- Mask downsampling preserves spatial structure
- Broadcasting: (B,1,H,W) × (B,C,H,W) → (B,C,H,W) correct
- Gradients flow through frozen seg model to gen_images input
- Decoder reuse produces identical results to full forward pass (diff = 0.0)
- Per-channel loss is independent (zero for identical images)
- `use_gen_mask` flag correctly switches between predicted and GT masks
