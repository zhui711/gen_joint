# SegMSE Loss Degradation Analysis & Diagnostic Toolkit

**Project:** OmniGen + Anatomy-Aware Fine-Tuning for Multi-View CXR Generation
**Date:** 2026-03-26
**Authors:** AI Research Analysis
**Status:** Diagnostic Investigation

---

## Executive Summary

This report provides a deep-dive analysis of the observed quality degradation when adding the SegMSE (Anatomy-Aware Feature Matching) loss to OmniGen fine-tuning for CXR generation. The key observations from training were:

| Metric | Baseline (30k steps) | + SegMSE (10.5k more) | Direction |
|--------|---------------------|----------------------|-----------|
| `Loss_diffusion` | ~0.44 | ~0.45 | **Increased** (worse) |
| `Loss_anatomy` | - | Decreased | Improved |
| FID | Better | Worse | **Degraded** |
| LPIPS | Better | Worse | **Degraded** |
| SSIM/PSNR | Higher | Slightly lower | **Degraded** |
| Visual Quality | Clean | Dark patches/artifacts | **Degraded** |

**Training Setup:**
- `batch_size_per_device` = 16
- `gradient_accumulation_steps` = 4
- `anatomy_subbatch_size` = 16 (full-batch, no stochastic noise)
- `lambda_anatomy` = 0.005

---

## Part 1: Theoretical Analysis of the Degradation

### 1.1 Gradient Conflict (Objective Mismatch)

#### The Core Problem

The diffusion objective and the anatomy objective operate in fundamentally different spaces with potentially conflicting optima:

```
Diffusion Loss:
  L_diff = E[||u_hat - u_true||^2]

  Where:
    - u_true = x_1 - x_0 (ground truth velocity in LATENT space)
    - u_hat = model(x_t, t) (predicted velocity)
    - Optimization target: Match the velocity field in VAE latent space

Anatomy Loss:
  L_anat = E[||f(Dec(x_1_hat)) - f(Dec(x_1))||^2]

  Where:
    - Dec() = VAE Decoder (latent -> pixel)
    - f() = UNet segmentation feature extractor
    - x_1_hat = x_t + (1-t) * u_hat (predicted clean latent)
    - Optimization target: Match UNet features in PIXEL space after decoding
```

#### Why This Causes Conflict

1. **Different Representation Spaces**:
   - Diffusion optimizes in VAE latent space (4×32×32 for 256×256 images)
   - Anatomy optimizes through the VAE decoder into pixel space (3×256×256), then through UNet

2. **Non-Aligned Jacobians**:
   The gradient of `L_anat` backpropagates through:
   ```
   ∂L_anat/∂θ = ∂L_anat/∂f * ∂f/∂img * ∂img/∂lat * ∂lat/∂u_hat * ∂u_hat/∂θ
                 \_______/   \_______/   \________/   \__________/
                  UNet grad   Pixel grad  VAE Decoder   Model grad
   ```

   The VAE decoder's Jacobian `∂img/∂lat` transforms gradients in ways that may not preserve alignment with the diffusion objective. Specifically:
   - VAE decoders are trained to maximize ELBO, not to have well-conditioned Jacobians
   - High-frequency pixel details map to complex latent manifold structures
   - The gradient may push latents toward regions that decode to "correct" features but lie off the learned diffusion manifold

3. **The Pareto Frontier Problem**:
   ```
   Loss landscape visualization:

   L_anat
     ^
     |    * (Current position)
     |   /|\
     |  / | \ Pareto frontier
     | /  |  \
     |/   |   \
     +----+-----> L_diff
          |
          Ideal (unachievable)
   ```

   If the two objectives have a non-trivial Pareto frontier, reducing `L_anat` necessarily increases `L_diff`. Your observation that `L_diff` increased from 0.44→0.45 while `L_anat` decreased is classic evidence of this conflict.

#### Mathematical Formalization

Let θ be the LoRA parameters. At each step:
```
g_diff = ∂L_diff/∂θ
g_anat = λ * ∂L_anat/∂θ

g_total = g_diff + g_anat
```

The update direction conflicts when:
```
cos(g_diff, g_anat) = <g_diff, g_anat> / (||g_diff|| * ||g_anat||) < 0
```

When this happens, the total gradient `g_total` points in a direction that:
- Neither fully satisfies `L_diff` (generation quality)
- Nor fully satisfies `L_anat` (anatomy alignment)
- But compromises both objectives, degrading overall performance

**Key Insight**: Even with small λ=0.005, if anatomy gradients point in the opposite direction of diffusion gradients, they will pull the model off its optimal diffusion trajectory.

---

### 1.2 The VAE Domain Shift Problem

#### Why MSE on UNet Logits Creates Adversarial Artifacts

The feature matching loss computes:
```python
L_anat = MSE(UNet(VAE.decode(x1_hat)), UNet(gt_image))
```

**Critical Asymmetry**: The GT image is raw pixels, but the generated image passes through VAE decode.

```
Ground Truth Path:
  gt_image (raw pixels) --> UNet --> gt_logits
                           [No VAE involved]

Generated Path:
  x1_hat (predicted latent) --> VAE.decode() --> gen_image --> UNet --> gen_logits
                                \___________/
                                 VAE artifacts
```

#### The Systematic Bias

VAE decoders introduce systematic artifacts:
1. **Checkerboard patterns** from transposed convolutions
2. **Slight blurring** compared to sharp originals
3. **Color/contrast shifts** from the learned prior

When we minimize `MSE(gen_logits, gt_logits)`:
- We're asking the generator to produce latents that, **after VAE artifacts**, match features of **clean GT images**
- The generator must **pre-compensate** for VAE artifacts
- This pre-compensation manifests as **adversarial patterns** that "undo" VAE issues

#### Dark Patch Artifact Mechanism

The dark patches you observe likely arise from:

```
Scenario:
  - GT image: Uniform lung field, pixel value ~128
  - VAE decode typically adds slight brightening (learned bias)
  - To match GT features, generator must produce darker latents
  - Result: Dark patches in high-uncertainty regions

Mathematical explanation:
  Let VAE.decode introduce brightness bias β > 0:
    decoded_value ≈ expected_value + β

  To achieve decoded_value = GT_value:
    latent must encode: GT_value - β

  This creates systematic darkening in the generated images
```

#### Why This Doesn't Happen Without Anatomy Loss

Without anatomy loss, the diffusion model only needs to match the velocity field in latent space. The VAE's artifacts are "baked in" to the training distribution—the model learns to generate latents that decode reasonably, without any pixel-space comparison.

With anatomy loss, we explicitly penalize differences after decoding, forcing the model to fight against the VAE's systematic biases rather than accepting them.

---

### 1.3 The ReLU Flaw (Mathematical Analysis)

#### Current Implementation Issue

Based on the user's description, ReLU was applied to logits before MSE calculation:

```python
# Problematic implementation (as described)
loss = F.mse_loss(F.relu(gen_logits), F.relu(gt_logits))
```

#### Why ReLU Destroys Gradient Information

**Logit Statistics in Segmentation:**
- Background regions: typically **negative logits** (sigmoid → low probability)
- Foreground regions: typically **positive logits** (sigmoid → high probability)

For a 10-class segmentation of CXR:
```
Typical logit distribution per pixel:
  - ~7-8 classes: negative (not present at this pixel)
  - ~2-3 classes: positive (overlapping anatomy)
```

**ReLU Effect:**
```python
ReLU(x) = max(0, x)

For gen_logit = -2.0, gt_logit = -5.0:
  Without ReLU: MSE = (−2.0 − (−5.0))² = 9.0  ← gradient exists
  With ReLU:    MSE = (0 − 0)² = 0.0          ← gradient = 0!

For gen_logit = 3.0, gt_logit = −2.0:
  Without ReLU: MSE = (3.0 − (−2.0))² = 25.0  ← correct large penalty
  With ReLU:    MSE = (3.0 − 0)² = 9.0        ← understated penalty
```

#### Mathematical Gradient Analysis

Let L = MSE(ReLU(g), ReLU(t)) where g = gen_logits, t = gt_logits

```
∂L/∂g = 2 * (ReLU(g) - ReLU(t)) * ∂ReLU(g)/∂g
      = 2 * (ReLU(g) - ReLU(t)) * 1_{g > 0}

Case analysis:
  1. g > 0, t > 0: ∂L/∂g = 2(g - t)     ← Normal gradient
  2. g > 0, t < 0: ∂L/∂g = 2g           ← Incorrect (should penalize more)
  3. g < 0, t > 0: ∂L/∂g = 0            ← ZERO gradient! Model can't learn
  4. g < 0, t < 0: ∂L/∂g = 0            ← ZERO gradient! Background lost
```

**Critical Issue (Case 3)**: When the generated logit is negative but GT logit is positive:
- This is a **false negative** (model missing anatomy)
- ReLU causes **zero gradient**
- The model receives **no learning signal** to fix this error!

**Critical Issue (Case 4)**: For background regions where both are negative:
- All background context provides **zero gradient**
- The model cannot learn subtle background texture matching
- Only foreground (positive logits) contributes to loss

#### Visual Impact

```
Gradient magnitude map (ReLU applied):

+-------------------+-------------------+
|                   |                   |
|   LUNG FIELD      |   BACKGROUND      |
|   (negative       |   (negative       |
|    logits)        |    logits)        |
|                   |                   |
|   GRADIENT = 0    |   GRADIENT = 0    |  ← No learning signal!
|                   |                   |
+-------------------+-------------------+
|                   |                   |
|   HEART           |   RIB             |
|   (positive       |   (positive       |
|    logits)        |    logits)        |
|                   |                   |
|   gradient ✓      |   gradient ✓      |  ← Only here!
|                   |                   |
+-------------------+-------------------+

Result: Model only learns from ~20-30% of spatial locations
        (those with positive logits for some class)
```

This concentration of gradients on small foreground regions causes:
1. **Unstable training**: Large sparse gradients
2. **Texture degradation**: Background gets no guidance
3. **Dark patches**: Model creates artifacts to maintain foreground features while ignoring background

---

## Part 2: Gradient Norm Tracking Implementation

### 2.1 Code Snippet for train_anatomy.py

Insert this code right before `optimizer.step()` in your training loop:

```python
# ===========================================================================
# GRADIENT NORM TRACKING - Insert in train_anatomy.py
# ===========================================================================
# Location: After loss_total = loss_dict["loss_total"]
#           Before accelerator.backward(loss)
# ===========================================================================

from typing import Dict, Optional
import torch

def compute_separate_grad_norms(
    model,
    loss_diffusion: torch.Tensor,
    loss_anatomy: torch.Tensor,
    lambda_anatomy: float,
    retain_graph: bool = True,
) -> Dict[str, float]:
    """
    Compute gradient norms separately for diffusion and anatomy losses.

    Uses torch.autograd.grad() to compute gradients without modifying .grad

    Args:
        model: The OmniGen model (with LoRA)
        loss_diffusion: Diffusion loss tensor (requires_grad=True)
        loss_anatomy: Raw anatomy loss before lambda scaling (requires_grad=True)
        lambda_anatomy: Scaling factor for anatomy loss
        retain_graph: Keep graph for subsequent backward() call

    Returns:
        Dict with:
            - grad_norm_diffusion: L2 norm of diffusion gradients
            - grad_norm_anatomy: L2 norm of scaled anatomy gradients
            - grad_norm_ratio: anatomy / diffusion
            - cosine_similarity: dot(g_diff, g_anat) / (|g_diff| * |g_anat|)
    """
    # Get LoRA parameters only
    lora_params = [p for n, p in model.named_parameters()
                   if "lora" in n.lower() and p.requires_grad]

    if len(lora_params) == 0:
        return {"grad_norm_diffusion": 0, "grad_norm_anatomy": 0,
                "grad_norm_ratio": 0, "cosine_similarity": 0}

    # Compute diffusion gradients
    try:
        grads_diff = torch.autograd.grad(
            loss_diffusion, lora_params,
            retain_graph=True, allow_unused=True
        )
    except RuntimeError:
        grads_diff = [None] * len(lora_params)

    # Compute anatomy gradients (with lambda scaling)
    scaled_anat = lambda_anatomy * loss_anatomy
    try:
        grads_anat = torch.autograd.grad(
            scaled_anat, lora_params,
            retain_graph=retain_graph, allow_unused=True
        )
    except RuntimeError:
        grads_anat = [None] * len(lora_params)

    # Compute norms and cosine similarity
    norm_diff_sq = 0.0
    norm_anat_sq = 0.0
    dot_product = 0.0

    for gd, ga in zip(grads_diff, grads_anat):
        if gd is not None:
            norm_diff_sq += gd.norm(2).item() ** 2
        if ga is not None:
            norm_anat_sq += ga.norm(2).item() ** 2
        if gd is not None and ga is not None:
            dot_product += (gd * ga).sum().item()

    norm_diff = norm_diff_sq ** 0.5
    norm_anat = norm_anat_sq ** 0.5

    if norm_diff > 1e-8 and norm_anat > 1e-8:
        cos_sim = dot_product / (norm_diff * norm_anat)
    else:
        cos_sim = 0.0

    ratio = norm_anat / norm_diff if norm_diff > 1e-8 else float('inf')

    return {
        "grad_norm_diffusion": norm_diff,
        "grad_norm_anatomy": norm_anat,
        "grad_norm_ratio": ratio,
        "cosine_similarity": cos_sim,
    }


# ===========================================================================
# Integration into training loop
# ===========================================================================
# Replace your existing loss computation and backward section with:

"""
    loss_dict = training_losses_with_anatomy(
        model=model,
        x1=output_images,
        model_kwargs=model_kwargs,
        output_images_pixel=output_images_pixel,
        vae=vae,
        seg_model=seg_model,
        lambda_anatomy=args.lambda_anatomy,
        anatomy_subbatch_size=args.anatomy_subbatch_size,
    )

    loss_total = loss_dict["loss_total"]
    loss_diffusion = loss_dict["loss_diffusion"]
    loss_anatomy = loss_dict["loss_anatomy"]

    # ===== GRADIENT TRACKING (every 100 steps) =====
    global_step = train_steps // args.gradient_accumulation_steps
    if global_step % 100 == 0 and loss_diffusion.requires_grad and loss_anatomy.requires_grad:
        grad_metrics = compute_separate_grad_norms(
            model=model,
            loss_diffusion=loss_diffusion,
            loss_anatomy=loss_anatomy,
            lambda_anatomy=args.lambda_anatomy,
            retain_graph=True,  # Keep graph for actual backward
        )

        if accelerator.is_main_process:
            logger.info(
                f"[GradNorm] Step {global_step}: "
                f"||g_diff||={grad_metrics['grad_norm_diffusion']:.4f}, "
                f"||g_anat||={grad_metrics['grad_norm_anatomy']:.4f}, "
                f"ratio={grad_metrics['grad_norm_ratio']:.4f}, "
                f"cos_sim={grad_metrics['cosine_similarity']:.4f}"
            )

            # Log to TensorBoard
            accelerator.log({
                "gradients/norm_diffusion": grad_metrics["grad_norm_diffusion"],
                "gradients/norm_anatomy": grad_metrics["grad_norm_anatomy"],
                "gradients/ratio_anat_over_diff": grad_metrics["grad_norm_ratio"],
                "gradients/cosine_similarity": grad_metrics["cosine_similarity"],
            }, step=global_step)

            # Alert on conflict
            if grad_metrics["cosine_similarity"] < -0.1:
                logger.warning(
                    f"[CONFLICT] Gradients opposing! cos_sim={grad_metrics['cosine_similarity']:.4f}"
                )
            if grad_metrics["grad_norm_ratio"] > 2.0:
                logger.warning(
                    f"[IMBALANCE] Anatomy gradients dominating! ratio={grad_metrics['grad_norm_ratio']:.4f}"
                )
    # ===== END GRADIENT TRACKING =====

    accelerator.backward(loss_total)
    # ... rest of training loop
"""
```

### 2.2 Interpretation Guide

After running with gradient tracking, interpret the logs as follows:

| Metric | Healthy Range | Warning | Critical |
|--------|--------------|---------|----------|
| `grad_norm_ratio` | 0.1 - 1.0 | > 2.0 | > 5.0 |
| `cosine_similarity` | > 0.3 | < 0.1 | < 0 |

**Diagnostic Table:**

| cos_sim | ratio | Diagnosis | Action |
|---------|-------|-----------|--------|
| > 0.5 | 0.1-1.0 | ✓ Healthy | Continue training |
| > 0.3 | > 2.0 | Anatomy dominates | Reduce λ by 2-5x |
| < 0.1 | any | Objectives orthogonal | Consider PCGrad or different loss |
| < 0 | any | **CONFLICT** | Stop and redesign loss |

---

## Part 3: The "Real Dice" Evaluation Script

This standalone script evaluates whether the anatomy actually improved by computing proper Dice scores against GT masks.

### 3.1 Complete Script: `eval_real_dice.py`

```python
#!/usr/bin/env python3
"""
eval_real_dice.py - Objective Anatomy Evaluation via True Dice Score

Compares Baseline OmniGen (30k) vs. OmniGen+SegMSE (10.5k more) by computing
actual Dice scores using the frozen ResNet34-UNet against Ground Truth masks.

This answers: "Did adding SegMSE loss actually improve anatomical correctness,
even though FID/LPIPS degraded?"

Usage:
    python eval_real_dice.py \
        --baseline_dir /path/to/outputs/cxr_finetune_lora_30000 \
        --segmse_dir /path/to/outputs/cxr_finetune_lora_30ksteps_SegMSE_lamda0.005_subbatch16_10500 \
        --gt_mask_dir /path/to/Segmentation/data/lidc_TotalSeg \
        --seg_model_ckpt /path/to/best_anatomy_model.pth \
        --output_csv anatomy_dice_comparison.csv \
        --output_json anatomy_dice_comparison.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add segmentation_models_pytorch to path
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")
import segmentation_models_pytorch as smp


# ===========================================================================
# Configuration - Must match your segmentation model training
# ===========================================================================

# Mapping from 119 TotalSegmentator classes to 10 target groups
TARGET_GROUPS = {
    "Lung_Left": [10, 11],                    # lung_upper_lobe_left, lung_lower_lobe_left
    "Lung_Right": [12, 13, 14],               # lung_upper/middle/lower_lobe_right
    "Heart": [51],                            # heart
    "Aorta": [52],                            # aorta
    "Liver": [5],                             # liver
    "Stomach": [6],                           # stomach
    "Trachea": [16],                          # trachea
    "Ribs": list(range(92, 116)),             # rib_left/right_1-12
    "Vertebrae": list(range(25, 51)),         # vertebrae_C1-L5
    "Upper_Skeleton": [69, 70, 71, 72, 73, 74]  # scapula, clavicle, humerus
}

CLASS_NAMES = list(TARGET_GROUPS.keys())
NUM_CLASSES = len(CLASS_NAMES)


# ===========================================================================
# Model Loading
# ===========================================================================

def load_seg_model(ckpt_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load the frozen ResNet34-UNet segmentation model.

    Args:
        ckpt_path: Path to checkpoint (.pth file)
        device: Target device

    Returns:
        Frozen evaluation-mode model
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,  # Raw logits
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    print(f"[INFO] Loaded seg model from: {ckpt_path}")
    if "val_dice" in ckpt:
        print(f"[INFO] Checkpoint validation Dice: {ckpt['val_dice']:.4f}")

    return model


# ===========================================================================
# Data Loading Utilities
# ===========================================================================

def load_image_tensor(path: str, device: str = "cuda") -> torch.Tensor:
    """
    Load image as tensor normalized to [-1, 1] (same as training).

    Args:
        path: Path to image file
        device: Target device

    Returns:
        Tensor of shape (1, 3, 256, 256) in [-1, 1] range
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256), Image.BICUBIC)

    arr = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    tensor = tensor / 127.5 - 1.0  # [0, 255] -> [-1, 1]

    return tensor.unsqueeze(0).to(device)


def load_gt_mask_10ch(
    mask_dir: str,
    patient_id: str,
    view_idx: int,
    device: str = "cuda"
) -> Optional[torch.Tensor]:
    """
    Load ground truth mask and convert from 119-channel to 10-channel format.

    Args:
        mask_dir: Root directory containing patient folders
        patient_id: e.g., "LIDC-IDRI-0001"
        view_idx: View index (integer)
        device: Target device

    Returns:
        Tensor of shape (10, 256, 256) with binary values, or None if not found
    """
    # Try multiple path formats
    possible_paths = [
        os.path.join(mask_dir, patient_id, "04_drr_256", "mask_compact", f"{view_idx:06d}.npz"),
        os.path.join(mask_dir, patient_id, "04_drr_256", "mask_compact", f"{view_idx:04d}.npz"),
        os.path.join(mask_dir, patient_id, "04_drr_256", "mask_compact", f"{view_idx}.npz"),
    ]

    mask_path = None
    for p in possible_paths:
        if os.path.exists(p):
            mask_path = p
            break

    if mask_path is None:
        return None

    # Load 119-channel mask
    data = np.load(mask_path)
    orig_mask = data["mask"]  # (119, 256, 256) or similar

    # Load labels_found metadata if available
    meta_path = os.path.join(mask_dir, patient_id, "02_totalseg", "phase2_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            labels_found = meta.get("labels_found", list(range(orig_mask.shape[0])))
    else:
        labels_found = list(range(orig_mask.shape[0]))

    # Convert to 10-channel format
    target_mask = np.zeros((NUM_CLASSES, 256, 256), dtype=np.float32)

    for ch_idx, (group_name, class_ids) in enumerate(TARGET_GROUPS.items()):
        channel_masks = []
        for cid in class_ids:
            if cid in labels_found and cid < orig_mask.shape[0]:
                channel_masks.append(orig_mask[cid])

        if channel_masks:
            merged = np.any(np.stack(channel_masks, axis=0), axis=0)
            target_mask[ch_idx] = merged.astype(np.float32)

    return torch.from_numpy(target_mask).to(device)


def discover_samples(gen_dir: str) -> List[Dict]:
    """
    Discover all generated images in a directory.

    Returns list of dicts with keys: patient_id, view_idx, gen_path
    """
    samples = []

    for patient_folder in os.listdir(gen_dir):
        patient_path = os.path.join(gen_dir, patient_folder)

        if not os.path.isdir(patient_path):
            continue
        if not patient_folder.startswith("LIDC-IDRI-"):
            continue

        for img_file in os.listdir(patient_path):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            view_idx_str = os.path.splitext(img_file)[0]
            try:
                view_idx = int(view_idx_str)
            except ValueError:
                continue

            samples.append({
                "patient_id": patient_folder,
                "view_idx": view_idx,
                "gen_path": os.path.join(patient_path, img_file),
            })

    return samples


# ===========================================================================
# Dice Score Computation
# ===========================================================================

def compute_dice(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> float:
    """
    Compute Dice score for binary masks.

    Args:
        pred: Binary prediction (H, W) or (C, H, W)
        gt: Binary ground truth (H, W) or (C, H, W)
        eps: Small constant to avoid division by zero

    Returns:
        Dice score in [0, 1]
    """
    pred = pred.float()
    gt = gt.float()

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()

    if union < eps:
        return 1.0  # Both empty = perfect match

    return (2.0 * intersection / (union + eps)).item()


@torch.no_grad()
def evaluate_single_image(
    model: torch.nn.Module,
    gen_path: str,
    gt_mask: torch.Tensor,
    device: str = "cuda",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate a single generated image against GT mask.

    Args:
        model: Segmentation model
        gen_path: Path to generated image
        gt_mask: Ground truth mask (10, H, W)
        device: Compute device
        threshold: Sigmoid threshold for binarization

    Returns:
        Dict with per-class Dice scores and macro Dice
    """
    # Load and predict
    img_tensor = load_image_tensor(gen_path, device)
    logits = model(img_tensor)  # (1, 10, H, W)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > threshold).squeeze(0).float()  # (10, H, W)

    # Compute per-class Dice
    results = {}
    all_dice = []

    for ch_idx, class_name in enumerate(CLASS_NAMES):
        dice = compute_dice(pred_mask[ch_idx], gt_mask[ch_idx])
        results[class_name] = dice
        all_dice.append(dice)

    results["macro_dice"] = np.mean(all_dice)

    return results


# ===========================================================================
# Main Evaluation Pipeline
# ===========================================================================

def evaluate_directory(
    gen_dir: str,
    gt_mask_dir: str,
    model: torch.nn.Module,
    device: str = "cuda",
    desc: str = "Evaluating",
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate all images in a generated directory.

    Returns:
        (aggregate_results, per_sample_results)
    """
    samples = discover_samples(gen_dir)
    print(f"[INFO] Found {len(samples)} generated images in {gen_dir}")

    # Accumulators for micro-averaged Dice
    tp_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    fp_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    fn_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)

    per_sample_results = []
    skipped = 0

    for sample in tqdm(samples, desc=desc):
        # Load GT mask
        gt_mask = load_gt_mask_10ch(
            gt_mask_dir, sample["patient_id"], sample["view_idx"], device
        )

        if gt_mask is None:
            skipped += 1
            continue

        # Evaluate
        sample_results = evaluate_single_image(
            model, sample["gen_path"], gt_mask, device
        )
        sample_results["patient_id"] = sample["patient_id"]
        sample_results["view_idx"] = sample["view_idx"]
        per_sample_results.append(sample_results)

        # Accumulate for micro-average
        img_tensor = load_image_tensor(sample["gen_path"], device)
        logits = model(img_tensor)
        pred_mask = (torch.sigmoid(logits) > 0.5).squeeze(0).float()

        tp = (pred_mask * gt_mask).sum(dim=(1, 2))
        fp = (pred_mask * (1 - gt_mask)).sum(dim=(1, 2))
        fn = ((1 - pred_mask) * gt_mask).sum(dim=(1, 2))

        tp_sum += tp.double()
        fp_sum += fp.double()
        fn_sum += fn.double()

    print(f"[INFO] Evaluated {len(per_sample_results)} samples, skipped {skipped}")

    # Compute aggregate metrics
    micro_dice = (2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum + 1e-7)).cpu().numpy()

    aggregate = {
        "n_samples": len(per_sample_results),
        "n_skipped": skipped,
        "micro_dice_per_class": {CLASS_NAMES[i]: float(micro_dice[i]) for i in range(NUM_CLASSES)},
        "micro_dice_macro": float(micro_dice.mean()),
        "macro_dice_mean": float(np.mean([r["macro_dice"] for r in per_sample_results])),
        "macro_dice_std": float(np.std([r["macro_dice"] for r in per_sample_results])),
    }

    return aggregate, per_sample_results


def format_comparison_report(
    baseline_agg: Dict,
    segmse_agg: Dict,
) -> str:
    """Generate formatted comparison report."""

    lines = []
    lines.append("=" * 85)
    lines.append("ANATOMY DICE SCORE COMPARISON: Baseline (30k) vs. +SegMSE (10.5k)")
    lines.append("=" * 85)
    lines.append("")
    lines.append(f"{'Class':<20} {'Baseline':<12} {'SegMSE':<12} {'Delta':<12} {'Status':<10}")
    lines.append("-" * 70)

    for class_name in CLASS_NAMES:
        b_dice = baseline_agg["micro_dice_per_class"][class_name]
        s_dice = segmse_agg["micro_dice_per_class"][class_name]
        delta = s_dice - b_dice

        if delta > 0.02:
            status = "+++"
        elif delta > 0.01:
            status = "++"
        elif delta > 0:
            status = "+"
        elif delta > -0.01:
            status = "~"
        elif delta > -0.02:
            status = "-"
        else:
            status = "---"

        delta_str = f"{delta:+.4f}"
        lines.append(f"{class_name:<20} {b_dice:.4f}       {s_dice:.4f}       {delta_str}      {status}")

    lines.append("-" * 70)

    # Macro averages
    b_macro = baseline_agg["micro_dice_macro"]
    s_macro = segmse_agg["micro_dice_macro"]
    delta_macro = s_macro - b_macro
    delta_str = f"{delta_macro:+.4f}"

    lines.append(f"{'MICRO-DICE (mean)':<20} {b_macro:.4f}       {s_macro:.4f}       {delta_str}")
    lines.append("")
    lines.append("=" * 85)
    lines.append("")

    # Interpretation
    lines.append("DIAGNOSTIC INTERPRETATION:")
    lines.append("-" * 40)

    if delta_macro > 0.02:
        lines.append("[SUCCESS] Anatomy significantly IMPROVED with SegMSE loss.")
        lines.append("          The FID/LPIPS degradation may be acceptable tradeoff.")
    elif delta_macro > 0:
        lines.append("[MILD GAIN] Slight anatomy improvement.")
        lines.append("          May not justify the visual quality degradation.")
    elif delta_macro > -0.01:
        lines.append("[NEUTRAL] Anatomy essentially unchanged.")
        lines.append("          SegMSE loss is NOT providing anatomical benefit.")
    else:
        lines.append("[FAILURE] Anatomy DEGRADED with SegMSE loss!")
        lines.append("          This indicates adversarial shortcuts or severe gradient conflict.")
        lines.append("          Recommendation: Disable SegMSE loss or redesign it.")

    lines.append("")

    # Check for class-specific patterns
    improved = [c for c in CLASS_NAMES
                if segmse_agg["micro_dice_per_class"][c] - baseline_agg["micro_dice_per_class"][c] > 0.02]
    degraded = [c for c in CLASS_NAMES
                if segmse_agg["micro_dice_per_class"][c] - baseline_agg["micro_dice_per_class"][c] < -0.02]

    if improved:
        lines.append(f"Classes with IMPROVEMENT (>2%): {', '.join(improved)}")
    if degraded:
        lines.append(f"Classes with DEGRADATION (<-2%): {', '.join(degraded)}")

    if improved and degraded:
        lines.append("")
        lines.append("[WARNING] Mixed pattern suggests localized artifact creation!")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Anatomy Dice Score: Baseline vs. SegMSE"
    )
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="Directory with baseline generated images")
    parser.add_argument("--segmse_dir", type=str, required=True,
                        help="Directory with SegMSE generated images")
    parser.add_argument("--gt_mask_dir", type=str, required=True,
                        help="Root directory with GT masks (lidc_TotalSeg)")
    parser.add_argument("--seg_model_ckpt", type=str, required=True,
                        help="Path to segmentation model checkpoint")
    parser.add_argument("--output_csv", type=str, default="anatomy_dice_comparison.csv",
                        help="Output CSV file for per-sample results")
    parser.add_argument("--output_json", type=str, default="anatomy_dice_comparison.json",
                        help="Output JSON file for aggregate results")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    model = load_seg_model(args.seg_model_ckpt, device)

    # Evaluate baseline
    print("\n" + "=" * 50)
    print("Phase 1: Evaluating Baseline (30k steps)")
    print("=" * 50)
    baseline_agg, baseline_samples = evaluate_directory(
        args.baseline_dir, args.gt_mask_dir, model, device, desc="Baseline"
    )

    # Evaluate SegMSE
    print("\n" + "=" * 50)
    print("Phase 2: Evaluating SegMSE (+10.5k steps)")
    print("=" * 50)
    segmse_agg, segmse_samples = evaluate_directory(
        args.segmse_dir, args.gt_mask_dir, model, device, desc="SegMSE"
    )

    # Generate report
    report = format_comparison_report(baseline_agg, segmse_agg)
    print("\n" + report)

    # Save results
    results = {
        "baseline": baseline_agg,
        "segmse": segmse_agg,
        "delta": {
            "micro_dice_per_class": {
                c: segmse_agg["micro_dice_per_class"][c] - baseline_agg["micro_dice_per_class"][c]
                for c in CLASS_NAMES
            },
            "micro_dice_macro": segmse_agg["micro_dice_macro"] - baseline_agg["micro_dice_macro"],
        }
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Aggregate results saved to: {args.output_json}")

    # Save per-sample CSV
    all_samples = []
    for s in baseline_samples:
        s["model"] = "baseline"
        all_samples.append(s)
    for s in segmse_samples:
        s["model"] = "segmse"
        all_samples.append(s)

    df = pd.DataFrame(all_samples)
    df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Per-sample results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
```

### 3.2 Running the Evaluation

```bash
cd /home/wenting/zr/gen_code

python diagnostics/eval_real_dice.py \
    --baseline_dir outputs/cxr_finetune_lora_30000 \
    --segmse_dir outputs/cxr_finetune_lora_30ksteps_SegMSE_lamda0.005_subbatch16_10500 \
    --gt_mask_dir /home/wenting/zr/Segmentation/data/lidc_TotalSeg \
    --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
    --output_json diagnostics/dice_comparison_results.json \
    --output_csv diagnostics/dice_comparison_per_sample.csv
```

---

## Part 4: Alternative Loss Formulation Proposals

Given the identified problems with the current approach, here are two alternative implementations that address the core issues.

### 4.1 Thresholded Logits Loss (Margin-Based)

**Motivation**: Allow OmniGen freedom to generate varied textures as long as the anatomy is "roughly correct" (within a margin).

```python
# ===========================================================================
# Alternative 1: Thresholded Logits Loss (loss_anatomy_thresholded.py)
# ===========================================================================

import torch
import torch.nn.functional as F
from typing import Tuple

def compute_thresholded_logits_loss(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    margin: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Thresholded Logits Loss: Only penalize if logit difference exceeds margin.

    This allows the generator freedom for texture variation as long as
    the anatomical structure is approximately correct.

    L = mean(max(0, |gen_logit - gt_logit| - margin)^2)

    Args:
        seg_model: Frozen ResNet34-UNet (eval mode)
        gen_images: (B, 3, H, W) generated images in [-1, 1], WITH grad
        gt_images: (B, 3, H, W) ground truth images in [-1, 1]
        margin: Tolerance margin in logit space (default 2.0)
                - Logit diff of 2.0 corresponds to ~73% vs ~27% prob difference
                - Only penalize errors larger than this
        reduction: "mean" or "sum"

    Returns:
        Scalar loss tensor

    Mathematical Properties:
        - When |diff| <= margin: loss = 0, gradient = 0
        - When |diff| > margin: loss = (|diff| - margin)^2
        - Gradient only flows for "large" errors
        - Preserves generator freedom for fine texture details
    """
    # GT logits: no grad needed
    with torch.no_grad():
        gt_logits = seg_model(gt_images)  # (B, C, H, W)

    # Gen logits: WITH grad
    gen_logits = seg_model(gen_images)  # (B, C, H, W)

    # Compute absolute difference
    diff = torch.abs(gen_logits - gt_logits)  # (B, C, H, W)

    # Apply margin threshold: max(0, |diff| - margin)
    # Only penalize differences larger than margin
    thresholded_diff = F.relu(diff - margin)  # (B, C, H, W)

    # Squared loss on thresholded differences
    loss = thresholded_diff ** 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def compute_asymmetric_thresholded_loss(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    margin_positive: float = 1.5,
    margin_negative: float = 3.0,
) -> torch.Tensor:
    """
    Asymmetric Thresholded Loss: Different margins for over/under-prediction.

    Rationale:
        - Under-predicting anatomy (missing structures) is worse than over-predicting
        - Use smaller margin for false negatives, larger for false positives

    Args:
        margin_positive: Margin when gen_logit > gt_logit (over-prediction)
        margin_negative: Margin when gen_logit < gt_logit (under-prediction)
    """
    with torch.no_grad():
        gt_logits = seg_model(gt_images)

    gen_logits = seg_model(gen_images)

    diff = gen_logits - gt_logits

    # Over-prediction: gen > gt (positive diff)
    over_pred = F.relu(diff - margin_positive)

    # Under-prediction: gen < gt (negative diff, use abs and separate margin)
    under_pred = F.relu(-diff - margin_negative)

    # Weight under-prediction more heavily (missing anatomy is worse)
    loss = over_pred ** 2 + 2.0 * under_pred ** 2

    return loss.mean()


# ===========================================================================
# Integration into training_losses_with_anatomy
# ===========================================================================

def compute_feature_matching_loss_thresholded(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """
    Drop-in replacement for compute_feature_matching_loss() in loss_anatomy.py

    Replace the call in training_losses_with_anatomy():
        OLD: loss_i = compute_feature_matching_loss(seg_model, gen_decoded, gt_img)
        NEW: loss_i = compute_feature_matching_loss_thresholded(seg_model, gen_decoded, gt_img, margin=2.0)
    """
    return compute_thresholded_logits_loss(
        seg_model=seg_model,
        gen_images=gen_images,
        gt_images=gt_images,
        margin=margin,
        reduction="mean",
    )


# ===========================================================================
# Hyperparameter Guide
# ===========================================================================
"""
Margin Selection Guide:

margin=1.0: Tight constraint
    - Probability tolerance: ~73% vs ~50%
    - Use when anatomy precision is critical
    - Higher risk of gradient conflict

margin=2.0: Moderate constraint (RECOMMENDED START)
    - Probability tolerance: ~88% vs ~50%
    - Good balance of anatomy guidance and generator freedom
    - Lower gradient conflict risk

margin=3.0: Loose constraint
    - Probability tolerance: ~95% vs ~50%
    - Maximum generator freedom
    - Minimal anatomy guidance (may be too weak)

Tuning Strategy:
    1. Start with margin=2.0
    2. If anatomy improves but quality still degrades, try margin=3.0
    3. If anatomy doesn't improve enough, try margin=1.5
    4. Monitor gradient conflict during tuning
"""
```

### 4.2 Blurred Feature Matching (Gaussian Smoothing)

**Motivation**: Align macro-structures (organ shapes) without overfitting to high-frequency pixel artifacts.

```python
# ===========================================================================
# Alternative 2: Blurred Feature Matching (loss_anatomy_blurred.py)
# ===========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class GaussianBlur2d(nn.Module):
    """
    Differentiable 2D Gaussian blur for feature map smoothing.

    Uses separable convolutions for efficiency.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 11,
        sigma: float = 3.0,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create 1D Gaussian kernel
        kernel_1d = self._gaussian_kernel_1d(kernel_size, sigma)

        # Create separable 2D kernels
        kernel_h = kernel_1d.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
        kernel_v = kernel_1d.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)

        self.register_buffer("kernel_h", kernel_h)
        self.register_buffer("kernel_v", kernel_v)

        self.padding = kernel_size // 2

    def _gaussian_kernel_1d(self, size: int, sigma: float) -> torch.Tensor:
        """Generate 1D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur.

        Args:
            x: (B, C, H, W) feature maps

        Returns:
            (B, C, H, W) blurred feature maps
        """
        # Separable convolution: horizontal then vertical
        x = F.conv2d(
            x, self.kernel_h, padding=(0, self.padding), groups=self.channels
        )
        x = F.conv2d(
            x, self.kernel_v, padding=(self.padding, 0), groups=self.channels
        )
        return x


def compute_blurred_feature_matching_loss(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    blur_sigma: float = 3.0,
    blur_kernel_size: int = 11,
    num_classes: int = 10,
) -> torch.Tensor:
    """
    Blurred Feature Matching Loss: Align macro-structures, ignore high-frequency.

    Strategy:
        1. Get logits from both generated and GT images
        2. Apply Gaussian blur to both
        3. Compute MSE on blurred logits

    This forces the model to align large-scale anatomical shapes without
    overfitting to pixel-level variations or artifacts.

    Args:
        seg_model: Frozen ResNet34-UNet
        gen_images: (B, 3, H, W) generated images, WITH grad
        gt_images: (B, 3, H, W) GT images
        blur_sigma: Gaussian sigma (higher = more blur, default 3.0)
        blur_kernel_size: Kernel size (must be odd, default 11)
        num_classes: Number of segmentation classes (default 10)

    Returns:
        Scalar loss tensor

    Hyperparameter Guide:
        blur_sigma=2.0: Light blur, preserves some fine structure
        blur_sigma=3.0: Moderate blur (RECOMMENDED)
        blur_sigma=5.0: Heavy blur, only coarse shapes matter
    """
    device = gen_images.device

    # Create blur module (cached in practice)
    blur = GaussianBlur2d(
        channels=num_classes,
        kernel_size=blur_kernel_size,
        sigma=blur_sigma,
    ).to(device)

    # Get logits
    with torch.no_grad():
        gt_logits = seg_model(gt_images)  # (B, C, H, W)
        gt_logits_blurred = blur(gt_logits)

    # Gen logits: WITH grad
    gen_logits = seg_model(gen_images)  # (B, C, H, W)
    gen_logits_blurred = blur(gen_logits)  # Gradient flows through blur

    # MSE on blurred logits
    loss = F.mse_loss(gen_logits_blurred, gt_logits_blurred)

    return loss


class MultiScaleFeatureMatchingLoss(nn.Module):
    """
    Multi-Scale Feature Matching: Combine multiple blur levels.

    Captures both coarse structure (large sigma) and finer detail (small sigma).
    """
    def __init__(
        self,
        num_classes: int = 10,
        sigmas: Tuple[float, ...] = (1.5, 3.0, 5.0),
        weights: Tuple[float, ...] = (0.2, 0.5, 0.3),
    ):
        super().__init__()

        assert len(sigmas) == len(weights)
        assert abs(sum(weights) - 1.0) < 1e-6

        self.sigmas = sigmas
        self.weights = weights
        self.num_classes = num_classes

        # Create blur modules for each scale
        self.blurs = nn.ModuleList([
            GaussianBlur2d(num_classes, kernel_size=int(sigma * 4) | 1, sigma=sigma)
            for sigma in sigmas
        ])

    def forward(
        self,
        seg_model: torch.nn.Module,
        gen_images: torch.Tensor,
        gt_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-scale feature matching loss.
        """
        # Get logits
        with torch.no_grad():
            gt_logits = seg_model(gt_images)
        gen_logits = seg_model(gen_images)

        total_loss = 0.0

        for blur, weight in zip(self.blurs, self.weights):
            gt_blurred = blur(gt_logits.detach())
            gen_blurred = blur(gen_logits)

            scale_loss = F.mse_loss(gen_blurred, gt_blurred)
            total_loss = total_loss + weight * scale_loss

        return total_loss


# ===========================================================================
# Alternative: Intermediate Feature Matching (Not Just Final Logits)
# ===========================================================================

def compute_intermediate_feature_matching_loss(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    feature_layers: Tuple[str, ...] = ("encoder.layer2", "encoder.layer3", "encoder.layer4"),
    weights: Tuple[float, ...] = (0.2, 0.3, 0.5),
    blur_sigma: float = 2.0,
) -> torch.Tensor:
    """
    Match intermediate encoder features, not just final logits.

    Rationale:
        - Early layers capture texture/edges (blur more heavily)
        - Later layers capture semantic structure (blur less)
        - Avoids overfitting to output space artifacts

    WARNING: Requires modifying seg_model to return intermediate features.
             This is a more advanced approach requiring hook-based extraction.
    """
    # This requires a forward hook to extract intermediate features
    # Implementation skeleton - needs seg_model modification

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook

    # Register hooks
    hooks = []
    for layer_name in feature_layers:
        layer = dict(seg_model.named_modules())[layer_name]
        hooks.append(layer.register_forward_hook(get_activation(layer_name)))

    try:
        # Forward pass for GT (no grad)
        with torch.no_grad():
            _ = seg_model(gt_images)
            gt_features = {k: v.clone() for k, v in activations.items()}

        activations.clear()

        # Forward pass for generated (with grad)
        _ = seg_model(gen_images)
        gen_features = activations

        # Compute loss at each layer
        total_loss = 0.0

        for (layer_name, weight) in zip(feature_layers, weights):
            gt_feat = gt_features[layer_name]
            gen_feat = gen_features[layer_name]

            # Optionally blur features
            if blur_sigma > 0:
                C = gt_feat.shape[1]
                blur = GaussianBlur2d(C, kernel_size=int(blur_sigma * 4) | 1, sigma=blur_sigma)
                blur = blur.to(gt_feat.device)
                gt_feat = blur(gt_feat)
                gen_feat = blur(gen_feat)

            layer_loss = F.mse_loss(gen_feat, gt_feat)
            total_loss = total_loss + weight * layer_loss

        return total_loss

    finally:
        # Remove hooks
        for h in hooks:
            h.remove()


# ===========================================================================
# Integration Example
# ===========================================================================

def compute_feature_matching_loss_blurred(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    blur_sigma: float = 3.0,
) -> torch.Tensor:
    """
    Drop-in replacement for compute_feature_matching_loss() in loss_anatomy.py

    Replace the call in training_losses_with_anatomy():
        OLD: loss_i = compute_feature_matching_loss(seg_model, gen_decoded, gt_img)
        NEW: loss_i = compute_feature_matching_loss_blurred(seg_model, gen_decoded, gt_img, blur_sigma=3.0)
    """
    return compute_blurred_feature_matching_loss(
        seg_model=seg_model,
        gen_images=gen_images,
        gt_images=gt_images,
        blur_sigma=blur_sigma,
        blur_kernel_size=int(blur_sigma * 4) | 1,  # Ensure odd
        num_classes=10,
    )


# ===========================================================================
# Hyperparameter Selection Guide
# ===========================================================================
"""
Blur Sigma Selection:

sigma=1.5: Light blur
    - Equivalent receptive field: ~6 pixels
    - Preserves organ boundaries
    - Still sensitive to local artifacts

sigma=3.0: Moderate blur (RECOMMENDED START)
    - Equivalent receptive field: ~12 pixels
    - Focuses on organ shapes, smooths edges
    - Good balance of structure and freedom

sigma=5.0: Heavy blur
    - Equivalent receptive field: ~20 pixels
    - Only coarse organ positions matter
    - Maximum generator freedom
    - Risk: Too loose, anatomy not enforced

Multi-Scale Recommendation:
    sigmas = (1.5, 3.0, 5.0)
    weights = (0.2, 0.5, 0.3)

    This captures:
    - 20% weight on fine structure
    - 50% weight on medium-scale shapes
    - 30% weight on coarse organ positions
"""
```

---

## Part 5: Recommended Next Steps

### 5.1 Immediate Diagnostic Actions

1. **Run Dice Evaluation** (Part 3):
   ```bash
   python diagnostics/eval_real_dice.py \
       --baseline_dir outputs/cxr_finetune_lora_30000 \
       --segmse_dir outputs/cxr_finetune_lora_30ksteps_SegMSE_lamda0.005_subbatch16_10500 \
       --gt_mask_dir /home/wenting/zr/Segmentation/data/lidc_TotalSeg \
       --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth
   ```

   **Expected Outcome**: If macro Dice did NOT improve, the SegMSE loss is not providing actual anatomical benefit.

2. **Enable Gradient Tracking** (Part 2):
   - Add the gradient monitoring code to `train_anatomy.py`
   - Retrain with monitoring enabled
   - Check for gradient conflict (cos_sim < 0)

### 5.2 If Dice Did NOT Improve

This confirms the SegMSE loss is creating adversarial shortcuts rather than learning anatomy.

**Action Plan**:
1. Remove SegMSE loss entirely and verify baseline quality recovers
2. Choose ONE alternative from Part 4:
   - **Thresholded Logits** (easier, robust)
   - **Blurred Feature Matching** (better for macro-shapes)
3. Retrain with reduced λ (try 0.001 instead of 0.005)

### 5.3 If Dice DID Improve But FID Degraded

This indicates a Pareto tradeoff between anatomy and visual quality.

**Action Plan**:
1. Accept that some FID increase is inevitable
2. Use Blurred Feature Matching to reduce artifact sensitivity
3. Tune λ to find the best tradeoff point
4. Consider reporting both FID and Dice as separate metrics

### 5.4 Long-Term Architectural Recommendations

1. **Consider Perceptual Anatomy Loss**: Replace MSE with a perceptual loss (LPIPS) computed on UNet features

2. **Gradient Projection (PCGrad)**: If conflict persists, implement PCGrad to project anatomy gradients onto the orthogonal complement of diffusion gradients

3. **Curriculum Training**: Start with diffusion-only training until convergence, then gradually introduce anatomy loss with warmup schedule

---

## Appendix A: Quick Reference

### Loss Implementations at a Glance

| Method | Code Change | Pros | Cons |
|--------|-------------|------|------|
| **Current (Raw MSE)** | None | Simple | Sensitive to artifacts |
| **Thresholded (Margin)** | Replace `compute_feature_matching_loss` | Robust to noise | Need to tune margin |
| **Blurred (Gaussian)** | Replace `compute_feature_matching_loss` | Focus on shapes | May miss fine detail |
| **Multi-Scale** | New module | Best of both | More compute |

### Key Hyperparameters

| Parameter | Current | Recommended Range | Notes |
|-----------|---------|-------------------|-------|
| `lambda_anatomy` | 0.005 | 0.001 - 0.01 | Lower if gradient conflict |
| `margin` (thresholded) | N/A | 1.5 - 3.0 | Higher = more freedom |
| `blur_sigma` | N/A | 2.0 - 5.0 | Higher = coarser |

---

*End of Report*
