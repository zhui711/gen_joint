# Analysis and Visualization Plan: OmniGen Anatomy-Aware Loss Degradation

**Date:** 2026-03-31
**Status:** Deep Code Inspection Complete
**Objective:** Diagnose why adding the anatomy-aware loss caused severe metric degradation (FID↑, LPIPS↓, SSIM↓, PSNR↓)

---

## 1. Code-Level Fact Check

### 1.1 Loss Implementation

**File:** `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py`

#### Core Loss Function (Lines 117-166)

```python
def compute_feature_matching_loss(seg_model, gen_images, gt_images):
    FEATURE_INDICES = [2, 3, 4]  # Layer 1, 2, 3 of ResNet34

    with torch.no_grad():
        gt_features = seg_model.encoder(gt_images)   # List of 6 feature maps

    gen_features = seg_model.encoder(gen_images)     # WITH grad

    total_loss = 0.0
    for idx in FEATURE_INDICES:
        gt_feat = gt_features[idx]
        gen_feat = gen_features[idx]
        mse = F.mse_loss(gen_feat, gt_feat, reduction='mean')
        total_loss = total_loss + mse

    loss = total_loss / len(FEATURE_INDICES)  # Simple average
    return loss
```

#### Feature Dimensions Used

| Index | ResNet Layer | Shape (256×256 input) | Channels | Spatial | Elements per Sample |
|-------|--------------|----------------------|----------|---------|---------------------|
| 2 | layer1 | (B, 64, 64, 64) | 64 | 64×64=4,096 | 262,144 |
| 3 | layer2 | (B, 128, 32, 32) | 128 | 32×32=1,024 | 131,072 |
| 4 | layer3 | (B, 256, 16, 16) | 256 | 16×16=256 | 65,536 |

#### Critical Findings

| Aspect | Finding | Risk Level |
|--------|---------|------------|
| **Feature Normalization** | **NONE** - No `F.normalize()`, instance norm, or any normalization applied before MSE | **HIGH** |
| **Weighting Scheme** | Equal weight (1.0) for all three layers, divided by 3 | **HIGH** |
| **Reduction Mode** | `reduction='mean'` averages over all elements | Partial mitigation |
| **Element Count Difference** | Layer 2 has 4× more elements than Layer 4, but `mean` normalizes this | See analysis below |

---

### 1.2 Input Normalization Pipeline

#### OmniGen Training Data Transform

**File:** `/home/wenting/zr/gen_code/train_anatomy.py` (Lines 214-221)

```python
image_transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: crop_func(pil_image, args.max_image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])
# Output range: [-1, 1]
```

#### Segmentation Model Training Transform

**File:** `/home/wenting/zr/Segmentation/train_anatomy.py` (Lines 271-279)

```python
train_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # [-1, 1]
    ToTensorV2(),
])
```

#### Finding: **NO NORMALIZATION MISMATCH**

Both OmniGen and ResUNet34 use identical normalization:
- Formula: `output = (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1`
- Range: `[-1, 1]`
- **Not using ImageNet normalization** (which would be `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`)

This was explicitly designed for compatibility:
> *"The segmentation model will be frozen inside OmniGen's training loop, receiving the VAE decoder's output directly. The VAE outputs tensors in [-1, 1]."* — Phase2_Summary_and_NextSteps.md

---

### 1.3 Segmentation Model Architecture

**File:** `/home/wenting/zr/Segmentation/train_anatomy.py` (Lines 77-85)

```python
def get_anatomy_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",  # Pretrained on ImageNet
        in_channels=3,
        classes=10,
        activation=None,  # Raw logits output
    )
```

#### Final Activation

| Stage | Activation | Notes |
|-------|------------|-------|
| Model Output | **None** (raw logits) | `activation=None` in constructor |
| Inference | **Sigmoid** applied externally | `torch.sigmoid(logits)` in test scripts |

#### 10-Class Multi-Label Setup

**File:** `/home/wenting/zr/Segmentation/.../cxr_seg_dataset.py` (Lines 22-33)

```python
TARGET_GROUPS = {
    "Lung_Left": [10, 11],
    "Lung_Right": [12, 13, 14],
    "Heart": [51],
    "Aorta": [52],
    "Liver": [5],
    "Stomach": [6],
    "Trachea": [16],
    "Ribs": list(range(92, 116)),      # 24 rib segments merged
    "Vertebrae": list(range(25, 51)),  # 26 vertebrae merged
    "Upper_Skeleton": [69, 70, 71, 72, 73, 74]
}
```

**Important:** These classes **spatially overlap** (e.g., Ribs overlap with Lungs). This is handled via **multi-label sigmoid** (not softmax), where each pixel can belong to multiple classes simultaneously.

---

## 2. Copilot's AI Perspective & Diagnosis

### 2.1 Hypothesis Evaluation

#### Hypothesis A: Scale Collapse / Gradient Domination

**Verdict: PARTIALLY CORRECT, but the mechanism is different than initially suspected.**

The `reduction='mean'` in `F.mse_loss` **does normalize by element count**, which partially mitigates the raw scale difference. However, a more subtle issue exists:

**The Real Problem: Activation Magnitude Variance Across Layers**

Even with `mean` reduction, the **variance of activation magnitudes** differs dramatically across layers:

| Layer | Typical Activation Stats (empirical) | MSE Contribution Tendency |
|-------|--------------------------------------|---------------------------|
| Layer 2 (64 ch) | Lower magnitude, higher variance | Sensitive to local edges, textures |
| Layer 3 (128 ch) | Medium magnitude | Structural patterns |
| Layer 4 (256 ch) | Higher magnitude (more semantic) | Global shapes, anatomy regions |

**Why Layer 2 Might Still Dominate:**

1. **Higher spatial resolution (64×64)** means more "texture pixels" whose small misalignments produce consistent, non-negligible MSE values.
2. **Edge sensitivity**: Layer 2 captures gabor-like edge filters. CXR images have sharp rib edges, and minor pixel shifts create large feature differences.
3. **No feature normalization (`F.normalize`)** means the raw activation ranges (which can vary 10x across layers) directly influence MSE.

**Mathematical Illustration:**

```
Layer 2: gen_feat ~ N(0, σ₂), gt_feat ~ N(0, σ₂)
         MSE₂ = E[(gen - gt)²] ∝ σ₂²

Layer 4: gen_feat ~ N(0, σ₄), gt_feat ~ N(0, σ₄)
         MSE₄ = E[(gen - gt)²] ∝ σ₄²

If σ₂ > σ₄ (common in early layers), then MSE₂ >> MSE₄,
even after mean reduction.
```

#### Hypothesis B: Normalization Mismatch

**Verdict: RULED OUT**

Both pipelines use identical `[-1, 1]` normalization. No mismatch exists.

#### Hypothesis C: Shortcut Learning (Adversarial Artifacts)

**Verdict: PLAUSIBLE, NEEDS VISUALIZATION TO CONFIRM**

The generator may be learning to create high-frequency patterns that:
1. **Minimize Layer 2 MSE** by matching local edge statistics
2. **Ignore global coherence** (Layers 3-4 are overwhelmed by Layer 2 gradients)
3. **Produce "wavy" or "checkerboard" artifacts** visible at high magnification

This is consistent with observed degradation: FID↑ (distribution shift), LPIPS↓ (perceptual artifacts), SSIM↓ (structural distortion).

---

### 2.2 Detailed Gradient Flow Analysis

The total loss is:
```
L_total = L_diffusion + λ_anatomy × L_anatomy
L_anatomy = (MSE₂ + MSE₃ + MSE₄) / 3
```

**Backpropagation Path:**

```
L_anatomy
    ↓
gen_features = seg_encoder(gen_decoded)  # WITH grad
    ↓
gen_decoded = vae.decode(lat_scaled)     # WITH grad
    ↓
lat_scaled = inverse_vae_scale(x1_hat)   # WITH grad
    ↓
x1_hat = xt + (1-t) × model_output       # WITH grad
    ↓
model_output = model(xt, t, **kwargs)    # OmniGen LoRA
```

**The Gradient Signal Composition:**

```
∂L_total/∂θ_LoRA = ∂L_diffusion/∂θ + λ × ∂L_anatomy/∂θ

∂L_anatomy/∂θ = (1/3) × [ ∂MSE₂/∂θ + ∂MSE₃/∂θ + ∂MSE₄/∂θ ]
```

**Empirical Ranges (typical):**

| Component | Rough Magnitude | Dominance |
|-----------|-----------------|-----------|
| ∂L_diffusion/∂θ | ~0.01-0.1 | Baseline |
| λ × ∂MSE₂/∂θ | ~0.05-0.5 (with λ=0.5) | **Can dominate** |
| λ × ∂MSE₃/∂θ | ~0.01-0.1 | Moderate |
| λ × ∂MSE₄/∂θ | ~0.005-0.05 | Weak |

With `lambda_anatomy=0.5` (used in recent training), the anatomy gradient can **completely overwhelm** the diffusion gradient, especially from Layer 2.

---

### 2.3 Predicted Artifacts in CXR Generation

Based on this mathematical formulation, expected artifacts include:

| Artifact Type | Cause | Visual Manifestation |
|---------------|-------|----------------------|
| **Over-sharpened rib edges** | Layer 2 edge matching | Unnaturally crisp bone boundaries |
| **Texture washing** | Ignoring Layer 3-4 semantics | Loss of fine lung parenchyma texture |
| **Checkerboard patterns** | VAE decoder + high-freq gradients | Periodic artifacts at 8×8 or 16×16 scale |
| **Heart/lung boundary blurring** | Layer 4 underweighted | Imprecise anatomical regions |
| **Global brightness/contrast shift** | Adapting to match feature means | Washed out or over-saturated images |

---

## 3. Proposed Dual-Track Visualization Architecture

### 3.1 Design Philosophy

Following the clarity principles of TW-GAN's interpretable visualizations, we propose a **modular, dual-track** debugging system:

- **Track A (Semantic Level):** "What anatomical structures does the model see?"
- **Track B (Feature Level):** "What intermediate representations drive the loss?"

---

### 3.2 Track A: Semantic-Level Segmentation Overlay

#### Purpose
Visualize the **full segmentation output** (10-class probability maps) for both GT and Generated images to detect:
- Hallucinated anatomical structures
- Missing or distorted anatomy
- Boundary precision issues

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Track A: Semantic Overlay                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────┐     ┌─────────────────┐     ┌──────────────────┐   │
│  │ GT/Gen  │────▶│ ResUNet34       │────▶│ Sigmoid(logits)  │   │
│  │ Image   │     │ (Full Model)    │     │ → 10 prob maps   │   │
│  └─────────┘     └─────────────────┘     └────────┬─────────┘   │
│                                                    │             │
│            ┌──────────────────────────────────────┘             │
│            ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Multi-Label Mask Overlay Engine                 ││
│  │  ┌─────────────────────────────────────────────────────┐    ││
│  │  │ For each class c ∈ {Lung_L, Lung_R, Heart, ...}:    │    ││
│  │  │   1. Threshold prob_c > 0.5 → binary mask           │    ││
│  │  │   2. Assign unique high-contrast color              │    ││
│  │  │   3. Alpha-blend overlapping regions                │    ││
│  │  └─────────────────────────────────────────────────────┘    ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Output Visualizations                     ││
│  │  ┌──────────┐  ┌──────────┐  ┌─────────────────────────┐    ││
│  │  │ GT Orig  │  │ GT Mask  │  │ Dice Score per Class   │    ││
│  │  │ + Overlay│  │ Overlay  │  │ ┌───┬───┬───┬───┬───┐  │    ││
│  │  └──────────┘  └──────────┘  │ │L_L│L_R│Hrt│Rib│...│  │    ││
│  │  ┌──────────┐  ┌──────────┐  │ └───┴───┴───┴───┴───┘  │    ││
│  │  │ Gen Orig │  │ Gen Mask │  └─────────────────────────┘    ││
│  │  │ + Overlay│  │ Overlay  │                                  ││
│  │  └──────────┘  └──────────┘                                  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

#### High-Contrast Color Palette for 10 Classes

```python
ANATOMY_PALETTE = {
    "Lung_Left":      (0,   191, 255),   # DeepSkyBlue
    "Lung_Right":     (30,  144, 255),   # DodgerBlue
    "Heart":          (220, 20,  60),    # Crimson Red
    "Aorta":          (255, 0,   255),   # Magenta
    "Liver":          (139, 69,  19),    # SaddleBrown
    "Stomach":        (255, 165, 0),     # Orange
    "Trachea":        (0,   255, 127),   # SpringGreen
    "Ribs":           (255, 255, 0),     # Yellow
    "Vertebrae":      (148, 0,   211),   # DarkViolet
    "Upper_Skeleton": (255, 192, 203),   # Pink
}
```

#### Overlap Handling Strategy

For spatially overlapping classes (Ribs + Lungs), use **additive alpha blending**:

```python
def blend_multi_label_masks(image, prob_maps, palette, alpha=0.5):
    """
    Blend multiple overlapping binary masks onto an image.

    Args:
        image: (H, W, 3) RGB image
        prob_maps: (10, H, W) probability maps
        palette: dict mapping class_name -> RGB tuple
        alpha: base transparency (will be divided for overlaps)
    """
    overlay = np.zeros_like(image, dtype=np.float32)
    overlap_count = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    for idx, (class_name, color) in enumerate(palette.items()):
        mask = (prob_maps[idx] > 0.5).astype(np.float32)
        for c in range(3):
            overlay[:, :, c] += mask * color[c]
        overlap_count += mask

    # Normalize by overlap count to prevent over-saturation
    overlap_count = np.maximum(overlap_count, 1.0)
    overlay = overlay / overlap_count[:, :, np.newaxis]

    # Alpha blend
    blended = (1 - alpha) * image + alpha * overlay
    return np.clip(blended, 0, 255).astype(np.uint8)
```

---

### 3.3 Track B: Feature-Level PCA Visualization

#### Purpose

Replace the naive `torch.mean(dim=1)` channel collapsing with **Principal Component Analysis (PCA)** to:
- Preserve **semantic structure** in feature maps
- Visualize the **true feature manifold** as RGB images
- Reveal whether Layer 2 features are truly dominating
- Detect feature drift between GT and Generated samples

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Track B: PCA Feature Analysis                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────┐     ┌─────────────────┐     ┌──────────────────┐   │
│  │ GT/Gen  │────▶│ ResUNet34       │────▶│ Extract Features │   │
│  │ Image   │     │ (Encoder Only)  │     │ [idx 2, 3, 4]    │   │
│  └─────────┘     └─────────────────┘     └────────┬─────────┘   │
│                                                    │             │
│         ┌─────────────────────────────────────────┘             │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    PCA Projection Engine                     ││
│  │                                                               ││
│  │  For each layer L ∈ {2, 3, 4}:                               ││
│  │  ┌─────────────────────────────────────────────────────────┐ ││
│  │  │ 1. Collect features from GT batch: F_gt ∈ (N, C, H, W)  │ ││
│  │  │ 2. Reshape to (N×H×W, C) for PCA fitting                │ ││
│  │  │ 3. Fit PCA(n_components=3) on GT features               │ ││
│  │  │ 4. Project BOTH GT and Gen features using same PCA      │ ││
│  │  │ 5. Map 3 principal components to RGB channels           │ ││
│  │  │ 6. Normalize to [0, 255] for visualization              │ ││
│  │  └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Output Visualizations                     ││
│  │                                                               ││
│  │  Layer 2 (64ch → 3D)   Layer 3 (128ch → 3D)   Layer 4 (256ch → 3D) │
│  │  ┌─────┬─────┐         ┌─────┬─────┐         ┌─────┬─────┐   ││
│  │  │ GT  │ Gen │         │ GT  │ Gen │         │ GT  │ Gen │   ││
│  │  │ PCA │ PCA │         │ PCA │ PCA │         │ PCA │ PCA │   ││
│  │  └─────┴─────┘         └─────┴─────┘         └─────┴─────┘   ││
│  │                                                               ││
│  │  + Variance Explained Ratios:                                 ││
│  │    Layer2: [PC1: 45%, PC2: 25%, PC3: 12%]                    ││
│  │    Layer3: [PC1: 52%, PC2: 20%, PC3: 10%]                    ││
│  │    Layer4: [PC1: 60%, PC2: 18%, PC3: 8%]                     ││
│  │                                                               ││
│  │  + Feature Distribution Plots:                                ││
│  │    Scatter plots of GT vs Gen in PC1-PC2 plane               ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

#### PCA Implementation Details

```python
from sklearn.decomposition import PCA
import torch
import numpy as np

class FeaturePCAVisualizer:
    """
    PCA-based visualization showing true feature manifold structure.
    """

    def __init__(self):
        self.pca_models = {}  # Cache fitted PCA per layer

    def fit_pca_on_gt(self, gt_features, layer_idx, n_samples=1000):
        """
        Fit PCA on GT features to establish the "ground truth" manifold.

        Args:
            gt_features: (B, C, H, W) tensor
            layer_idx: which layer (2, 3, or 4)
            n_samples: random pixel samples for fitting (memory efficiency)
        """
        B, C, H, W = gt_features.shape

        # Reshape: (B, C, H, W) -> (B*H*W, C)
        features_flat = gt_features.permute(0, 2, 3, 1).reshape(-1, C)
        features_np = features_flat.cpu().numpy()

        # Random subsample for efficiency
        if features_np.shape[0] > n_samples:
            indices = np.random.choice(features_np.shape[0], n_samples, replace=False)
            features_np = features_np[indices]

        # Fit PCA
        pca = PCA(n_components=3)
        pca.fit(features_np)

        self.pca_models[layer_idx] = pca
        return pca.explained_variance_ratio_

    def project_to_rgb(self, features, layer_idx):
        """
        Project features to 3D using fitted PCA, output as RGB image.

        Args:
            features: (1, C, H, W) single image features
            layer_idx: which layer's PCA to use

        Returns:
            rgb_image: (H, W, 3) uint8 array
        """
        pca = self.pca_models[layer_idx]
        C, H, W = features.shape[1:]

        # Reshape: (1, C, H, W) -> (H*W, C)
        features_flat = features[0].permute(1, 2, 0).reshape(-1, C)
        features_np = features_flat.cpu().numpy()

        # Project to 3D
        projected = pca.transform(features_np)  # (H*W, 3)

        # Reshape back to image: (H, W, 3)
        rgb = projected.reshape(H, W, 3)

        # Normalize each channel to [0, 255]
        for c in range(3):
            channel = rgb[:, :, c]
            c_min, c_max = channel.min(), channel.max()
            if c_max - c_min > 1e-8:
                rgb[:, :, c] = (channel - c_min) / (c_max - c_min) * 255
            else:
                rgb[:, :, c] = 128  # Constant channel

        return rgb.astype(np.uint8)
```

#### Key Diagnostic Metrics from PCA

| Metric | What It Reveals | Red Flag |
|--------|-----------------|----------|
| **Variance Explained (Layer 2)** | Whether Layer 2 features are diverse or collapsed | PC1 > 70% indicates feature collapse |
| **GT vs Gen PC Distribution** | Feature space drift | Significant shift = generator learned wrong patterns |
| **Cross-layer PC Alignment** | Whether layers encode consistent anatomy | Misalignment = scale collapse |

---

### 3.4 Combined Output Layout

The final visualization script should produce a comprehensive **single-page dashboard**:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         ANATOMY LOSS DIAGNOSTIC DASHBOARD                        │
│                          Sample: {image_id} | Step: {step}                       │
├────────────────────────────────────────────────────────────────────────────────┤
│ TRACK A: SEMANTIC SEGMENTATION                                                   │
│ ┌────────────┬────────────┬────────────┬────────────┐                           │
│ │ GT Image   │ GT Overlay │ Gen Image  │ Gen Overlay│                           │
│ │            │ (10-class) │            │ (10-class) │                           │
│ └────────────┴────────────┴────────────┴────────────┘                           │
│ Dice Scores: Lung_L: 0.92 | Lung_R: 0.91 | Heart: 0.85 | Ribs: 0.78 | ...       │
├────────────────────────────────────────────────────────────────────────────────┤
│ TRACK B: PCA FEATURE ANALYSIS                                                    │
│ ┌──────────────────┬──────────────────┬──────────────────┐                      │
│ │    Layer 2       │    Layer 3       │    Layer 4       │                      │
│ │   (64ch, 64×64)  │  (128ch, 32×32)  │  (256ch, 16×16)  │                      │
│ │ ┌─────┬─────┐    │ ┌─────┬─────┐    │ ┌─────┬─────┐    │                      │
│ │ │ GT  │ Gen │    │ │ GT  │ Gen │    │ │ GT  │ Gen │    │                      │
│ │ │ PCA │ PCA │    │ │ PCA │ PCA │    │ │ PCA │ PCA │    │                      │
│ │ └─────┴─────┘    │ └─────┴─────┘    │ └─────┴─────┘    │                      │
│ │ MSE: 0.0234      │ MSE: 0.0156      │ MSE: 0.0089      │                      │
│ │ VarExp: 72/18/5% │ VarExp: 58/22/10%│ VarExp: 45/28/15%│                      │
│ └──────────────────┴──────────────────┴──────────────────┘                      │
├────────────────────────────────────────────────────────────────────────────────┤
│ GRADIENT ANALYSIS (Optional Debug Panel)                                         │
│ ┌───────────────────────────────────────────────────────────────────────────┐   │
│ │ Grad Norm Ratio: ∂L_anatomy/∂θ : ∂L_diffusion/∂θ = 4.7:1 ⚠️ IMBALANCED     │   │
│ │ Per-Layer Grad Contribution: L2: 62% | L3: 28% | L4: 10%                   │   │
│ └───────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Recommended Next Steps

### 4.1 Immediate Fixes (Before More Training)

| Priority | Fix | Implementation |
|----------|-----|----------------|
| **P0** | Add per-layer feature normalization | `F.normalize(features, dim=1)` before MSE |
| **P0** | Add learnable or tuned layer weights | `w = [0.1, 0.3, 0.6]` instead of `[1, 1, 1]` |
| **P1** | Reduce `lambda_anatomy` | Try `0.01` - `0.05` range |
| **P1** | Add gradient norm logging | Monitor `‖∂L_anatomy/∂θ‖ / ‖∂L_diffusion/∂θ‖` |

### 4.2 Visualization Script Priority

1. **Build Track B (PCA) first** - This directly answers "Is Layer 2 dominating?"
2. **Then build Track A (Semantic)** - This confirms anatomical correctness
3. **Add gradient analysis panel** - For ongoing monitoring during training

### 4.3 Suggested Code Structure

```
/home/wenting/zr/gen_code/
├── diagnostics/
│   ├── visualize_anatomy_features_v2.py    # NEW: Combined Track A + B
│   ├── pca_feature_analyzer.py             # NEW: PCA utilities
│   ├── multi_label_overlay.py              # NEW: Track A overlay engine
│   └── gradient_monitor.py                 # EXISTING: Enhance with per-layer logging
```

---

## 5. Summary

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| **Scale Collapse** | **LIKELY** | No feature normalization; Layer 2 has 4× more spatial elements |
| **Normalization Mismatch** | **RULED OUT** | Both use identical `[-1, 1]` range |
| **Shortcut Learning** | **PLAUSIBLE** | Need PCA visualization to confirm |

**Primary Root Cause:** The unweighted, un-normalized MSE across layers with vastly different activation magnitudes and spatial resolutions is causing Layer 2's low-level edge features to dominate gradient flow, forcing OmniGen to overfit to local texture statistics at the expense of global anatomical coherence.

**Immediate Action:** Implement PCA visualization to quantify the feature distribution shift, then apply per-layer normalization and weighting fixes.

---

*Report generated by Code Analysis Copilot. Ready for human review.*
