#!/usr/bin/env python3
"""
Dual-Track Anatomy Visualization Script (v2)

This diagnostic tool analyzes why the anatomy-aware loss caused metric degradation.
It provides two visualization tracks:

Track A (Semantic Level):
    - Full 10-class segmentation overlays for GT and Generated images
    - High-contrast color palette for multi-label mask blending

Track B (Feature Level via PCA):
    - PCA projection of Layer 2, 3, 4 features to RGB images
    - Reveals true feature manifold and potential Layer 2 domination

Track C (Lung-Region Analysis):
    - Calculates MSE inside vs outside lungs
    - Validates hypothesis that loss ignores lung parenchyma

Usage:
    python visualize_anatomy_v2.py \
        --gt_image /path/to/gt_image.png \
        --gen_image /path/to/gen_image.png \
        --seg_ckpt /path/to/best_anatomy_model.pth \
        --output_path ./anatomy_diagnostic_dashboard.png
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

# Add segmentation_models_pytorch to path
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")
import segmentation_models_pytorch as smp


# =============================================================================
# Constants
# =============================================================================

CLASS_NAMES = [
    "Lung_Left", "Lung_Right", "Heart", "Aorta", "Liver",
    "Stomach", "Trachea", "Ribs", "Vertebrae", "Upper_Skeleton"
]

# High-contrast color palette for 10 anatomical classes (RGB 0-255)
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

# Feature layer configurations
FEATURE_INDICES = [2, 3, 4]  # layer1, layer2, layer3 of ResNet34
FEATURE_INFO = {
    2: {"name": "Layer 2", "channels": 64, "spatial": 64, "stride": 4},
    3: {"name": "Layer 3", "channels": 128, "spatial": 32, "stride": 8},
    4: {"name": "Layer 4", "channels": 256, "spatial": 16, "stride": 16},
}


# =============================================================================
# Model Loading
# =============================================================================

def load_frozen_seg_model(checkpoint_path, device="cuda"):
    """Load the frozen ResNet34-UNet segmentation model."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    print(f"[INFO] Loaded segmentation model from: {checkpoint_path}")
    if "val_dice" in ckpt:
        print(f"[INFO] Model validation Dice: {ckpt['val_dice']:.4f}")

    return model


# =============================================================================
# Image Loading
# =============================================================================

def load_image_as_tensor(image_path, device="cuda"):
    """
    Load image and convert to [-1, 1] range tensor (256x256).
    Matches the normalization used during segmentation model training.
    """
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)

    # Resize if needed
    if img.shape[0] != 256 or img.shape[1] != 256:
        img = np.array(Image.fromarray(img).resize((256, 256), Image.BICUBIC))

    # Normalize to [-1, 1]
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
    tensor = tensor.unsqueeze(0).to(device)

    return tensor, img


def tensor_to_numpy_image(tensor):
    """Convert [-1, 1] tensor to [0, 255] numpy image."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return img


# =============================================================================
# Track A: Semantic Overlay
# =============================================================================

def blend_multi_label_masks(image, prob_maps, palette, alpha=0.5, threshold=0.5):
    """
    Blend multiple overlapping binary masks onto an image.

    Args:
        image: (H, W, 3) RGB image, uint8
        prob_maps: (10, H, W) probability maps, numpy array
        palette: dict mapping class_name -> RGB tuple
        alpha: base transparency
        threshold: probability threshold for mask

    Returns:
        blended: (H, W, 3) uint8 array
    """
    H, W = image.shape[:2]
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlap_count = np.zeros((H, W), dtype=np.float32)

    for idx, class_name in enumerate(CLASS_NAMES):
        color = palette[class_name]
        mask = (prob_maps[idx] > threshold).astype(np.float32)

        for c in range(3):
            overlay[:, :, c] += mask * color[c]
        overlap_count += mask

    # Normalize by overlap count to prevent over-saturation
    overlap_count = np.maximum(overlap_count, 1.0)
    overlay = overlay / overlap_count[:, :, np.newaxis]

    # Alpha blend
    image_float = image.astype(np.float32)
    mask_any = (overlap_count > 0).astype(np.float32)[:, :, np.newaxis]

    blended = image_float * (1 - alpha * mask_any) + overlay * alpha * mask_any
    blended = np.where(mask_any > 0, blended, image_float)

    return np.clip(blended, 0, 255).astype(np.uint8)


def get_segmentation_probs(model, image_tensor):
    """
    Run full segmentation model and return probability maps.

    Args:
        model: ResUNet34 model
        image_tensor: (1, 3, H, W) tensor in [-1, 1]

    Returns:
        prob_maps: (10, H, W) numpy array of probabilities
    """
    with torch.no_grad():
        logits = model(image_tensor)  # (1, 10, H, W)
        probs = torch.sigmoid(logits)  # (1, 10, H, W)

    return probs.squeeze(0).cpu().numpy()


# =============================================================================
# Track B: PCA Feature Visualization
# =============================================================================

class FeaturePCAVisualizer:
    """
    PCA-based visualization showing true feature manifold structure.
    """

    def __init__(self):
        self.pca_models = {}
        self.variance_ratios = {}

    def fit_pca_on_gt(self, gt_features, layer_idx, n_samples=2000):
        """
        Fit PCA on GT features to establish the "ground truth" manifold.

        Args:
            gt_features: (B, C, H, W) tensor
            layer_idx: which layer (2, 3, or 4)
            n_samples: random pixel samples for fitting

        Returns:
            variance_ratio: explained variance ratio for 3 components
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
        self.variance_ratios[layer_idx] = pca.explained_variance_ratio_

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
        if layer_idx not in self.pca_models:
            raise ValueError(f"PCA not fitted for layer {layer_idx}")

        pca = self.pca_models[layer_idx]
        _, C, H, W = features.shape

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


def extract_encoder_features(model, image_tensor):
    """
    Extract intermediate features from ResNet34 encoder.

    Args:
        model: ResUNet34 model
        image_tensor: (1, 3, H, W) tensor in [-1, 1]

    Returns:
        features: list of feature tensors [f0, f1, f2, f3, f4, f5]
    """
    with torch.no_grad():
        features = model.encoder(image_tensor)
    return features


# =============================================================================
# Track C: Lung-Region MSE Analysis
# =============================================================================

def compute_lung_region_mse(gt_features, gen_features, lung_mask, layer_idx):
    """
    Compute MSE inside and outside lung regions.

    Args:
        gt_features: (1, C, H, W) GT feature tensor
        gen_features: (1, C, H, W) Gen feature tensor
        lung_mask: (1, 1, 256, 256) binary lung mask
        layer_idx: layer index for spatial size lookup

    Returns:
        dict with mse_inside, mse_outside, ratio
    """
    C, H, W = gt_features.shape[1:]

    # Resize lung mask to match feature spatial size
    mask_resized = F.interpolate(
        lung_mask.float(),
        size=(H, W),
        mode="nearest"
    ).squeeze(0).squeeze(0)  # (H, W)

    # Compute squared error per pixel
    sq_error = (gt_features - gen_features).pow(2)  # (1, C, H, W)
    sq_error = sq_error.squeeze(0)  # (C, H, W)

    # Mean over channels -> (H, W)
    sq_error_spatial = sq_error.mean(dim=0)

    # Create masks
    mask_inside = mask_resized > 0.5
    mask_outside = ~mask_inside

    # Compute MSE for each region
    n_inside = mask_inside.sum().item()
    n_outside = mask_outside.sum().item()

    if n_inside > 0:
        mse_inside = sq_error_spatial[mask_inside].mean().item()
    else:
        mse_inside = 0.0

    if n_outside > 0:
        mse_outside = sq_error_spatial[mask_outside].mean().item()
    else:
        mse_outside = 0.0

    # Compute ratio (outside/inside) - higher means edges dominating
    ratio = mse_outside / (mse_inside + 1e-8) if mse_inside > 1e-8 else float('inf')

    return {
        "mse_inside": mse_inside,
        "mse_outside": mse_outside,
        "ratio_out_in": ratio,
        "n_pixels_inside": n_inside,
        "n_pixels_outside": n_outside,
    }


def create_lung_mask_from_probs(prob_maps, threshold=0.5):
    """
    Create combined lung mask from Lung_Left (idx 0) and Lung_Right (idx 1).

    Args:
        prob_maps: (10, H, W) probability maps

    Returns:
        lung_mask: (1, 1, H, W) tensor
    """
    lung_left = prob_maps[0] > threshold
    lung_right = prob_maps[1] > threshold
    combined = (lung_left | lung_right).astype(np.float32)

    return torch.from_numpy(combined).unsqueeze(0).unsqueeze(0)


# =============================================================================
# Main Visualization Dashboard
# =============================================================================

def create_dashboard(
    gt_image_np,
    gen_image_np,
    gt_probs,
    gen_probs,
    gt_features,
    gen_features,
    pca_visualizer,
    lung_mse_results,
    output_path,
):
    """
    Create comprehensive diagnostic dashboard.
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=(24, 18), dpi=150)
    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.25)

    # Title
    fig.suptitle(
        "ANATOMY LOSS DIAGNOSTIC DASHBOARD",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )

    # =========================================================================
    # Row 1: Track A - Semantic Overlays
    # =========================================================================
    ax_title_a = fig.add_subplot(gs[0, :])
    ax_title_a.axis("off")
    ax_title_a.text(
        0.5, 0.5,
        "TRACK A: SEMANTIC SEGMENTATION OVERLAYS",
        fontsize=14, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8)
    )

    # GT Original
    ax_gt_orig = fig.add_subplot(gs[1, 0])
    ax_gt_orig.imshow(gt_image_np)
    ax_gt_orig.set_title("GT Original", fontsize=11)
    ax_gt_orig.axis("off")

    # GT Overlay
    ax_gt_overlay = fig.add_subplot(gs[1, 1])
    gt_overlay = blend_multi_label_masks(gt_image_np, gt_probs, ANATOMY_PALETTE, alpha=0.6)
    ax_gt_overlay.imshow(gt_overlay)
    ax_gt_overlay.set_title("GT + Segmentation", fontsize=11)
    ax_gt_overlay.axis("off")

    # Gen Original
    ax_gen_orig = fig.add_subplot(gs[1, 2])
    ax_gen_orig.imshow(gen_image_np)
    ax_gen_orig.set_title("Generated Original", fontsize=11)
    ax_gen_orig.axis("off")

    # Gen Overlay
    ax_gen_overlay = fig.add_subplot(gs[1, 3])
    gen_overlay = blend_multi_label_masks(gen_image_np, gen_probs, ANATOMY_PALETTE, alpha=0.6)
    ax_gen_overlay.imshow(gen_overlay)
    ax_gen_overlay.set_title("Generated + Segmentation", fontsize=11)
    ax_gen_overlay.axis("off")

    # Color Legend
    ax_legend = fig.add_subplot(gs[1, 4:])
    ax_legend.axis("off")
    legend_text = "CLASS LEGEND:\n"
    for i, (name, color) in enumerate(ANATOMY_PALETTE.items()):
        r, g, b = [c/255 for c in color]
        ax_legend.add_patch(plt.Rectangle((0.05, 0.9 - i*0.09), 0.08, 0.07,
                                           facecolor=(r, g, b), edgecolor='black'))
        ax_legend.text(0.16, 0.93 - i*0.09, name, fontsize=9, va='center')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    # =========================================================================
    # Row 2: Track B - PCA Feature Visualization
    # =========================================================================
    ax_title_b = fig.add_subplot(gs[2, :])
    ax_title_b.axis("off")
    ax_title_b.text(
        0.5, 0.5,
        "TRACK B: PCA FEATURE VISUALIZATION (3 Principal Components → RGB)",
        fontsize=14, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8)
    )

    # Plot PCA visualizations for each layer
    for col, layer_idx in enumerate(FEATURE_INDICES):
        info = FEATURE_INFO[layer_idx]

        # GT PCA
        ax_gt_pca = fig.add_subplot(gs[3, col * 2])
        # gt_features[layer_idx] is already (1, C, H, W) - no unsqueeze needed
        gt_feat_4d = gt_features[layer_idx]
        if gt_feat_4d.dim() == 3:
            gt_feat_4d = gt_feat_4d.unsqueeze(0)
        gt_pca_rgb = pca_visualizer.project_to_rgb(gt_feat_4d, layer_idx)
        # Resize for better visibility
        gt_pca_rgb_resized = np.array(Image.fromarray(gt_pca_rgb).resize((128, 128), Image.NEAREST))
        ax_gt_pca.imshow(gt_pca_rgb_resized)
        var_ratio = pca_visualizer.variance_ratios[layer_idx]
        ax_gt_pca.set_title(
            f"GT {info['name']}\n({info['channels']}ch, {info['spatial']}×{info['spatial']})\n"
            f"VarExp: {var_ratio[0]*100:.0f}/{var_ratio[1]*100:.0f}/{var_ratio[2]*100:.0f}%",
            fontsize=9
        )
        ax_gt_pca.axis("off")

        # Gen PCA
        ax_gen_pca = fig.add_subplot(gs[3, col * 2 + 1])
        # gen_features[layer_idx] is already (1, C, H, W) - no unsqueeze needed
        gen_feat_4d = gen_features[layer_idx]
        if gen_feat_4d.dim() == 3:
            gen_feat_4d = gen_feat_4d.unsqueeze(0)
        gen_pca_rgb = pca_visualizer.project_to_rgb(gen_feat_4d, layer_idx)
        gen_pca_rgb_resized = np.array(Image.fromarray(gen_pca_rgb).resize((128, 128), Image.NEAREST))
        ax_gen_pca.imshow(gen_pca_rgb_resized)

        # Get MSE for this layer
        mse_result = lung_mse_results[layer_idx]
        ax_gen_pca.set_title(
            f"Gen {info['name']}\n"
            f"MSE_in: {mse_result['mse_inside']:.4f}\n"
            f"MSE_out: {mse_result['mse_outside']:.4f}",
            fontsize=9
        )
        ax_gen_pca.axis("off")

    # =========================================================================
    # Add Lung MSE Analysis Summary Text
    # =========================================================================
    summary_text = "LUNG-REGION MSE ANALYSIS (Inside Lungs vs Outside):\n"
    summary_text += "=" * 60 + "\n"

    for layer_idx in FEATURE_INDICES:
        info = FEATURE_INFO[layer_idx]
        result = lung_mse_results[layer_idx]
        summary_text += (
            f"{info['name']:10s}: "
            f"MSE_inside={result['mse_inside']:.6f}, "
            f"MSE_outside={result['mse_outside']:.6f}, "
            f"Ratio(out/in)={result['ratio_out_in']:.2f}\n"
        )

    summary_text += "=" * 60 + "\n"
    summary_text += "If Ratio > 1: Loss focuses MORE on edges/ribs than lungs (PROBLEM!)\n"
    summary_text += "If Ratio < 1: Loss focuses MORE on lung parenchyma (DESIRED!)\n"

    # Add text box at bottom
    fig.text(
        0.5, 0.02, summary_text,
        fontsize=10, family="monospace",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9)
    )

    # Save
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\n[SUCCESS] Dashboard saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dual-Track Anatomy Visualization for Loss Debugging"
    )
    parser.add_argument(
        "--gt_image", type=str, required=True,
        help="Path to ground truth image"
    )
    parser.add_argument(
        "--gen_image", type=str, required=True,
        help="Path to generated image"
    )
    parser.add_argument(
        "--seg_ckpt", type=str, required=True,
        help="Path to frozen ResUNet34 checkpoint"
    )
    parser.add_argument(
        "--output_path", type=str, default="./anatomy_diagnostic_dashboard.png",
        help="Path to save the output dashboard"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 70)
    print("DUAL-TRACK ANATOMY VISUALIZATION DIAGNOSTIC")
    print("=" * 70)

    # =========================================================================
    # Load Model
    # =========================================================================
    print("\n[1/5] Loading segmentation model...")
    model = load_frozen_seg_model(args.seg_ckpt, device=args.device)

    # =========================================================================
    # Load Images
    # =========================================================================
    print("\n[2/5] Loading images...")
    gt_tensor, gt_image_np = load_image_as_tensor(args.gt_image, device=args.device)
    gen_tensor, gen_image_np = load_image_as_tensor(args.gen_image, device=args.device)
    print(f"  GT image: {args.gt_image}")
    print(f"  Gen image: {args.gen_image}")

    # =========================================================================
    # Track A: Get Segmentation Probabilities
    # =========================================================================
    print("\n[3/5] Running Track A: Semantic Segmentation...")
    gt_probs = get_segmentation_probs(model, gt_tensor)  # (10, 256, 256)
    gen_probs = get_segmentation_probs(model, gen_tensor)

    # Create lung mask from GT probabilities
    lung_mask = create_lung_mask_from_probs(gt_probs).to(args.device)
    print(f"  Lung mask coverage: {lung_mask.sum().item() / (256*256) * 100:.1f}% of image")

    # =========================================================================
    # Track B: Extract Features and Fit PCA
    # =========================================================================
    print("\n[4/5] Running Track B: PCA Feature Analysis...")
    gt_features = extract_encoder_features(model, gt_tensor)
    gen_features = extract_encoder_features(model, gen_tensor)

    pca_visualizer = FeaturePCAVisualizer()
    for layer_idx in FEATURE_INDICES:
        gt_feat = gt_features[layer_idx]
        var_ratio = pca_visualizer.fit_pca_on_gt(gt_feat, layer_idx)
        print(f"  {FEATURE_INFO[layer_idx]['name']}: "
              f"Variance Explained = [{var_ratio[0]*100:.1f}%, {var_ratio[1]*100:.1f}%, {var_ratio[2]*100:.1f}%]")

    # =========================================================================
    # Track C: Lung-Region MSE Analysis
    # =========================================================================
    print("\n[5/5] Computing Lung-Region MSE Analysis...")
    lung_mse_results = {}

    for layer_idx in FEATURE_INDICES:
        gt_feat = gt_features[layer_idx]
        gen_feat = gen_features[layer_idx]

        result = compute_lung_region_mse(gt_feat, gen_feat, lung_mask, layer_idx)
        lung_mse_results[layer_idx] = result

        print(f"  {FEATURE_INFO[layer_idx]['name']}: "
              f"MSE_inside={result['mse_inside']:.6f}, "
              f"MSE_outside={result['mse_outside']:.6f}, "
              f"Ratio(out/in)={result['ratio_out_in']:.2f}")

    # =========================================================================
    # Create Dashboard
    # =========================================================================
    print("\n[6/6] Creating visualization dashboard...")

    # Need to move features to CPU for PCA projection
    gt_features_cpu = [f.cpu() for f in gt_features]
    gen_features_cpu = [f.cpu() for f in gen_features]

    create_dashboard(
        gt_image_np=gt_image_np,
        gen_image_np=gen_image_np,
        gt_probs=gt_probs,
        gen_probs=gen_probs,
        gt_features=gt_features_cpu,
        gen_features=gen_features_cpu,
        pca_visualizer=pca_visualizer,
        lung_mse_results=lung_mse_results,
        output_path=args.output_path,
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
