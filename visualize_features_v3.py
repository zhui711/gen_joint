#!/usr/bin/env python3
"""
Visualize Mask-Weighted Feature Maps for Anatomy-Aware Loss Analysis (V3).

This script visualizes what the segmentation model's encoder "sees" when
processing GT and generated CXR images. It rigorously replicates the logic 
of loss_anatomy_v3.py:
  - External linear weighting: Loss = M_gt * (F_gen - F_gt)^2
  - Area normalization: Loss / (mask_area * C_feat)
  - use_gen_mask=False defaults (uses GT mask for both features)

Usage:
    # Basic (raw features only):
    python visualize_features.py \
        --gt_image /path/to/gt.png \
        --gen_image /path/to/gen.png \
        --seg_model_ckpt /path/to/best_anatomy_model.pth

    # With mask-weighted organ visualization (V3 logic):
    python visualize_features.py \
        --gt_image /path/to/gt.png \
        --gen_image /path/to/gen.png \
        --mask_gt /path/to/mask.npz \
        --seg_model_ckpt /path/to/best_anatomy_model.pth \
        --organ_channels 0 1 2 7

Layer Selection (ResNet34 encoder, 256x256 input):
    Feature 2: (B, 64, 64, 64)   stride=4  - 1/4 resolution (DEFAULT)
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add segmentation library to path
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")
import segmentation_models_pytorch as smp


# Anatomical channel names
ANATOMY_CHANNELS =[
    "Lung_Left",      # 0
    "Lung_Right",     # 1
    "Heart",          # 2
    "Aorta",          # 3
    "Liver",          # 4
    "Stomach",        # 5
    "Trachea",        # 6
    "Ribs",           # 7
    "Vertebrae",      # 8
    "Upper_Skeleton", # 9
]


def get_anatomy_model():
    """Create UNet with ResNet34 encoder (weights loaded from checkpoint)."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )


def load_seg_model(ckpt_path, device):
    """Load frozen segmentation model for inference."""
    model = get_anatomy_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def load_and_preprocess_image(image_path, target_size=256):
    """
    Load image and preprocess for the segmentation model.
    """
    img = Image.open(image_path).convert("RGB")
    original_np = np.array(img)

    img = img.resize((target_size, target_size), Image.BILINEAR)
    img_np = np.array(img)

    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    tensor = tensor / 127.5 - 1.0  # [0, 255] ->[-1, 1]
    tensor = tensor.unsqueeze(0)

    original_resized = cv2.resize(original_np, (target_size, target_size))
    return tensor, original_resized


def feature_to_heatmap(feature_2d_np, colormap=cv2.COLORMAP_JET):
    """
    Convert a 2D numpy feature map to a colormap heatmap (H, W, 3) uint8 RGB.
    """
    feat_min, feat_max = feature_2d_np.min(), feature_2d_np.max()
    if feat_max - feat_min > 1e-8:
        norm = (feature_2d_np - feat_min) / (feat_max - feat_min)
    else:
        norm = np.zeros_like(feature_2d_np)

    uint8 = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(uint8, colormap)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay_heatmap(original_img, heatmap, alpha=0.5):
    """Overlay heatmap on original image with alpha blending."""
    if heatmap.shape[:2] != original_img.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    blended = (1 - alpha) * original_img + alpha * heatmap
    return np.clip(blended, 0, 255).astype(np.uint8)


def extract_all(model, image_tensor, device):
    """
    Extract encoder features AND predicted mask from the model.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = model.encoder(image_tensor)
        decoder_output = model.decoder(features)
        logits = model.segmentation_head(decoder_output)
        mask_pred = torch.sigmoid(logits)
    return features, mask_pred


# =============================================================================
# Visualization: Raw Feature Heatmaps (Global view)
# =============================================================================

def create_summary_figure(
    gt_original, gen_original,
    gt_features, gen_features,
    output_path
):
    """
    Summary figure: GT vs Gen raw features at layers 2, 3, 4 + difference.
    """
    layer_info =[
        (2, "Layer2\n(1/4 res)"),
        (3, "Layer3\n(1/8 res)"),
        (4, "Layer4\n(1/16 res)"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 0: GT
    axes[0, 0].imshow(gt_original)
    axes[0, 0].set_title("GT Original", fontsize=10)
    axes[0, 0].axis("off")

    for col, (idx, name) in enumerate(layer_info):
        feat_avg = torch.mean(gt_features[idx], dim=1).squeeze().cpu().numpy()
        heatmap = feature_to_heatmap(feat_avg)
        axes[0, col + 1].imshow(overlay_heatmap(gt_original, heatmap))
        axes[0, col + 1].set_title(f"GT {name}", fontsize=9)
        axes[0, col + 1].axis("off")

    # Row 1: Gen
    axes[1, 0].imshow(gen_original)
    axes[1, 0].set_title("Gen Original", fontsize=10)
    axes[1, 0].axis("off")

    for col, (idx, name) in enumerate(layer_info):
        feat_avg = torch.mean(gen_features[idx], dim=1).squeeze().cpu().numpy()
        heatmap = feature_to_heatmap(feat_avg)
        axes[1, col + 1].imshow(overlay_heatmap(gen_original, heatmap))
        axes[1, col + 1].set_title(f"Gen {name}", fontsize=9)
        axes[1, col + 1].axis("off")

    # Row 2: Difference
    axes[2, 0].text(0.5, 0.5, "Feature\nDifference\n|GT - Gen|",
                    ha="center", va="center", fontsize=10, fontweight="bold")
    axes[2, 0].axis("off")

    for col, (idx, name) in enumerate(layer_info):
        diff = torch.abs(gt_features[idx] - gen_features[idx])
        diff_avg = torch.mean(diff, dim=1).squeeze().cpu().numpy()
        heatmap = feature_to_heatmap(diff_avg, colormap=cv2.COLORMAP_HOT)
        heatmap = cv2.resize(heatmap, (256, 256))
        mse_val = ((gt_features[idx] - gen_features[idx]) ** 2).mean().item()
        axes[2, col + 1].imshow(heatmap)
        axes[2, col + 1].set_title(f"Diff {name}\nMSE: {mse_val:.4f}", fontsize=9)
        axes[2, col + 1].axis("off")

    plt.suptitle("Feature Matching Analysis (Raw Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Visualization: Mask-Weighted Organ-Specific Feature Heatmaps (V3)
# =============================================================================

def create_organ_feature_figure(
    gt_original, gen_original,
    gt_features, gen_features,
    mask_gt, mask_gen,
    feature_layer_idx,
    organ_channels,
    use_gen_mask,
    output_path,
):
    """
    Per-organ mask-weighted feature visualization using V3 logic.
    """
    F_gt = gt_features[feature_layer_idx]   # (1, C, H, W)
    F_gen = gen_features[feature_layer_idx]  # (1, C, H, W)
    _, C_feat, H_feat, W_feat = F_gt.shape

    # Downsample masks to feature resolution (using avg_pool2d as in loss)
    input_H = mask_gt.shape[-2]
    if H_feat != input_H:
        kernel_size = input_H // H_feat
        mask_gt_down = F.avg_pool2d(mask_gt, kernel_size=kernel_size)
        mask_gen_down = F.avg_pool2d(mask_gen, kernel_size=kernel_size)
    else:
        mask_gt_down = mask_gt
        mask_gen_down = mask_gen

    n_organs = len(organ_channels)
    fig, axes = plt.subplots(n_organs, 6, figsize=(24, 4 * n_organs))
    if n_organs == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    for row, c in enumerate(organ_channels):
        organ_name = ANATOMY_CHANNELS[c]

        M_gt_c = mask_gt_down[:, c:c+1, :, :]
        M_gen_c = mask_gen_down[:, c:c+1, :, :]

        # --- V3 Logic Calculation ---
        if use_gen_mask:
            OrganFeat_gt = M_gt_c * F_gt
            OrganFeat_gen = M_gen_c * F_gen
            squared_diff = (OrganFeat_gen - OrganFeat_gt) ** 2
        else:
            # STRICT V3: Gen features are masked by GT mask!
            OrganFeat_gt = M_gt_c * F_gt
            OrganFeat_gen = M_gt_c * F_gen
            squared_diff = M_gt_c * ((F_gen - F_gt) ** 2)

        # Calculate V3 Area-Normalized MSE
        mask_area = M_gt_c.sum().item()
        sum_sq_diff = squared_diff.sum().item()
        denominator = max(mask_area * C_feat, 1.0)
        mse_c_v3 = sum_sq_diff / denominator

        # Channel-average to 2D spatial heatmap for display
        organ_gt_2d = torch.mean(OrganFeat_gt, dim=1).squeeze().cpu().numpy()
        organ_gen_2d = torch.mean(OrganFeat_gen, dim=1).squeeze().cpu().numpy()

        # Mask at original resolution for display
        mask_gt_orig_c = mask_gt[0, c].cpu().numpy()
        mask_gen_orig_c = mask_gen[0, c].cpu().numpy()

        # Col 0: GT original with mask overlay
        mask_overlay_gt = overlay_heatmap(
            gt_original,
            feature_to_heatmap(mask_gt_orig_c, cv2.COLORMAP_BONE),
            alpha=0.4
        )
        axes[row, 0].imshow(mask_overlay_gt)
        axes[row, 0].set_title(f"GT Mask: {organ_name}", fontsize=9)
        axes[row, 0].axis("off")

        # Col 1: GT organ feature heatmap
        gt_heatmap = feature_to_heatmap(organ_gt_2d)
        gt_heatmap_resized = cv2.resize(gt_heatmap, (256, 256))
        axes[row, 1].imshow(overlay_heatmap(gt_original, gt_heatmap_resized, alpha=0.6))
        axes[row, 1].set_title(f"GT OrganFeat", fontsize=9)
        axes[row, 1].axis("off")

        # Col 2: Gen original with GEN mask overlay (Diagnostic only if use_gen_mask=False)
        mask_overlay_gen = overlay_heatmap(
            gen_original,
            feature_to_heatmap(mask_gen_orig_c, cv2.COLORMAP_BONE),
            alpha=0.4
        )
        axes[row, 2].imshow(mask_overlay_gen)
        title_gen_mask = f"Gen Mask (Pred): {organ_name}" if use_gen_mask else f"Gen Mask (Diagnostic Only)"
        axes[row, 2].set_title(title_gen_mask, fontsize=9)
        axes[row, 2].axis("off")

        # Col 3: Gen organ feature heatmap
        gen_heatmap = feature_to_heatmap(organ_gen_2d)
        gen_heatmap_resized = cv2.resize(gen_heatmap, (256, 256))
        axes[row, 3].imshow(overlay_heatmap(gen_original, gen_heatmap_resized, alpha=0.6))
        title_gen_feat = "Gen OrganFeat" if use_gen_mask else "Gen OrganFeat (Masked by GT)"
        axes[row, 3].set_title(title_gen_feat, fontsize=9)
        axes[row, 3].axis("off")

        # Col 4: Difference heatmap (Masked)
        diff_2d_viz = np.abs(organ_gt_2d - organ_gen_2d)
        diff_heatmap = feature_to_heatmap(diff_2d_viz, cv2.COLORMAP_HOT)
        diff_heatmap_resized = cv2.resize(diff_heatmap, (256, 256))
        axes[row, 4].imshow(diff_heatmap_resized)
        axes[row, 4].set_title(f"|Diff| V3_MSE={mse_c_v3:.5f}", fontsize=10, fontweight="bold")
        axes[row, 4].axis("off")

        # Col 5: Side-by-side raw feature comparison (no mask)
        raw_gt_2d = torch.mean(F_gt, dim=1).squeeze().cpu().numpy()
        raw_gen_2d = torch.mean(F_gen, dim=1).squeeze().cpu().numpy()
        raw_diff = np.abs(raw_gt_2d - raw_gen_2d)
        raw_heatmap = feature_to_heatmap(raw_diff, cv2.COLORMAP_JET)
        raw_heatmap_resized = cv2.resize(raw_heatmap, (256, 256))
        axes[row, 5].imshow(raw_heatmap_resized)
        raw_mse = ((F_gt - F_gen) ** 2).mean().item()
        axes[row, 5].set_title(f"Raw Diff (Global)\nMSE={raw_mse:.5f}", fontsize=9)
        axes[row, 5].axis("off")

    mode_str = "use_gen_mask=True" if use_gen_mask else "use_gen_mask=False (GT Anchored)"
    layer_name = f"Layer{feature_layer_idx} (1/{2**feature_layer_idx} res)"
    plt.suptitle(
        f"Mask-Weighted Feature Analysis (V3) — {layer_name} | {mode_str}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def create_loss_summary_table(
    gt_features, gen_features,
    mask_gt, mask_gen,
    feature_layer_idx,
    use_gen_mask
):
    """Print per-organ V3 Area-Normalized MSE table."""
    F_gt = gt_features[feature_layer_idx]
    F_gen = gen_features[feature_layer_idx]
    _, C_feat, H_feat, W_feat = F_gt.shape

    input_H = mask_gt.shape[-2]
    if H_feat != input_H:
        kernel_size = input_H // H_feat
        mask_gt_down = F.avg_pool2d(mask_gt, kernel_size=kernel_size)
        mask_gen_down = F.avg_pool2d(mask_gen, kernel_size=kernel_size)
    else:
        mask_gt_down = mask_gt
        mask_gen_down = mask_gen

    print("\n" + "=" * 70)
    print(f"Per-Organ V3 MSE (Layer {feature_layer_idx}, {H_feat}x{W_feat} resolution)")
    print(f"Settings: Area-Normalized, use_gen_mask={use_gen_mask}")
    print("=" * 70)

    total_loss = 0.0
    for c in range(10):
        M_gt_c = mask_gt_down[:, c:c+1, :, :]
        M_gen_c = mask_gen_down[:, c:c+1, :, :]

        if use_gen_mask:
            OrganFeat_gt = M_gt_c * F_gt
            OrganFeat_gen = M_gen_c * F_gen
            squared_diff = (OrganFeat_gen - OrganFeat_gt) ** 2
        else:
            squared_diff = M_gt_c * ((F_gen - F_gt) ** 2)

        mask_area = M_gt_c.sum().item()
        sum_sq_diff = squared_diff.sum().item()
        denominator = max(mask_area * C_feat, 1.0)
        mse_c_v3 = sum_sq_diff / denominator
        
        mask_coverage = M_gt_c.mean().item() * 100
        total_loss += mse_c_v3

        print(f"  Ch {c:2d}[{ANATOMY_CHANNELS[c]:>15s}]  Loss={mse_c_v3:.6f}  "
              f"(Area: {mask_area:.1f} px | Coverage: {mask_coverage:.1f}%)")

    print("-" * 70)
    print(f"  Total Anatomy Loss (sum): {total_loss:.6f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mask-weighted encoder features for anatomy-aware loss (V3)."
    )
    parser.add_argument("--gt_image", type=str, required=True,
                        help="Path to ground truth image.")
    parser.add_argument("--gen_image", type=str, required=True,
                        help="Path to generated image.")
    parser.add_argument("--mask_gt", type=str, default=None,
                        help="Path to GT mask .npz file (key='mask', shape 10x256x256).")
    parser.add_argument("--seg_model_ckpt", type=str,
                        default="/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth",
                        help="Path to frozen segmentation model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./feature_vis_output",
                        help="Directory to save output visualizations.")
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--feature_layer_idx", type=int, default=2,
                        help="Encoder layer index (2=1/4 res, 3=1/8, 4=1/16).")
    parser.add_argument("--organ_channels", type=int, nargs="+",
                        default=[0, 1, 2, 3, 7],
                        help="Organ channel indices to visualize (default: Lungs, Heart, Aorta, Ribs).")
    parser.add_argument("--use_gen_mask", action="store_true",
                        help="If passed, replicates use_gen_mask=True behavior. "
                             "By default (V3), uses strictly GT mask for stability.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading segmentation model from {args.seg_model_ckpt}...")
    model = load_seg_model(args.seg_model_ckpt, args.device)

    print(f"Loading GT:  {args.gt_image}")
    gt_tensor, gt_original = load_and_preprocess_image(args.gt_image, args.target_size)
    print(f"Loading Gen: {args.gen_image}")
    gen_tensor, gen_original = load_and_preprocess_image(args.gen_image, args.target_size)

    print("Extracting encoder features...")
    gt_features, mask_pred_gt = extract_all(model, gt_tensor, args.device)
    gen_features, mask_pred_gen = extract_all(model, gen_tensor, args.device)

    print("\nGenerating raw feature summary...")
    create_summary_figure(
        gt_original, gen_original,
        gt_features, gen_features,
        os.path.join(args.output_dir, "summary_raw_features.png")
    )

    if args.mask_gt is not None:
        print(f"\nLoading GT mask: {args.mask_gt}")
        mask_data = np.load(args.mask_gt)
        mask_gt_np = mask_data["mask"].astype(np.float32)
        mask_gt_tensor = torch.from_numpy(mask_gt_np).unsqueeze(0).to(args.device)

        mask_gen_tensor = mask_pred_gen

        valid_channels =[c for c in args.organ_channels if 0 <= c < 10]
        if not valid_channels:
            valid_channels = list(range(10))

        create_organ_feature_figure(
            gt_original, gen_original,
            gt_features, gen_features,
            mask_gt_tensor, mask_gen_tensor,
            feature_layer_idx=args.feature_layer_idx,
            organ_channels=valid_channels,
            use_gen_mask=args.use_gen_mask,
            output_path=os.path.join(args.output_dir, "organ_feature_analysis_v3.png"),
        )

        create_loss_summary_table(
            gt_features, gen_features,
            mask_gt_tensor, mask_gen_tensor,
            feature_layer_idx=args.feature_layer_idx,
            use_gen_mask=args.use_gen_mask
        )
    else:
        print("\nNo --mask_gt provided. Skipping mask-weighted organ visualization.")

    print(f"\nDone! Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()