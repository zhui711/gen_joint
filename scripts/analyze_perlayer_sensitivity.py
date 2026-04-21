"""
Quick Per-Layer Feature Sensitivity Analysis.

This script analyzes per-layer MSE sensitivity WITHOUT loading OmniGen.
It compares GT image features with perturbed versions to understand
which encoder layer is most sensitive to different types of distortions.

This helps explain the training-time per-layer loss distribution.

Usage:
    python scripts/analyze_perlayer_sensitivity.py --num_samples 100
"""

import sys
import os
import argparse
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add paths
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")
import segmentation_models_pytorch as smp


def get_anatomy_model():
    """Create UNet with ResNet34 encoder."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )


def load_seg_model(ckpt_path, device):
    """Load frozen segmentation model."""
    model = get_anatomy_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def compute_perlayer_mse(seg_model, img1, img2):
    """Compute per-layer MSE between two images' features."""
    FEATURE_INDICES = [2, 3, 4]

    with torch.no_grad():
        feat1 = seg_model.encoder(img1)
        feat2 = seg_model.encoder(img2)

    per_layer_mse = {}
    per_layer_info = {}

    for idx in FEATURE_INDICES:
        f1 = feat1[idx]
        f2 = feat2[idx]

        mse = F.mse_loss(f1, f2, reduction='mean').item()
        per_layer_mse[f"layer_{idx}"] = mse

        # Feature statistics
        per_layer_info[f"layer_{idx}"] = {
            "mse": mse,
            "shape": list(f1.shape),
            "channels": f1.shape[1],
            "spatial": f"{f1.shape[2]}x{f1.shape[3]}",
            "numel": f1.numel(),
            "feat_mean_abs": f1.abs().mean().item(),
            "feat_std": f1.std().item(),
        }

    return per_layer_mse, per_layer_info


def add_gaussian_noise(img, std):
    """Add Gaussian noise to image."""
    noise = torch.randn_like(img) * std
    return (img + noise).clamp(-1, 1)


def add_blur(img, kernel_size=5):
    """Add Gaussian blur to image."""
    # Simple box blur approximation
    padding = kernel_size // 2
    return F.avg_pool2d(img, kernel_size, stride=1, padding=padding)


def random_shift(img, max_shift=10):
    """Randomly shift image pixels."""
    shift_x = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    shift_y = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    return torch.roll(img, shifts=(shift_y, shift_x), dims=(2, 3))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load segmentation model
    print(f"Loading segmentation model from {args.seg_model_ckpt}...")
    seg_model = load_seg_model(args.seg_model_ckpt, device)

    # Print encoder layer info
    print("\n## Encoder Layer Information (ResNet34)")
    print("-" * 60)
    dummy = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        features = seg_model.encoder(dummy)
    for i, f in enumerate(features):
        print(f"Layer {i}: shape={list(f.shape)}, numel={f.numel()}")
    print("-" * 60)

    # Image transform
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load training data (handle both JSON and JSONL formats)
    print(f"\nLoading training data from {args.json_file}...")
    data = []
    with open(args.json_file, 'r') as f:
        if args.json_file.endswith('.jsonl'):
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        else:
            data = json.load(f)

    # Results storage
    results = {
        "identical": {f"layer_{i}": [] for i in [2, 3, 4]},
        "noise_0.01": {f"layer_{i}": [] for i in [2, 3, 4]},
        "noise_0.05": {f"layer_{i}": [] for i in [2, 3, 4]},
        "noise_0.10": {f"layer_{i}": [] for i in [2, 3, 4]},
        "noise_0.20": {f"layer_{i}": [] for i in [2, 3, 4]},
        "blur_3": {f"layer_{i}": [] for i in [2, 3, 4]},
        "blur_5": {f"layer_{i}": [] for i in [2, 3, 4]},
        "shift_5px": {f"layer_{i}": [] for i in [2, 3, 4]},
        "shift_10px": {f"layer_{i}": [] for i in [2, 3, 4]},
    }

    print(f"\nAnalyzing {min(args.num_samples, len(data))} samples...")

    for i, item in enumerate(tqdm(data[:args.num_samples])):
        try:
            # Load GT image (handle both absolute and relative paths)
            output_img = item.get("output_image", item.get("image", ""))
            if os.path.isabs(output_img):
                img_path = output_img
            else:
                img_path = os.path.join(args.image_path, output_img)
            if not os.path.exists(img_path):
                continue

            gt_img = image_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

            # Test 1: Identical (sanity check - should be ~0)
            mse, _ = compute_perlayer_mse(seg_model, gt_img, gt_img)
            for k, v in mse.items():
                results["identical"][k].append(v)

            # Test 2: Various noise levels
            for noise_std, key in [(0.01, "noise_0.01"), (0.05, "noise_0.05"),
                                    (0.10, "noise_0.10"), (0.20, "noise_0.20")]:
                noisy = add_gaussian_noise(gt_img, noise_std)
                mse, _ = compute_perlayer_mse(seg_model, gt_img, noisy)
                for k, v in mse.items():
                    results[key][k].append(v)

            # Test 3: Blur
            for ks, key in [(3, "blur_3"), (5, "blur_5")]:
                blurred = add_blur(gt_img, ks)
                mse, _ = compute_perlayer_mse(seg_model, gt_img, blurred)
                for k, v in mse.items():
                    results[key][k].append(v)

            # Test 4: Spatial shift
            for shift, key in [(5, "shift_5px"), (10, "shift_10px")]:
                shifted = random_shift(gt_img, shift)
                mse, _ = compute_perlayer_mse(seg_model, gt_img, shifted)
                for k, v in mse.items():
                    results[key][k].append(v)

        except Exception as e:
            continue

    # Print results
    print("\n" + "="*80)
    print("PER-LAYER MSE SENSITIVITY ANALYSIS")
    print("="*80)

    print("\n## Summary Table: Mean MSE by Perturbation Type")
    print("-" * 85)
    print(f"{'Perturbation':<20} {'Layer 2':<15} {'Layer 3':<15} {'Layer 4':<15} {'Ratio 4/2':<15}")
    print("-" * 85)

    summary_data = []
    for pert_type in ["identical", "noise_0.01", "noise_0.05", "noise_0.10", "noise_0.20",
                      "blur_3", "blur_5", "shift_5px", "shift_10px"]:
        l2 = np.mean(results[pert_type]["layer_2"]) if results[pert_type]["layer_2"] else 0
        l3 = np.mean(results[pert_type]["layer_3"]) if results[pert_type]["layer_3"] else 0
        l4 = np.mean(results[pert_type]["layer_4"]) if results[pert_type]["layer_4"] else 0
        ratio = l4 / l2 if l2 > 1e-10 else float('inf')

        print(f"{pert_type:<20} {l2:<15.6f} {l3:<15.6f} {l4:<15.6f} {ratio:<15.2f}")

        summary_data.append({
            "perturbation": pert_type,
            "layer_2": l2,
            "layer_3": l3,
            "layer_4": l4,
            "ratio_4_over_2": ratio,
        })

    print("-" * 85)

    # Contribution analysis
    print("\n## Contribution % by Layer (for noise_0.10)")
    noise_data = results["noise_0.10"]
    if noise_data["layer_2"]:
        l2 = np.mean(noise_data["layer_2"])
        l3 = np.mean(noise_data["layer_3"])
        l4 = np.mean(noise_data["layer_4"])
        total = l2 + l3 + l4
        print(f"  Layer 2: {l2:.6f} ({l2/total*100:.1f}%)")
        print(f"  Layer 3: {l3:.6f} ({l3/total*100:.1f}%)")
        print(f"  Layer 4: {l4:.6f} ({l4/total*100:.1f}%)")
        print(f"  Total:   {total:.6f} → Average: {total/3:.6f}")

    # Compare with standalone test values
    print("\n## Comparison with Standalone Test (raw MSE ~12.68)")
    print("-" * 60)
    print("Standalone test reported:")
    print("  Layer 2: ~0.45")
    print("  Layer 3: ~1.98")
    print("  Layer 4: ~35.61")
    print("  Average: ~12.68")
    print("\nThis analysis shows per-layer sensitivity to perturbations.")
    print("The ~70x discrepancy likely comes from:")
    print("  1. Standalone test comparing GT vs heavily corrupted images")
    print("  2. Training compares GT vs predicted x1_hat (close to GT)")

    # Save results
    output_path = "perlayer_sensitivity_analysis.json"
    with open(output_path, 'w') as f:
        json.dump({
            "summary": summary_data,
            "raw_results": {k: {lk: {"mean": np.mean(lv), "std": np.std(lv), "n": len(lv)}
                               for lk, lv in v.items()}
                          for k, v in results.items()}
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_model_ckpt", type=str,
                        default="/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth")
    parser.add_argument("--json_file", type=str,
                        default="/home/wenting/zr/data/nih-cxr-lt/meta/train_prompts_balanced_conditions_v2.json")
    parser.add_argument("--image_path", type=str,
                        default="/home/wenting/zr/data/nih-cxr-lt/images_rescaled_256")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    main(args)
