#!/usr/bin/env python3
"""
Tool 1: Objective Anatomy Evaluation Script (eval_anatomy_dice.py)

Evaluates generated CXR images using the frozen ResNet34-UNet segmentation model.
Compares Baseline OmniGen vs. OmniGen+Seg by computing Macro-Dice scores against
Ground Truth masks.

This tool answers: "Did the anatomy actually improve despite worse FID?"

Usage:
    python eval_anatomy_dice.py \
        --baseline_dir /path/to/baseline_generated_images \
        --seg_dir /path/to/omnigen_seg_generated_images \
        --gt_mask_dir /path/to/lidc_TotalSeg \
        --seg_model_ckpt /path/to/best_anatomy_model.pth \
        --output_report anatomy_comparison_report.json

Example:
    python diagnostics/eval_anatomy_dice.py \
        --baseline_dir outputs/cxr_finetune_lora_30000 \
        --seg_dir outputs/cxr_finetune_lora_30ksteps_SegMSE_lamda0.005_subbatch16_10500 \
        --gt_mask_dir /home/wenting/zr/Segmentation/data/lidc_TotalSeg \
        --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
        --output_report diagnostics/anatomy_comparison_report.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add segmentation_models_pytorch to path
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")
import segmentation_models_pytorch as smp


# ===========================================================================
# Constants (must match training)
# ===========================================================================

TARGET_GROUPS = {
    "Lung_Left": [10, 11],
    "Lung_Right": [12, 13, 14],
    "Heart": [51],
    "Aorta": [52],
    "Liver": [5],
    "Stomach": [6],
    "Trachea": [16],
    "Ribs": list(range(92, 116)),
    "Vertebrae": list(range(25, 51)),
    "Upper_Skeleton": [69, 70, 71, 72, 73, 74]
}

CLASS_NAMES = list(TARGET_GROUPS.keys())


# ===========================================================================
# Model Loading
# ===========================================================================

def load_frozen_seg_model(checkpoint_path, device="cuda"):
    """Load the frozen ResNet34-UNet segmentation model."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # weights from checkpoint
        in_channels=3,
        classes=10,
        activation=None,  # raw logits
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    print(f"[INFO] Loaded segmentation model from: {checkpoint_path}")
    if "val_dice" in ckpt:
        print(f"[INFO] Model validation Dice: {ckpt['val_dice']:.4f}")

    return model


# ===========================================================================
# Data Loading
# ===========================================================================

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

    # Normalize to [-1, 1] (same as OmniGen VAE output)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
    tensor = tensor.unsqueeze(0).to(device)

    return tensor


def load_gt_mask(mask_dir, patient_id, image_idx, labels_found=None):
    """
    Load ground truth mask and convert from 119-channel to 10-channel format.

    Args:
        mask_dir: Path to lidc_TotalSeg directory
        patient_id: e.g., "LIDC-IDRI-0030"
        image_idx: e.g., "0001" (without .png extension)
        labels_found: List of label IDs present in this patient (from metadata)

    Returns:
        torch.Tensor of shape (10, 256, 256) with binary values
    """
    # Construct mask path
    # Format: {mask_dir}/{patient_id}/04_drr_256/mask_compact/{idx:06d}.npz
    mask_path = os.path.join(
        mask_dir, patient_id, "04_drr_256", "mask_compact",
        f"{int(image_idx):06d}.npz"
    )

    if not os.path.exists(mask_path):
        # Try alternative format without zero-padding
        mask_path = os.path.join(
            mask_dir, patient_id, "04_drr_256", "mask_compact",
            f"{int(image_idx)}.npz"
        )

    if not os.path.exists(mask_path):
        print(f"[WARN] Mask not found: {mask_path}")
        return None

    # Load 119-channel mask
    npz_data = np.load(mask_path)
    orig_mask = npz_data['mask']  # (119, 256, 256) bool

    # Load labels_found from metadata if not provided
    if labels_found is None:
        meta_path = os.path.join(mask_dir, patient_id, "02_totalseg", "phase2_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                labels_found = meta.get("labels_found", list(range(119)))
        else:
            labels_found = list(range(119))

    # Convert to 10-channel format
    target_mask = np.zeros((10, 256, 256), dtype=np.float32)

    for i, (group_name, ids) in enumerate(TARGET_GROUPS.items()):
        valid_slices = []
        for class_id in ids:
            if class_id in labels_found and class_id < orig_mask.shape[0]:
                valid_slices.append(orig_mask[class_id, :, :])

        if valid_slices:
            merged = np.any(np.stack(valid_slices, axis=0), axis=0)
            target_mask[i] = merged.astype(np.float32)

    return torch.from_numpy(target_mask)


def discover_image_pairs(gen_dir, gt_mask_dir):
    """
    Discover all generated images and their corresponding GT masks.

    Returns:
        List of dicts with keys: patient_id, image_idx, gen_path, mask_available
    """
    pairs = []

    for patient_folder in os.listdir(gen_dir):
        patient_path = os.path.join(gen_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        # Skip non-LIDC folders
        if not patient_folder.startswith("LIDC-IDRI-"):
            continue

        for img_file in os.listdir(patient_path):
            if not img_file.endswith(".png"):
                continue

            image_idx = os.path.splitext(img_file)[0]  # e.g., "0001"
            gen_path = os.path.join(patient_path, img_file)

            # Check if GT mask exists
            mask_path = os.path.join(
                gt_mask_dir, patient_folder, "04_drr_256", "mask_compact",
                f"{int(image_idx):06d}.npz"
            )
            mask_available = os.path.exists(mask_path)

            pairs.append({
                "patient_id": patient_folder,
                "image_idx": image_idx,
                "gen_path": gen_path,
                "mask_available": mask_available,
            })

    return pairs


# ===========================================================================
# Dice Score Computation
# ===========================================================================

def compute_dice_scores(pred_mask, gt_mask, eps=1e-7):
    """
    Compute per-class Dice scores.

    Args:
        pred_mask: (10, H, W) predicted binary mask
        gt_mask: (10, H, W) ground truth binary mask

    Returns:
        dict mapping class_name -> dice_score
    """
    scores = {}

    for i, class_name in enumerate(CLASS_NAMES):
        pred = pred_mask[i].float()
        gt = gt_mask[i].float()

        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum()

        if union < eps:
            # Both empty -> perfect match
            dice = 1.0
        else:
            dice = (2.0 * intersection / (union + eps)).item()

        scores[class_name] = dice

    return scores


@torch.no_grad()
def predict_mask(model, image_tensor, threshold=0.5):
    """
    Run inference to get predicted segmentation mask.

    Args:
        model: Frozen segmentation model
        image_tensor: (1, 3, 256, 256) in [-1, 1]
        threshold: Sigmoid threshold for binarization

    Returns:
        (10, 256, 256) binary mask tensor
    """
    logits = model(image_tensor)  # (1, 10, H, W)
    probs = torch.sigmoid(logits)
    mask = (probs > threshold).squeeze(0).float()  # (10, H, W)
    return mask


# ===========================================================================
# Main Evaluation
# ===========================================================================

def evaluate_directory(gen_dir, gt_mask_dir, model, device, desc="Evaluating"):
    """
    Evaluate all generated images in a directory.

    Returns:
        dict with per_class_dice, macro_dice, per_sample_results
    """
    pairs = discover_image_pairs(gen_dir, gt_mask_dir)
    valid_pairs = [p for p in pairs if p["mask_available"]]

    print(f"[INFO] Found {len(pairs)} images, {len(valid_pairs)} with GT masks")

    if len(valid_pairs) == 0:
        return None

    # Accumulators for micro-average
    tp_sum = torch.zeros(10, dtype=torch.float64, device=device)
    fp_sum = torch.zeros(10, dtype=torch.float64, device=device)
    fn_sum = torch.zeros(10, dtype=torch.float64, device=device)

    per_sample_results = []

    for pair in tqdm(valid_pairs, desc=desc):
        # Load generated image
        gen_tensor = load_image_as_tensor(pair["gen_path"], device)

        # Predict mask
        pred_mask = predict_mask(model, gen_tensor)

        # Load GT mask
        gt_mask = load_gt_mask(gt_mask_dir, pair["patient_id"], pair["image_idx"])
        if gt_mask is None:
            continue
        gt_mask = gt_mask.to(device)

        # Compute per-sample Dice
        sample_dice = compute_dice_scores(pred_mask, gt_mask)
        sample_dice["patient_id"] = pair["patient_id"]
        sample_dice["image_idx"] = pair["image_idx"]
        sample_dice["macro_dice"] = np.mean(list({k: v for k, v in sample_dice.items()
                                                  if k in CLASS_NAMES}.values()))
        per_sample_results.append(sample_dice)

        # Accumulate stats for micro-average
        pred_binary = pred_mask.long()
        gt_binary = gt_mask.long()

        tp = (pred_binary * gt_binary).sum(dim=(1, 2))
        fp = (pred_binary * (1 - gt_binary)).sum(dim=(1, 2))
        fn = ((1 - pred_binary) * gt_binary).sum(dim=(1, 2))

        tp_sum += tp.double()
        fp_sum += fp.double()
        fn_sum += fn.double()

    # Compute per-class Dice (micro-average across all samples)
    per_class_dice = (2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum + 1e-7)).cpu().numpy()

    results = {
        "n_samples": len(per_sample_results),
        "per_class_dice": {CLASS_NAMES[i]: float(per_class_dice[i]) for i in range(10)},
        "macro_dice": float(per_class_dice.mean()),
        "per_sample_results": per_sample_results,
    }

    return results


def print_comparison_table(baseline_results, seg_results):
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 80)
    print("ANATOMY DICE COMPARISON: Baseline vs. OmniGen+Seg")
    print("=" * 80)

    print(f"\n{'Class':<20} {'Baseline':<15} {'OmniGen+Seg':<15} {'Delta':<15}")
    print("-" * 65)

    for class_name in CLASS_NAMES:
        b_dice = baseline_results["per_class_dice"][class_name]
        s_dice = seg_results["per_class_dice"][class_name]
        delta = s_dice - b_dice
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        indicator = "+++" if delta > 0.02 else ("++" if delta > 0.01 else
                    ("+" if delta > 0 else ("---" if delta < -0.02 else
                    ("--" if delta < -0.01 else ("-" if delta < 0 else "=")))))

        print(f"{class_name:<20} {b_dice:.4f}         {s_dice:.4f}         {delta_str} {indicator}")

    print("-" * 65)
    b_macro = baseline_results["macro_dice"]
    s_macro = seg_results["macro_dice"]
    delta_macro = s_macro - b_macro
    delta_str = f"+{delta_macro:.4f}" if delta_macro >= 0 else f"{delta_macro:.4f}"

    print(f"{'MACRO DICE':<20} {b_macro:.4f}         {s_macro:.4f}         {delta_str}")
    print("=" * 80)

    # Diagnostic interpretation
    print("\nDIAGNOSTIC INTERPRETATION:")
    print("-" * 40)

    if delta_macro > 0.02:
        print("[GOOD] Anatomy significantly improved with segmentation loss.")
        print("       The model learned better anatomical structure.")
    elif delta_macro > 0:
        print("[MILD] Slight anatomy improvement with segmentation loss.")
        print("       But gains may not justify FID degradation.")
    elif delta_macro > -0.01:
        print("[NEUTRAL] Anatomy roughly unchanged.")
        print("       Segmentation loss may not be contributing effectively.")
    else:
        print("[CONCERNING] Anatomy WORSE with segmentation loss!")
        print("       This strongly suggests adversarial shortcuts or gradient conflict.")

    # Check for adversarial shortcut signatures
    large_improvements = [c for c in CLASS_NAMES
                         if seg_results["per_class_dice"][c] - baseline_results["per_class_dice"][c] > 0.03]
    large_degradations = [c for c in CLASS_NAMES
                         if seg_results["per_class_dice"][c] - baseline_results["per_class_dice"][c] < -0.03]

    if large_improvements and large_degradations:
        print("\n[WARNING] Mixed pattern: Some classes improved, others degraded significantly.")
        print(f"         Improved: {large_improvements}")
        print(f"         Degraded: {large_degradations}")
        print("         This suggests the model may be creating localized artifacts.")


def main():
    parser = argparse.ArgumentParser(description="Objective Anatomy Dice Evaluation")
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="Directory with baseline OmniGen generated images")
    parser.add_argument("--seg_dir", type=str, required=True,
                        help="Directory with OmniGen+Seg generated images")
    parser.add_argument("--gt_mask_dir", type=str, required=True,
                        help="Directory with ground truth masks (lidc_TotalSeg)")
    parser.add_argument("--seg_model_ckpt", type=str, required=True,
                        help="Path to frozen segmentation model checkpoint")
    parser.add_argument("--output_report", type=str, default="anatomy_comparison_report.json",
                        help="Path to save JSON report")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda/cpu)")

    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_frozen_seg_model(args.seg_model_ckpt, device)

    # Evaluate baseline
    print("\n[1/2] Evaluating Baseline OmniGen...")
    baseline_results = evaluate_directory(
        args.baseline_dir, args.gt_mask_dir, model, device,
        desc="Baseline"
    )

    # Evaluate OmniGen+Seg
    print("\n[2/2] Evaluating OmniGen+Seg...")
    seg_results = evaluate_directory(
        args.seg_dir, args.gt_mask_dir, model, device,
        desc="OmniGen+Seg"
    )

    if baseline_results is None or seg_results is None:
        print("[ERROR] Could not evaluate one or both directories!")
        return

    # Print comparison
    print_comparison_table(baseline_results, seg_results)

    # Save report
    report = {
        "baseline": {
            "dir": args.baseline_dir,
            "n_samples": baseline_results["n_samples"],
            "per_class_dice": baseline_results["per_class_dice"],
            "macro_dice": baseline_results["macro_dice"],
        },
        "omnigen_seg": {
            "dir": args.seg_dir,
            "n_samples": seg_results["n_samples"],
            "per_class_dice": seg_results["per_class_dice"],
            "macro_dice": seg_results["macro_dice"],
        },
        "delta": {
            "per_class_dice": {c: seg_results["per_class_dice"][c] - baseline_results["per_class_dice"][c]
                              for c in CLASS_NAMES},
            "macro_dice": seg_results["macro_dice"] - baseline_results["macro_dice"],
        }
    }

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[INFO] Report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
