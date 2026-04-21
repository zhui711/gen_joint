#!/usr/bin/env python3
"""
eval_dice_standalone.py - Standalone Dice Evaluation for Baseline vs. SegMSE

Compares Baseline OmniGen (30k) vs. OmniGen+SegMSE by computing actual Dice scores
against Ground Truth masks using the frozen ResNet34-UNet segmentation model.

This answers: "Did adding SegMSE loss actually improve anatomical correctness?"

Usage:
    python eval_dice_standalone.py \
        --baseline_dir /home/wenting/zr/gen_code/outputs/cxr_finetune_lora_30000 \
        --segmse_dir /home/wenting/zr/gen_code/outputs/cxr_finetune_lora_30ksteps_SegMSE_lamda0.005_subbatch16_10500 \
        --gt_mask_dir /home/wenting/zr/Segmentation/data/lidc_TotalSeg \
        --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
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
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add segmentation_models_pytorch to path
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")
import segmentation_models_pytorch as smp


# ===========================================================================
# Configuration - Must match the segmentation model training
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
NUM_CLASSES = len(CLASS_NAMES)


# ===========================================================================
# Model Loading
# ===========================================================================

def load_seg_model(ckpt_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load the frozen ResNet34-UNet segmentation model."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
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
# Data Loading
# ===========================================================================

def load_image_tensor(path: str, device: str = "cuda") -> torch.Tensor:
    """Load image as tensor normalized to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = tensor / 127.5 - 1.0
    return tensor.unsqueeze(0).to(device)


def load_gt_mask_10ch(
    mask_dir: str,
    patient_id: str,
    view_idx: int,
    device: str = "cuda"
) -> Optional[torch.Tensor]:
    """Load ground truth mask and convert from 119-channel to 10-channel format."""
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

    data = np.load(mask_path)
    orig_mask = data["mask"]

    meta_path = os.path.join(mask_dir, patient_id, "02_totalseg", "phase2_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            labels_found = meta.get("labels_found", list(range(orig_mask.shape[0])))
    else:
        labels_found = list(range(orig_mask.shape[0]))

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
    """Discover all generated images in a directory."""
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
# Dice Computation
# ===========================================================================

def compute_dice(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> float:
    """Compute Dice score for binary masks."""
    pred = pred.float()
    gt = gt.float()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    if union < eps:
        return 1.0
    return (2.0 * intersection / (union + eps)).item()


@torch.no_grad()
def evaluate_directory(
    gen_dir: str,
    gt_mask_dir: str,
    model: torch.nn.Module,
    device: str = "cuda",
    desc: str = "Evaluating",
    threshold: float = 0.5,
) -> Dict:
    """Evaluate all images in a generated directory."""
    samples = discover_samples(gen_dir)
    print(f"[INFO] Found {len(samples)} generated images in {gen_dir}")

    tp_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    fp_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    fn_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)

    per_sample_dice = []
    skipped = 0

    for sample in tqdm(samples, desc=desc):
        gt_mask = load_gt_mask_10ch(
            gt_mask_dir, sample["patient_id"], sample["view_idx"], device
        )

        if gt_mask is None:
            skipped += 1
            continue

        img_tensor = load_image_tensor(sample["gen_path"], device)
        logits = model(img_tensor)
        pred_mask = (torch.sigmoid(logits) > threshold).squeeze(0).float()

        # Per-sample Dice
        sample_dice = []
        for ch_idx in range(NUM_CLASSES):
            d = compute_dice(pred_mask[ch_idx], gt_mask[ch_idx])
            sample_dice.append(d)
        per_sample_dice.append(np.mean(sample_dice))

        # Accumulate for micro-average
        tp = (pred_mask * gt_mask).sum(dim=(1, 2))
        fp = (pred_mask * (1 - gt_mask)).sum(dim=(1, 2))
        fn = ((1 - pred_mask) * gt_mask).sum(dim=(1, 2))

        tp_sum += tp.double()
        fp_sum += fp.double()
        fn_sum += fn.double()

    print(f"[INFO] Evaluated {len(per_sample_dice)} samples, skipped {skipped}")

    micro_dice = (2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum + 1e-7)).cpu().numpy()

    return {
        "n_samples": len(per_sample_dice),
        "n_skipped": skipped,
        "micro_dice_per_class": {CLASS_NAMES[i]: float(micro_dice[i]) for i in range(NUM_CLASSES)},
        "micro_dice_macro": float(micro_dice.mean()),
        "macro_dice_mean": float(np.mean(per_sample_dice)),
        "macro_dice_std": float(np.std(per_sample_dice)),
    }


def format_report(baseline: Dict, segmse: Dict) -> str:
    """Generate formatted comparison report."""
    lines = []
    lines.append("=" * 85)
    lines.append("ANATOMY DICE SCORE COMPARISON: Baseline (30k) vs. +SegMSE (10.5k)")
    lines.append("=" * 85)
    lines.append("")
    lines.append(f"Samples evaluated: Baseline={baseline['n_samples']}, SegMSE={segmse['n_samples']}")
    lines.append("")
    lines.append(f"{'Class':<20} {'Baseline':<12} {'SegMSE':<12} {'Delta':<12} {'Status':<10}")
    lines.append("-" * 70)

    for class_name in CLASS_NAMES:
        b = baseline["micro_dice_per_class"][class_name]
        s = segmse["micro_dice_per_class"][class_name]
        d = s - b

        if d > 0.02:
            status = "+++"
        elif d > 0.01:
            status = "++"
        elif d > 0:
            status = "+"
        elif d > -0.01:
            status = "~"
        elif d > -0.02:
            status = "-"
        else:
            status = "---"

        lines.append(f"{class_name:<20} {b:.4f}       {s:.4f}       {d:+.4f}      {status}")

    lines.append("-" * 70)

    b_macro = baseline["micro_dice_macro"]
    s_macro = segmse["micro_dice_macro"]
    d_macro = s_macro - b_macro

    lines.append(f"{'MICRO-DICE (mean)':<20} {b_macro:.4f}       {s_macro:.4f}       {d_macro:+.4f}")
    lines.append("")
    lines.append("=" * 85)
    lines.append("")

    lines.append("DIAGNOSTIC INTERPRETATION:")
    lines.append("-" * 40)

    if d_macro > 0.02:
        lines.append("[SUCCESS] Anatomy significantly IMPROVED with SegMSE loss.")
        lines.append("          The FID/LPIPS degradation may be acceptable tradeoff.")
    elif d_macro > 0:
        lines.append("[MILD GAIN] Slight anatomy improvement.")
        lines.append("          May not justify the visual quality degradation.")
    elif d_macro > -0.01:
        lines.append("[NEUTRAL] Anatomy essentially unchanged.")
        lines.append("          SegMSE loss is NOT providing anatomical benefit.")
    else:
        lines.append("[FAILURE] Anatomy DEGRADED with SegMSE loss!")
        lines.append("          This indicates adversarial shortcuts or severe gradient conflict.")
        lines.append("          Recommendation: Disable SegMSE loss or redesign it.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Dice Evaluation: Baseline vs. SegMSE")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--segmse_dir", type=str, required=True)
    parser.add_argument("--gt_mask_dir", type=str, required=True)
    parser.add_argument("--seg_model_ckpt", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="anatomy_dice_comparison.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = load_seg_model(args.seg_model_ckpt, device)

    print("\n[1/2] Evaluating Baseline...")
    baseline = evaluate_directory(
        args.baseline_dir, args.gt_mask_dir, model, device, "Baseline", args.threshold
    )

    print("\n[2/2] Evaluating SegMSE...")
    segmse = evaluate_directory(
        args.segmse_dir, args.gt_mask_dir, model, device, "SegMSE", args.threshold
    )

    report = format_report(baseline, segmse)
    print("\n" + report)

    results = {
        "baseline": baseline,
        "segmse": segmse,
        "delta": {
            "micro_dice_per_class": {
                c: segmse["micro_dice_per_class"][c] - baseline["micro_dice_per_class"][c]
                for c in CLASS_NAMES
            },
            "micro_dice_macro": segmse["micro_dice_macro"] - baseline["micro_dice_macro"],
        }
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
