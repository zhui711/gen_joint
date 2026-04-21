#!/usr/bin/env python3
"""
Analyze segmentation masks for generated CXR images.

This script samples generated images, runs them through a frozen segmentation model,
compares predictions against ground truth masks, and saves results with metrics.
"""

import argparse
import glob
import logging
import os
import random
import sys
from pathlib import Path

# Add local segmentation_models_pytorch to path (same as train_anatomy.py)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SEG_PKG_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), "Segmentation", "segmentation_models_pytorch")
if os.path.isdir(_SEG_PKG_PATH):
    sys.path.insert(0, _SEG_PKG_PATH)

import numpy as np
import torch
from PIL import Image

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "segmentation_models_pytorch not found. Either install it via pip "
        "or ensure the local package exists at: " + _SEG_PKG_PATH
    )

# Channel mapping for the 10 anatomical structures
CHANNEL_NAMES = {
    0: "Lung_Left",
    1: "Lung_Right",
    2: "Heart",
    3: "Aorta",
    4: "Liver",
    5: "Stomach",
    6: "Trachea",
    7: "Ribs",
    8: "Vertebrae",
    9: "Upper_Skeleton",
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze segmentation masks for generated CXR images"
    )
    parser.add_argument(
        "--gen_images_dir",
        type=str,
        default="/home/wenting/zr/gen_code/outputs/cxr_finetune_lora_plus_anatomy_5000_lamda0.1_subbatch4",
        help="Directory containing generated images",
    )
    parser.add_argument(
        "--gt_masks_dir",
        type=str,
        default="/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch",
        help="Directory containing ground truth masks",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth",
        help="Path to segmentation model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/wenting/zr/gen_code/analysis_results/cxr_finetune_lora_plus_anatomy_5000_lamda0.1_subbatch4",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of images to randomly sample for analysis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform a dry run without actually processing (for testing)",
    )
    return parser.parse_args()


def calculate_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Dice score between prediction and ground truth.

    Args:
        pred: Binary prediction array
        gt: Binary ground truth array

    Returns:
        Dice score (0-1), returns 1.0 if both are empty
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()

    if denominator == 0:
        return 1.0  # Both empty, perfect match

    return (2.0 * intersection) / denominator


def calculate_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate IoU (Intersection over Union) between prediction and ground truth.

    Args:
        pred: Binary prediction array
        gt: Binary ground truth array

    Returns:
        IoU score (0-1), returns 1.0 if both are empty
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0  # Both empty, perfect match

    return intersection / union


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load the segmentation model with pretrained weights.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from {checkpoint_path}")

    model = smp.Unet(
        encoder_name="resnet34",
        in_channels=3,
        classes=10,
        activation=None,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def preprocess_image(image_path: str, target_size: int = 256) -> torch.Tensor:
    """
    Load and preprocess an image for the segmentation model.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing

    Returns:
        Preprocessed image tensor of shape (1, 3, H, W)
    """
    img = Image.open(image_path)

    # Convert to RGB if grayscale
    if img.mode == "L":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if necessary
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.BILINEAR)

    # Convert to numpy and normalize to [-1, 1]
    img_np = np.array(img, dtype=np.float32)
    img_np = (img_np / 127.5) - 1.0

    # Convert to tensor (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def load_gt_mask(mask_path: str) -> np.ndarray:
    """
    Load ground truth mask from npz file.

    Args:
        mask_path: Path to the npz file

    Returns:
        Boolean mask array of shape (10, 256, 256)
    """
    data = np.load(mask_path)
    mask = data["mask"]
    return mask.astype(bool)


def save_mask_as_png(mask: np.ndarray, save_path: str):
    """
    Save a binary mask as a PNG image.

    Args:
        mask: Binary mask array (H, W)
        save_path: Path to save the PNG
    """
    # Scale boolean to uint8 (0 or 255)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    img = Image.fromarray(mask_uint8, mode="L")
    img.save(save_path)


def discover_images(gen_images_dir: str) -> list:
    """
    Recursively find all PNG images in the generated images directory.

    Args:
        gen_images_dir: Root directory to search

    Returns:
        List of image paths
    """
    pattern = os.path.join(gen_images_dir, "**", "*.png")
    images = glob.glob(pattern, recursive=True)
    return sorted(images)


def get_gt_mask_path(image_path: str, gen_images_dir: str, gt_masks_dir: str) -> str:
    """
    Construct the GT mask path from the generated image path.

    Args:
        image_path: Path to generated image
        gen_images_dir: Root directory of generated images
        gt_masks_dir: Root directory of GT masks

    Returns:
        Path to corresponding GT mask
    """
    # Get relative path from gen_images_dir
    rel_path = os.path.relpath(image_path, gen_images_dir)

    # Change extension from .png to .npz
    rel_path_npz = os.path.splitext(rel_path)[0] + ".npz"

    # Construct GT mask path
    gt_path = os.path.join(gt_masks_dir, rel_path_npz)

    return gt_path


def get_sample_identifier(image_path: str, gen_images_dir: str) -> str:
    """
    Get a unique identifier for the sample based on its path.

    Args:
        image_path: Path to the image
        gen_images_dir: Root directory of generated images

    Returns:
        Identifier string (e.g., "LIDC-IDRI-0030_0002")
    """
    rel_path = os.path.relpath(image_path, gen_images_dir)
    parts = Path(rel_path).parts

    if len(parts) >= 2:
        patient_id = parts[0]  # e.g., "LIDC-IDRI-0030"
        filename = Path(parts[-1]).stem  # e.g., "0002"
        return f"{patient_id}_{filename}"
    else:
        return Path(rel_path).stem


def process_sample(
    image_path: str,
    gt_mask_path: str,
    model: torch.nn.Module,
    device: torch.device,
    output_subdir: str,
    dry_run: bool = False,
) -> dict:
    """
    Process a single sample: run inference, calculate metrics, save outputs.

    Args:
        image_path: Path to the generated image
        gt_mask_path: Path to the GT mask
        model: Segmentation model
        device: Torch device
        output_subdir: Directory to save outputs for this sample
        dry_run: If True, skip actual processing

    Returns:
        Dictionary containing metrics for all channels
    """
    if dry_run:
        logger.info(f"  [DRY RUN] Would process: {image_path}")
        logger.info(f"  [DRY RUN] GT mask path: {gt_mask_path}")
        logger.info(f"  [DRY RUN] Output dir: {output_subdir}")
        return {}

    # Create output directory
    os.makedirs(output_subdir, exist_ok=True)

    # Load and preprocess image
    img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    # Load GT mask
    gt_mask = load_gt_mask(gt_mask_path)

    # Run inference
    with torch.inference_mode():
        logits = model(img_tensor)
        pred_probs = torch.sigmoid(logits)
        pred_mask = (pred_probs > 0.5).cpu().numpy()[0]  # (10, 256, 256)

    # Save the generated CXR image
    gen_img = Image.open(image_path)
    gen_img.save(os.path.join(output_subdir, "generated_cxr.png"))

    # Calculate metrics and save masks for each channel
    metrics = {}
    for ch_idx in range(10):
        ch_name = CHANNEL_NAMES[ch_idx]

        pred_ch = pred_mask[ch_idx]  # (256, 256)
        gt_ch = gt_mask[ch_idx]  # (256, 256)

        # Calculate metrics
        dice = calculate_dice(pred_ch, gt_ch)
        iou = calculate_iou(pred_ch, gt_ch)

        metrics[ch_idx] = {
            "name": ch_name,
            "dice": dice,
            "iou": iou,
        }

        # Save GT mask
        gt_filename = f"ch{ch_idx:02d}_{ch_name}_gt.png"
        save_mask_as_png(gt_ch, os.path.join(output_subdir, gt_filename))

        # Save predicted mask
        pred_filename = f"ch{ch_idx:02d}_{ch_name}_pred.png"
        save_mask_as_png(pred_ch, os.path.join(output_subdir, pred_filename))

    # Calculate average metrics
    avg_dice = np.mean([m["dice"] for m in metrics.values()])
    avg_iou = np.mean([m["iou"] for m in metrics.values()])

    # Write metrics.txt
    metrics_path = os.path.join(output_subdir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Segmentation Metrics Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"GT Mask: {gt_mask_path}\n\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Channel':<5} {'Name':<20} {'Dice':>10} {'IoU':>10}\n")
        f.write("-" * 60 + "\n")

        for ch_idx in range(10):
            m = metrics[ch_idx]
            f.write(f"{ch_idx:<5} {m['name']:<20} {m['dice']:>10.4f} {m['iou']:>10.4f}\n")

        f.write("-" * 60 + "\n")
        f.write(f"{'AVG':<5} {'':<20} {avg_dice:>10.4f} {avg_iou:>10.4f}\n")
        f.write("=" * 60 + "\n")

    return {
        "channel_metrics": metrics,
        "avg_dice": avg_dice,
        "avg_iou": avg_iou,
    }


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("Segmentation Mask Analysis")
    logger.info("=" * 60)
    logger.info(f"Generated Images Dir: {args.gen_images_dir}")
    logger.info(f"GT Masks Dir: {args.gt_masks_dir}")
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info(f"Num Samples: {args.num_samples}")
    logger.info(f"Random Seed: {args.seed}")
    if args.dry_run:
        logger.info("*** DRY RUN MODE - No actual processing ***")
    logger.info("=" * 60)

    # Discover all generated images
    logger.info("Discovering generated images...")
    all_images = discover_images(args.gen_images_dir)
    logger.info(f"Found {len(all_images)} generated images")

    if len(all_images) == 0:
        logger.error("No images found! Check the gen_images_dir path.")
        return

    # Sample images
    num_to_sample = min(args.num_samples, len(all_images))
    sampled_images = random.sample(all_images, num_to_sample)
    logger.info(f"Randomly sampled {num_to_sample} images")

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if not args.dry_run:
        model = load_model(args.checkpoint_path, device)
    else:
        model = None
        logger.info("[DRY RUN] Skipping model loading")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each sample
    all_sample_metrics = []
    skipped_count = 0

    for idx, image_path in enumerate(sampled_images):
        logger.info(f"\nProcessing sample {idx + 1}/{num_to_sample}")
        logger.info(f"  Image: {image_path}")

        # Get GT mask path
        gt_mask_path = get_gt_mask_path(image_path, args.gen_images_dir, args.gt_masks_dir)

        # Check if GT mask exists
        if not os.path.exists(gt_mask_path) and not args.dry_run:
            logger.warning(f"  GT mask not found: {gt_mask_path} - SKIPPING")
            skipped_count += 1
            continue

        # Get sample identifier and create output subdir
        sample_id = get_sample_identifier(image_path, args.gen_images_dir)
        output_subdir = os.path.join(args.output_dir, sample_id)

        # Process the sample
        sample_metrics = process_sample(
            image_path=image_path,
            gt_mask_path=gt_mask_path,
            model=model,
            device=device,
            output_subdir=output_subdir,
            dry_run=args.dry_run,
        )

        if sample_metrics:
            all_sample_metrics.append({
                "sample_id": sample_id,
                **sample_metrics,
            })
            logger.info(f"  Avg Dice: {sample_metrics['avg_dice']:.4f}, Avg IoU: {sample_metrics['avg_iou']:.4f}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN] No metrics calculated")
    elif all_sample_metrics:
        overall_avg_dice = np.mean([m["avg_dice"] for m in all_sample_metrics])
        overall_avg_iou = np.mean([m["avg_iou"] for m in all_sample_metrics])

        logger.info(f"Processed: {len(all_sample_metrics)} samples")
        logger.info(f"Skipped (missing GT): {skipped_count} samples")
        logger.info(f"Overall Average Dice: {overall_avg_dice:.4f}")
        logger.info(f"Overall Average IoU: {overall_avg_iou:.4f}")

        # Write overall summary
        summary_path = os.path.join(args.output_dir, "overall_summary.txt")
        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Overall Segmentation Analysis Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Samples Analyzed: {len(all_sample_metrics)}\n")
            f.write(f"Samples Skipped (missing GT): {skipped_count}\n\n")
            f.write(f"Overall Average Dice: {overall_avg_dice:.4f}\n")
            f.write(f"Overall Average IoU: {overall_avg_iou:.4f}\n\n")

            # Per-channel averages
            f.write("-" * 60 + "\n")
            f.write("Per-Channel Average Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Channel':<5} {'Name':<20} {'Dice':>10} {'IoU':>10}\n")
            f.write("-" * 60 + "\n")

            for ch_idx in range(10):
                ch_name = CHANNEL_NAMES[ch_idx]
                ch_dices = [m["channel_metrics"][ch_idx]["dice"] for m in all_sample_metrics]
                ch_ious = [m["channel_metrics"][ch_idx]["iou"] for m in all_sample_metrics]
                avg_ch_dice = np.mean(ch_dices)
                avg_ch_iou = np.mean(ch_ious)
                f.write(f"{ch_idx:<5} {ch_name:<20} {avg_ch_dice:>10.4f} {avg_ch_iou:>10.4f}\n")

            f.write("=" * 60 + "\n")

            # Individual sample results
            f.write("\n\nIndividual Sample Results:\n")
            f.write("-" * 60 + "\n")
            for m in all_sample_metrics:
                f.write(f"{m['sample_id']}: Dice={m['avg_dice']:.4f}, IoU={m['avg_iou']:.4f}\n")

        logger.info(f"Summary saved to: {summary_path}")
    else:
        logger.warning("No samples were processed!")

    logger.info("=" * 60)
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
