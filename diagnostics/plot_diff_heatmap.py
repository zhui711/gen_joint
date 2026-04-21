#!/usr/bin/env python3
"""
Tool 3: Visual Difference Heatmap Generator (plot_diff_heatmap.py)

Creates side-by-side comparison images showing:
  [Baseline] | [OmniGen+Seg] | [Difference Heatmap]

This helps visually identify where the "adversarial patches" or localized
artifacts might be located when comparing Baseline vs. OmniGen+Seg outputs.

Usage:
    # Single image comparison
    python plot_diff_heatmap.py \
        --baseline_img /path/to/baseline/0001.png \
        --seg_img /path/to/seg/0001.png \
        --output comparison.png

    # Batch comparison (entire directories)
    python plot_diff_heatmap.py \
        --baseline_dir outputs/cxr_finetune_lora_30000 \
        --seg_dir outputs/cxr_finetune_lora_30ksteps_SegMSE_lamda0.005_subbatch16_10500 \
        --output_dir diagnostics/diff_heatmaps \
        --n_samples 50

Example:
    python diagnostics/plot_diff_heatmap.py \
        --baseline_dir /home/wenting/zr/gen_code/outputs/cxr_finetune_lora_30000 \
        --seg_dir /home/wenting/zr/gen_code/outputs/cxr_finetune_lora_30ksteps_SegMSE_lamda0.005_subbatch16_10500 \
        --output_dir /home/wenting/zr/gen_code/diagnostics/diff_heatmaps \
        --n_samples 100 \
        --threshold 0.1
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm


def load_image_grayscale(image_path: str) -> np.ndarray:
    """
    Load an image and convert to grayscale float32 [0, 1].

    Args:
        image_path: Path to image file

    Returns:
        numpy array of shape (H, W) with values in [0, 1]
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def load_image_rgb(image_path: str) -> np.ndarray:
    """
    Load an image as RGB uint8.

    Args:
        image_path: Path to image file

    Returns:
        numpy array of shape (H, W, 3) with values in [0, 255]
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def compute_difference_map(
    baseline: np.ndarray,
    seg: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute absolute pixel-wise difference between two grayscale images.

    Args:
        baseline: Grayscale image [0, 1]
        seg: Grayscale image [0, 1]
        normalize: If True, normalize difference to [0, 1]

    Returns:
        Difference map as numpy array (H, W) in [0, 1]
    """
    # Ensure same shape
    if baseline.shape != seg.shape:
        # Resize seg to match baseline
        seg = cv2.resize(seg, (baseline.shape[1], baseline.shape[0]))

    diff = np.abs(baseline - seg)

    if normalize and diff.max() > 1e-6:
        diff = diff / diff.max()

    return diff


def apply_colormap(
    diff_map: np.ndarray,
    colormap: str = "jet",
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Apply a colormap to a difference map.

    Args:
        diff_map: Grayscale difference map [0, 1]
        colormap: Matplotlib colormap name ('jet', 'hot', 'viridis', etc.)
        threshold: Values below this are set to black (useful for noise suppression)

    Returns:
        RGB image as uint8 numpy array (H, W, 3)
    """
    # Apply threshold
    diff_map = diff_map.copy()
    diff_map[diff_map < threshold] = 0

    # Get colormap
    cmap = cm.get_cmap(colormap)

    # Apply colormap (returns RGBA)
    colored = cmap(diff_map)[:, :, :3]  # Drop alpha

    # Convert to uint8
    colored = (colored * 255).astype(np.uint8)

    return colored


def create_comparison_image(
    baseline_img: np.ndarray,
    seg_img: np.ndarray,
    diff_heatmap: np.ndarray,
    labels: Tuple[str, str, str] = ("Baseline", "OmniGen+Seg", "Difference"),
    add_colorbar: bool = True,
    diff_range: Tuple[float, float] = (0, 1),
) -> np.ndarray:
    """
    Create a side-by-side comparison image with labels.

    Args:
        baseline_img: RGB baseline image (H, W, 3)
        seg_img: RGB seg image (H, W, 3)
        diff_heatmap: RGB difference heatmap (H, W, 3)
        labels: Labels for each panel
        add_colorbar: Whether to add a colorbar for the difference
        diff_range: Range for colorbar

    Returns:
        Combined image as numpy array
    """
    # Ensure all images have same height
    h = baseline_img.shape[0]
    w = baseline_img.shape[1]

    # Resize if needed
    if seg_img.shape[0] != h or seg_img.shape[1] != w:
        seg_img = cv2.resize(seg_img, (w, h))
    if diff_heatmap.shape[0] != h or diff_heatmap.shape[1] != w:
        diff_heatmap = cv2.resize(diff_heatmap, (w, h))

    # Create figure
    if add_colorbar:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4),
                                  gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot images
    axes[0].imshow(baseline_img)
    axes[0].set_title(labels[0], fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(seg_img)
    axes[1].set_title(labels[1], fontsize=12, fontweight='bold')
    axes[1].axis('off')

    im = axes[2].imshow(diff_heatmap)
    axes[2].set_title(labels[2], fontsize=12, fontweight='bold')
    axes[2].axis('off')

    if add_colorbar:
        # Create a normalized colorbar
        sm = plt.cm.ScalarMappable(cmap='jet',
                                    norm=plt.Normalize(vmin=diff_range[0], vmax=diff_range[1]))
        plt.colorbar(sm, cax=axes[3], label='Pixel Difference')

    plt.tight_layout()

    # Convert figure to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return img_array


def create_simple_comparison(
    baseline_path: str,
    seg_path: str,
    output_path: str,
    colormap: str = "jet",
    threshold: float = 0.0,
    add_stats: bool = True,
):
    """
    Create a simple side-by-side comparison image using OpenCV.

    This is faster than matplotlib and suitable for batch processing.

    Args:
        baseline_path: Path to baseline image
        seg_path: Path to seg image
        output_path: Path to save comparison
        colormap: OpenCV colormap name (COLORMAP_JET, COLORMAP_HOT, etc.)
        threshold: Threshold for difference map
        add_stats: Add statistics text overlay
    """
    # Load images
    baseline_gray = load_image_grayscale(baseline_path)
    seg_gray = load_image_grayscale(seg_path)
    baseline_rgb = load_image_rgb(baseline_path)
    seg_rgb = load_image_rgb(seg_path)

    # Compute difference
    diff_map = compute_difference_map(baseline_gray, seg_gray, normalize=False)

    # Statistics
    mean_diff = diff_map.mean()
    max_diff = diff_map.max()
    std_diff = diff_map.std()

    # Normalize and threshold for visualization
    diff_visual = diff_map.copy()
    if diff_visual.max() > 1e-6:
        diff_visual = diff_visual / diff_visual.max()
    diff_visual[diff_visual < threshold] = 0

    # Apply colormap
    diff_uint8 = (diff_visual * 255).astype(np.uint8)

    # OpenCV colormap
    cv_colormap = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
    diff_colored = cv2.applyColorMap(diff_uint8, cv_colormap)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

    # Ensure same size
    h, w = baseline_rgb.shape[:2]
    if seg_rgb.shape[:2] != (h, w):
        seg_rgb = cv2.resize(seg_rgb, (w, h))
    if diff_colored.shape[:2] != (h, w):
        diff_colored = cv2.resize(diff_colored, (w, h))

    # Add labels
    label_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    def add_label(img, text):
        labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
        labeled[label_height:] = img

        # White background for label
        labeled[:label_height] = 255

        # Add text
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = label_height - 8
        cv2.putText(labeled, text, (text_x, text_y), font, font_scale,
                    (0, 0, 0), font_thickness)

        return labeled

    baseline_labeled = add_label(baseline_rgb, "Baseline")
    seg_labeled = add_label(seg_rgb, "OmniGen+Seg")
    diff_labeled = add_label(diff_colored, "Difference Heatmap")

    # Concatenate horizontally
    comparison = np.hstack([baseline_labeled, seg_labeled, diff_labeled])

    # Add statistics panel if requested
    if add_stats:
        stats_height = 40
        stats_panel = np.ones((stats_height, comparison.shape[1], 3), dtype=np.uint8) * 255

        stats_text = f"Mean Diff: {mean_diff:.4f} | Max Diff: {max_diff:.4f} | Std: {std_diff:.4f}"
        text_size = cv2.getTextSize(stats_text, font, 0.6, 1)[0]
        text_x = (comparison.shape[1] - text_size[0]) // 2
        cv2.putText(stats_panel, stats_text, (text_x, 28), font, 0.6, (0, 0, 0), 1)

        comparison = np.vstack([comparison, stats_panel])

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(comparison).save(output_path)


def find_matching_images(
    baseline_dir: str,
    seg_dir: str,
) -> List[Tuple[str, str, str, str]]:
    """
    Find matching image pairs between baseline and seg directories.

    Returns:
        List of tuples: (patient_id, image_idx, baseline_path, seg_path)
    """
    pairs = []

    for patient_folder in os.listdir(baseline_dir):
        baseline_patient = os.path.join(baseline_dir, patient_folder)
        seg_patient = os.path.join(seg_dir, patient_folder)

        if not os.path.isdir(baseline_patient):
            continue
        if not os.path.isdir(seg_patient):
            continue

        for img_file in os.listdir(baseline_patient):
            if not img_file.endswith(".png"):
                continue

            baseline_path = os.path.join(baseline_patient, img_file)
            seg_path = os.path.join(seg_patient, img_file)

            if os.path.exists(seg_path):
                image_idx = os.path.splitext(img_file)[0]
                pairs.append((patient_folder, image_idx, baseline_path, seg_path))

    return pairs


def compute_global_statistics(
    baseline_dir: str,
    seg_dir: str,
    n_samples: Optional[int] = None,
) -> dict:
    """
    Compute global difference statistics across all image pairs.

    Returns:
        Dict with mean, std, max difference statistics
    """
    pairs = find_matching_images(baseline_dir, seg_dir)

    if n_samples and len(pairs) > n_samples:
        import random
        pairs = random.sample(pairs, n_samples)

    all_diffs = []
    all_means = []
    all_maxs = []

    for _, _, baseline_path, seg_path in tqdm(pairs, desc="Computing statistics"):
        baseline_gray = load_image_grayscale(baseline_path)
        seg_gray = load_image_grayscale(seg_path)
        diff = compute_difference_map(baseline_gray, seg_gray, normalize=False)

        all_diffs.extend(diff.flatten().tolist())
        all_means.append(diff.mean())
        all_maxs.append(diff.max())

    all_diffs = np.array(all_diffs)

    return {
        "n_pairs": len(pairs),
        "pixel_diff_mean": float(np.mean(all_diffs)),
        "pixel_diff_std": float(np.std(all_diffs)),
        "pixel_diff_median": float(np.median(all_diffs)),
        "pixel_diff_p95": float(np.percentile(all_diffs, 95)),
        "pixel_diff_p99": float(np.percentile(all_diffs, 99)),
        "image_mean_diff": {
            "mean": float(np.mean(all_means)),
            "std": float(np.std(all_means)),
            "min": float(np.min(all_means)),
            "max": float(np.max(all_means)),
        },
        "image_max_diff": {
            "mean": float(np.mean(all_maxs)),
            "std": float(np.std(all_maxs)),
            "min": float(np.min(all_maxs)),
            "max": float(np.max(all_maxs)),
        },
    }


def create_summary_grid(
    pairs: List[Tuple[str, str, str, str]],
    output_path: str,
    n_cols: int = 4,
    colormap: str = "jet",
    threshold: float = 0.0,
    img_size: int = 128,
):
    """
    Create a grid showing multiple difference heatmaps.

    Args:
        pairs: List of (patient_id, image_idx, baseline_path, seg_path)
        output_path: Where to save the grid
        n_cols: Number of columns in grid
        colormap: Colormap for difference
        threshold: Threshold for difference visualization
        img_size: Size to resize each thumbnail
    """
    n_samples = len(pairs)
    n_rows = (n_samples + n_cols - 1) // n_cols

    # Create canvas
    grid_w = n_cols * img_size
    grid_h = n_rows * img_size
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    for idx, (patient_id, image_idx, baseline_path, seg_path) in enumerate(pairs):
        row = idx // n_cols
        col = idx % n_cols

        # Load and compute difference
        baseline_gray = load_image_grayscale(baseline_path)
        seg_gray = load_image_grayscale(seg_path)
        diff_map = compute_difference_map(baseline_gray, seg_gray, normalize=True)

        # Threshold and colormap
        diff_map[diff_map < threshold] = 0
        diff_uint8 = (diff_map * 255).astype(np.uint8)

        cv_colormap = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
        diff_colored = cv2.applyColorMap(diff_uint8, cv_colormap)
        diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

        # Resize
        diff_thumb = cv2.resize(diff_colored, (img_size, img_size))

        # Place in grid
        y0, y1 = row * img_size, (row + 1) * img_size
        x0, x1 = col * img_size, (col + 1) * img_size
        grid[y0:y1, x0:x1] = diff_thumb

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(grid).save(output_path)
    print(f"[INFO] Saved summary grid to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visual Difference Heatmap Generator")

    # Single image mode
    parser.add_argument("--baseline_img", type=str, help="Path to single baseline image")
    parser.add_argument("--seg_img", type=str, help="Path to single seg image")
    parser.add_argument("--output", type=str, help="Output path for single comparison")

    # Batch mode
    parser.add_argument("--baseline_dir", type=str, help="Directory with baseline images")
    parser.add_argument("--seg_dir", type=str, help="Directory with OmniGen+Seg images")
    parser.add_argument("--output_dir", type=str, help="Output directory for batch comparisons")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of random samples for batch mode (default: all)")

    # Options
    parser.add_argument("--colormap", type=str, default="jet",
                        choices=["jet", "hot", "viridis", "inferno", "plasma", "magma"],
                        help="Colormap for difference heatmap")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Threshold for difference (values below are zeroed)")
    parser.add_argument("--create_grid", action="store_true",
                        help="Also create a summary grid of all differences")
    parser.add_argument("--compute_stats", action="store_true",
                        help="Compute and print global statistics")

    args = parser.parse_args()

    # Single image mode
    if args.baseline_img and args.seg_img:
        if not args.output:
            args.output = "comparison.png"

        print(f"[INFO] Single image comparison mode")
        print(f"[INFO] Baseline: {args.baseline_img}")
        print(f"[INFO] OmniGen+Seg: {args.seg_img}")

        create_simple_comparison(
            args.baseline_img,
            args.seg_img,
            args.output,
            colormap=args.colormap,
            threshold=args.threshold,
        )

        print(f"[INFO] Saved comparison to: {args.output}")
        return

    # Batch mode
    if args.baseline_dir and args.seg_dir:
        if not args.output_dir:
            args.output_dir = "diff_heatmaps"

        print(f"[INFO] Batch comparison mode")
        print(f"[INFO] Baseline dir: {args.baseline_dir}")
        print(f"[INFO] OmniGen+Seg dir: {args.seg_dir}")

        # Find matching pairs
        pairs = find_matching_images(args.baseline_dir, args.seg_dir)
        print(f"[INFO] Found {len(pairs)} matching image pairs")

        if len(pairs) == 0:
            print("[ERROR] No matching pairs found!")
            return

        # Sample if requested
        if args.n_samples and len(pairs) > args.n_samples:
            import random
            random.seed(42)
            pairs = random.sample(pairs, args.n_samples)
            print(f"[INFO] Sampled {len(pairs)} pairs for visualization")

        # Compute global statistics if requested
        if args.compute_stats:
            print("\n[INFO] Computing global statistics...")
            stats = compute_global_statistics(args.baseline_dir, args.seg_dir, args.n_samples)

            print("\n" + "=" * 60)
            print("GLOBAL DIFFERENCE STATISTICS")
            print("=" * 60)
            print(f"Number of pairs analyzed: {stats['n_pairs']}")
            print(f"\nPixel-level differences:")
            print(f"  Mean: {stats['pixel_diff_mean']:.4f}")
            print(f"  Std:  {stats['pixel_diff_std']:.4f}")
            print(f"  Median: {stats['pixel_diff_median']:.4f}")
            print(f"  95th percentile: {stats['pixel_diff_p95']:.4f}")
            print(f"  99th percentile: {stats['pixel_diff_p99']:.4f}")
            print(f"\nPer-image mean differences:")
            print(f"  Mean: {stats['image_mean_diff']['mean']:.4f}")
            print(f"  Std:  {stats['image_mean_diff']['std']:.4f}")
            print(f"  Range: [{stats['image_mean_diff']['min']:.4f}, {stats['image_mean_diff']['max']:.4f}]")
            print("=" * 60 + "\n")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Process each pair
        for patient_id, image_idx, baseline_path, seg_path in tqdm(pairs, desc="Creating heatmaps"):
            output_path = os.path.join(
                args.output_dir,
                f"{patient_id}_{image_idx}_diff.png"
            )

            create_simple_comparison(
                baseline_path,
                seg_path,
                output_path,
                colormap=args.colormap,
                threshold=args.threshold,
            )

        print(f"\n[INFO] Saved {len(pairs)} comparison images to: {args.output_dir}")

        # Create summary grid if requested
        if args.create_grid:
            grid_samples = pairs[:min(64, len(pairs))]  # Max 64 for grid
            grid_path = os.path.join(args.output_dir, "_summary_grid.png")
            create_summary_grid(
                grid_samples,
                grid_path,
                colormap=args.colormap,
                threshold=args.threshold,
            )

        return

    # No valid arguments
    parser.print_help()


if __name__ == "__main__":
    main()
