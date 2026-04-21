import argparse
import json
import os
from pathlib import Path

import torch
import torchmetrics
from einops import rearrange
from PIL import Image
from skimage.exposure import match_histograms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm


def get_image_paths(gt_dir, pred_dir, src_dir):
    """
    Get matching image paths from ground truth, prediction, and source directories.
    Assuming the structure matches what was used for OmniGen evaluation.
    """
    gt_paths = []
    pred_paths = []
    src_paths = []

    # Iterate over patient directories in pred_dir
    for patient_dir in sorted(os.listdir(pred_dir)):
        patient_path = os.path.join(pred_dir, patient_dir)
        if not os.path.isdir(patient_path):
            continue

        gt_patient_path = os.path.join(gt_dir, patient_dir)
        if not os.path.exists(gt_patient_path):
            print(f"Warning: GT directory {gt_patient_path} not found.")
            continue

        for filename in sorted(os.listdir(patient_path)):
            if filename.endswith(".png"):
                pred_img_path = os.path.join(patient_path, filename)
                gt_img_path = os.path.join(gt_patient_path, filename)
                
                # Try to find exactly corresponding source image
                # In SV-DRR / OmniGen setup, usually the source is the AP view
                # This logic might need adjustment based on exact naming
                src_filename = "0000.png" # Assuming 0000.png is the source view
                src_img_path = os.path.join(src_dir, patient_dir, src_filename)

                if os.path.exists(gt_img_path) and os.path.exists(src_img_path):
                    gt_paths.append(gt_img_path)
                    pred_paths.append(pred_img_path)
                    src_paths.append(src_img_path)

    return gt_paths, pred_paths, src_paths


def calculate_metrics(gt_imgs, pred_imgs, device="cuda"):
    """
    Calculate SSIM, PSNR, LPIPS, FID for a list of GT and Pred images.
    Returns a dictionary of metrics.
    """
    to_tensor = transforms.ToTensor()
    # Normalize images for LPIPS and FID (-1 to 1 for LPIPS, 0-255 uint8 tensor for FID)
    # Actually Torchmetrics FID expects uint8 0-255
    # LPIPS expects float32 -1 to +1

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    total_ssim = 0
    total_psnr = 0
    count = 0

    print("Extracting features for FID and calculating LPIPS/SSIM/PSNR...")
    for gt_img_np, pred_img_np in tqdm(zip(gt_imgs, pred_imgs), total=len(gt_imgs)):
        # Calculate traditional metrics (skimage)
        # Ensure images are properly scaled for skimage metrics if they are float
        if gt_img_np.dtype != pred_img_np.dtype or gt_img_np.max() > 1.0 or pred_img_np.max() > 1.0:
            # Assuming inputs are 0-255 uint8 arrays for skimage
            _gt = gt_img_np
            _pred = pred_img_np
        else:
             _gt = (gt_img_np * 255).astype('uint8')
             _pred = (pred_img_np * 255).astype('uint8')

        curr_ssim = ssim(_gt, _pred, data_range=255, channel_axis=-1 if _gt.ndim == 3 else None)
        curr_psnr = psnr(_gt, _pred, data_range=255)
        
        total_ssim += curr_ssim
        total_psnr += curr_psnr
        count += 1

        # Convert to tensors
        # Ensure 3 channels for torchmetrics
        if _gt.ndim == 2:
            _gt = _gt[..., None].repeat(3, axis=-1)
        if _pred.ndim == 2:
            _pred = _pred[..., None].repeat(3, axis=-1)

        # PyTorch expects BxCxHxW
        gt_tensor_uint8 = torch.from_numpy(_gt).permute(2, 0, 1).unsqueeze(0).to(torch.uint8).to(device)
        pred_tensor_uint8 = torch.from_numpy(_pred).permute(2, 0, 1).unsqueeze(0).to(torch.uint8).to(device)
        
        # LPIPS expects float in [0, 1] when normalize=True (we let it handle the norm internally)
        gt_tensor_float = gt_tensor_uint8.float() / 255.0
        pred_tensor_float = pred_tensor_uint8.float() / 255.0

        # Calculate LPIPS
        lpips_metric.update(pred_tensor_float, gt_tensor_float)

        # Update FID
        fid_metric.update(gt_tensor_uint8, real=True)
        fid_metric.update(pred_tensor_uint8, real=False)

    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count
    
    print("Computing final LPIPS...")
    final_lpips = lpips_metric.compute().item()
    print("Computing final FID...")
    final_fid = fid_metric.compute().item()

    return {
        "SSIM": avg_ssim,
        "PSNR": avg_psnr,
        "LPIPS": final_lpips,
        "FID": final_fid
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose Metrics with Histogram Matching")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing Ground Truth images")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing Original Prediction images")
    parser.add_argument("--src_dir", type=str, required=True, help="Directory containing Source images")
    parser.add_argument("--output_dir", type=str, default="diagnostics_output", help="Directory to save comparison grids")
    parser.add_argument("--num_cases", type=int, default=5, help="Number of cases to save for visual comparison")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to process for faster diagnosis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Gathering image paths...")
    gt_paths, pred_paths, src_paths = get_image_paths(args.gt_dir, args.pred_dir, args.src_dir)
    
    if args.max_samples is not None and len(gt_paths) > args.max_samples:
        gt_paths = gt_paths[:args.max_samples]
        pred_paths = pred_paths[:args.max_samples]
        src_paths = src_paths[:args.max_samples]

    print(f"Found {len(gt_paths)} valid image pairs to evaluate.")

    if len(gt_paths) == 0:
        print("No valid image pairs found. Exiting.")
        return

    # To calculate metrics over the whole dataset, load images
    # Note: For large datasets, loading all images into RAM might be prohibitive.
    # In that case, modify to load in batches. Assuming dataset is manageable.
    
    original_preds = []
    matched_preds = []
    gts = []
    srcs_for_vis = []

    print("Loading images and executing Histogram Matching...")
    for i in tqdm(range(len(gt_paths))):
        gt_img = Image.open(gt_paths[i]).convert('RGB')
        pred_img = Image.open(pred_paths[i]).convert('RGB')
        
        # Load and store for visual comparison if needed
        import numpy as np
        gt_np = np.array(gt_img)
        pred_np = np.array(pred_img)

        # 1. Histogram Matching
        # match_histograms returns float64 if input is float, or float64 if input is int sometimes,
        # need to ensure it's cast properly. Let's cast to uint8
        matched_np = match_histograms(pred_np, gt_np, channel_axis=-1)
        matched_np = np.clip(matched_np, 0, 255).astype(np.uint8)

        # Store for full dataset metrics
        original_preds.append(pred_np)
        matched_preds.append(matched_np)
        gts.append(gt_np)

        if i < args.num_cases:
            try:
                src_img = Image.open(src_paths[i]).convert('RGB')
                src_np = np.array(src_img)
                srcs_for_vis.append(src_np)
            except Exception as e:
                print(f"Failed to load source image for visualization {src_paths[i]}: {e}")
                srcs_for_vis.append(np.zeros_like(gt_np))

    # 4. Save visual comparison grid for the first `num_cases` cases
    print(f"Saving top {args.num_cases} conceptual comparative visualisations...")
    for i in range(min(args.num_cases, len(srcs_for_vis))):
        # Create a 1x4 grid: Source | GT | Original Pred | Matched Pred
        # Concatenate horizontally
        grid_img_np = np.concatenate(
            (srcs_for_vis[i], gts[i], original_preds[i], matched_preds[i]), axis=1
        )
        grid_img = Image.fromarray(grid_img_np)
        
        # Draw labels (rudimentary way: add padding or just simple concat)
        # For simplicity, we just save the concat image. In a real script, drawing text is preferred.
        out_path = os.path.join(args.output_dir, f"comparison_case_{i+1:02d}.png")
        grid_img.save(out_path)
        print(f"Saved visualization to {out_path}")

    # 2. Recalculate metrics
    print("\n--- Calculating Metrics for [Original Pred vs GT] ---")
    orig_metrics = calculate_metrics(gts, original_preds, device=device)
    
    print("\n--- Calculating Metrics for [Matched Pred vs GT] ---")
    matched_metrics = calculate_metrics(gts, matched_preds, device=device)

    # 3. Print the comparison
    print("\n=======================================================")
    print("                    DIAGNOSTIC RESULTS                  ")
    print("=======================================================")
    print(f"{'Metric':<10} | {'Original Pred':<15} | {'Matched Pred':<15}")
    print("-" * 55)
    for k in orig_metrics.keys():
        print(f"{k:<10} | {orig_metrics[k]:<15.4f} | {matched_metrics[k]:<15.4f}")
    print("=======================================================")

    # Optional: save to json
    results = {
        "original": orig_metrics,
        "matched": matched_metrics
    }
    with open(os.path.join(args.output_dir, "diagnostic_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        print(f"Saved metrics to {os.path.join(args.output_dir, 'diagnostic_metrics.json')}")

if __name__ == "__main__":
    main()
