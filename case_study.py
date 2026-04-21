#!/usr/bin/env python3
"""
case_study.py — CXR Multi-View Generation Case Study Tool
=========================================================

Functionality:
  1. Reads the test dataset JSONL (source/target paths, prompts).
  2. Scans the prediction directory to find matching generated images.
  3. Selects N random cases (match found) or all cases.
  4. Creates a structured 'case_study' folder with SYMLINKS to source/target/pred files.
  5. Computes per-image metrics (SSIM, PSNR, LPIPS) for each case.
  6. Generates a 'report.txt' with the prompt and metrics.

Usage:
  python case_study.py \
    --jsonl_path /path/to/test.jsonl \
    --pred_dir /path/to/generated_images \
    --output_dir case_study_results \
    --num_cases 50

Dependencies:
  - torch (for LPIPS)
  - scikit-image (for SSIM/PSNR)
  - lpips
  - tqdm
"""

import os
import sys
import json
import random
import argparse
import logging
import shutil
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Metrics Helpers ---
class MetricsCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info(f"Initializing LPIPS on {device}...")
        # LPIPS demands input in range [-1, 1]
        self.lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    def load_image_gray(self, path):
        """Load image as grayscale numpy array [H, W] in range [0, 255] for SSIM/PSNR."""
        img = Image.open(path).convert('L')
        img = img.resize((256, 256), Image.BICUBIC)
        return np.array(img)

    def load_image_rgb_tensor(self, path):
        """Load image as RGB torch tensor [1, 3, H, W] in range [-1, 1] for LPIPS."""
        img = Image.open(path).convert('RGB')
        img = img.resize((256, 256), Image.BICUBIC)
        # Transform: [0, 255] -> [0, 1] -> [-1, 1]
        img_np = (np.array(img).astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def compute(self, target_path, pred_path):
        # 1. SSIM & PSNR (Grayscale, uint8-like range handling)
        gt_gray = self.load_image_gray(target_path)
        pred_gray = self.load_image_gray(pred_path)
        
        # scikit-image's SSIM/PSNR data_range=255
        score_ssim = ssim(gt_gray, pred_gray, data_range=255)
        score_psnr = psnr(gt_gray, pred_gray, data_range=255)

        # 2. LPIPS (RGB, -1 to 1)
        gt_rgb = self.load_image_rgb_tensor(target_path)
        pred_rgb = self.load_image_rgb_tensor(pred_path)
        
        with torch.no_grad():
            score_lpips = self.lpips_fn(gt_rgb, pred_rgb).item()

        return {
            "SSIM": score_ssim,
            "PSNR": score_psnr,
            "LPIPS": score_lpips
        }

# --- Main Logic ---

def derive_relative_path(target_path):
    """
    Derive the relative path (PatientID/ViewID.png) from the absolute target path.
    Assumes structure: .../Category/PatientID/Filename.png 
    We just take the last two components.
    """
    parts = target_path.replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 2:
        return os.path.basename(target_path)
    return os.path.join(parts[-2], parts[-1])

def main():
    parser = argparse.ArgumentParser(description="Generate a case study report for CXR view synthesis.")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing generated outputs")
    parser.add_argument("--output_dir", type=str, default="case_study", help="Output directory for symlinks and reports")
    parser.add_argument("--num_cases", type=int, default=50, help="Number of cases to sample (-1 for all)")
    args = parser.parse_args()

    # 1. Load JSONL Data
    logger.info(f"Loading annotations from {args.jsonl_path}...")
    valid_entries = []
    
    if not os.path.exists(args.jsonl_path):
        logger.error(f"JSONL file not found: {args.jsonl_path}")
        sys.exit(1)
        
    if not os.path.exists(args.pred_dir):
        logger.error(f"Prediction dir not found: {args.pred_dir}")
        sys.exit(1)

    with open(args.jsonl_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                
                # Check required fields
                if 'input_images' not in entry or 'output_image' not in entry:
                    continue
                    
                source_path = entry['input_images'][0]
                target_path = entry['output_image']
                
                # Derive pred path: pred_dir + PatientID/ViewID.png
                rel_path = derive_relative_path(target_path)
                pred_path = os.path.join(args.pred_dir, rel_path)
                
                # Check if generated image actually exists
                if os.path.exists(pred_path):
                    valid_entries.append({
                        "source": source_path,
                        "target": target_path,
                        "pred": pred_path,
                        "instruction": entry.get("instruction", "No instruction provided")
                    })
            except Exception as e:
                logger.warning(f"Skipping malformed line: {e}")

    total_matched = len(valid_entries)
    logger.info(f"Found {total_matched} matched cases (generated file exists).")

    if total_matched == 0:
        logger.warning("No matched cases found. Check paths and prediction directory structure.")
        sys.exit(0)

    # 2. Sample Cases
    if args.num_cases > 0 and args.num_cases < total_matched:
        selected_cases = random.sample(valid_entries, args.num_cases)
        logger.info(f"Randomly selected {len(selected_cases)} cases.")
    else:
        selected_cases = valid_entries
        logger.info(f"Processing all {len(selected_cases)} cases.")

    # 3. Setup Metrics
    metrics_calc = MetricsCalculator()

    # 4. Process Cases
    if os.path.exists(args.output_dir):
        logger.warning(f"Output directory {args.output_dir} exists. New cases will be added/overwritten.")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Processing cases...")
    for item in tqdm(selected_cases):
        try:
            # 4.1 Create Case Directory
            # Use PatientID_Filename as folder name
            rel_path = derive_relative_path(item['target']) # e.g. LIDC-IDRI-0251/0001.png
            folder_name = rel_path.replace("/", "_").replace("\\", "_").rsplit(".", 1)[0]
            case_dir = os.path.join(args.output_dir, folder_name)
            os.makedirs(case_dir, exist_ok=True)

            # 4.2 Create Symlinks (Safe removal if exists)
            def create_symlink(src, link_name):
                dst = os.path.join(case_dir, link_name)
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(src), dst)

            create_symlink(item['source'], "source.png")
            create_symlink(item['target'], "target.png")
            create_symlink(item['pred'], "pred.png")

            # 4.3 Compute Metrics
            metrics = metrics_calc.compute(item['target'], item['pred'])

            # 4.4 Write Report
            report_path = os.path.join(case_dir, "report.txt")
            with open(report_path, "w") as f:
                f.write("[Instruction]\n")
                f.write(f"{item['instruction']}\n\n")
                
                f.write("[Metrics]\n")
                f.write(f"SSIM : {metrics['SSIM']:.4f}\n")
                f.write(f"PSNR : {metrics['PSNR']:.4f}\n")
                f.write(f"LPIPS: {metrics['LPIPS']:.4f}\n")
                
                f.write("\n[Original Paths]\n")
                f.write(f"Source: {item['source']}\n")
                f.write(f"Target: {item['target']}\n")
                f.write(f"Pred  : {item['pred']}\n")

        except Exception as e:
            logger.error(f"Error processing case {item.get('target', 'unknown')}: {e}")

    logger.info(f"Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
