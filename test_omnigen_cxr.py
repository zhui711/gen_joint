#!/usr/bin/env python3
"""
test_omnigen_cxr.py — CXR Multi-View Generation: Batch Inference & Evaluation
==============================================================================

Architecture (Map-Reduce):
  Phase 1 — Generation (Parallel):
    Main process loads JSONL → splits into N chunks → mp.spawn on N GPUs.
    Each worker: load OmniGenPipeline + LoRA → batch inference → save PNGs.

  Phase 2 — Synchronization:
    mp.spawn joins all workers.

  Phase 3 — Evaluation (Serial):
    Main process: load (Generated, GT) image pairs → compute SSIM, PSNR,
    LPIPS, FID → print & save metrics_report.json.

Evaluation Details:
  The SV-DRR repository (CVPR/MICCAI) does NOT open-source its metric
  computation code.  We therefore use the standard libraries that are
  universally adopted in medical-imaging generation papers to faithfully
  reproduce the four metrics reported in SV-DRR Table 1:

    • SSIM  — scikit-image  structural_similarity   (grayscale, data_range=255)
    • PSNR  — scikit-image  peak_signal_noise_ratio  (grayscale, data_range=255)
    • LPIPS — lpips package  (AlexNet backbone, RGB images in [-1, 1])
    • FID   — torchmetrics   FrechetInceptionDistance (InceptionV3 2048-d features)

  All comparisons are performed on 256×256 images.  Generated images (RGB
  from OmniGen) are converted to grayscale for SSIM/PSNR; GT images
  (originally grayscale) are replicated to 3-channel for LPIPS/FID.

Usage:
  # Quick sanity check (first 100 samples, single GPU)
  python test_omnigen_cxr.py --jsonl_path ... --output_dir ... --max_samples 100

  # Full evaluation on 2 GPUs
  CUDA_VISIBLE_DEVICES=2,3 python test_omnigen_cxr.py --jsonl_path ... --output_dir ... --num_gpus 2
"""

import os
import sys
import gc
import json
import math
import time
import logging
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm


# ============================================================
# Constants
# ============================================================
IMAGE_SIZE = 256  # Fixed CXR image resolution


# ============================================================
# Utility Functions
# ============================================================

def derive_save_path(gt_path: str, output_dir: str) -> str:
    """
    Map a Ground Truth absolute path to a generated-image save path.

    Example:
      gt_path   = ".../img_complex_fb_256/LIDC-IDRI-0251/0001.png"
      output_dir = ".../output/cxr_synth_results"
      returns   → ".../output/cxr_synth_results/LIDC-IDRI-0251/0001.png"

    This deterministic mapping is also used by the evaluation function to
    locate the generated counterpart for every GT, ensuring pairwise
    alignment without any external manifest file.
    """
    parts = gt_path.replace("\\", "/").rstrip("/").split("/")
    relative = os.path.join(parts[-2], parts[-1])  # patient_id/view_id.png
    return os.path.join(output_dir, relative)


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file, skipping blank or malformed lines."""
    data = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping malformed line {line_no} in {path}: {e}")
    return data


def setup_logging(output_dir: str, rank: Optional[int] = None):
    """Configure logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_gpu{rank}" if rank is not None else ""
    log_file = os.path.join(output_dir, f"run{suffix}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


# ============================================================
# Phase 1: Generation — Worker Function
# ============================================================

def generation_worker(rank: int, num_gpus: int, chunks: List[List[Dict]], args_dict: dict):
    """
    Per-GPU worker process.
    Loads the full OmniGenPipeline (+ optional LoRA) on cuda:{rank},
    iterates over its data chunk in batches, and saves generated images.
    """
    # ---- Rebuild args from dict (Namespace is picklable, but dict is safer) ----
    args = argparse.Namespace(**args_dict)

    device = torch.device(f"cuda:{rank}")
    setup_logging(args.output_dir, rank=rank)
    logger = logging.getLogger(f"worker-{rank}")

    logger.info(f"[GPU {rank}] Starting — device={device}, samples={len(chunks[rank])}")

    # ---- Load model ----
    from OmniGen import OmniGenPipeline                         # import inside worker (fresh process)
    pipe = OmniGenPipeline.from_pretrained(args.model_path)
    if args.lora_path:
        logger.info(f"[GPU {rank}] Merging LoRA from {args.lora_path}")
        pipe.merge_lora(args.lora_path)
    pipe.to(device)

    my_data = chunks[rank]
    batch_size = args.batch_size
    success_count = 0
    fail_count = 0

    num_batches = math.ceil(len(my_data) / batch_size)
    pbar = tqdm(range(num_batches), desc=f"[GPU {rank}]", position=rank, leave=True)

    for batch_idx in pbar:
        batch_start = batch_idx * batch_size
        batch = my_data[batch_start : batch_start + batch_size]

        try:
            # ---- Assemble batch inputs with strict ordering ----
            prompts: List[str] = [item["instruction"] for item in batch]
            input_imgs: List[List[str]] = [item["input_images"] for item in batch]

            # ------------------------------------------------------------------
            # OmniGen API note:
            #   use_input_image_size_as_output=True  requires prompt to be a
            #   single str (not a list).  Since all our CXR images are exactly
            #   256×256, we pass height/width explicitly and set that flag to
            #   False so that true batch inference is possible.
            #   When batch_size==1, the single-item path is equivalent.
            # ------------------------------------------------------------------
            if len(prompts) == 1:
                # Single-item batch: can safely use use_input_image_size_as_output
                outputs = pipe(
                    prompt=prompts[0],
                    input_images=input_imgs[0],
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    num_inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    img_guidance_scale=args.img_guidance_scale,
                    use_input_image_size_as_output=True,
                    use_kv_cache=True,
                    offload_kv_cache=False,
                    separate_cfg_infer=False,
                    offload_model=False,
                    seed=args.seed,
                )
            else:
                # True batch inference: explicit dimensions
                outputs = pipe(
                    prompt=prompts,
                    input_images=input_imgs,
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    num_inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    img_guidance_scale=args.img_guidance_scale,
                    use_input_image_size_as_output=False,
                    use_kv_cache=True,
                    offload_kv_cache=False,
                    separate_cfg_infer=False,
                    offload_model=False,
                    seed=args.seed,
                )

            # ---- Strict zip: bind output images to metadata ----
            assert len(outputs) == len(batch), (
                f"Pipeline returned {len(outputs)} images for batch of {len(batch)}"
            )
            for img, item in zip(outputs, batch):
                save_path = derive_save_path(item["output_image"], args.output_dir)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img.save(save_path)
                success_count += 1

        except Exception as e:
            logger.error(
                f"[GPU {rank}] Batch {batch_idx} error (indices {batch_start}-"
                f"{batch_start + len(batch) - 1}): {e}",
                exc_info=True,
            )
            fail_count += len(batch)
            torch.cuda.empty_cache()
            gc.collect()

        pbar.set_postfix(ok=success_count, fail=fail_count)

    # ---- Save per-GPU stats ----
    stats = {
        "rank": rank,
        "total": len(my_data),
        "success": success_count,
        "fail": fail_count,
    }
    stats_path = os.path.join(args.output_dir, f"gen_stats_gpu{rank}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(
        f"[GPU {rank}] Done: {success_count}/{len(my_data)} success, {fail_count} failed"
    )


# ============================================================
# Phase 3: Evaluation
# ============================================================

def evaluate(args, all_data: List[Dict]):
    """
    Compute SSIM, PSNR, LPIPS, FID over all generated-vs-GT pairs.

    Metric library choices (aligned with common medical imaging practice):
      • SSIM/PSNR — scikit-image (the de-facto standard in radiology papers).
        Computed on **grayscale** uint8 images with data_range=255.
      • LPIPS — original lpips package by Zhang et al. (AlexNet backbone).
        Computed on **RGB** float tensors normalized to [-1, 1].
      • FID — torchmetrics FrechetInceptionDistance (InceptionV3, 2048-d).
        Computed on **RGB** uint8 tensors [0, 255].

    For grayscale CXR images, we replicate the single channel to 3 channels
    when computing LPIPS and FID, following standard practice in the field.
    """
    try:
        from skimage.metrics import structural_similarity as skimage_ssim
        from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
    except ImportError:
        print("ERROR: scikit-image is required.  pip install scikit-image")
        return None

    try:
        import lpips as lpips_pkg
    except ImportError:
        print("ERROR: lpips is required.  pip install lpips")
        return None

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("ERROR: torchmetrics[image] is required.  pip install torchmetrics[image]")
        return None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Evaluation Phase — device={device}")
    print(f"{'='*60}")

    # ---- Collect valid pairs ----
    valid_pairs: List[Tuple[str, str]] = []  # (gen_path, gt_path)
    for sample in all_data:
        gt_path = sample["output_image"]
        print("gt_path:",gt_path)
        gen_path = derive_save_path(gt_path, args.output_dir)
        print("gen_path:",gen_path)
        if os.path.exists(gen_path) and os.path.exists(gt_path):
            valid_pairs.append((gen_path, gt_path))

    num_skipped = len(all_data) - len(valid_pairs)
    print(f"  Valid pairs : {len(valid_pairs)}")
    print(f"  Skipped     : {num_skipped}")
    if len(valid_pairs) == 0:
        print("  No valid pairs found — skipping evaluation.")
        return None

    # ---- Initialise metrics ----
    lpips_fn = lpips_pkg.LPIPS(net="alex").to(device).eval()
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    ssim_scores: List[float] = []
    psnr_scores: List[float] = []
    lpips_scores: List[float] = []

    # ---- Per-patient accumulators for fine-grained reporting ----
    patient_metrics: Dict[str, Dict[str, List[float]]] = {}

    # ---- Batched evaluation (for LPIPS / FID efficiency) ----
    EVAL_BATCH = 32
    gt_fid_buf: List[torch.Tensor] = []
    gen_fid_buf: List[torch.Tensor] = []
    gt_lpips_buf: List[torch.Tensor] = []
    gen_lpips_buf: List[torch.Tensor] = []

    def _flush_fid():
        """Push accumulated buffers into torchmetrics FID state."""
        nonlocal gt_fid_buf, gen_fid_buf
        if gt_fid_buf:
            fid_metric.update(torch.stack(gt_fid_buf).to(device), real=True)
            fid_metric.update(torch.stack(gen_fid_buf).to(device), real=False)
            gt_fid_buf, gen_fid_buf = [], []

    def _flush_lpips():
        """Compute LPIPS for the accumulated buffer and extend scores."""
        nonlocal gt_lpips_buf, gen_lpips_buf
        if gt_lpips_buf:
            with torch.no_grad():
                gt_batch = torch.stack(gt_lpips_buf).to(device)
                gen_batch = torch.stack(gen_lpips_buf).to(device)
                d = lpips_fn(gen_batch, gt_batch)          # [B, 1, 1, 1]
                lpips_scores.extend(d.view(-1).cpu().tolist())
            gt_lpips_buf, gen_lpips_buf = [], []

    # ---- Main evaluation loop ----
    for idx, (gen_path, gt_path) in enumerate(tqdm(valid_pairs, desc="Evaluating")):
        try:
            gen_pil = Image.open(gen_path).convert("RGB")
            gt_pil = Image.open(gt_path).convert("RGB")   # grayscale → 3-ch

            # ---- SSIM & PSNR (grayscale, uint8, data_range=255) ----
            gen_gray = np.array(gen_pil.convert("L"))      # [H, W] uint8
            gt_gray = np.array(gt_pil.convert("L"))

            ssim_val = skimage_ssim(gt_gray, gen_gray, data_range=255)
            psnr_val = skimage_psnr(gt_gray, gen_gray, data_range=255)
            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)

            # ---- Patient-level tracking ----
            patient_id = gt_path.replace("\\", "/").split("/")[-2]
            if patient_id not in patient_metrics:
                patient_metrics[patient_id] = {"ssim": [], "psnr": [], "lpips": []}
            patient_metrics[patient_id]["ssim"].append(ssim_val)
            patient_metrics[patient_id]["psnr"].append(psnr_val)

            # ---- Prepare tensors for batched LPIPS ----
            gen_np = np.array(gen_pil)   # [H, W, 3] uint8
            gt_np = np.array(gt_pil)

            # LPIPS: float32, [-1, 1], [3, H, W]
            gen_lpips_buf.append(
                torch.from_numpy(gen_np).permute(2, 0, 1).float() / 127.5 - 1.0
            )
            gt_lpips_buf.append(
                torch.from_numpy(gt_np).permute(2, 0, 1).float() / 127.5 - 1.0
            )

            # FID: uint8, [0, 255], [3, H, W]
            gen_fid_buf.append(torch.from_numpy(gen_np).permute(2, 0, 1))
            gt_fid_buf.append(torch.from_numpy(gt_np).permute(2, 0, 1))

            # ---- Flush buffers when full ----
            if len(gt_fid_buf) >= EVAL_BATCH:
                _flush_lpips()
                _flush_fid()

                # Attach LPIPS to patient-level tracking
                # (the last EVAL_BATCH lpips_scores correspond to the last EVAL_BATCH pairs)
                start_idx = idx + 1 - EVAL_BATCH
                for offset in range(EVAL_BATCH):
                    pair_idx = start_idx + offset
                    pid = valid_pairs[pair_idx][1].replace("\\", "/").split("/")[-2]
                    patient_metrics[pid]["lpips"].append(lpips_scores[pair_idx])

        except Exception as e:
            logging.warning(f"Eval error on {gen_path}: {e}")
            continue

    # ---- Flush remaining buffers ----
    remaining_count = len(gt_lpips_buf)
    _flush_lpips()
    _flush_fid()

    # Attach remaining LPIPS to patient-level tracking
    if remaining_count > 0:
        start_idx = len(valid_pairs) - remaining_count
        for offset in range(remaining_count):
            pair_idx = start_idx + offset
            if pair_idx < len(lpips_scores):
                pid = valid_pairs[pair_idx][1].replace("\\", "/").split("/")[-2]
                patient_metrics[pid]["lpips"].append(lpips_scores[pair_idx])

    # ---- Compute final aggregate metrics ----
    fid_score = float(fid_metric.compute().item()) if len(valid_pairs) > 0 else float("nan")

    results = {
        "SSIM": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        "PSNR": float(np.mean(psnr_scores)) if psnr_scores else 0.0,
        "LPIPS": float(np.mean(lpips_scores)) if lpips_scores else 0.0,
        "FID": fid_score,
        "num_evaluated": len(ssim_scores),
        "num_total": len(all_data),
        "num_skipped": num_skipped,
    }

    # ---- Per-patient summary ----
    per_patient = {}
    for pid, m in sorted(patient_metrics.items()):
        per_patient[pid] = {
            "SSIM": float(np.mean(m["ssim"])) if m["ssim"] else 0.0,
            "PSNR": float(np.mean(m["psnr"])) if m["psnr"] else 0.0,
            "LPIPS": float(np.mean(m["lpips"])) if m["lpips"] else 0.0,
            "count": len(m["ssim"]),
        }
    results["per_patient"] = per_patient

    # ---- Print report ----
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  SSIM  ↑ : {results['SSIM']:.4f}")
    print(f"  PSNR  ↑ : {results['PSNR']:.2f} dB")
    print(f"  LPIPS ↓ : {results['LPIPS']:.4f}")
    print(f"  FID   ↓ : {results['FID']:.2f}")
    print(f"  Evaluated : {results['num_evaluated']} / {results['num_total']}")
    print(f"{'='*60}")
    print("\n  Per-patient breakdown:")
    for pid, pm in per_patient.items():
        print(
            f"    {pid:20s}  SSIM={pm['SSIM']:.4f}  PSNR={pm['PSNR']:.2f}  "
            f"LPIPS={pm['LPIPS']:.4f}  (n={pm['count']})"
        )
    print()

    # ---- Save to disk ----
    report_path = os.path.join(args.output_dir, "metrics_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Report saved to {report_path}")

    return results


# ============================================================
# Main Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CXR Multi-View OmniGen — Batch Inference & Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- Model ----
    parser.add_argument(
        "--model_path", type=str, default="Shitao/OmniGen-v1",
        help="Path or HF repo for the base OmniGen model.",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None,
        help="Path to a LoRA adapter checkpoint directory (e.g. results/cxr_finetune_lora/checkpoints/0000500).",
    )

    # ---- Data ----
    parser.add_argument(
        "--jsonl_path", type=str, required=True,
        help="Path to the test JSONL file (e.g. cxr_synth_anno_test.jsonl).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save generated images and metrics report.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="If set, only process the first N samples (for quick sanity checks).",
    )

    # ---- Inference ----
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for data-parallel generation.")
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of diffusion denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="CFG guidance scale.")
    parser.add_argument("--img_guidance_scale", type=float, default=2.0, help="Image guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")

    # ---- Workflow ----
    parser.add_argument(
        "--skip_generation", action="store_true",
        help="Skip generation phase; only run evaluation on existing images.",
    )
    parser.add_argument(
        "--skip_eval", action="store_true",
        help="Skip evaluation phase; only generate images.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    # ---- Load data ----
    print(f"Loading JSONL from {args.jsonl_path} ...")
    all_data = load_jsonl(args.jsonl_path)
    print(f"  Total samples in file: {len(all_data)}")

    if args.max_samples is not None and args.max_samples < len(all_data):
        all_data = all_data[: args.max_samples]
        print(f"  Sliced to first {args.max_samples} samples (--max_samples)")

    # ---- Save run config ----
    config_path = os.path.join(args.output_dir, "eval_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # ================================================================
    # Phase 1: Generation
    # ================================================================
    if not args.skip_generation:
        print(f"\n{'='*60}")
        print(f"  Generation Phase — {args.num_gpus} GPU(s), batch_size={args.batch_size}")
        print(f"  Output  → {args.output_dir}")
        print(f"{'='*60}\n")

        # Uniform data sharding: consecutive split keeps same-patient views
        # on the same GPU, improving filesystem cache locality for the shared
        # condition image (0000.png).
        chunk_size = math.ceil(len(all_data) / args.num_gpus)
        chunks = [
            all_data[i * chunk_size : (i + 1) * chunk_size]
            for i in range(args.num_gpus)
        ]
        for i, c in enumerate(chunks):
            print(f"  GPU {i}: {len(c)} samples")

        # Convert Namespace → dict for safe pickling across processes
        args_dict = vars(args)

        t0 = time.time()
        if args.num_gpus == 1:
            # Single-GPU: run directly without spawning (simpler debugging)
            generation_worker(0, 1, chunks, args_dict)
        else:
            mp.spawn(
                generation_worker,
                args=(args.num_gpus, chunks, args_dict),
                nprocs=args.num_gpus,
                join=True,
            )
        elapsed = time.time() - t0

        # Print generation summary
        print(f"\n  Generation completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        total_ok, total_fail = 0, 0
        for i in range(args.num_gpus):
            sp = os.path.join(args.output_dir, f"gen_stats_gpu{i}.json")
            if os.path.exists(sp):
                with open(sp) as f:
                    s = json.load(f)
                print(f"    GPU {s['rank']}: {s['success']}/{s['total']} ok, {s['fail']} fail")
                total_ok += s["success"]
                total_fail += s["fail"]
        print(f"    Total: {total_ok}/{total_ok + total_fail}")

    # ================================================================
    # Phase 3: Evaluation
    # ================================================================
    if not args.skip_eval:
        evaluate(args, all_data)

    print("\nAll done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
