#!/usr/bin/env python3
"""
ROI self-consistency metrics for joint image-mask co-generation.

The predicted mask defines the ROI.  These metrics compare generated images to
GT edited images inside and outside the model-induced ROI; they are not
segmentation metrics and do not require paired GT masks.
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass(frozen=True)
class Sample:
    sample_id: str
    gt_path: str
    gen_path: str
    mask_path: str


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_no} of {path}: {exc}") from exc
    return records


def derive_relative_image_path(gt_path: str) -> str:
    parts = gt_path.replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot derive patient/view relative path from {gt_path}")
    return os.path.join(parts[-2], parts[-1])


def derive_generated_path(gt_path: str, inference_dir: str) -> str:
    return os.path.join(inference_dir, derive_relative_image_path(gt_path))


def derive_mask_path(gt_path: str, inference_dir: str) -> str:
    rel = derive_relative_image_path(gt_path)
    stem, _ = os.path.splitext(rel)
    return os.path.join(inference_dir, "masks", f"{stem}.npz")


def sample_id_candidates(record: Dict) -> List[str]:
    gt_path = record["output_image"]
    parts = gt_path.replace("\\", "/").rstrip("/").split("/")
    patient = parts[-2] if len(parts) >= 2 else ""
    stem = os.path.splitext(parts[-1])[0] if parts else ""
    candidates = [
        record.get("sample_id", ""),
        record.get("id", ""),
        patient,
        stem,
        f"{patient}_{stem}" if patient and stem else "",
    ]
    return [str(x) for x in candidates if x is not None and str(x) != ""]


def canonical_sample_id(record: Dict) -> str:
    if record.get("sample_id"):
        return str(record["sample_id"])
    if record.get("id"):
        return str(record["id"])
    gt_path = record["output_image"]
    parts = gt_path.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2:
        patient = parts[-2]
        stem = os.path.splitext(parts[-1])[0]
        return f"{patient}_{stem}"
    return os.path.splitext(os.path.basename(gt_path))[0]


def select_records(records: Sequence[Dict], sample_ids: Optional[str], max_samples: Optional[int]) -> List[Dict]:
    if sample_ids:
        requested = {x.strip() for x in sample_ids.split(",") if x.strip()}
        selected = [
            rec for rec in records
            if requested.intersection(sample_id_candidates(rec))
        ]
        found = {sid for rec in selected for sid in sample_id_candidates(rec)}
        missing = sorted(requested - found)
        if missing:
            print(f"WARNING: requested sample_ids not found: {', '.join(missing)}")
        return selected
    if max_samples is not None:
        return list(records[:max_samples])
    return list(records)


def build_samples(records: Sequence[Dict], inference_dir: str) -> List[Sample]:
    return [
        Sample(
            sample_id=canonical_sample_id(rec),
            gt_path=rec["output_image"],
            gen_path=derive_generated_path(rec["output_image"], inference_dir),
            mask_path=derive_mask_path(rec["output_image"], inference_dir),
        )
        for rec in records
    ]


def read_gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image.astype(np.float32)


def load_mask_npz(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predicted mask NPZ not found: {path}")
    with np.load(path) as data:
        key = "mask" if "mask" in data.files else data.files[0]
        mask = np.asarray(data[key])
    mask = np.squeeze(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask array after squeeze, got {mask.shape} from {path}")
    if mask.shape[0] != 10 and mask.shape[-1] == 10:
        mask = np.moveaxis(mask, -1, 0)
    if mask.shape[0] != 10:
        raise ValueError(f"Expected 10 mask channels, got {mask.shape} from {path}")
    return mask.astype(np.float32)


def make_roi(mask_10ch: np.ndarray, height: int, width: int, threshold: float) -> np.ndarray:
    union = np.sum(mask_10ch, axis=0) > threshold
    union = union.astype(np.uint8)
    if union.shape != (height, width):
        union = cv2.resize(union, (width, height), interpolation=cv2.INTER_NEAREST)
    return union.astype(bool)


def masked_psnr(gt: np.ndarray, gen: np.ndarray, roi: np.ndarray) -> float:
    if not np.any(roi):
        return float("nan")
    diff = gt[roi] - gen[roi]
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def masked_ssim_from_map(ssim_map: np.ndarray, roi: np.ndarray) -> float:
    if not np.any(roi):
        return float("nan")
    return float(np.mean(ssim_map[roi]))


def compute_sample_metrics(sample: Sample, mask_threshold: float) -> Dict[str, float]:
    gt = read_gray(sample.gt_path)
    gen = read_gray(sample.gen_path)
    if gt.shape != gen.shape:
        gen = cv2.resize(gen, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)

    mask = load_mask_npz(sample.mask_path)
    roi = make_roi(mask, gt.shape[0], gt.shape[1], mask_threshold)
    background = ~roi

    global_ssim, ssim_map = structural_similarity(
        gt,
        gen,
        data_range=255,
        full=True,
    )
    global_psnr = peak_signal_noise_ratio(gt, gen, data_range=255)

    return {
        "sample_id": sample.sample_id,
        "gt_path": sample.gt_path,
        "gen_path": sample.gen_path,
        "mask_path": sample.mask_path,
        "roi_pixels": int(np.sum(roi)),
        "background_pixels": int(np.sum(background)),
        "roi_fraction": float(np.mean(roi)),
        "global_psnr": float(global_psnr),
        "global_ssim": float(global_ssim),
        "roi_inside_psnr": masked_psnr(gt, gen, roi),
        "roi_inside_ssim": masked_ssim_from_map(ssim_map, roi),
        "roi_outside_psnr": masked_psnr(gt, gen, background),
        "roi_outside_ssim": masked_ssim_from_map(ssim_map, background),
    }


def finite_values(rows: Sequence[Dict], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def summarize(rows: Sequence[Dict]) -> Dict:
    metric_keys = [
        "global_psnr",
        "global_ssim",
        "roi_inside_psnr",
        "roi_inside_ssim",
        "roi_outside_psnr",
        "roi_outside_ssim",
        "roi_fraction",
    ]
    summary = {
        "num_samples": len(rows),
        "metrics": {},
        "interpretation_note": (
            "Predicted masks define ROI regions for local image-fidelity analysis. "
            "These are self-consistency metrics, not Dice/IoU segmentation metrics "
            "and not causal proof of anatomical reasoning."
        ),
    }
    for key in metric_keys:
        values = finite_values(rows, key)
        summary["metrics"][key] = {
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "n": int(values.size),
        }
    return summary


def write_csv(rows: Sequence[Dict], path: str):
    fieldnames = [
        "sample_id",
        "gt_path",
        "gen_path",
        "mask_path",
        "roi_pixels",
        "background_pixels",
        "roi_fraction",
        "global_psnr",
        "global_ssim",
        "roi_inside_psnr",
        "roi_inside_ssim",
        "roi_outside_psnr",
        "roi_outside_ssim",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute global and predicted-ROI local fidelity metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl_path", required=True, help="Test JSONL used for inference.")
    parser.add_argument(
        "--inference_dir",
        "--results_dir",
        dest="inference_dir",
        required=True,
        help="Directory containing generated images and masks/ from test_joint_mask.py.",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis_roi_metrics",
        help="Directory for sample_metrics.csv and dataset_summary.json.",
    )
    parser.add_argument(
        "--sample_ids",
        default=None,
        help="Optional comma-separated IDs. Matches sample_id/id, patient ID, view stem, or patient_stem.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Process first N samples when --sample_ids is not provided.",
    )
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="ROI threshold after summing channels.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_jsonl(args.jsonl_path)
    selected = select_records(records, args.sample_ids, args.max_samples)
    samples = build_samples(selected, args.inference_dir)
    if not samples:
        raise RuntimeError("No samples selected for ROI analysis.")

    rows = []
    skipped = 0
    for sample in samples:
        try:
            rows.append(compute_sample_metrics(sample, args.mask_threshold))
        except Exception as exc:
            skipped += 1
            print(f"WARNING: skipping {sample.sample_id}: {exc}")

    if not rows:
        raise RuntimeError("No valid samples were available for ROI analysis.")

    csv_path = os.path.join(args.output_dir, "sample_metrics.csv")
    summary_path = os.path.join(args.output_dir, "dataset_summary.json")
    write_csv(rows, csv_path)
    summary = summarize(rows)
    summary["num_requested"] = len(samples)
    summary["num_skipped"] = skipped
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote per-sample metrics: {csv_path}")
    print(f"Wrote dataset summary: {summary_path}")
    if skipped:
        print(f"Skipped {skipped} samples due to missing/corrupt inputs.")


if __name__ == "__main__":
    main()
