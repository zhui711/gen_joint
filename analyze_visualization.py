#!/usr/bin/env python3
"""
Publication visualizations for joint image-mask co-generation.

This script does not evaluate segmentation quality.  The predicted 10-channel
mask is used only as an interpretability overlay for the generated image.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


ORGAN_NAMES = [
    "Left Lung",
    "Right Lung",
    "Heart",
    "Aorta",
    "Liver",
    "Stomach",
    "Trachea",
    "Ribs",
    "Vertebrae",
    "Upper Skeleton",
]

CHANNEL_COLORS = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [214, 39, 40],
        [148, 103, 189],
        [44, 160, 44],
        [140, 86, 75],
        [23, 190, 207],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
    ],
    dtype=np.float32,
) / 255.0


@dataclass(frozen=True)
class Sample:
    sample_id: str
    input_path: str
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
    input_images = record.get("input_images") or []
    gt_parts = gt_path.replace("\\", "/").rstrip("/").split("/")
    patient = gt_parts[-2] if len(gt_parts) >= 2 else ""
    stem = os.path.splitext(gt_parts[-1])[0] if gt_parts else ""
    candidates = [
        record.get("sample_id", ""),
        record.get("id", ""),
        patient,
        stem,
        f"{patient}_{stem}" if patient and stem else "",
        os.path.splitext(os.path.basename(gt_path))[0],
    ]
    if input_images:
        inp = input_images[0].replace("\\", "/").rstrip("/").split("/")
        if len(inp) >= 2:
            candidates.append(f"{inp[-2]}_{os.path.splitext(inp[-1])[0]}")
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


def select_records(
    records: Sequence[Dict],
    sample_ids: Optional[str],
    max_samples: Optional[int],
) -> List[Dict]:
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
    samples = []
    for rec in records:
        input_images = rec.get("input_images") or []
        if not input_images:
            print(f"WARNING: skipping {canonical_sample_id(rec)} because input_images is empty")
            continue
        gt_path = rec["output_image"]
        samples.append(
            Sample(
                sample_id=canonical_sample_id(rec),
                input_path=input_images[0],
                gt_path=gt_path,
                gen_path=derive_generated_path(gt_path, inference_dir),
                mask_path=derive_mask_path(gt_path, inference_dir),
            )
        )
    return samples


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    resized = []
    for ch in mask:
        resized.append(cv2.resize(ch, (width, height), interpolation=cv2.INTER_NEAREST))
    return np.stack(resized, axis=0)


def normalize_binary(mask: np.ndarray, threshold: float) -> np.ndarray:
    return (mask > threshold).astype(np.float32)


def overlay_mask(
    image_rgb: np.ndarray,
    mask_10ch: np.ndarray,
    alpha: float = 0.4,
    channels: Optional[Iterable[int]] = None,
    threshold: float = 0.0,
) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    h, w = image.shape[:2]
    mask = resize_mask(mask_10ch, h, w)
    mask = normalize_binary(mask, threshold)

    if channels is None:
        channels = range(mask.shape[0])

    color_layer = np.zeros_like(image)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    for ch in channels:
        ch_mask = mask[ch][..., None]
        color_layer += ch_mask * CHANNEL_COLORS[ch]
        weight += ch_mask

    positive = weight[..., 0] > 0
    if np.any(positive):
        color_layer[positive] /= np.maximum(weight[positive], 1.0)

    out = image.copy()
    out[positive] = (1.0 - alpha) * image[positive] + alpha * color_layer[positive]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def plot_image(ax, image: np.ndarray, title: str):
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def save_panorama(sample: Sample, output_dir: str, alpha: float, threshold: float, dpi: int):
    input_img = read_rgb(sample.input_path)
    gt_img = read_rgb(sample.gt_path)
    gen_img = read_rgb(sample.gen_path)
    mask = load_mask_npz(sample.mask_path)
    overlay = overlay_mask(gen_img, mask, alpha=alpha, threshold=threshold)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    plot_image(axes[0], input_img, "Input Image")
    plot_image(axes[1], gt_img, "GT Edited Image")
    plot_image(axes[2], gen_img, "Generated Image")
    plot_image(axes[3], overlay, "Generated + Predicted Mask")

    out_path = os.path.join(output_dir, f"{sample.sample_id}_panorama.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def save_channel_breakdown(sample: Sample, output_dir: str, alpha: float, threshold: float, dpi: int):
    gen_img = read_rgb(sample.gen_path)
    mask = load_mask_npz(sample.mask_path)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for ch, ax in enumerate(axes.flat):
        overlay = overlay_mask(gen_img, mask, alpha=alpha, channels=[ch], threshold=threshold)
        plot_image(ax, overlay, f"Ch {ch}: {ORGAN_NAMES[ch]}")

    out_path = os.path.join(output_dir, f"{sample.sample_id}_channels.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create publication overlays for predicted joint masks.",
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
        default="analysis_visualizations",
        help="Directory where visualization PNGs will be saved.",
    )
    parser.add_argument(
        "--sample_ids",
        default=None,
        help="Comma-separated IDs to process. Matches sample_id/id, patient ID, view stem, or patient_stem.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Process the first N samples when --sample_ids is not provided.",
    )
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha.")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="Mask binarization threshold.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_jsonl(args.jsonl_path)
    selected = select_records(records, args.sample_ids, args.max_samples)
    samples = build_samples(selected, args.inference_dir)

    if not samples:
        raise RuntimeError("No samples selected for visualization.")

    written = []
    skipped = 0
    for sample in samples:
        try:
            written.append(save_panorama(sample, args.output_dir, args.alpha, args.mask_threshold, args.dpi))
            written.append(save_channel_breakdown(sample, args.output_dir, args.alpha, args.mask_threshold, args.dpi))
        except Exception as exc:
            skipped += 1
            print(f"WARNING: skipping {sample.sample_id}: {exc}")

    print(f"Saved {len(written)} figures for {len(samples) - skipped} samples to {args.output_dir}")
    if skipped:
        print(f"Skipped {skipped} samples due to missing/corrupt inputs.")


if __name__ == "__main__":
    main()
