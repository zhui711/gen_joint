#!/usr/bin/env python3
"""
Diagnose whether mask blurriness comes from the mask autoencoder bottleneck
or from the flow-matching / ODE generation path.

This script performs an upper-bound autoencoder round trip:
    GT mask -> MaskEncoder -> MaskDecoder -> threshold > 0

No diffusion, no ODE, no image model.
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from OmniGen.mask_autoencoder import MaskDecoder, MaskEncoder


DEFAULT_JSONL = "/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl"
DEFAULT_MASK_MODULES = "/home/wenting/zr/gen_code_plan2_1/results/10000/mask_modules.bin"


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
                print(f"WARNING: skipping malformed line {line_no}: {exc}")
    return records


def canonical_sample_id(record: Dict) -> str:
    if record.get("sample_id"):
        return str(record["sample_id"])
    if record.get("id"):
        return str(record["id"])
    mask_path = record.get("output_mask") or record.get("output_image") or ""
    parts = mask_path.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}_{os.path.splitext(parts[-1])[0]}"
    return os.path.splitext(os.path.basename(mask_path))[0] or "sample"


def load_mask_modules(path: str, latent_channels: int, device: torch.device):
    state = torch.load(path, map_location="cpu")
    enc_state = {
        k.replace("mask_encoder.", ""): v
        for k, v in state.items()
        if k.startswith("mask_encoder.")
    }
    dec_state = {
        k.replace("mask_decoder.", ""): v
        for k, v in state.items()
        if k.startswith("mask_decoder.")
    }
    if not enc_state:
        raise RuntimeError(f"No mask_encoder.* keys found in {path}")
    if not dec_state:
        raise RuntimeError(f"No mask_decoder.* keys found in {path}")

    latent_weight = enc_state.get("net.12.weight")
    if latent_weight is not None:
        latent_channels = int(latent_weight.shape[0])

    encoder = MaskEncoder(in_channels=10, latent_channels=latent_channels)
    decoder = MaskDecoder(latent_channels=latent_channels, out_channels=10)
    encoder.load_state_dict(enc_state, strict=True)
    decoder.load_state_dict(dec_state, strict=True)
    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder, latent_channels


def select_records(records: Sequence[Dict], max_samples: int) -> List[Dict]:
    selected = []
    for record in records:
        mask_path = record.get("output_mask")
        if mask_path and os.path.exists(mask_path):
            selected.append(record)
        if len(selected) >= max_samples:
            break
    if not selected:
        raise RuntimeError("No usable records with existing output_mask found.")
    return selected


def load_mask_npz(path: str, key: str) -> np.ndarray:
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(f"{key!r} not found in {path}; available keys={data.files}")
        mask = data[key].astype(np.float32)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.shape} from {path}")
    if mask.shape[0] != 10 and mask.shape[-1] == 10:
        mask = np.moveaxis(mask, -1, 0)
    if mask.shape[0] != 10:
        raise ValueError(f"Expected 10 channels, got {mask.shape} from {path}")
    if mask.shape[1:] != (256, 256):
        mask = np.stack(
            [cv2.resize(ch, (256, 256), interpolation=cv2.INTER_NEAREST) for ch in mask],
            axis=0,
        )
    return mask


def make_grid(mask: np.ndarray, title: str = "") -> np.ndarray:
    """mask: [10,H,W] in {0,1} or [0,1]."""
    mask = np.clip(mask, 0.0, 1.0)
    panels = []
    for ch in range(mask.shape[0]):
        panel = (mask[ch] * 255).astype(np.uint8)
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2RGB)
        cv2.putText(panel, f"ch{ch}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 64, 64), 1, cv2.LINE_AA)
        panels.append(panel)
    rows = [np.concatenate(panels[i:i + 5], axis=1) for i in range(0, 10, 5)]
    grid = np.concatenate(rows, axis=0)
    if title:
        header = np.full((28, grid.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(header, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        grid = np.concatenate([header, grid], axis=0)
    return grid


def save_comparison(gt: np.ndarray, recon_bin: np.ndarray, recon_cont: np.ndarray, output_path: str, sample_id: str):
    gt_grid = make_grid(gt, f"{sample_id} GT mask")
    recon_bin_grid = make_grid(recon_bin, f"{sample_id} AE recon threshold > 0")
    recon_cont_grid = make_grid((recon_cont + 1.0) / 2.0, f"{sample_id} AE recon continuous")
    spacer = np.full((gt_grid.shape[0], 10, 3), 255, dtype=np.uint8)
    comparison = np.concatenate([gt_grid, spacer, recon_bin_grid, spacer, recon_cont_grid], axis=1)
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def latent_stats(z: torch.Tensor) -> Dict[str, float]:
    x = z.detach().float().cpu().numpy().reshape(-1)
    mean = float(np.mean(x))
    std = float(np.std(x))
    centered = x - mean
    if std > 0:
        skew = float(np.mean((centered / std) ** 3))
        kurtosis = float(np.mean((centered / std) ** 4) - 3.0)
    else:
        skew = float("nan")
        kurtosis = float("nan")
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": mean,
        "std": std,
        "skewness": skew,
        "excess_kurtosis": kurtosis,
        "p01": float(np.percentile(x, 1)),
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
    }


def binary_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_bool = gt > 0.5
    pred_bool = pred > 0.5
    inter = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()
    return float(inter / union) if union > 0 else float("nan")


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_bool = gt > 0.5
    pred_bool = pred > 0.5
    inter = np.logical_and(gt_bool, pred_bool).sum()
    denom = gt_bool.sum() + pred_bool.sum()
    return float(2 * inter / denom) if denom > 0 else float("nan")


def edge_density(mask_bin: np.ndarray) -> float:
    total = 0
    pixels = 0
    for ch in mask_bin:
        edges = cv2.Canny((ch > 0.5).astype(np.uint8) * 255, 50, 150)
        total += int((edges > 0).sum())
        pixels += edges.size
    return float(total / pixels)


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def write_report(args, aggregate: Dict, sample_rows: List[Dict], example_paths: List[str]):
    stats = aggregate["latent_stats"]
    visual_verdict = aggregate["visual_verdict"]
    conclusion = aggregate["conclusion"]

    lines = []
    lines.append("# Mask Blur Diagnosis Report\n\n")
    lines.append("## Setup\n\n")
    lines.append(f"- checkpoint: `{args.mask_modules_path}`\n")
    lines.append(f"- jsonl: `{args.jsonl_path}`\n")
    lines.append(f"- samples: `{len(sample_rows)}`\n")
    lines.append("- test: `GT mask -> MaskEncoder -> MaskDecoder -> threshold > 0`\n")
    lines.append("- no ODE / no diffusion / no image model used\n\n")

    lines.append("## Visual Verdict\n\n")
    lines.append(f"**{visual_verdict}**\n\n")
    lines.append("Saved comparison grids:\n\n")
    for path in example_paths:
        lines.append(f"- `{path}`\n")
    lines.append("\n")

    lines.append("## Mathematical Verdict\n\n")
    lines.append(f"**{conclusion}**\n\n")
    lines.append("Round-trip binary metrics are used only as an autoencoder sanity check, not as test-set segmentation evaluation.\n\n")
    lines.append(f"- mean autoencoder Dice: `{aggregate['mean_dice']:.6f}`\n")
    lines.append(f"- mean autoencoder IoU: `{aggregate['mean_iou']:.6f}`\n")
    lines.append(f"- mean GT edge density: `{aggregate['mean_gt_edge_density']:.6f}`\n")
    lines.append(f"- mean recon edge density: `{aggregate['mean_recon_edge_density']:.6f}`\n\n")

    lines.append("## Latent Distribution\n\n")
    lines.append("| statistic | value |\n")
    lines.append("|---|---:|\n")
    for key in ["min", "max", "mean", "std", "skewness", "excess_kurtosis", "p01", "p50", "p99"]:
        lines.append(f"| `{key}` | {stats[key]:.6f} |\n")
    lines.append("\n")

    lines.append("KL-divergence hypothesis note: these clean encoder latents are not constrained to a unit Gaussian. ")
    lines.append("A unit-Gaussian prior would have mean near 0, std near 1, skew near 0, and excess kurtosis near 0. ")
    lines.append("The reported statistics show the actual learned latent support that the flow model must hit.\n\n")

    lines.append("## Per-Sample CSV\n\n")
    lines.append(f"- `{os.path.join(args.output_dir, 'sample_metrics.csv')}`\n")

    report_path = os.path.join(args.output_dir, "BLUR_DIAGNOSIS_REPORT.md")
    with open(report_path, "w") as f:
        f.write("".join(lines))
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose whether mask blurriness is from the autoencoder bottleneck or ODE trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mask_modules_path", type=str, default=DEFAULT_MASK_MODULES)
    parser.add_argument("--jsonl_path", type=str, default=DEFAULT_JSONL)
    parser.add_argument("--output_dir", type=str, default="mask_blur_diagnostic")
    parser.add_argument("--mask_key", type=str, default="mask")
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--num_visuals", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: {args.device} requested but CUDA unavailable; using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    encoder, decoder, latent_channels = load_mask_modules(args.mask_modules_path, args.latent_channels, device)
    records = select_records(load_jsonl(args.jsonl_path), args.max_samples)

    all_latents = []
    rows = []
    example_paths = []
    with torch.no_grad():
        for idx, record in enumerate(records):
            sample_id = canonical_sample_id(record)
            gt = load_mask_npz(record["output_mask"], args.mask_key)
            gt_cont = torch.from_numpy(gt).unsqueeze(0).to(device=device, dtype=torch.float32) * 2.0 - 1.0
            z = encoder(gt_cont)
            recon_cont = decoder(z)
            recon_bin = (recon_cont > 0.0).float()

            z_stats = latent_stats(z)
            all_latents.append(z.detach().float().cpu())

            recon_np = recon_bin.squeeze(0).cpu().numpy()
            recon_cont_np = recon_cont.squeeze(0).cpu().numpy()
            row = {
                "sample_id": sample_id,
                "mask_path": record["output_mask"],
                "dice": dice_score(gt, recon_np),
                "iou": binary_iou(gt, recon_np),
                "gt_edge_density": edge_density(gt),
                "recon_edge_density": edge_density(recon_np),
                **{f"latent_{k}": v for k, v in z_stats.items()},
            }
            rows.append(row)

            if idx < args.num_visuals:
                out_path = os.path.join(args.output_dir, f"{sample_id}_ae_roundtrip_grid.png")
                save_comparison(gt, recon_np, recon_cont_np, out_path, sample_id)
                example_paths.append(out_path)

    all_z = torch.cat(all_latents, dim=0)
    aggregate_latent_stats = latent_stats(all_z)
    mean_dice = finite_mean([r["dice"] for r in rows])
    mean_iou = finite_mean([r["iou"] for r in rows])
    mean_gt_edge = finite_mean([r["gt_edge_density"] for r in rows])
    mean_recon_edge = finite_mean([r["recon_edge_density"] for r in rows])

    # Conservative automatic verdict: high binary overlap and similar edge density
    # indicates the AE upper bound is structurally sharp after thresholding.
    edge_ratio = mean_recon_edge / mean_gt_edge if mean_gt_edge > 0 else float("nan")
    if mean_dice >= 0.85 and 0.75 <= edge_ratio <= 1.35:
        visual_verdict = "Autoencoder upper-bound masks are structurally sharp after thresholding."
        conclusion = (
            "The dominant blur/patchiness is not an inherent MaskEncoder/MaskDecoder bottleneck. "
            "It comes from the flow-matching trajectory not landing exactly on the clean mask-latent manifold."
        )
    else:
        visual_verdict = "Autoencoder upper-bound masks are already degraded after thresholding."
        conclusion = (
            "The autoencoder bottleneck contributes materially to the blur/patchiness. "
            "The ODE path cannot exceed this upper bound without improving the mask autoencoder."
        )

    csv_path = os.path.join(args.output_dir, "sample_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        "latent_stats": aggregate_latent_stats,
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "mean_gt_edge_density": mean_gt_edge,
        "mean_recon_edge_density": mean_recon_edge,
        "visual_verdict": visual_verdict,
        "conclusion": conclusion,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"aggregate": aggregate, "samples": rows}, f, indent=2)

    report_path = write_report(args, aggregate, rows, example_paths)
    print(json.dumps(aggregate, indent=2))
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
