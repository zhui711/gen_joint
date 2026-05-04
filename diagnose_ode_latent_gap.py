#!/usr/bin/env python3
"""
Face-to-face diagnostic for mask latent ODE accuracy.

This script answers two specific questions without changing training or
inference code:

1. Is a static mask latent scale factor still present in the active loss or
   inference path?
2. How close is the final ODE-predicted mask latent to the clean
   MaskEncoder(GT mask) latent for the same training sample?

It intentionally computes no image metrics and makes no architecture changes.
"""

import argparse
import gc
import json
import math
import os
import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from OmniGen import OmniGenProcessor, OmniGenScheduler
from test_joint_mask import initialize_joint_mask_modules, load_jsonl, setup_logging


DEFAULT_TRAIN_JSONL = "/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl"
DEFAULT_LORA_PATH = "results/10000"
DEFAULT_MASK_MODULES = "results/10000/mask_modules.bin"
DEFAULT_SCALE_CHECK_FILES = [
    "OmniGen/train_helper/loss_joint_mask.py",
    "test_joint_mask.py",
    "OmniGen/pipeline.py",
    "OmniGen/scheduler.py",
]


def canonical_sample_id(record: Dict) -> str:
    if record.get("sample_id"):
        return str(record["sample_id"])
    if record.get("id"):
        return str(record["id"])
    path = record.get("output_mask") or record.get("output_image") or ""
    parts = path.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}_{os.path.splitext(parts[-1])[0]}"
    return os.path.splitext(os.path.basename(path))[0] or "sample"


def sample_id_candidates(record: Dict) -> List[str]:
    paths = [
        record.get("output_mask") or "",
        record.get("output_image") or "",
    ]
    candidates = [record.get("sample_id", ""), record.get("id", "")]
    for path in paths:
        parts = path.replace("\\", "/").rstrip("/").split("/")
        if len(parts) >= 2:
            patient = parts[-2]
            stem = os.path.splitext(parts[-1])[0]
            candidates.extend([patient, stem, f"{patient}_{stem}", f"{patient}/{stem}"])
    return [str(x) for x in candidates if x is not None and str(x) != ""]


def pick_sample(records: List[Dict], sample_id: Optional[str]) -> Dict:
    usable = []
    for record in records:
        if not record.get("instruction"):
            continue
        if not record.get("input_images"):
            continue
        if not record.get("output_mask") or not os.path.exists(record["output_mask"]):
            continue
        usable.append(record)

    if not usable:
        raise RuntimeError("No usable training samples with instruction, input image, and existing output_mask.")

    if not sample_id:
        return usable[0]

    requested = {x.strip() for x in sample_id.split(",") if x.strip()}
    for record in usable:
        if requested.intersection(sample_id_candidates(record)):
            return record

    raise RuntimeError(f"Could not find requested sample_id in training JSONL: {sample_id}")


def inspect_static_scale_factors(files: List[str]) -> Dict:
    """Search active code paths for explicit static mask latent scaling."""
    patterns = [
        re.compile(r"MASK[_A-Z0-9]*SCALE[_A-Z0-9]*\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        re.compile(r"mask[_a-zA-Z0-9]*scale[_a-zA-Z0-9]*\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        re.compile(r"scale[_a-zA-Z0-9]*mask[_a-zA-Z0-9]*\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    ]
    loose_terms = re.compile(r"(MASK_SCALE_FACTOR|mask_scale|scale_mask|mask.*scale|scale.*mask)", re.IGNORECASE)
    matches = []
    numeric_factors = []

    for path in files:
        if not os.path.exists(path):
            matches.append({"file": path, "line": None, "text": "FILE_NOT_FOUND", "numeric_factor": None})
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                stripped = line.strip()
                if not loose_terms.search(stripped):
                    continue
                numeric_factor = None
                for pattern in patterns:
                    found = pattern.search(stripped)
                    if found:
                        numeric_factor = float(found.group(1))
                        numeric_factors.append(numeric_factor)
                        break
                matches.append(
                    {
                        "file": path,
                        "line": line_no,
                        "text": stripped,
                        "numeric_factor": numeric_factor,
                    }
                )

    hardcoded_mask_scale_found = any(m["numeric_factor"] is not None for m in matches)
    return {
        "files": files,
        "matches": matches,
        "numeric_factors": numeric_factors,
        "hardcoded_mask_scale_found": hardcoded_mask_scale_found,
    }


def load_mask_cont(mask_path: str, key: str, device: torch.device) -> torch.Tensor:
    with np.load(mask_path) as data:
        if key not in data.files:
            raise KeyError(f"{key!r} not found in {mask_path}; available keys={data.files}")
        mask = data[key].astype(np.float32)

    if mask.ndim != 3:
        raise ValueError(f"Expected GT mask shape [10,H,W] or [H,W,10], got {mask.shape}")
    if mask.shape[0] != 10 and mask.shape[-1] == 10:
        mask = np.moveaxis(mask, -1, 0)
    if mask.shape[0] != 10:
        raise ValueError(f"Expected 10 mask channels, got shape {mask.shape}")
    if mask.shape[1:] != (256, 256):
        mask = np.stack(
            [cv2.resize(ch, (256, 256), interpolation=cv2.INTER_NEAREST) for ch in mask],
            axis=0,
        )

    mask = np.clip(mask, 0.0, 1.0)
    return (2.0 * torch.from_numpy(mask).unsqueeze(0) - 1.0).to(device=device, dtype=torch.float32)


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    x = tensor.detach().float().cpu().numpy().reshape(-1)
    mean = float(np.mean(x))
    std = float(np.std(x))
    centered = x - mean
    if std > 0:
        skew = float(np.mean((centered / std) ** 3))
        excess_kurtosis = float(np.mean((centered / std) ** 4) - 3.0)
    else:
        skew = float("nan")
        excess_kurtosis = float("nan")
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": mean,
        "std": std,
        "skewness": skew,
        "excess_kurtosis": excess_kurtosis,
        "p01": float(np.percentile(x, 1)),
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
    }


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    delta = a.detach().float().cpu().reshape(-1) - b.detach().float().cpu().reshape(-1)
    return float(torch.linalg.vector_norm(delta).item())


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().cpu().reshape(-1)
    y = b.detach().float().cpu().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) == 0.0:
        return float("nan")
    return float(torch.dot(x, y).div(denom).item())


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    numerator = l2_distance(a, b)
    denom = float(torch.linalg.vector_norm(b.detach().float().cpu().reshape(-1)).item())
    return float(numerator / denom) if denom > 0 else float("nan")


def save_mask_grid(mask_cont: torch.Tensor, path: str, title: str) -> None:
    mask = mask_cont.detach().float().cpu().squeeze(0).numpy()
    if mask.shape[0] != 10:
        raise ValueError(f"Expected mask tensor with 10 channels, got {mask.shape}")
    mask01 = np.clip((mask + 1.0) / 2.0, 0.0, 1.0)
    panels = []
    for channel in range(10):
        panel = (mask01[channel] * 255).astype(np.uint8)
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2RGB)
        cv2.putText(panel, f"ch{channel}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 64, 64), 1, cv2.LINE_AA)
        panels.append(panel)
    rows = [np.concatenate(panels[i:i + 5], axis=1) for i in range(0, 10, 5)]
    grid = np.concatenate(rows, axis=0)
    header = np.full((30, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    grid = np.concatenate([header, grid], axis=0)
    Image.fromarray(grid).save(path)


def save_mask_union(mask_cont: torch.Tensor, path: str) -> None:
    mask = mask_cont.detach().float().cpu().squeeze(0).numpy()
    mask01 = np.clip((mask + 1.0) / 2.0, 0.0, 1.0)
    union = np.max(mask01, axis=0)
    Image.fromarray((union * 255).astype(np.uint8)).save(path)


def save_latent_histogram(z_clean: torch.Tensor, z_ode: torch.Tensor, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = z_clean.detach().float().cpu().numpy().reshape(-1)
    ode = z_ode.detach().float().cpu().numpy().reshape(-1)
    plt.figure(figsize=(7, 4.5), dpi=160)
    plt.hist(clean, bins=80, alpha=0.65, density=True, label="z_clean")
    plt.hist(ode, bins=80, alpha=0.55, density=True, label="z_ode_unscaled")
    plt.xlabel("latent value")
    plt.ylabel("density")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def load_pipeline(args, device: torch.device):
    from OmniGen import OmniGenPipeline

    pipe = OmniGenPipeline.from_pretrained(args.model_path)
    pipe.model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)
    if args.lora_path:
        print(f"Loading LoRA checkpoint: {args.lora_path}")
        pipe.merge_lora(args.lora_path)
    initialize_joint_mask_modules(pipe, args, setup_logging(args.output_dir) or __import__("logging").getLogger("latent-gap"))
    pipe.to(device)
    pipe.model.eval()
    pipe.vae.eval()
    pipe.mask_encoder.eval()
    pipe.mask_decoder.eval()
    return pipe


@torch.no_grad()
def run_current_ode_and_return_mask_latent(pipe, record: Dict, args) -> Tuple[torch.Tensor, Image.Image]:
    """Mirror the current joint inference path and return the final clean mask latent."""
    prompt = [record["instruction"]]
    input_images = [record.get("input_images") or None]
    height = width = 256
    dtype = args.torch_dtype
    use_img_guidance = input_images is not None
    separate_cfg_infer = False
    num_prompt = len(prompt)
    num_cfg = 2 if use_img_guidance else 1

    if args.max_image_size != pipe.processor.max_image_size:
        pipe.processor = OmniGenProcessor(pipe.processor.text_tokenizer, max_image_size=args.max_image_size)

    pipe.model.to(dtype)
    pipe.disable_model_cpu_offload()

    use_joint = pipe.model.use_joint_mask and pipe.mask_decoder is not None
    if not use_joint:
        raise RuntimeError("Joint mask mode is not active after loading the checkpoint.")

    latent_size_h, latent_size_w = height // 8, width // 8
    num_mask_tokens = (latent_size_h // pipe.model.patch_size) * (latent_size_w // pipe.model.patch_size)
    input_data = pipe.processor(
        prompt,
        input_images,
        height=height,
        width=width,
        use_img_cfg=use_img_guidance,
        separate_cfg_input=separate_cfg_infer,
        use_input_image_size_as_output=False,
        num_mask_tokens=num_mask_tokens,
    )

    generator = torch.Generator(device=pipe.device).manual_seed(args.seed) if args.seed is not None else None
    latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=pipe.device, generator=generator)
    latents = torch.cat([latents] * (1 + num_cfg), dim=0).to(dtype)
    mask_latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=pipe.device, generator=generator)
    mask_latents = torch.cat([mask_latents] * (1 + num_cfg), dim=0).to(dtype)

    input_img_latents = []
    for image_tensor in input_data["input_pixel_values"]:
        input_img_latents.append(pipe.vae_encode(image_tensor.to(pipe.device), dtype))

    model_kwargs = dict(
        input_ids=pipe.move_to_device(input_data["input_ids"]),
        input_img_latents=input_img_latents,
        input_image_sizes=deepcopy(input_data["input_image_sizes"]),
        attention_mask=pipe.move_to_device(input_data["attention_mask"]),
        position_ids=pipe.move_to_device(input_data["position_ids"]),
        cfg_scale=args.guidance_scale,
        img_cfg_scale=args.img_guidance_scale,
        use_img_cfg=use_img_guidance,
        use_kv_cache=args.use_kv_cache,
        offload_model=False,
    )

    scheduler = OmniGenScheduler(num_steps=args.inference_steps)
    samples, mask_samples = scheduler.__call_joint__(
        latents,
        mask_latents,
        pipe.model.forward_with_cfg,
        model_kwargs,
        use_kv_cache=args.use_kv_cache,
        offload_kv_cache=args.offload_kv_cache,
    )

    clean_img_latent = samples.chunk((1 + num_cfg), dim=0)[0]
    clean_mask_latent = mask_samples.chunk((1 + num_cfg), dim=0)[0].to(torch.float32)

    pipe.vae.to(pipe.device)
    image_latent = clean_img_latent.to(torch.float32)
    if pipe.vae.config.shift_factor is not None:
        image_latent = image_latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    else:
        image_latent = image_latent / pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(image_latent).sample
    decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
    image_np = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    torch.cuda.empty_cache()
    gc.collect()
    return clean_mask_latent, Image.fromarray(image_np)


def markdown_stats_row(name: str, stats: Dict[str, float]) -> str:
    return (
        f"| {name} | {stats['min']:.6f} | {stats['max']:.6f} | "
        f"{stats['mean']:.6f} | {stats['std']:.6f} | {stats['p01']:.6f} | "
        f"{stats['p50']:.6f} | {stats['p99']:.6f} |\n"
    )


def write_report(args, sample: Dict, scale_check: Dict, results: Dict) -> str:
    clean_stats = results["z_clean_stats"]
    ode_stats = results["z_ode_unscaled_stats"]
    std_ratio = ode_stats["std"] / clean_stats["std"] if clean_stats["std"] > 0 else float("nan")

    scale_found = scale_check["hardcoded_mask_scale_found"]
    numeric_factors = scale_check["numeric_factors"]
    if numeric_factors:
        largest_factor = max(abs(x) for x in numeric_factors)
        implied_target_std = clean_stats["std"] * largest_factor
    else:
        largest_factor = None
        implied_target_std = clean_stats["std"]

    cosine = results["cosine_similarity"]
    rel_l2 = results["relative_l2_to_clean"]
    huge_gap = (not math.isnan(cosine) and cosine < 0.5) or (not math.isnan(rel_l2) and rel_l2 > 1.0)
    stats_mismatch = (
        clean_stats["std"] > 0
        and (std_ratio > 2.0 or std_ratio < 0.5 or abs(ode_stats["mean"] - clean_stats["mean"]) > clean_stats["std"])
    )

    lines = []
    lines.append("# Latent Gap Report\n\n")
    lines.append("## Setup\n\n")
    lines.append(f"- sample_id: `{canonical_sample_id(sample)}`\n")
    lines.append(f"- training jsonl: `{args.jsonl_path}`\n")
    lines.append(f"- base model: `{args.model_path}`\n")
    lines.append(f"- LoRA checkpoint: `{args.lora_path}`\n")
    lines.append(f"- mask modules: `{args.mask_modules_path}`\n")
    lines.append(f"- inference steps: `{args.inference_steps}`\n")
    lines.append(f"- guidance_scale: `{args.guidance_scale}`\n")
    lines.append(f"- img_guidance_scale: `{args.img_guidance_scale}`\n")
    lines.append(f"- seed: `{args.seed}`\n\n")

    lines.append("## 1. Scale Factor Code Check\n\n")
    if scale_found:
        lines.append("**A static mask-latent scale factor was found in the inspected active code paths.**\n\n")
        lines.append(f"- numeric factors found: `{numeric_factors}`\n")
        lines.append(f"- clean latent std for this sample: `{clean_stats['std']:.6f}`\n")
        lines.append(f"- implied scaled target std using largest factor `{largest_factor:.6f}`: `{implied_target_std:.6f}`\n\n")
    else:
        lines.append("**No hardcoded `MASK_SCALE_FACTOR`, `mask_scale`, or equivalent static mask-latent scaling assignment was found in the inspected active code paths.**\n\n")
        lines.append("The current loss path uses raw `MaskEncoder(GT_mask)` latents, and the current inference path initializes and updates `mask_latents` directly with no unscale step.\n\n")
        lines.append(f"- clean latent std for this sample: `{clean_stats['std']:.6f}`\n")
        lines.append("- implied scaled target std from current code: not applicable\n\n")

    lines.append("Inspected lines matching scale-related terms:\n\n")
    if scale_check["matches"]:
        lines.append("| file | line | numeric factor | text |\n")
        lines.append("|---|---:|---:|---|\n")
        for match in scale_check["matches"]:
            line = "" if match["line"] is None else str(match["line"])
            factor = "" if match["numeric_factor"] is None else f"{match['numeric_factor']:.6f}"
            text = str(match["text"]).replace("|", "\\|")
            lines.append(f"| `{match['file']}` | {line} | {factor} | `{text}` |\n")
    else:
        lines.append("- No scale-related lines matched in the inspected files.\n")
    lines.append("\n")

    lines.append("## 2. Face-to-Face Latent Probe\n\n")
    lines.append("| latent | min | max | mean | std | p01 | p50 | p99 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    lines.append(markdown_stats_row("z_clean = MaskEncoder(GT mask)", clean_stats))
    lines.append(markdown_stats_row("z_ode_unscaled = current 50-step ODE output", ode_stats))
    lines.append("\n")
    lines.append(f"- `||z_clean - z_ode_unscaled||_2`: `{results['l2_to_clean']:.6f}`\n")
    lines.append(f"- relative L2 vs `||z_clean||_2`: `{rel_l2:.6f}`\n")
    lines.append(f"- cosine similarity: `{cosine:.6f}`\n")
    lines.append(f"- ODE/clean std ratio: `{std_ratio:.6f}`\n\n")

    lines.append("## 3. Saved Evidence\n\n")
    for key in ["gt_mask_grid", "clean_recon_grid", "ode_recon_grid", "clean_recon_union", "ode_recon_union", "generated_image", "latent_histogram"]:
        lines.append(f"- `{results[key]}`\n")
    lines.append("\n")

    lines.append("## 4. Strategic Conclusion\n\n")
    if scale_found and implied_target_std > 10.0:
        lines.append(
            "**The static scale-factor bug is confirmed.** The mask flow target is being expanded to a scale far outside "
            "the clean encoder latent support, so the next fix should be dynamic latent normalization rather than KL-first architecture changes.\n\n"
        )
        lines.append(
            "Recommended next fix: replace hardcoded mask scaling with checkpointed dataset statistics, e.g. normalize "
            "`z_mask = (z_clean - mean_train) / std_train` during training and unnormalize exactly once before `MaskDecoder` at inference.\n"
        )
    elif huge_gap or stats_mismatch:
        lines.append(
            "**The scale-factor hypothesis is not supported by the current code, but the ODE latent gap is confirmed.** "
            "The ODE output is not landing close to the clean `MaskEncoder(GT)` latent for this sample.\n\n"
        )
        lines.append(
            "This points to the flow trajectory / latent-manifold alignment as the immediate failure source. "
            "A KL-style prior may still be useful, but the hard data says the first target is to constrain the generated "
            "mask latent distribution to the measured clean encoder distribution.\n\n"
        )
        lines.append(
            "Recommended next fixes to test, in order: per-channel latent mean/std normalization using training-set "
            "statistics; branch-specific mask CFG reduction or disabling CFG for mask tokens; then a lightweight KL "
            "regularizer on `MaskEncoder` if the normalized flow still leaves holes in latent space.\n"
        )
    else:
        lines.append(
            "**The ODE latent is numerically close to the clean latent under these probes.** If decoded masks remain patchy, "
            "the likely issue is not global scale but local latent topology or decoder sensitivity to small off-manifold perturbations.\n\n"
        )
        lines.append(
            "Recommended next fix: add a lightweight KL or latent smoothness regularizer to make the encoder latent manifold "
            "more connected, while preserving the already validated autoencoder reconstruction capacity.\n"
        )

    report_text = "".join(lines)
    with open(args.report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)
    output_report_path = os.path.join(args.output_dir, "LATENT_GAP_REPORT.md")
    with open(output_report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)
    return args.report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose static scale factor and ODE-vs-clean mask latent gap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl_path", type=str, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--sample_id", type=str, default=None, help="Optional sample/patient ID. First usable sample is used if omitted.")
    parser.add_argument("--model_path", type=str, default="Shitao/OmniGen-v1")
    parser.add_argument("--lora_path", type=str, default=DEFAULT_LORA_PATH)
    parser.add_argument("--mask_modules_path", type=str, default=DEFAULT_MASK_MODULES)
    parser.add_argument("--output_dir", type=str, default="ode_latent_gap_diagnostic")
    parser.add_argument("--report_path", type=str, default="LATENT_GAP_REPORT.md")
    parser.add_argument("--mask_key", type=str, default="mask")
    parser.add_argument("--mask_latent_channels", type=int, default=4)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--img_guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--use_kv_cache", action="store_true")
    parser.add_argument("--offload_kv_cache", action="store_true")
    parser.add_argument("--scale_check_files", type=str, default=",".join(DEFAULT_SCALE_CHECK_FILES), help="Comma-separated files to inspect for mask scale factors.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")
    args.torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    scale_files = [path.strip() for path in args.scale_check_files.split(",") if path.strip()]
    scale_check = inspect_static_scale_factors(scale_files)
    print("Hardcoded mask scale found:", scale_check["hardcoded_mask_scale_found"])
    if scale_check["matches"]:
        print("Scale-related matches:")
        for match in scale_check["matches"]:
            print(f"  {match['file']}:{match['line']} factor={match['numeric_factor']} text={match['text']}")

    records = load_jsonl(args.jsonl_path)
    sample = pick_sample(records, args.sample_id)
    sample_id = canonical_sample_id(sample)
    print(f"Selected sample: {sample_id}")

    pipe = load_pipeline(args, device)

    gt_mask = load_mask_cont(sample["output_mask"], args.mask_key, device)
    with torch.no_grad():
        z_clean = pipe.mask_encoder(gt_mask)
        clean_recon = pipe.mask_decoder(z_clean)

    z_ode_raw, generated_image = run_current_ode_and_return_mask_latent(pipe, sample, args)
    # Current test_joint_mask.py has no mask latent unscale step. Keep this explicit
    # so the report captures the exact inference semantics.
    z_ode_unscaled = z_ode_raw
    with torch.no_grad():
        ode_recon = pipe.mask_decoder(z_ode_unscaled.to(device))

    paths = {
        "gt_mask_grid": os.path.join(args.output_dir, f"{sample_id}_gt_mask_grid.png"),
        "clean_recon_grid": os.path.join(args.output_dir, f"{sample_id}_clean_recon_grid.png"),
        "ode_recon_grid": os.path.join(args.output_dir, f"{sample_id}_ode_recon_grid.png"),
        "clean_recon_union": os.path.join(args.output_dir, f"{sample_id}_clean_recon_union.png"),
        "ode_recon_union": os.path.join(args.output_dir, f"{sample_id}_ode_recon_union.png"),
        "generated_image": os.path.join(args.output_dir, f"{sample_id}_generated_image.png"),
        "latent_histogram": os.path.join(args.output_dir, f"{sample_id}_latent_histogram.png"),
    }

    save_mask_grid(gt_mask, paths["gt_mask_grid"], f"{sample_id} GT mask")
    save_mask_grid(clean_recon, paths["clean_recon_grid"], f"{sample_id} MaskDecoder(z_clean)")
    save_mask_grid(ode_recon, paths["ode_recon_grid"], f"{sample_id} MaskDecoder(z_ode_unscaled)")
    save_mask_union(clean_recon, paths["clean_recon_union"])
    save_mask_union(ode_recon, paths["ode_recon_union"])
    generated_image.save(paths["generated_image"])
    save_latent_histogram(z_clean, z_ode_unscaled, paths["latent_histogram"])

    results = {
        "sample_id": sample_id,
        "sample": sample,
        "z_clean_stats": tensor_stats(z_clean),
        "z_ode_raw_stats": tensor_stats(z_ode_raw),
        "z_ode_unscaled_stats": tensor_stats(z_ode_unscaled),
        "l2_to_clean": l2_distance(z_ode_unscaled, z_clean),
        "relative_l2_to_clean": relative_l2(z_ode_unscaled, z_clean),
        "cosine_similarity": cosine_similarity(z_ode_unscaled, z_clean),
        **paths,
    }

    metrics_path = os.path.join(args.output_dir, "latent_gap_metrics.json")
    serializable_args = {
        key: (str(value) if key == "torch_dtype" else value)
        for key, value in vars(args).items()
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": serializable_args,
                "scale_check": scale_check,
                "results": results,
            },
            handle,
            indent=2,
        )

    print("z_clean stats:", results["z_clean_stats"])
    print("z_ode_unscaled stats:", results["z_ode_unscaled_stats"])
    print("L2:", results["l2_to_clean"])
    print("Relative L2:", results["relative_l2_to_clean"])
    print("Cosine similarity:", results["cosine_similarity"])
    print(f"Wrote metrics: {metrics_path}")
    report_path = write_report(args, sample, scale_check, results)
    print(f"Wrote report: {report_path}")
    print(f"Wrote report copy: {os.path.join(args.output_dir, 'LATENT_GAP_REPORT.md')}")


if __name__ == "__main__":
    main()
