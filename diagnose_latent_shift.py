#!/usr/bin/env python3
"""
Diagnose mask-latent distribution shift during joint image-mask inference.

Hypothesis:
  The trained MaskDecoder reconstructs clean encoder latents well, but the ODE
  inference loop produces out-of-distribution mask latents, especially under CFG.

This script compares:
  1. Z_m0_true  = MaskEncoder(GT_mask_cont)
  2. Z_m0_cfg   = final inferred mask latent with guidance_scale/img_guidance=2.5/2.0
  3. Z_m0_nocfg = final inferred mask latent with guidance_scale/img_guidance=1.0/1.0

It saves decoded mask visualizations and writes EVIDENCE_REPORT.md.
"""

import argparse
import gc
import json
import math
import os
from copy import deepcopy
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from OmniGen import OmniGenProcessor, OmniGenScheduler
from test_joint_mask import initialize_joint_mask_modules, load_jsonl, setup_logging


DEFAULT_TRAIN_JSONL = "/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl"


def canonical_sample_id(record: Dict) -> str:
    if record.get("sample_id"):
        return str(record["sample_id"])
    if record.get("id"):
        return str(record["id"])
    gt_path = record.get("output_image", "")
    parts = gt_path.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}_{os.path.splitext(parts[-1])[0]}"
    return os.path.splitext(os.path.basename(gt_path))[0] or "sample"


def sample_id_candidates(record: Dict) -> List[str]:
    gt_path = record.get("output_image", "")
    mask_path = record.get("output_mask", "")
    parts = gt_path.replace("\\", "/").rstrip("/").split("/")
    patient = parts[-2] if len(parts) >= 2 else ""
    stem = os.path.splitext(parts[-1])[0] if parts else ""
    candidates = [
        record.get("sample_id", ""),
        record.get("id", ""),
        patient,
        stem,
        f"{patient}_{stem}" if patient and stem else "",
        mask_path.replace("\\", "/").rstrip("/").split("/")[-2] if "/" in mask_path else "",
    ]
    return [str(x) for x in candidates if x is not None and str(x) != ""]


def pick_sample(records: List[Dict], sample_id: str = None) -> Dict:
    usable = [
        rec for rec in records
        if rec.get("instruction") and rec.get("input_images") and rec.get("output_mask") and os.path.exists(rec["output_mask"])
    ]
    if not usable:
        raise RuntimeError("No usable training records with instruction, input_images, and existing output_mask.")
    if sample_id is None:
        return usable[0]
    requested = {x.strip() for x in sample_id.split(",") if x.strip()}
    for rec in usable:
        if requested.intersection(sample_id_candidates(rec)):
            return rec
    raise RuntimeError(f"Could not find requested sample_id in training JSONL: {sample_id}")


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
        resized = [cv2.resize(ch, (256, 256), interpolation=cv2.INTER_NEAREST) for ch in mask]
        mask = np.stack(resized, axis=0)
    tensor = torch.from_numpy(mask).unsqueeze(0).to(device=device, dtype=torch.float32)
    return 2.0 * tensor - 1.0


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    t = tensor.detach().float().cpu()
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
    }


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((a.detach().float() - b.detach().float()).reshape(-1)).item())


def save_mask_grid(mask_cont: torch.Tensor, path: str, title: str = ""):
    mask = mask_cont.detach().float().cpu().squeeze(0).numpy()
    mask01 = np.clip((mask + 1.0) / 2.0, 0.0, 1.0)
    panels = []
    for ch in range(mask01.shape[0]):
        img = (mask01[ch] * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.putText(img, f"ch{ch}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 64, 64), 1, cv2.LINE_AA)
        panels.append(img)
    rows = [np.concatenate(panels[i:i + 5], axis=1) for i in range(0, 10, 5)]
    grid = np.concatenate(rows, axis=0)
    if title:
        header = np.full((28, grid.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(header, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        grid = np.concatenate([header, grid], axis=0)
    Image.fromarray(grid).save(path)


def save_mask_union(mask_cont: torch.Tensor, path: str):
    mask = mask_cont.detach().float().cpu().squeeze(0).numpy()
    mask01 = np.clip((mask + 1.0) / 2.0, 0.0, 1.0)
    union = np.max(mask01, axis=0)
    Image.fromarray((union * 255).astype(np.uint8)).save(path)


def load_pipeline(args, device: torch.device):
    from OmniGen import OmniGenPipeline

    pipe = OmniGenPipeline.from_pretrained(args.model_path)
    pipe.model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)
    if args.lora_path:
        print(f"Loading LoRA: {args.lora_path}")
        pipe.merge_lora(args.lora_path)
    initialize_joint_mask_modules(pipe, args, setup_logging(args.output_dir) or __import__("logging").getLogger("latent-shift"))
    pipe.to(device)
    pipe.model.eval()
    pipe.vae.eval()
    pipe.mask_encoder.eval()
    pipe.mask_decoder.eval()
    return pipe


@torch.no_grad()
def run_joint_inference_return_mask_latent(
    pipe,
    record: Dict,
    guidance_scale: float,
    img_guidance_scale: float,
    args,
) -> Tuple[torch.Tensor, Image.Image]:
    """Mirror OmniGenPipeline.__call__ but return final clean mask latent."""
    prompt = [record["instruction"]]
    input_images = [record.get("input_images") or None]
    height = width = 256
    dtype = args.dtype
    use_img_guidance = True if input_images is not None else False
    separate_cfg_infer = False
    use_kv_cache = args.use_kv_cache
    offload_kv_cache = args.offload_kv_cache
    offload_model = False

    if args.max_image_size != pipe.processor.max_image_size:
        pipe.processor = OmniGenProcessor(pipe.processor.text_tokenizer, max_image_size=args.max_image_size)

    pipe.model.to(dtype)
    pipe.disable_model_cpu_offload()

    use_joint = pipe.model.use_joint_mask and pipe.mask_decoder is not None
    if not use_joint:
        raise RuntimeError("Joint mask mode is not active after loading checkpoint.")

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

    num_prompt = len(prompt)
    num_cfg = 2 if use_img_guidance else 1
    generator = torch.Generator(device=pipe.device).manual_seed(args.seed) if args.seed is not None else None
    latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=pipe.device, generator=generator)
    latents = torch.cat([latents] * (1 + num_cfg), 0).to(dtype)
    mask_latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=pipe.device, generator=generator)
    mask_latents = torch.cat([mask_latents] * (1 + num_cfg), 0).to(dtype)

    input_img_latents = []
    for img in input_data["input_pixel_values"]:
        input_img_latents.append(pipe.vae_encode(img.to(pipe.device), dtype))

    model_kwargs = dict(
        input_ids=pipe.move_to_device(input_data["input_ids"]),
        input_img_latents=input_img_latents,
        input_image_sizes=deepcopy(input_data["input_image_sizes"]),
        attention_mask=pipe.move_to_device(input_data["attention_mask"]),
        position_ids=pipe.move_to_device(input_data["position_ids"]),
        cfg_scale=guidance_scale,
        img_cfg_scale=img_guidance_scale,
        use_img_cfg=use_img_guidance,
        use_kv_cache=use_kv_cache,
        offload_model=offload_model,
    )

    scheduler = OmniGenScheduler(num_steps=args.inference_steps)
    samples, mask_samples = scheduler.__call_joint__(
        latents,
        mask_latents,
        pipe.model.forward_with_cfg,
        model_kwargs,
        use_kv_cache=use_kv_cache,
        offload_kv_cache=offload_kv_cache,
    )

    clean_img_latent = samples.chunk((1 + num_cfg), dim=0)[0]
    clean_mask_latent = mask_samples.chunk((1 + num_cfg), dim=0)[0].to(torch.float32)

    pipe.vae.to(pipe.device)
    img_latent = clean_img_latent.to(torch.float32)
    if pipe.vae.config.shift_factor is not None:
        img_latent = img_latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    else:
        img_latent = img_latent / pipe.vae.config.scaling_factor
    decoded_img = pipe.vae.decode(img_latent).sample
    decoded_img = (decoded_img * 0.5 + 0.5).clamp(0, 1)
    img_np = (decoded_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(img_np)

    torch.cuda.empty_cache()
    gc.collect()
    return clean_mask_latent, image


def markdown_stats(name: str, stats: Dict[str, float]) -> str:
    return (
        f"| {name} | {stats['min']:.6f} | {stats['max']:.6f} | "
        f"{stats['mean']:.6f} | {stats['std']:.6f} |\n"
    )


def write_report(output_dir: str, sample: Dict, results: Dict):
    cfg_stats = results["cfg"]["stats"]
    nocfg_stats = results["nocfg"]["stats"]
    true_stats = results["true"]["stats"]

    true_range = max(abs(true_stats["min"]), abs(true_stats["max"]), 1e-6)
    cfg_range = max(abs(cfg_stats["min"]), abs(cfg_stats["max"]))
    nocfg_range = max(abs(nocfg_stats["min"]), abs(nocfg_stats["max"]))
    cfg_exploded = cfg_range > 2.0 * true_range or cfg_stats["std"] > 2.0 * max(true_stats["std"], 1e-6)
    nocfg_better = results["nocfg"]["l2_to_true"] < results["cfg"]["l2_to_true"]

    report = []
    report.append("# Latent Shift Evidence Report\n\n")
    report.append("## Sample\n\n")
    report.append(f"- `sample_id`: `{canonical_sample_id(sample)}`\n")
    report.append(f"- `input_image`: `{sample.get('input_images', [''])[0]}`\n")
    report.append(f"- `gt_image`: `{sample.get('output_image', '')}`\n")
    report.append(f"- `gt_mask`: `{sample.get('output_mask', '')}`\n\n")

    report.append("## Latent Statistics\n\n")
    report.append("| Latent | min | max | mean | std |\n")
    report.append("|---|---:|---:|---:|---:|\n")
    report.append(markdown_stats("Z_m0_true = MaskEncoder(GT mask)", true_stats))
    report.append(markdown_stats("Z_m0_cfg, guidance=CFG", cfg_stats))
    report.append(markdown_stats("Z_m0_nocfg, guidance=1.0", nocfg_stats))
    report.append("\n")
    report.append("## Distances to Clean GT Latent\n\n")
    report.append(f"- `||Z_m0_cfg - Z_m0_true||_2`: `{results['cfg']['l2_to_true']:.6f}`\n")
    report.append(f"- `||Z_m0_nocfg - Z_m0_true||_2`: `{results['nocfg']['l2_to_true']:.6f}`\n\n")

    report.append("## Saved Visual Evidence\n\n")
    for key in ["true", "cfg", "nocfg"]:
        report.append(f"- `{results[key]['grid_path']}`\n")
        report.append(f"- `{results[key]['union_path']}`\n")
    report.append(f"- `{results['cfg']['generated_path']}`\n")
    report.append(f"- `{results['nocfg']['generated_path']}`\n\n")

    report.append("## Interpretation\n\n")
    report.append(f"1. Did the latent explode under CFG? **{'Yes' if cfg_exploded else 'No'}**.\n")
    report.append(
        f"   The clean latent max-absolute range is `{true_range:.6f}`; CFG range is `{cfg_range:.6f}`; "
        f"no-CFG range is `{nocfg_range:.6f}`.\n"
    )
    report.append(
        f"2. Is no-CFG numerically closer to the clean latent? **{'Yes' if nocfg_better else 'No'}**.\n"
    )
    report.append(
        "3. Visual assessment should compare `mask_true_recon_grid.png`, `mask_cfg_grid.png`, and "
        "`mask_nocfg_grid.png`. The GT-latent reconstruction is the decoder sanity check; if CFG/no-CFG "
        "outputs are mosaic-like while GT reconstruction is anatomical, the failure is upstream of the decoder.\n\n"
    )

    report.append("## Mathematically Sound Fixes to Test\n\n")
    report.append(
        "1. Disable CFG for the mask branch while keeping image CFG. In `forward_with_cfg`, compute the guided "
        "image prediction normally, but use the conditional mask prediction directly for `mask_out` instead of "
        "applying the CFG extrapolation formula to mask tokens.\n"
    )
    report.append(
        "2. Clamp or normalize `z_mask` during each ODE step to the empirical clean-latent support measured from "
        "training masks, e.g. per-channel clamp to training percentiles or global clamp to `[p0.1, p99.9]` of "
        "`MaskEncoder(GT_mask)`.\n"
    )
    report.append(
        "3. Add a mask-latent regularizer during training or inference calibration: penalize drift in mean/std "
        "relative to the clean encoder-latent distribution, or rescale inferred `z_mask` to match clean latent "
        "per-channel mean/std before decoding.\n"
    )
    report.append(
        "4. If no-CFG is much closer than CFG, prefer branch-specific guidance: image branch uses CFG, mask branch "
        "uses conditional prediction or a much smaller guidance scale.\n"
    )

    path = os.path.join(output_dir, "EVIDENCE_REPORT.md")
    with open(path, "w") as f:
        f.write("".join(report))
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose OOD mask-latent shift between GT MaskEncoder latents and inference latents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl_path", type=str, default=DEFAULT_TRAIN_JSONL, help="Training JSONL with output_mask.")
    parser.add_argument("--sample_id", type=str, default=None, help="Optional sample/patient ID to diagnose.")
    parser.add_argument("--model_path", type=str, default="Shitao/OmniGen-v1", help="Base OmniGen model.")
    parser.add_argument("--lora_path", type=str, default="results/30000", help="Option A LoRA checkpoint directory.")
    parser.add_argument("--mask_modules_path", type=str, default="results/30000/mask_modules.bin", help="Option A mask_modules.bin.")
    parser.add_argument("--output_dir", type=str, default="latent_shift_diagnostic", help="Directory for images/report.")
    parser.add_argument("--mask_key", type=str, default="mask", help="Key inside GT mask NPZ.")
    parser.add_argument("--mask_latent_channels", type=int, default=4, help="Mask latent channel count.")
    parser.add_argument("--inference_steps", type=int, default=50, help="ODE inference steps.")
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.5, help="CFG text guidance for CFG run.")
    parser.add_argument("--cfg_img_guidance_scale", type=float, default=2.0, help="Image guidance for CFG run.")
    parser.add_argument("--nocfg_guidance_scale", type=float, default=1.0, help="Text guidance for no-CFG run.")
    parser.add_argument("--nocfg_img_guidance_scale", type=float, default=1.0, help="Image guidance for no-CFG run.")
    parser.add_argument("--seed", type=int, default=42, help="Shared random seed for both inference runs.")
    parser.add_argument("--max_image_size", type=int, default=1024, help="OmniGen processor max input image size.")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="Unused except for module API compatibility.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Inference dtype.")
    parser.add_argument("--use_kv_cache", action="store_true", help="Use the standard KV cache during ODE inference.")
    parser.add_argument("--offload_kv_cache", action="store_true", help="Offload KV cache if --use_kv_cache is set.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")
    args.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    records = load_jsonl(args.jsonl_path)
    sample = pick_sample(records, args.sample_id)
    sample_id = canonical_sample_id(sample)
    print(f"Diagnosing sample: {sample_id}")

    pipe = load_pipeline(args, device)

    gt_mask = load_mask_cont(sample["output_mask"], args.mask_key, device)
    with torch.no_grad():
        z_true = pipe.mask_encoder(gt_mask)
        recon_true = pipe.mask_decoder(z_true)

    results = {"true": {"stats": tensor_stats(z_true)}}
    print("Z_m0_true stats:", results["true"]["stats"])

    true_grid = os.path.join(args.output_dir, "mask_true_recon_grid.png")
    true_union = os.path.join(args.output_dir, "mask_true_recon_union.png")
    save_mask_grid(recon_true, true_grid, "MaskDecoder(MaskEncoder(GT mask))")
    save_mask_union(recon_true, true_union)
    results["true"]["grid_path"] = true_grid
    results["true"]["union_path"] = true_union

    z_cfg, img_cfg = run_joint_inference_return_mask_latent(
        pipe,
        sample,
        args.cfg_guidance_scale,
        args.cfg_img_guidance_scale,
        args,
    )
    with torch.no_grad():
        mask_cfg = pipe.mask_decoder(z_cfg.to(device))
    cfg_grid = os.path.join(args.output_dir, "mask_cfg_grid.png")
    cfg_union = os.path.join(args.output_dir, "mask_cfg_union.png")
    cfg_img_path = os.path.join(args.output_dir, "generated_cfg.png")
    save_mask_grid(mask_cfg, cfg_grid, f"MaskDecoder(Z_hat_cfg), CFG={args.cfg_guidance_scale}")
    save_mask_union(mask_cfg, cfg_union)
    img_cfg.save(cfg_img_path)
    results["cfg"] = {
        "stats": tensor_stats(z_cfg),
        "l2_to_true": l2_distance(z_cfg.cpu(), z_true.cpu()),
        "grid_path": cfg_grid,
        "union_path": cfg_union,
        "generated_path": cfg_img_path,
    }
    print("Z_m0_cfg stats:", results["cfg"]["stats"])
    print("L2(cfg,true):", results["cfg"]["l2_to_true"])

    z_nocfg, img_nocfg = run_joint_inference_return_mask_latent(
        pipe,
        sample,
        args.nocfg_guidance_scale,
        args.nocfg_img_guidance_scale,
        args,
    )
    with torch.no_grad():
        mask_nocfg = pipe.mask_decoder(z_nocfg.to(device))
    nocfg_grid = os.path.join(args.output_dir, "mask_nocfg_grid.png")
    nocfg_union = os.path.join(args.output_dir, "mask_nocfg_union.png")
    nocfg_img_path = os.path.join(args.output_dir, "generated_nocfg.png")
    save_mask_grid(mask_nocfg, nocfg_grid, f"MaskDecoder(Z_hat_nocfg), guidance=1.0")
    save_mask_union(mask_nocfg, nocfg_union)
    img_nocfg.save(nocfg_img_path)
    results["nocfg"] = {
        "stats": tensor_stats(z_nocfg),
        "l2_to_true": l2_distance(z_nocfg.cpu(), z_true.cpu()),
        "grid_path": nocfg_grid,
        "union_path": nocfg_union,
        "generated_path": nocfg_img_path,
    }
    print("Z_m0_nocfg stats:", results["nocfg"]["stats"])
    print("L2(nocfg,true):", results["nocfg"]["l2_to_true"])

    metrics_path = os.path.join(args.output_dir, "latent_shift_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "sample_id": sample_id,
                "sample": sample,
                "results": results,
                "config": {k: str(v) if k == "dtype" else v for k, v in vars(args).items()},
            },
            f,
            indent=2,
        )
    report_path = write_report(args.output_dir, sample, results)
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
