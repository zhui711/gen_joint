#!/usr/bin/env python3
"""
Forensic trace for the joint image-mask ODE inference loop.

Runs a single sample for a small number of steps and prints z_img/z_mask
statistics at each step. This does not modify training or inference code.
"""

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List

import torch

from OmniGen import OmniGenProcessor, OmniGenScheduler
from OmniGen.scheduler import OmniGenCache
from test_joint_mask import initialize_joint_mask_modules, load_jsonl, setup_logging


DEFAULT_JSONL = "/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl"


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
    parts = gt_path.replace("\\", "/").rstrip("/").split("/")
    patient = parts[-2] if len(parts) >= 2 else ""
    stem = os.path.splitext(parts[-1])[0] if parts else ""
    return [x for x in [record.get("sample_id"), record.get("id"), patient, stem, f"{patient}_{stem}"] if x]


def pick_sample(records: List[Dict], sample_id: str = None) -> Dict:
    usable = [r for r in records if r.get("instruction") and r.get("input_images")]
    if not usable:
        raise RuntimeError("No usable records with instruction and input_images.")
    if sample_id is None:
        return usable[0]
    requested = {x.strip() for x in sample_id.split(",") if x.strip()}
    for record in usable:
        if requested.intersection(sample_id_candidates(record)):
            return record
    raise RuntimeError(f"Requested sample not found: {sample_id}")


def stats(tensor: torch.Tensor) -> Dict[str, float]:
    t = tensor.detach().float()
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def fmt(s: Dict[str, float]) -> str:
    return f"mean={s['mean']:+.6f} std={s['std']:.6f} min={s['min']:+.6f} max={s['max']:+.6f}"


def load_pipeline(args, device: torch.device):
    from OmniGen import OmniGenPipeline

    pipe = OmniGenPipeline.from_pretrained(args.model_path)
    pipe.model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)
    if args.lora_path:
        print(f"Loading LoRA: {args.lora_path}")
        pipe.merge_lora(args.lora_path)
    initialize_joint_mask_modules(pipe, args, setup_logging(args.output_dir) or __import__("logging").getLogger("debug-loop"))
    pipe.to(device)
    pipe.model.eval()
    pipe.vae.eval()
    pipe.mask_encoder.eval()
    pipe.mask_decoder.eval()
    return pipe


@torch.no_grad()
def build_initial_state(pipe, sample: Dict, args):
    prompt = [sample["instruction"]]
    input_images = [sample.get("input_images") or None]
    height = width = 256
    dtype = args.dtype

    if args.max_image_size != pipe.processor.max_image_size:
        pipe.processor = OmniGenProcessor(pipe.processor.text_tokenizer, max_image_size=args.max_image_size)

    pipe.model.to(dtype)
    pipe.disable_model_cpu_offload()

    latent_size_h, latent_size_w = height // 8, width // 8
    num_mask_tokens = (latent_size_h // pipe.model.patch_size) * (latent_size_w // pipe.model.patch_size)
    input_data = pipe.processor(
        prompt,
        input_images,
        height=height,
        width=width,
        use_img_cfg=True,
        separate_cfg_input=False,
        use_input_image_size_as_output=False,
        num_mask_tokens=num_mask_tokens,
    )

    generator = torch.Generator(device=pipe.device).manual_seed(args.seed) if args.seed is not None else None
    latents = torch.randn(1, 4, latent_size_h, latent_size_w, device=pipe.device, generator=generator)
    mask_latents = torch.randn(1, 4, latent_size_h, latent_size_w, device=pipe.device, generator=generator)

    # Match OmniGenPipeline.__call__: pack [conditional, unconditional, image-conditional].
    z_img = torch.cat([latents] * 3, dim=0).to(dtype)
    z_mask = torch.cat([mask_latents] * 3, dim=0).to(dtype)

    input_img_latents = []
    for img in input_data["input_pixel_values"]:
        input_img_latents.append(pipe.vae_encode(img.to(pipe.device), dtype))

    model_kwargs = dict(
        input_ids=pipe.move_to_device(input_data["input_ids"]),
        input_img_latents=input_img_latents,
        input_image_sizes=deepcopy(input_data["input_image_sizes"]),
        attention_mask=pipe.move_to_device(input_data["attention_mask"]),
        position_ids=pipe.move_to_device(input_data["position_ids"]),
        cfg_scale=args.guidance_scale,
        img_cfg_scale=args.img_guidance_scale,
        use_img_cfg=True,
        use_kv_cache=args.use_kv_cache,
        offload_model=False,
    )
    return z_img, z_mask, model_kwargs


@torch.no_grad()
def trace_loop(pipe, z_img, z_mask, model_kwargs, args):
    scheduler = OmniGenScheduler(num_steps=args.num_steps)
    rows = []
    num_img_tokens = z_img.size(-1) * z_img.size(-2) // 4
    num_mask_tokens = z_mask.size(-1) * z_mask.size(-2) // 4
    num_total_gen_tokens = num_img_tokens + num_mask_tokens
    cache = OmniGenCache(num_total_gen_tokens, False) if args.use_kv_cache else None

    print("\nstep,phase,z_img_mean,z_img_std,z_mask_mean,z_mask_std,pred_img_std,pred_mask_std,dt")
    for step in range(args.num_steps):
        before_img = stats(z_img)
        before_mask = stats(z_mask)
        timesteps = torch.zeros(size=(len(z_img),), device=z_img.device) + scheduler.sigma[step]
        pred, cache = pipe.model.forward_with_cfg(
            z_img,
            timesteps,
            past_key_values=cache,
            x_mask=z_mask,
            **model_kwargs,
        )
        pred_img, pred_mask = pred
        pred_img_stats = stats(pred_img)
        pred_mask_stats = stats(pred_mask)
        dt = scheduler.sigma[step + 1] - scheduler.sigma[step]

        print(
            f"{step},before,{before_img['mean']:+.6f},{before_img['std']:.6f},"
            f"{before_mask['mean']:+.6f},{before_mask['std']:.6f},"
            f"{pred_img_stats['std']:.6f},{pred_mask_stats['std']:.6f},{float(dt):+.6f}"
        )

        z_img = z_img + dt * pred_img
        z_mask = z_mask + dt * pred_mask

        after_img = stats(z_img)
        after_mask = stats(z_mask)
        print(
            f"{step},after,{after_img['mean']:+.6f},{after_img['std']:.6f},"
            f"{after_mask['mean']:+.6f},{after_mask['std']:.6f},"
            f"{pred_img_stats['std']:.6f},{pred_mask_stats['std']:.6f},{float(dt):+.6f}"
        )

        rows.append(
            {
                "step": step,
                "dt": float(dt),
                "before": {"z_img": before_img, "z_mask": before_mask},
                "after": {"z_img": after_img, "z_mask": after_mask},
                "pred_img": pred_img_stats,
                "pred_mask": pred_mask_stats,
            }
        )

        if step == 0 and args.use_kv_cache:
            model_kwargs["input_ids"] = None
            model_kwargs["position_ids"] = scheduler.crop_position_ids_for_cache(model_kwargs["position_ids"], num_total_gen_tokens)
            model_kwargs["attention_mask"] = scheduler.crop_attention_mask_for_cache(model_kwargs["attention_mask"], num_total_gen_tokens)

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace z_img/z_mask stats during the first few joint ODE inference steps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl_path", type=str, default=DEFAULT_JSONL)
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="Shitao/OmniGen-v1")
    parser.add_argument("--lora_path", type=str, default="results/30000")
    parser.add_argument("--mask_modules_path", type=str, default="results/30000/mask_modules.bin")
    parser.add_argument("--output_dir", type=str, default="debug_inference_loop_output")
    parser.add_argument("--mask_latent_channels", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--img_guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--use_kv_cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")
    args.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    sample = pick_sample(load_jsonl(args.jsonl_path), args.sample_id)
    print(f"Tracing sample: {canonical_sample_id(sample)}")

    pipe = load_pipeline(args, device)
    z_img, z_mask, model_kwargs = build_initial_state(pipe, sample, args)
    rows = trace_loop(pipe, z_img, z_mask, model_kwargs, args)

    out_path = os.path.join(args.output_dir, "trace_stats.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "sample_id": canonical_sample_id(sample),
                "sample": sample,
                "trace": rows,
                "config": {k: str(v) if k == "dtype" else v for k, v in vars(args).items()},
            },
            f,
            indent=2,
        )
    print(f"\nWrote trace JSON: {out_path}")


if __name__ == "__main__":
    main()
