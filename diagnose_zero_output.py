#!/usr/bin/env python3
"""
Pinpoint why pred_mask is exactly zero during joint-mask inference.

This is a forensic script only. It does not modify training or inference code.
"""

import argparse
import gc
import json
import os
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch

from OmniGen import OmniGenProcessor
from test_joint_mask import initialize_joint_mask_modules, load_jsonl, setup_logging, _get_inner_omnigen_model


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
    usable = [r for r in records if r.get("instruction") and r.get("input_images") and r.get("output_mask") and os.path.exists(r["output_mask"])]
    if not usable:
        raise RuntimeError("No usable records found.")
    if sample_id is None:
        return usable[0]
    requested = {x.strip() for x in sample_id.split(",") if x.strip()}
    for record in usable:
        if requested.intersection(sample_id_candidates(record)):
            return record
    raise RuntimeError(f"Requested sample not found: {sample_id}")


def stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "std": float(t.std(unbiased=False).item()),
        "mean": float(t.mean().item()),
        "abs_sum": float(t.abs().sum().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def format_stats(name: str, s: Dict[str, float]) -> str:
    return f"{name}: shape={s['shape']} mean={s['mean']:+.6f} std={s['std']:.6f} min={s['min']:+.6f} max={s['max']:+.6f} abs_sum={s['abs_sum']:.6f}"


def load_pipeline(args, device: torch.device):
    from OmniGen import OmniGenPipeline

    pipe = OmniGenPipeline.from_pretrained(args.model_path)
    pipe.model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)
    if args.lora_path:
        print(f"Loading LoRA: {args.lora_path}")
        pipe.merge_lora(args.lora_path)
    initialize_joint_mask_modules(pipe, args, setup_logging(args.output_dir) or __import__("logging").getLogger("zero-mask"))
    pipe.to(device)
    pipe.model.eval()
    pipe.vae.eval()
    pipe.mask_encoder.eval()
    pipe.mask_decoder.eval()
    return pipe


def weight_summary(model) -> Dict[str, Dict[str, float]]:
    inner = _get_inner_omnigen_model(model)
    result = {}
    for name in ["final_layer", "mask_final_layer", "x_embedder", "mask_x_embedder"]:
        module = getattr(inner, name, None)
        if module is None:
            result[name] = {"present": False}
            continue
        state = module.state_dict()
        abs_sum = 0.0
        param_count = 0
        for v in state.values():
            abs_sum += float(v.detach().float().abs().sum().item())
            param_count += v.numel()
        result[name] = {
            "present": True,
            "param_count": param_count,
            "abs_sum": abs_sum,
            "state_keys": list(state.keys()),
        }
    return result


def state_dict_keys(path: str) -> List[str]:
    state = torch.load(path, map_location="cpu")
    return list(state.keys())


@torch.no_grad()
def trace_one_forward(pipe, sample: Dict, args) -> Dict:
    prompt = [sample["instruction"]]
    input_images = [sample["input_images"]]
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

    generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
    z_img = torch.randn(1, 4, latent_size_h, latent_size_w, device=pipe.device, generator=generator)
    z_mask = torch.randn(1, 4, latent_size_h, latent_size_w, device=pipe.device, generator=generator)
    z_img = torch.cat([z_img] * 3, dim=0).to(dtype)
    z_mask = torch.cat([z_mask] * 3, dim=0).to(dtype)

    input_img_latents = [pipe.vae_encode(img.to(pipe.device), dtype) for img in input_data["input_pixel_values"]]
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

    inner = _get_inner_omnigen_model(pipe.model)

    raw = {}

    orig_forward = inner.forward

    def wrapped_forward(*f_args, **f_kwargs):
        out = orig_forward(*f_args, **f_kwargs)
        model_out = out[0] if isinstance(out, tuple) else out
        if isinstance(model_out, tuple) and len(model_out) == 2:
            img_out, mask_out = model_out
            raw["model_out_type"] = "joint"
            raw["img_out"] = stats(img_out)
            raw["mask_out"] = stats(mask_out)
        else:
            raw["model_out_type"] = "single"
            raw["model_out"] = stats(model_out)
        return out

    inner.forward = wrapped_forward
    try:
        pred, _ = pipe.model.forward_with_cfg(
            z_img,
            torch.zeros(size=(len(z_img),), device=z_img.device) + 0.0,
            past_key_values=None if not args.use_kv_cache else [],
            x_mask=z_mask,
            **model_kwargs,
        )
    finally:
        inner.forward = orig_forward

    if isinstance(pred, tuple) and len(pred) == 2:
        pred_img, pred_mask = pred
        raw["pred_img"] = stats(pred_img)
        raw["pred_mask"] = stats(pred_mask)
    else:
        raw["pred"] = stats(pred)

    return raw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose exact zero output in joint mask inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl_path", type=str, default=DEFAULT_JSONL)
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="Shitao/OmniGen-v1")
    parser.add_argument("--lora_path", type=str, default="results/30000")
    parser.add_argument("--mask_modules_path", type=str, default="results/30000/mask_modules.bin")
    parser.add_argument("--output_dir", type=str, default="diagnose_zero_output")
    parser.add_argument("--mask_latent_channels", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--img_guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_image_size", type=int, default=1024)
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

    weights = weight_summary(pipe.model)
    print("\n=== Loaded Weights ===")
    for name, info in weights.items():
        if not info.get("present"):
            print(f"{name}: not present")
            continue
        print(f"{name}: abs_sum={info['abs_sum']:.6f} param_count={info['param_count']}")
        if name in {"final_layer", "mask_final_layer"}:
            print(f"  keys: {info['state_keys']}")

    mask_modules_keys = state_dict_keys(args.mask_modules_path)
    print("\n=== mask_modules.bin keys ===")
    for k in mask_modules_keys:
        print(k)

    print("\n=== Forward Trace ===")
    trace = trace_one_forward(pipe, sample, args)
    for k, v in trace.items():
        if isinstance(v, dict) and "shape" in v:
            print(format_stats(k, v))
        else:
            print(f"{k}: {v}")

    verdict = {
        "sample_id": canonical_sample_id(sample),
        "weights": weights,
        "mask_modules_keys": mask_modules_keys,
        "forward_trace": trace,
    }
    out_path = os.path.join(args.output_dir, "zero_mask_verdict.json")
    with open(out_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nWrote verdict JSON: {out_path}")


if __name__ == "__main__":
    main()
