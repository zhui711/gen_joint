#!/usr/bin/env python3
"""
Standalone attention heatmap extractor for joint image-mask co-generation.

This diagnostic is intentionally narrow:
  - run only a few selected samples,
  - hook one LLM self-attention layer,
  - extract image-token query -> mask-token key attention,
  - save heatmap overlays.

It does not compute PSNR, SSIM, FID, Dice, IoU, or any other metric.
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from test_joint_mask import (
    initialize_joint_mask_modules,
    load_jsonl,
    setup_logging,
    _get_inner_omnigen_model,
)


IMAGE_SIZE = 256
NUM_IMAGE_TOKENS = 256
NUM_MASK_TOKENS = 256
GRID_SIZE = 16


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
        return f"{parts[-2]}_{os.path.splitext(parts[-1])[0]}"
    return os.path.splitext(os.path.basename(gt_path))[0]


def select_records(records: Sequence[Dict], sample_ids: Optional[str], max_samples: Optional[int]) -> List[Dict]:
    if sample_ids:
        requested = {x.strip() for x in sample_ids.split(",") if x.strip()}
        selected = [rec for rec in records if requested.intersection(sample_id_candidates(rec))]
        found = {sid for rec in selected for sid in sample_id_candidates(rec)}
        missing = sorted(requested - found)
        if missing:
            print(f"WARNING: requested sample_ids not found: {', '.join(missing)}")
        return selected
    if max_samples is not None:
        return list(records[:max_samples])
    return list(records)


def _extract_attention_tensor(module_output) -> Optional[torch.Tensor]:
    if isinstance(module_output, tuple) and len(module_output) >= 2:
        attn = module_output[1]
        if isinstance(attn, torch.Tensor):
            return attn
    return None


def resolve_layer_index(num_layers: int, layer_index: int) -> int:
    resolved = layer_index if layer_index >= 0 else num_layers + layer_index
    if resolved < 0 or resolved >= num_layers:
        raise IndexError(f"layer_index={layer_index} resolves to {resolved}, but model has {num_layers} layers.")
    return resolved


def register_attention_hook(pipe, layer_index: int, capture: Dict[str, torch.Tensor]):
    inner = _get_inner_omnigen_model(pipe.model)
    if not hasattr(inner, "llm") or not hasattr(inner.llm, "layers"):
        raise AttributeError("Could not locate model.llm.layers for attention hook registration.")

    resolved = resolve_layer_index(len(inner.llm.layers), layer_index)
    inner.llm.config.output_attentions = True
    if hasattr(inner.llm.config, "_attn_implementation"):
        inner.llm.config._attn_implementation = "eager"
    if hasattr(inner.llm.config, "attn_implementation"):
        inner.llm.config.attn_implementation = "eager"

    target = inner.llm.layers[resolved].self_attn

    def hook(_module, _inputs, output):
        attn = _extract_attention_tensor(output)
        if attn is not None:
            capture["attn"] = attn.detach().float().cpu()

    return target.register_forward_hook(hook), resolved


def image_to_mask_attention_heatmap(attn: torch.Tensor) -> np.ndarray:
    """
    Convert attention [batch, heads, query, key] into one 256x256 heatmap.

    The OmniGen joint suffix is [image_tokens, mask_tokens]. With KV cache
    disabled, the final 512 tokens are exactly [256 image tokens, 256 mask
    tokens] for 256x256 generation.
    """
    if attn.ndim != 4:
        raise ValueError(f"Expected attention tensor [B, heads, Q, K], got {tuple(attn.shape)}")
    if attn.shape[-1] < NUM_IMAGE_TOKENS + NUM_MASK_TOKENS:
        raise ValueError(f"Attention key length {attn.shape[-1]} is too short for 256 image + 256 mask tokens.")
    if attn.shape[-2] < NUM_IMAGE_TOKENS + NUM_MASK_TOKENS:
        raise ValueError(f"Attention query length {attn.shape[-2]} is too short for 256 image + 256 mask tokens.")

    suffix_start = attn.shape[-1] - (NUM_IMAGE_TOKENS + NUM_MASK_TOKENS)
    img_start = suffix_start
    img_end = img_start + NUM_IMAGE_TOKENS
    mask_start = img_end
    mask_end = mask_start + NUM_MASK_TOKENS

    # First batch row is the prompt-conditioned branch when CFG is packed as
    # [conditional, unconditional, image-conditional].
    sample_attn = attn[0]
    img_q_mask_k = sample_attn[:, img_start:img_end, mask_start:mask_end]
    avg_heads = img_q_mask_k.mean(dim=0)       # [256 image queries, 256 mask keys]
    per_image_token = avg_heads.mean(dim=1)    # [256]

    heat = per_image_token.numpy().reshape(GRID_SIZE, GRID_SIZE)
    heat = heat - np.min(heat)
    denom = np.max(heat)
    if denom > 0:
        heat = heat / denom
    heat = cv2.resize(heat.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    return np.clip(heat, 0.0, 1.0)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    image_np = np.asarray(image.convert("RGB"))
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (1.0 - alpha) * image_np.astype(np.float32) + alpha * colored.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_heatmap_outputs(
    output_dir: str,
    sample_id: str,
    layer_index: int,
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float,
    save_generated: bool,
):
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{sample_id}_layer{layer_index}_attn"
    overlay = overlay_heatmap(image, heatmap, alpha=alpha)
    Image.fromarray(overlay).save(os.path.join(output_dir, f"{stem}.png"))
    np.save(os.path.join(output_dir, f"{stem}.npy"), heatmap.astype(np.float32))
    if save_generated:
        image.save(os.path.join(output_dir, f"{sample_id}_generated.png"))


def load_pipeline(args, device: torch.device):
    from OmniGen import OmniGenPipeline

    pipe = OmniGenPipeline.from_pretrained(args.model_path)
    pipe.model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)
    if args.lora_path:
        logging.info(f"Merging LoRA from {args.lora_path}")
        pipe.merge_lora(args.lora_path)
    initialize_joint_mask_modules(pipe, args, logging.getLogger("extract-attn"))
    pipe.to(device)
    return pipe


def run_sample(pipe, record: Dict, args, device: torch.device, resolved_layer: Optional[int]) -> int:
    sample_id = canonical_sample_id(record)
    capture: Dict[str, torch.Tensor] = {}
    handle = None
    try:
        handle, resolved = register_attention_hook(pipe, args.layer_index, capture)
        if resolved_layer is None:
            print(f"Using resolved layer index: {resolved}")

        result = pipe(
            prompt=record["instruction"],
            input_images=record.get("input_images") or None,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            img_guidance_scale=args.img_guidance_scale,
            max_input_image_size=args.max_image_size,
            use_input_image_size_as_output=False,
            use_kv_cache=False,
            offload_kv_cache=False,
            separate_cfg_infer=False,
            offload_model=False,
            seed=args.seed,
            save_mask=False,
            mask_threshold=args.mask_threshold,
        )
    finally:
        if handle is not None:
            handle.remove()

    images = result[0] if isinstance(result, tuple) else result
    if not images:
        raise RuntimeError(f"No generated image returned for {sample_id}")
    if "attn" not in capture:
        raise RuntimeError(
            "No attention weights were captured. Confirm that this transformers/Phi3 setup supports "
            "output_attentions=True with eager attention."
        )

    heatmap = image_to_mask_attention_heatmap(capture["attn"])
    layer_for_name = resolve_layer_index(len(_get_inner_omnigen_model(pipe.model).llm.layers), args.layer_index)
    save_heatmap_outputs(
        args.output_dir,
        sample_id,
        args.layer_index,
        images[0],
        heatmap,
        args.heatmap_alpha,
        args.save_generated,
    )
    torch.cuda.empty_cache()
    return layer_for_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract image-token to mask-token attention heatmaps for selected joint-mask samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default="Shitao/OmniGen-v1", help="Base OmniGen model path or HF repo.")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA checkpoint directory.")
    parser.add_argument("--mask_modules_path", type=str, required=True, help="Path to mask_modules.bin.")
    parser.add_argument("--mask_latent_channels", type=int, default=4, help="Mask latent channels.")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Test JSONL containing instruction/input/output paths.")
    parser.add_argument("--output_dir", type=str, default="analysis_attention_heatmaps", help="Output directory for heatmaps.")
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma-separated sample IDs/patient IDs to process.")
    parser.add_argument("--max_samples", type=int, default=4, help="Process first N samples when --sample_ids is not provided.")
    parser.add_argument("--layer_index", type=int, default=-1, help="LLM layer index to hook; negative values count from the end.")
    parser.add_argument("--inference_steps", type=int, default=50, help="Diffusion denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Text CFG guidance scale.")
    parser.add_argument("--img_guidance_scale", type=float, default=2.0, help="Image guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="Mask threshold passed to the pipeline.")
    parser.add_argument("--max_image_size", type=int, default=1024, help="Maximum input image size for preprocessing.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for extraction.")
    parser.add_argument("--heatmap_alpha", type=float, default=0.45, help="Heatmap overlay alpha.")
    parser.add_argument("--save_generated", action="store_true", help="Also save the generated image beside the heatmap.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")
    device = torch.device(args.device)

    records = load_jsonl(args.jsonl_path)
    selected = select_records(records, args.sample_ids, args.max_samples)
    if not selected:
        raise RuntimeError("No samples selected for attention extraction.")

    with open(os.path.join(args.output_dir, "extract_attention_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    pipe = load_pipeline(args, device)
    resolved_layer = None
    failures = 0
    for record in tqdm(selected, desc="Extracting attention"):
        sample_id = canonical_sample_id(record)
        try:
            resolved_layer = run_sample(pipe, record, args, device, resolved_layer)
        except Exception as exc:
            failures += 1
            logging.exception(f"Failed to extract attention for {sample_id}: {exc}")

    print(f"Saved attention heatmaps to {args.output_dir}")
    print(f"Processed {len(selected) - failures}/{len(selected)} samples successfully.")
    if failures:
        print(f"Failures: {failures}. See run.log for details.")


if __name__ == "__main__":
    main()
