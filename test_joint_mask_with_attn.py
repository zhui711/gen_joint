#!/usr/bin/env python3
"""
Joint image-mask inference with last-layer image-to-mask attention heatmaps.

The hook is attached to llm.layers[-1].self_attn.  For each generated sample,
we extract attention where Q = generated image tokens and K = generated mask
tokens, average over heads and mask-token keys, reshape the image-token vector
to a 16x16 grid for 256x256 images, and save a heatmap overlay.
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

from test_joint_mask import (
    build_generation_groups,
    derive_mask_save_path,
    derive_save_path,
    initialize_joint_mask_modules,
    load_jsonl,
    setup_logging,
    _get_inner_omnigen_model,
)


IMAGE_SIZE = 256


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


def filter_records(records: Sequence[Dict], sample_ids: Optional[str], max_samples: Optional[int]) -> List[Dict]:
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


def _extract_attention_tensor(module_output) -> Optional[torch.Tensor]:
    if isinstance(module_output, tuple) and len(module_output) >= 2:
        attn = module_output[1]
        if isinstance(attn, torch.Tensor):
            return attn
    return None


def register_last_self_attention_hook(pipe, capture: Dict[str, torch.Tensor]):
    inner = _get_inner_omnigen_model(pipe.model)
    if not hasattr(inner, "llm") or not hasattr(inner.llm, "layers"):
        raise AttributeError("Could not locate inner.llm.layers for attention hook registration.")
    if len(inner.llm.layers) == 0:
        raise ValueError("inner.llm.layers is empty.")

    inner.llm.config.output_attentions = True
    if hasattr(inner.llm.config, "_attn_implementation"):
        inner.llm.config._attn_implementation = "eager"
    if hasattr(inner.llm.config, "attn_implementation"):
        inner.llm.config.attn_implementation = "eager"

    target = inner.llm.layers[-1].self_attn

    def hook(_module, _inputs, output):
        attn = _extract_attention_tensor(output)
        if attn is not None:
            capture["attn"] = attn.detach().float().cpu()

    return target.register_forward_hook(hook)


def attention_to_heatmaps(
    attn: torch.Tensor,
    num_samples: int,
    target_height: int,
    target_width: int,
    force_grid_size: int = 16,
) -> List[np.ndarray]:
    if attn.ndim != 4:
        raise ValueError(f"Expected attention tensor [B,H,Q,K], got {tuple(attn.shape)}")

    num_img_tokens = (target_height // 8) * (target_width // 8) // 4
    num_mask_tokens = num_img_tokens
    key_len = attn.shape[-1]
    query_len = attn.shape[-2]
    suffix_len = num_img_tokens + num_mask_tokens
    if key_len < suffix_len or query_len < suffix_len:
        raise ValueError(
            f"Attention tensor too short for image+mask suffix: attn={tuple(attn.shape)}, "
            f"img_tokens={num_img_tokens}, mask_tokens={num_mask_tokens}"
        )

    img_start = key_len - suffix_len
    img_end = key_len - num_mask_tokens
    mask_start = key_len - num_mask_tokens
    mask_end = key_len

    batch_heatmaps = []
    for sample_idx in range(num_samples):
        # The collator orders CFG batches as [conditional, unconditional, image-conditional].
        # The first num_samples rows are therefore the prompt-conditioned samples.
        sample_attn = attn[sample_idx]
        sub = sample_attn[:, img_start:img_end, mask_start:mask_end]
        heat = sub.mean(dim=(0, 2)).numpy()

        if num_img_tokens == force_grid_size * force_grid_size:
            grid_h = grid_w = force_grid_size
        else:
            grid_h = max(1, target_height // 16)
            grid_w = max(1, target_width // 16)
            if grid_h * grid_w != num_img_tokens:
                side = int(round(math.sqrt(num_img_tokens)))
                if side * side != num_img_tokens:
                    raise ValueError(f"Cannot reshape {num_img_tokens} image-token attentions into a grid.")
                grid_h = grid_w = side

        heat = heat.reshape(grid_h, grid_w)
        heat = heat - np.min(heat)
        denom = np.max(heat)
        if denom > 0:
            heat = heat / denom
        heat = cv2.resize(heat.astype(np.float32), (256, 256), interpolation=cv2.INTER_CUBIC)
        batch_heatmaps.append(np.clip(heat, 0.0, 1.0))
    return batch_heatmaps


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.clip(heatmap_resized * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (1.0 - alpha) * rgb.astype(np.float32) + alpha * colored.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_attention_outputs(
    item: Dict,
    image: Image.Image,
    heatmap: np.ndarray,
    output_dir: str,
    alpha: float,
):
    rel_gen_path = derive_save_path(item["output_image"], output_dir)
    stem, _ = os.path.splitext(rel_gen_path)
    heatmap_png = f"{stem}_lastlayer_imgQ_maskK_attn.png"
    heatmap_npy = f"{stem}_lastlayer_imgQ_maskK_attn.npy"
    os.makedirs(os.path.dirname(heatmap_png), exist_ok=True)

    overlay = overlay_heatmap(image, heatmap, alpha=alpha)
    Image.fromarray(overlay).save(heatmap_png)
    np.save(heatmap_npy, heatmap.astype(np.float32))

    fig_path = f"{stem}_lastlayer_imgQ_maskK_attn_figure.png"
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    axes[0].imshow(image.convert("RGB"))
    axes[0].set_title("Generated")
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Image Q -> Mask K")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generation_worker_with_attn(rank: int, num_gpus: int, chunks: List[List[Dict]], args_dict: dict):
    args = argparse.Namespace(**args_dict)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    setup_logging(args.output_dir, rank=rank)
    logger = logging.getLogger(f"attn-worker-{rank}")
    logger.info(f"[GPU {rank}] Starting attention extraction on {device}; samples={len(chunks[rank])}")

    from OmniGen import OmniGenPipeline

    pipe = OmniGenPipeline.from_pretrained(args.model_path)
    pipe.model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)
    if args.lora_path:
        logger.info(f"[GPU {rank}] Merging LoRA from {args.lora_path}")
        pipe.merge_lora(args.lora_path)
    initialize_joint_mask_modules(pipe, args, logger)
    pipe.to(device)

    success_count = 0
    fail_count = 0
    mask_output_dir = os.path.join(args.output_dir, "masks")
    if args.save_masks:
        os.makedirs(mask_output_dir, exist_ok=True)

    my_data = chunks[rank]
    num_batches = math.ceil(len(my_data) / args.batch_size)
    pbar = tqdm(range(num_batches), desc=f"[GPU {rank} attn]", position=rank, leave=True)

    for batch_idx in pbar:
        batch = my_data[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        try:
            generation_groups = build_generation_groups(batch, args)
        except Exception as exc:
            logger.error(f"[GPU {rank}] preprocessing error: {exc}", exc_info=True)
            fail_count += len(batch)
            continue

        for target_height, target_width, use_input_size_output, subbatch in generation_groups:
            saved_in_subbatch = 0
            hook_handle = None
            capture: Dict[str, torch.Tensor] = {}
            try:
                prompts = [item["instruction"] for item in subbatch]
                input_imgs = [item["input_images"] for item in subbatch]

                hook_handle = register_last_self_attention_hook(pipe, capture)
                call_kwargs = dict(
                    height=target_height,
                    width=target_width,
                    num_inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    img_guidance_scale=args.img_guidance_scale,
                    max_input_image_size=args.max_image_size,
                    use_input_image_size_as_output=use_input_size_output,
                    use_kv_cache=False,
                    offload_kv_cache=False,
                    separate_cfg_infer=False,
                    offload_model=False,
                    seed=args.seed,
                    save_mask=args.save_masks,
                    mask_threshold=args.mask_threshold,
                )
                if len(subbatch) == 1:
                    result = pipe(prompt=prompts[0], input_images=input_imgs[0], **call_kwargs)
                else:
                    result = pipe(prompt=prompts, input_images=input_imgs, **call_kwargs)
                if hook_handle is not None:
                    hook_handle.remove()
                    hook_handle = None

                if args.save_masks:
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise RuntimeError("Pipeline did not return (images, masks) with --save_masks.")
                    outputs, masks = result
                else:
                    outputs = result[0] if isinstance(result, tuple) else result
                    masks = None

                if "attn" not in capture:
                    raise RuntimeError(
                        "No attention weights captured. Ensure transformers Phi3 attention is using eager "
                        "attention and output_attentions=True."
                    )
                heatmaps = attention_to_heatmaps(capture["attn"], len(subbatch), target_height, target_width)

                if len(outputs) != len(subbatch):
                    raise RuntimeError(f"Pipeline returned {len(outputs)} images for {len(subbatch)} samples.")

                for idx, (img, item) in enumerate(zip(outputs, subbatch)):
                    save_path = derive_save_path(item["output_image"], args.output_dir)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    img.save(save_path)

                    if masks is not None:
                        mask_save_path = derive_mask_save_path(item["output_image"], mask_output_dir)
                        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                        np.savez_compressed(mask_save_path, mask=masks[idx].cpu().numpy())

                    save_attention_outputs(item, img, heatmaps[idx], args.output_dir, args.heatmap_alpha)
                    success_count += 1
                    saved_in_subbatch += 1

            except Exception as exc:
                logger.error(
                    f"[GPU {rank}] attention save error for subgroup size={len(subbatch)}: {exc}",
                    exc_info=True,
                )
                fail_count += len(subbatch) - saved_in_subbatch
            finally:
                if hook_handle is not None:
                    hook_handle.remove()

            torch.cuda.empty_cache()
            gc.collect()
        pbar.set_postfix(ok=success_count, fail=fail_count)

    stats = {
        "rank": rank,
        "total": len(my_data),
        "success": success_count,
        "fail": fail_count,
    }
    with open(os.path.join(args.output_dir, f"attn_stats_gpu{rank}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"[GPU {rank}] Done: {success_count}/{len(my_data)} success, {fail_count} failed")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CXR Joint OmniGen inference with image-token to mask-token attention extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default="Shitao/OmniGen-v1")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--mask_modules_path", type=str, required=True)
    parser.add_argument("--mask_latent_channels", type=int, default=4)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_ids", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--img_guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--keep_raw_resolution", action="store_true")
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--heatmap_alpha", type=float, default=0.45)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    print(f"Loading JSONL from {args.jsonl_path} ...")
    all_data = load_jsonl(args.jsonl_path)
    all_data = filter_records(all_data, args.sample_ids, args.max_samples)
    print(f"  Selected samples: {len(all_data)}")
    if not all_data:
        raise RuntimeError("No samples selected for attention extraction.")

    config_path = os.path.join(args.output_dir, "attn_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    chunk_size = math.ceil(len(all_data) / args.num_gpus)
    chunks = [all_data[i * chunk_size : (i + 1) * chunk_size] for i in range(args.num_gpus)]
    for rank, chunk in enumerate(chunks):
        print(f"  GPU {rank}: {len(chunk)} samples")

    t0 = time.time()
    if args.num_gpus == 1:
        generation_worker_with_attn(0, 1, chunks, vars(args))
    else:
        mp.spawn(
            generation_worker_with_attn,
            args=(args.num_gpus, chunks, vars(args)),
            nprocs=args.num_gpus,
            join=True,
        )
    print(f"Attention extraction completed in {(time.time() - t0) / 60.0:.2f} min")


if __name__ == "__main__":
    main()
