#!/usr/bin/env python3
"""Joint Image-Mask Co-Generation Inference Script.

This script loads a trained joint model (OmniGen + LoRA + mask modules)
and generates both edited images and 10-channel anatomy masks.

Usage:
  python test_joint_mask.py \
    --model_path Shitao/OmniGen-v1 \
    --lora_path results/joint_mask/checkpoints/0010000/ \
    --mask_modules_path results/joint_mask/checkpoints/0010000/mask_modules.bin \
    --jsonl_path data/cxr_synth_anno_mask_test.jsonl \
    --output_dir results/joint_mask_inference/
"""

import os
import sys
import gc
import json
import math
import argparse
from typing import List, Dict

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm


IMAGE_SIZE = 256


def derive_save_path(gt_path: str, output_dir: str) -> str:
    parts = gt_path.replace("\\", "/").rstrip("/").split("/")
    relative = os.path.join(parts[-2], parts[-1])
    return os.path.join(output_dir, relative)


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def generation_worker(rank: int, num_gpus: int, chunks: List[List[Dict]], args_dict: dict):
    args = argparse.Namespace(**args_dict)
    device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] Starting — device={device}, samples={len(chunks[rank])}")

    from OmniGen import OmniGenPipeline, OmniGen
    from OmniGen.mask_autoencoder import MaskEncoder, MaskDecoder

    # Load pipeline
    pipe = OmniGenPipeline.from_pretrained(args.model_path)

    # Initialize mask modules on the model
    pipe.model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)

    # Load LoRA
    if args.lora_path:
        print(f"[GPU {rank}] Merging LoRA from {args.lora_path}")
        pipe.merge_lora(args.lora_path)

    # Load mask modules
    if args.mask_modules_path:
        print(f"[GPU {rank}] Loading mask modules from {args.mask_modules_path}")
        mask_state = torch.load(args.mask_modules_path, map_location="cpu")

        # Load mask encoder/decoder
        mask_encoder = MaskEncoder(in_channels=10, latent_channels=args.mask_latent_channels)
        mask_decoder = MaskDecoder(latent_channels=args.mask_latent_channels, out_channels=10)

        enc_state = {k.replace("mask_encoder.", ""): v for k, v in mask_state.items() if k.startswith("mask_encoder.")}
        dec_state = {k.replace("mask_decoder.", ""): v for k, v in mask_state.items() if k.startswith("mask_decoder.")}
        if enc_state:
            mask_encoder.load_state_dict(enc_state)
        if dec_state:
            mask_decoder.load_state_dict(dec_state)

        pipe.mask_encoder = mask_encoder
        pipe.mask_decoder = mask_decoder

        # Load model-internal mask modules
        base_model = pipe.model
        if hasattr(base_model, 'base_model'):
            inner = base_model.base_model
            if hasattr(inner, 'model'):
                inner = inner.model
        else:
            inner = base_model

        emb_state = {k.replace("mask_x_embedder.", ""): v for k, v in mask_state.items() if k.startswith("mask_x_embedder.")}
        if emb_state and inner.mask_x_embedder is not None:
            inner.mask_x_embedder.load_state_dict(emb_state)

        fl_state = {k.replace("mask_final_layer.", ""): v for k, v in mask_state.items() if k.startswith("mask_final_layer.")}
        if fl_state and inner.mask_final_layer is not None:
            inner.mask_final_layer.load_state_dict(fl_state)

        if "image_modality_embed" in mask_state and inner.image_modality_embed is not None:
            inner.image_modality_embed.data.copy_(mask_state["image_modality_embed"])
        if "mask_modality_embed" in mask_state and inner.mask_modality_embed is not None:
            inner.mask_modality_embed.data.copy_(mask_state["mask_modality_embed"])

    pipe.to(device)

    my_data = chunks[rank]
    batch_size = args.batch_size
    success_count = 0

    num_batches = math.ceil(len(my_data) / batch_size)
    pbar = tqdm(range(num_batches), desc=f"[GPU {rank}]", position=rank, leave=True)

    mask_output_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(mask_output_dir, exist_ok=True)

    for batch_idx in pbar:
        batch_start = batch_idx * batch_size
        batch = my_data[batch_start : batch_start + batch_size]

        try:
            prompts = [item["instruction"] for item in batch]
            input_imgs = [item["input_images"] for item in batch]

            if len(prompts) == 1:
                result = pipe(
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
                    save_mask=args.save_masks,
                    mask_threshold=args.mask_threshold,
                )
            else:
                result = pipe(
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
                    save_mask=args.save_masks,
                    mask_threshold=args.mask_threshold,
                )

            if args.save_masks and isinstance(result, tuple):
                outputs, masks = result
            else:
                outputs = result if not isinstance(result, tuple) else result[0]
                masks = None

            for idx_in_batch, (img, item) in enumerate(zip(outputs, batch)):
                save_path = derive_save_path(item["output_image"], args.output_dir)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img.save(save_path)

                # Save mask as .npz
                if masks is not None:
                    mask_save_path = derive_save_path(
                        item["output_image"], mask_output_dir
                    ).replace(".png", ".npz")
                    os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                    mask_np = masks[idx_in_batch].numpy()  # (10, 256, 256)
                    np.savez_compressed(mask_save_path, mask=mask_np)

                success_count += 1

        except Exception as e:
            print(f"[GPU {rank}] Batch {batch_idx} error: {e}")
            torch.cuda.empty_cache()
            gc.collect()

        pbar.set_postfix(ok=success_count)

    print(f"[GPU {rank}] Done: {success_count}/{len(my_data)} success")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Shitao/OmniGen-v1")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--mask_modules_path", type=str, default=None)
    parser.add_argument("--mask_latent_channels", type=int, default=4)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--img_guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_masks", action="store_true", help="Save predicted 10-channel masks.")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="Threshold for binarizing masks.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_data = load_jsonl(args.jsonl_path)
    print(f"Total samples: {len(all_data)}")

    if args.max_samples is not None:
        all_data = all_data[:args.max_samples]

    chunk_size = math.ceil(len(all_data) / args.num_gpus)
    chunks = [all_data[i * chunk_size : (i + 1) * chunk_size] for i in range(args.num_gpus)]

    args_dict = vars(args)

    if args.num_gpus == 1:
        generation_worker(0, 1, chunks, args_dict)
    else:
        mp.spawn(
            generation_worker,
            args=(args.num_gpus, chunks, args_dict),
            nprocs=args.num_gpus,
            join=True,
        )

    print("All done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
