#!/usr/bin/env python3
"""
Probe Option C: estimate raw loss magnitudes for a soft reconstruction term.

This script builds one real joint-training batch, runs a single forward pass,
and reports:
  - L_img        : raw image flow loss
  - L_flow_mask  : raw mask flow loss
  - L_recon      : raw mask reconstruction MSE in the decoder's native [-1, 1] space

It then derives:
  lambda_recon = 0.1 * L_img / L_recon

so the weighted reconstruction term is exactly one tenth of the image flow loss
for that measured batch.
"""

import argparse
import json
import os
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from OmniGen import OmniGenPipeline, OmniGenProcessor
from OmniGen.mask_autoencoder import MaskDecoder, MaskEncoder
from OmniGen.train_helper import DatasetFromJson, TrainDataCollator
from OmniGen.train_helper.loss_joint_mask import training_losses_joint_mask
from OmniGen.utils import center_crop_arr, crop_arr, vae_encode, vae_encode_list


DEFAULT_MODEL_NAME = "Shitao/OmniGen-v1"
DEFAULT_JSONL_PATH = "/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe raw loss magnitudes for Option C.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base OmniGen model path or repo id.",
    )
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default=DEFAULT_JSONL_PATH,
        help="Training JSONL containing real image/mask pairs.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./",
        help="Base image path prefix passed to DatasetFromJson.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional LoRA adapter path. If omitted, base OmniGen weights are used.",
    )
    parser.add_argument(
        "--mask_modules_path",
        type=str,
        default=None,
        help=(
            "Optional mask_modules.bin path. If omitted, fresh joint mask modules and "
            "a fresh MaskEncoder/MaskDecoder are used."
        ),
    )
    parser.add_argument(
        "--mask_latent_channels",
        type=int,
        default=4,
        help="Mask latent channel count.",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=1024,
        help="Max image size used by OmniGen preprocessing.",
    )
    parser.add_argument(
        "--keep_raw_resolution",
        action="store_true",
        help="Mirror training-time raw-resolution preprocessing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of real samples to collate into the probe batch.",
    )
    parser.add_argument(
        "--sample_offset",
        type=int,
        default=0,
        help="Offset into the JSONL dataset for choosing the probe batch.",
    )
    parser.add_argument(
        "--lambda_mask",
        type=float,
        default=0.25,
        help="Current training-time lambda_mask. Raw losses are reported separately.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="probe_option_c_report.json",
        help="Where to save the probe summary JSON.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_weight_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _unwrap_module(module):
    while hasattr(module, "module"):
        module = module.module
    return module


def _get_inner_omnigen_model(model):
    base_model = _unwrap_module(model)
    if hasattr(base_model, "base_model"):
        inner = base_model.base_model
        if hasattr(inner, "model"):
            inner = inner.model
        return inner
    return base_model


def load_mask_module_state_dict(model, mask_encoder, mask_decoder, state_dict):
    mask_encoder = _unwrap_module(mask_encoder)
    mask_decoder = _unwrap_module(mask_decoder)

    enc_state = {
        key.replace("mask_encoder.", ""): value
        for key, value in state_dict.items()
        if key.startswith("mask_encoder.")
    }
    dec_state = {
        key.replace("mask_decoder.", ""): value
        for key, value in state_dict.items()
        if key.startswith("mask_decoder.")
    }
    if enc_state:
        mask_encoder.load_state_dict(enc_state)
    if dec_state:
        mask_decoder.load_state_dict(dec_state)

    inner = _get_inner_omnigen_model(model)

    emb_state = {
        key.replace("mask_x_embedder.", ""): value
        for key, value in state_dict.items()
        if key.startswith("mask_x_embedder.")
    }
    if emb_state and inner.mask_x_embedder is not None:
        inner.mask_x_embedder.load_state_dict(emb_state)

    fl_state = {
        key.replace("mask_final_layer.", ""): value
        for key, value in state_dict.items()
        if key.startswith("mask_final_layer.")
    }
    if fl_state and inner.mask_final_layer is not None:
        inner.mask_final_layer.load_state_dict(fl_state)

    if "image_modality_embed" in state_dict and inner.image_modality_embed is not None:
        inner.image_modality_embed.data.copy_(state_dict["image_modality_embed"])
    if "mask_modality_embed" in state_dict and inner.mask_modality_embed is not None:
        inner.mask_modality_embed.data.copy_(state_dict["mask_modality_embed"])


def build_probe_batch(
    processor: OmniGenProcessor,
    model_hidden_size: int,
    args,
) -> Dict:
    crop_func = crop_arr if args.keep_raw_resolution else center_crop_arr
    image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: crop_func(pil_image, args.max_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            ),
        ]
    )

    dataset = DatasetFromJson(
        json_file=args.jsonl_path,
        image_path=args.image_path,
        processer=processor,
        image_transform=image_transform,
        max_input_length_limit=18000,
        condition_dropout_prob=0.0,
        keep_raw_resolution=args.keep_raw_resolution,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No data found in {args.jsonl_path}")
    if args.sample_offset >= len(dataset):
        raise ValueError(
            f"sample_offset={args.sample_offset} is out of range for dataset size {len(dataset)}"
        )

    features = []
    for index in range(args.sample_offset, min(len(dataset), args.sample_offset + args.batch_size)):
        sample = dataset[index]
        if len(sample) < 3:
            raise RuntimeError(
                "Probe batch is missing `output_mask`. Ensure the JSONL includes `output_mask`."
            )
        features.append(sample)

    if not features:
        raise RuntimeError("Failed to construct a non-empty probe batch.")

    first_output_image = features[0][1]
    img_h, img_w = int(first_output_image.size(-2)), int(first_output_image.size(-1))
    num_mask_tokens = (img_h // 8 // 2) * (img_w // 8 // 2)

    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=model_hidden_size,
        keep_raw_resolution=args.keep_raw_resolution,
        num_mask_tokens=num_mask_tokens,
    )
    batch = collate_fn(features)
    batch["_meta"] = {
        "dataset_size": len(dataset),
        "batch_size": len(features),
        "image_size": [img_h, img_w],
        "num_mask_tokens": num_mask_tokens,
        "sample_offset": args.sample_offset,
    }
    return batch


def encode_image_batch(batch, vae, device: torch.device, weight_dtype: torch.dtype):
    output_images = batch["output_images"]
    input_pixel_values = batch["input_pixel_values"]

    with torch.no_grad():
        if isinstance(output_images, list):
            output_images = [x.to(device=device, dtype=torch.float32) for x in output_images]
            output_images = vae_encode_list(vae, output_images, weight_dtype)
            if input_pixel_values is not None:
                input_pixel_values = [x.to(device=device, dtype=torch.float32) for x in input_pixel_values]
                input_pixel_values = vae_encode_list(vae, input_pixel_values, weight_dtype)
        else:
            output_images = output_images.to(device=device, dtype=torch.float32)
            output_images = vae_encode(vae, output_images, weight_dtype)
            if input_pixel_values is not None:
                input_pixel_values = input_pixel_values.to(device=device, dtype=torch.float32)
                input_pixel_values = vae_encode(vae, input_pixel_values, weight_dtype)

    return output_images, input_pixel_values


def move_padding_latents(padding_images, device: torch.device, dtype: torch.dtype):
    if padding_images is None:
        return None
    moved = []
    for item in padding_images:
        if item is None:
            moved.append(None)
        else:
            moved.append(item.to(device=device, dtype=dtype))
    return moved


def main():
    args = parse_args()
    if not args.keep_raw_resolution:
        # The real training run used keep_raw_resolution=True. Make the current
        # behavior explicit to avoid silent mismatches.
        print(
            "Warning: --keep_raw_resolution was not provided. "
            "The production joint training used keep_raw_resolution=True."
        )

    if not os.path.exists(args.jsonl_path):
        raise FileNotFoundError(f"Training JSONL not found: {args.jsonl_path}")

    set_seed(args.seed)
    device = choose_device()
    weight_dtype = choose_weight_dtype(device)
    print(f"Using device: {device}, weight_dtype={weight_dtype}")

    t0 = time.time()

    pipe = OmniGenPipeline.from_pretrained(args.model_name_or_path)
    pipe.model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)
    if args.lora_path:
        if not os.path.exists(args.lora_path):
            raise FileNotFoundError(f"LoRA path not found: {args.lora_path}")
        print(f"Merging LoRA from {args.lora_path}")
        pipe.merge_lora(args.lora_path)

    mask_encoder = MaskEncoder(
        in_channels=10,
        latent_channels=args.mask_latent_channels,
    )
    mask_decoder = MaskDecoder(
        latent_channels=args.mask_latent_channels,
        out_channels=10,
    )

    mask_modules_loaded = False
    if args.mask_modules_path:
        if not os.path.exists(args.mask_modules_path):
            raise FileNotFoundError(f"mask_modules.bin not found: {args.mask_modules_path}")
        mask_state = torch.load(args.mask_modules_path, map_location="cpu")
        load_mask_module_state_dict(pipe.model, mask_encoder, mask_decoder, mask_state)
        mask_modules_loaded = True
        print(f"Loaded mask modules from {args.mask_modules_path}")
    else:
        print("No mask_modules.bin provided; using fresh mask encoder/decoder and fresh joint mask heads.")

    pipe.model.to(device=device, dtype=weight_dtype)
    pipe.vae.to(device=device, dtype=torch.float32)
    pipe.model.eval()
    pipe.vae.eval()
    mask_encoder.to(device=device, dtype=weight_dtype).eval()
    mask_decoder.to(device=device, dtype=weight_dtype).eval()

    inner_model = _get_inner_omnigen_model(pipe.model)
    hidden_size = inner_model.llm.config.hidden_size
    batch = build_probe_batch(pipe.processor, hidden_size, args)
    print(
        "Probe batch:",
        json.dumps(batch["_meta"], indent=2),
    )

    output_images, input_pixel_values = encode_image_batch(batch, pipe.vae, device, weight_dtype)

    output_anatomy_masks = batch.get("output_anatomy_masks")
    if output_anatomy_masks is None:
        raise RuntimeError("output_anatomy_masks missing from probe batch.")
    mask_cont = 2.0 * output_anatomy_masks.to(device=device, dtype=weight_dtype) - 1.0

    with torch.no_grad():
        x1_mask = mask_encoder(mask_cont)

        model_kwargs = dict(
            input_ids=batch["input_ids"].to(device),
            input_img_latents=input_pixel_values,
            input_image_sizes=batch["input_image_sizes"],
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch["position_ids"].to(device),
            padding_latent=move_padding_latents(batch["padding_images"], device, weight_dtype),
            past_key_values=None,
            return_past_key_values=False,
        )

        loss_dict = training_losses_joint_mask(
            model=pipe.model,
            x1_img=output_images,
            x1_mask=x1_mask,
            model_kwargs=model_kwargs,
            lambda_mask=args.lambda_mask,
        )

        recon = mask_decoder(x1_mask)
        loss_recon = F.mse_loss(recon, mask_cont)

    l_img = float(loss_dict["loss_img"].item())
    l_flow_mask = float(loss_dict["loss_mask"].item())
    l_recon = float(loss_recon.item())
    lambda_recon = (0.1 * l_img / max(l_recon, 1e-12))
    weighted_recon_target = lambda_recon * l_recon

    report = {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name_or_path": args.model_name_or_path,
        "jsonl_path": args.jsonl_path,
        "image_path": args.image_path,
        "lora_path": args.lora_path,
        "mask_modules_path": args.mask_modules_path,
        "mask_modules_loaded": mask_modules_loaded,
        "device": str(device),
        "weight_dtype": str(weight_dtype),
        "batch_meta": batch["_meta"],
        "loss_img_raw": l_img,
        "loss_flow_mask_raw": l_flow_mask,
        "loss_recon_raw": l_recon,
        "lambda_mask_current": args.lambda_mask,
        "lambda_recon_recommended": lambda_recon,
        "weighted_recon_at_lambda": weighted_recon_target,
        "weighted_recon_to_img_ratio": (
            weighted_recon_target / l_img if l_img > 0 else None
        ),
        "formula": "lambda_recon = 0.1 * L_img / L_recon",
        "elapsed_seconds": time.time() - t0,
        "notes": [
            "L_recon is computed in the decoder's native tanh space against masks mapped from {0,1} to [-1,1].",
            "If no LoRA path is supplied, the image branch is the base OmniGen model.",
        ],
    }

    with open(args.report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Report saved to {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
