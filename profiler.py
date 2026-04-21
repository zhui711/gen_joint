#!/usr/bin/env python3
"""Mini profiler for joint image-mask training initialization.

Goals:
1. Load one real batch from the CXR train manifest.
2. Measure the initial loss magnitude ratio between L_img and L_mask.
3. Check whether the lightweight mask autoencoder can quickly overfit / improve
   on a single batch without separate pretraining.

This is intentionally lightweight and safe to run on a single GPU.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers.models import AutoencoderKL

from OmniGen import OmniGen, OmniGenProcessor
from OmniGen.mask_autoencoder import MaskEncoder, MaskDecoder
from OmniGen.train_helper.data import DatasetFromJson, TrainDataCollator
from OmniGen.train_helper.loss_joint_mask import training_losses_joint_mask
from OmniGen.utils import crop_arr, vae_encode, vae_encode_list


@dataclass
class ProfileResult:
    loss_total: float
    loss_img: float
    loss_mask: float
    suggested_lambda_mask: float
    ae_recon_initial: float
    ae_recon_final: float
    ae_recon_improvement_pct: float
    pretrain_strictly_required: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_hidden_size(model):
    if hasattr(model, "llm"):
        return model.llm.config.hidden_size
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model.llm.config.hidden_size
    raise RuntimeError("Unable to resolve model hidden size")


def build_loader(args, processor, hidden_size):
    image_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_arr(pil_image, args.max_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    dataset = DatasetFromJson(
        json_file=args.json_file,
        image_path=args.image_path,
        processer=processor,
        image_transform=image_transform,
        max_input_length_limit=args.max_input_length_limit,
        condition_dropout_prob=0.0,
        keep_raw_resolution=args.keep_raw_resolution,
    )

    num_mask_tokens = (args.max_image_size // 8 // 2) * (args.max_image_size // 8 // 2)
    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=hidden_size,
        keep_raw_resolution=args.keep_raw_resolution,
        num_mask_tokens=num_mask_tokens,
    )

    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def encode_output_images(vae, output_images, input_pixel_values, weight_dtype):
    with torch.no_grad():
        if isinstance(output_images, list):
            device = next(vae.parameters()).device
            output_images = [x.to(device=device, dtype=torch.float32) for x in output_images]
            output_images = vae_encode_list(vae, output_images, weight_dtype)
            if input_pixel_values is not None:
                input_pixel_values = [x.to(device=device, dtype=torch.float32) for x in input_pixel_values]
                input_pixel_values = vae_encode_list(vae, input_pixel_values, weight_dtype)
        else:
            device = next(vae.parameters()).device
            output_images = output_images.to(device=device, dtype=torch.float32)
            output_images = vae_encode(vae, output_images, weight_dtype)
            if input_pixel_values is not None:
                input_pixel_values = input_pixel_values.to(device=device, dtype=torch.float32)
                input_pixel_values = vae_encode(vae, input_pixel_values, weight_dtype)
    return output_images, input_pixel_values


def run_single_batch_profile(args) -> ProfileResult:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    weight_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    processor = OmniGenProcessor.from_pretrained(args.model_name_or_path)
    model = OmniGen.from_pretrained(args.model_name_or_path)
    model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)
    model = model.to(device=device, dtype=weight_dtype).eval()

    vae_path = os.path.join(args.model_name_or_path, "vae") if os.path.exists(args.model_name_or_path) else None
    if vae_path and os.path.exists(vae_path):
        vae = AutoencoderKL.from_pretrained(vae_path).to(device=device, dtype=torch.float32).eval()
    else:
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device=device, dtype=torch.float32).eval()

    mask_encoder = MaskEncoder(in_channels=10, latent_channels=args.mask_latent_channels).to(device=device, dtype=weight_dtype).train()
    mask_decoder = MaskDecoder(latent_channels=args.mask_latent_channels, out_channels=10).to(device=device, dtype=weight_dtype).train()

    loader = build_loader(args, processor, resolve_hidden_size(model))
    batch = next(iter(loader))

    output_images = batch["output_images"]
    input_pixel_values = batch["input_pixel_values"]
    output_images, input_pixel_values = encode_output_images(vae, output_images, input_pixel_values, weight_dtype)

    output_anatomy_masks = batch["output_anatomy_masks"].to(device=device, dtype=weight_dtype)
    mask_cont = 2.0 * output_anatomy_masks - 1.0
    x1_mask = mask_encoder(mask_cont)

    model_kwargs = dict(
        input_ids=batch["input_ids"].to(device),
        input_img_latents=input_pixel_values,
        input_image_sizes=batch["input_image_sizes"],
        attention_mask=batch["attention_mask"].to(device),
        position_ids=batch["position_ids"].to(device),
        padding_latent=batch["padding_images"],
        past_key_values=None,
        return_past_key_values=False,
    )

    with torch.no_grad():
        loss_dict = training_losses_joint_mask(
            model=model,
            x1_img=output_images,
            x1_mask=x1_mask,
            model_kwargs=model_kwargs,
            lambda_mask=1.0,
        )

    loss_img = float(loss_dict["loss_img"].detach().cpu())
    loss_mask = float(loss_dict["loss_mask"].detach().cpu())
    loss_total = float(loss_dict["loss"].detach().cpu())

    # Recommend lambda so the weighted mask loss is initially on the same order as image loss.
    suggested_lambda = loss_img / max(loss_mask, 1e-8)
    suggested_lambda = float(max(0.05, min(2.0, suggested_lambda)))

    # Quick single-batch AE improvement probe
    ae_optimizer = torch.optim.AdamW(
        list(mask_encoder.parameters()) + list(mask_decoder.parameters()),
        lr=args.ae_probe_lr,
        weight_decay=0.0,
    )

    with torch.no_grad():
        z0 = mask_encoder(mask_cont)
        recon0 = mask_decoder(z0)
        recon_loss0 = torch.mean((recon0 - mask_cont) ** 2).item()

    for _ in range(args.ae_probe_steps):
        z = mask_encoder(mask_cont)
        recon = mask_decoder(z)
        recon_loss = torch.mean((recon - mask_cont) ** 2)
        ae_optimizer.zero_grad(set_to_none=True)
        recon_loss.backward()
        ae_optimizer.step()

    with torch.no_grad():
        z1 = mask_encoder(mask_cont)
        recon1 = mask_decoder(z1)
        recon_loss1 = torch.mean((recon1 - mask_cont) ** 2).item()

    improvement_pct = 100.0 * (recon_loss0 - recon_loss1) / max(recon_loss0, 1e-8)
    strictly_required = improvement_pct < args.ae_improvement_threshold_pct

    return ProfileResult(
        loss_total=loss_total,
        loss_img=loss_img,
        loss_mask=loss_mask,
        suggested_lambda_mask=suggested_lambda,
        ae_recon_initial=recon_loss0,
        ae_recon_final=recon_loss1,
        ae_recon_improvement_pct=improvement_pct,
        pretrain_strictly_required=strictly_required,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Shitao/OmniGen-v1")
    parser.add_argument("--json_file", type=str, default="/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl")
    parser.add_argument("--image_path", type=str, default="./")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_image_size", type=int, default=256)
    parser.add_argument("--max_input_length_limit", type=int, default=18000)
    parser.add_argument("--keep_raw_resolution", action="store_true")
    parser.add_argument("--mask_latent_channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ae_probe_steps", type=int, default=25)
    parser.add_argument("--ae_probe_lr", type=float, default=2e-3)
    parser.add_argument("--ae_improvement_threshold_pct", type=float, default=10.0)
    parser.add_argument("--output_json", type=str, default="/home/wenting/zr/gen_code_plan2/profiler_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    result = run_single_batch_profile(args)
    result_dict = result.__dict__
    print(json.dumps(result_dict, indent=2))
    with open(args.output_json, "w") as f:
        json.dump(result_dict, f, indent=2)


if __name__ == "__main__":
    main()
