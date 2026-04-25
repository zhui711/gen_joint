#!/usr/bin/env python3
"""Post-hoc MaskDecoder training for Option A.

This script loads MaskEncoder and MaskDecoder from an existing
`mask_modules.bin`, freezes the encoder, and trains only the decoder on real
GT masks using reconstruction MSE:

  L = MSE(MaskDecoder(MaskEncoder(mask_cont)), mask_cont)

The updated decoder weights are written back into a compatible
`mask_modules.bin` file without touching the main joint generation model.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from OmniGen.mask_autoencoder import MaskDecoder, MaskEncoder


DEFAULT_JSONL_PATH = "/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl"


class MaskOnlyDataset(Dataset):
    """Dataset that yields only GT masks mapped to [-1, 1]."""

    def __init__(self, json_file: str, mask_key: str = "mask", max_samples: int = None):
        self.mask_key = mask_key
        self.mask_paths: List[str] = []

        with open(json_file, "r") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                mask_path = item.get("output_mask")
                if mask_path is None or not os.path.exists(mask_path):
                    continue

                self.mask_paths.append(mask_path)
                if max_samples is not None and len(self.mask_paths) >= max_samples:
                    break

        if len(self.mask_paths) == 0:
            raise RuntimeError(
                f"No usable output_mask entries found in {json_file}."
            )

    def __len__(self) -> int:
        return len(self.mask_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        mask_path = self.mask_paths[index]
        data = np.load(mask_path)
        if self.mask_key not in data:
            raise KeyError(f"{self.mask_key!r} not found in {mask_path}")
        mask = data[self.mask_key].astype(np.float32)
        mask = torch.from_numpy(mask)
        return 2.0 * mask - 1.0


def infer_mask_latent_channels(state_dict: Dict[str, torch.Tensor], fallback: int) -> int:
    weight = state_dict.get("mask_encoder.net.12.weight")
    if weight is not None:
        return int(weight.shape[0])
    return fallback


def load_encoder_decoder(
    mask_modules_path: str,
    mask_latent_channels: int,
):
    state_dict = torch.load(mask_modules_path, map_location="cpu")
    latent_channels = infer_mask_latent_channels(state_dict, mask_latent_channels)

    mask_encoder = MaskEncoder(in_channels=10, latent_channels=latent_channels)
    mask_decoder = MaskDecoder(latent_channels=latent_channels, out_channels=10)

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
    if not enc_state:
        raise RuntimeError(
            f"No MaskEncoder weights found in {mask_modules_path}."
        )
    if not dec_state:
        raise RuntimeError(
            f"No MaskDecoder weights found in {mask_modules_path}."
        )

    mask_encoder.load_state_dict(enc_state, strict=True)
    mask_decoder.load_state_dict(dec_state, strict=True)
    return state_dict, mask_encoder, mask_decoder, latent_channels


def update_mask_modules_state_dict(
    state_dict: Dict[str, torch.Tensor],
    mask_encoder: MaskEncoder,
    mask_decoder: MaskDecoder,
) -> Dict[str, torch.Tensor]:
    updated = dict(state_dict)
    for key, value in mask_encoder.state_dict().items():
        updated[f"mask_encoder.{key}"] = value.detach().cpu()
    for key, value in mask_decoder.state_dict().items():
        updated[f"mask_decoder.{key}"] = value.detach().cpu()
    return updated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train only the MaskDecoder post-hoc from an existing mask_modules.bin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mask_modules_path",
        type=str,
        required=True,
        help="Path to input mask_modules.bin.",
    )
    parser.add_argument(
        "--output_mask_modules_path",
        type=str,
        default=None,
        help=(
            "Where to save the updated mask_modules.bin. "
            "Defaults to overwriting --mask_modules_path."
        ),
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=DEFAULT_JSONL_PATH,
        help="Training JSONL containing output_mask entries.",
    )
    parser.add_argument(
        "--mask_key",
        type=str,
        default="mask",
        help="Key inside GT mask .npz files.",
    )
    parser.add_argument(
        "--mask_latent_channels",
        type=int,
        default=4,
        help="Fallback latent channel count if it cannot be inferred from the checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of decoder-only training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per process.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers per process.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, train on only the first N masks for quick experiments.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Accelerate mixed precision mode.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Log every N optimization steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed)

    if not os.path.exists(args.mask_modules_path):
        raise FileNotFoundError(
            f"mask_modules.bin not found: {args.mask_modules_path}"
        )
    if not os.path.exists(args.json_file):
        raise FileNotFoundError(f"JSONL not found: {args.json_file}")

    output_path = args.output_mask_modules_path or args.mask_modules_path
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    original_state, mask_encoder, mask_decoder, latent_channels = load_encoder_decoder(
        args.mask_modules_path,
        args.mask_latent_channels,
    )

    for param in mask_encoder.parameters():
        param.requires_grad = False
    mask_encoder.eval()
    mask_decoder.train()

    dataset = MaskOnlyDataset(
        json_file=args.json_file,
        mask_key=args.mask_key,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        mask_decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    mask_decoder, optimizer, loader = accelerator.prepare(mask_decoder, optimizer, loader)
    mask_encoder.to(accelerator.device)

    if accelerator.is_main_process:
        print("=== Mask Decoder Only Training ===")
        print(f"Input mask_modules.bin : {args.mask_modules_path}")
        print(f"Output mask_modules.bin: {output_path}")
        print(f"Dataset size           : {len(dataset)}")
        print(f"Latent channels        : {latent_channels}")
        print(f"Epochs                 : {args.epochs}")
        print(f"Batch size / process   : {args.batch_size}")
        print(f"Mixed precision        : {args.mixed_precision}")

    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        progress = tqdm(
            loader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for batch_masks in progress:
            batch_masks = batch_masks.to(
                device=accelerator.device,
                dtype=torch.float32,
                non_blocking=True,
            )

            with torch.no_grad():
                with accelerator.autocast():
                    latents = mask_encoder(batch_masks)

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                recon_masks = mask_decoder(latents)
            loss = F.mse_loss(recon_masks.float(), batch_masks.float())
            accelerator.backward(loss)

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(mask_decoder.parameters(), args.max_grad_norm)

            optimizer.step()

            reduced_loss = accelerator.gather_for_metrics(loss.detach().unsqueeze(0)).mean()
            epoch_loss_sum += reduced_loss.item()
            epoch_loss_count += 1
            global_step += 1

            if accelerator.is_local_main_process:
                progress.set_postfix(loss=f"{reduced_loss.item():.4f}")
            if accelerator.is_main_process and global_step % args.log_every == 0:
                print(
                    f"step={global_step:07d} epoch={epoch + 1} "
                    f"recon_mse={reduced_loss.item():.6f}"
                )

        avg_epoch_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        if accelerator.is_main_process:
            print(
                f"Epoch {epoch + 1}/{args.epochs} complete: "
                f"avg_recon_mse={avg_epoch_loss:.6f}"
            )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_decoder = accelerator.unwrap_model(mask_decoder)
        updated_state = update_mask_modules_state_dict(
            original_state,
            mask_encoder.cpu(),
            unwrapped_decoder.cpu(),
        )
        torch.save(updated_state, output_path)
        print(f"Saved updated mask_modules.bin to {output_path}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
