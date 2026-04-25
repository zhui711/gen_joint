#!/usr/bin/env python3
"""
Probe Option A: can the trained MaskEncoder support a freshly initialized decoder?

This script freezes the MaskEncoder loaded from `mask_modules.bin`, initializes a
fresh MaskDecoder, and tries to overfit that decoder on a small subset of real GT
anatomy masks. If the decoder can drive reconstruction MSE very low quickly, the
latent still contains enough information and post-hoc decoder training is viable.
"""

import argparse
import itertools
import json
import os
import random
import time
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from OmniGen.mask_autoencoder import MaskDecoder, MaskEncoder


DEFAULT_JSONL_PATH = "/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl"
DEFAULT_MASK_MODULES_PATH = "./mask_modules.bin"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe whether Option A is viable by overfitting a fresh decoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mask_modules_path",
        type=str,
        default=DEFAULT_MASK_MODULES_PATH,
        help="Path to mask_modules.bin containing a trained MaskEncoder.",
    )
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default=DEFAULT_JSONL_PATH,
        help="Training JSONL containing `output_mask` entries.",
    )
    parser.add_argument(
        "--mask_key",
        type=str,
        default="mask",
        help="Key inside each GT .npz file.",
    )
    parser.add_argument(
        "--mask_latent_channels",
        type=int,
        default=4,
        help="Mask latent channel count.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=64,
        help="Number of real masks to use for the overfitting probe.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for decoder training.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=300,
        help="Number of decoder optimization steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the fresh decoder.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for the fresh decoder optimizer.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=50,
        help="Full-subset evaluation interval during training.",
    )
    parser.add_argument(
        "--loss_threshold",
        type=float,
        default=0.05,
        help="If final full-subset MSE drops below this, Option A is considered viable.",
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
        default="probe_option_a_report.json",
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


def load_mask_paths(jsonl_path: str, max_items: int) -> List[str]:
    mask_paths: List[str] = []
    with open(jsonl_path, "r") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            mask_path = item.get("output_mask")
            if mask_path and os.path.exists(mask_path):
                mask_paths.append(mask_path)
            if len(mask_paths) >= max_items:
                break
    return mask_paths


def load_mask_tensor(mask_path: str, mask_key: str) -> torch.Tensor:
    data = np.load(mask_path)
    if mask_key not in data:
        raise KeyError(f"{mask_key!r} not found in {mask_path}")
    mask = data[mask_key].astype(np.float32)
    return torch.from_numpy(mask)


def infer_mask_latent_channels(state_dict: dict, fallback: int) -> int:
    for key, value in state_dict.items():
        if key == "mask_encoder.net.12.weight":
            return int(value.shape[0])
    return fallback


def load_encoder_from_checkpoint(mask_modules_path: str, mask_latent_channels: int, device: torch.device):
    state_dict = torch.load(mask_modules_path, map_location="cpu")
    inferred_channels = infer_mask_latent_channels(state_dict, mask_latent_channels)
    encoder = MaskEncoder(in_channels=10, latent_channels=inferred_channels).to(device)
    encoder_state = {
        key.replace("mask_encoder.", ""): value
        for key, value in state_dict.items()
        if key.startswith("mask_encoder.")
    }
    if not encoder_state:
        raise ValueError(
            f"No MaskEncoder weights found in {mask_modules_path}."
        )
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, inferred_channels


def encode_subset(
    encoder: MaskEncoder,
    masks: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    latents: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, masks.size(0), batch_size):
            chunk = masks[start : start + batch_size].to(device=device, dtype=torch.float32)
            latents.append(encoder(chunk).cpu())
    return torch.cat(latents, dim=0)


def evaluate_decoder(
    decoder: MaskDecoder,
    latents: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    decoder.eval()
    losses: List[float] = []
    with torch.no_grad():
        for start in range(0, latents.size(0), batch_size):
            z = latents[start : start + batch_size].to(device=device, dtype=torch.float32)
            target = targets[start : start + batch_size].to(device=device, dtype=torch.float32)
            recon = decoder(z)
            losses.append(float(F.mse_loss(recon, target).item()))
    decoder.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    set_seed(args.seed)

    report = {
        "status": "not_run",
        "mask_modules_path": args.mask_modules_path,
        "jsonl_path": args.jsonl_path,
        "num_examples_requested": args.num_examples,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "loss_threshold": args.loss_threshold,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if not os.path.exists(args.mask_modules_path):
        report["status"] = "missing_checkpoint"
        report["message"] = (
            f"mask_modules.bin not found at {args.mask_modules_path}. "
            "Transfer the checkpoint and rerun this probe."
        )
        print(report["message"])
        with open(args.report_path, "w") as handle:
            json.dump(report, handle, indent=2)
        return 2

    if not os.path.exists(args.jsonl_path):
        report["status"] = "missing_jsonl"
        report["message"] = f"Training JSONL not found at {args.jsonl_path}."
        print(report["message"])
        with open(args.report_path, "w") as handle:
            json.dump(report, handle, indent=2)
        return 2

    device = choose_device()
    print(f"Using device: {device}")

    mask_paths = load_mask_paths(args.jsonl_path, args.num_examples)
    if not mask_paths:
        report["status"] = "no_masks_found"
        report["message"] = f"No usable `output_mask` entries were found in {args.jsonl_path}."
        print(report["message"])
        with open(args.report_path, "w") as handle:
            json.dump(report, handle, indent=2)
        return 2

    print(f"Loaded {len(mask_paths)} real mask paths for the probe subset.")
    masks = torch.stack(
        [load_mask_tensor(path, args.mask_key) for path in mask_paths],
        dim=0,
    )
    targets = 2.0 * masks - 1.0

    encoder, latent_channels = load_encoder_from_checkpoint(
        args.mask_modules_path,
        args.mask_latent_channels,
        device,
    )
    print(f"Loaded frozen MaskEncoder with latent_channels={latent_channels}.")

    latents = encode_subset(
        encoder=encoder,
        masks=targets,
        device=device,
        batch_size=args.batch_size,
    )
    print(f"Encoded subset latents: {tuple(latents.shape)}")

    decoder = MaskDecoder(
        latent_channels=latent_channels,
        out_channels=10,
    ).to(device)
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_dataset = TensorDataset(latents, targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    loader_iter: Iterable = itertools.cycle(train_loader)

    initial_eval_loss = evaluate_decoder(decoder, latents, targets, device, args.batch_size)
    print(f"Initial full-subset MSE: {initial_eval_loss:.6f}")

    best_batch_loss = float("inf")
    eval_history = [{"step": 0, "full_subset_mse": initial_eval_loss}]
    train_loss_history = []

    t0 = time.time()
    decoder.train()
    for step in range(1, args.num_steps + 1):
        batch_latents, batch_targets = next(loader_iter)
        batch_latents = batch_latents.to(device=device, dtype=torch.float32, non_blocking=True)
        batch_targets = batch_targets.to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        recon = decoder(batch_latents)
        loss = F.mse_loss(recon, batch_targets)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        best_batch_loss = min(best_batch_loss, batch_loss)
        train_loss_history.append({"step": step, "batch_mse": batch_loss})

        if step == 1 or step % args.eval_every == 0 or step == args.num_steps:
            full_subset_loss = evaluate_decoder(
                decoder,
                latents,
                targets,
                device,
                args.batch_size,
            )
            eval_history.append({"step": step, "full_subset_mse": full_subset_loss})
            print(
                f"step={step:03d} batch_mse={batch_loss:.6f} "
                f"full_subset_mse={full_subset_loss:.6f}"
            )

    final_eval_loss = eval_history[-1]["full_subset_mse"]
    elapsed = time.time() - t0
    viable = final_eval_loss < args.loss_threshold

    report.update(
        {
            "status": "ok",
            "device": str(device),
            "num_examples_used": len(mask_paths),
            "latent_channels": latent_channels,
            "initial_full_subset_mse": initial_eval_loss,
            "final_full_subset_mse": final_eval_loss,
            "best_batch_mse": best_batch_loss,
            "elapsed_seconds": elapsed,
            "option_a_viable": viable,
            "decision_rule": (
                f"final full-subset MSE < {args.loss_threshold} => Option A viable"
            ),
            "eval_history": eval_history,
            "train_loss_history_tail": train_loss_history[-20:],
        }
    )

    with open(args.report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    if viable:
        print(
            f"Result: VIABLE. Final full-subset MSE={final_eval_loss:.6f} "
            f"is below threshold {args.loss_threshold:.6f}."
        )
    else:
        print(
            f"Result: NOT VIABLE (or latent severely compressed). "
            f"Final full-subset MSE={final_eval_loss:.6f} remains above "
            f"threshold {args.loss_threshold:.6f}."
        )
    print(f"Report saved to {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
