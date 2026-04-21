"""
Offline Pseudo-Mask Generator for Anatomy-Aware Loss.

Uses torch.multiprocessing.spawn to run inference across 4 GPUs.
Loads grayscale images, converts to 3-channel, normalizes to [-1, 1],
runs through frozen ResNet34-UNet, and saves boolean masks as .npz.

Usage:
    python generate_pseudo_masks.py \
        --image_root /home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256 \
        --mask_root  /home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch \
        --seg_ckpt   /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
        --num_gpus 4 --batch_size 64
"""

import os
import sys
import argparse
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Add segmentation library to path
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")
import segmentation_models_pytorch as smp


def get_anatomy_model():
    """Create UNet with ResNet34 encoder (no pretrained weights; loaded from ckpt)."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )


def load_seg_model(ckpt_path, device):
    """Load frozen segmentation model onto specified device."""
    model = get_anatomy_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def image_path_to_mask_path(image_path, image_root, mask_root, suffix=".npz"):
    """Convert an image path to the corresponding mask path."""
    rel = os.path.relpath(image_path, image_root)
    base, _ = os.path.splitext(rel)
    return os.path.join(mask_root, base + suffix)


def process_batch(model, image_paths, device):
    """Process a batch of images and return boolean masks."""
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)
        # Normalize to [-1, 1]
        arr = arr / 127.5 - 1.0
        # Grayscale -> 3-channel
        tensor = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1)
        tensors.append(tensor)

    batch = torch.stack(tensors, dim=0).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits = model(batch)  # (B, 10, 256, 256)

    probs = torch.sigmoid(logits)
    masks = (probs > 0.5).cpu().numpy().astype(bool)  # (B, 10, 256, 256)
    return masks


def worker(rank, args, all_image_paths):
    """Worker function for each GPU."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model = load_seg_model(args.seg_ckpt, device)

    # Stride-based partitioning: rank, rank+world_size, rank+2*world_size, ...
    my_paths = all_image_paths[rank::args.num_gpus]
    total = len(my_paths)

    # Filter paths that already have masks (resume/skip logic)
    pending_paths = []
    for p in my_paths:
        mask_path = image_path_to_mask_path(p, args.image_root, args.mask_root)
        if os.path.exists(mask_path):
            # Quick integrity check
            try:
                npz = np.load(mask_path)
                if "mask" in npz and npz["mask"].shape == (10, 256, 256):
                    continue
            except Exception:
                pass
        pending_paths.append(p)

    skipped = total - len(pending_paths)
    if rank == 0:
        print(f"[GPU {rank}] {total} images assigned, {skipped} already done, {len(pending_paths)} to process")

    # Process in batches
    for i in range(0, len(pending_paths), args.batch_size):
        batch_paths = pending_paths[i : i + args.batch_size]
        masks = process_batch(model, batch_paths, device)

        for j, path in enumerate(batch_paths):
            mask_path = image_path_to_mask_path(path, args.image_root, args.mask_root)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)

            # Atomic save: write to .tmp then rename
            tmp_path = mask_path + ".tmp"
            # np.savez_compressed(tmp_path, mask=masks[j])
            with open(tmp_path, 'wb') as f:
                np.savez_compressed(f, mask=masks[j])
            os.replace(tmp_path, mask_path)

        if rank == 0 and (i // args.batch_size) % 20 == 0:
            done = min(i + args.batch_size, len(pending_paths))
            print(f"[GPU {rank}] Progress: {done}/{len(pending_paths)}")

    if rank == 0:
        print(f"[GPU {rank}] Done!")


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo anatomy masks")
    parser.add_argument(
        "--image_root",
        type=str,
        default="/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256",
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        default="/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch",
    )
    parser.add_argument(
        "--seg_ckpt",
        type=str,
        default="/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth",
    )
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # Collect all image paths
    pattern = os.path.join(args.image_root, "*", "*.png")
    all_image_paths = sorted(glob.glob(pattern))
    print(f"Found {len(all_image_paths)} images in {args.image_root}")

    if len(all_image_paths) == 0:
        raise RuntimeError(f"No images found matching {pattern}")

    os.makedirs(args.mask_root, exist_ok=True)

    if args.num_gpus == 1:
        worker(0, args, all_image_paths)
    else:
        mp.spawn(worker, args=(args, all_image_paths), nprocs=args.num_gpus, join=True)

    print("All GPUs finished. Pseudo-mask generation complete.")


if __name__ == "__main__":
    main()
