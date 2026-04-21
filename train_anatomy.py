"""
Anatomy-Aware Training Script for OmniGen LoRA Fine-tuning.

Duplicated from train.py with additions:
  - Loads frozen segmentation model (NOT passed to accelerator.prepare)
  - Uses training_losses_with_anatomy instead of training_losses
  - Logs loss_diffusion, loss_anatomy, loss_total separately
"""

import json
from time import time
import argparse
import logging
import os
import sys
from pathlib import Path
import math

import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedType
from peft import LoraConfig, set_peft_model_state_dict, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from diffusers.models import AutoencoderKL

from OmniGen import OmniGen, OmniGenProcessor
from OmniGen.train_helper import DatasetFromJson, TrainDataCollator
# Import both loss versions for runtime selection
from OmniGen.train_helper.loss_anatomy_v2 import training_losses_with_anatomy_v2
from OmniGen.train_helper.loss_anatomy_v3 import training_losses_with_anatomy_v3
from OmniGen.utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    crop_arr,
    vae_encode,
    vae_encode_list
)

# Add segmentation library to path for loading the seg model
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
    """
    Load frozen segmentation model.

    CRITICAL: Do NOT pass to accelerator.prepare().
    Do NOT attach as attribute of OmniGen model.
    """
    model = get_anatomy_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def main(args):
    # Setup accelerator:
    from accelerate import DistributedDataParallelKwargs as DDPK
    kwargs = DDPK(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.results_dir,
        kwargs_handlers=[kwargs],
    )
    device = accelerator.device
    accelerator.init_trackers("tensorboard_log", config=args.__dict__)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    logger = create_logger(args.results_dir)
    checkpoint_dir = f"{args.results_dir}/checkpoints"
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created at {args.results_dir}")
        json.dump(args.__dict__, open(os.path.join(args.results_dir, 'train_args.json'), 'w'))

    # Create model:
    if not os.path.exists(args.model_name_or_path):
        cache_folder = os.getenv('HF_HUB_CACHE')
        args.model_name_or_path = snapshot_download(
            repo_id=args.model_name_or_path,
            cache_dir=cache_folder,
            ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
        )
        logger.info(f"Downloaded model to {args.model_name_or_path}")
    model = OmniGen.from_pretrained(args.model_name_or_path)
    model.llm.config.use_cache = False
    model.llm.gradient_checkpointing_enable()
    model = model.to(device)

    if args.vae_path is None:
        vae_path = os.path.join(args.model_name_or_path, "vae")
        if os.path.exists(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)
        else:
            logger.info("No VAE found in model, downloading stabilityai/sdxl-vae from HF")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # VAE must stay in fp32 for differentiable decoding
    vae.to(dtype=torch.float32)
    model.to(weight_dtype)

    # ---- Load frozen segmentation model (NOT managed by accelerator) ----
    if accelerator.is_main_process:
        logger.info(f"Loading segmentation model from {args.seg_model_ckpt}")
    seg_model = load_seg_model(args.seg_model_ckpt, device)
    if accelerator.is_main_process:
        logger.info(f"Segmentation model loaded. loss_version={args.loss_version}, lambda_anatomy={args.lambda_anatomy}, subbatch={args.anatomy_subbatch_size}")
        if args.loss_version == "v3":
            logger.info(f"  v3 options: t_threshold={args.t_threshold}, use_gen_mask={args.use_gen_mask}")
        else:
            logger.info(f"  v2 options: use_gen_mask={args.use_gen_mask}")

    processor = OmniGenProcessor.from_pretrained(args.model_name_or_path)

    requires_grad(vae, False)
    if args.use_lora:
        if accelerator.distributed_type == DistributedType.FSDP:
            raise NotImplementedError("FSDP does not support LoRA")
        requires_grad(model, False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["qkv_proj", "o_proj"],
        )
        model.llm.enable_input_require_grads()
        model = get_peft_model(model, transformer_lora_config)

        # ---- Scenario A: Load LoRA weights only (no optimizer state) ----
        # Used for transitioning from pure-diffusion to anatomy-aware training.
        # We load the adapter weights BEFORE creating the optimizer, so the
        # optimizer is initialized fresh with zero momentum/variance buffers.
        if args.lora_resume_path:
            if accelerator.is_main_process:
                logger.info(f"[Scenario A] Loading LoRA adapter weights from {args.lora_resume_path}")
                logger.info("[Scenario A] Optimizer will be created fresh (no momentum transfer).")
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file as safe_load_file
            import glob as glob_mod
            adapter_path = args.lora_resume_path
            # Try safetensors first, fallback to bin
            safetensor_files = glob_mod.glob(os.path.join(adapter_path, "adapter_model*.safetensors"))
            bin_files = glob_mod.glob(os.path.join(adapter_path, "adapter_model*.bin"))
            if safetensor_files:
                adapter_state = {}
                for f in safetensor_files:
                    adapter_state.update(safe_load_file(f, device="cpu"))
            elif bin_files:
                adapter_state = {}
                for f in bin_files:
                    adapter_state.update(torch.load(f, map_location="cpu"))
            else:
                raise FileNotFoundError(
                    f"No adapter_model.safetensors or adapter_model.bin found in {adapter_path}"
                )
            set_peft_model_state_dict(model, adapter_state)
            if accelerator.is_main_process:
                logger.info(f"[Scenario A] Loaded {len(adapter_state)} LoRA parameter tensors.")

        model.to(weight_dtype)
        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        opt = torch.optim.AdamW(transformer_lora_parameters, lr=args.lr, weight_decay=args.adam_weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    ema = None
    if args.use_ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)

    # Setup data:
    crop_func = crop_arr
    if not args.keep_raw_resolution:
        crop_func = center_crop_arr
    image_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_func(pil_image, args.max_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = DatasetFromJson(
        json_file=args.json_file,
        image_path=args.image_path,
        processer=processor,
        image_transform=image_transform,
        max_input_length_limit=args.max_input_length_limit,
        condition_dropout_prob=args.condition_dropout_prob,
        keep_raw_resolution=args.keep_raw_resolution
    )
    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=model.llm.config.hidden_size,
        keep_raw_resolution=args.keep_raw_resolution
    )

    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size_per_device,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,}")

    num_update_steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    model.train()

    if ema is not None:
        update_ema(ema, model, decay=0)
        ema.eval()

    if ema is not None:
        model, ema = accelerator.prepare(model, ema)
    else:
        model = accelerator.prepare(model)
    # NOTE: seg_model is NOT passed to accelerator.prepare()

    opt, loader, lr_scheduler = accelerator.prepare(opt, loader, lr_scheduler)

    # ---- Scenario B: Full checkpoint resume (weights + optimizer + scheduler + RNG) ----
    global_step = 0
    first_epoch = 0
    resume_step_in_epoch = 0

    if args.resume_from_checkpoint:
        if not os.path.isdir(args.resume_from_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {args.resume_from_checkpoint}"
            )
        accelerator.load_state(args.resume_from_checkpoint)
        # Extract step number from checkpoint path (e.g., "checkpoint-1000")
        path = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
        if "checkpoint-" in path:
            global_step = int(path.split("-")[-1])
            # Use post-prepare loader length (distributed) for correct epoch/step math
            actual_steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
            first_epoch = global_step // actual_steps_per_epoch
            resume_step_in_epoch = global_step % actual_steps_per_epoch
        if accelerator.is_main_process:
            logger.info(f"[Scenario B] Resumed from checkpoint: {args.resume_from_checkpoint}")
            logger.info(f"[Scenario B] global_step={global_step}, epoch={first_epoch}, step_in_epoch={resume_step_in_epoch}")
            logger.info(f"[Scenario B] Distributed loader has {len(loader)} batches per epoch per GPU")

    # Variables for monitoring:
    train_steps = global_step * args.gradient_accumulation_steps
    log_steps = 0
    running_loss = 0
    running_loss_diffusion = 0
    running_loss_anatomy = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(first_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")

        # Efficiently skip already-trained batches (only needed in the first resumed epoch)
        if epoch == first_epoch and resume_step_in_epoch > 0:
            active_loader = accelerator.skip_first_batches(loader, resume_step_in_epoch * args.gradient_accumulation_steps)
            if accelerator.is_main_process:
                logger.info(f"Skipping first {resume_step_in_epoch * args.gradient_accumulation_steps} batches")
        else:
            active_loader = loader

        for data in active_loader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    output_images = data['output_images']
                    # Save GT pixel-space images BEFORE VAE encoding (needed by v2 loss)
                    output_images_pixel = output_images if not isinstance(output_images, list) \
                        else [img.clone() for img in output_images]
                    input_pixel_values = data['input_pixel_values']
                    if isinstance(output_images, list):
                        output_images = vae_encode_list(vae, output_images, weight_dtype)
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode_list(vae, input_pixel_values, weight_dtype)
                    else:
                        output_images = vae_encode(vae, output_images, weight_dtype)
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode(vae, input_pixel_values, weight_dtype)

                model_kwargs = dict(
                    input_ids=data['input_ids'],
                    input_img_latents=input_pixel_values,
                    input_image_sizes=data['input_image_sizes'],
                    attention_mask=data['attention_mask'],
                    position_ids=data['position_ids'],
                    padding_latent=data['padding_images'],
                    past_key_values=None,
                    return_past_key_values=False,
                )

                # Get anatomy masks from batch
                output_anatomy_masks = data.get('output_anatomy_masks', None)
                if output_anatomy_masks is None:
                    raise RuntimeError(
                        "output_anatomy_masks not found in batch. "
                        "Ensure your JSONL file has 'output_mask' field. "
                        "Use gen_mask_jsonl.py to generate it."
                    )

                # Select loss function based on version
                if args.loss_version == "v3":
                    loss_dict = training_losses_with_anatomy_v3(
                        model=model,
                        x1=output_images,
                        model_kwargs=model_kwargs,
                        output_images_pixel=output_images_pixel,
                        output_anatomy_masks=output_anatomy_masks,
                        vae=vae,
                        seg_model=seg_model,
                        lambda_anatomy=args.lambda_anatomy,
                        anatomy_subbatch_size=args.anatomy_subbatch_size,
                        feature_layer_idx=getattr(args, 'feature_layer_idx', 2),
                        use_gen_mask=getattr(args, 'use_gen_mask', False),
                        t_threshold=getattr(args, 't_threshold', 0.5),
                    )
                else:  # v2
                    loss_dict = training_losses_with_anatomy_v2(
                        model=model,
                        x1=output_images,
                        model_kwargs=model_kwargs,
                        output_images_pixel=output_images_pixel,
                        output_anatomy_masks=output_anatomy_masks,
                        vae=vae,
                        seg_model=seg_model,
                        lambda_anatomy=args.lambda_anatomy,
                        anatomy_subbatch_size=args.anatomy_subbatch_size,
                        feature_layer_idx=getattr(args, 'feature_layer_idx', 2),
                        use_gen_mask=getattr(args, 'use_gen_mask', True),
                    )
                loss = loss_dict["loss_total"]

                running_loss += loss.item()
                running_loss_diffusion += loss_dict["loss_diffusion"].item()
                running_loss_anatomy += loss_dict["loss_anatomy"].item()

                accelerator.backward(loss)
                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()

                log_steps += 1
                train_steps += 1

                accelerator.log({
                    "training_loss_total": loss.item(),
                    "training_loss_diffusion": loss_dict["loss_diffusion"].item(),
                    "training_loss_anatomy": loss_dict["loss_anatomy"].item(),
                }, step=train_steps)

                if train_steps % args.gradient_accumulation_steps == 0:
                    if accelerator.sync_gradients and ema is not None:
                        update_ema(ema, model)

                if train_steps % (args.log_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / args.gradient_accumulation_steps / (end_time - start_time)
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    avg_loss_diff = torch.tensor(running_loss_diffusion / log_steps, device=device)
                    avg_loss_anat = torch.tensor(running_loss_anatomy / log_steps, device=device)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_loss_diff, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_loss_anat, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / accelerator.num_processes
                    avg_loss_diff = avg_loss_diff.item() / accelerator.num_processes
                    avg_loss_anat = avg_loss_anat.item() / accelerator.num_processes

                    if accelerator.is_main_process:
                        cur_lr = opt.param_groups[0]["lr"]
                        logger.info(
                            f"(step={int(train_steps/args.gradient_accumulation_steps):07d}) "
                            f"Loss: {avg_loss:.4f} (diff={avg_loss_diff:.4f}, anat={avg_loss_anat:.4f}), "
                            f"Steps/Sec: {steps_per_sec:.2f}, Epoch: {train_steps/len(loader):.2f}, LR: {cur_lr}"
                        )

                    running_loss = 0
                    running_loss_diffusion = 0
                    running_loss_anatomy = 0
                    log_steps = 0
                    start_time = time()

            if train_steps % (args.ckpt_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                # Save full accelerator state for resume (optimizer, scheduler, RNG, model)
                current_step = int(train_steps / args.gradient_accumulation_steps)
                accelerator_state_path = f"{checkpoint_dir}/checkpoint-{current_step}"
                accelerator.save_state(accelerator_state_path)
                if accelerator.is_main_process:
                    logger.info(f"Saved accelerator state to {accelerator_state_path}")

                # Also save standalone LoRA / model weights for inference
                if accelerator.distributed_type == DistributedType.FSDP:
                    state_dict = accelerator.get_state_dict(model)
                    ema_state_dict = accelerator.get_state_dict(ema) if ema is not None else None
                else:
                    if not args.use_lora:
                        if hasattr(model, "module"):
                            state_dict = model.module.state_dict()
                        else:
                            state_dict = model.state_dict()
                        ema_state_dict = accelerator.get_state_dict(ema) if ema is not None else None

                if accelerator.is_main_process:
                    if args.use_lora:
                        checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}/"
                        os.makedirs(checkpoint_path, exist_ok=True)

                        if hasattr(model, "module"):
                            model.module.save_pretrained(checkpoint_path)
                        else:
                            model.save_pretrained(checkpoint_path)
                    else:
                        checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}/"
                        os.makedirs(checkpoint_path, exist_ok=True)
                        torch.save(state_dict, os.path.join(checkpoint_path, "model.pt"))
                        processor.text_tokenizer.save_pretrained(checkpoint_path)
                        model.llm.config.save_pretrained(checkpoint_path)
                        if ema_state_dict is not None:
                            checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}_ema"
                            os.makedirs(checkpoint_path, exist_ok=True)
                            torch.save(ema_state_dict, os.path.join(checkpoint_path, "model.pt"))
                            processor.text_tokenizer.save_pretrained(checkpoint_path)
                            model.llm.config.save_pretrained(checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    accelerator.end_training()
    model.eval()

    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name_or_path", type=str, default="OmniGen")
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size_per_device", type=int, default=1)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=20000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_input_length_limit", type=int, default=1024)
    parser.add_argument("--condition_dropout_prob", type=float, default=0.1)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--keep_raw_resolution", action="store_true")
    parser.add_argument("--max_image_size", type=int, default=1344)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant",
        help='["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # ---- Checkpoint Resuming arguments ----
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Path to an accelerator state checkpoint folder to fully resume training "
             "(weights + optimizer + scheduler + RNG). "
             "E.g., results/cxr_finetune_lora_anatomy/checkpoints/checkpoint-2000",
    )
    parser.add_argument(
        "--lora_resume_path", type=str, default=None,
        help="Path to a LoRA adapter checkpoint to load ONLY the weights (no optimizer state). "
             "Use this for transitioning from pure-diffusion to anatomy-aware training. "
             "E.g., results/cxr_finetune_lora/checkpoints/0008000/",
    )

    # ---- Anatomy-Aware Loss arguments ----
    parser.add_argument(
        "--seg_model_ckpt", type=str,
        default="/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth",
        help="Path to the frozen segmentation model checkpoint.",
    )
    parser.add_argument(
        "--lambda_anatomy", type=float, default=0.1,
        help="Weight for the anatomy segmentation loss.",
    )
    parser.add_argument(
        "--anatomy_subbatch_size", type=int, default=4,
        help="Max number of samples to VAE-decode per step (VRAM safety).",
    )
    parser.add_argument(
        "--feature_layer_idx", type=int, default=2,
        help="Encoder feature layer index for feature matching (2=1/4 res, 3=1/8, 4=1/16).",
    )
    parser.add_argument(
        "--use_gen_mask", type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,  # CHANGED: Default to False for stability
        help="Use predicted mask from gen image (True) or GT mask for both (False, recommended).",
    )
    parser.add_argument(
        "--t_threshold", type=float, default=0.0,  #先默认全时间步应用
        help="Timestep threshold for anatomy loss. Only compute when t > threshold (default: 0.5). [v3 only]",
    )
    parser.add_argument(
        "--loss_version", type=str, default="v3", choices=["v2", "v3"],
        help="Loss function version: v2 (original) or v3 (with timestep gating, feature normalization).",
    )

    args = parser.parse_args()
    assert args.max_image_size % 16 == 0, "Image size must be divisible by 16."

    # Mutual exclusivity check
    if args.lora_resume_path and args.resume_from_checkpoint:
        raise ValueError(
            "--lora_resume_path and --resume_from_checkpoint are mutually exclusive.\n"
            "  Use --lora_resume_path for weights-only transition (diffusion -> anatomy).\n"
            "  Use --resume_from_checkpoint to fully resume an interrupted anatomy run."
        )

    main(args)
