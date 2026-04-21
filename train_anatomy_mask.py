"""
Mask-based Anatomy-Aware Training Script for OmniGen LoRA fine-tuning.

This is a clean, independent training entrypoint for Plan 1:
  - Keep the standard OmniGen diffusion objective unchanged.
  - Decode generated latents through the frozen VAE.
  - Run a frozen segmentation model on the decoded images.
  - Compare predicted anatomy masks against GT 10-channel masks with MSE.

Unlike the older anatomy scripts, this file does not carry any feature-matching,
timestep gating, or mask-prediction branches beyond the final sigmoid mask.
"""

import json
from time import time
import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path
import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from peft import set_peft_model_state_dict
from safetensors.torch import save_file
from diffusers.models import AutoencoderKL

from OmniGen import OmniGen, OmniGenProcessor
from OmniGen.train_helper import DatasetFromJson, TrainDataCollator
from OmniGen.train_helper.loss_anatomy_mask import training_losses_with_anatomy_mask
from OmniGen.utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    crop_arr,
    vae_encode,
    vae_encode_list,
)

# Add segmentation library to path for loading the frozen seg model.
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")
import segmentation_models_pytorch as smp


def get_anatomy_model():
    """Create the ResNet34 U-Net anatomy model."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )


def load_seg_model(ckpt_path, device):
    """
    Load the frozen segmentation model used to produce anatomy masks.

    The model is intentionally kept outside accelerator.prepare() and frozen so
    only the OmniGen / LoRA parameters receive gradients.
    """
    model = get_anatomy_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def main(args):
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

    os.makedirs(args.results_dir, exist_ok=True)
    logger = create_logger(args.results_dir)
    checkpoint_dir = f"{args.results_dir}/checkpoints"
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created at {args.results_dir}")
        json.dump(args.__dict__, open(os.path.join(args.results_dir, "train_args.json"), "w"))

    if not os.path.exists(args.model_name_or_path):
        cache_folder = os.getenv("HF_HUB_CACHE")
        args.model_name_or_path = snapshot_download(
            repo_id=args.model_name_or_path,
            cache_dir=cache_folder,
            ignore_patterns=["flax_model.msgpack", "rust_model.ot", "tf_model.h5"],
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

    # Keep the VAE in fp32 for stable differentiable decoding.
    vae.to(dtype=torch.float32)
    model.to(weight_dtype)

    if accelerator.is_main_process:
        logger.info(f"Loading segmentation model from {args.seg_model_ckpt}")
    seg_model = load_seg_model(args.seg_model_ckpt, device)
    if accelerator.is_main_process:
        logger.info(
            f"Segmentation model loaded. lambda_anatomy={args.lambda_anatomy}, "
            f"subbatch={args.anatomy_subbatch_size}"
        )

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

        if args.lora_resume_path:
            if accelerator.is_main_process:
                logger.info(f"Loading LoRA adapter weights from {args.lora_resume_path}")
                logger.info("Optimizer will be created fresh (no momentum transfer).")
            from safetensors.torch import load_file as safe_load_file
            import glob as glob_mod

            adapter_path = args.lora_resume_path
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
                logger.info(f"Loaded {len(adapter_state)} LoRA parameter tensors.")

        model.to(weight_dtype)
        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        opt = torch.optim.AdamW(transformer_lora_parameters, lr=args.lr, weight_decay=args.adam_weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    ema = None
    if args.use_ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)

    crop_func = crop_arr
    if not args.keep_raw_resolution:
        crop_func = center_crop_arr
    image_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_func(pil_image, args.max_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    dataset = DatasetFromJson(
        json_file=args.json_file,
        image_path=args.image_path,
        processer=processor,
        image_transform=image_transform,
        max_input_length_limit=args.max_input_length_limit,
        condition_dropout_prob=args.condition_dropout_prob,
        keep_raw_resolution=args.keep_raw_resolution,
    )
    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=model.llm.config.hidden_size,
        keep_raw_resolution=args.keep_raw_resolution,
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

    model.train()
    if ema is not None:
        update_ema(ema, model, decay=0)
        ema.eval()

    if ema is not None:
        model, ema = accelerator.prepare(model, ema)
    else:
        model = accelerator.prepare(model)
    # seg_model is intentionally excluded from accelerator.prepare()
    opt, loader, lr_scheduler = accelerator.prepare(opt, loader, lr_scheduler)

    global_step = 0
    first_epoch = 0
    resume_step_in_epoch = 0

    if args.resume_from_checkpoint:
        if not os.path.isdir(args.resume_from_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {args.resume_from_checkpoint}"
            )
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
        if "checkpoint-" in path:
            global_step = int(path.split("-")[-1])
            actual_steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
            first_epoch = global_step // actual_steps_per_epoch
            resume_step_in_epoch = global_step % actual_steps_per_epoch
        if accelerator.is_main_process:
            logger.info(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            logger.info(f"global_step={global_step}, epoch={first_epoch}, step_in_epoch={resume_step_in_epoch}")

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

        if epoch == first_epoch and resume_step_in_epoch > 0:
            active_loader = accelerator.skip_first_batches(
                loader, resume_step_in_epoch * args.gradient_accumulation_steps
            )
            if accelerator.is_main_process:
                logger.info(f"Skipping first {resume_step_in_epoch * args.gradient_accumulation_steps} batches")
        else:
            active_loader = loader

        for data in active_loader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    output_images = data["output_images"]
                    input_pixel_values = data["input_pixel_values"]
                    if isinstance(output_images, list):
                        output_images = vae_encode_list(vae, output_images, weight_dtype)
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode_list(vae, input_pixel_values, weight_dtype)
                    else:
                        output_images = vae_encode(vae, output_images, weight_dtype)
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode(vae, input_pixel_values, weight_dtype)

                model_kwargs = dict(
                    input_ids=data["input_ids"],
                    input_img_latents=input_pixel_values,
                    input_image_sizes=data["input_image_sizes"],
                    attention_mask=data["attention_mask"],
                    position_ids=data["position_ids"],
                    padding_latent=data["padding_images"],
                    past_key_values=None,
                    return_past_key_values=False,
                )

                output_anatomy_masks = data.get("output_anatomy_masks", None)
                if output_anatomy_masks is None:
                    raise RuntimeError(
                        "output_anatomy_masks not found in batch. Ensure your JSONL file has 'output_mask'."
                    )

                loss_dict = training_losses_with_anatomy_mask(
                    model=model,
                    x1=output_images,
                    model_kwargs=model_kwargs,
                    output_anatomy_masks=output_anatomy_masks,
                    vae=vae,
                    seg_model=seg_model,
                    lambda_anatomy=args.lambda_anatomy,
                    anatomy_subbatch_size=args.anatomy_subbatch_size,
                    anatomy_alpha=args.anatomy_alpha,
                    debug_vis=args.debug_vis,
                    debug_global_step=train_steps // args.gradient_accumulation_steps,
                )
                loss = loss_dict["loss"]

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

                accelerator.log(
                    {
                        "training_loss_total": loss.item(),
                        "training_loss_diffusion": loss_dict["loss_diffusion"].item(),
                        "training_loss_anatomy": loss_dict["loss_anatomy"].item(),
                    },
                    step=train_steps,
                )

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
                            f"(step={int(train_steps / args.gradient_accumulation_steps):07d}) "
                            f"Loss: {avg_loss:.4f} (diff={avg_loss_diff:.4f}, anat={avg_loss_anat:.4f}), "
                            f"Steps/Sec: {steps_per_sec:.2f}, Epoch: {train_steps / len(loader):.2f}, LR: {cur_lr}"
                        )

                    running_loss = 0
                    running_loss_diffusion = 0
                    running_loss_anatomy = 0
                    log_steps = 0
                    start_time = time()

            if train_steps % (args.ckpt_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                current_step = int(train_steps / args.gradient_accumulation_steps)
                accelerator_state_path = f"{checkpoint_dir}/checkpoint-{current_step}"
                accelerator.save_state(accelerator_state_path)
                if accelerator.is_main_process:
                    logger.info(f"Saved accelerator state to {accelerator_state_path}")

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
                        checkpoint_path = f"{checkpoint_dir}/{int(train_steps / args.gradient_accumulation_steps):07d}/"
                        os.makedirs(checkpoint_path, exist_ok=True)
                        if hasattr(model, "module"):
                            model.module.save_pretrained(checkpoint_path)
                        else:
                            model.save_pretrained(checkpoint_path)
                    else:
                        checkpoint_path = f"{checkpoint_dir}/{int(train_steps / args.gradient_accumulation_steps):07d}/"
                        os.makedirs(checkpoint_path, exist_ok=True)
                        torch.save(state_dict, os.path.join(checkpoint_path, "model.pt"))
                        processor.text_tokenizer.save_pretrained(checkpoint_path)
                        model.llm.config.save_pretrained(checkpoint_path)
                        if ema_state_dict is not None:
                            checkpoint_path = f"{checkpoint_dir}/{int(train_steps / args.gradient_accumulation_steps):07d}_ema"
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
        "--lr_scheduler",
        type=str,
        default="constant",
        help='["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to an accelerator state checkpoint folder to fully resume training.",
    )
    parser.add_argument(
        "--lora_resume_path",
        type=str,
        default=None,
        help="Path to a LoRA adapter checkpoint to load ONLY the weights.",
    )

    parser.add_argument(
        "--seg_model_ckpt",
        type=str,
        default="/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth",
        help="Path to the frozen segmentation model checkpoint.",
    )
    parser.add_argument(
        "--lambda_anatomy",
        type=float,
        default=0.1,
        help="Weight for the anatomy mask loss.",
    )
    parser.add_argument(
        "--anatomy_subbatch_size",
        type=int,
        default=4,
        help="Max number of samples to VAE-decode per step.",
    )
    parser.add_argument(
        "--anatomy_alpha",
        type=float,
        default=4.0,
        help="Polynomial power for timestep weighting.",
    )
    parser.add_argument(
        "--debug_vis",
        action="store_true",
        help="If set, saves decoded images and masks to ./debug_vis/ for debugging.",
    )

    args = parser.parse_args()
    assert args.max_image_size % 16 == 0, "Image size must be divisible by 16."

    if args.lora_resume_path and args.resume_from_checkpoint:
        raise ValueError(
            "--lora_resume_path and --resume_from_checkpoint are mutually exclusive."
        )

    main(args)
