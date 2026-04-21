"""Joint Image-Mask Co-Generation Training Script for OmniGen.

This is the main training entrypoint for Plan 2 (Joint Distribution):
  - Encode GT images to image latents via frozen VAE.
  - Encode GT masks to mask latents via trainable MaskEncoder.
  - Co-denoise both modalities through the shared Transformer.
  - Compute flow-matching loss directly in latent space for both branches.
  - No frozen segmentation model needed.

Checkpointing:
  - PEFT saves LoRA weights automatically.
  - All new mask modules (MaskEncoder, MaskDecoder, mask_x_embedder,
    mask_final_layer, modality_embeddings) are saved to mask_modules.bin.
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
from safetensors.torch import save_file, load_file
from diffusers.models import AutoencoderKL

from OmniGen import OmniGen, OmniGenProcessor
from OmniGen.mask_autoencoder import MaskEncoder, MaskDecoder
from OmniGen.train_helper import DatasetFromJson, TrainDataCollator
from OmniGen.train_helper.loss_joint_mask import training_losses_joint_mask
from OmniGen.utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    crop_arr,
    vae_encode,
    vae_encode_list,
)


def _unwrap_module(module):
    """Recursively unwrap DDP/FSDP-like wrappers that expose `.module`."""
    while hasattr(module, "module"):
        module = module.module
    return module


def _get_inner_omnigen_model(model):
    """Resolve the underlying OmniGen instance through DDP and PEFT wrappers."""
    base_model = _unwrap_module(model)
    if hasattr(base_model, "base_model"):
        inner = base_model.base_model
        if hasattr(inner, "model"):
            inner = inner.model
        return inner
    return base_model


def _maybe_load_mask_modules_from_checkpoint(model, mask_encoder, mask_decoder, mask_modules_path, logger=None):
    if mask_modules_path and os.path.exists(mask_modules_path):
        if logger is not None:
            logger.info(f"Loading mask modules from {mask_modules_path}")
        mask_state = torch.load(mask_modules_path, map_location="cpu")
        load_mask_module_state_dict(model, mask_encoder, mask_decoder, mask_state)
        return True
    return False


def _sanitize_tracker_config(config_dict):
    """Convert argparse values into TensorBoard-hparams-safe scalars."""
    safe = {}
    for key, value in config_dict.items():
        if isinstance(value, (int, float, str, bool, torch.Tensor)) or value is None:
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = ",".join(str(v) for v in value)
        else:
            safe[key] = str(value)
    return safe


def get_mask_module_state_dict(model, mask_encoder, mask_decoder):
    """Collect state dicts of all newly introduced mask modules.

    Returns a flat dict suitable for torch.save / safetensors.save_file.
    """
    state = {}

    # Mask encoder/decoder (standalone modules)
    mask_encoder = _unwrap_module(mask_encoder)
    mask_decoder = _unwrap_module(mask_decoder)

    for k, v in mask_encoder.state_dict().items():
        state[f"mask_encoder.{k}"] = v
    for k, v in mask_decoder.state_dict().items():
        state[f"mask_decoder.{k}"] = v

    # Modules inside the OmniGen model (may be wrapped by PEFT/DDP)
    inner = _get_inner_omnigen_model(model)

    if inner.mask_x_embedder is not None:
        for k, v in inner.mask_x_embedder.state_dict().items():
            state[f"mask_x_embedder.{k}"] = v
    if inner.mask_final_layer is not None:
        for k, v in inner.mask_final_layer.state_dict().items():
            state[f"mask_final_layer.{k}"] = v
    if inner.image_modality_embed is not None:
        state["image_modality_embed"] = inner.image_modality_embed.data
    if inner.mask_modality_embed is not None:
        state["mask_modality_embed"] = inner.mask_modality_embed.data

    return state


def load_mask_module_state_dict(model, mask_encoder, mask_decoder, state_dict):
    """Load mask module weights from a flat state dict."""
    mask_encoder = _unwrap_module(mask_encoder)
    mask_decoder = _unwrap_module(mask_decoder)
    enc_state = {k.replace("mask_encoder.", ""): v for k, v in state_dict.items() if k.startswith("mask_encoder.")}
    dec_state = {k.replace("mask_decoder.", ""): v for k, v in state_dict.items() if k.startswith("mask_decoder.")}
    mask_encoder.load_state_dict(enc_state)
    mask_decoder.load_state_dict(dec_state)

    inner = _get_inner_omnigen_model(model)

    emb_state = {k.replace("mask_x_embedder.", ""): v for k, v in state_dict.items() if k.startswith("mask_x_embedder.")}
    if emb_state and inner.mask_x_embedder is not None:
        inner.mask_x_embedder.load_state_dict(emb_state)

    fl_state = {k.replace("mask_final_layer.", ""): v for k, v in state_dict.items() if k.startswith("mask_final_layer.")}
    if fl_state and inner.mask_final_layer is not None:
        inner.mask_final_layer.load_state_dict(fl_state)

    if "image_modality_embed" in state_dict and inner.image_modality_embed is not None:
        inner.image_modality_embed.data.copy_(state_dict["image_modality_embed"])
    if "mask_modality_embed" in state_dict and inner.mask_modality_embed is not None:
        inner.mask_modality_embed.data.copy_(state_dict["mask_modality_embed"])


def main(args):
    from accelerate import DistributedDataParallelKwargs as DDPK

    # The joint mask training graph is static: every iteration uses the image
    # branch, mask branch, and LoRA-adapted Transformer in a single forward.
    # Keeping find_unused_parameters=True together with LoRA + gradient
    # checkpointing can make DDP register duplicate reducer hooks on adapter
    # weights, producing "marked ready twice" errors.
    kwargs = DDPK(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.results_dir,
        kwargs_handlers=[kwargs],
    )
    device = accelerator.device
    accelerator.init_trackers("tensorboard_log", config=_sanitize_tracker_config(args.__dict__))

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

    # --- Load base OmniGen model ---
    model = OmniGen.from_pretrained(args.model_name_or_path)
    model.llm.config.use_cache = False
    model.llm.gradient_checkpointing_enable()

    # --- Initialize joint mask modules ---
    model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)
    model = model.to(device)

    if accelerator.is_main_process:
        logger.info("Initialized joint mask modules (mask_x_embedder, mask_final_layer, modality_embeddings)")

    # --- Load VAE ---
    if args.vae_path is None:
        vae_path = os.path.join(args.model_name_or_path, "vae")
        if os.path.exists(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)
        else:
            logger.info("No VAE found in model, downloading stabilityai/sdxl-vae from HF")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    # --- Load/Create Mask Autoencoder ---
    mask_encoder = MaskEncoder(in_channels=10, latent_channels=args.mask_latent_channels).to(device)
    mask_decoder = MaskDecoder(latent_channels=args.mask_latent_channels, out_channels=10).to(device)

    if args.mask_ae_ckpt:
        if accelerator.is_main_process:
            logger.info(f"Loading mask autoencoder from {args.mask_ae_ckpt}")
        ckpt = torch.load(args.mask_ae_ckpt, map_location="cpu")
        if 'encoder' in ckpt:
            mask_encoder.load_state_dict(ckpt['encoder'])
            mask_decoder.load_state_dict(ckpt['decoder'])
        else:
            # Try loading as MaskAutoencoder state dict
            enc_state = {k.replace('encoder.', ''): v for k, v in ckpt.items() if k.startswith('encoder.')}
            dec_state = {k.replace('decoder.', ''): v for k, v in ckpt.items() if k.startswith('decoder.')}
            if enc_state:
                mask_encoder.load_state_dict(enc_state)
                mask_decoder.load_state_dict(dec_state)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(dtype=torch.float32)
    model.to(weight_dtype)
    mask_encoder.to(weight_dtype)
    mask_decoder.to(weight_dtype)

    processor = OmniGenProcessor.from_pretrained(args.model_name_or_path)

    requires_grad(vae, False)

    # --- LoRA setup ---
    if args.use_lora:
        if accelerator.distributed_type == DistributedType.FSDP:
            raise NotImplementedError("FSDP does not support LoRA")
        requires_grad(model, False)

        # Re-enable gradients for new mask modules inside the model
        if model.mask_x_embedder is not None:
            requires_grad(model.mask_x_embedder, True)
        if model.mask_final_layer is not None:
            requires_grad(model.mask_final_layer, True)
        if model.image_modality_embed is not None:
            model.image_modality_embed.requires_grad_(True)
        if model.mask_modality_embed is not None:
            model.mask_modality_embed.requires_grad_(True)

        # Parse target modules
        lora_target_modules = args.lora_target_modules
        if isinstance(lora_target_modules, str):
            lora_target_modules = [m.strip() for m in lora_target_modules.split(",")]

        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
        )
        model.llm.enable_input_require_grads()
        model = get_peft_model(model, transformer_lora_config)

        if args.lora_resume_path:
            if accelerator.is_main_process:
                logger.info(f"Loading LoRA adapter weights from {args.lora_resume_path}")
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

        # Load mask modules explicitly if provided, or auto-discover alongside LoRA resume path.
        loaded_mask_modules = False
        if args.mask_modules_resume_path:
            loaded_mask_modules = _maybe_load_mask_modules_from_checkpoint(
                model, mask_encoder, mask_decoder, args.mask_modules_resume_path, logger=logger if accelerator.is_main_process else None
            )
        elif args.lora_resume_path:
            auto_mask_modules_path = os.path.join(args.lora_resume_path, "mask_modules.bin")
            loaded_mask_modules = _maybe_load_mask_modules_from_checkpoint(
                model, mask_encoder, mask_decoder, auto_mask_modules_path, logger=logger if accelerator.is_main_process else None
            )
            if accelerator.is_main_process and not loaded_mask_modules:
                logger.info(f"No companion mask_modules.bin found at {auto_mask_modules_path}; continuing without it.")

        model.to(weight_dtype)

        # Collect all trainable parameters
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        trainable_params += list(mask_encoder.parameters())
        trainable_params += list(mask_decoder.parameters())
        opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.adam_weight_decay)
    else:
        trainable_params = list(model.parameters()) + list(mask_encoder.parameters()) + list(mask_decoder.parameters())
        opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.adam_weight_decay)

    if accelerator.is_main_process:
        total_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Total trainable parameters: {total_trainable:,}")
        logger.info(f"LoRA targets: {args.lora_target_modules}")
        logger.info(f"LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
        logger.info(f"lambda_mask={args.lambda_mask}")

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
    # Compute num_mask_tokens for the collator
    # For 256x256 images: latent = (4, 32, 32), patch_size=2 -> 16x16 = 256 tokens
    # This must match the mask latent spatial dims
    num_mask_tokens = (256 // 8 // 2) * (256 // 8 // 2)  # = 256 for 256x256

    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=model.llm.config.hidden_size if hasattr(model, 'llm') else model.base_model.model.llm.config.hidden_size,
        keep_raw_resolution=args.keep_raw_resolution,
        num_mask_tokens=num_mask_tokens,
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
    if args.max_train_steps is not None:
        max_train_steps = min(max_train_steps, args.max_train_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    model.train()
    mask_encoder.train()
    mask_decoder.train()

    if ema is not None:
        update_ema(ema, model, decay=0)
        ema.eval()

    if ema is not None:
        model, ema, mask_encoder, mask_decoder = accelerator.prepare(model, ema, mask_encoder, mask_decoder)
    else:
        model, mask_encoder, mask_decoder = accelerator.prepare(model, mask_encoder, mask_decoder)
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

        # Explicitly reload mask modules from the sibling step directory if available.
        mask_modules_resume_path = args.mask_modules_resume_path
        if mask_modules_resume_path is None and global_step > 0:
            candidate_dir = os.path.join(os.path.dirname(args.resume_from_checkpoint.rstrip("/")), f"{global_step:07d}")
            candidate_path = os.path.join(candidate_dir, "mask_modules.bin")
            if os.path.exists(candidate_path):
                mask_modules_resume_path = candidate_path

        if mask_modules_resume_path is not None:
            _maybe_load_mask_modules_from_checkpoint(
                model, mask_encoder, mask_decoder, mask_modules_resume_path, logger=logger if accelerator.is_main_process else None
            )

    train_steps = global_step * args.gradient_accumulation_steps
    log_steps = 0
    running_loss = 0
    running_loss_img = 0
    running_loss_mask = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs (max_train_steps={max_train_steps})...")

    for epoch in range(first_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")

        if epoch == first_epoch and resume_step_in_epoch > 0:
            active_loader = accelerator.skip_first_batches(
                loader, resume_step_in_epoch * args.gradient_accumulation_steps
            )
        else:
            active_loader = loader

        for data in active_loader:
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(model):
                # --- Encode images and masks ---
                with torch.no_grad():
                    output_images = data["output_images"]
                    input_pixel_values = data["input_pixel_values"]
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

                # Encode GT masks to mask latent
                output_anatomy_masks = data.get("output_anatomy_masks", None)
                if output_anatomy_masks is None:
                    raise RuntimeError(
                        "output_anatomy_masks not found in batch. Ensure your JSONL file has 'output_mask'."
                    )

                # Map {0,1} -> [-1,1]
                mask_cont = 2.0 * output_anatomy_masks.to(device=device, dtype=weight_dtype) - 1.0
                # Encode through mask encoder (WITH gradients for mask_encoder training)
                x1_mask = mask_encoder(mask_cont)

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

                loss_dict = training_losses_joint_mask(
                    model=model,
                    x1_img=output_images,
                    x1_mask=x1_mask,
                    model_kwargs=model_kwargs,
                    lambda_mask=args.lambda_mask,
                )
                loss = loss_dict["loss"]

                running_loss += loss.item()
                running_loss_img += loss_dict["loss_img"].item()
                running_loss_mask += loss_dict["loss_mask"].item()

                accelerator.backward(loss)
                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    params_to_clip = list(model.parameters()) + list(mask_encoder.parameters()) + list(mask_decoder.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()

                log_steps += 1
                train_steps += 1

                accelerator.log(
                    {
                        "training_loss_total": loss.item(),
                        "training_loss_img": loss_dict["loss_img"].item(),
                        "training_loss_mask": loss_dict["loss_mask"].item(),
                    },
                    step=train_steps,
                )

                if train_steps % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    if accelerator.sync_gradients and ema is not None:
                        update_ema(ema, model)

                if train_steps % (args.log_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / args.gradient_accumulation_steps / (end_time - start_time)
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    avg_loss_img = torch.tensor(running_loss_img / log_steps, device=device)
                    avg_loss_mask = torch.tensor(running_loss_mask / log_steps, device=device)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_loss_img, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_loss_mask, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / accelerator.num_processes
                    avg_loss_img = avg_loss_img.item() / accelerator.num_processes
                    avg_loss_mask = avg_loss_mask.item() / accelerator.num_processes

                    if accelerator.is_main_process:
                        cur_lr = opt.param_groups[0]["lr"]
                        logger.info(
                            f"(step={global_step:07d}) "
                            f"Loss: {avg_loss:.4f} (img={avg_loss_img:.4f}, mask={avg_loss_mask:.4f}), "
                            f"Steps/Sec: {steps_per_sec:.2f}, Epoch: {train_steps / len(loader):.2f}, LR: {cur_lr}"
                        )

                    running_loss = 0
                    running_loss_img = 0
                    running_loss_mask = 0
                    log_steps = 0
                    start_time = time()

            if train_steps % (args.ckpt_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                current_step = int(train_steps / args.gradient_accumulation_steps)

                # Save accelerator state
                accelerator_state_path = f"{checkpoint_dir}/checkpoint-{current_step}"
                accelerator.save_state(accelerator_state_path)
                if accelerator.is_main_process:
                    logger.info(f"Saved accelerator state to {accelerator_state_path}")

                if accelerator.is_main_process:
                    checkpoint_path = f"{checkpoint_dir}/{current_step:07d}/"
                    os.makedirs(checkpoint_path, exist_ok=True)

                    # Save LoRA weights
                    if args.use_lora:
                        if hasattr(model, "module"):
                            model.module.save_pretrained(checkpoint_path)
                        else:
                            model.save_pretrained(checkpoint_path)

                    # Save mask modules (CRITICAL: not covered by PEFT)
                    mask_state = get_mask_module_state_dict(model, mask_encoder, mask_decoder)
                    mask_modules_path = os.path.join(checkpoint_path, "mask_modules.bin")
                    torch.save(mask_state, mask_modules_path)
                    logger.info(f"Saved mask_modules.bin ({len(mask_state)} tensors) to {checkpoint_path}")

                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

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
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        help="LoRA target module names.",
    )
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
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
    )
    parser.add_argument(
        "--lora_resume_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mask_modules_resume_path",
        type=str,
        default=None,
        help="Path to mask_modules.bin for resuming mask module weights.",
    )

    # --- Joint mask specific ---
    parser.add_argument(
        "--mask_latent_channels",
        type=int,
        default=4,
        help="Number of channels in the mask latent space.",
    )
    parser.add_argument(
        "--lambda_mask",
        type=float,
        default=0.25,
        help="Weight for the mask velocity loss.",
    )
    parser.add_argument(
        "--mask_ae_ckpt",
        type=str,
        default=None,
        help="Path to pretrained mask autoencoder checkpoint.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Max training steps (overrides epochs if set).",
    )

    args = parser.parse_args()
    assert args.max_image_size % 16 == 0, "Image size must be divisible by 16."

    if args.lora_resume_path and args.resume_from_checkpoint:
        raise ValueError(
            "--lora_resume_path and --resume_from_checkpoint are mutually exclusive."
        )

    main(args)
