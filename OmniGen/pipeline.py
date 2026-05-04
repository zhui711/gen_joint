import os
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import gc

from PIL import Image
import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from diffusers.models import AutoencoderKL
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from safetensors.torch import load_file

from OmniGen import OmniGen, OmniGenProcessor, OmniGenScheduler


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from OmniGen import OmniGenPipeline
        >>> pipe = FluxControlNetPipeline.from_pretrained(
        ...     base_model
        ... )
        >>> prompt = "A woman holds a bouquet of flowers and faces the camera"
        >>> image = pipe(
        ...     prompt,
        ...     guidance_scale=2.5,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("t2i.png")
        ```
"""


class OmniGenPipeline:
    def __init__(
        self,
        vae: AutoencoderKL,
        model: OmniGen,
        processor: OmniGenProcessor,
        device: Union[str, torch.device] = None,
        mask_encoder=None,
        mask_decoder=None,
    ):
        self.vae = vae
        self.model = model
        self.processor = processor
        self.device = device
        self.mask_encoder = mask_encoder
        self.mask_decoder = mask_decoder

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                logger.info("Don't detect any available GPUs, using CPU instead, this may take long time to generate image!!!")
                self.device = torch.device("cpu")

        self.model.eval()
        self.vae.eval()
        if self.mask_encoder is not None:
            self.mask_encoder.eval()
        if self.mask_decoder is not None:
            self.mask_decoder.eval()

        self.model_cpu_offload = False

    @classmethod
    def from_pretrained(cls, model_name, vae_path: str=None):
        if not os.path.exists(model_name) or (not os.path.exists(os.path.join(model_name, 'model.safetensors')) and model_name == "Shitao/OmniGen-v1"):
            print("Model not found, downloading...")
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5', 'model.pt'])
            print(f"Downloaded model to {model_name}")
        model = OmniGen.from_pretrained(model_name)
        processor = OmniGenProcessor.from_pretrained(model_name)

        if os.path.exists(os.path.join(model_name, "vae")):
            vae = AutoencoderKL.from_pretrained(os.path.join(model_name, "vae"))
        elif vae_path is not None:
            vae = AutoencoderKL.from_pretrained(vae_path)
        else:
            logger.info(f"No VAE found in {model_name}, downloading stabilityai/sdxl-vae from HF")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

        return cls(vae, model, processor)

    def merge_lora(self, lora_path: str):
        model = PeftModel.from_pretrained(self.model, lora_path)
        model.merge_and_unload()

        self.model = model

    def to(self, device: Union[str, torch.device]):
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)
        self.vae.to(device)
        if self.mask_encoder is not None:
            self.mask_encoder.to(device)
        if self.mask_decoder is not None:
            self.mask_decoder.to(device)
        self.device = device

    def vae_encode(self, x, dtype):
        if self.vae.config.shift_factor is not None:
            x = self.vae.encode(x).latent_dist.sample()
            x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            x = self.vae.encode(x).latent_dist.sample().mul_(self.vae.config.scaling_factor)
        x = x.to(dtype)
        return x

    def move_to_device(self, data):
        if isinstance(data, list):
            return [x.to(self.device) for x in data]
        return data.to(self.device)

    def enable_model_cpu_offload(self):
        self.model_cpu_offload = True
        self.model.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

    def disable_model_cpu_offload(self):
        self.model_cpu_offload = False
        self.model.to(self.device)
        self.vae.to(self.device)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Union[List[str], List[List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        offload_model: bool = False,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = True,
        use_input_image_size_as_output: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = None,
        output_type: str = "pil",
        save_mask: bool = False,
        mask_threshold: float = 0.0,
        mask_scale_factor: float = 1.0,
        ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            input_images (`List[str]` or `List[List[str]]`, *optional*):
                The list of input images.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 3.0):
                Guidance scale for CFG.
            use_img_guidance (`bool`, *optional*, defaults to True):
                Whether to use image guidance.
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Image guidance scale.
            max_input_image_size (`int`, *optional*, defaults to 1024):
                Maximum input image size.
            separate_cfg_infer (`bool`, *optional*, defaults to True):
                Whether to use separate CFG inference.
            offload_model (`bool`, *optional*, defaults to False):
                Whether to offload model to CPU.
            use_kv_cache (`bool`, *optional*, defaults to True):
                Whether to use KV cache.
            offload_kv_cache (`bool`, *optional*, defaults to True):
                Whether to offload KV cache.
            use_input_image_size_as_output (`bool`, *optional*, defaults to False):
                Whether to use input image size as output.
            dtype (`torch.dtype`, *optional*, defaults to torch.bfloat16):
                Data type.
            seed (`int`, *optional*):
                Random seed.
            output_type (`str`, *optional*, defaults to "pil"):
                Output type.
            save_mask (`bool`, *optional*, defaults to False):
                If True and joint mask mode is active, also return decoded masks.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                Threshold for binarizing predicted masks.
            mask_scale_factor (`float`, *optional*, defaults to 1.0):
                Scale factor used during mask flow training. Generated mask
                latents are divided by this value before MaskDecoder.
        Examples:

        Returns:
            A list of generated images. If save_mask=True, returns (images, masks).
        """
        # check inputs:
        if use_input_image_size_as_output:
            assert isinstance(prompt, str) and len(input_images) == 1
        else:
            assert height%16 == 0 and width%16 == 0
        if input_images is None:
            use_img_guidance = False
        if mask_scale_factor <= 0:
            raise ValueError("mask_scale_factor must be > 0.")
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images] if input_images is not None else None


        # set model and processor
        if max_input_image_size != self.processor.max_image_size:
            self.processor = OmniGenProcessor(self.processor.text_tokenizer, max_image_size=max_input_image_size)
        self.model.to(dtype)
        if offload_model:
            self.enable_model_cpu_offload()
        else:
            self.disable_model_cpu_offload()

        # Compute mask token count for joint mode
        use_joint = self.model.use_joint_mask and self.mask_decoder is not None
        if use_joint:
            _latent_h, _latent_w = height // 8, width // 8
            _num_mask_tokens = (_latent_h // self.model.patch_size) * (_latent_w // self.model.patch_size)
        else:
            _num_mask_tokens = 0

        input_data = self.processor(prompt, input_images, height=height, width=width, use_img_cfg=use_img_guidance, separate_cfg_input=separate_cfg_infer, use_input_image_size_as_output=use_input_image_size_as_output, num_mask_tokens=_num_mask_tokens)

        num_prompt = len(prompt)
        num_cfg = 2 if use_img_guidance else 1
        if use_input_image_size_as_output:
            if separate_cfg_infer:
                height, width = input_data['input_pixel_values'][0][0].shape[-2:]
            else:
                height, width = input_data['input_pixel_values'][0].shape[-2:]
        latent_size_h, latent_size_w = height//8, width//8

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=self.device, generator=generator)
        latents = torch.cat([latents]*(1+num_cfg), 0).to(dtype)

        # Initialize mask latents if in joint mode (use_joint already computed above)
        mask_latents = None
        if use_joint:
            mask_latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=self.device, generator=generator)
            mask_latents = torch.cat([mask_latents]*(1+num_cfg), 0).to(dtype)

        if input_images is not None and self.model_cpu_offload: self.vae.to(self.device)
        input_img_latents = []
        if separate_cfg_infer:
            for temp_pixel_values in input_data['input_pixel_values']:
                temp_input_latents = []
                for img in temp_pixel_values:
                    img = self.vae_encode(img.to(self.device), dtype)
                    temp_input_latents.append(img)
                input_img_latents.append(temp_input_latents)
        else:
            for img in input_data['input_pixel_values']:
                img = self.vae_encode(img.to(self.device), dtype)
                input_img_latents.append(img)
        if input_images is not None and self.model_cpu_offload:
            self.vae.to('cpu')
            torch.cuda.empty_cache()
            gc.collect()

        model_kwargs = dict(input_ids=self.move_to_device(input_data['input_ids']),
            input_img_latents=input_img_latents,
            input_image_sizes=input_data['input_image_sizes'],
            attention_mask=self.move_to_device(input_data["attention_mask"]),
            position_ids=self.move_to_device(input_data["position_ids"]),
            cfg_scale=guidance_scale,
            img_cfg_scale=img_guidance_scale,
            use_img_cfg=use_img_guidance,
            use_kv_cache=use_kv_cache,
            offload_model=offload_model,
            )

        if separate_cfg_infer:
            func = self.model.forward_with_separate_cfg
        else:
            func = self.model.forward_with_cfg

        if self.model_cpu_offload:
            for name, param in self.model.named_parameters():
                if 'layers' in name and 'layers.0' not in name:
                    param.data = param.data.cpu()
                else:
                    param.data = param.data.to(self.device)
            for buffer_name, buffer in self.model.named_buffers():
                setattr(self.model, buffer_name, buffer.to(self.device))

        scheduler = OmniGenScheduler(num_steps=num_inference_steps)

        if use_joint:
            # Joint co-denoising
            samples, mask_samples = scheduler.__call_joint__(
                latents, mask_latents, func, model_kwargs,
                use_kv_cache=use_kv_cache, offload_kv_cache=offload_kv_cache
            )
        else:
            # Standard image-only denoising
            samples = scheduler(latents, func, model_kwargs, use_kv_cache=use_kv_cache, offload_kv_cache=offload_kv_cache)
            mask_samples = None

        samples = samples.chunk((1+num_cfg), dim=0)[0]

        if self.model_cpu_offload:
            self.model.to('cpu')
            torch.cuda.empty_cache()
            gc.collect()

        self.vae.to(self.device)
        samples = samples.to(torch.float32)
        if self.vae.config.shift_factor is not None:
            samples = samples / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            samples = samples / self.vae.config.scaling_factor
        samples = self.vae.decode(samples).sample

        if self.model_cpu_offload:
            self.vae.to('cpu')
            torch.cuda.empty_cache()
            gc.collect()

        samples = (samples * 0.5 + 0.5).clamp(0, 1)

        if output_type == "pt":
            output_images = samples
        else:
            output_samples = (samples * 255).to("cpu", dtype=torch.uint8)
            output_samples = output_samples.permute(0, 2, 3, 1).numpy()
            output_images = []
            for i, sample in enumerate(output_samples):
                output_images.append(Image.fromarray(sample))

        # Decode mask if requested
        output_masks = None
        if use_joint and save_mask and mask_samples is not None:
            mask_samples_clean = mask_samples.chunk((1+num_cfg), dim=0)[0]
            mask_samples_clean = mask_samples_clean.to(torch.float32)
            if self.mask_decoder is not None:
                self.mask_decoder.to(self.device)
                mask_samples_unscaled = mask_samples_clean / mask_scale_factor
                decoded_masks = self.mask_decoder(mask_samples_unscaled)  # (B, 10, 256, 256)
                # Threshold: tanh output in [-1, 1], threshold at mask_threshold
                binary_masks = (decoded_masks > mask_threshold).float()
                output_masks = binary_masks.cpu()
                if self.model_cpu_offload:
                    self.mask_decoder.to('cpu')

        torch.cuda.empty_cache()
        gc.collect()

        if save_mask:
            return output_images, output_masks
        return output_images
