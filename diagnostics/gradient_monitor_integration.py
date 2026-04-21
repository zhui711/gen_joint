"""
gradient_monitor_integration.py - Ready-to-integrate gradient monitoring code

This module provides the exact code snippet to add to train_anatomy.py
for tracking gradient norms from diffusion vs. anatomy losses.

INTEGRATION STEPS:
1. Copy the compute_separate_grad_norms() function to your training script
2. Add the logging code block inside your training loop after loss computation
3. Monitor the logs for gradient conflict signals

See SEGMSE_DEGRADATION_ANALYSIS_REPORT.md Part 2 for full documentation.
"""

import torch
from typing import Dict, List


def compute_separate_grad_norms(
    model,
    loss_diffusion: torch.Tensor,
    loss_anatomy: torch.Tensor,
    lambda_anatomy: float,
    retain_graph: bool = True,
) -> Dict[str, float]:
    """
    Compute gradient norms separately for diffusion and anatomy losses.

    This function uses torch.autograd.grad() to compute gradients without
    modifying the .grad attribute of parameters, allowing us to analyze
    gradient conflict before the actual backward pass.

    Args:
        model: The OmniGen model (with LoRA)
        loss_diffusion: Diffusion loss tensor (requires_grad=True)
        loss_anatomy: Raw anatomy loss before lambda scaling (requires_grad=True)
        lambda_anatomy: Scaling factor for anatomy loss
        retain_graph: Keep computation graph for subsequent backward() call

    Returns:
        Dict with:
            - grad_norm_diffusion: L2 norm of diffusion gradients
            - grad_norm_anatomy: L2 norm of scaled anatomy gradients
            - grad_norm_ratio: anatomy / diffusion (>1 means anatomy dominates)
            - cosine_similarity: dot(g_diff, g_anat) / (|g_diff| * |g_anat|)
                                 (<0 means CONFLICT)

    Usage:
        After computing losses but BEFORE backward():

        grad_metrics = compute_separate_grad_norms(
            model=model,
            loss_diffusion=loss_dict["loss_diffusion"],
            loss_anatomy=loss_dict["loss_anatomy"],
            lambda_anatomy=args.lambda_anatomy,
            retain_graph=True,
        )

        if grad_metrics["cosine_similarity"] < 0:
            logger.warning("Gradient conflict detected!")

        loss_total.backward()  # Then proceed with backward
    """
    # Get LoRA parameters only (these are the trainable parameters)
    lora_params = [p for n, p in model.named_parameters()
                   if "lora" in n.lower() and p.requires_grad]

    if len(lora_params) == 0:
        # Fallback: get all trainable parameters
        lora_params = [p for p in model.parameters() if p.requires_grad]

    if len(lora_params) == 0:
        return {
            "grad_norm_diffusion": 0.0,
            "grad_norm_anatomy": 0.0,
            "grad_norm_ratio": 0.0,
            "cosine_similarity": 0.0,
        }

    # Compute diffusion gradients
    try:
        grads_diff = torch.autograd.grad(
            loss_diffusion,
            lora_params,
            retain_graph=True,
            allow_unused=True,
        )
    except RuntimeError as e:
        print(f"[GradNorm] Warning: Could not compute diffusion gradients: {e}")
        grads_diff = [None] * len(lora_params)

    # Compute anatomy gradients (with lambda scaling applied)
    scaled_anat = lambda_anatomy * loss_anatomy
    try:
        grads_anat = torch.autograd.grad(
            scaled_anat,
            lora_params,
            retain_graph=retain_graph,
            allow_unused=True,
        )
    except RuntimeError as e:
        print(f"[GradNorm] Warning: Could not compute anatomy gradients: {e}")
        grads_anat = [None] * len(lora_params)

    # Compute norms and cosine similarity
    norm_diff_sq = 0.0
    norm_anat_sq = 0.0
    dot_product = 0.0

    for gd, ga in zip(grads_diff, grads_anat):
        if gd is not None:
            norm_diff_sq += gd.norm(2).item() ** 2
        if ga is not None:
            norm_anat_sq += ga.norm(2).item() ** 2
        if gd is not None and ga is not None:
            dot_product += (gd * ga).sum().item()

    norm_diff = norm_diff_sq ** 0.5
    norm_anat = norm_anat_sq ** 0.5

    # Cosine similarity
    if norm_diff > 1e-8 and norm_anat > 1e-8:
        cos_sim = dot_product / (norm_diff * norm_anat)
    else:
        cos_sim = 0.0

    # Ratio (anatomy / diffusion)
    ratio = norm_anat / norm_diff if norm_diff > 1e-8 else float('inf')

    return {
        "grad_norm_diffusion": norm_diff,
        "grad_norm_anatomy": norm_anat,
        "grad_norm_ratio": ratio,
        "cosine_similarity": cos_sim,
    }


# ===========================================================================
# INTEGRATION SNIPPET FOR train_anatomy.py
# ===========================================================================
# Copy everything below this line into your training loop

INTEGRATION_CODE = '''
# ===========================================================================
# GRADIENT MONITORING - Add after loss computation, before backward()
# ===========================================================================
# Step 1: Import at the top of train_anatomy.py:
# from diagnostics.gradient_monitor_integration import compute_separate_grad_norms

# Step 2: Add this code block inside your training loop:

    loss_dict = training_losses_with_anatomy(
        model=model,
        x1=output_images,
        model_kwargs=model_kwargs,
        output_images_pixel=output_images_pixel,
        vae=vae,
        seg_model=seg_model,
        lambda_anatomy=args.lambda_anatomy,
        anatomy_subbatch_size=args.anatomy_subbatch_size,
    )

    loss_total = loss_dict["loss_total"]
    loss_diffusion = loss_dict["loss_diffusion"]
    loss_anatomy = loss_dict["loss_anatomy"]

    # ===== GRADIENT MONITORING (every 100 steps) =====
    global_step = train_steps // args.gradient_accumulation_steps
    if global_step % 100 == 0:
        if loss_diffusion.requires_grad and loss_anatomy.requires_grad:
            grad_metrics = compute_separate_grad_norms(
                model=model,
                loss_diffusion=loss_diffusion,
                loss_anatomy=loss_anatomy,
                lambda_anatomy=args.lambda_anatomy,
                retain_graph=True,  # IMPORTANT: Keep graph for backward()
            )

            if accelerator.is_main_process:
                # Log to console
                logger.info(
                    f"[GradNorm] Step {global_step}: "
                    f"||g_diff||={grad_metrics['grad_norm_diffusion']:.4f}, "
                    f"||g_anat||={grad_metrics['grad_norm_anatomy']:.4f}, "
                    f"ratio={grad_metrics['grad_norm_ratio']:.4f}, "
                    f"cos_sim={grad_metrics['cosine_similarity']:.4f}"
                )

                # Log to TensorBoard
                accelerator.log({
                    "gradients/norm_diffusion": grad_metrics["grad_norm_diffusion"],
                    "gradients/norm_anatomy": grad_metrics["grad_norm_anatomy"],
                    "gradients/ratio_anat_over_diff": grad_metrics["grad_norm_ratio"],
                    "gradients/cosine_similarity": grad_metrics["cosine_similarity"],
                }, step=global_step)

                # ALERT on conflict or imbalance
                if grad_metrics["cosine_similarity"] < -0.1:
                    logger.warning(
                        f"[GRADIENT CONFLICT] Step {global_step}: "
                        f"Gradients opposing! cos_sim={grad_metrics['cosine_similarity']:.4f}"
                    )
                if grad_metrics["grad_norm_ratio"] > 2.0:
                    logger.warning(
                        f"[GRADIENT IMBALANCE] Step {global_step}: "
                        f"Anatomy gradients dominating! ratio={grad_metrics['grad_norm_ratio']:.4f}"
                    )
    # ===== END GRADIENT MONITORING =====

    accelerator.backward(loss_total)
    # ... rest of training loop
'''


# ===========================================================================
# Interpretation Guide
# ===========================================================================

INTERPRETATION_GUIDE = """
GRADIENT MONITORING INTERPRETATION GUIDE
========================================

1. GRADIENT NORM RATIO (anatomy / diffusion)
   -----------------------------------------
   ratio < 0.1  : Diffusion dominates; anatomy loss has minimal effect
   ratio 0.1-1.0: HEALTHY - balanced gradient contributions
   ratio 1.0-2.0: Anatomy starting to dominate; consider reducing lambda
   ratio > 2.0  : Anatomy DOMINATES; reduce lambda_anatomy by 2-5x
   ratio > 5.0  : CRITICAL; anatomy gradients overwhelming diffusion

2. COSINE SIMILARITY
   ------------------
   cos_sim > 0.7 : EXCELLENT - losses cooperate toward same goal
   cos_sim 0.3-0.7: OK - some alignment, reasonable tradeoff
   cos_sim 0.0-0.3: WARNING - losses are nearly orthogonal
   cos_sim < 0   : CONFLICT - losses push in opposite directions!

3. ACTION RECOMMENDATIONS
   ----------------------

   If cos_sim < 0 (conflict):
     -> The losses fundamentally disagree on gradient direction
     -> Options:
        a) Reduce lambda_anatomy significantly (try 0.1x current)
        b) Switch to alternative loss (thresholded or blurred)
        c) Implement gradient projection (PCGrad)
        d) Remove anatomy loss entirely

   If ratio > 2.0 (anatomy dominates):
     -> Anatomy gradients are overwhelming diffusion training
     -> Actions:
        a) Reduce lambda_anatomy (try 0.5x current)
        b) Apply separate gradient clipping to anatomy loss
        c) Use gradient normalization

   If ratio < 0.1 (anatomy negligible):
     -> Anatomy loss is not affecting training
     -> Actions:
        a) Increase lambda_anatomy (try 2-5x current)
        b) Check if anatomy loss is computing correctly
        c) Check if seg_model gradients are flowing

4. EXAMPLE LOG OUTPUT
   ------------------
   [GradNorm] Step 1000: ||g_diff||=0.0234, ||g_anat||=0.0456, ratio=1.95, cos_sim=0.23

   Interpretation:
     - ratio=1.95: Anatomy gradients almost 2x diffusion (borderline high)
     - cos_sim=0.23: Weak alignment (losses not well coordinated)
     -> Consider reducing lambda_anatomy slightly
"""


def print_interpretation_guide():
    """Print the interpretation guide."""
    print(INTERPRETATION_GUIDE)


if __name__ == "__main__":
    print("Gradient Monitoring Integration Code")
    print("=" * 60)
    print(INTEGRATION_CODE)
    print("\n")
    print_interpretation_guide()
