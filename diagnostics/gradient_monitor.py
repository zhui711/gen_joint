#!/usr/bin/env python3
"""
Tool 2: Gradient Norm Monitoring Hook

This module provides utilities to measure the L2 gradient norm contribution
from Loss_diffusion vs. Loss_anatomy on LoRA parameters.

PROBLEM: When we do `loss_total.backward()`, we can't directly separate
the gradient contributions from each loss term.

SOLUTION: We use two approaches:
  1. Two-pass method (more accurate, higher cost): Compute gradients separately
     using `torch.autograd.grad()` with `retain_graph=True`
  2. Gradient ratio estimation (efficient): Log gradient norms at different
     points and use loss magnitudes to infer relative contributions

Usage:
    # In your training script, import and use:
    from diagnostics.gradient_monitor import GradientMonitor, log_gradient_analysis

    # Option 1: Full monitoring (slower, every N steps)
    monitor = GradientMonitor(model, log_every=100)

    for step, batch in enumerate(dataloader):
        loss_dict = training_losses_with_anatomy(...)
        loss_total = loss_dict["loss_total"]
        loss_total.backward()

        # Log gradient analysis every N steps
        monitor.log_step(
            step=step,
            loss_diffusion=loss_dict["loss_diffusion"],
            loss_anatomy=loss_dict["loss_anatomy"],
            lambda_anatomy=args.lambda_anatomy,
        )

        optimizer.step()
        optimizer.zero_grad()

    # Option 2: Quick logging function for existing code
    if step % 100 == 0:
        log_gradient_analysis(
            model=model,
            loss_diffusion=loss_diffusion,
            loss_anatomy=loss_anatomy,
            lambda_anatomy=0.005,
            step=step,
            retain_graph=True,  # If you need gradients for backprop
        )
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GradientMonitor")


def get_lora_parameters(model) -> Dict[str, torch.nn.Parameter]:
    """
    Extract LoRA parameters from a PEFT model.

    Returns a dict mapping parameter names to parameters.
    Only includes parameters with 'lora' in the name and requires_grad=True.
    """
    lora_params = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_params[name] = param
    return lora_params


def get_transformer_layer_parameters(model, layer_idx: int = 0) -> Dict[str, torch.nn.Parameter]:
    """
    Extract parameters from a specific transformer layer.

    Args:
        model: The OmniGen model
        layer_idx: Which transformer layer to monitor (0-indexed)

    Returns:
        Dict mapping parameter names to parameters
    """
    layer_params = {}
    for name, param in model.named_parameters():
        if f"layers.{layer_idx}." in name and param.requires_grad:
            layer_params[name] = param
    return layer_params


def compute_grad_norm(params: Dict[str, torch.nn.Parameter], norm_type: float = 2.0) -> float:
    """
    Compute the L2 (or Lp) norm of gradients for given parameters.

    Args:
        params: Dict of named parameters
        norm_type: Type of norm (default L2)

    Returns:
        Total gradient norm as float
    """
    total_norm = 0.0
    for name, param in params.items():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    return total_norm ** (1.0 / norm_type)


def compute_separate_gradient_norms(
    model,
    loss_diffusion: torch.Tensor,
    loss_anatomy: torch.Tensor,
    lambda_anatomy: float,
    params: Optional[Dict[str, torch.nn.Parameter]] = None,
    retain_graph: bool = True,
) -> Dict[str, float]:
    """
    Compute gradient norms separately for diffusion and anatomy losses.

    This uses `torch.autograd.grad()` to compute gradients without modifying
    the parameter's `.grad` attribute.

    WARNING: This requires `retain_graph=True` on both calls, which doubles
    the memory usage during gradient computation. Use sparingly (every N steps).

    Args:
        model: The model being trained
        loss_diffusion: Diffusion loss tensor (with grad)
        loss_anatomy: Raw anatomy loss tensor (before lambda scaling, with grad)
        lambda_anatomy: The lambda weight for anatomy loss
        params: Optional dict of parameters to measure (default: LoRA params)
        retain_graph: Whether to retain the computation graph (True for training)

    Returns:
        Dict with:
            - grad_norm_diffusion: L2 norm from diffusion loss
            - grad_norm_anatomy: L2 norm from anatomy loss (after lambda scaling)
            - grad_norm_ratio: anatomy / diffusion (conflict indicator)
            - cosine_similarity: Cosine similarity of gradient directions
    """
    if params is None:
        params = get_lora_parameters(model)

    if len(params) == 0:
        logger.warning("No LoRA parameters found! Check model configuration.")
        return {
            "grad_norm_diffusion": 0.0,
            "grad_norm_anatomy": 0.0,
            "grad_norm_ratio": 0.0,
            "cosine_similarity": 0.0,
        }

    param_list = list(params.values())

    # Compute gradients from diffusion loss
    try:
        grads_diffusion = torch.autograd.grad(
            loss_diffusion,
            param_list,
            retain_graph=True,
            allow_unused=True,
        )
    except RuntimeError as e:
        logger.warning(f"Failed to compute diffusion gradients: {e}")
        grads_diffusion = [None] * len(param_list)

    # Compute gradients from anatomy loss (scaled by lambda)
    scaled_anatomy_loss = lambda_anatomy * loss_anatomy
    try:
        grads_anatomy = torch.autograd.grad(
            scaled_anatomy_loss,
            param_list,
            retain_graph=retain_graph,
            allow_unused=True,
        )
    except RuntimeError as e:
        logger.warning(f"Failed to compute anatomy gradients: {e}")
        grads_anatomy = [None] * len(param_list)

    # Compute norms
    norm_diff = 0.0
    norm_anat = 0.0
    dot_product = 0.0

    for g_diff, g_anat in zip(grads_diffusion, grads_anatomy):
        if g_diff is not None:
            norm_diff += g_diff.norm(2).item() ** 2
        if g_anat is not None:
            norm_anat += g_anat.norm(2).item() ** 2
        if g_diff is not None and g_anat is not None:
            dot_product += (g_diff * g_anat).sum().item()

    norm_diff = norm_diff ** 0.5
    norm_anat = norm_anat ** 0.5

    # Cosine similarity between gradient directions
    if norm_diff > 1e-8 and norm_anat > 1e-8:
        cosine_sim = dot_product / (norm_diff * norm_anat)
    else:
        cosine_sim = 0.0

    # Ratio (anatomy / diffusion)
    if norm_diff > 1e-8:
        ratio = norm_anat / norm_diff
    else:
        ratio = float("inf") if norm_anat > 1e-8 else 0.0

    return {
        "grad_norm_diffusion": norm_diff,
        "grad_norm_anatomy": norm_anat,
        "grad_norm_ratio": ratio,
        "cosine_similarity": cosine_sim,
    }


class GradientMonitor:
    """
    A class to monitor gradient norms during training.

    Tracks:
        - Per-step gradient norms from each loss component
        - Running statistics (mean, std)
        - Gradient conflict detection

    Example:
        monitor = GradientMonitor(model, log_every=100)

        for step, batch in enumerate(dataloader):
            loss_dict = compute_losses(...)
            loss_dict["loss_total"].backward()

            monitor.log_step(
                step=step,
                loss_diffusion=loss_dict["loss_diffusion"],
                loss_anatomy=loss_dict["loss_anatomy"],
                lambda_anatomy=0.005,
            )

            optimizer.step()
            optimizer.zero_grad()

        # Get summary at the end
        summary = monitor.get_summary()
    """

    def __init__(
        self,
        model,
        log_every: int = 100,
        params: Optional[Dict[str, torch.nn.Parameter]] = None,
        writer=None,  # Optional TensorBoard SummaryWriter
    ):
        """
        Args:
            model: The model being trained
            log_every: Log detailed gradient analysis every N steps
            params: Specific parameters to monitor (default: LoRA params)
            writer: Optional TensorBoard writer for logging
        """
        self.model = model
        self.log_every = log_every
        self.params = params or get_lora_parameters(model)
        self.writer = writer

        # History tracking
        self.history = defaultdict(list)
        self.conflict_count = 0

        logger.info(f"GradientMonitor initialized with {len(self.params)} LoRA parameters")

    def log_step(
        self,
        step: int,
        loss_diffusion: torch.Tensor,
        loss_anatomy: torch.Tensor,
        lambda_anatomy: float,
        force_log: bool = False,
    ):
        """
        Log gradient analysis for a single training step.

        Args:
            step: Current training step
            loss_diffusion: Diffusion loss (must have grad)
            loss_anatomy: Anatomy loss (must have grad)
            lambda_anatomy: Anatomy loss weight
            force_log: Force logging even if not at log_every interval
        """
        if not force_log and step % self.log_every != 0:
            return

        # Don't compute if losses don't require grad
        if not loss_diffusion.requires_grad or not loss_anatomy.requires_grad:
            logger.warning(f"Step {step}: Losses don't require grad, skipping analysis")
            return

        # Compute separate gradient norms
        metrics = compute_separate_gradient_norms(
            self.model,
            loss_diffusion,
            loss_anatomy,
            lambda_anatomy,
            self.params,
            retain_graph=True,  # Keep graph for actual backward
        )

        # Also compute total gradient norm (from combined loss)
        # This is computed from .grad after backward, so we estimate from sum
        metrics["grad_norm_total_estimated"] = (
            metrics["grad_norm_diffusion"] ** 2 +
            metrics["grad_norm_anatomy"] ** 2 +
            2 * metrics["grad_norm_diffusion"] * metrics["grad_norm_anatomy"] * metrics["cosine_similarity"]
        ) ** 0.5

        # Store in history
        for key, value in metrics.items():
            self.history[key].append(value)
        self.history["step"].append(step)
        self.history["loss_diffusion"].append(loss_diffusion.item())
        self.history["loss_anatomy"].append(loss_anatomy.item())
        self.history["loss_anatomy_weighted"].append(lambda_anatomy * loss_anatomy.item())

        # Detect gradient conflict (cosine similarity < 0)
        if metrics["cosine_similarity"] < -0.1:
            self.conflict_count += 1
            logger.warning(
                f"Step {step}: GRADIENT CONFLICT DETECTED! "
                f"Cosine similarity = {metrics['cosine_similarity']:.4f}"
            )

        # Log to console
        logger.info(
            f"Step {step} | "
            f"GradNorm(diff)={metrics['grad_norm_diffusion']:.4f} | "
            f"GradNorm(anat)={metrics['grad_norm_anatomy']:.4f} | "
            f"Ratio={metrics['grad_norm_ratio']:.4f} | "
            f"CosSim={metrics['cosine_similarity']:.4f}"
        )

        # Log to TensorBoard if available
        if self.writer is not None:
            self.writer.add_scalar("gradients/norm_diffusion", metrics["grad_norm_diffusion"], step)
            self.writer.add_scalar("gradients/norm_anatomy", metrics["grad_norm_anatomy"], step)
            self.writer.add_scalar("gradients/ratio", metrics["grad_norm_ratio"], step)
            self.writer.add_scalar("gradients/cosine_similarity", metrics["cosine_similarity"], step)

    def get_summary(self) -> Dict:
        """
        Get summary statistics from all logged steps.

        Returns:
            Dict with mean, std, min, max for each metric
        """
        import numpy as np

        summary = {
            "n_logged_steps": len(self.history["step"]),
            "conflict_count": self.conflict_count,
        }

        for key in ["grad_norm_diffusion", "grad_norm_anatomy", "grad_norm_ratio", "cosine_similarity"]:
            if self.history[key]:
                values = np.array(self.history[key])
                summary[key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }

        return summary

    def diagnose(self) -> str:
        """
        Generate a diagnostic report based on gradient monitoring.

        Returns:
            String with diagnostic interpretation
        """
        summary = self.get_summary()

        report = []
        report.append("=" * 60)
        report.append("GRADIENT MONITORING DIAGNOSTIC REPORT")
        report.append("=" * 60)
        report.append(f"Total logged steps: {summary['n_logged_steps']}")
        report.append(f"Gradient conflict events: {summary['conflict_count']}")
        report.append("")

        if "grad_norm_ratio" in summary:
            ratio = summary["grad_norm_ratio"]
            report.append(f"Gradient Norm Ratio (anatomy/diffusion):")
            report.append(f"  Mean: {ratio['mean']:.4f}")
            report.append(f"  Std:  {ratio['std']:.4f}")
            report.append(f"  Range: [{ratio['min']:.4f}, {ratio['max']:.4f}]")
            report.append("")

            if ratio["mean"] > 2.0:
                report.append("[WARNING] Anatomy gradients DOMINATE diffusion gradients!")
                report.append("  -> Reduce lambda_anatomy (try 0.5x current value)")
                report.append("  -> Consider gradient clipping for anatomy loss")
            elif ratio["mean"] < 0.1:
                report.append("[INFO] Diffusion gradients dominate (anatomy has little effect)")
                report.append("  -> Consider increasing lambda_anatomy if anatomy matters")
            else:
                report.append("[OK] Gradient magnitudes are reasonably balanced")

        report.append("")

        if "cosine_similarity" in summary:
            cos_sim = summary["cosine_similarity"]
            report.append(f"Gradient Cosine Similarity:")
            report.append(f"  Mean: {cos_sim['mean']:.4f}")
            report.append(f"  Min:  {cos_sim['min']:.4f}")
            report.append("")

            if cos_sim["mean"] < 0:
                report.append("[CRITICAL] Gradients are OPPOSED on average!")
                report.append("  -> The losses are conflicting - model is being pulled in opposite directions")
                report.append("  -> Consider: gradient projection, PCGrad, or removing anatomy loss")
            elif cos_sim["mean"] < 0.3:
                report.append("[WARNING] Gradients are nearly orthogonal")
                report.append("  -> Losses may be competing rather than cooperating")
            elif cos_sim["mean"] > 0.7:
                report.append("[GOOD] Gradients are well-aligned")
                report.append("  -> Losses are cooperating towards similar goals")
            else:
                report.append("[OK] Moderate gradient alignment")

        if summary["conflict_count"] > summary["n_logged_steps"] * 0.1:
            report.append("")
            report.append(f"[WARNING] {summary['conflict_count']}/{summary['n_logged_steps']} "
                         f"({100*summary['conflict_count']/summary['n_logged_steps']:.1f}%) "
                         f"steps had gradient conflicts (cos_sim < -0.1)")

        report.append("=" * 60)

        return "\n".join(report)


# ===========================================================================
# Quick logging function for existing training code
# ===========================================================================

def log_gradient_analysis(
    model,
    loss_diffusion: torch.Tensor,
    loss_anatomy: torch.Tensor,
    lambda_anatomy: float,
    step: int,
    retain_graph: bool = True,
    logger_fn=None,
):
    """
    Quick one-shot function to log gradient analysis.

    Insert this into your training loop:

        if step % 100 == 0:
            log_gradient_analysis(
                model=model,
                loss_diffusion=loss_diffusion,
                loss_anatomy=loss_anatomy,
                lambda_anatomy=args.lambda_anatomy,
                step=step,
                retain_graph=True,
            )

    Args:
        model: The model being trained
        loss_diffusion: Diffusion loss tensor
        loss_anatomy: Anatomy loss tensor (raw, before lambda)
        lambda_anatomy: Weight for anatomy loss
        step: Current training step
        retain_graph: Keep computation graph for actual backward
        logger_fn: Optional custom logging function (default: print)
    """
    if logger_fn is None:
        logger_fn = print

    params = get_lora_parameters(model)
    if len(params) == 0:
        logger_fn(f"[GradAnalysis] Step {step}: No LoRA params found!")
        return

    metrics = compute_separate_gradient_norms(
        model,
        loss_diffusion,
        loss_anatomy,
        lambda_anatomy,
        params,
        retain_graph,
    )

    logger_fn(
        f"[GradAnalysis] Step {step}: "
        f"||g_diff||={metrics['grad_norm_diffusion']:.4f}, "
        f"||g_anat||={metrics['grad_norm_anatomy']:.4f}, "
        f"ratio={metrics['grad_norm_ratio']:.4f}, "
        f"cos_sim={metrics['cosine_similarity']:.4f}"
    )

    # Alert on potential issues
    if metrics["cosine_similarity"] < -0.1:
        logger_fn(f"[GradAnalysis] Step {step}: WARNING - Gradient conflict (cos_sim={metrics['cosine_similarity']:.4f})")
    if metrics["grad_norm_ratio"] > 5.0:
        logger_fn(f"[GradAnalysis] Step {step}: WARNING - Anatomy gradients dominating (ratio={metrics['grad_norm_ratio']:.4f})")


# ===========================================================================
# Integration snippet for train_anatomy.py
# ===========================================================================

INTEGRATION_SNIPPET = """
# ===========================================================================
# GRADIENT MONITORING INTEGRATION
# Add this to your train_anatomy.py training loop
# ===========================================================================

# At the top of your file:
from diagnostics.gradient_monitor import GradientMonitor, log_gradient_analysis

# Before training loop:
gradient_monitor = GradientMonitor(model, log_every=100)

# Inside your training loop, AFTER loss computation but BEFORE backward:
#   loss_dict = training_losses_with_anatomy(...)
#   loss_total = loss_dict["loss_total"]
#   loss_diffusion = loss_dict["loss_diffusion"]
#   loss_anatomy = loss_dict["loss_anatomy"]
#
#   # Option A: Quick logging (minimal overhead)
#   if global_step % 100 == 0:
#       log_gradient_analysis(
#           model=model,
#           loss_diffusion=loss_diffusion,
#           loss_anatomy=loss_anatomy,
#           lambda_anatomy=args.lambda_anatomy,
#           step=global_step,
#           retain_graph=True,
#       )
#
#   loss_total.backward()

# After training:
print(gradient_monitor.diagnose())
"""


if __name__ == "__main__":
    # Print usage information
    print(__doc__)
    print("\n" + "=" * 60)
    print("INTEGRATION SNIPPET:")
    print("=" * 60)
    print(INTEGRATION_SNIPPET)
