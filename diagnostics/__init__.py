"""
Diagnostic Tools for OmniGen CXR Fine-tuning with Anatomy-Aware Loss.

This package provides tools to diagnose:
1. Whether anatomy actually improved despite worse FID
2. Whether gradient conflicts are causing issues
3. Where visual differences/artifacts are located

Tools:
    - eval_anatomy_dice.py: Objective Dice score comparison
    - gradient_monitor.py: Gradient norm monitoring hooks
    - plot_diff_heatmap.py: Visual difference heatmap generator
"""

from .gradient_monitor import (
    GradientMonitor,
    log_gradient_analysis,
    compute_separate_gradient_norms,
    get_lora_parameters,
)
