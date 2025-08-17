"""
Factory functions and utilities for creating food CNN models.
Provides model creation, efficiency analysis, and benchmarking.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any
from .food_architectures import FoodNet
from .mobile_net import MobileFoodNet


def create_food_cnn(architecture: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to create different FoodNet variants.
    
    Args:
        architecture: Architecture type ('standard', 'efficient', 'deep', 'mobile')
        **kwargs: Additional arguments for model creation
        
    Returns:
        nn.Module: FoodNet model
    """
    
    if architecture == "mobile":
        return MobileFoodNet(**kwargs)
    else:
        return FoodNet(architecture=architecture, **kwargs)


def calculate_model_efficiency(model: nn.Module) -> Dict[str, Any]:
    """
    Calculate efficiency metrics for food CNN models.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dict with efficiency metrics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    # Count different layer types
    conv_layers = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
    
    # Efficiency score (higher is more efficient)
    efficiency_score = 100 / (1 + math.log10(total_params / 100000))
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'conv_layers': conv_layers,
        'efficiency_score': efficiency_score
    }