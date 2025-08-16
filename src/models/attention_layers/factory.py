"""
Factory functions and utilities for attention mechanisms.
Provides easy creation and management of attention layers.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from .channel_spatial import ChannelAttention, SpatialAttention, CBAM, SqueezeExcitation
from .self_attention import MultiHeadSelfAttention
from .cross_modal import CrossModalAttention, MultiTaskAttentionFusion
from .food_specific import FoodSpecificAttention


def create_attention_layer(attention_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create different types of attention layers.
    
    Args:
        attention_type: Type of attention ('channel', 'spatial', 'cbam', 'se', 'mhsa', 'food_specific')
        **kwargs: Arguments specific to each attention type
        
    Returns:
        nn.Module: Attention layer
    """
    
    attention_registry = {
        'channel': ChannelAttention,
        'spatial': SpatialAttention,
        'cbam': CBAM,
        'se': SqueezeExcitation,
        'mhsa': MultiHeadSelfAttention,
        'cross_modal': CrossModalAttention,
        'food_specific': FoodSpecificAttention,
        'multitask_fusion': MultiTaskAttentionFusion
    }
    
    if attention_type not in attention_registry:
        raise ValueError(f"Unknown attention type: {attention_type}. Available: {list(attention_registry.keys())}")
    
    return attention_registry[attention_type](**kwargs)


def get_attention_info(attention_module: nn.Module) -> Dict[str, Any]:
    """Get comprehensive info about attention module."""
    
    # Basic info
    info = {
        'type': attention_module.__class__.__name__,
        'parameters': sum(p.numel() for p in attention_module.parameters()),
        'trainable_parameters': sum(p.numel() for p in attention_module.parameters() if p.requires_grad),
        'memory_mb': sum(p.numel() * 4 for p in attention_module.parameters()) / (1024 * 1024)
    }
    
    # Add module-specific info
    if hasattr(attention_module, 'get_info'):
        info.update(attention_module.get_info())
    
    return info


def compare_attention_mechanisms(modules: List[nn.Module], 
                               input_shape: tuple = (2, 256, 32, 32)) -> Dict[str, Dict[str, Any]]:
    """Compare multiple attention mechanisms."""
    
    results = {}
    test_input = torch.randn(input_shape)
    
    for module in modules:
        module_name = module.__class__.__name__
        
        try:
            # Test forward pass
            with torch.no_grad():
                if module_name == 'MultiHeadSelfAttention':
                    # Reshape for transformer attention
                    batch_size, channels, height, width = input_shape
                    seq_input = test_input.view(batch_size, channels, -1).transpose(1, 2)
                    output = module(seq_input)
                else:
                    output = module(test_input)
            
            # Get module info
            info = get_attention_info(module)
            info['output_shape'] = tuple(output.shape)
            info['forward_pass_success'] = True
            
        except Exception as e:
            info = get_attention_info(module)
            info['forward_pass_success'] = False
            info['error'] = str(e)
        
        results[module_name] = info
    
    return results


def create_attention_suite(in_channels: int = 256) -> Dict[str, nn.Module]:
    """Create a complete suite of attention mechanisms for testing."""
    
    return {
        'channel': create_attention_layer('channel', in_channels=in_channels),
        'spatial': create_attention_layer('spatial'),
        'cbam': create_attention_layer('cbam', in_channels=in_channels),
        'squeeze_excitation': create_attention_layer('se', in_channels=in_channels),
        'food_specific': create_attention_layer('food_specific', in_channels=in_channels),
        'multi_head_self': create_attention_layer('mhsa', embed_dim=in_channels),
        'cross_modal': create_attention_layer('cross_modal', embed_dim=in_channels, cross_dim=in_channels),
    }


def benchmark_attention_speed(attention_module: nn.Module, 
                            input_tensor: torch.Tensor,
                            num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark attention mechanism speed."""
    import time
    
    attention_module.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = attention_module(input_tensor)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = attention_module(input_tensor)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    import numpy as np
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times))
    }