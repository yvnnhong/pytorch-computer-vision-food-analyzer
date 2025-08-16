"""
Attention visualization and interpretability tools.
Provides utilities for visualizing and understanding attention mechanisms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple
from .channel_spatial import ChannelAttention, SpatialAttention, CBAM


def visualize_attention_maps(attention_module: nn.Module, 
                           input_tensor: torch.Tensor,
                           save_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Visualize attention maps for interpretability.
    
    Args:
        attention_module: Attention module to visualize
        input_tensor: Input tensor to compute attention for
        save_path: Optional path to save visualization
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary of attention maps
    """
    attention_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                # For spatial attention maps
                attention_maps[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in attention_module.named_modules():
        if isinstance(module, (ChannelAttention, SpatialAttention, CBAM)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = attention_module(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps


def extract_attention_weights(attention_module: nn.Module, 
                            input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Extract attention weights from various attention mechanisms."""
    
    attention_weights = {}
    
    # Handle different attention types
    if hasattr(attention_module, 'get_attention_weights'):
        weights = attention_module.get_attention_weights(input_tensor)
        if weights is not None:
            attention_weights['attention_weights'] = weights
    
    # Handle food-specific attention
    if hasattr(attention_module, 'get_attention_components'):
        components = attention_module.get_attention_components(input_tensor)
        attention_weights.update(components)
    
    return attention_weights


def create_attention_heatmap(attention_weights: torch.Tensor, 
                           original_image: Optional[torch.Tensor] = None) -> np.ndarray:
    """Create heatmap visualization from attention weights."""
    
    # Convert to numpy and handle different tensor shapes
    if attention_weights.dim() == 4:
        # Spatial attention (B, 1, H, W)
        heatmap = attention_weights[0, 0].cpu().numpy()
    elif attention_weights.dim() == 3:
        # Multi-head attention weights (B, H, W)
        heatmap = attention_weights[0].cpu().numpy()
    elif attention_weights.dim() == 2:
        # Channel attention or other 2D weights
        heatmap = attention_weights[0].cpu().numpy()
    else:
        # Flatten to 2D
        heatmap = attention_weights.flatten().cpu().numpy()
        size = int(np.sqrt(len(heatmap)))
        heatmap = heatmap[:size*size].reshape(size, size)
    
    # Normalize to 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap


def analyze_attention_patterns(attention_weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """Analyze attention patterns for insights."""
    
    analysis = {}
    
    for name, weights in attention_weights.items():
        stats = {
            'mean': float(weights.mean()),
            'std': float(weights.std()),
            'max': float(weights.max()),
            'min': float(weights.min()),
            'sparsity': float((weights < 0.1).sum()) / weights.numel(),
            'concentration': float((weights > 0.8).sum()) / weights.numel()
        }
        
        analysis[name] = stats
    
    return analysis


def compare_attention_across_models(models: List[nn.Module], 
                                  input_tensor: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compare attention patterns across different models."""
    
    model_attention = {}
    
    for i, model in enumerate(models):
        model_name = f"model_{i}_{model.__class__.__name__}"
        
        try:
            attention_maps = visualize_attention_maps(model, input_tensor)
            attention_weights = extract_attention_weights(model, input_tensor)
            
            model_attention[model_name] = {
                **attention_maps,
                **attention_weights
            }
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            model_attention[model_name] = {}
    
    return model_attention


def plot_attention_summary(attention_data: Dict[str, torch.Tensor], 
                         save_path: Optional[str] = None) -> None:
    """Create summary plot of attention mechanisms."""
    
    try:
        import matplotlib.pyplot as plt
        
        num_attention = len(attention_data)
        if num_attention == 0:
            return
        
        fig, axes = plt.subplots(1, min(num_attention, 4), figsize=(16, 4))
        if num_attention == 1:
            axes = [axes]
        
        for idx, (name, weights) in enumerate(list(attention_data.items())[:4]):
            heatmap = create_attention_heatmap(weights)
            
            axes[idx].imshow(heatmap, cmap='hot', interpolation='nearest')
            axes[idx].set_title(name.replace('_', ' ').title())
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")


def attention_statistics_report(attention_module: nn.Module,
                              input_tensor: torch.Tensor) -> str:
    """Generate comprehensive attention statistics report."""
    
    # Extract attention information
    attention_weights = extract_attention_weights(attention_module, input_tensor)
    attention_analysis = analyze_attention_patterns(attention_weights)
    
    # Generate report
    report = []
    report.append(f"ATTENTION ANALYSIS REPORT")
    report.append(f"=" * 50)
    report.append(f"Module: {attention_module.__class__.__name__}")
    report.append(f"Input Shape: {tuple(input_tensor.shape)}")
    report.append(f"Number of Attention Components: {len(attention_weights)}")
    report.append("")
    
    for component, stats in attention_analysis.items():
        report.append(f"{component.upper()}:")
        report.append(f"  Mean Activation: {stats['mean']:.4f}")
        report.append(f"  Standard Deviation: {stats['std']:.4f}")
        report.append(f"  Sparsity (< 0.1): {stats['sparsity']:.1%}")
        report.append(f"  Concentration (> 0.8): {stats['concentration']:.1%}")
        report.append("")
    
    return "\n".join(report)