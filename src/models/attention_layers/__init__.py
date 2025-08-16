"""
Advanced Attention Mechanisms for Computer Vision and Multi-Task Learning.

This package provides production-ready attention layers optimized for food image analysis
and multi-task learning scenarios.

Mathematical Foundation:
- Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Channel Attention: A_c = σ(W_2 δ(W_1 GAP(F)))  
- Spatial Attention: A_s = σ(conv([AvgPool(F); MaxPool(F)]))
- Cross-Modal Attention: A_cm = softmax(F_1 × W × F_2^T)

Available Modules:
- channel_spatial: Channel & Spatial attention (CBAM family)
- self_attention: Multi-head self-attention (Transformer family)
- cross_modal: Cross-modal & multi-task attention
- food_specific: Domain-adapted food attention
- factory: Factory functions & utilities
- visualization: Attention visualization tools
"""

# Import all attention mechanisms
from .channel_spatial import (
    ChannelAttention, 
    SpatialAttention, 
    CBAM, 
    SqueezeExcitation
)

from .self_attention import (
    MultiHeadSelfAttention
)

from .cross_modal import (
    CrossModalAttention, 
    MultiTaskAttentionFusion
)

from .food_specific import (
    FoodSpecificAttention
)

from .factory import (
    create_attention_layer,
    get_attention_info,
    compare_attention_mechanisms,
    create_attention_suite,
    benchmark_attention_speed
)

from .visualization import (
    visualize_attention_maps,
    extract_attention_weights,
    create_attention_heatmap,
    analyze_attention_patterns,
    attention_statistics_report
)

from .base_attention import (
    BaseAttention,
    AttentionMixin
)

# Define what gets imported with "from attention_layers import *"
__all__ = [
    # Core attention mechanisms
    'ChannelAttention',
    'SpatialAttention', 
    'CBAM',
    'SqueezeExcitation',
    'MultiHeadSelfAttention',
    'CrossModalAttention',
    'MultiTaskAttentionFusion',
    'FoodSpecificAttention',
    
    # Factory functions
    'create_attention_layer',
    'get_attention_info',
    'compare_attention_mechanisms',
    'create_attention_suite',
    'benchmark_attention_speed',
    
    # Visualization tools
    'visualize_attention_maps',
    'extract_attention_weights',
    'create_attention_heatmap',
    'analyze_attention_patterns',
    'attention_statistics_report',
    
    # Base classes
    'BaseAttention',
    'AttentionMixin'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Yvonne Hong"
__description__ = "Advanced attention mechanisms for computer vision and multi-task learning"

# Quick access functions
def list_available_attention():
    """List all available attention mechanisms."""
    return [
        'channel', 'spatial', 'cbam', 'se', 'mhsa', 
        'cross_modal', 'food_specific', 'multitask_fusion'
    ]

def get_attention_family(family: str):
    """Get all attention mechanisms from a specific family."""
    families = {
        'cbam': [ChannelAttention, SpatialAttention, CBAM, SqueezeExcitation],
        'transformer': [MultiHeadSelfAttention],
        'cross_modal': [CrossModalAttention, MultiTaskAttentionFusion],
        'food_specific': [FoodSpecificAttention]
    }
    return families.get(family, [])

# Example usage documentation
def example_usage():
    """Show example usage of attention mechanisms."""
    examples = """
    # Basic Usage
    from attention_layers import ChannelAttention, create_attention_layer
    
    # Create attention mechanism
    channel_att = ChannelAttention(in_channels=256)
    
    # Or use factory
    spatial_att = create_attention_layer('spatial')
    
    # Food-specific attention
    food_att = create_attention_layer('food_specific', 
                                    in_channels=512, 
                                    num_food_classes=101)
    
    # Visualization
    from attention_layers import visualize_attention_maps
    attention_maps = visualize_attention_maps(channel_att, input_tensor)
    
    # Benchmarking
    from attention_layers import benchmark_attention_speed
    speed_stats = benchmark_attention_speed(channel_att, input_tensor)
    """
    return examples