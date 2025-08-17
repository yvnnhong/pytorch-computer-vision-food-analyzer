"""
Custom CNN architectures for food image analysis.

This package provides domain-specific CNN architectures optimized for food recognition:
- Food texture, color, and structure analysis blocks
- Efficient mobile-optimized architectures
- Multi-task learning for food/cuisine classification and nutrition regression
"""

# Import building blocks
from .food_blocks import (
    FoodTextureBlock,
    ColorAnalysisBlock, 
    FoodStructureBlock,
    FoodCNNBlock
)

# Import efficient convolutions
from .efficient_conv import (
    DepthwiseSeparableConv,
    EfficientFoodBlock
)

# Import main architectures
from .food_architectures import FoodNet

# Import mobile architecture
from .mobile_net import MobileFoodNet

# Import factory functions
from .model_factory import (
    create_food_cnn,
    calculate_model_efficiency
)

# Define exports
__all__ = [
    # Building blocks
    'FoodTextureBlock',
    'ColorAnalysisBlock',
    'FoodStructureBlock', 
    'FoodCNNBlock',
    
    # Efficient convolutions
    'DepthwiseSeparableConv',
    'EfficientFoodBlock',
    
    # Main architectures
    'FoodNet',
    'MobileFoodNet',
    
    # Factory functions
    'create_food_cnn',
    'calculate_model_efficiency'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Yvonne Hong"