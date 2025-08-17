"""
Advanced loss functions for multi-task learning with mathematical rigor.

Mathematical Foundation:
- Multi-task loss with learned uncertainty weighting (Kendall & Gal, 2017)
- Focal Loss for addressing class imbalance (Lin et al., 2017)  
- Task correlation regularization for multi-task optimization
- Gradient harmonization for balanced task learning
- Contrastive learning for representation optimization
"""

# Import base classes
from .base_losses import BaseLoss, FocalLoss

# Import uncertainty-weighted losses
from .uncertainty_losses import (
    UncertaintyWeightedLoss,
    GradientHarmonizationLoss
)

# Import contrastive losses
from .contrastive_losses import (
    ContrastiveLoss,
    InfoNCELoss
)

# Import adaptive losses
from .adaptive_losses import (
    AdaptiveMultiTaskLoss,
    TaskCorrelationRegularizer
)

# Import factory and analysis
from .loss_factory import (
    create_loss_function,
    analyze_loss_landscape
)

# Define exports
__all__ = [
    # Base classes
    'BaseLoss',
    'FocalLoss',
    
    # Uncertainty-weighted losses
    'UncertaintyWeightedLoss',
    'GradientHarmonizationLoss',
    
    # Contrastive losses
    'ContrastiveLoss',
    'InfoNCELoss',
    
    # Adaptive losses
    'AdaptiveMultiTaskLoss',
    'TaskCorrelationRegularizer',
    
    # Factory and analysis
    'create_loss_function',
    'analyze_loss_landscape'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Yvonne Hong"

# Quick access function
def create_multitask_loss(loss_type: str = 'adaptive', **kwargs):
    """Quick multi-task loss creation function."""
    return create_loss_function(loss_type, **kwargs)