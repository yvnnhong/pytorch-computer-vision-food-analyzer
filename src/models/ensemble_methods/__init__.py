"""
Ensemble methods for multi-task food classification.

Mathematical Foundation:
- Weighted Ensemble: f_ensemble(x) = Σᵢ wᵢ * fᵢ(x) where Σwᵢ = 1
- Stacked Ensemble: f_stack(x) = g(f₁(x), f₂(x), ..., fₙ(x))
- Bayesian Model Averaging: P(y|x,D) = Σᵢ P(y|x,Mᵢ) * P(Mᵢ|D)
- Mixture of Experts: P(y|x) = Σᵢ gᵢ(x) * fᵢ(x) where gᵢ is gating function

Key Features:
- Dynamic model weighting based on confidence
- Architecture diversity for robustness
- Task-specific ensemble strategies
- Real-time inference optimization
- Uncertainty quantification
"""

# Import ensemble implementations
from .weighted_ensemble import WeightedEnsemble
from .stacked_ensemble import StackedEnsemble
from .adaptive_ensemble import AdaptiveEnsemble
from .mixture_experts import MixtureOfExperts

# Import utilities
from .uncertainty_utils import ModelUncertainty, evaluate_ensemble_diversity

# Import factory and configuration
from .ensemble_factory import EnsembleFactory, EnsembleConfig

# Define exports
__all__ = [
    # Ensemble implementations
    'WeightedEnsemble',
    'StackedEnsemble', 
    'AdaptiveEnsemble',
    'MixtureOfExperts',
    
    # Utilities
    'ModelUncertainty',
    'evaluate_ensemble_diversity',
    
    # Factory and configuration
    'EnsembleFactory',
    'EnsembleConfig'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Yvonne Hong"

# Quick access functions
def create_ensemble(ensemble_type: str, models: list, **kwargs):
    """Quick ensemble creation function."""
    return EnsembleFactory.create_ensemble(ensemble_type, models, **kwargs)

def create_diverse_ensemble(**kwargs):
    """Quick diverse ensemble creation function."""
    return EnsembleFactory.create_diverse_ensemble(**kwargs)