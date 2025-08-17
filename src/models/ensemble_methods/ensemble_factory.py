"""
Factory functions and configuration for ensemble creation.
Provides unified interface for creating different ensemble types.
"""

import torch.nn as nn
from typing import List, Optional
from dataclasses import dataclass
from .weighted_ensemble import WeightedEnsemble
from .stacked_ensemble import StackedEnsemble
from .adaptive_ensemble import AdaptiveEnsemble
from .mixture_experts import MixtureOfExperts


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models."""
    model_weights: List[float]
    combination_method: str  # 'weighted', 'voting', 'stacking', 'adaptive'
    confidence_threshold: float = 0.8
    diversity_weight: float = 0.1
    temperature_scaling: bool = True
    uncertainty_estimation: bool = True


class EnsembleFactory:
    """
    Factory class for creating different types of ensemble models.
    """
    
    @staticmethod
    def create_ensemble(ensemble_type: str, 
                       models: List[nn.Module], 
                       config: Optional[EnsembleConfig] = None,
                       **kwargs) -> nn.Module:
        """
        Create ensemble model of specified type.
        
        Args:
            ensemble_type: Type of ensemble ('weighted', 'stacked', 'adaptive', 'mixture')
            models: List of base models
            config: Ensemble configuration
            **kwargs: Additional arguments
            
        Returns:
            nn.Module: Ensemble model
        """
        
        if ensemble_type == 'weighted':
            weights = config.model_weights if config else None
            return WeightedEnsemble(
                models=models,
                weights=weights,
                learnable_weights=kwargs.get('learnable_weights', False),
                temperature_scaling=config.temperature_scaling if config else True
            )
        
        elif ensemble_type == 'stacked':
            return StackedEnsemble(
                base_models=models,
                meta_learner_hidden_dim=kwargs.get('meta_hidden_dim', 128),
                **kwargs
            )
        
        elif ensemble_type == 'adaptive':
            return AdaptiveEnsemble(
                models=models,
                feature_extractor_dim=kwargs.get('feature_dim', 512),
                gating_hidden_dim=kwargs.get('gating_dim', 64)
            )
        
        elif ensemble_type == 'mixture':
            specializations = kwargs.get('specializations', [f'expert_{i}' for i in range(len(models))])
            return MixtureOfExperts(
                expert_models=models,
                expert_specializations=specializations,
                gating_dim=kwargs.get('gating_dim', 256)
            )
        
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    @staticmethod
    def create_diverse_ensemble(num_food_classes: int = 101,
                              num_cuisine_classes: int = 10,
                              ensemble_type: str = 'weighted') -> nn.Module:
        """
        Create ensemble with diverse architectures for maximum performance.
        
        Args:
            num_food_classes: Number of food classes
            num_cuisine_classes: Number of cuisine classes
            ensemble_type: Type of ensemble to create
            
        Returns:
            nn.Module: Diverse ensemble model
        """
        # Import here to avoid circular imports
        from .food_classifier import MultiTaskFoodModel
        from .resnet_multitask import AdvancedResNetMultiTask
        from .custom_cnn import FoodNet, MobileFoodNet
        
        # Create diverse base models
        models = [
            # Basic multi-task model
            MultiTaskFoodModel(
                num_food_classes=num_food_classes,
                num_cuisine_classes=num_cuisine_classes
            ),
            
            # Advanced ResNet with attention
            AdvancedResNetMultiTask(
                num_food_classes=num_food_classes,
                num_cuisine_classes=num_cuisine_classes,
                backbone='resnet50'
            ),
            
            # Custom food-specific CNN
            FoodNet(
                num_food_classes=num_food_classes,
                num_cuisine_classes=num_cuisine_classes,
                architecture='standard'
            ),
            
            # Efficient mobile variant
            MobileFoodNet(
                num_food_classes=num_food_classes,
                num_cuisine_classes=num_cuisine_classes,
                width_multiplier=1.0
            )
        ]
        
        # Create ensemble
        if ensemble_type == 'mixture':
            specializations = ['general', 'attention_based', 'food_specific', 'mobile_optimized']
            return EnsembleFactory.create_ensemble(
                ensemble_type, models, specializations=specializations
            )
        else:
            return EnsembleFactory.create_ensemble(ensemble_type, models)