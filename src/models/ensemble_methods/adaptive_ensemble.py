"""
Adaptive ensemble with input-dependent gating networks.
Dynamically weights models based on input characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from .uncertainty_utils import ModelUncertainty


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that dynamically weights models based on input characteristics.
    
    The gating network learns to predict which models are most reliable for each input.
    
    Mathematical formulation:
    f_adaptive(x) = Σᵢ gᵢ(x) * fᵢ(x)
    
    Where gᵢ(x) is the input-dependent gating weight for model i.
    """
    
    def __init__(self, 
                 models: List[nn.Module],
                 feature_extractor_dim: int = 512,
                 gating_hidden_dim: int = 64):
        super(AdaptiveEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        # Lightweight feature extractor for gating
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  # Reduce spatial dimensions
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, feature_extractor_dim),  # Assuming RGB input
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Gating network
        self.gating_network = nn.Sequential(
            nn.Linear(feature_extractor_dim, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, self.num_models),
            nn.Softmax(dim=1)  # Ensure weights sum to 1
        )
        
        # Task-specific gating (optional)
        self.task_specific_gating = True
        if self.task_specific_gating:
            self.food_gating = nn.Sequential(
                nn.Linear(feature_extractor_dim, gating_hidden_dim),
                nn.ReLU(),
                nn.Linear(gating_hidden_dim, self.num_models),
                nn.Softmax(dim=1)
            )
            
            self.cuisine_gating = nn.Sequential(
                nn.Linear(feature_extractor_dim, gating_hidden_dim),
                nn.ReLU(),
                nn.Linear(gating_hidden_dim, self.num_models),
                nn.Softmax(dim=1)
            )
            
            self.nutrition_gating = nn.Sequential(
                nn.Linear(feature_extractor_dim, gating_hidden_dim),
                nn.ReLU(),
                nn.Linear(gating_hidden_dim, self.num_models),
                nn.Softmax(dim=1)
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through adaptive ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (food_logits, cuisine_logits, nutrition_values, gating_weights)
        """
        # Extract features for gating
        gating_features = self.feature_extractor(x)
        
        # Calculate gating weights
        if self.task_specific_gating:
            food_weights = self.food_gating(gating_features)
            cuisine_weights = self.cuisine_gating(gating_features)
            nutrition_weights = self.nutrition_gating(gating_features)
        else:
            shared_weights = self.gating_network(gating_features)
            food_weights = cuisine_weights = nutrition_weights = shared_weights
        
        # Collect model predictions
        food_predictions = []
        cuisine_predictions = []
        nutrition_predictions = []
        
        for model in self.models:
            food_logits, cuisine_logits, nutrition_values = model(x)
            food_predictions.append(food_logits)
            cuisine_predictions.append(cuisine_logits)
            nutrition_predictions.append(nutrition_values)
        
        # Adaptive weighted combination
        ensemble_food_logits = sum(
            w.unsqueeze(-1) * pred 
            for w, pred in zip(food_weights.split(1, dim=1), food_predictions)
        )
        
        ensemble_cuisine_logits = sum(
            w.unsqueeze(-1) * pred 
            for w, pred in zip(cuisine_weights.split(1, dim=1), cuisine_predictions)
        )
        
        ensemble_nutrition = sum(
            w.unsqueeze(-1) * pred 
            for w, pred in zip(nutrition_weights.split(1, dim=1), nutrition_predictions)
        )
        
        # Metadata with gating information
        metadata = {
            'food_gating_weights': food_weights.detach().cpu().numpy(),
            'cuisine_gating_weights': cuisine_weights.detach().cpu().numpy(),
            'nutrition_gating_weights': nutrition_weights.detach().cpu().numpy(),
            'gating_entropy': ModelUncertainty.predictive_entropy(food_weights).mean().item()
        }
        
        return ensemble_food_logits, ensemble_cuisine_logits, ensemble_nutrition, metadata