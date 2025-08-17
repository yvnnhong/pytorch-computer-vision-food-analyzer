"""
Weighted ensemble implementation for multi-task learning.
Combines models with learnable or fixed weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from .uncertainty_utils import ModelUncertainty


class WeightedEnsemble(nn.Module):
    """
    Weighted ensemble that combines models with learnable or fixed weights.
    
    Mathematical formulation:
    f_ensemble(x) = Σᵢ wᵢ * fᵢ(x)
    
    Where wᵢ are weights (learnable or fixed) and fᵢ are individual models.
    """
    
    def __init__(self, 
                 models: List[nn.Module],
                 weights: Optional[List[float]] = None,
                 learnable_weights: bool = False,
                 temperature_scaling: bool = True):
        super(WeightedEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.temperature_scaling = temperature_scaling
        
        # Initialize weights
        if weights is None:
            weights = [1.0 / self.num_models] * self.num_models
        
        if learnable_weights:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
        # Temperature parameters for calibration
        if temperature_scaling:
            self.temperature_food = nn.Parameter(torch.ones(1))
            self.temperature_cuisine = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through weighted ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (food_logits, cuisine_logits, nutrition_values, metadata)
        """
        # Normalize weights
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Collect predictions from all models
        food_predictions = []
        cuisine_predictions = []
        nutrition_predictions = []
        model_confidences = []
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'forward') and callable(model.forward):
                    food_logits, cuisine_logits, nutrition_values = model(x)
                else:
                    # Handle different model interfaces
                    outputs = model(x)
                    if isinstance(outputs, tuple):
                        food_logits, cuisine_logits, nutrition_values = outputs
                    else:
                        raise ValueError(f"Unexpected model output format from model {i}")
                
                food_predictions.append(food_logits)
                cuisine_predictions.append(cuisine_logits)
                nutrition_predictions.append(nutrition_values)
                
                # Calculate model confidence (max probability)
                food_probs = F.softmax(food_logits, dim=1)
                max_confidence = torch.max(food_probs, dim=1)[0].mean().item()
                model_confidences.append(max_confidence)
        
        # Weighted combination
        ensemble_food_logits = sum(w * pred for w, pred in zip(normalized_weights, food_predictions))
        ensemble_cuisine_logits = sum(w * pred for w, pred in zip(normalized_weights, cuisine_predictions))
        ensemble_nutrition = sum(w * pred for w, pred in zip(normalized_weights, nutrition_predictions))
        
        # Apply temperature scaling
        if self.temperature_scaling:
            ensemble_food_logits = ensemble_food_logits / self.temperature_food
            ensemble_cuisine_logits = ensemble_cuisine_logits / self.temperature_cuisine
        
        # Calculate uncertainty metrics
        food_probs = [F.softmax(pred, dim=1) for pred in food_predictions]
        cuisine_probs = [F.softmax(pred, dim=1) for pred in cuisine_predictions]
        
        food_uncertainty = ModelUncertainty.mutual_information(food_probs)
        cuisine_uncertainty = ModelUncertainty.mutual_information(cuisine_probs)
        nutrition_variance = ModelUncertainty.prediction_variance(nutrition_predictions)
        
        # Metadata
        metadata = {
            'model_weights': normalized_weights.detach().cpu().numpy().tolist(),
            'model_confidences': model_confidences,
            'food_uncertainty': food_uncertainty.mean().item(),
            'cuisine_uncertainty': cuisine_uncertainty.mean().item(),
            'nutrition_variance': nutrition_variance.mean().item(),
            'ensemble_confidence': max(model_confidences),
            'prediction_diversity': float(torch.std(torch.tensor(model_confidences)).item())
        }
        
        return ensemble_food_logits, ensemble_cuisine_logits, ensemble_nutrition, metadata