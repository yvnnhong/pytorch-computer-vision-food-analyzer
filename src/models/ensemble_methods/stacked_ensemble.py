"""
Stacked ensemble implementation with meta-learner.
Uses Level-1 meta-learner to optimally combine Level-0 base models.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class StackedEnsemble(nn.Module):
    """
    Stacked ensemble with meta-learner that learns optimal combination.
    
    Architecture:
    Level 0: Base models (ResNet, Custom CNN, etc.)
    Level 1: Meta-learner that combines base model predictions
    
    Mathematical formulation:
    f_stack(x) = g(f₁(x), f₂(x), ..., fₙ(x))
    
    Where g is the meta-learner neural network.
    """
    
    def __init__(self, 
                 base_models: List[nn.Module],
                 meta_learner_hidden_dim: int = 128,
                 num_food_classes: int = 101,
                 num_cuisine_classes: int = 10,
                 nutrition_dim: int = 4):
        super(StackedEnsemble, self).__init__()
        
        self.base_models = nn.ModuleList(base_models)
        self.num_models = len(base_models)
        
        # Calculate input dimension for meta-learner
        meta_input_dim = self.num_models * (num_food_classes + num_cuisine_classes + nutrition_dim)
        
        # Meta-learner networks for each task
        self.food_meta_learner = nn.Sequential(
            nn.Linear(meta_input_dim, meta_learner_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(meta_learner_hidden_dim, meta_learner_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(meta_learner_hidden_dim // 2, num_food_classes)
        )
        
        self.cuisine_meta_learner = nn.Sequential(
            nn.Linear(meta_input_dim, meta_learner_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(meta_learner_hidden_dim, meta_learner_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(meta_learner_hidden_dim // 2, num_cuisine_classes)
        )
        
        self.nutrition_meta_learner = nn.Sequential(
            nn.Linear(meta_input_dim, meta_learner_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(meta_learner_hidden_dim, meta_learner_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(meta_learner_hidden_dim // 2, nutrition_dim),
            nn.ReLU()  # Ensure positive nutrition values
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through stacked ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (food_logits, cuisine_logits, nutrition_values)
        """
        # Collect base model predictions
        base_predictions = []
        
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                food_logits, cuisine_logits, nutrition_values = model(x)
                
                # Concatenate all outputs from this model
                model_output = torch.cat([
                    food_logits, cuisine_logits, nutrition_values
                ], dim=1)
                base_predictions.append(model_output)
        
        # Concatenate all base model predictions
        meta_input = torch.cat(base_predictions, dim=1)
        
        # Meta-learner predictions
        ensemble_food_logits = self.food_meta_learner(meta_input)
        ensemble_cuisine_logits = self.cuisine_meta_learner(meta_input)
        ensemble_nutrition = self.nutrition_meta_learner(meta_input)
        
        return ensemble_food_logits, ensemble_cuisine_logits, ensemble_nutrition