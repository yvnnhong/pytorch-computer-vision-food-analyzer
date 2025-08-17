"""
Uncertainty-weighted multi-task loss implementations.
Based on Kendall & Gal (2017) mathematical framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List
from .base_losses import BaseLoss, FocalLoss


class UncertaintyWeightedLoss(BaseLoss):
    """
    Multi-task loss with learnable uncertainty weighting.
    
    Mathematical foundation (Kendall & Gal, 2017):
    L_total = Σ_i [1/(2σ_i²) * L_i + log(σ_i)]
    
    where:
    - L_i is the loss for task i
    - σ_i is the learned uncertainty (noise parameter) for task i
    """
    
    def __init__(self, num_tasks: int = 3, init_uncertainty: float = 1.0):
        super(UncertaintyWeightedLoss, self).__init__()
        
        # Initialize learnable uncertainty parameters
        # log_sigma to ensure σ > 0 through exponential
        self.log_uncertainty = nn.Parameter(
            torch.full((num_tasks,), math.log(init_uncertainty))
        )
        
        # Individual loss functions
        self.food_loss_fn = FocalLoss(gamma=2.0)  # Use focal loss for food classification
        self.cuisine_loss_fn = nn.CrossEntropyLoss()
        self.nutrition_loss_fn = nn.MSELoss()
        
        self.task_names = ['food', 'cuisine', 'nutrition']
        
    def forward(self, predictions: Tuple[torch.Tensor, ...], 
                targets: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate uncertainty-weighted multi-task loss.
        
        Args:
            predictions: Tuple of (food_logits, cuisine_logits, nutrition_values)
            targets: Tuple of (food_labels, cuisine_labels, nutrition_targets)
            
        Returns:
            tuple: (total_loss, loss_breakdown_dict)
        """
        food_logits, cuisine_logits, nutrition_values = predictions
        food_labels, cuisine_labels, nutrition_targets = targets
        
        # Calculate individual losses
        food_loss = self.food_loss_fn(food_logits, food_labels)
        cuisine_loss = self.cuisine_loss_fn(cuisine_logits, cuisine_labels)
        nutrition_loss = self.nutrition_loss_fn(nutrition_values, nutrition_targets)
        
        losses = torch.stack([food_loss, cuisine_loss, nutrition_loss])
        
        # Get uncertainty values (σ = exp(log_σ))
        uncertainty = torch.exp(self.log_uncertainty)
        
        # Calculate uncertainty-weighted loss
        # L = 1/(2σ²) * L_task + log(σ)
        weighted_losses = losses / (2 * uncertainty ** 2) + torch.log(uncertainty)
        total_loss = weighted_losses.sum()
        
        # Create detailed loss breakdown
        loss_breakdown = {
            'total_loss': total_loss.item(),
            'food_loss': food_loss.item(),
            'cuisine_loss': cuisine_loss.item(),
            'nutrition_loss': nutrition_loss.item(),
            'food_uncertainty': uncertainty[0].item(),
            'cuisine_uncertainty': uncertainty[1].item(),
            'nutrition_uncertainty': uncertainty[2].item(),
            'food_weighted': weighted_losses[0].item(),
            'cuisine_weighted': weighted_losses[1].item(),
            'nutrition_weighted': weighted_losses[2].item()
        }
        
        self.loss_history.append(total_loss.item())
        return total_loss, loss_breakdown
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task importance weights (1/σ²)"""
        uncertainty = torch.exp(self.log_uncertainty)
        weights = 1.0 / (uncertainty ** 2)
        
        return {
            task: weight.item() 
            for task, weight in zip(self.task_names, weights)
        }


class GradientHarmonizationLoss(BaseLoss):
    """
    Multi-task loss with gradient harmonization for balanced learning.
    
    Addresses the problem of conflicting gradients in multi-task learning
    by harmonizing gradient directions across tasks.
    """
    
    def __init__(self, alpha: float = 0.1):
        super(GradientHarmonizationLoss, self).__init__()
        self.alpha = alpha  # Harmonization strength
        
        self.food_loss_fn = nn.CrossEntropyLoss()
        self.cuisine_loss_fn = nn.CrossEntropyLoss()
        self.nutrition_loss_fn = nn.MSELoss()
        
        self.gradient_history = []
    
    def forward(self, predictions: Tuple[torch.Tensor, ...], 
                targets: Tuple[torch.Tensor, ...], 
                model_parameters: List[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate loss with gradient harmonization.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            model_parameters: Model parameters for gradient calculation
            
        Returns:
            tuple: (harmonized_loss, loss_breakdown)
        """
        food_logits, cuisine_logits, nutrition_values = predictions
        food_labels, cuisine_labels, nutrition_targets = targets
        
        # Calculate individual losses
        food_loss = self.food_loss_fn(food_logits, food_labels)
        cuisine_loss = self.cuisine_loss_fn(cuisine_logits, cuisine_labels)
        nutrition_loss = self.nutrition_loss_fn(nutrition_values, nutrition_targets)
        
        # Base multi-task loss
        base_loss = food_loss + 0.5 * cuisine_loss + nutrition_loss
        
        # Add gradient harmonization if model parameters provided
        harmonization_penalty = 0.0
        if model_parameters is not None:
            harmonization_penalty = self._calculate_gradient_harmonization(
                [food_loss, cuisine_loss, nutrition_loss], model_parameters
            )
        
        total_loss = base_loss + self.alpha * harmonization_penalty
        
        loss_breakdown = {
            'total_loss': total_loss.item(),
            'base_loss': base_loss.item(),
            'food_loss': food_loss.item(),
            'cuisine_loss': cuisine_loss.item(),
            'nutrition_loss': nutrition_loss.item(),
            'harmonization_penalty': harmonization_penalty
        }
        
        return total_loss, loss_breakdown
    
    def _calculate_gradient_harmonization(self, task_losses: List[torch.Tensor], 
                                        parameters: List[torch.Tensor]) -> float:
        """Calculate gradient harmonization penalty"""
        
        task_gradients = []
        
        for loss in task_losses:
            # Calculate gradients for this task
            grad = torch.autograd.grad(loss, parameters, retain_graph=True, create_graph=True)
            
            # Flatten and concatenate gradients
            flat_grad = torch.cat([g.flatten() for g in grad])
            task_gradients.append(flat_grad)
        
        # Calculate pairwise gradient conflicts
        conflicts = 0.0
        num_pairs = 0
        
        for i in range(len(task_gradients)):
            for j in range(i + 1, len(task_gradients)):
                # Cosine similarity between gradient vectors
                cos_sim = F.cosine_similarity(
                    task_gradients[i].unsqueeze(0), 
                    task_gradients[j].unsqueeze(0)
                )
                
                # Penalty for conflicting gradients (negative cosine similarity)
                if cos_sim < 0:
                    conflicts += -cos_sim
                    num_pairs += 1
        
        return conflicts / max(num_pairs, 1)