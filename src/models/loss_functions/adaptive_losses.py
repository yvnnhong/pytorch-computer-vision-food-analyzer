"""
Adaptive multi-task loss with dynamic weighting and task correlation.
Combines multiple advanced techniques for optimal multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
from .base_losses import BaseLoss, FocalLoss
from .uncertainty_losses import UncertaintyWeightedLoss


class TaskCorrelationRegularizer(nn.Module):
    """
    Regularizer that encourages or discourages correlation between tasks.
    
    Mathematical formulation:
    R_corr = λ * |corr(f_food, f_cuisine) - target_correlation|²
    """
    
    def __init__(self, target_correlation: float = 0.3, lambda_reg: float = 0.01):
        super(TaskCorrelationRegularizer, self).__init__()
        self.target_correlation = target_correlation
        self.lambda_reg = lambda_reg
    
    def forward(self, food_features: torch.Tensor, 
                cuisine_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate task correlation regularization loss.
        
        Args:
            food_features: Features for food classification (N, D)
            cuisine_features: Features for cuisine classification (N, D)
            
        Returns:
            torch.Tensor: Regularization loss
        """
        # Calculate correlation coefficient
        food_centered = food_features - food_features.mean(dim=0)
        cuisine_centered = cuisine_features - cuisine_features.mean(dim=0)
        
        # Correlation = E[XY] / (σ_X * σ_Y)
        covariance = (food_centered * cuisine_centered).mean(dim=0)
        food_std = food_centered.std(dim=0) + 1e-8
        cuisine_std = cuisine_centered.std(dim=0) + 1e-8
        
        correlation = covariance / (food_std * cuisine_std)
        mean_correlation = correlation.mean()
        
        # Regularization loss
        reg_loss = self.lambda_reg * (mean_correlation - self.target_correlation) ** 2
        
        return reg_loss


class AdaptiveMultiTaskLoss(BaseLoss):
    """
    Advanced multi-task loss that adapts based on training dynamics.
    
    Combines multiple techniques:
    1. Uncertainty weighting
    2. Focal loss for classification
    3. Dynamic task weighting based on learning progress
    4. Task correlation regularization
    """
    
    def __init__(self, 
                 num_food_classes: int = 101,
                 num_cuisine_classes: int = 10,
                 adaptation_rate: float = 0.01):
        super(AdaptiveMultiTaskLoss, self).__init__()
        
        # Core loss components
        self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=3)
        self.correlation_reg = TaskCorrelationRegularizer()
        
        # Task-specific losses with class balancing
        self.food_focal_loss = FocalLoss(gamma=2.0)
        self.cuisine_focal_loss = FocalLoss(gamma=1.5)  # Less aggressive for cuisine
        
        # Adaptive components
        self.adaptation_rate = adaptation_rate
        self.task_difficulty_history = {
            'food': [],
            'cuisine': [], 
            'nutrition': []
        }
        
        # Dynamic task weights (learned)
        self.dynamic_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, predictions: Tuple[torch.Tensor, ...], 
                targets: Tuple[torch.Tensor, ...],
                epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate adaptive multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            epoch: Current training epoch (for adaptation)
            
        Returns:
            tuple: (adaptive_loss, comprehensive_breakdown)
        """
        food_logits, cuisine_logits, nutrition_values = predictions
        food_labels, cuisine_labels, nutrition_targets = targets
        
        # 1. Calculate base uncertainty-weighted loss
        uncertainty_loss, base_breakdown = self.uncertainty_loss(predictions, targets)
        
        # 2. Calculate task-specific focal losses
        food_focal = self.food_focal_loss(food_logits, food_labels)
        cuisine_focal = self.cuisine_focal_loss(cuisine_logits, cuisine_labels)
        
        # 3. Task correlation regularization
        # Extract features from logits (simplified)
        food_features = F.softmax(food_logits, dim=1)
        cuisine_features = F.softmax(cuisine_logits, dim=1)
        correlation_loss = self.correlation_reg(food_features, cuisine_features)
        
        # 4. Dynamic weighting based on learning progress
        dynamic_food_weight = torch.sigmoid(self.dynamic_weights[0])
        dynamic_cuisine_weight = torch.sigmoid(self.dynamic_weights[1])
        dynamic_nutrition_weight = torch.sigmoid(self.dynamic_weights[2])
        
        # 5. Combine all components
        total_loss = (
            uncertainty_loss +
            0.1 * (dynamic_food_weight * food_focal +
                   dynamic_cuisine_weight * cuisine_focal) +
            correlation_loss
        )
        
        # 6. Update task difficulty tracking
        self._update_difficulty_tracking(base_breakdown, epoch)
        
        # 7. Comprehensive loss breakdown
        comprehensive_breakdown = {
            **base_breakdown,
            'food_focal': food_focal.item(),
            'cuisine_focal': cuisine_focal.item(),
            'correlation_reg': correlation_loss.item(),
            'dynamic_food_weight': dynamic_food_weight.item(),
            'dynamic_cuisine_weight': dynamic_cuisine_weight.item(),
            'dynamic_nutrition_weight': dynamic_nutrition_weight.item(),
            'total_adaptive_loss': total_loss.item()
        }
        
        return total_loss, comprehensive_breakdown
    
    def _update_difficulty_tracking(self, loss_breakdown: Dict[str, float], epoch: int):
        """Update task difficulty tracking for adaptive weighting"""
        
        # Track task-specific loss trends
        self.task_difficulty_history['food'].append(loss_breakdown['food_loss'])
        self.task_difficulty_history['cuisine'].append(loss_breakdown['cuisine_loss'])
        self.task_difficulty_history['nutrition'].append(loss_breakdown['nutrition_loss'])
        
        # Keep only recent history (sliding window)
        window_size = 50
        for task in self.task_difficulty_history:
            if len(self.task_difficulty_history[task]) > window_size:
                self.task_difficulty_history[task] = self.task_difficulty_history[task][-window_size:]
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about loss adaptation"""
        
        stats = {}
        
        # Task difficulty trends
        for task, history in self.task_difficulty_history.items():
            if len(history) >= 2:
                recent_avg = np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)
                early_avg = np.mean(history[:10]) if len(history) >= 10 else recent_avg
                
                stats[f'{task}_difficulty_trend'] = recent_avg - early_avg
                stats[f'{task}_recent_avg'] = recent_avg
                stats[f'{task}_variance'] = np.var(history)
        
        # Current dynamic weights
        with torch.no_grad():
            stats['current_dynamic_weights'] = {
                'food': torch.sigmoid(self.dynamic_weights[0]).item(),
                'cuisine': torch.sigmoid(self.dynamic_weights[1]).item(),
                'nutrition': torch.sigmoid(self.dynamic_weights[2]).item()
            }
        
        # Uncertainty weights
        uncertainty_weights = self.uncertainty_loss.get_task_weights()
        stats['uncertainty_weights'] = uncertainty_weights
        
        return stats