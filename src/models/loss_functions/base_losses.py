"""
Base loss classes and fundamental loss implementations.
Includes abstract base class and focal loss for class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseLoss(ABC, nn.Module):
    """Abstract base class for all loss functions with common functionality"""
    
    def __init__(self, reduction: str = 'mean'):
        super(BaseLoss, self).__init__()
        self.reduction = reduction
        self.loss_history = []
    
    @abstractmethod
    def forward(self, predictions, targets, **kwargs):
        pass
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """Get statistical summary of loss history"""
        if not self.loss_history:
            return {}
        
        history = torch.tensor(self.loss_history)
        return {
            'mean': history.mean().item(),
            'std': history.std().item(),
            'min': history.min().item(),
            'max': history.max().item(),
            'last': history[-1].item()
        }


class FocalLoss(BaseLoss):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    
    Mathematical formulation:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t is the predicted probability for the true class
    - α_t is the class-specific weighting factor
    - γ (gamma) is the focusing parameter
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            predictions: Logits tensor (N, C)
            targets: Target labels (N,)
            
        Returns:
            torch.Tensor: Focal loss value
        """
        # Calculate cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Calculate p_t (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        self.loss_history.append(focal_loss.item())
        return focal_loss