"""
Factory functions and analysis tools for loss functions.
Provides unified interface for creating and analyzing different loss types.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from .base_losses import BaseLoss, FocalLoss
from .uncertainty_losses import UncertaintyWeightedLoss, GradientHarmonizationLoss
from .contrastive_losses import ContrastiveLoss, InfoNCELoss
from .adaptive_losses import AdaptiveMultiTaskLoss


def create_loss_function(loss_type: str = 'adaptive', **kwargs) -> BaseLoss:
    """
    Factory function to create different loss functions.
    
    Args:
        loss_type: Type of loss ('adaptive', 'uncertainty', 'focal', 'contrastive')
        **kwargs: Additional arguments for loss function
        
    Returns:
        BaseLoss: Initialized loss function
    """
    
    loss_registry = {
        'focal': FocalLoss,
        'uncertainty': UncertaintyWeightedLoss,
        'gradient_harmonization': GradientHarmonizationLoss,
        'adaptive': AdaptiveMultiTaskLoss,
        'contrastive': ContrastiveLoss,
        'infonce': InfoNCELoss
    }
    
    if loss_type not in loss_registry:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_registry.keys())}")
    
    return loss_registry[loss_type](**kwargs)


def analyze_loss_landscape(model: nn.Module, 
                          dataloader: torch.utils.data.DataLoader,
                          loss_fn: BaseLoss,
                          device: str = 'cpu') -> Dict[str, Any]:
    """
    Analyze the loss landscape for multi-task learning insights.
    
    Args:
        model: Trained model
        dataloader: Data loader for analysis
        loss_fn: Loss function to analyze
        device: Computing device
        
    Returns:
        dict: Loss landscape analysis results
    """
    
    model.eval()
    
    analysis_results = {
        'task_losses': {'food': [], 'cuisine': [], 'nutrition': []},
        'total_losses': [],
        'gradient_norms': [],
        'task_correlations': []
    }
    
    with torch.no_grad():
        for batch_idx, (images, food_labels, cuisine_labels, nutrition_targets) in enumerate(dataloader):
            if batch_idx >= 100:  # Limit analysis for efficiency
                break
                
            images = images.to(device)
            food_labels = food_labels.to(device)
            cuisine_labels = cuisine_labels.to(device)
            nutrition_targets = nutrition_targets.to(device)
            
            # Forward pass
            predictions = model(images)
            targets = (food_labels, cuisine_labels, nutrition_targets)
            
            # Calculate losses
            if hasattr(loss_fn, 'forward') and callable(getattr(loss_fn, 'forward')):
                if isinstance(loss_fn, (UncertaintyWeightedLoss, AdaptiveMultiTaskLoss)):
                    total_loss, loss_breakdown = loss_fn(predictions, targets)
                    analysis_results['task_losses']['food'].append(loss_breakdown['food_loss'])
                    analysis_results['task_losses']['cuisine'].append(loss_breakdown['cuisine_loss'])
                    analysis_results['task_losses']['nutrition'].append(loss_breakdown['nutrition_loss'])
                else:
                    total_loss = loss_fn(predictions, targets)
                
                analysis_results['total_losses'].append(total_loss.item())
    
    # Calculate summary statistics
    for task in analysis_results['task_losses']:
        task_losses = analysis_results['task_losses'][task]
        if task_losses:
            analysis_results[f'{task}_loss_stats'] = {
                'mean': np.mean(task_losses),
                'std': np.std(task_losses),
                'min': np.min(task_losses),
                'max': np.max(task_losses)
            }
    
    # Calculate task correlations
    if all(analysis_results['task_losses'].values()):
        food_losses = np.array(analysis_results['task_losses']['food'])
        cuisine_losses = np.array(analysis_results['task_losses']['cuisine'])
        nutrition_losses = np.array(analysis_results['task_losses']['nutrition'])
        
        analysis_results['task_correlations'] = {
            'food_cuisine': np.corrcoef(food_losses, cuisine_losses)[0, 1],
            'food_nutrition': np.corrcoef(food_losses, nutrition_losses)[0, 1],
            'cuisine_nutrition': np.corrcoef(cuisine_losses, nutrition_losses)[0, 1]
        }
    
    return analysis_results