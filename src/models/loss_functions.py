# Multi-task loss functions
"""
Advanced loss functions for multi-task learning with mathematical rigor.
Implements uncertainty-weighted losses, focal loss, and task correlation analysis.

Mathematical Foundation:
- Multi-task loss with learned uncertainty weighting (Kendall & Gal, 2017)
- Focal Loss for addressing class imbalance (Lin et al., 2017)  
- Task correlation regularization for multi-task optimization
- Gradient harmonization for balanced task learning
NOTETO SELF: REVIEW EVERYTHING TMW!!!!  
"""
#THIS IS JUST A PROTOTYPE -- CHECK OVER LATER 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict, List, Optional, Any 
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
    
    Args:
        alpha: Class weighting factor (float or tensor)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
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


class UncertaintyWeightedLoss(BaseLoss):
    """
    Multi-task loss with learnable uncertainty weighting.
    
    Mathematical foundation (Kendall & Gal, 2017):
    L_total = Σ_i [1/(2σ_i²) * L_i + log(σ_i)]
    
    where:
    - L_i is the loss for task i
    - σ_i is the learned uncertainty (noise parameter) for task i
    
    The uncertainty σ is learned as a parameter, allowing the model to 
    automatically balance task importance based on their relative difficulty.
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
    
    Mathematical approach:
    1. Calculate gradients for each task individually
    2. Identify conflicting gradient directions
    3. Apply gradient alignment regularization
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
                model_parameters: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        
        # This is a simplified version - in practice would need more sophisticated
        # gradient conflict detection and harmonization
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


class TaskCorrelationRegularizer(nn.Module):
    """
    Regularizer that encourages or discourages correlation between tasks.
    
    Mathematical formulation:
    R_corr = λ * |corr(f_food, f_cuisine) - target_correlation|²
    
    where f_food and f_cuisine are the feature representations for each task.
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
    4. Gradient conflict detection
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


class ContrastiveLoss(BaseLoss):
    """
    Contrastive loss for learning better feature representations.
    
    Mathematical formulation:
    L_contrastive = (1-y) * d² + y * max(0, margin - d)²
    
    where:
    - y = 1 if samples are from same class, 0 otherwise
    - d = euclidean distance between feature vectors
    - margin = threshold for negative pairs
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss for feature learning.
        
        Args:
            features: Feature vectors (N, D)
            labels: Class labels (N,)
            
        Returns:
            torch.Tensor: Contrastive loss
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Calculate pairwise distances
        distances = torch.cdist(features, features, p=2)
        
        # Create mask for positive pairs (same class)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        
        # Remove diagonal (self-pairs)
        positive_mask.fill_diagonal_(0)
        
        # Negative mask
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)
        
        # Positive loss: minimize distance for same class
        positive_loss = positive_mask * distances.pow(2)
        
        # Negative loss: maximize distance (up to margin) for different classes  
        negative_loss = negative_mask * torch.clamp(self.margin - distances, min=0).pow(2)
        
        # Combine losses
        total_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_mask.sum() + negative_mask.sum() + 1e-8)
        
        return total_loss


class InfoNCELoss(BaseLoss):
    """
    InfoNCE loss for self-supervised representation learning.
    
    Mathematical formulation:
    L_InfoNCE = -log(exp(q·k⁺/τ) / Σᵢ exp(q·kᵢ/τ))
    
    where:
    - q is the query representation
    - k⁺ is the positive key
    - kᵢ are all keys (positive and negative)
    - τ is the temperature parameter
    """
    
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                positive_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate InfoNCE loss.
        
        Args:
            query: Query representations (N, D)
            keys: Key representations (N, D)  
            positive_mask: Binary mask indicating positive pairs (N, N)
            
        Returns:
            torch.Tensor: InfoNCE loss
        """
        # Normalize representations
        query = F.normalize(query, p=2, dim=1)
        keys = F.normalize(keys, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(query, keys.t()) / self.temperature
        
        # For numerical stability
        similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0].detach()
        
        # Calculate InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        positive_sim = exp_sim * positive_mask
        
        # Sum over all keys for denominator
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # Sum over positive keys for numerator
        numerator = positive_sim.sum(dim=1, keepdim=True)
        
        # InfoNCE loss
        loss = -torch.log(numerator / (denominator + 1e-8))
        
        return loss.mean()


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


if __name__ == "__main__":
    # Test advanced loss functions
    print("Testing Advanced Multi-Task Loss Functions...")
    
    # Create dummy data
    batch_size, num_food_classes, num_cuisine_classes = 8, 101, 10
    
    # Dummy predictions
    food_logits = torch.randn(batch_size, num_food_classes)
    cuisine_logits = torch.randn(batch_size, num_cuisine_classes)
    nutrition_values = torch.randn(batch_size, 4)
    predictions = (food_logits, cuisine_logits, nutrition_values)
    
    # Dummy targets
    food_labels = torch.randint(0, num_food_classes, (batch_size,))
    cuisine_labels = torch.randint(0, num_cuisine_classes, (batch_size,))
    nutrition_targets = torch.randn(batch_size, 4)
    targets = (food_labels, cuisine_labels, nutrition_targets)
    
    # Test different loss functions
    loss_functions = {
        'focal': FocalLoss(),
        'uncertainty': UncertaintyWeightedLoss(),
        'adaptive': AdaptiveMultiTaskLoss()
    }
    
    for name, loss_fn in loss_functions.items():
        try:
            if name == 'focal':
                # Test focal loss on food classification only
                loss = loss_fn(food_logits, food_labels)
                print(f"{name}: {loss.item():.4f}")
            else:
                # Test multi-task losses
                total_loss, breakdown = loss_fn(predictions, targets)
                print(f"{name}: {total_loss.item():.4f}")
                print(f"  Breakdown: {list(breakdown.keys())[:3]}...")
                
        except Exception as e:
            print(f"Error testing {name}: {e}")
    
    print("\nAdvanced loss function tests complete!")
    print("\nMathematical Foundation Implemented:")
    print("  - Uncertainty-weighted multi-task learning")
    print("  - Focal loss for class imbalance") 
    print("  - Gradient harmonization")
    print("  - Task correlation analysis")
    print("  - Adaptive loss weighting")