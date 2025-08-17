"""
Uncertainty quantification utilities for ensemble predictions.
Implements mathematical methods for measuring prediction uncertainty.
"""

import torch
import numpy as np
from typing import List


class ModelUncertainty:
    """
    Uncertainty quantification for ensemble predictions.
    
    Methods:
    - Predictive Entropy: H(y|x) = -Σ p(y|x) log p(y|x)
    - Mutual Information: I(y;θ|x,D) = H(y|x,D) - E[H(y|x,θ)]
    - Variance of Predictions: Var[fᵢ(x)]
    """
    
    @staticmethod
    def predictive_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        """
        Calculate predictive entropy for uncertainty estimation.
        
        Args:
            probabilities: Softmax probabilities (B, C)
            
        Returns:
            torch.Tensor: Entropy values (B,)
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probabilities = torch.clamp(probabilities, eps, 1.0 - eps)
        entropy = -torch.sum(probabilities * torch.log(probabilities), dim=1)
        return entropy
    
    @staticmethod
    def mutual_information(model_predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculate mutual information between predictions and model parameters.
        
        Args:
            model_predictions: List of prediction tensors from different models
            
        Returns:
            torch.Tensor: Mutual information values
        """
        # Stack predictions
        stacked_preds = torch.stack(model_predictions, dim=0)  # (num_models, batch_size, num_classes)
        
        # Mean prediction across models
        mean_pred = torch.mean(stacked_preds, dim=0)
        
        # Entropy of mean prediction
        mean_entropy = ModelUncertainty.predictive_entropy(mean_pred)
        
        # Mean entropy of individual predictions
        individual_entropies = torch.stack([
            ModelUncertainty.predictive_entropy(pred) for pred in model_predictions
        ], dim=0)
        mean_individual_entropy = torch.mean(individual_entropies, dim=0)
        
        # Mutual information
        mutual_info = mean_entropy - mean_individual_entropy
        return mutual_info
    
    @staticmethod
    def prediction_variance(predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculate variance across model predictions.
        
        Args:
            predictions: List of prediction tensors
            
        Returns:
            torch.Tensor: Variance values
        """
        stacked = torch.stack(predictions, dim=0)
        variance = torch.var(stacked, dim=0)
        return torch.mean(variance, dim=1)  # Mean variance across classes


def evaluate_ensemble_diversity(models: List[torch.nn.Module], 
                               dataloader: torch.utils.data.DataLoader,
                               device: str = 'cpu') -> dict:
    """
    Evaluate diversity metrics for ensemble models.
    
    Args:
        models: List of models in ensemble
        dataloader: Data loader for evaluation
        device: Computing device
        
    Returns:
        Dict with diversity metrics
    """
    all_predictions = []
    
    # Collect predictions from all models
    for model in models:
        model.eval()
        model_preds = []
        
        with torch.no_grad():
            for batch_idx, (images, *_) in enumerate(dataloader):
                if batch_idx >= 10:  # Limit for efficiency
                    break
                
                images = images.to(device)
                food_logits, _, _ = model(images)
                
                # Convert to predictions
                predictions = torch.argmax(food_logits, dim=1)
                model_preds.append(predictions.cpu())
        
        all_predictions.append(torch.cat(model_preds))
    
    # Calculate diversity metrics
    num_models = len(models)
    pairwise_disagreements = []
    
    for i in range(num_models):
        for j in range(i + 1, num_models):
            disagreement = (all_predictions[i] != all_predictions[j]).float().mean().item()
            pairwise_disagreements.append(disagreement)
    
    # Overall diversity metrics
    mean_disagreement = np.mean(pairwise_disagreements)
    prediction_entropy = []
    
    for sample_idx in range(len(all_predictions[0])):
        sample_preds = [preds[sample_idx].item() for preds in all_predictions]
        pred_counts = torch.bincount(torch.tensor(sample_preds))
        pred_probs = pred_counts.float() / num_models
        pred_probs = pred_probs[pred_probs > 0]  # Remove zeros
        entropy = -torch.sum(pred_probs * torch.log(pred_probs)).item()
        prediction_entropy.append(entropy)
    
    return {
        'mean_pairwise_disagreement': mean_disagreement,
        'prediction_entropy_mean': np.mean(prediction_entropy),
        'prediction_entropy_std': np.std(prediction_entropy),
        'max_disagreement': max(pairwise_disagreements),
        'min_disagreement': min(pairwise_disagreements)
    }