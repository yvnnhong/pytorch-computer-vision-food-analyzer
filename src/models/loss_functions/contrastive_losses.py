"""
Contrastive and self-supervised loss implementations.
Includes contrastive loss and InfoNCE for representation learning.
"""

import torch
import torch.nn.functional as F
from .base_losses import BaseLoss


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