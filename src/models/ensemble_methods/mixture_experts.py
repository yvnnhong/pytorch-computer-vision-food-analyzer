"""
Mixture of Experts ensemble implementation.
Each expert specializes in different food types with confidence weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts ensemble where each expert specializes in different food types.
    
    Mathematical formulation:
    P(y|x) = Σᵢ P(expert_i|x) * P(y|x, expert_i)
    
    Where P(expert_i|x) is the gating probability for expert i.
    """
    
    def __init__(self, 
                 expert_models: List[nn.Module],
                 expert_specializations: List[str],
                 gating_dim: int = 256):
        super(MixtureOfExperts, self).__init__()
        
        self.experts = nn.ModuleList(expert_models)
        self.expert_specializations = expert_specializations
        self.num_experts = len(expert_models)
        
        # Gating network that decides which expert to use
        self.gating_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, gating_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gating_dim, gating_dim // 2),
            nn.ReLU(),
            nn.Linear(gating_dim // 2, self.num_experts),
            nn.Softmax(dim=1)
        )
        
        # Expert confidence networks
        self.expert_confidence = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gating_dim, 1),
                nn.Sigmoid()
            ) for _ in range(self.num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through mixture of experts.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (food_logits, cuisine_logits, nutrition_values, expert_info)
        """
        # Calculate expert gating weights
        gating_features = self.gating_network[:-1](x)  # All layers except softmax
        expert_weights = F.softmax(self.gating_network[-1](gating_features), dim=1)
        
        # Get expert confidences
        expert_confidences = [
            conf_net(gating_features) for conf_net in self.expert_confidence
        ]
        expert_confidences = torch.cat(expert_confidences, dim=1)
        
        # Weight experts by both gating weights and confidence
        combined_weights = expert_weights * expert_confidences
        combined_weights = F.normalize(combined_weights, p=1, dim=1)  # Renormalize
        
        # Collect expert predictions
        expert_food_preds = []
        expert_cuisine_preds = []
        expert_nutrition_preds = []
        
        for expert in self.experts:
            food_logits, cuisine_logits, nutrition_values = expert(x)
            expert_food_preds.append(food_logits)
            expert_cuisine_preds.append(cuisine_logits)
            expert_nutrition_preds.append(nutrition_values)
        
        # Weighted combination
        final_food_logits = sum(
            w.unsqueeze(-1) * pred 
            for w, pred in zip(combined_weights.split(1, dim=1), expert_food_preds)
        )
        
        final_cuisine_logits = sum(
            w.unsqueeze(-1) * pred 
            for w, pred in zip(combined_weights.split(1, dim=1), expert_cuisine_preds)
        )
        
        final_nutrition = sum(
            w.unsqueeze(-1) * pred 
            for w, pred in zip(combined_weights.split(1, dim=1), expert_nutrition_preds)
        )
        
        # Expert metadata
        metadata = {
            'expert_weights': expert_weights.detach().cpu().numpy(),
            'expert_confidences': expert_confidences.detach().cpu().numpy(),
            'combined_weights': combined_weights.detach().cpu().numpy(),
            'dominant_expert': torch.argmax(combined_weights, dim=1).cpu().numpy(),
            'specializations': self.expert_specializations
        }
        
        return final_food_logits, final_cuisine_logits, final_nutrition, metadata