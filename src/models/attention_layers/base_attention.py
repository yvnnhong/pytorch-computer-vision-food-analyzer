"""
Abstract base classes for attention mechanisms.
Provides common interface and utilities for all attention types.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAttention(nn.Module, ABC):
    """Abstract base class for all attention mechanisms."""
    
    def __init__(self):
        super(BaseAttention, self).__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to input tensor."""
        pass
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return attention weights for visualization."""
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """Return module information."""
        return {
            'type': self.__class__.__name__,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def complexity_score(self) -> float:
        """Calculate relative complexity score."""
        params = sum(p.numel() for p in self.parameters())
        return min(params / 100000, 1.0) * 100  # Normalize to 0-100


class AttentionMixin:
    """Mixin class providing common attention utilities."""
    
    @staticmethod
    def apply_attention_mask(attention_scores: torch.Tensor, 
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention mask to scores."""
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        return attention_scores
    
    @staticmethod
    def scaled_dot_product(query: torch.Tensor, 
                          key: torch.Tensor, 
                          value: torch.Tensor,
                          scale: float = None) -> torch.Tensor:
        """Scaled dot-product attention computation."""
        if scale is None:
            scale = 1.0 / (query.size(-1) ** 0.5)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attention_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)