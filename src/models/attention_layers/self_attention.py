"""
Multi-head self-attention mechanisms (Transformer family).
Implements transformer-style attention for computer vision tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .base_attention import BaseAttention, AttentionMixin


class MultiHeadSelfAttention(BaseAttention, AttentionMixin):
    """
    Multi-Head Self-Attention mechanism for vision transformers.
    
    Mathematical formulation:
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        
        # Generate Q, K, V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        attention_scores = self.apply_attention_mask(attention_scores, mask)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Final linear projection
        output = self.out_linear(attended_values)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention weights for visualization."""
        with torch.no_grad():
            batch_size, seq_len, embed_dim = x.size()
            
            Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            return attention_weights.mean(dim=1)  # Average across heads