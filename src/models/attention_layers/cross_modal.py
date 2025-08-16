"""
Cross-modal and multi-task attention mechanisms.
Handles attention between different modalities and tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from .base_attention import BaseAttention
from .self_attention import MultiHeadSelfAttention


class CrossModalAttention(BaseAttention):
    """
    Cross-Modal Attention for fusing features from different modalities/tasks.
    
    Mathematical formulation:
    A = softmax(F_1 W_q (F_2 W_k)^T / sqrt(d_k))
    Output = A (F_2 W_v)
    """
    
    def __init__(self, embed_dim: int, cross_dim: int, dropout: float = 0.1):
        super(CrossModalAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.cross_dim = cross_dim
        self.scale = math.sqrt(embed_dim)
        
        # Linear projections
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(cross_dim, embed_dim)
        self.v_linear = nn.Linear(cross_dim, embed_dim)
        
        # Output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_features: torch.Tensor, key_value_features: torch.Tensor) -> torch.Tensor:
        # Generate Q from query features, K and V from key-value features
        Q = self.q_linear(query_features)
        K = self.k_linear(key_value_features)
        V = self.v_linear(key_value_features)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Final linear projection
        output = self.out_linear(attended_values)
        
        return output


class MultiTaskAttentionFusion(BaseAttention):
    """
    Multi-task attention fusion for sharing information between tasks.
    Designed for food classification, cuisine classification, and nutrition regression.
    """
    
    def __init__(self, feature_dim: int, num_tasks: int = 3):
        super(MultiTaskAttentionFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_tasks = num_tasks
        
        # Cross-attention between tasks
        self.cross_attentions = nn.ModuleList([
            CrossModalAttention(feature_dim, feature_dim) for _ in range(num_tasks)
        ])
        
        # Self-attention for each task
        self.self_attentions = nn.ModuleList([
            MultiHeadSelfAttention(feature_dim, num_heads=8) for _ in range(num_tasks)
        ])
        
        # Task-specific projections
        self.task_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_tasks)
        ])
        
        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_tasks, num_tasks))
        
    def forward(self, task_features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(task_features) == self.num_tasks, f"Expected {self.num_tasks} task features"
        
        fused_features = []
        fusion_weights_norm = F.softmax(self.fusion_weights, dim=1)
        
        for i, features in enumerate(task_features):
            # Self-attention for current task
            self_attended = self.self_attentions[i](features)
            
            # Cross-attention with other tasks
            cross_attended_sum = torch.zeros_like(features)
            
            for j, other_features in enumerate(task_features):
                if i != j:  # Don't attend to self
                    cross_attended = self.cross_attentions[i](features, other_features)
                    cross_attended_sum += fusion_weights_norm[i, j] * cross_attended
            
            # Combine self and cross attention
            combined = self_attended + cross_attended_sum
            
            # Task-specific projection
            projected = self.task_projections[i](combined)
            
            fused_features.append(projected)
        
        return fused_features