"""
Attention Mechanisms for Computer Vision and Multi-Task Learning.

Mathematical Foundation:
- Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Channel Attention: A_c = σ(W_2 δ(W_1 GAP(F)))  
- Spatial Attention: A_s = σ(conv([AvgPool(F); MaxPool(F)]))
- Cross-Modal Attention: A_cm = softmax(F_1 × W × F_2^T)

This module provides production-ready attention layers optimized for food image analysis
and multi-task learning scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict, Any
from torch.nn.parameter import Parameter


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for emphasizing important feature channels.
    
    Mathematical formulation:
    A_c = σ(W_2 * ReLU(W_1 * GAP(F)) + W_2 * ReLU(W_1 * GMP(F)))
    
    Where GAP = Global Average Pooling, GMP = Global Max Pooling
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = max(in_channels // reduction_ratio, 1)
        
        # Shared MLP for both average and max pooling paths
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, self.reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduced_channels, in_channels)
        )
        
        # Global pooling operations
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input feature map.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Channel-attended features (B, C, H, W)
        """
        batch_size, channels, height, width = x.size()
        
        # Global Average Pooling path
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.shared_mlp(avg_pool)
        
        # Global Max Pooling path
        max_pool = self.max_pool(x).view(batch_size, channels)
        max_out = self.shared_mlp(max_pool)
        
        # Combine and apply sigmoid
        channel_attention = self.sigmoid(avg_out + max_out)
        channel_attention = channel_attention.view(batch_size, channels, 1, 1)
        
        # Apply attention weights
        return x * channel_attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for emphasizing important spatial locations.
    
    Mathematical formulation:
    A_s = σ(conv([AvgPool_c(F); MaxPool_c(F)]))
    
    Where AvgPool_c and MaxPool_c are channel-wise pooling operations.
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # Convolution layer for spatial attention
        self.conv = nn.Conv2d(
            in_channels=2,  # Average and max pooled features
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to input feature map.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Spatially-attended features (B, C, H, W)
        """
        # Channel-wise pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate pooled features
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        
        # Apply convolution and sigmoid
        spatial_attention = self.sigmoid(self.conv(pooled))  # (B, 1, H, W)
        
        # Apply attention weights
        return x * spatial_attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines both channel and spatial attention sequentially.
    Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
    
    Mathematical flow:
    F' = Channel_Attention(F) ⊗ F
    F'' = Spatial_Attention(F') ⊗ F'
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM attention (channel then spatial).
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: CBAM-attended features (B, C, H, W)
        """
        # Apply channel attention first
        x = self.channel_attention(x)
        
        # Then apply spatial attention
        x = self.spatial_attention(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for vision transformers.
    
    Mathematical formulation:
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    
    Where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
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
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor (B, N, D) where N is sequence length
            mask: Optional attention mask (B, N, N)
            
        Returns:
            torch.Tensor: Self-attended features (B, N, D)
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Generate Q, K, V
        Q = self.q_linear(x)  # (B, N, D)
        K = self.k_linear(x)  # (B, N, D)
        V = self.v_linear(x)  # (B, N, D)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D/H)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D/H)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D/H)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # (B, H, N, D/H)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )  # (B, N, D)
        
        # Final linear projection
        output = self.out_linear(attended_values)
        
        return output


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention for fusing features from different modalities/tasks.
    
    Mathematical formulation:
    A = softmax(F_1 W_q (F_2 W_k)^T / sqrt(d_k))
    Output = A (F_2 W_v)
    
    Useful for multi-task learning where different tasks attend to each other.
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
        """
        Apply cross-modal attention.
        
        Args:
            query_features: Query features (B, N_q, D_q)
            key_value_features: Key-Value features (B, N_kv, D_kv)
            
        Returns:
            torch.Tensor: Cross-attended features (B, N_q, D_q)
        """
        # Generate Q from query features, K and V from key-value features
        Q = self.q_linear(query_features)  # (B, N_q, D)
        K = self.k_linear(key_value_features)  # (B, N_kv, D)
        V = self.v_linear(key_value_features)  # (B, N_kv, D)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, N_q, N_kv)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # (B, N_q, D)
        
        # Final linear projection
        output = self.out_linear(attended_values)
        
        return output


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(SqueezeExcitation, self).__init__()
        
        reduced_channels = max(in_channels // reduction_ratio, 1)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: Global Average Pooling
        squeezed = self.squeeze(x).view(batch_size, channels)
        
        # Excitation: FC layers with sigmoid
        excited = self.excitation(squeezed).view(batch_size, channels, 1, 1)
        
        # Scale the input
        return x * excited


class FoodSpecificAttention(nn.Module):
    """
    Food-specific attention mechanism designed for food image analysis.
    
    Combines multiple attention mechanisms optimized for food characteristics:
    - Texture attention (for food surface details)
    - Color attention (for ingredient identification)  
    - Shape attention (for food structure recognition)
    """
    
    def __init__(self, in_channels: int, num_food_classes: int = 101):
        super(FoodSpecificAttention, self).__init__()
        
        self.in_channels = in_channels
        self.num_food_classes = num_food_classes
        
        # Texture attention - focuses on surface patterns
        self.texture_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Color attention - emphasizes color information
        self.color_attention = ChannelAttention(in_channels, reduction_ratio=8)
        
        # Shape attention - focuses on structural elements
        self.shape_attention = SpatialAttention(kernel_size=7)
        
        # Food-class specific weighting
        self.class_weights = nn.Parameter(torch.ones(num_food_classes, 3))  # 3 attention types
        
        # Fusion layer
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, food_class_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply food-specific attention.
        
        Args:
            x: Input features (B, C, H, W)
            food_class_logits: Optional food class predictions for adaptive weighting (B, num_classes)
            
        Returns:
            torch.Tensor: Food-specific attended features (B, C, H, W)
        """
        # Apply different attention mechanisms
        texture_attended = x * self.texture_attention(x)
        color_attended = self.color_attention(x)
        shape_attended = self.shape_attention(x)
        
        # Adaptive weighting based on predicted food class
        if food_class_logits is not None:
            batch_size = x.size(0)
            
            # Get class probabilities
            class_probs = F.softmax(food_class_logits, dim=1)  # (B, num_classes)
            
            # Weight attention mechanisms by predicted class
            attention_weights = torch.matmul(class_probs, F.softmax(self.class_weights, dim=1))  # (B, 3)
            
            # Apply weights
            texture_weight = attention_weights[:, 0].view(batch_size, 1, 1, 1)
            color_weight = attention_weights[:, 1].view(batch_size, 1, 1, 1)
            shape_weight = attention_weights[:, 2].view(batch_size, 1, 1, 1)
            
            # Weighted combination
            combined = (texture_weight * texture_attended + 
                       color_weight * color_attended + 
                       shape_weight * shape_attended)
        else:
            # Equal weighting if no class information
            combined = (texture_attended + color_attended + shape_attended) / 3
        
        # Final fusion
        output = self.fusion(combined)
        
        return output


class MultiTaskAttentionFusion(nn.Module):
    """
    Multi-task attention fusion for sharing information between tasks.
    
    Designed for food classification, cuisine classification, and nutrition regression.
    Each task can attend to features from other tasks.
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
        """
        Fuse features across multiple tasks using attention.
        
        Args:
            task_features: List of task-specific features [(B, N, D), ...]
            
        Returns:
            List[torch.Tensor]: Fused task features
        """
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


def create_attention_layer(attention_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create different types of attention layers.
    
    Args:
        attention_type: Type of attention ('channel', 'spatial', 'cbam', 'se', 'mhsa', 'food_specific')
        **kwargs: Arguments specific to each attention type
        
    Returns:
        nn.Module: Attention layer
    """
    
    attention_registry = {
        'channel': ChannelAttention,
        'spatial': SpatialAttention,
        'cbam': CBAM,
        'se': SqueezeExcitation,
        'mhsa': MultiHeadSelfAttention,
        'cross_modal': CrossModalAttention,
        'food_specific': FoodSpecificAttention,
        'multitask_fusion': MultiTaskAttentionFusion
    }
    
    if attention_type not in attention_registry:
        raise ValueError(f"Unknown attention type: {attention_type}. Available: {list(attention_registry.keys())}")
    
    return attention_registry[attention_type](**kwargs)


def visualize_attention_maps(attention_module: nn.Module, 
                           input_tensor: torch.Tensor,
                           save_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Visualize attention maps for interpretability.
    
    Args:
        attention_module: Attention module to visualize
        input_tensor: Input tensor to compute attention for
        save_path: Optional path to save visualization
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary of attention maps
    """
    attention_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                # For spatial attention maps
                attention_maps[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in attention_module.named_modules():
        if isinstance(module, (ChannelAttention, SpatialAttention, CBAM)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = attention_module(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps


if __name__ == "__main__":
    print("Testing Advanced Attention Mechanisms...")
    
    # Test parameters
    batch_size, channels, height, width = 2, 256, 32, 32
    seq_len, embed_dim = 100, 512
    
    # Create test tensors
    conv_input = torch.randn(batch_size, channels, height, width)
    seq_input = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test different attention mechanisms
    attention_tests = {
        'Channel Attention': ChannelAttention(channels),
        'Spatial Attention': SpatialAttention(),
        'CBAM': CBAM(channels),
        'Squeeze-Excitation': SqueezeExcitation(channels),
        'Multi-Head Self-Attention': MultiHeadSelfAttention(embed_dim),
        'Food-Specific Attention': FoodSpecificAttention(channels),
    }
    
    print(f"Input shapes: Conv({conv_input.shape}), Seq({seq_input.shape})")
    print("-" * 60)
    
    for name, attention_module in attention_tests.items():
        try:
            if 'Self-Attention' in name:
                output = attention_module(seq_input)
            else:
                output = attention_module(conv_input)
            
            # Calculate parameter count
            params = sum(p.numel() for p in attention_module.parameters())
            
            print(f"{name:25} | Output: {tuple(output.shape)} | Params: {params:,}")
            
        except Exception as e:
            print(f"{name:25} | Error: {str(e)}")
    
    print("-" * 60)
    print("Advanced attention mechanisms testing complete!")
    print("\nKey Features Implemented:")
    print("  - Channel & Spatial Attention (CBAM)")
    print("  - Multi-Head Self-Attention (Transformer)")
    print("  - Cross-Modal Attention (Multi-task)")
    print("  - Food-Specific Attention (Domain-adapted)")
    print("  - Squeeze-and-Excitation (Efficient)")
    print("  - Multi-Task Attention Fusion")
    print("  - Attention Visualization Tools")
    print("\nMathematical Rigor:")
    print("  - Proper normalization with √d_k scaling")
    print("  - Softmax attention with dropout regularization")
    print("  - Learnable parameters with proper initialization")
    print("  - Memory-efficient implementations")