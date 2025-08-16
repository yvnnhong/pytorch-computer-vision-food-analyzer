# Attention Mechanisms for Computer Vision

Advanced attention mechanisms optimized for multi-task food image analysis and computer vision applications.

## Overview

This module provides production-ready attention layers with mathematical rigor and practical optimizations for real-world deployment.

### Mathematical Foundation
- **Self-Attention**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **Channel Attention**: `A_c = σ(W_2 δ(W_1 GAP(F)))`
- **Spatial Attention**: `A_s = σ(conv([AvgPool(F); MaxPool(F)]))`
- **Cross-Modal Attention**: `A_cm = softmax(F_1 × W × F_2^T)`

## Module Structure

```
attention_layers/
├── base_attention.py          # Abstract base classes and utilities
├── channel_spatial.py         # Channel & Spatial attention (CBAM family)
├── self_attention.py          # Multi-head self-attention (Transformer)
├── cross_modal.py             # Cross-modal & multi-task attention
├── food_specific.py           # Domain-adapted food attention
├── factory.py                 # Factory functions & benchmarking
├── visualization.py           # Attention visualization tools
└── __init__.py               # Unified imports
```

## Quick Start

### Basic Usage
```python
from attention_layers import ChannelAttention, CBAM, create_attention_layer

# Direct instantiation
channel_att = ChannelAttention(in_channels=256)
cbam = CBAM(in_channels=512)

# Factory pattern
spatial_att = create_attention_layer('spatial')
food_att = create_attention_layer('food_specific', 
                                 in_channels=512, 
                                 num_food_classes=101)

# Apply attention
x = torch.randn(2, 256, 32, 32)
attended_features = channel_att(x)
```

### Multi-Task Learning
```python
from attention_layers import MultiTaskAttentionFusion

# Fuse features across tasks
fusion = MultiTaskAttentionFusion(feature_dim=256, num_tasks=3)
task_features = [food_features, cuisine_features, nutrition_features]
fused_features = fusion(task_features)
```

### Visualization & Analysis
```python
from attention_layers import visualize_attention_maps, benchmark_attention_speed

# Visualize attention patterns
attention_maps = visualize_attention_maps(channel_att, input_tensor)

# Performance benchmarking
speed_stats = benchmark_attention_speed(channel_att, input_tensor)
print(f"Average inference: {speed_stats['mean_ms']:.2f}ms")
```

## Available Attention Mechanisms

### Channel & Spatial Attention (CBAM Family)
- **ChannelAttention**: Emphasizes important feature channels
- **SpatialAttention**: Focuses on important spatial locations  
- **CBAM**: Combines channel and spatial attention sequentially
- **SqueezeExcitation**: Efficient channel attention mechanism

### Transformer-Style Attention
- **MultiHeadSelfAttention**: Multi-head self-attention for vision transformers

### Cross-Modal & Multi-Task
- **CrossModalAttention**: Attention between different modalities
- **MultiTaskAttentionFusion**: Shares information across multiple tasks

### Domain-Specific
- **FoodSpecificAttention**: Specialized for food image analysis
  - Texture attention for surface patterns
  - Color attention for ingredient identification
  - Shape attention for structural elements

## Performance Characteristics

| Mechanism | Parameters | Memory | Speed | Use Case |
|-----------|------------|---------|-------|----------|
| ChannelAttention | Low | Low | Fast | Feature selection |
| SpatialAttention | Minimal | Low | Fast | Region focus |
| CBAM | Low | Low | Fast | General enhancement |
| MultiHeadSelfAttention | High | High | Moderate | Global context |
| FoodSpecificAttention | Medium | Medium | Moderate | Domain adaptation |

## Factory Functions

### Create Attention Layers
```python
from attention_layers import create_attention_layer

# Available types: 'channel', 'spatial', 'cbam', 'se', 'mhsa', 
#                  'cross_modal', 'food_specific', 'multitask_fusion'

attention = create_attention_layer('cbam', in_channels=256)
```

### Benchmarking & Comparison
```python
from attention_layers import compare_attention_mechanisms, create_attention_suite

# Create suite of attention mechanisms
attention_suite = create_attention_suite(in_channels=256)

# Compare performance
comparison = compare_attention_mechanisms(list(attention_suite.values()))
```

## Visualization Tools

### Attention Heatmaps
```python
from attention_layers import create_attention_heatmap, plot_attention_summary

# Extract and visualize attention weights
attention_weights = extract_attention_weights(attention_module, input_tensor)
heatmap = create_attention_heatmap(attention_weights['attention_weights'])

# Generate summary plot
plot_attention_summary(attention_weights, save_path='attention_analysis.png')
```

### Analysis Reports
```python
from attention_layers import attention_statistics_report

# Generate comprehensive analysis
report = attention_statistics_report(attention_module, input_tensor)
print(report)
```

## Integration Examples

### With ResNet Models
```python
import torch.nn as nn
from attention_layers import CBAM

class ResNetWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.attention = CBAM(in_channels=2048)
        
    def forward(self, x):
        features = self.resnet.features(x)
        attended = self.attention(features)
        return self.resnet.classifier(attended)
```

### With Custom CNNs
```python
from attention_layers import FoodSpecificAttention

class FoodCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = self._build_conv_layers()
        self.food_attention = FoodSpecificAttention(in_channels=512)
        
    def forward(self, x):
        features = self.conv_layers(x)
        attended_features = self.food_attention(features)
        return self.classifier(attended_features)
```

## 🔬 Mathematical Details

### Channel Attention
```
A_c = σ(MLP(GAP(F)) + MLP(GMP(F)))
```
Where GAP/GMP are global average/max pooling operations.

### Spatial Attention  
```
A_s = σ(Conv([AvgPool_c(F); MaxPool_c(F)]))
```
Channel-wise pooling followed by convolution.

### Multi-Head Self-Attention
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

## Research References

- **CBAM**: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
- **Squeeze-and-Excitation**: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
- **Self-Attention**: "Attention Is All You Need" (Vaswani et al., 2017)

## Production Features

- **Memory Efficient**: Optimized implementations for reduced memory footprint
- **GPU Accelerated**: Full CUDA support for all attention mechanisms
- **Quantization Ready**: Compatible with model quantization for mobile deployment
- **Benchmarking Tools**: Built-in performance analysis and comparison utilities
- **Visualization Support**: Comprehensive attention map visualization and analysis