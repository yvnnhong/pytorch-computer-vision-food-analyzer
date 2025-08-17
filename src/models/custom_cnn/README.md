# Custom CNN Architectures for Food Image Analysis

Domain-specific CNN architectures optimized for food recognition tasks with multi-task learning capabilities.

## Overview

This module implements custom CNN architectures specifically designed for food image analysis, featuring:

- **Food-specific feature extraction** (texture, color, structure)
- **Mobile-optimized architectures** using depthwise separable convolutions
- **Multi-task learning** for food classification, cuisine detection, and nutrition regression
- **Production-ready deployment** with efficiency optimizations

## Architecture Features

### Food-Specific Analysis
- **Texture Analysis**: Multi-scale kernels (3x3, 5x5, 7x7) for surface patterns
- **Color Analysis**: HSV-inspired color space transformations for ripeness/cooking detection
- **Structure Analysis**: Dilated convolutions for food arrangement and composition

### Mobile Optimization
- **Depthwise Separable Convolutions**: 8-9x parameter reduction vs standard convolutions
- **Width Multipliers**: Configurable model sizes (0.5x, 1.0x, 1.5x)
- **Quantization-Ready**: Designed for mobile deployment and edge inference

## Module Structure

```
custom_cnn/
├── __init__.py                # Module exports and interface
├── food_blocks.py            # Domain-specific building blocks
├── efficient_conv.py         # Mobile-optimized convolutions
├── food_architectures.py     # Main network architectures
├── mobile_net.py             # Ultra-efficient mobile variant
└── model_factory.py          # Factory functions and utilities
```

## Quick Start

### Basic Usage
```python
from custom_cnn import create_food_cnn

# Create standard FoodNet
model = create_food_cnn('standard', num_food_classes=101)

# Create mobile-optimized version
mobile_model = create_food_cnn('mobile', width_multiplier=0.5)

# Forward pass
food_logits, cuisine_logits, nutrition_values = model(images)
```

### Custom Architecture Building
```python
from custom_cnn import FoodCNNBlock, DepthwiseSeparableConv

# Build custom food analysis block
texture_block = FoodCNNBlock(128, 256, block_type="texture_focus")

# Use efficient convolutions
efficient_conv = DepthwiseSeparableConv(64, 128)
```

## Available Architectures

### FoodNet Variants
- **Standard**: Full food analysis with texture, color, and structure blocks
- **Efficient**: Mobile-optimized using depthwise separable convolutions
- **Deep**: Maximum accuracy with deeper architecture

### MobileFoodNet
- **Ultra-lightweight**: Designed for mobile and edge deployment
- **Configurable width**: Adjustable model size via width multiplier
- **Single classifier**: Unified output head for efficiency

## Performance Characteristics

| Architecture | Parameters | Model Size | Use Case |
|-------------|------------|------------|----------|
| FoodNet Standard | ~25M | ~100MB | High accuracy server deployment |
| FoodNet Efficient | ~8M | ~32MB | Balanced accuracy/efficiency |
| FoodNet Deep | ~45M | ~180MB | Maximum accuracy research |
| MobileFoodNet 1.0x | ~3M | ~12MB | Mobile apps |
| MobileFoodNet 0.5x | ~1M | ~4MB | Edge devices |

## Factory Functions

### Model Creation
```python
from custom_cnn import create_food_cnn

# Available architectures: 'standard', 'efficient', 'deep', 'mobile'
model = create_food_cnn(
    architecture='standard',
    num_food_classes=101,
    num_cuisine_classes=13,
    nutrition_dim=4,
    dropout_rate=0.3
)
```

### Efficiency Analysis
```python
from custom_cnn import calculate_model_efficiency

metrics = calculate_model_efficiency(model)
print(f"Parameters: {metrics['total_parameters']:,}")
print(f"Model size: {metrics['model_size_mb']:.1f} MB")
print(f"Efficiency score: {metrics['efficiency_score']:.1f}")
```

## Building Blocks

### Food-Specific Blocks

#### FoodTextureBlock
```python
from custom_cnn import FoodTextureBlock

texture_block = FoodTextureBlock(in_channels=128, out_channels=256)
# Analyzes food textures at multiple scales with attention
```

#### ColorAnalysisBlock  
```python
from custom_cnn import ColorAnalysisBlock

color_block = ColorAnalysisBlock(in_channels=128, out_channels=256)
# HSV-inspired color analysis for ingredient identification
```

#### FoodStructureBlock
```python
from custom_cnn import FoodStructureBlock

structure_block = FoodStructureBlock(in_channels=128, out_channels=256)
# Dilated convolutions for food arrangement analysis
```

### Efficient Convolutions

#### DepthwiseSeparableConv
```python
from custom_cnn import DepthwiseSeparableConv

# 8-9x parameter reduction vs standard convolution
efficient_conv = DepthwiseSeparableConv(
    in_channels=64, 
    out_channels=128,
    kernel_size=3,
    stride=1
)
```

## Multi-Task Learning

All architectures support multi-task learning with three heads:

1. **Food Classification**: 101 food categories (Food-101 dataset)
2. **Cuisine Classification**: Regional cuisine types (Italian, Asian, etc.)
3. **Nutrition Regression**: Calories, protein, carbohydrates, fat

```python
# Forward pass returns all three tasks
food_logits, cuisine_logits, nutrition_values = model(images)

# Food classification (101 classes)
food_predictions = torch.argmax(food_logits, dim=1)

# Cuisine classification (13 classes) 
cuisine_predictions = torch.argmax(cuisine_logits, dim=1)

# Nutrition regression (4 values: calories, protein, carbs, fat)
nutrition_estimates = nutrition_values  # [batch_size, 4]
```

## Integration Examples

### With Attention Mechanisms
```python
from custom_cnn import FoodNet
from attention_layers import CBAM

class FoodNetWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FoodNet('standard')
        self.attention = CBAM(in_channels=512)
    
    def forward(self, x):
        features = self.backbone.get_feature_maps(x)['stage4']
        attended = self.attention(features)
        return self.backbone.classifier(attended)
```

### Mobile Deployment
```python
# Create mobile-optimized model
mobile_model = create_food_cnn('mobile', width_multiplier=0.5)

# Check efficiency
efficiency = calculate_model_efficiency(mobile_model)
print(f"Mobile model: {efficiency['model_size_mb']:.1f} MB")

# Quantization-ready forward pass
with torch.no_grad():
    mobile_output = mobile_model(images)
```

## Design Principles

### Food-Specific Inductive Biases
- **Multi-scale texture analysis**: Captures fine to coarse food surface patterns
- **Color space awareness**: Understands color variations indicating freshness/cooking
- **Structure understanding**: Analyzes food arrangement and composition

### Mobile-First Design
- **Depthwise separable convolutions**: Dramatic parameter reduction
- **Efficient attention**: Lightweight attention mechanisms when needed
- **Quantization-friendly**: Clean activation patterns for mobile optimization

### Production Readiness
- **Modular architecture**: Easy to customize and extend
- **Comprehensive testing**: Validated on Food-101 dataset
- **Performance monitoring**: Built-in efficiency analysis tools

## Research Foundation

Based on established computer vision techniques:
- **MobileNets**: Depthwise separable convolution efficiency
- **ResNet**: Residual connections for deep networks
- **CBAM**: Channel and spatial attention mechanisms
- **Multi-task Learning**: Shared representation learning

## Requirements

- PyTorch 2.0+
- torchvision
- Python 3.8+

## Performance Notes

- **Training**: Optimized for Food-101 dataset with 101 food classes
- **Inference**: Real-time capable on mobile devices (MobileFoodNet variants)
- **Memory**: Efficient memory usage through depthwise convolutions
- **Accuracy**: Competitive performance with domain-specific optimizations