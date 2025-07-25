# Models Directory

This directory contains PyTorch model architectures and utilities for the multi-task food classification system.

## Core Models

### food_classifier.py
Main multi-task model class combining food classification, cuisine classification, and nutrition regression.

### resnet_multitask.py
ResNet-50 based architecture adapted for multi-task learning with shared backbone and task-specific heads.

### efficientnet_model.py
EfficientNet implementation optimized for mobile deployment and inference speed.

### custom_cnn.py
Custom CNN architecture designed specifically for food image analysis with attention mechanisms.

## Supporting Components

### ensemble_model.py
Model ensemble techniques combining multiple architectures for improved accuracy.

### loss_functions.py
Custom multi-task loss functions with configurable task weighting and balancing strategies.

### attention_layers.py
Attention mechanism implementations for enhanced feature learning and model interpretability.

### model_utils.py
Utility functions for model loading, saving, architecture comparison, and performance profiling.

## Usage

```python
from src.models.food_classifier import FoodNutritionModel

# Initialize multi-task model
model = FoodNutritionModel(num_food_classes=101, num_cuisine_classes=10)

# Load pre-trained weights
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
```

## Model Requirements

- PyTorch >= 2.0.0
- torchvision for pre-trained backbones
- GPU recommended for training (8GB+ VRAM)