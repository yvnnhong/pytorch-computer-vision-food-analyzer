# Advanced Loss Functions for Multi-Task Learning

Mathematically rigorous loss function implementations for multi-task food classification with uncertainty quantification, gradient harmonization, and adaptive weighting strategies.

## Overview

This module provides production-ready loss functions that address key challenges in multi-task learning: task balancing, class imbalance, gradient conflicts, and uncertainty quantification. All implementations are based on established research with proper mathematical foundations.

## Mathematical Foundation

### Core Formulations

**Uncertainty-Weighted Multi-Task Loss (Kendall & Gal, 2017)**:
```
L_total = Σᵢ [1/(2σᵢ²) * Lᵢ + log(σᵢ)]
```

**Focal Loss (Lin et al., 2017)**:
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**Contrastive Loss**:
```
L_contrastive = (1-y) * d² + y * max(0, margin - d)²
```

**InfoNCE Loss**:
```
L_InfoNCE = -log(exp(q·k⁺/τ) / Σᵢ exp(q·kᵢ/τ))
```

**Task Correlation Regularization**:
```
R_corr = λ * |corr(f_food, f_cuisine) - target_correlation|²
```

## Module Structure

```
loss_functions/
├── __init__.py              # Module interface and exports
├── base_losses.py           # Base classes and focal loss
├── uncertainty_losses.py    # Uncertainty-weighted learning
├── contrastive_losses.py    # Representation learning losses
├── adaptive_losses.py       # Advanced multi-task adaptation
└── loss_factory.py          # Factory and analysis tools
```

## Quick Start

### Basic Usage
```python
from loss_functions import create_loss_function

# Create uncertainty-weighted loss
loss_fn = create_loss_function('uncertainty', num_tasks=3)

# Forward pass
predictions = (food_logits, cuisine_logits, nutrition_values)
targets = (food_labels, cuisine_labels, nutrition_targets)
total_loss, breakdown = loss_fn(predictions, targets)
```

### Advanced Configuration
```python
from loss_functions import AdaptiveMultiTaskLoss

# Create adaptive loss with custom parameters
adaptive_loss = AdaptiveMultiTaskLoss(
    num_food_classes=101,
    num_cuisine_classes=13,
    adaptation_rate=0.01
)

# Training loop with adaptation
for epoch in range(num_epochs):
    total_loss, breakdown = adaptive_loss(predictions, targets, epoch=epoch)
    
    # Monitor adaptation
    stats = adaptive_loss.get_adaptation_statistics()
    print(f"Dynamic weights: {stats['current_dynamic_weights']}")
```

## Available Loss Functions

### 1. Focal Loss
**Purpose**: Address class imbalance in classification tasks by focusing on hard examples.

**Mathematical Foundation**:
- Modifies cross-entropy with focusing parameter γ
- Down-weights easy examples, up-weights hard examples
- Class-specific weighting factor α for additional balance

**Usage**:
```python
from loss_functions import FocalLoss

focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
loss = focal_loss(predictions, targets)
```

**Best For**: Highly imbalanced classification tasks (e.g., rare food categories).

### 2. Uncertainty-Weighted Loss
**Purpose**: Automatically balance multiple tasks using learned uncertainty parameters.

**Key Features**:
- Learnable task-specific uncertainty σᵢ
- Automatic task importance weighting
- Prevents task domination during training
- Based on homoscedastic uncertainty modeling

**Usage**:
```python
from loss_functions import UncertaintyWeightedLoss

uncertainty_loss = UncertaintyWeightedLoss(num_tasks=3)
total_loss, breakdown = uncertainty_loss(predictions, targets)

# Get current task importance
weights = uncertainty_loss.get_task_weights()
```

**Best For**: Multi-task scenarios where optimal task weighting is unknown.

### 3. Gradient Harmonization Loss
**Purpose**: Address conflicting gradients between tasks for balanced learning.

**Key Features**:
- Detects gradient conflicts using cosine similarity
- Applies harmonization penalty for conflicting directions
- Maintains task-specific learning while reducing conflicts

**Usage**:
```python
from loss_functions import GradientHarmonizationLoss

harmonization_loss = GradientHarmonizationLoss(alpha=0.1)
total_loss, breakdown = harmonization_loss(
    predictions, targets, model_parameters=list(model.parameters())
)
```

**Best For**: Multi-task scenarios with known gradient conflicts.

### 4. Contrastive Loss
**Purpose**: Learn better feature representations through contrastive learning.

**Key Features**:
- Minimizes distance for same-class pairs
- Maximizes distance for different-class pairs (up to margin)
- Improves feature discriminability

**Usage**:
```python
from loss_functions import ContrastiveLoss

contrastive_loss = ContrastiveLoss(margin=1.0)
loss = contrastive_loss(features, labels)
```

**Best For**: Improving feature representations for better classification.

### 5. InfoNCE Loss
**Purpose**: Self-supervised representation learning with normalized temperature scaling.

**Key Features**:
- Contrastive learning with temperature parameter
- Normalized similarity computation
- Suitable for self-supervised pretraining

**Usage**:
```python
from loss_functions import InfoNCELoss

infonce_loss = InfoNCELoss(temperature=0.07)
loss = infonce_loss(query, keys, positive_mask)
```

**Best For**: Self-supervised pretraining and representation learning.

### 6. Adaptive Multi-Task Loss
**Purpose**: Advanced multi-task learning with multiple adaptation mechanisms.

**Key Features**:
- Combines uncertainty weighting, focal loss, and correlation regularization
- Dynamic task weighting based on learning progress
- Task difficulty tracking and adaptation
- Comprehensive adaptation statistics

**Components**:
- **Uncertainty Weighting**: Automatic task balancing
- **Focal Loss**: Class imbalance handling
- **Correlation Regularization**: Task relationship management
- **Dynamic Weights**: Learning progress adaptation

**Usage**:
```python
from loss_functions import AdaptiveMultiTaskLoss

adaptive_loss = AdaptiveMultiTaskLoss(
    num_food_classes=101,
    num_cuisine_classes=13
)

# Training with adaptation
for epoch in range(num_epochs):
    total_loss, breakdown = adaptive_loss(predictions, targets, epoch=epoch)
    
    # Monitor adaptation progress
    adaptation_stats = adaptive_loss.get_adaptation_statistics()
```

**Best For**: Complex multi-task scenarios requiring sophisticated adaptation.

## Factory Pattern

### Loss Creation
```python
from loss_functions import create_loss_function

# Available types: 'focal', 'uncertainty', 'gradient_harmonization', 
#                  'adaptive', 'contrastive', 'infonce'

loss_fn = create_loss_function(
    loss_type='adaptive',
    num_food_classes=101,
    num_cuisine_classes=13
)
```

### Loss Analysis
```python
from loss_functions import analyze_loss_landscape

# Comprehensive loss landscape analysis
analysis = analyze_loss_landscape(
    model=trained_model,
    dataloader=val_loader,
    loss_fn=loss_function,
    device='cuda'
)

print(f"Task correlations: {analysis['task_correlations']}")
print(f"Loss statistics: {analysis['food_loss_stats']}")
```

## Advanced Features

### Task Correlation Regularization
Controls the correlation between task representations:
```python
from loss_functions import TaskCorrelationRegularizer

correlation_reg = TaskCorrelationRegularizer(
    target_correlation=0.3,  # Desired correlation level
    lambda_reg=0.01         # Regularization strength
)

reg_loss = correlation_reg(food_features, cuisine_features)
```

### Loss History Tracking
All loss functions inherit history tracking:
```python
# Train for several epochs
for epoch in range(num_epochs):
    loss = loss_fn(predictions, targets)

# Get loss statistics
stats = loss_fn.get_loss_statistics()
print(f"Loss trend: {stats['mean']:.4f} ± {stats['std']:.4f}")
```

### Adaptation Monitoring
Monitor how adaptive losses change over training:
```python
adaptive_stats = adaptive_loss.get_adaptation_statistics()

# Task difficulty trends
for task in ['food', 'cuisine', 'nutrition']:
    trend = adaptive_stats[f'{task}_difficulty_trend']
    print(f"{task} difficulty trend: {trend:.4f}")

# Current dynamic weights
weights = adaptive_stats['current_dynamic_weights']
print(f"Dynamic task weights: {weights}")
```

## Integration Examples

### With Multi-Task Models
```python
from loss_functions import UncertaintyWeightedLoss

# Create loss function
loss_fn = UncertaintyWeightedLoss(num_tasks=3)

# Training loop
for batch in dataloader:
    # Forward pass
    food_logits, cuisine_logits, nutrition_values = model(images)
    predictions = (food_logits, cuisine_logits, nutrition_values)
    targets = (food_labels, cuisine_labels, nutrition_targets)
    
    # Calculate loss
    total_loss, breakdown = loss_fn(predictions, targets)
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
```

### Multi-Loss Training
```python
# Combine multiple loss types
uncertainty_loss = create_loss_function('uncertainty')
contrastive_loss = create_loss_function('contrastive')

for batch in dataloader:
    # Multi-task loss
    mt_loss, _ = uncertainty_loss(predictions, targets)
    
    # Contrastive loss on features
    features = model.get_features(images)
    contr_loss = contrastive_loss(features, food_labels)
    
    # Combined loss
    total_loss = mt_loss + 0.1 * contr_loss
```

### Progressive Loss Adaptation
```python
adaptive_loss = AdaptiveMultiTaskLoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        # Adaptive loss with epoch info
        total_loss, breakdown = adaptive_loss(
            predictions, targets, epoch=epoch
        )
        
        # Monitor adaptation every 10 epochs
        if epoch % 10 == 0:
            stats = adaptive_loss.get_adaptation_statistics()
            print(f"Epoch {epoch} adaptation: {stats}")
```

## Performance Characteristics

### Computational Complexity
| Loss Function | Forward Pass | Memory | Gradients |
|--------------|-------------|---------|-----------|
| Focal Loss | O(N×C) | Low | Standard |
| Uncertainty-Weighted | O(N×C×T) | Low | + uncertainty params |
| Gradient Harmonization | O(N×C×T×P) | High | + gradient computation |
| Adaptive | O(N×C×T) | Medium | + dynamic weights |
| Contrastive | O(N²×D) | Medium | Feature-dependent |

Where N=batch_size, C=num_classes, T=num_tasks, P=num_parameters, D=feature_dim.

### Training Recommendations
- **Start Simple**: Begin with uncertainty-weighted loss
- **Add Complexity**: Progress to adaptive loss if needed
- **Monitor Adaptation**: Use statistics to understand training dynamics
- **Tune Hyperparameters**: Adjust weights based on task importance

## Mathematical Details

### Uncertainty Weighting Derivation
The uncertainty-weighted loss assumes homoscedastic uncertainty and derives from maximum likelihood estimation:

```
p(y|f(x), σ) = N(y; f(x), σ²)
log p(y|f(x), σ) = -½log(2πσ²) - ½(y-f(x))²/σ²
L = -log p(y|f(x), σ) = ½log(σ²) + ½(y-f(x))²/σ²
L ≈ ½log(σ²) + L_task/2σ² ∝ L_task/2σ² + log(σ)
```

### Focal Loss Motivation
Standard cross-entropy: L_CE = -log(p_t)
Problem: Easy examples (high p_t) dominate training
Solution: Weight by difficulty: L_FL = -(1-p_t)^γ log(p_t)

### Gradient Conflict Detection
For tasks i,j with gradients g_i, g_j:
- Conflict when cos(g_i, g_j) < 0
- Harmonization penalty: Σ max(0, -cos(g_i, g_j))

## Research Foundation

Based on established research:
- **Kendall & Gal (2017)**: "Multi-Task Learning Using Uncertainty to Weigh Losses"
- **Lin et al. (2017)**: "Focal Loss for Dense Object Detection"
- **Chen et al. (2020)**: "A Simple Framework for Contrastive Learning"
- **Oord et al. (2018)**: "Representation Learning with Contrastive Predictive Coding"

## Best Practices

### Loss Selection Guidelines
1. **Uncertainty-Weighted**: Default choice for multi-task learning
2. **Focal Loss**: When dealing with severe class imbalance
3. **Adaptive**: For complex scenarios requiring sophisticated balancing
4. **Contrastive**: When feature quality is limiting performance

### Hyperparameter Tuning
- **Focal Loss γ**: Start with 2.0, reduce for less imbalance
- **Uncertainty Initialization**: Use 1.0 for balanced start
- **Correlation Target**: 0.3 encourages moderate task sharing
- **Adaptation Rate**: 0.01 for stable adaptation

### Monitoring and Debugging
- Track loss component trends over training
- Monitor uncertainty parameter evolution
- Analyze task correlation changes
- Use loss landscape analysis for insights

## Requirements

- PyTorch 2.0+
- NumPy
- Python 3.8+

## Performance Notes

- **Memory**: Gradient harmonization requires additional memory for gradient computation
- **Speed**: Uncertainty-weighted and adaptive losses have minimal overhead
- **Convergence**: Adaptive methods may require longer training for stabilization
- **Stability**: All implementations include numerical stability safeguards