# Ensemble Methods for Multi-Task Food Classification

Advanced ensemble learning implementations for improving model robustness and performance through strategic model combination.

## Overview

This module provides production-ready ensemble methods that combine multiple neural network models to achieve superior performance on multi-task food classification. The implementations are mathematically rigorous and optimized for real-world deployment scenarios.

## Mathematical Foundation

### Core Ensemble Formulations

**Weighted Ensemble**:
```
f_ensemble(x) = Σᵢ wᵢ * fᵢ(x) where Σwᵢ = 1
```

**Stacked Ensemble**:
```
f_stack(x) = g(f₁(x), f₂(x), ..., fₙ(x))
```

**Mixture of Experts**:
```
P(y|x) = Σᵢ gᵢ(x) * fᵢ(x) where gᵢ is gating function
```

**Uncertainty Quantification**:
```
H(y|x) = -Σ p(y|x) log p(y|x)  (Predictive Entropy)
I(y;θ|x,D) = H(y|x,D) - E[H(y|x,θ)]  (Mutual Information)
```

## Module Structure

```
ensemble_methods/
├── __init__.py                # Module interface and exports
├── weighted_ensemble.py       # Linear combination with weights
├── stacked_ensemble.py        # Meta-learner combination
├── adaptive_ensemble.py       # Input-dependent gating
├── mixture_experts.py         # Expert specialization
├── uncertainty_utils.py       # Uncertainty quantification
└── ensemble_factory.py        # Factory and configuration
```

## Quick Start

### Basic Ensemble Creation
```python
from ensemble_methods import create_ensemble, EnsembleFactory

# Create weighted ensemble
models = [model1, model2, model3]
ensemble = create_ensemble('weighted', models)

# Forward pass
food_logits, cuisine_logits, nutrition_values, metadata = ensemble(images)
```

### Advanced Configuration
```python
from ensemble_methods import EnsembleConfig, EnsembleFactory

# Configure ensemble
config = EnsembleConfig(
    model_weights=[0.4, 0.3, 0.3],
    combination_method='weighted',
    temperature_scaling=True,
    uncertainty_estimation=True
)

# Create configured ensemble
ensemble = EnsembleFactory.create_ensemble('weighted', models, config)
```

## Available Ensemble Types

### 1. Weighted Ensemble
**Purpose**: Linear combination of model predictions with learnable or fixed weights.

**Key Features**:
- Temperature scaling for calibration
- Learnable weight optimization
- Confidence-based metadata
- Uncertainty quantification

**Usage**:
```python
from ensemble_methods import WeightedEnsemble

ensemble = WeightedEnsemble(
    models=models,
    weights=[0.4, 0.3, 0.3],
    learnable_weights=True,
    temperature_scaling=True
)
```

**Best For**: Simple, interpretable model combination with proven effectiveness.

### 2. Stacked Ensemble
**Purpose**: Meta-learner that learns optimal combination of base model predictions.

**Architecture**:
- **Level 0**: Base models (ResNet, Custom CNN, etc.)
- **Level 1**: Meta-learner networks for each task

**Key Features**:
- Task-specific meta-learners
- End-to-end trainable
- Optimal combination learning
- High expressiveness

**Usage**:
```python
from ensemble_methods import StackedEnsemble

ensemble = StackedEnsemble(
    base_models=models,
    meta_learner_hidden_dim=128,
    num_food_classes=101,
    num_cuisine_classes=13
)
```

**Best For**: Maximum performance when training data is abundant.

### 3. Adaptive Ensemble
**Purpose**: Input-dependent model selection through gating networks.

**Key Features**:
- Task-specific gating
- Input-dependent weighting
- Dynamic model selection
- Lightweight feature extraction

**Usage**:
```python
from ensemble_methods import AdaptiveEnsemble

ensemble = AdaptiveEnsemble(
    models=models,
    feature_extractor_dim=512,
    gating_hidden_dim=64
)
```

**Best For**: Scenarios where different models excel on different input types.

### 4. Mixture of Experts
**Purpose**: Expert specialization with confidence-weighted gating.

**Key Features**:
- Expert specialization tracking
- Confidence-based weighting
- Interpretable expert selection
- Specialization metadata

**Usage**:
```python
from ensemble_methods import MixtureOfExperts

ensemble = MixtureOfExperts(
    expert_models=models,
    expert_specializations=['texture', 'color', 'structure'],
    gating_dim=256
)
```

**Best For**: Domain-specific expertise and interpretable model selection.

## Uncertainty Quantification

### ModelUncertainty Class
Provides comprehensive uncertainty estimation for ensemble predictions.

#### Predictive Entropy
```python
from ensemble_methods import ModelUncertainty

probabilities = torch.softmax(logits, dim=1)
entropy = ModelUncertainty.predictive_entropy(probabilities)
```

#### Mutual Information
```python
model_predictions = [model1_probs, model2_probs, model3_probs]
mutual_info = ModelUncertainty.mutual_information(model_predictions)
```

#### Prediction Variance
```python
predictions = [model1_preds, model2_preds, model3_preds]
variance = ModelUncertainty.prediction_variance(predictions)
```

### Diversity Evaluation
```python
from ensemble_methods import evaluate_ensemble_diversity

diversity_metrics = evaluate_ensemble_diversity(models, dataloader)
print(f"Mean disagreement: {diversity_metrics['mean_pairwise_disagreement']:.3f}")
```

## Factory Pattern

### Quick Creation
```python
from ensemble_methods import EnsembleFactory

# Create any ensemble type
ensemble = EnsembleFactory.create_ensemble(
    ensemble_type='adaptive',
    models=models,
    feature_dim=512
)
```

### Diverse Architecture Ensemble
```python
# Automatically creates ensemble with diverse architectures
diverse_ensemble = EnsembleFactory.create_diverse_ensemble(
    num_food_classes=101,
    num_cuisine_classes=13,
    ensemble_type='mixture'
)
```

## Performance Characteristics

### Computational Complexity
| Ensemble Type | Training Overhead | Inference Speed | Memory Usage |
|---------------|------------------|-----------------|--------------|
| Weighted | Low | Fast | 1x base models |
| Stacked | High | Fast | 1x + meta-learner |
| Adaptive | Medium | Medium | 1x + gating network |
| Mixture | Medium | Medium | 1x + expert gating |

### Accuracy vs Efficiency Trade-offs
- **Weighted**: Best efficiency, good accuracy improvement
- **Stacked**: Highest accuracy, moderate efficiency
- **Adaptive**: Balanced accuracy/efficiency, input-adaptive
- **Mixture**: Interpretable, specialized performance

## Advanced Features

### Temperature Scaling
Improves probability calibration for better confidence estimates:
```python
ensemble = WeightedEnsemble(models, temperature_scaling=True)
# Automatically learns optimal temperature parameters
```

### Learnable Weights
Optimizes ensemble weights during training:
```python
ensemble = WeightedEnsemble(models, learnable_weights=True)
# Weights updated via gradient descent
```

### Task-Specific Gating
Different model combinations for different tasks:
```python
adaptive_ensemble = AdaptiveEnsemble(models)
# Separate gating networks for food/cuisine/nutrition tasks
```

## Integration Examples

### With Custom CNN Models
```python
from custom_cnn import create_food_cnn
from ensemble_methods import create_ensemble

# Create diverse base models
models = [
    create_food_cnn('standard'),
    create_food_cnn('efficient'), 
    create_food_cnn('mobile')
]

# Create ensemble
ensemble = create_ensemble('weighted', models)
```

### With Attention Mechanisms
```python
from attention_layers import CBAM
from ensemble_methods import WeightedEnsemble

# Models with different attention mechanisms
models_with_attention = [base_model + attention for base_model in models]
ensemble = WeightedEnsemble(models_with_attention)
```

### Multi-Task Training
```python
# All ensembles support multi-task learning
food_logits, cuisine_logits, nutrition_values = ensemble(images)

# Task-specific losses
food_loss = F.cross_entropy(food_logits, food_labels)
cuisine_loss = F.cross_entropy(cuisine_logits, cuisine_labels)
nutrition_loss = F.mse_loss(nutrition_values, nutrition_targets)

total_loss = food_loss + cuisine_loss + nutrition_loss
```

## Configuration and Customization

### EnsembleConfig
```python
from ensemble_methods import EnsembleConfig

config = EnsembleConfig(
    model_weights=[0.4, 0.3, 0.3],
    combination_method='weighted',
    confidence_threshold=0.8,
    diversity_weight=0.1,
    temperature_scaling=True,
    uncertainty_estimation=True
)
```

### Custom Ensemble Creation
```python
# Extend base classes for custom behavior
class CustomEnsemble(WeightedEnsemble):
    def __init__(self, models, **kwargs):
        super().__init__(models, **kwargs)
        # Add custom functionality
        
    def forward(self, x):
        # Custom forward logic
        return super().forward(x)
```

## Best Practices

### Model Diversity
- **Architecture Diversity**: Combine CNN, ResNet, Transformer models
- **Training Diversity**: Different initialization, augmentation, hyperparameters
- **Data Diversity**: Train on different subsets or cross-validation folds

### Ensemble Selection
- **Weighted**: Start here for simplicity and interpretability
- **Stacked**: When you have abundant training data and need maximum performance
- **Adaptive**: When different models excel on different input types
- **Mixture**: When you need interpretable expert specialization

### Performance Optimization
- **Model Pruning**: Remove redundant ensemble members
- **Knowledge Distillation**: Train single model to mimic ensemble
- **Quantization**: Reduce precision for deployment efficiency

## Evaluation Metrics

### Ensemble-Specific Metrics
```python
# Diversity metrics
diversity_metrics = evaluate_ensemble_diversity(models, dataloader)

# Uncertainty calibration
reliability_diagram = plot_reliability_diagram(predictions, confidences)

# Individual model contribution
ablation_results = ablation_study(ensemble, test_data)
```

### Multi-Task Evaluation
```python
# Task-specific performance
food_accuracy = calculate_accuracy(food_predictions, food_labels)
cuisine_accuracy = calculate_accuracy(cuisine_predictions, cuisine_labels)
nutrition_mae = calculate_mae(nutrition_predictions, nutrition_values)

# Overall ensemble benefit
ensemble_improvement = ensemble_score - best_individual_score
```

## Research Foundation

Based on established ensemble learning principles:
- **Bootstrap Aggregating (Bagging)**: Variance reduction through averaging
- **Boosting**: Bias reduction through sequential learning
- **Stacking**: Optimal combination learning via meta-models
- **Mixture of Experts**: Specialized model selection and gating

## Production Deployment

### Model Serving
```python
# Ensemble inference pipeline
def ensemble_predict(image_batch):
    with torch.no_grad():
        food_logits, cuisine_logits, nutrition_values, metadata = ensemble(image_batch)
    
    return {
        'food_predictions': torch.argmax(food_logits, dim=1),
        'cuisine_predictions': torch.argmax(cuisine_logits, dim=1), 
        'nutrition_estimates': nutrition_values,
        'confidence_scores': metadata['model_confidences'],
        'uncertainty_estimates': metadata['food_uncertainty']
    }
```

### Monitoring and Maintenance
- **Performance Tracking**: Monitor ensemble vs individual model performance
- **Drift Detection**: Track prediction diversity and confidence over time
- **Model Updates**: Add/remove ensemble members based on performance
- **A/B Testing**: Compare ensemble variants in production

## Requirements

- PyTorch 2.0+
- NumPy
- Python 3.8+

## Performance Notes

- **Training**: Ensemble training is computationally intensive but parallelizable
- **Inference**: Real-time capable for weighted/adaptive ensembles
- **Memory**: Scales linearly with number of base models
- **Accuracy**: Typically 2-5% improvement over best individual model