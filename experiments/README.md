# Experiments Directory

ML research and experimentation scripts for model optimization and analysis.

## Architecture Research

### architecture_comparison.py
Systematic comparison of CNN architectures including ResNet, EfficientNet, and custom models with performance benchmarking.

### transfer_learning.py
Transfer learning experiments evaluating different pre-trained backbones and fine-tuning strategies for food classification.

## Model Optimization

### hyperparameter_tuning.py
Automated hyperparameter search using grid search, random search, and Bayesian optimization techniques.

### ablation_studies.py
Component ablation analysis measuring the impact of individual model components on overall performance.

## Usage

```python
# Compare architectures
python experiments/architecture_comparison.py --models resnet50,efficientnet_b0

# Hyperparameter tuning
python experiments/hyperparameter_tuning.py --method bayesian --trials 50

# Ablation study
python experiments/ablation_studies.py --component attention
```

## Output

- Experiment results saved to `results/experiments/`
- Performance metrics in JSON format
- Visualization plots and comparison tables
- Best configuration recommendations