# Evaluation Directory

Model evaluation and analysis tools for multi-task food classification system.

## Performance Metrics

### metrics.py
Multi-task evaluation metrics including accuracy, F1-score, MAE for regression, and task-specific performance measures.

### model_comparison.py
Architecture comparison utilities for benchmarking different CNN models and ensemble techniques.

## Analysis Tools

### error_analysis.py
Systematic error analysis identifying failure cases, misclassification patterns, and performance bottlenecks.

### task_correlation.py
Inter-task relationship analysis measuring correlation between food classification, cuisine prediction, and nutrition regression.

### interpretability.py
Model interpretation tools including feature visualization, attention maps, and prediction explanations.

## Visualization

### visualizations.py
Result visualization including confusion matrices, learning curves, prediction distributions, and performance plots.

## Usage

```python
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualizations import plot_results

# Evaluate model performance
results = evaluate_model(model, test_loader)

# Generate visualization reports
plot_results(results, save_dir='results/')
```

## Output Formats

- Metrics saved as JSON and CSV
- Plots exported as PNG/PDF
- Analysis reports in markdown format