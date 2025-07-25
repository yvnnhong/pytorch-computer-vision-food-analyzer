# Training Directory

PyTorch training pipeline components for multi-task food classification models.

## Core Training

### train.py
Main training script with command-line interface for model training and experiment execution.

### trainer.py
Training orchestration class handling epoch loops, loss computation, and model checkpointing.

### validation.py
Validation loop implementation with multi-task metrics computation and model evaluation.

## Training Configuration

### hyperparameters.py
Hyperparameter configuration management including learning rates, batch sizes, and optimization settings.

### lr_scheduler.py
Learning rate scheduling strategies including step decay, cosine annealing, and warmup schedules.

### early_stopping.py
Early stopping implementation with patience-based training termination and best model preservation.

## Multi-Task Training

### multitask_trainer.py
Specialized trainer for coordinating multi-task learning with task balancing and loss weighting strategies.

## Usage

```python
from src.training.trainer import Trainer
from src.training.hyperparameters import get_config

# Initialize trainer
config = get_config()
trainer = Trainer(model, train_loader, val_loader, config)

# Start training
trainer.train(epochs=100)
```

## Training Requirements

- GPU recommended (8GB+ VRAM for full dataset)
- Mixed precision training supported
- Automatic checkpoint saving and resumption