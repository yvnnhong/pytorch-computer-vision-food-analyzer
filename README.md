# Multi-Task Food Classification System

A production-ready computer vision system for food image analysis using PyTorch. Combines food classification, cuisine detection, and nutrition estimation through advanced multi-task learning architectures.

## Features

- **Multi-task CNN architectures** with shared backbones and task-specific heads
- **Advanced attention mechanisms** including channel, spatial, and cross-modal attention
- **Multiple model variants** optimized for different deployment scenarios
- **Ensemble methods** with weighted combination, stacking, and mixture of experts
- **Real-time inference API** with FastAPI backend
- **Interactive web interface** using Gradio
- **Production utilities** for model management, benchmarking, and optimization

## Architecture Overview

### Core Models
- **Basic Multi-Task Model**: ResNet50 backbone with task-specific heads
- **Advanced ResNet**: Task-specific attention with cross-task feature fusion
- **Custom Food CNN**: Domain-specific architecture for food image analysis
- **Mobile-Optimized**: Depthwise separable convolutions for edge deployment
- **Ensemble Methods**: Weighted, stacked, adaptive, and mixture of experts

### Tasks
1. **Food Classification**: 101 food categories (Food-101 dataset)
2. **Cuisine Classification**: 13 regional cuisine types
3. **Nutrition Regression**: Calories, protein, carbohydrates, fat estimation

## Project Structure

```
pytorch-computer-vision-food-analyzer/
├── src/
│   ├── models/                    # PyTorch model architectures
│   │   ├── food_classifier.py     # Basic multi-task model
│   │   ├── resnet_multitask.py    # Advanced ResNet with attention
│   │   ├── custom_cnn.py          # Food-specific CNN architectures
│   │   ├── ensemble_model.py      # Ensemble methods and combinations
│   │   ├── attention_layers.py    # Attention mechanism implementations
│   │   ├── loss_functions.py      # Multi-task loss functions
│   │   └── model_utils.py         # Model management and benchmarking
│   ├── datasets/                  # Data handling and preprocessing
│   │   ├── dataset.py             # Multi-task dataset class
│   │   ├── data_loaders.py        # DataLoader configurations
│   │   ├── transforms.py          # Image preprocessing pipeline
│   │   └── preprocessing.py       # Data preparation utilities
│   ├── training/                  # Training pipeline components
│   │   ├── train.py               # Main training script
│   │   └── trainer.py             # Training orchestration
│   ├── evaluation/                # Model evaluation and analysis
│   ├── inference/                 # Deployment and serving
│   │   └── api.py                 # FastAPI REST API
│   └── utils/                     # Configuration and utilities
├── web_demo/
│   └── gradio_app.py              # Interactive web interface
├── data/                          # Dataset storage (excluded from repo)
├── models/                        # Model checkpoints (excluded from repo)
└── logs/                          # Training logs and metrics
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

**Windows 11:**
```cmd
git clone https://github.com/yvnnhong/pytorch-computer-vision-food-analyzer.git
cd pytorch-computer-vision-food-analyzer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS:**
```bash
git clone https://github.com/yvnnhong/pytorch-computer-vision-food-analyzer.git
cd pytorch-computer-vision-food-analyzer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Setup

Download and prepare the Food-101 dataset:
```bash
python src/datasets/preprocessing.py
```

## Usage

### Training

Train the multi-task model:
```bash
python src/training/train.py
```

### Deployment

The system requires two components running simultaneously:

#### Terminal 1: Start the API Backend

**Windows 11:**
```cmd
.venv\Scripts\activate
python src/inference/api.py
```

**macOS:**
```bash
source .venv/bin/activate
python src/inference/api.py
```

The API will be available at `http://localhost:8000`

#### Terminal 2: Start the Web Interface

**Windows 11:**
```cmd
.venv\Scripts\activate
python web_demo/gradio_app.py
```

**macOS:**
```bash
source .venv/bin/activate
python web_demo/gradio_app.py
```

The web interface will be available at `http://localhost:7860`

### API Usage

```python
import requests

# Analyze food image
with open('food_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze',
        files={'file': f}
    )
    
result = response.json()
print(f"Food: {result['predictions']['food']['class_name']}")
print(f"Cuisine: {result['predictions']['cuisine']['class_name']}")
print(f"Calories: {result['predictions']['nutrition']['calories']}")
```

## Model Performance

### Current Results
- **Nutrition Regression**: MAE of 24.4 (significant improvement from baseline)
- **Cuisine Classification**: 22-32% accuracy across 13 classes
- **Food Classification**: Limited by training data constraints

### Performance Limitations

The current system has several known limitations:

#### Training Data Constraints
- **Class Mismatch**: Model architecture supports 101 classes, dataset contains 126 classes
- **Insufficient Samples**: Only 1-3 training samples per food class in current subset
- **Data Imbalance**: Severe underrepresentation across food categories

#### Impact on Results
- Food classification performance significantly limited by sample size
- Model may misclassify visually similar foods due to inadequate training examples
- Confidence scores may be lower than optimal for food classification task

## Advanced Features

### Attention Mechanisms
- **Channel Attention**: Emphasizes important feature channels
- **Spatial Attention**: Focuses on relevant spatial regions
- **Cross-Modal Attention**: Enables information sharing between tasks
- **Food-Specific Attention**: Domain-adapted attention for food characteristics

### Ensemble Methods
- **Weighted Ensemble**: Learnable combination weights with temperature scaling
- **Stacked Ensemble**: Meta-learner for optimal model combination
- **Adaptive Ensemble**: Input-dependent model selection via gating networks
- **Mixture of Experts**: Specialized models with confidence-based weighting

### Model Utilities
- **Performance Benchmarking**: Speed, memory, and throughput analysis
- **Model Comparison**: Architecture evaluation and selection tools
- **Quantization Support**: Model compression for mobile deployment
- **Uncertainty Estimation**: Predictive entropy and mutual information

### Current Performance Limitations

**Food Classification Issues:**
- Model expects 101 classes but dataset contains 126 classes
- Training subset limited to 200 samples total (1.6 samples per class average)
- Results in low confidence predictions (7-15%) and frequent misclassification
- Example: Sushi images may be classified as "strawberry_shortcake"

**Root Cause Analysis:**
- Severe class imbalance: 126 food categories with only 1-3 training examples each
- Insufficient data for deep learning model to learn discriminative features
- Food-101 subset inadequate for production-quality classification

**Successful Components:**
- Nutrition regression: Achieved 78% improvement (MAE 107→24)
- Cuisine classification: 22-32% accuracy acceptable for 13-class problem
- Multi-task learning architecture: Proven effective for shared feature learning

## Future Improvements

### Data Enhancement
- Expand training dataset beyond Food-101 subset
- Implement advanced data augmentation strategies
- Balance class distribution through strategic sampling
- Consider self-supervised pre-training on larger food datasets

### Architecture Improvements
- Integration of advanced models (ResNet with attention, custom CNN) into production API
- Dynamic model selection based on input characteristics
- Ensemble deployment for improved robustness

### Production Optimization
- Model quantization for edge deployment
- Caching and batch processing for high-throughput scenarios
- A/B testing framework for model comparison

## Technical Specifications

### Model Architectures
- **Parameters**: 26-30M (basic), 25-60M (ensemble)
- **Input Size**: 224×224 RGB images
- **Inference Time**: <300ms per image (CPU)
- **Memory Usage**: <4GB during training

### Dependencies
- PyTorch 2.0+
- torchvision
- FastAPI
- Gradio
- NumPy, Pandas
- PIL, OpenCV


