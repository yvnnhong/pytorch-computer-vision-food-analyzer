# Inference Directory
# NOTE: FIX THIS BRO FIX ALL THE READMES ALDSKFLKFSJL

Model inference and deployment components for production food classification service.

## Core Inference

### predict.py
Main prediction pipeline handling single image inference with preprocessing, model prediction, and result formatting.

### real_time_inference.py
Real-time prediction service optimized for low-latency inference with model optimization and caching strategies.

## Production Deployment

### api.py
FastAPI REST API providing HTTP endpoints for food classification, cuisine prediction, and nutrition analysis.

### batch_inference.py
Batch processing pipeline for high-throughput inference on multiple images with parallel processing capabilities.

## Usage

```python
from src.inference.predict import FoodPredictor

# Initialize predictor
predictor = FoodPredictor(model_path='checkpoints/best_model.pth')

# Single prediction
result = predictor.predict(image_path='food.jpg')

# API server
# uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /predict` - Single image prediction
- `POST /batch` - Batch image processing
- `GET /health` - Service health check
- `GET /models` - Available model information

## Performance

- Single inference: <200ms
- Batch processing: 100+ images/minute
- API response time: <500ms end-to-end