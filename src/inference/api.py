from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
from PIL import Image
import io
import sys
import os
from pathlib import Path
import json
import time
import logging

# Add src to path for imports
sys.path.append('.')
sys.path.append('./src')

from src.models.food_classifier import MultiTaskFoodModel, create_model
from src.datasets.dataset import get_transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the analyzer globally
analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global analyzer
    logger.info("Starting Food Analyzer API...")
    analyzer = FoodAnalyzer()
    try:
        analyzer.load_model()
        logger.info("API ready for food analysis!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down Food Analyzer API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Multi-Task Food Analyzer API",
    description="AI-powered food classification, cuisine detection, and nutrition estimation",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc",  # Alternative docs at /redoc
    lifespan=lifespan
)

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and transforms
model = None
transform = None
device = None

class FoodAnalyzer:
    """Main class for food analysis functionality"""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None
        self.model_loaded = False
        
    def load_model(self, model_path="./models/best_model.pth"):
        """Load the trained multi-task model"""
        try:
            # Determine device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load class info to get number of cuisine classes
            try:
                with open('./data/cuisine_mappings.json', 'r') as f:
                    cuisine_mapping = json.load(f)
                num_cuisine_classes = len(set(cuisine_mapping.values()))
            except:
                num_cuisine_classes = 10  # Default fallback
                logger.warning("Could not load cuisine mapping, using default 10 classes")
            
            # Create model architecture
            self.model = create_model(
                num_cuisine_classes=num_cuisine_classes,
                device=self.device
            )
            
            # Load trained weights if available
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded trained model from {model_path}")
            else:
                logger.warning(f"Model file {model_path} not found, using untrained model")
            
            # Set to evaluation mode
            self.model.eval()
            
            # Load transforms
            self.transform = get_transforms('val')  # Use validation transforms (no augmentation)
            
            self.model_loaded = True
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def analyze_image(self, image: Image.Image):
        """
        Analyze a single food image
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Analysis results with food, cuisine, and nutrition predictions
        """
        if not self.model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            start_time = time.time()
            
            # Preprocess image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                prediction = self.model.predict_single_image(image_tensor.squeeze(0))
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Format response
            result = {
                "success": True,
                "predictions": {
                    "food": {
                        "class_name": prediction['food']['class_name'],
                        "confidence": round(prediction['food']['confidence'], 4)
                    },
                    "cuisine": {
                        "class_name": prediction['cuisine']['class_name'],
                        "confidence": round(prediction['cuisine']['confidence'], 4)
                    },
                    "nutrition": {
                        "calories": float(round(prediction['nutrition']['calories'], 1)),
                        "protein": float(round(prediction['nutrition']['protein'], 1)),
                        "carbs": float(round(prediction['nutrition']['carbs'], 1)),
                        "fat": float(round(prediction['nutrition']['fat'], 1))
                    }
                },
                "metadata": {
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "model_device": str(self.device),
                    "image_size": image.size
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Initialize the analyzer will be done in lifespan
# analyzer = FoodAnalyzer()  # Moved to lifespan function



@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Multi-Task Food Analyzer API",
        "status": "running",
        "model_loaded": analyzer.model_loaded,
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze - POST with image file",
            "health": "/health - GET health status",
            "docs": "/docs - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if analyzer.model_loaded else "degraded",
        "model_loaded": analyzer.model_loaded,
        "device": str(analyzer.device) if analyzer.device else "unknown",
        "timestamp": time.time()
    }

@app.post("/analyze")
async def analyze_food(file: UploadFile = File(...)):
    """
    Analyze uploaded food image
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
        
    Returns:
        JSON with food classification, cuisine type, and nutrition estimates
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Analyze image
        result = analyzer.analyze_image(image)
        
        # Add file info to response
        result["metadata"]["filename"] = file.filename
        result["metadata"]["file_size_kb"] = round(len(image_data) / 1024, 2)
        
        logger.info(f"Successfully analyzed {file.filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if not analyzer.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get model parameters count
        total_params = sum(p.numel() for p in analyzer.model.parameters())
        trainable_params = sum(p.numel() for p in analyzer.model.parameters() if p.requires_grad)
        
        return {
            "model_architecture": "Multi-Task CNN (ResNet50 backbone)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "tasks": [
                "Food Classification (101 classes)",
                "Cuisine Classification (10+ classes)", 
                "Nutrition Regression (4 values)"
            ],
            "device": str(analyzer.device),
            "input_size": "224x224 RGB"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/analyze/batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    """
    Analyze multiple food images at once
    
    Args:
        files: List of uploaded image files
        
    Returns:
        JSON with analysis results for each image
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue
            
            # Read and analyze image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            result = analyzer.analyze_image(image)
            result["metadata"]["filename"] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_images": len(files),
        "successful_analyses": sum(1 for r in results if r.get("success", False))
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Multi-Task Food Analyzer API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )