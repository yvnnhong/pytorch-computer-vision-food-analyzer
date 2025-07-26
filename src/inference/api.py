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

# Import BOTH model architectures
from src.models.food_classifier import MultiTaskFoodModel, create_model
from src.models.resnet_multitask import create_resnet_multitask, AdvancedResNetMultiTask
from src.datasets.transforms import get_transforms

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
    title="Multi-Task Food Analyzer API with Advanced ResNet",
    description="AI-powered food classification, cuisine detection, and nutrition estimation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FoodAnalyzer:
    """Main class for food analysis functionality with model selection"""
    
    def __init__(self):
        self.models = {}  # Store multiple models
        self.current_model = None
        self.current_model_type = "basic"
        self.transform = None
        self.device = None
        self.model_loaded = False
        
    def load_model(self, 
                   model_type="basic",  # "basic", "advanced", "ensemble"
                   model_path="./models/best_model.pth"):
        """Load the selected multi-task model"""
        try:
            # Determine device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load class info
            try:
                with open('./data/cuisine_mappings.json', 'r') as f:
                    cuisine_mapping = json.load(f)
                num_cuisine_classes = len(set(cuisine_mapping.values()))
            except:
                num_cuisine_classes = 10
                logger.warning("Could not load cuisine mapping, using default 10 classes")
            
            # Create the selected model architecture
            if model_type == "basic":
                self.current_model = create_model(
                    num_cuisine_classes=num_cuisine_classes,
                    device=self.device
                )
                logger.info("Loaded BASIC MultiTaskFoodModel")
                
            elif model_type == "advanced":
                self.current_model = create_resnet_multitask(
                    architecture='advanced',
                    num_food_classes=101,
                    num_cuisine_classes=num_cuisine_classes,
                    backbone='resnet50',
                    pretrained=True
                ).to(self.device)
                logger.info("Loaded ADVANCED ResNet Multi-Task Model")
                
            elif model_type == "ensemble":
                self.current_model = create_resnet_multitask(
                    architecture='ensemble',
                    num_food_classes=101,
                    num_cuisine_classes=num_cuisine_classes
                ).to(self.device)
                logger.info("Loaded ENSEMBLE ResNet Multi-Task Model")
            
            # Store model in registry
            self.models[model_type] = self.current_model
            self.current_model_type = model_type
            
            # Load trained weights if available (only for basic model)
            if model_type == "basic" and os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # Handle different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    self.current_model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded trained weights from {model_path}")
                        
                except Exception as e:
                    logger.warning(f"Could not load trained weights: {e}")
                    logger.info("Using pre-trained ImageNet weights only")
            else:
                logger.info("Using pre-trained ImageNet weights for advanced models")
            
            # Set to evaluation mode
            self.current_model.eval()
            
            # Load transforms
            self.transform = get_transforms('val')
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully! Architecture: {model_type}")
            
            # Print model info
            total_params = sum(p.numel() for p in self.current_model.parameters())
            logger.info(f"Total parameters: {total_params:,}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def switch_model(self, model_type: str):
        """Switch between different model architectures"""
        if model_type in self.models:
            self.current_model = self.models[model_type]
            self.current_model_type = model_type
            logger.info(f"Switched to {model_type} model")
        else:
            # Load new model
            self.load_model(model_type=model_type)
    
    def analyze_image(self, image: Image.Image, return_attention=False):
        """
        Analyze a single food image with optional attention maps
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
            
            # Get prediction based on model type
            with torch.no_grad():
                if self.current_model_type == "basic":
                    # Use the predict_single_image method for basic model
                    prediction = self.current_model.predict_single_image(image_tensor.squeeze(0))
                    result_format = {
                        'food': prediction['food'],
                        'cuisine': prediction['cuisine'],
                        'nutrition': prediction['nutrition']
                    }
                else:
                    # For advanced models, use forward pass
                    food_logits, cuisine_logits, nutrition_values = self.current_model(image_tensor)
                    
                    # Get class predictions
                    food_probs = torch.softmax(food_logits, dim=1)
                    cuisine_probs = torch.softmax(cuisine_logits, dim=1)
                    
                    food_pred = torch.argmax(food_probs, dim=1).item()
                    cuisine_pred = torch.argmax(cuisine_probs, dim=1).item()
                    
                    food_confidence = food_probs[0, food_pred].item()
                    cuisine_confidence = cuisine_probs[0, cuisine_pred].item()
                    
                    # Format nutrition values
                    nutrition = nutrition_values[0].cpu().numpy()
                    nutrition = [max(0, val) for val in nutrition]
                    
                    result_format = {
                        'food': {
                            'class_name': f'food_class_{food_pred}',
                            'confidence': food_confidence
                        },
                        'cuisine': {
                            'class_name': f'cuisine_class_{cuisine_pred}',
                            'confidence': cuisine_confidence
                        },
                        'nutrition': {
                            'calories': float(round(nutrition[0], 1)),
                            'protein': float(round(nutrition[1], 1)),
                            'carbs': float(round(nutrition[2], 1)),
                            'fat': float(round(nutrition[3], 1))
                        }
                    }
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Get attention maps if requested and available
            attention_info = {}
            if return_attention and hasattr(self.current_model, 'get_attention_maps'):
                try:
                    attention_maps = self.current_model.get_attention_maps(image_tensor)
                    attention_info = {
                        'attention_available': True,
                        'attention_tasks': list(attention_maps.keys())
                    }
                except Exception as e:
                    logger.warning(f"Could not extract attention maps: {e}")
                    attention_info = {'attention_available': False}
            
            # Format response
            result = {
                "success": True,
                "predictions": {
                    "food": {
                        "class_name": result_format['food']['class_name'],
                        "confidence": round(result_format['food']['confidence'], 4)
                    },
                    "cuisine": {
                        "class_name": result_format['cuisine']['class_name'],
                        "confidence": round(result_format['cuisine']['confidence'], 4)
                    },
                    "nutrition": result_format['nutrition']
                },
                "metadata": {
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "model_type": self.current_model_type,
                    "model_device": str(self.device),
                    "image_size": image.size,
                    **attention_info
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Initialize the analyzer
analyzer = FoodAnalyzer()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Multi-Task Food Analyzer API with Advanced ResNet",
        "status": "running",
        "model_loaded": analyzer.model_loaded,
        "current_model": analyzer.current_model_type if analyzer.model_loaded else "none",
        "version": "2.0.0",
        "endpoints": {
            "analyze": "/analyze - POST with image file",
            "analyze_advanced": "/analyze/advanced - POST with attention maps",
            "switch_model": "/model/switch/{model_type} - Switch architecture",
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
        "current_model": analyzer.current_model_type if analyzer.model_loaded else "none",
        "device": str(analyzer.device) if analyzer.device else "unknown",
        "available_models": list(analyzer.models.keys()),
        "timestamp": time.time()
    }

@app.post("/analyze")
async def analyze_food(file: UploadFile = File(...)):
    """
    Analyze uploaded food image using current model
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
        
        logger.info(f"Successfully analyzed {file.filename} with {analyzer.current_model_type} model")
        return result
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.post("/analyze/advanced")
async def analyze_food_advanced(file: UploadFile = File(...)):
    """
    Analyze uploaded food image with attention maps (if available)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Analyze with attention maps
        result = analyzer.analyze_image(image, return_attention=True)
        
        result["metadata"]["filename"] = file.filename
        result["metadata"]["file_size_kb"] = round(len(image_data) / 1024, 2)
        
        logger.info(f"Advanced analysis complete for {file.filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error in advanced analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")

@app.post("/model/switch/{model_type}")
async def switch_model(model_type: str):
    """
    Switch between different model architectures
    """
    valid_types = ["basic", "advanced", "ensemble"]
    if model_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Choose from: {valid_types}")
    
    try:
        analyzer.switch_model(model_type)
        return {
            "message": f"Successfully switched to {model_type} model",
            "current_model": analyzer.current_model_type,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the current model"""
    if not analyzer.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get model parameters count
        total_params = sum(p.numel() for p in analyzer.current_model.parameters())
        trainable_params = sum(p.numel() for p in analyzer.current_model.parameters() if p.requires_grad)
        
        model_info = {
            "current_architecture": analyzer.current_model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "tasks": [
                "Food Classification (101 classes)",
                "Cuisine Classification (10+ classes)", 
                "Nutrition Regression (4 values)"
            ],
            "device": str(analyzer.device),
            "input_size": "224x224 RGB",
            "available_models": list(analyzer.models.keys())
        }
        
        # Add architecture-specific info
        if analyzer.current_model_type == "advanced":
            model_info["special_features"] = [
                "Task-specific attention mechanisms",
                "Cross-task feature fusion",
                "Multi-scale feature extraction",
                "Attention map visualization"
            ]
        
        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/analyze/batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    """
    Analyze multiple food images at once
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
    
    print("Starting Advanced Multi-Task Food Analyzer API...")
    print("Available Models: basic, advanced, ensemble")
    print("API Documentation: http://localhost:8000/docs")
    print("Switch models: POST /model/switch/{model_type}")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )