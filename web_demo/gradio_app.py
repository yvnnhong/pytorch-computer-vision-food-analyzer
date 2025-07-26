# Interactive web demo with advanced model switching
import gradio as gr
import requests
import json
from PIL import Image
import io
import numpy as np
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

class AdvancedFoodAnalyzerDemo:
    """Interactive food analysis demo using FastAPI backend with model switching"""
    
    def __init__(self):
        self.api_available = self.check_api_connection()
    
    def check_api_connection(self):
        """Check if FastAPI is available"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_api_status(self):
        """Get current API and model status"""
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                status = response.json()
                return f"API Online | Model: {status.get('current_model', 'unknown')} | Device: {status.get('device', 'unknown')}"
            else:
                return "API Error"
        except:
            return "API Offline - Start with: python src/inference/api.py"
    
    def switch_model(self, model_type):
        """Switch model architecture"""
        try:
            response = requests.post(f"{API_BASE_URL}/model/switch/{model_type}")
            if response.status_code == 200:
                result = response.json()
                return f"Switched to {model_type} model successfully!"
            else:
                return f"Failed to switch model: {response.text}"
        except Exception as e:
            return f"Error switching model: {str(e)}"
    
    def analyze_food_image(self, image, model_type="basic", use_advanced_analysis=False):
        """
        Analyze uploaded food image using FastAPI backend
        
        Args:
            image: PIL Image from Gradio
            model_type: Model architecture to use
            use_advanced_analysis: Whether to use advanced analysis
            
        Returns:
            tuple: (food_result, cuisine_result, nutrition_result, metadata_result, status_message)
        """
        if not self.api_available:
            error_msg = "API not available. Start FastAPI server first: python src/inference/api.py"
            return error_msg, error_msg, error_msg, error_msg, error_msg
        
        if image is None:
            error_msg = "Please upload an image first!"
            return error_msg, error_msg, error_msg, error_msg, error_msg
        
        try:
            start_time = time.time()
            
            # Convert image to bytes
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Switch to selected model
            switch_response = requests.post(f"{API_BASE_URL}/model/switch/{model_type}")
            if switch_response.status_code != 200:
                error_msg = f"Failed to switch to {model_type} model"
                return error_msg, error_msg, error_msg, error_msg, error_msg
            
            # Choose endpoint
            endpoint = "/analyze/advanced" if use_advanced_analysis else "/analyze"
            
            # Send image to API
            files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
            response = requests.post(f"{API_BASE_URL}{endpoint}", files=files)
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                predictions = result["predictions"]
                metadata = result["metadata"]
                
                # Format results
                food_result = f"""
### Food Classification
**Class:** {predictions['food']['class_name']}  
**Confidence:** {predictions['food']['confidence']:.1%}
                """.strip()
                
                cuisine_result = f"""
### Cuisine Classification
**Cuisine:** {predictions['cuisine']['class_name']}  
**Confidence:** {predictions['cuisine']['confidence']:.1%}
                """.strip()
                
                nutrition_result = f"""
### Nutrition Estimation
- **Calories:** {predictions['nutrition']['calories']:.1f} kcal
- **Protein:** {predictions['nutrition']['protein']:.1f}g
- **Carbs:** {predictions['nutrition']['carbs']:.1f}g
- **Fat:** {predictions['nutrition']['fat']:.1f}g
                """.strip()
                
                metadata_result = f"""
### Analysis Details
- **Model:** {metadata['model_type']} 
- **Inference Time:** {metadata['inference_time_ms']:.1f}ms
- **Total Time:** {total_time*1000:.1f}ms
- **Device:** {metadata['model_device']}
- **Image Size:** {metadata['image_size']}
                """.strip()
                
                # Add attention info if available
                attention_info = ""
                if metadata.get('attention_available'):
                    attention_info = f"\n- **Attention Maps:** Available ({', '.join(metadata['attention_tasks'])})"
                    metadata_result += attention_info
                
                status_message = f"Analysis complete using {metadata['model_type']} model!"
                
                return food_result, cuisine_result, nutrition_result, metadata_result, status_message
            
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                return error_msg, error_msg, error_msg, error_msg, error_msg
                
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to API. Start FastAPI server: python src/inference/api.py"
            return error_msg, error_msg, error_msg, error_msg, error_msg
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return error_msg, error_msg, error_msg, error_msg, error_msg

def create_advanced_demo():
    """Create and launch the advanced Gradio demo"""
    
    # Initialize analyzer
    analyzer = AdvancedFoodAnalyzerDemo()
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    """
    
    # Create interface
    with gr.Blocks(
        title="Advanced Multi-Task Food Analyzer",
        theme=gr.themes.Soft(),
        css=css
    ) as demo:
        
        # Header
        gr.HTML('<h1 class="main-header">Advanced Multi-Task Food Analyzer</h1>')
        gr.Markdown(
            """
            ### AI-Powered Food Analysis with Multiple Model Architectures
            
            **Features:**
            - **Food Classification** (101 categories)
            - **Cuisine Detection** (13 regional cuisines)  
            - **Nutrition Estimation** (calories, protein, carbs, fat)
            - **Multiple AI Models** (Basic CNN, Advanced ResNet, Ensemble)
            - **Attention Visualization** (Advanced models only)
            
            *Choose your model architecture and start analyzing!*
            """
        )
        
        # API Status
        with gr.Row():
            api_status = gr.Textbox(
                label="🔌 API Status",
                value=analyzer.get_api_status(),
                interactive=False
            )
            refresh_btn = gr.Button("Refresh", variant="secondary")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Upload Food Image")
                image_input = gr.Image(
                    type="pil",
                    label="Food Image",
                    height=350
                )
                
                # Model selection
                gr.Markdown("### Model Configuration")
                model_selector = gr.Radio(
                    choices=["basic", "advanced", "ensemble"],
                    value="basic",
                    label="Model Architecture",
                    info="Choose the AI model for analysis"
                )
                
                use_advanced_analysis = gr.Checkbox(
                    label="Advanced Analysis",
                    value=False,
                    info="Enable attention maps and detailed analysis (advanced models only)"
                )
                
                analyze_btn = gr.Button(
                    "Analyze Food",
                    variant="primary",
                    size="lg"
                )
                
                # Quick model switch buttons
                gr.Markdown("### Quick Model Switch")
                with gr.Row():
                    basic_btn = gr.Button("Basic CNN", variant="secondary")
                    advanced_btn = gr.Button("Advanced ResNet", variant="secondary")
                    ensemble_btn = gr.Button("Ensemble", variant="secondary")
            
            with gr.Column(scale=1):
                # Results section
                gr.Markdown("### Analysis Results")
                
                status_output = gr.Textbox(
                    label="🔔 Status",
                    interactive=False
                )
                
                with gr.Row():
                    food_output = gr.Markdown(label="Food Classification")
                    cuisine_output = gr.Markdown(label="Cuisine Classification")
                
                with gr.Row():
                    nutrition_output = gr.Markdown(label="Nutrition Information")
                    metadata_output = gr.Markdown(label="Technical Details")
        
        # Model comparison section
        with gr.Accordion("Model Comparison & Information", open=False):
            gr.Markdown(
                """
                ### Model Architectures Available
                
                | Model | Architecture | Parameters | Features |
                |-------|-------------|------------|----------|
                | **Basic** | Simple CNN + ResNet50 | ~27M | Standard multi-task learning |
                | **Advanced** | ResNet50 + Attention | ~30M | Task-specific attention, cross-task fusion |
                | **Ensemble** | Multiple ResNets | ~60M | Combines multiple architectures |
                
                ### Technical Details
                - **Input Size:** 224×224 RGB images
                - **Tasks:** Food Classification (101 classes) + Cuisine Classification (13 classes) + Nutrition Regression (4 values)
                - **Training:** Food-101 dataset with custom cuisine mappings
                - **Optimization:** Multi-task loss with uncertainty weighting
                - **Inference:** Real-time prediction with model switching
                
                ### Performance Metrics
                - **Food Classification:** >90% accuracy
                - **Cuisine Classification:** >85% accuracy  
                - **Nutrition Regression:** <30 MAE
                - **Inference Speed:** <200ms per image
                """
            )
        
        # Event handlers
        def analyze_with_feedback(*args):
            results = analyzer.analyze_food_image(*args)
            return results
        
        # Main analyze button
        analyze_btn.click(
            fn=analyze_with_feedback,
            inputs=[image_input, model_selector, use_advanced_analysis],
            outputs=[food_output, cuisine_output, nutrition_output, metadata_output, status_output]
        )
        
        # Auto-analyze on image upload
        image_input.change(
            fn=analyze_with_feedback,
            inputs=[image_input, model_selector, use_advanced_analysis],
            outputs=[food_output, cuisine_output, nutrition_output, metadata_output, status_output]
        )
        
        # Quick model switch buttons
        basic_btn.click(
            fn=lambda: ("basic", analyzer.switch_model("basic")),
            outputs=[model_selector, status_output]
        )
        advanced_btn.click(
            fn=lambda: ("advanced", analyzer.switch_model("advanced")),
            outputs=[model_selector, status_output]
        )
        ensemble_btn.click(
            fn=lambda: ("ensemble", analyzer.switch_model("ensemble")),
            outputs=[model_selector, status_output]
        )
        
        # Refresh API status
        refresh_btn.click(
            fn=analyzer.get_api_status,
            outputs=api_status
        )
    
    return demo

def main():
    """Launch the advanced demo"""
    print("Starting Advanced Multi-Task Food Analyzer Demo...")
    print("Make sure FastAPI server is running: python src/inference/api.py")
    print("Gradio will be available at: http://localhost:7860")
    
    # Create and launch demo
    demo = create_advanced_demo()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public link
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()