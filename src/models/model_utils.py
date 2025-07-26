"""
Production-Ready Model Utilities for Multi-Task Computer Vision Systems.

Key Features ~:
- Model loading/saving with version control
- Performance benchmarking and profiling
- Model architecture comparison and analysis
- Memory optimization and quantization
- Deployment utilities and model serving
- Model interpretability and visualization tools
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from collections import OrderedDict
import psutil
import gc
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import platform


@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    memory_usage_mb: float
    model_size_mb: float
    parameters_count: int
    flops: Optional[int] = None
    throughput_fps: Optional[float] = None


@dataclass
class ModelInfo:
    """Complete model information and metadata."""
    name: str
    architecture: str
    version: str
    created_at: str
    parameters: int
    model_size_mb: float
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    framework: str = "PyTorch"
    quantized: bool = False
    pruned: bool = False
    task_type: str = "multi-task"
    dataset: str = "Food-101"
    metrics: Optional[ModelMetrics] = None
    checksum: Optional[str] = None


class ModelManager:
    """
    Advanced model management system with versioning and metadata tracking.
    """
    
    def __init__(self, models_dir: str = "./models", metadata_file: str = "model_registry.json"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = self.models_dir / metadata_file
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, ModelInfo]:
        """Load model registry from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                return {name: ModelInfo(**info) for name, info in data.items()}
            except Exception as e:
                warnings.warn(f"Could not load registry: {e}")
        return {}
    
    def _save_registry(self):
        """Save model registry to disk."""
        data = {name: asdict(info) for name, info in self.registry.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _calculate_checksum(self, model_path: Path) -> str:
        """Calculate SHA256 checksum of model file."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def save_model(self, 
                   model: nn.Module, 
                   name: str, 
                   version: str = "1.0.0",
                   metadata: Optional[Dict[str, Any]] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None) -> str:
        """
        Save model with comprehensive metadata.
        
        Args:
            model: PyTorch model to save
            name: Model name
            version: Model version
            metadata: Additional metadata
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            
        Returns:
            str: Path to saved model
        """
        # Create model filename
        model_filename = f"{name}_v{version}.pth"
        model_path = self.models_dir / model_filename
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_architecture': model.__class__.__name__,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'platform': platform.platform(),
            'metadata': metadata or {}
        }
        
        # Add optimizer and scheduler if provided
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save model
        torch.save(checkpoint, model_path)
        
        # Calculate model info
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        param_count = sum(p.numel() for p in model.parameters())
        checksum = self._calculate_checksum(model_path)
        
        # Create model info
        model_info = ModelInfo(
            name=name,
            architecture=model.__class__.__name__,
            version=version,
            created_at=datetime.now().isoformat(),
            parameters=param_count,
            model_size_mb=model_size,
            input_shape=(3, 224, 224),  # Default for vision models
            output_shape=self._get_output_shape(model),
            checksum=checksum
        )
        
        # Register model
        self.registry[f"{name}_v{version}"] = model_info
        self._save_registry()
        
        print(f"  Model saved: {model_path}")
        print(f"  Parameters: {param_count:,}")
        print(f"  Size: {model_size:.2f} MB")
        print(f"  Checksum: {checksum[:16]}...")
        
        return str(model_path)
    
    def load_model(self, 
                   name: str, 
                   version: Optional[str] = None,
                   model_class: Optional[Callable] = None,
                   device: str = 'cpu') -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load model with metadata.
        
        Args:
            name: Model name
            version: Model version (latest if None)
            model_class: Model class for instantiation
            device: Device to load model on
            
        Returns:
            Tuple of (model, metadata)
        """
        # Find model
        if version:
            model_key = f"{name}_v{version}"
        else:
            # Find latest version
            model_keys = [k for k in self.registry.keys() if k.startswith(f"{name}_v")]
            if not model_keys:
                raise ValueError(f"No models found with name: {name}")
            model_key = sorted(model_keys)[-1]  # Latest version
        
        if model_key not in self.registry:
            raise ValueError(f"Model not found: {model_key}")
        
        # Load checkpoint
        model_info = self.registry[model_key]
        model_filename = f"{name}_v{model_info.version}.pth"
        model_path = self.models_dir / model_filename
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Verify checksum
        current_checksum = self._calculate_checksum(model_path)
        if model_info.checksum and current_checksum != model_info.checksum:
            warnings.warn("Model checksum mismatch - file may be corrupted")
        
        print(f"  Loaded model: {model_key}")
        print(f"  Architecture: {model_info.architecture}")
        print(f"  Parameters: {model_info.parameters:,}")
        print(f"  Created: {model_info.created_at}")
        
        return checkpoint, asdict(model_info)
    
    def _get_output_shape(self, model: nn.Module) -> Tuple[int, ...]:
        """Infer output shape from model."""
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                output = model(dummy_input)
                if isinstance(output, tuple):
                    return tuple(o.shape[1:] for o in output)
                return output.shape[1:]
        except:
            return (101,)  # Default for classification
    
    def list_models(self) -> Dict[str, ModelInfo]:
        """List all registered models."""
        return self.registry.copy()
    
    def delete_model(self, name: str, version: str):
        """Delete model and its metadata."""
        model_key = f"{name}_v{version}"
        if model_key in self.registry:
            # Delete file
            model_filename = f"{name}_v{version}.pth"
            model_path = self.models_dir / model_filename
            if model_path.exists():
                model_path.unlink()
            
            # Remove from registry
            del self.registry[model_key]
            self._save_registry()
            
            print(f"Deleted model: {model_key}")
        else:
            print(f"Model not found: {model_key}")


class ModelBenchmark:
    """
    Comprehensive model benchmarking and performance analysis.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results = {}
    
    def benchmark_inference_speed(self, 
                                 model: nn.Module, 
                                 input_shape: Tuple[int, ...],
                                 num_iterations: int = 100,
                                 warmup_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dict with timing statistics
        """
        model.eval()
        model = model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
        
        # Synchronize GPU
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        times = np.array(times)
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_fps': 1000.0 / np.mean(times)
        }
    
    def benchmark_memory_usage(self, 
                              model: nn.Module, 
                              input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """
        Benchmark model memory usage.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            
        Returns:
            Dict with memory statistics
        """
        model = model.to(self.device)
        
        # Clear cache
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        gc.collect()
        
        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if self.device.startswith('cuda'):
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Forward pass
        dummy_input = torch.randn(input_shape).to(self.device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory
        
        results = {
            'model_memory_mb': memory_usage,
            'peak_memory_mb': peak_memory
        }
        
        if self.device.startswith('cuda'):
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            gpu_memory_usage = peak_gpu_memory - initial_gpu_memory
            results.update({
                'gpu_memory_mb': gpu_memory_usage,
                'peak_gpu_memory_mb': peak_gpu_memory
            })
        
        return results
    
    def benchmark_throughput(self, 
                           model: nn.Module, 
                           dataloader: DataLoader,
                           max_batches: int = 50) -> Dict[str, float]:
        """
        Benchmark model throughput on real data.
        
        Args:
            model: Model to benchmark
            dataloader: DataLoader for benchmarking
            max_batches: Maximum number of batches to process
            
        Returns:
            Dict with throughput statistics
        """
        model.eval()
        model = model.to(self.device)
        
        total_samples = 0
        total_time = 0
        batch_times = []
        
        with torch.no_grad():
            for batch_idx, (images, *_) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                images = images.to(self.device)
                batch_size = images.size(0)
                
                start_time = time.perf_counter()
                _ = model(images)
                
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                batch_time = end_time - start_time
                batch_times.append(batch_time)
                total_samples += batch_size
                total_time += batch_time
        
        avg_throughput = total_samples / total_time
        batch_times = np.array(batch_times)
        
        return {
            'avg_throughput_fps': avg_throughput,
            'total_samples': total_samples,
            'total_time_s': total_time,
            'avg_batch_time_s': float(np.mean(batch_times)),
            'min_batch_time_s': float(np.min(batch_times)),
            'max_batch_time_s': float(np.max(batch_times))
        }
    
    def profile_model_layers(self, 
                           model: nn.Module, 
                           input_shape: Tuple[int, ...]) -> Dict[str, Dict[str, Any]]:
        """
        Profile individual model layers for performance analysis.
        
        Args:
            model: Model to profile
            input_shape: Input tensor shape
            
        Returns:
            Dict with per-layer profiling results
        """
        model.eval()
        model = model.to(self.device)
        
        layer_times = {}
        layer_outputs = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                start_time = time.perf_counter()
                # Small delay to measure
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                layer_times[name] = (end_time - start_time) * 1000  # ms
                
                if isinstance(output, torch.Tensor):
                    layer_outputs[name] = {
                        'output_shape': tuple(output.shape),
                        'memory_mb': output.numel() * output.element_size() / 1024 / 1024
                    }
                elif isinstance(output, (tuple, list)):
                    layer_outputs[name] = {
                        'output_shapes': [tuple(o.shape) for o in output],
                        'memory_mb': sum(o.numel() * o.element_size() for o in output) / 1024 / 1024
                    }
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Run forward pass
        dummy_input = torch.randn(input_shape).to(self.device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Combine results
        results = {}
        for name in layer_times:
            results[name] = {
                'execution_time_ms': layer_times.get(name, 0),
                **layer_outputs.get(name, {})
            }
        
        return results


class ModelComparator:
    """
    Advanced model comparison and analysis tools.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.benchmark = ModelBenchmark(device)
    
    def compare_models(self, 
                      models: Dict[str, nn.Module], 
                      input_shape: Tuple[int, ...],
                      dataloader: Optional[DataLoader] = None) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive comparison of multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            input_shape: Input tensor shape
            dataloader: Optional dataloader for accuracy evaluation
            
        Returns:
            Dict with comparison results
        """
        results = {}
        
        print("Comparing models...")
        print("=" * 60)
        
        for name, model in models.items():
            print(f"\nAnalyzing {name}...")
            
            # Basic model info
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Inference speed
            speed_results = self.benchmark.benchmark_inference_speed(model, input_shape)
            
            # Memory usage
            memory_results = self.benchmark.benchmark_memory_usage(model, input_shape)
            
            # Throughput (if dataloader provided)
            throughput_results = {}
            if dataloader:
                throughput_results = self.benchmark.benchmark_throughput(model, dataloader)
            
            # Model size estimation
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            
            results[name] = {
                'parameters': {
                    'total': param_count,
                    'trainable': trainable_params,
                    'non_trainable': param_count - trainable_params
                },
                'model_size_mb': model_size_mb,
                'inference_speed': speed_results,
                'memory_usage': memory_results,
                'throughput': throughput_results,
                'architecture': model.__class__.__name__
            }
            
            print(f"  Parameters: {param_count:,}")
            print(f"  Model size: {model_size_mb:.2f} MB")
            print(f"  Avg inference: {speed_results['mean_ms']:.2f} ms")
            print(f"  Memory usage: {memory_results['model_memory_mb']:.2f} MB")
        
        return results
    
    def generate_comparison_report(self, 
                                 comparison_results: Dict[str, Dict[str, Any]],
                                 save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive comparison report.
        
        Args:
            comparison_results: Results from compare_models
            save_path: Optional path to save report
            
        Returns:
            str: Formatted report
        """
        report = []
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("SUMMARY TABLE")
        report.append("-" * 30)
        
        header = f"{'Model':<20} {'Params':<12} {'Size(MB)':<10} {'Speed(ms)':<12} {'Memory(MB)':<12}"
        report.append(header)
        report.append("-" * len(header))
        
        for name, results in comparison_results.items():
            params = results['parameters']['total']
            size = results['model_size_mb']
            speed = results['inference_speed']['mean_ms']
            memory = results['memory_usage']['model_memory_mb']
            
            row = f"{name:<20} {params:<12,} {size:<10.2f} {speed:<12.2f} {memory:<12.2f}"
            report.append(row)
        
        report.append("")
        
        # Detailed analysis
        report.append("DETAILED ANALYSIS")
        report.append("-" * 30)
        
        for name, results in comparison_results.items():
            report.append(f"\n{name.upper()}:")
            report.append(f"  Architecture: {results['architecture']}")
            report.append(f"  Total Parameters: {results['parameters']['total']:,}")
            report.append(f"  Trainable Parameters: {results['parameters']['trainable']:,}")
            report.append(f"  Model Size: {results['model_size_mb']:.2f} MB")
            
            # Speed analysis
            speed = results['inference_speed']
            report.append(f"  Inference Speed:")
            report.append(f"    Mean: {speed['mean_ms']:.2f} ms")
            report.append(f"    Std: {speed['std_ms']:.2f} ms")
            report.append(f"    P95: {speed['p95_ms']:.2f} ms")
            report.append(f"    Throughput: {speed['throughput_fps']:.1f} FPS")
            
            # Memory analysis
            memory = results['memory_usage']
            report.append(f"  Memory Usage:")
            report.append(f"    Model: {memory['model_memory_mb']:.2f} MB")
            if 'gpu_memory_mb' in memory:
                report.append(f"    GPU: {memory['gpu_memory_mb']:.2f} MB")
        
        # Recommendations
        report.append("\nRECOMMENDations:")
        report.append("-" * 20)
        
        # Find best models for different criteria
        models = list(comparison_results.keys())
        speeds = [comparison_results[m]['inference_speed']['mean_ms'] for m in models]
        params = [comparison_results[m]['parameters']['total'] for m in models]
        memories = [comparison_results[m]['memory_usage']['model_memory_mb'] for m in models]
        
        fastest_model = models[np.argmin(speeds)]
        smallest_model = models[np.argmin(params)]
        memory_efficient = models[np.argmin(memories)]
        
        report.append(f"  Fastest inference: {fastest_model}")
        report.append(f"  Smallest model: {smallest_model}")
        report.append(f"  Most memory efficient: {memory_efficient}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text


class ModelOptimizer:
    """
    Model optimization utilities including quantization and pruning.
    """
    
    @staticmethod
    def quantize_model(model: nn.Module, 
                      calibration_loader: Optional[DataLoader] = None,
                      quantization_type: str = 'dynamic') -> nn.Module:
        """
        Quantize model for inference optimization.
        
        Args:
            model: Model to quantize
            calibration_loader: DataLoader for calibration (static quantization)
            quantization_type: 'dynamic' or 'static'
            
        Returns:
            nn.Module: Quantized model
        """
        model.eval()
        
        if quantization_type == 'dynamic':
            # Dynamic quantization (no calibration needed)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            print("Applied dynamic quantization")
            
        elif quantization_type == 'static':
            if calibration_loader is None:
                raise ValueError("Calibration loader required for static quantization")
            
            # Prepare model for static quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibration
            print("Calibrating model...")
            with torch.no_grad():
                for batch_idx, (images, *_) in enumerate(calibration_loader):
                    if batch_idx >= 10:  # Limit calibration batches
                        break
                    _ = model(images)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=False)
            print("Applied static quantization")
        
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        return quantized_model
    
    @staticmethod
    def prune_model(model: nn.Module, 
                   pruning_ratio: float = 0.2,
                   structured: bool = False) -> nn.Module:
        """
        Prune model to reduce parameters.
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of parameters to prune
            structured: Whether to use structured pruning
            
        Returns:
            nn.Module: Pruned model
        """
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        if structured:
            # Structured pruning (remove entire channels/filters)
            for module, param_name in parameters_to_prune:
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name=param_name, amount=pruning_ratio, n=2, dim=0)
                else:
                    prune.l1_unstructured(module, name=param_name, amount=pruning_ratio)
        else:
            # Unstructured pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
        
        print(f"Applied {pruning_ratio:.1%} pruning ({'structured' if structured else 'unstructured'})")
        
        return model
    
    @staticmethod
    def optimize_for_mobile(model: nn.Module, 
                          input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> torch.jit.ScriptModule:
        """
        Optimize model for mobile deployment.
        
        Args:
            model: Model to optimize
            input_shape: Input tensor shape for tracing
            
        Returns:
            torch.jit.ScriptModule: Mobile-optimized model
        """
        model.eval()
        
        # Create example input
        example_input = torch.randn(input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimize for mobile
        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
        
        print("✓ Optimized model for mobile deployment")
        
        return optimized_model


def analyze_model_complexity(model: nn.Module, 
                           input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
    """
    Comprehensive model complexity analysis.
    
    Args:
        model: Model to analyze
        input_shape: Input tensor shape
        
    Returns:
        Dict with complexity metrics
    """
    model.eval()
    
    # Parameter analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory analysis
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    # Layer analysis
    layer_count = len(list(model.modules()))
    conv_layers = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
    linear_layers = len([m for m in model.modules() if isinstance(m, nn.Linear)])
    
    # Model size estimation
    model_size_mb = param_memory
    
    # Receptive field analysis (simplified)
    receptive_field = estimate_receptive_field(model, input_shape)
    
    return {
        'parameters': {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        },
        'memory': {
            'parameters_mb': param_memory,
            'estimated_model_size_mb': model_size_mb
        },
        'architecture': {
            'total_layers': layer_count,
            'conv_layers': conv_layers,
            'linear_layers': linear_layers,
            'layer_types': list(set(type(m).__name__ for m in model.modules()))
        },
        'receptive_field': receptive_field,
        'input_shape': input_shape,
        'complexity_score': calculate_complexity_score(total_params, conv_layers, linear_layers)
    }


def estimate_receptive_field(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, int]:
    """
    Estimate receptive field size of the model.
    
    Args:
        model: Model to analyze
        input_shape: Input tensor shape
        
    Returns:
        Dict with receptive field information
    """
    # Simplified receptive field calculation
    receptive_field = 1
    stride = 1
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            module_stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            
            receptive_field += (kernel_size - 1) * stride
            stride *= module_stride
            
        elif isinstance(module, nn.MaxPool2d):
            kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            module_stride = module.stride if isinstance(module.stride, int) else module.stride[0]
            
            receptive_field += (kernel_size - 1) * stride
            stride *= module_stride
    
    return {
        'receptive_field_size': receptive_field,
        'effective_stride': stride
    }


def calculate_complexity_score(params: int, conv_layers: int, linear_layers: int) -> float:
    """
    Calculate a normalized complexity score for the model.
    
    Args:
        params: Total number of parameters
        conv_layers: Number of convolutional layers
        linear_layers: Number of linear layers
        
    Returns:
        float: Complexity score (0-100)
    """
    # Normalize based on typical model sizes
    param_score = min(params / 100_000_000, 1.0) * 50  # Up to 100M params = 50 points
    layer_score = min((conv_layers + linear_layers) / 100, 1.0) * 50  # Up to 100 layers = 50 points
    
    return param_score + layer_score


def export_model_for_deployment(model: nn.Module,
                               export_format: str = 'onnx',
                               output_path: str = 'model_export',
                               input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                               optimize: bool = True) -> str:
    """
    Export model for deployment in various formats.
    
    Args:
        model: Model to export
        export_format: Format ('onnx', 'torchscript', 'coreml')
        output_path: Output file path (without extension)
        input_shape: Input tensor shape
        optimize: Whether to apply optimizations
        
    Returns:
        str: Path to exported model
    """
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    if export_format.lower() == 'onnx':
        output_file = f"{output_path}.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
            output_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=optimize,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
    elif export_format.lower() == 'torchscript':
        output_file = f"{output_path}.pt"
        
        traced_model = torch.jit.trace(model, dummy_input)
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        traced_model.save(output_file)
        
    elif export_format.lower() == 'coreml':
        try:
            import coremltools as ct
            output_file = f"{output_path}.mlmodel"
            
            traced_model = torch.jit.trace(model, dummy_input)
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=input_shape)]
            )
            coreml_model.save(output_file)
            
        except ImportError:
            raise ImportError("coremltools required for Core ML export")
    
    else:
        raise ValueError(f"Unsupported export format: {export_format}")
    
    print(f"Model exported to: {output_file}")
    return output_file


class ModelValidator:
    """
    Model validation and testing utilities.
    """
    
    @staticmethod
    def validate_model_outputs(model: nn.Module,
                             input_shape: Tuple[int, ...],
                             expected_output_shapes: List[Tuple[int, ...]],
                             num_classes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Validate model output shapes and ranges.
        
        Args:
            model: Model to validate
            input_shape: Input tensor shape
            expected_output_shapes: Expected output shapes
            num_classes: Expected number of classes for each output
            
        Returns:
            Dict with validation results
        """
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        
        results = {
            'valid': True,
            'errors': [],
            'output_analysis': []
        }
        
        for i, (output, expected_shape) in enumerate(zip(outputs, expected_output_shapes)):
            analysis = {
                'output_index': i,
                'actual_shape': tuple(output.shape),
                'expected_shape': expected_shape,
                'shape_match': tuple(output.shape[1:]) == expected_shape[1:],
                'value_range': (float(output.min()), float(output.max())),
                'contains_nan': bool(torch.isnan(output).any()),
                'contains_inf': bool(torch.isinf(output).any())
            }
            
            # Check for issues
            if not analysis['shape_match']:
                results['valid'] = False
                results['errors'].append(f"Output {i} shape mismatch: {output.shape} vs {expected_shape}")
            
            if analysis['contains_nan']:
                results['valid'] = False
                results['errors'].append(f"Output {i} contains NaN values")
            
            if analysis['contains_inf']:
                results['valid'] = False
                results['errors'].append(f"Output {i} contains Inf values")
            
            # Validate classification outputs
            if num_classes and i < len(num_classes):
                if output.shape[1] != num_classes[i]:
                    results['valid'] = False
                    results['errors'].append(f"Output {i} class count mismatch: {output.shape[1]} vs {num_classes[i]}")
            
            results['output_analysis'].append(analysis)
        
        return results
    
    @staticmethod
    def test_model_consistency(model: nn.Module,
                             input_shape: Tuple[int, ...],
                             num_runs: int = 5) -> Dict[str, Any]:
        """
        Test model consistency across multiple runs.
        
        Args:
            model: Model to test
            input_shape: Input tensor shape
            num_runs: Number of test runs
            
        Returns:
            Dict with consistency test results
        """
        model.eval()
        
        outputs_list = []
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        dummy_input = torch.randn(input_shape)
        
        # Multiple runs with same input
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model(dummy_input)
            outputs_list.append(outputs)
        
        # Check consistency
        if isinstance(outputs_list[0], tuple):
            num_outputs = len(outputs_list[0])
            consistent = True
            max_differences = []
            
            for i in range(num_outputs):
                output_values = [outputs[i] for outputs in outputs_list]
                differences = []
                
                for j in range(1, num_runs):
                    diff = torch.abs(output_values[0] - output_values[j]).max().item()
                    differences.append(diff)
                
                max_diff = max(differences)
                max_differences.append(max_diff)
                
                if max_diff > 1e-6:  # Tolerance for floating point
                    consistent = False
        
        else:
            # Single output
            differences = []
            for j in range(1, num_runs):
                diff = torch.abs(outputs_list[0] - outputs_list[j]).max().item()
                differences.append(diff)
            
            max_differences = [max(differences)]
            consistent = max(differences) <= 1e-6
        
        return {
            'consistent': consistent,
            'max_differences': max_differences,
            'tolerance': 1e-6,
            'num_runs': num_runs
        }


if __name__ == "__main__":
    print("Testing Model Utilities...")
    print("=" * 50)
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Test model manager
    print("\n1. Testing Model Manager...")
    manager = ModelManager()
    
    model = SimpleModel()
    saved_path = manager.save_model(model, "test_model", "1.0.0")
    
    loaded_checkpoint, metadata = manager.load_model("test_model")
    print(f"Model saved and loaded successfully")
    
    # Test benchmarking
    print("\n2. Testing Model Benchmarking...")
    benchmark = ModelBenchmark()
    
    speed_results = benchmark.benchmark_inference_speed(model, (1, 3, 224, 224))
    print(f"Speed benchmark: {speed_results['mean_ms']:.2f} ms")
    
    memory_results = benchmark.benchmark_memory_usage(model, (1, 3, 224, 224))
    print(f"Memory benchmark: {memory_results['model_memory_mb']:.2f} MB")
    
    # Test model analysis
    print("\n3. Testing Model Analysis...")
    complexity = analyze_model_complexity(model)
    print(f"Model complexity: {complexity['parameters']['total']:,} parameters")
    print(f"Complexity score: {complexity['complexity_score']:.1f}/100")
    
    # Test model validation
    print("\n4. Testing Model Validation...")
    validator = ModelValidator()
    
    validation_results = validator.validate_model_outputs(
        model, 
        (1, 3, 224, 224), 
        [(1, 10)],
        [10]
    )
    print(f"Model validation: {'PASSED' if validation_results['valid'] else 'FAILED'}")
    
    consistency_results = validator.test_model_consistency(model, (1, 3, 224, 224))
    print(f"Consistency test: {'PASSED' if consistency_results['consistent'] else 'FAILED'}")
    
    # Test export
    print("\n5. Testing Model Export...")
    try:
        export_path = export_model_for_deployment(model, 'torchscript', 'test_export')
        print(f"Model exported successfully")
    except Exception as e:
        print(f"Export test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Model Utilities Testing Complete!")