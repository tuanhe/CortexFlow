"""
Utility functions for CortexFlow
"""

from typing import Dict, Any
import torch
import numpy as np


def normalize_images(
    images: np.ndarray,
    mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
    std: np.ndarray = np.array([0.229, 0.224, 0.225])
) -> torch.Tensor:
    """归一化图像"""
    images = images.astype(np.float32) / 255.0
    images = (images - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    return torch.from_numpy(images).to(dtype=torch.bfloat16)


def benchmark_inference(engine, num_iterations: int = 100, warmup_iterations: int = 10) -> Dict[str, float]:
    """性能基准测试"""
    import time
    
    num_views = engine.inference_config.num_views
    images = torch.randn(
        num_views, 224, 224, 3,
        dtype=engine.inference_config.dtype,
        device=engine.inference_config.device
    )
    state_tokens = np.random.randint(0, 255, size=5)
    
    # 预热
    for _ in range(warmup_iterations):
        _ = engine.predict(images, "test task", state_tokens)
    
    # 测试
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        _ = engine.predict(images, "test task", state_tokens)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency = total_time / num_iterations
    throughput = num_iterations / total_time
    
    return {
        'total_time_sec': total_time,
        'avg_latency_ms': avg_latency * 1000,
        'throughput_fps': throughput,
        'num_iterations': num_iterations,
    }
