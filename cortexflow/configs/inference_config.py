"""
Inference runtime configuration
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import torch


class BackendType(Enum):
    """支持的计算后端"""
    TRITON = "triton"
    CUDA = "cuda"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


@dataclass
class InferenceConfig:
    """推理运行时配置"""
    num_views: int
    chunk_size: int
    backend: BackendType = BackendType.TRITON
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    use_cuda_graphs: bool = True
    tokenizer_path: Optional[str] = None
    max_tokenize_len: int = 200
    discrete_state_input: bool = True
    compile_mode: Optional[str] = None
    enable_profiling: bool = False
    
    # 性能优化选项
    prefetch_weights: bool = False
    pin_memory: bool = True
    num_warmup_iters: int = 3
    
    # 批处理选项
    max_batch_size: int = 1
    dynamic_batching: bool = False
    
    def __post_init__(self):
        """验证配置"""
        if self.use_cuda_graphs and self.device == "cpu":
            raise ValueError("CUDA graphs require CUDA device")
        
        if self.backend == BackendType.TRITON and "cuda" not in self.device:
            raise ValueError("Triton backend requires CUDA device")
        
        if self.num_views <= 0:
            raise ValueError(f"num_views must be positive, got {self.num_views}")
        
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")