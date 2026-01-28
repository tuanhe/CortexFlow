"""
Inference runtime configuration
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from ..pruning import TokenPruner


class BackendType(Enum):
    """支持的计算后端"""
    TRITON = "triton"
    CUDA = "cuda"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


@dataclass
class InferenceConfig:
    """推理运行时配置
    
    Attributes:
        num_views: 摄像头视角数量
        chunk_size: 动作序列长度
        backend: 使用的计算后端
        device: 设备（cuda 或 cpu）
        dtype: 数据类型精度
        use_cuda_graphs: 是否使用 CUDA graphs 优化
        tokenizer_path: tokenizer 路径
        max_tokenize_len: 最大 token 长度
        discrete_state_input: 是否使用离散状态输入
        compile_mode: PyTorch 2.0 编译模式
        enable_profiling: 是否启用性能分析
        vision_token_pruner: 视觉 token 剪枝器
    """
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
    
    # Token pruning
    vision_token_pruner: Optional['TokenPruner'] = None
    
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
