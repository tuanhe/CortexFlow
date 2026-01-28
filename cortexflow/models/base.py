"""
Abstract base class for VLA models
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import torch

from ..configs.model_config import ModelConfig


class VLAModel(ABC):
    """VLA 模型的抽象基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def get_weight_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """返回权重张量的形状字典"""
        pass
    
    @abstractmethod
    def get_buffer_shapes(self, num_views: int, chunk_size: int) -> Dict[str, Tuple[int, ...]]:
        """返回缓冲区张量的形状字典"""
        pass
    
    @abstractmethod
    def preprocess_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """预处理 checkpoint 权重"""
        pass
    
    def postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        """后处理模型输出"""
        return output
