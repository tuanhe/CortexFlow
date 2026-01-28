"""
Abstract base class for compute backends device
"""

from abc import ABC, abstractmethod
from typing import Dict
import torch

from ..configs.model_config import ModelConfig


class ComputeBackend(ABC):
    """计算后端的抽象基类
    
    所有后端实现必须继承这个类并实现以下方法：
    - vision_encoder: 处理视觉编码
    - transformer_encoder: 处理 transformer 编码
    - transformer_decoder: 处理 transformer 解码（带扩散）
    """
    
    def __init__(self, config: ModelConfig):
        """初始化后端
        
        Args:
            config: 模型配置
        """
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def vision_encoder(
        self, 
        inputs: Dict[str, torch.Tensor], 
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """处理视觉编码器
        
        Args:
            inputs: 输入张量字典，包含图像等
            weights: 权重张量字典
            
        Returns:
            视觉特征张量
        """
        pass
    
    @abstractmethod
    def transformer_encoder(
        self, 
        inputs: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """处理 Transformer 编码器
        
        Args:
            inputs: 输入张量字典
            weights: 权重张量字典
            
        Returns:
            编码后的张量
        """
        pass
    
    @abstractmethod
    def transformer_decoder(
        self, 
        inputs: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor],
        num_steps: int
    ) -> torch.Tensor:
        """处理 Transformer 解码器（带扩散）
        
        Args:
            inputs: 输入张量字典
            weights: 权重张量字典
            num_steps: 扩散步数
            
        Returns:
            解码后的动作张量
        """
        pass
    
    def initialize(self):
        """初始化后端（可选）"""
        self._initialized = True
    
    def cleanup(self):
        """清理资源（可选）"""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """检查后端是否已初始化"""
        return self._initialized
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()