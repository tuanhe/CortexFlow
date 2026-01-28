"""
Pure PyTorch fallback backend implementation
"""

from typing import Dict
import torch
import torch.nn.functional as F

from .base import ComputeBackend
from ..configs.model_config import ModelConfig


class PyTorchBackend(ComputeBackend):
    """纯 PyTorch 实现的后端
    
    作为 fallback 实现，不依赖 Triton 等优化库
    性能较低但兼容性最好
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    def vision_encoder(
        self, 
        inputs: Dict[str, torch.Tensor], 
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """纯 PyTorch 视觉编码器实现"""
        x = inputs['observation_images_normalized']
        
        # Patch embedding
        x_flat = x.reshape(-1, 224, 224, 3).permute(0, 3, 1, 2)
        patch_weight = weights['vision_patch_embedding_w'].permute(3, 2, 0, 1)
        patch_bias = weights['vision_patch_embedding_b']
        
        patches = F.conv2d(x_flat, patch_weight, bias=patch_bias, stride=14)
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches + weights['vision_position_embedding']
        
        # Transformer layers (简化)
        for layer_idx in range(min(3, self.config.vision_num_layers)):
            patches = self._vision_layer(patches, weights, layer_idx)
        
        # Final norm
        patches = F.layer_norm(
            patches,
            (self.config.vision_encoder_dim,),
            weight=weights['vision_final_norm_w'],
            bias=weights['vision_final_norm_b']
        )
        
        num_views = inputs['observation_images_normalized'].shape[0]
        inputs['vision_x'] = patches.reshape(num_views, 256, self.config.vision_encoder_dim)
        return inputs['vision_x']
    
    def _vision_layer(self, x, weights, layer_idx):
        """简化的 vision transformer 层"""
        # 简化实现
        return x
    
    def transformer_encoder(
        self, 
        inputs: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """纯 PyTorch Transformer 编码器"""
        # 简化实现 - 实际使用时需要完整实现
        return inputs['encoder_x']
    
    def transformer_decoder(
        self, 
        inputs: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor],
        num_steps: int
    ) -> torch.Tensor:
        """纯 PyTorch Transformer 解码器"""
        # 简化实现 - 实际使用时需要完整实现
        return inputs['diffusion_noise']
