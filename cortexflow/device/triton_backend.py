"""
Triton-optimized backend implementation
"""

from typing import Dict
import torch

from .base import ComputeBackend
from ..configs.model_config import ModelConfig


class TritonBackend(ComputeBackend):
    """基于 Triton 的优化后端
    
    使用 Triton JIT 编译的内核实现高性能推理
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._import_kernels()
    
    def _import_kernels(self):
        """导入 Triton 内核"""
        try:
            # 从 pi0_infer 导入基础内核
            from pi0_infer import (
                vision_encoder,
                layer_norm_matmul_n256_1152_2048_bias,
                rms_matmul_n_2048_2560_qkv_rope,
                matmul_n_2048_2048_res,
                matmul_n_16384_2048_res,
                rms_matmul_n_2048_16384_gate,
                matmul_small_bias,
                matmul_small_bias_res,
                matmul_small_bias_silu,
                matmul_small_gate,
                matmul_k8_n_256,
                matmul_abT_scale,
            )
            
            self.kernels = {
                'vision_encoder': vision_encoder,
                'layer_norm_matmul': layer_norm_matmul_n256_1152_2048_bias,
                'rms_matmul_qkv': rms_matmul_n_2048_2560_qkv_rope,
                'matmul_res': matmul_n_2048_2048_res,
                'matmul_16384_res': matmul_n_16384_2048_res,
                'rms_matmul_gate': rms_matmul_n_2048_16384_gate,
                'matmul_small_bias': matmul_small_bias,
                'matmul_small_bias_res': matmul_small_bias_res,
                'matmul_small_bias_silu': matmul_small_bias_silu,
                'matmul_small_gate': matmul_small_gate,
                'matmul_k8': matmul_k8_n_256,
                'matmul_scale': matmul_abT_scale,
            }
            
            # 从 pi05_infer 导入特定内核
            try:
                from pi05_infer import (
                    transformer_encoder,
                    transformer_decoder,
                )
                
                self.kernels.update({
                    'transformer_encoder': transformer_encoder,
                    'transformer_decoder': transformer_decoder,
                })
            except ImportError:
                print("Warning: Pi0.5 specific kernels not available")
            
        except ImportError as e:
            raise ImportError(
                "Triton kernels not available. "
                "Install triton or use PyTorch backend."
            ) from e
    
    def vision_encoder(
        self, 
        inputs: Dict[str, torch.Tensor], 
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """使用优化的 Triton 视觉编码器"""
        num_views = inputs['observation_images_normalized'].shape[0]
        self.kernels['vision_encoder'](weights, inputs, num_views)
        return inputs['vision_x']
    
    def transformer_encoder(
        self, 
        inputs: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """使用优化的 Triton Transformer 编码器"""
        if 'transformer_encoder' not in self.kernels:
            raise NotImplementedError(
                "Transformer encoder kernel not available"
            )
        
        encoder_seq_len = inputs['encoder_x'].shape[0]
        self.kernels['transformer_encoder'](weights, inputs, encoder_seq_len)
        return inputs['encoder_x']
    
    def transformer_decoder(
        self, 
        inputs: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor],
        num_steps: int
    ) -> torch.Tensor:
        """使用优化的 Triton Transformer 解码器"""
        if 'transformer_decoder' not in self.kernels:
            raise NotImplementedError(
                "Transformer decoder kernel not available"
            )
        
        encoder_seq_len = inputs['encoder_x'].shape[0] - inputs['diffusion_noise'].shape[0]
        self.kernels['transformer_decoder'](weights, inputs, encoder_seq_len, num_steps)
        return inputs['diffusion_noise']
