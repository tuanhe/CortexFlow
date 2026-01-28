"""
Model configuration definitions
"""

from dataclasses import dataclass
from enum import Enum


class VLAModelType(Enum):
    """支持的 VLA 模型架构"""
    PI0 = "pi0"
    PI05 = "pi05"
    OPENVLA = "openvla"
    RT1 = "rt1"
    RT2 = "rt2"
    OCTO = "octo"


@dataclass
class ModelConfig:
    """VLA 模型的架构配置"""
    model_type: VLAModelType
    vision_encoder_dim: int = 1152
    encoder_dim: int = 2048
    decoder_dim: int = 1024
    num_encoder_layers: int = 18
    num_decoder_layers: int = 18
    num_heads: int = 8
    head_dim: int = 256
    action_dim: int = 32
    chunk_size: int = 10
    num_diffusion_steps: int = 10
    max_seq_len: int = 512
    
    # Vision encoder specific
    vision_patch_size: int = 14
    vision_num_layers: int = 27
    vision_ffn_dim: int = 4304
    
    # Encoder specific
    encoder_ffn_dim: int = 16384
    
    # Decoder specific
    decoder_ffn_dim: int = 4096
    
    def __post_init__(self):
        """验证配置参数"""
        assert self.head_dim * self.num_heads <= self.encoder_dim
        assert self.chunk_size > 0
        assert self.num_diffusion_steps > 0


# 预定义的模型配置
PI05_CONFIG = ModelConfig(
    model_type=VLAModelType.PI05,
    vision_encoder_dim=1152,
    encoder_dim=2048,
    decoder_dim=1024,
    num_encoder_layers=18,
    num_decoder_layers=18,
    num_heads=8,
    head_dim=256,
    action_dim=32,
    chunk_size=10,
    num_diffusion_steps=10,
    vision_num_layers=27,
    vision_ffn_dim=4304,
    encoder_ffn_dim=16384,
    decoder_ffn_dim=4096,
)

OPENVLA_CONFIG = ModelConfig(
    model_type=VLAModelType.OPENVLA,
    vision_encoder_dim=1024,
    encoder_dim=2048,
    decoder_dim=2048,
    num_encoder_layers=12,
    num_decoder_layers=12,
    num_heads=16,
    head_dim=128,
    action_dim=7,
    chunk_size=1,
    num_diffusion_steps=1,
)