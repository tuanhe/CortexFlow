"""
Factory functions for creating CortexFlow inference engines
"""

from typing import Optional, Union
import torch

from .engine import CortexFlowEngine
from .configs import ModelConfig, InferenceConfig, BackendType, VLAModelType, PI05_CONFIG, OPENVLA_CONFIG


# 模型默认配置映射
_MODEL_CONFIGS = {
    VLAModelType.PI05: PI05_CONFIG,
    VLAModelType.PI0: PI05_CONFIG,
    VLAModelType.OPENVLA: OPENVLA_CONFIG,
}

# 模型默认推理配置
_MODEL_DEFAULTS = {
    VLAModelType.PI05: {
        'backend': BackendType.TRITON,
        'use_cuda_graphs': True,
        'discrete_state_input': True,
    },
    VLAModelType.PI0: {
        'backend': BackendType.TRITON,
        'use_cuda_graphs': True,
        'discrete_state_input': True,
    },
    VLAModelType.OPENVLA: {
        'backend': BackendType.PYTORCH,
        'use_cuda_graphs': False,
        'discrete_state_input': False,
    },
}


def create_inference(
    model_type: Union[str, VLAModelType],
    checkpoint_path: str,
    num_views: int = 1,
    chunk_size: Optional[int] = None,
    backend: Optional[BackendType] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_cuda_graphs: Optional[bool] = None,
    tokenizer_path: Optional[str] = None,
    discrete_state_input: Optional[bool] = None,
    model_config: Optional[ModelConfig] = None,
    # Token pruning
    pruning_strategy: Optional[str] = None,
    pruning_keep_ratio: float = 0.5,
    pruning_min_tokens: int = 64,
    **kwargs
) -> CortexFlowEngine:
    """统一的推理引擎创建接口
    
    Args:
        model_type: 模型类型 ("pi05", "pi0", "openvla", "rt1", "rt2" 或 VLAModelType)
        checkpoint_path: checkpoint 文件路径
        num_views: 摄像头视角数量
        chunk_size: 动作序列长度 (None 则使用模型默认值)
        backend: 计算后端 (None 则使用模型默认值)
        device: 设备
        dtype: 数据类型
        use_cuda_graphs: 是否使用 CUDA graphs (None 则使用模型默认值)
        tokenizer_path: tokenizer 路径
        discrete_state_input: 是否使用离散状态输入 (None 则使用模型默认值)
        model_config: 自定义模型配置 (提供则忽略其他模型配置参数)
        pruning_strategy: Token 剪枝策略 ("topk", "attention", "spatial", "hybrid", None=不剪枝)
        pruning_keep_ratio: 剪枝保留比例 (0.0-1.0)
        pruning_min_tokens: 最少保留的 token 数量
        **kwargs: 其他配置参数
        
    Returns:
        配置好的 CortexFlowEngine
        
    Examples:
        >>> # Pi0.5 with Top-K pruning
        >>> engine = create_inference(
        ...     "pi05", "checkpoint.pt",
        ...     num_views=2,
        ...     pruning_strategy="topk",
        ...     pruning_keep_ratio=0.5
        ... )
        
        >>> # OpenVLA
        >>> engine = create_inference("openvla", "checkpoint.pt")
        
        >>> # 自定义后端 + 空间剪枝
        >>> engine = create_inference(
        ...     "pi05", "checkpoint.pt",
        ...     backend=BackendType.PYTORCH,
        ...     pruning_strategy="spatial",
        ...     pool_size=2
        ... )
    """
    # 转换字符串为枚举
    if isinstance(model_type, str):
        model_type = VLAModelType(model_type.lower())
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取模型默认配置
    if model_config is None:
        base_config = _MODEL_CONFIGS.get(model_type)
        if base_config is None:
            raise ValueError(f"No default config for model type: {model_type}")
        
        # 使用提供的 chunk_size 或默认值
        if chunk_size is None:
            chunk_size = base_config.chunk_size
        
        model_config = ModelConfig(
            model_type=model_type,
            vision_encoder_dim=base_config.vision_encoder_dim,
            encoder_dim=base_config.encoder_dim,
            decoder_dim=base_config.decoder_dim,
            num_encoder_layers=base_config.num_encoder_layers,
            num_decoder_layers=base_config.num_decoder_layers,
            num_heads=base_config.num_heads,
            head_dim=base_config.head_dim,
            action_dim=base_config.action_dim,
            chunk_size=chunk_size,
            num_diffusion_steps=base_config.num_diffusion_steps,
            vision_num_layers=base_config.vision_num_layers,
            vision_ffn_dim=base_config.vision_ffn_dim,
            encoder_ffn_dim=base_config.encoder_ffn_dim,
            decoder_ffn_dim=base_config.decoder_ffn_dim,
        )
    else:
        if chunk_size is None:
            chunk_size = model_config.chunk_size
    
    # 获取推理默认配置
    defaults = _MODEL_DEFAULTS.get(model_type, {})
    
    if backend is None:
        backend = defaults.get('backend', BackendType.TRITON)
    if use_cuda_graphs is None:
        use_cuda_graphs = defaults.get('use_cuda_graphs', True)
    if discrete_state_input is None:
        discrete_state_input = defaults.get('discrete_state_input', True)
    
    # 创建 token pruner
    vision_token_pruner = None
    if pruning_strategy is not None:
        from .pruning import create_pruner
        vision_token_pruner = create_pruner(
            strategy=pruning_strategy,
            keep_ratio=pruning_keep_ratio,
            min_tokens=pruning_min_tokens,
            **kwargs
        )
    
    inference_config = InferenceConfig(
        num_views=num_views,
        chunk_size=chunk_size,
        backend=backend,
        device=device,
        dtype=dtype,
        use_cuda_graphs=use_cuda_graphs,
        tokenizer_path=tokenizer_path,
        discrete_state_input=discrete_state_input,
        vision_token_pruner=vision_token_pruner,
        **{k: v for k, v in kwargs.items() 
           if k not in ['pool_size', 'use_spatial', 'spatial_pool_size']}
    )
    
    return CortexFlowEngine(checkpoint, model_config, inference_config)
