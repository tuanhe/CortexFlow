# Token Pruning 使用示例

from cortexflow import create_inference, create_pruner

# ==================== 方式 1: 使用内置策略 ====================

# Top-K 剪枝：保留 50% 最重要的 tokens
engine = create_inference(
    "pi05",
    "checkpoint.pt",
    num_views=2,
    pruning_strategy="topk",
    pruning_keep_ratio=0.5,
    pruning_min_tokens=64,
)

# 空间池化：2x2 池化
engine = create_inference(
    "pi05",
    "checkpoint.pt",
    num_views=2,
    pruning_strategy="spatial",
    pool_size=2,
)

# 混合策略：先空间池化再 Top-K
engine = create_inference(
    "pi05",
    "checkpoint.pt",
    num_views=2,
    pruning_strategy="hybrid",
    pruning_keep_ratio=0.7,
    use_spatial=True,
    spatial_pool_size=2,
)

# ==================== 方式 2: 自定义 Pruner ====================

from cortexflow import TokenPruner
import torch

class MyCustomPruner(TokenPruner):
    """自定义剪枝策略"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def prune(self, tokens, attention_weights=None, **kwargs):
        # 自定义剪枝逻辑
        # tokens: (num_views, num_tokens, hidden_dim)
        
        # 示例：基于 token 的方差剪枝
        variance = tokens.var(dim=-1)  # (num_views, num_tokens)
        mask = variance > self.threshold
        
        pruned_tokens = []
        for view_idx in range(tokens.shape[0]):
            selected = tokens[view_idx, mask[view_idx]]
            pruned_tokens.append(selected)
        
        return torch.stack(pruned_tokens, dim=0)
    
    def get_pruned_length(self, original_length):
        # 估计剪枝后的长度
        return int(original_length * 0.6)


# 使用自定义 pruner
my_pruner = MyCustomPruner(threshold=0.3)

from cortexflow import InferenceConfig, ModelConfig, VLAModelType, BackendType, CortexFlowEngine

model_config = ModelConfig(model_type=VLAModelType.PI05)
inference_config = InferenceConfig(
    num_views=2,
    chunk_size=10,
    backend=BackendType.TRITON,
    vision_token_pruner=my_pruner,  # 传入自定义 pruner
)

checkpoint = torch.load("checkpoint.pt")
engine = CortexFlowEngine(checkpoint, model_config, inference_config)

# ==================== 方式 3: 使用 create_pruner 工具函数 ====================

# 创建不同的 pruner
identity_pruner = create_pruner("identity")  # 不剪枝
topk_pruner = create_pruner("topk", keep_ratio=0.5)
attention_pruner = create_pruner("attention", keep_ratio=0.6)
spatial_pruner = create_pruner("spatial", pool_size=2)
random_pruner = create_pruner("random", keep_ratio=0.5)  # 用于测试

# 动态切换 pruner
inference_config.vision_token_pruner = topk_pruner
# ... 使用 engine

inference_config.vision_token_pruner = spatial_pruner
# ... 使用 engine

# ==================== 性能对比 ====================

import numpy as np
from cortexflow.utils import benchmark_inference

strategies = ["identity", "topk", "spatial", "hybrid"]
results = {}

for strategy in strategies:
    engine = create_inference(
        "pi05",
        "checkpoint.pt",
        num_views=2,
        pruning_strategy=strategy if strategy != "identity" else None,
        pruning_keep_ratio=0.5,
    )
    
    stats = benchmark_inference(engine, num_iterations=50)
    results[strategy] = stats
    
    print(f"{strategy:10s}: {stats['avg_latency_ms']:.2f} ms, "
          f"{stats['throughput_fps']:.2f} FPS")
