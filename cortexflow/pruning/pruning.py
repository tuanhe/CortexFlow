"""
Token pruning strategies for vision encoder outputs
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch


class TokenPruner(ABC):
    """Token 剪枝策略的抽象基类"""
    
    @abstractmethod
    def prune(
        self,
        tokens: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """剪枝 tokens
        
        Args:
            tokens: (num_views, num_tokens, hidden_dim) 视觉 tokens
            attention_weights: 可选的注意力权重
            **kwargs: 其他参数
            
        Returns:
            剪枝后的 tokens
        """
        pass
    
    @abstractmethod
    def get_pruned_length(self, original_length: int) -> int:
        """获取剪枝后的长度"""
        pass


class IdentityPruner(TokenPruner):
    """不做任何剪枝（默认）"""
    
    def prune(
        self,
        tokens: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        return tokens
    
    def get_pruned_length(self, original_length: int) -> int:
        return original_length


class TopKPruner(TokenPruner):
    """保留 Top-K 最重要的 tokens
    
    根据 token 的 L2 norm 选择最重要的 K 个
    """
    
    def __init__(self, keep_ratio: float = 0.5, min_tokens: int = 64):
        """
        Args:
            keep_ratio: 保留的 token 比例 (0.0-1.0)
            min_tokens: 最少保留的 token 数量
        """
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
    
    def prune(
        self,
        tokens: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """基于 L2 norm 的 Top-K 剪枝"""
        num_views, num_tokens, hidden_dim = tokens.shape
        
        # 计算每个 token 的重要性分数（L2 norm）
        importance = torch.norm(tokens, p=2, dim=-1)  # (num_views, num_tokens)
        
        # 计算保留数量
        keep_k = max(self.min_tokens, int(num_tokens * self.keep_ratio))
        
        # 每个视角独立剪枝
        pruned_tokens = []
        for view_idx in range(num_views):
            # 获取 Top-K indices
            _, top_indices = torch.topk(importance[view_idx], keep_k)
            top_indices = top_indices.sort()[0]  # 保持原始顺序
            
            # 选择 tokens
            pruned_tokens.append(tokens[view_idx, top_indices])
        
        return torch.stack(pruned_tokens, dim=0)
    
    def get_pruned_length(self, original_length: int) -> int:
        return max(self.min_tokens, int(original_length * self.keep_ratio))


class AttentionPruner(TokenPruner):
    """基于注意力权重的剪枝
    
    保留注意力权重最高的 tokens
    """
    
    def __init__(self, keep_ratio: float = 0.5, min_tokens: int = 64):
        """
        Args:
            keep_ratio: 保留的 token 比例
            min_tokens: 最少保留的 token 数量
        """
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
    
    def prune(
        self,
        tokens: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """基于注意力权重的剪枝"""
        if attention_weights is None:
            raise ValueError("AttentionPruner requires attention_weights")
        
        num_views, num_tokens, hidden_dim = tokens.shape
        keep_k = max(self.min_tokens, int(num_tokens * self.keep_ratio))
        
        # attention_weights: (num_views, num_tokens) 或 (num_views, num_heads, num_tokens)
        if attention_weights.dim() == 3:
            # 平均所有注意力头
            importance = attention_weights.mean(dim=1)
        else:
            importance = attention_weights
        
        pruned_tokens = []
        for view_idx in range(num_views):
            _, top_indices = torch.topk(importance[view_idx], keep_k)
            top_indices = top_indices.sort()[0]
            pruned_tokens.append(tokens[view_idx, top_indices])
        
        return torch.stack(pruned_tokens, dim=0)
    
    def get_pruned_length(self, original_length: int) -> int:
        return max(self.min_tokens, int(original_length * self.keep_ratio))


class RandomPruner(TokenPruner):
    """随机剪枝（用于测试）"""
    
    def __init__(self, keep_ratio: float = 0.5, min_tokens: int = 64):
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
    
    def prune(
        self,
        tokens: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        num_views, num_tokens, hidden_dim = tokens.shape
        keep_k = max(self.min_tokens, int(num_tokens * self.keep_ratio))
        
        pruned_tokens = []
        for view_idx in range(num_views):
            indices = torch.randperm(num_tokens, device=tokens.device)[:keep_k]
            indices = indices.sort()[0]
            pruned_tokens.append(tokens[view_idx, indices])
        
        return torch.stack(pruned_tokens, dim=0)
    
    def get_pruned_length(self, original_length: int) -> int:
        return max(self.min_tokens, int(original_length * self.keep_ratio))


class SpatialPruner(TokenPruner):
    """空间池化剪枝
    
    将空间上相邻的 tokens 合并
    """
    
    def __init__(self, pool_size: int = 2):
        """
        Args:
            pool_size: 池化窗口大小
        """
        self.pool_size = pool_size
    
    def prune(
        self,
        tokens: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """空间池化"""
        num_views, num_tokens, hidden_dim = tokens.shape
        
        # 假设 tokens 是 16x16 的 grid
        grid_size = int(num_tokens ** 0.5)
        assert grid_size * grid_size == num_tokens, "Tokens must form a square grid"
        
        # Reshape to spatial grid
        tokens_grid = tokens.view(num_views, grid_size, grid_size, hidden_dim)
        
        # 池化
        new_size = grid_size // self.pool_size
        pooled = torch.nn.functional.avg_pool2d(
            tokens_grid.permute(0, 3, 1, 2),  # (views, dim, h, w)
            kernel_size=self.pool_size,
            stride=self.pool_size
        ).permute(0, 2, 3, 1)  # (views, h', w', dim)
        
        # Reshape back
        return pooled.reshape(num_views, new_size * new_size, hidden_dim)
    
    def get_pruned_length(self, original_length: int) -> int:
        grid_size = int(original_length ** 0.5)
        new_size = grid_size // self.pool_size
        return new_size * new_size


class HybridPruner(TokenPruner):
    """混合剪枝策略
    
    结合多种策略的优点
    """
    
    def __init__(
        self,
        keep_ratio: float = 0.5,
        min_tokens: int = 64,
        use_spatial: bool = False,
        spatial_pool_size: int = 2
    ):
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
        self.use_spatial = use_spatial
        self.spatial_pool_size = spatial_pool_size
    
    def prune(
        self,
        tokens: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # 第一步：空间池化（可选）
        if self.use_spatial:
            spatial_pruner = SpatialPruner(self.spatial_pool_size)
            tokens = spatial_pruner.prune(tokens)
        
        # 第二步：Top-K 剪枝
        topk_pruner = TopKPruner(self.keep_ratio, self.min_tokens)
        tokens = topk_pruner.prune(tokens, attention_weights)
        
        return tokens
    
    def get_pruned_length(self, original_length: int) -> int:
        if self.use_spatial:
            grid_size = int(original_length ** 0.5)
            new_size = grid_size // self.spatial_pool_size
            length = new_size * new_size
        else:
            length = original_length
        
        return max(self.min_tokens, int(length * self.keep_ratio))


# 便捷函数
def create_pruner(
    strategy: str = "identity",
    keep_ratio: float = 0.5,
    min_tokens: int = 64,
    **kwargs
) -> TokenPruner:
    """创建 token pruner
    
    Args:
        strategy: 剪枝策略 ("identity", "topk", "attention", "random", "spatial", "hybrid")
        keep_ratio: 保留比例
        min_tokens: 最小 token 数
        **kwargs: 其他参数
        
    Returns:
        TokenPruner 实例
        
    Examples:
        >>> # 不剪枝
        >>> pruner = create_pruner("identity")
        
        >>> # Top-K 剪枝，保留 50%
        >>> pruner = create_pruner("topk", keep_ratio=0.5)
        
        >>> # 空间池化
        >>> pruner = create_pruner("spatial", pool_size=2)
    """
    strategy = strategy.lower()
    
    pruners = {
        "identity": IdentityPruner,
        "topk": lambda: TopKPruner(keep_ratio, min_tokens),
        "attention": lambda: AttentionPruner(keep_ratio, min_tokens),
        "random": lambda: RandomPruner(keep_ratio, min_tokens),
        "spatial": lambda: SpatialPruner(kwargs.get("pool_size", 2)),
        "hybrid": lambda: HybridPruner(
            keep_ratio, min_tokens,
            kwargs.get("use_spatial", False),
            kwargs.get("spatial_pool_size", 2)
        ),
    }
    
    if strategy not in pruners:
        raise ValueError(f"Unknown pruning strategy: {strategy}. "
                        f"Available: {list(pruners.keys())}")
    
    pruner_fn = pruners[strategy]
    return pruner_fn() if callable(pruner_fn) else pruner_fn
