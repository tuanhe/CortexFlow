import logging
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from cortexflow.policies.vision_pruning.configuration_vision_pruning import VisionPruningConfig

logger = logging.getLogger(__name__)


class VisionPrunerRegistry:
    """Registry for vision pruning strategies."""

    _registry: dict[str, type["VisionPruner"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(pruner_cls):
            cls._registry[name] = pruner_cls
            return pruner_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type["VisionPruner"]:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown vision pruning strategy: {name!r}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]


class VisionPruner(ABC):
    """Base class for vision token pruning strategies."""

    def __init__(self, config: VisionPruningConfig):
        self.config = config

    @abstractmethod
    def prune(self, img_emb: Tensor) -> Tensor:
        """Prune vision tokens from image embeddings.

        Args:
            img_emb: Image embeddings of shape [B, N, D].

        Returns:
            Pruned embeddings of shape [B, M, D] where M <= N.
        """


@VisionPrunerRegistry.register("stride")
class StridePruner(VisionPruner):
    """Uniform stride-based vision token pruning."""

    def prune(self, img_emb: Tensor) -> Tensor:
        if not self.config.enabled:
            return img_emb

        n_tokens = img_emb.shape[1]
        n_keep = max(1, int(n_tokens * self.config.keep_ratio))
        indices = torch.linspace(0, n_tokens - 1, n_keep, dtype=torch.long, device=img_emb.device)
        return img_emb[:, indices, :]


def create_vision_pruner(config: VisionPruningConfig) -> VisionPruner:
    """Factory function to create a vision pruner from config."""
    pruner_cls = VisionPrunerRegistry.get(config.strategy)
    return pruner_cls(config)
