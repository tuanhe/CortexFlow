from __future__ import annotations

from pathlib import Path
from typing import Any

from cortexflow.configs.policies import PreTrainedConfig
from cortexflow.policies.factory import get_policy_class
from cortexflow.policies.pretrained import PreTrainedPolicy


class AutoPolicy:
    """Automatically select and load the correct policy class based on model config.

    Similar to HuggingFace's AutoModel, this class inspects the config.json of a
    pretrained model to determine which concrete Policy class to instantiate.

    Example::

        from cortexflow import AutoPolicy
        policy = AutoPolicy.from_pretrained("physical-intelligence/pi0.5_base").to("cuda").eval()
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoPolicy is designed to be instantiated using the "
            "`AutoPolicy.from_pretrained(pretrained_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        **kwargs: Any,
    ) -> PreTrainedPolicy:
        """Load a pretrained policy, automatically selecting the correct class.

        Args:
            pretrained_name_or_path: Either a repo ID on HuggingFace Hub or a local
                directory path containing a saved policy (with config.json).
            **kwargs: Additional keyword arguments forwarded to the underlying
                policy's ``from_pretrained`` method.

        Returns:
            An instance of the appropriate ``PreTrainedPolicy`` subclass.
        """
        config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        policy_cls = get_policy_class(config.type)
        return policy_cls.from_pretrained(
            pretrained_name_or_path, config=config, **kwargs
        )
