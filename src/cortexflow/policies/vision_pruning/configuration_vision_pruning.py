from dataclasses import dataclass


@dataclass
class VisionPruningConfig:
    enabled: bool = False
    strategy: str = "stride"
    keep_ratio: float = 0.5
