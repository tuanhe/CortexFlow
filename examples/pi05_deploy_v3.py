"""
PI0.5 deployment demo (v3) — directly construct processor objects from src/lerobot,
no JSON config files, no pi05_processor.py.

Usage:
    python examples/pi05_deploy_v3.py
"""

import cv2
import numpy as np
import torch
from cortexflow.configs.types import FeatureType, NormalizationMode, PolicyFeature
from cortexflow import AutoPolicy
from cortexflow.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from cortexflow.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from cortexflow.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

# ── config ──────────────────────────────────────────────────────────
model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"
tokenizer_path = "/home/x/Documents/models/paligemma-3b-pt-224/"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── feature & normalization definitions (matches PI05Config defaults) ──
input_features = {
    "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
}
output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
}
norm_map = {
    FeatureType.VISUAL: NormalizationMode.MEAN_STD,
    FeatureType.STATE: NormalizationMode.MIN_MAX,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}

# ── build preprocessor pipeline ─────────────────────────────────────
preprocessor = PolicyProcessorPipeline(
    steps=[
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**input_features, **output_features},
            norm_map=norm_map,
            stats=None,  # no stats file → normalization is identity
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=32),
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_path,
            max_length=200,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=device),
    ],
    name="policy_preprocessor",
    to_transition=batch_to_transition,
    to_output=transition_to_batch,
)

# ── build postprocessor pipeline ─────────────────────────────────────
postprocessor = PolicyProcessorPipeline(
    steps=[
        UnnormalizerProcessorStep(
            features=output_features,
            norm_map=norm_map,
            stats=None,
        ),
        DeviceProcessorStep(device="cpu"),
    ],
    name="policy_postprocessor",
    to_transition=policy_action_to_transition,
    to_output=transition_to_policy_action,
)

# ── load policy ─────────────────────────────────────────────────────
policy = AutoPolicy.from_pretrained(model_id).to(device).eval()


# ── helper: BGR numpy → model tensor ────────────────────────────────
def image_to_tensor(bgr: np.ndarray, size=(256, 256)) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def predict(
    image1: np.ndarray,
    image2: np.ndarray,
    state: np.ndarray,
    task: str,
) -> np.ndarray:
    """Run one inference step from raw camera/robot inputs."""
    frame = {
        "observation.images.image": image_to_tensor(image1),
        "observation.images.image2": image_to_tensor(image2),
        "observation.state": torch.from_numpy(state).float(),
        "task": task,
    }
    batch = preprocessor(frame)
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = postprocessor(action)
    return action.squeeze(0).numpy()


# ── demo: simulate camera input from dataset ────────────────────────
if __name__ == "__main__":
    from cortexflow.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("lerobot/libero")
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    frame = dict(dataset[from_idx])

    def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
        rgb = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    img1_bgr = tensor_to_bgr(frame["observation.images.image"])
    img2_bgr = tensor_to_bgr(frame["observation.images.image2"])
    state = frame["observation.state"].numpy()
    task = frame["task"]

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    action = predict(img1_bgr, img2_bgr, state, task)
    print(f"task   : {task}")
    print(f"state  : {state}")
    print(f"action : {action}")
