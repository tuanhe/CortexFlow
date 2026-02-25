"""
PI0.5 Processor v2 — no JSON config files, all parameters explicit.

Usage:
    from pi05_processor_v2 import PI05Processor

    processor = PI05Processor(device="cuda")
    batch     = processor.preprocess(images, state, task)
    action    = processor.postprocess(raw_action)
"""

import cv2
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer


class PI05Processor:
    """Self-contained pre/post processor for PI0.5 inference.

    All parameters are explicit — no JSON configs, no safetensors stats files.
    """

    def __init__(
        self,
        tokenizer_path: str = "/home/x/Documents/models/paligemma-3b-pt-224/",
        device: str = "cuda",
        image_size: tuple[int, int] = (256, 256),
        max_state_dim: int = 32,
        tokenizer_max_length: int = 200,
    ):
        self.device = device
        self.image_size = image_size
        self.max_state_dim = max_state_dim
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ────────────────────────────────────────────────────────────────
    #  preprocess
    # ────────────────────────────────────────────────────────────────

    def preprocess(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        task: str,
    ) -> dict:
        """Build a model-ready batch from raw inputs.

        Args:
            images: Camera name → BGR numpy image (any resolution).
                    e.g. {"image": cam1_bgr, "image2": cam2_bgr}
            state:  Robot joint state, float array of shape (D,).
            task:   Natural language task description.

        Returns:
            Dict ready to feed into policy.select_action().
        """
        batch: dict[str, Tensor | str] = {}

        # 1) Images: BGR numpy → [1, C, H, W] float32 tensor in [0, 1]
        for name, bgr in images.items():
            key = f"observation.images.{name}"
            batch[key] = self._image_to_tensor(bgr).unsqueeze(0)

        # 2) State: numpy → pad to max_state_dim → [1, D] tensor
        state_t = torch.from_numpy(state).float()
        state_t = _pad(state_t, self.max_state_dim).unsqueeze(0)
        batch["observation.state"] = state_t

        # 3) Task + state → discretized prompt → language tokens
        prompt = _build_prompt(state_t[0], task)
        tokens = self.tokenizer(
            [prompt],
            max_length=self.tokenizer_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        batch["observation.language.tokens"] = tokens["input_ids"]
        batch["observation.language.attention_mask"] = tokens["attention_mask"].to(torch.bool)

        # 4) Move everything to device
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    # ────────────────────────────────────────────────────────────────
    #  postprocess
    # ────────────────────────────────────────────────────────────────

    def postprocess(self, action: Tensor) -> np.ndarray:
        """Model output tensor → numpy action on CPU.

        No unnormalization needed for pi05_base (stats files don't exist).
        """
        return action.cpu().squeeze(0).numpy()

    # ────────────────────────────────────────────────────────────────
    #  helpers
    # ────────────────────────────────────────────────────────────────

    def _image_to_tensor(self, bgr: np.ndarray) -> Tensor:
        """BGR uint8 numpy → [C, H, W] float32 tensor in [0, 1]."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


# ════════════════════════════════════════════════════════════════════
#  Pure helper functions
# ════════════════════════════════════════════════════════════════════

def _pad(x: Tensor, target_dim: int) -> Tensor:
    """Pad (or truncate) last dimension to target_dim."""
    d = x.shape[-1]
    if d >= target_dim:
        return x[..., :target_dim]
    zeros = torch.zeros(target_dim - d, device=x.device, dtype=x.dtype)
    return torch.cat([x, zeros], dim=-1)


def _build_prompt(state: Tensor, task: str) -> str:
    """Discretize state into 256 bins, build prompt string."""
    bins = np.linspace(-1, 1, 257)[:-1]
    disc = np.digitize(state.cpu().numpy(), bins) - 1
    state_str = " ".join(map(str, disc))
    text = task.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {text}, State: {state_str};\nAction: "
