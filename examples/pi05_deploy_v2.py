"""
PI0.5 deployment demo (v2) — processor without JSON config dependency.

Usage:
    python examples/pi05_deploy_v2.py
"""

import os

import cv2
import numpy as np
import torch
from cortexflow import AutoPolicy

import sys
sys.path.insert(0, ".")
from pi05_processor_v2 import PI05Processor

# ── config ──────────────────────────────────────────────────────────
model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load policy & processor ─────────────────────────────────────────
policy = AutoPolicy.from_pretrained(model_id).to(device).eval()
processor = PI05Processor(device=str(device))

# ── load sample frame to simulate camera input ──────────────────────
frame = torch.load(os.path.join(os.path.dirname(__file__), "sample_frame.pt"), weights_only=False)

# Convert dataset tensors to what cameras would produce (BGR uint8 numpy)
def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    rgb = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

images = {
    "image": tensor_to_bgr(frame["observation.images.image"]),
    "image2": tensor_to_bgr(frame["observation.images.image2"]),
}
state = frame["observation.state"].numpy()
task = frame["task"]

# ── inference ───────────────────────────────────────────────────────
torch.manual_seed(42)
torch.cuda.manual_seed(42)

batch = processor.preprocess(images, state, task)
with torch.inference_mode():
    raw_action = policy.select_action(batch)
action = processor.postprocess(raw_action)

print(f"task   : {task}")
print(f"state  : {state}")
print(f"action : {action}")
