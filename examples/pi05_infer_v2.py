"""
PI0.5 Inference demo (v2) — uses PI05Processor instead of the pipeline system.

Usage:
    python pi05_demo_v2.py
"""

import os

import torch
from cortexflow import AutoPolicy
from pi05_processor import PI05Processor

# ── config ──────────────────────────────────────────────────────────
model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load policy ─────────────────────────────────────────────────────
policy = AutoPolicy.from_pretrained(model_id).to(device).eval()

# ── load processor ──────────────────────────────────────────────────
processor = PI05Processor(
    model_id, device=str(device),
    tokenizer_path="/home/x/Documents/models/paligemma-3b-pt-224/",
)

# ── load sample frame ──────────────────────────────────────────────
frame = torch.load(os.path.join(os.path.dirname(__file__), "sample_frame.pt"), weights_only=False)
print(f"frame length: {len(frame)}")

# ── inference ───────────────────────────────────────────────────────
torch.manual_seed(42)
torch.cuda.manual_seed(42)

batch = processor.preprocess(frame)
with torch.inference_mode():
    print(f"frame : {frame}")
    pred_action = policy.select_action(batch)
    print(f"pred_action : {pred_action}")
    pred_action = processor.postprocess(pred_action)
    print(f"pred_action : {pred_action}")
