"""
PI0.5 Inference — minimal, self-contained example.

Usage:
    python pi05_infer.py
"""

import os

import torch
from cortexflow import AutoPolicy
from cortexflow.processor import PolicyProcessorPipeline
from cortexflow.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

# ── config ──────────────────────────────────────────────────────────
model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load policy ─────────────────────────────────────────────────────
policy = AutoPolicy.from_pretrained(model_id).to(device).eval()

# ── load pre/post processors directly from pretrained JSON configs ──
preprocess = PolicyProcessorPipeline.from_pretrained(
    model_id,
    config_filename="policy_preprocessor.json",
    overrides={"device_processor": {"device": str(device)}},
    to_transition=batch_to_transition,
    to_output=transition_to_batch,
)
postprocess = PolicyProcessorPipeline.from_pretrained(
    model_id,
    config_filename="policy_postprocessor.json",
    to_transition=policy_action_to_transition,
    to_output=transition_to_policy_action,
)

# ── load sample frame ──────────────────────────────────────────────
frame = torch.load(os.path.join(os.path.dirname(__file__), "sample_frame.pt"), weights_only=False)
print(f"frame length: {len(frame)}")

# ── inference ───────────────────────────────────────────────────────
torch.manual_seed(42)
torch.cuda.manual_seed(42)

batch = preprocess(frame)
with torch.inference_mode():
    print(f"frame : {frame}")
    pred_action = policy.select_action(batch)
    print(f"pred_action : {pred_action}")
    pred_action = postprocess(pred_action)
    print(f"pred_action : {pred_action}")
