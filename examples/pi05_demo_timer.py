import os
import time

import torch
from cortexflow.policies.factory import make_pre_post_processors

from cortexflow import AutoPolicy

# load a policy
# model_id = "/home/x/Documents/models/lerobot/pi05_base/"  # <- swap checkpoint
model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"  # <- swap checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t0 = time.perf_counter()
policy = AutoPolicy.from_pretrained(model_id).to(device).eval()
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(f"[Timer] Model load:    {(time.perf_counter() - t0) * 1000:.2f} ms")

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# load a sample frame
frame = torch.load(os.path.join(os.path.dirname(__file__), "sample_frame.pt"), weights_only=False)

print(f"frame length: {len(frame)}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# ── warmup (first run includes CUDA kernel compilation overhead) ──
with torch.inference_mode():
    warmup_batch = preprocess(frame)
    _ = policy.predict_action_chunk(warmup_batch)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("[Timer] Warmup done")

# ── timed run ─────────────────────────────────────────────────────
# reset to clear action cache, force a real model forward pass
policy.reset()
torch.manual_seed(42)
torch.cuda.manual_seed(42)

t_pre_start = time.perf_counter()
batch = preprocess(frame)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_pre_end = time.perf_counter()

with torch.inference_mode():
    t_infer_start = time.perf_counter()
    pred_action = policy.select_action(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_infer_end = time.perf_counter()

t_post_start = time.perf_counter()
pred_action = postprocess(pred_action)
t_post_end = time.perf_counter()

t_total = t_post_end - t_pre_start

print(f"\npred_action : {pred_action}")
print(f"\n{'=' * 50}")
print(f"[Timer] Preprocess:    {(t_pre_end - t_pre_start) * 1000:7.2f} ms")
print(f"[Timer] Inference:     {(t_infer_end - t_infer_start) * 1000:7.2f} ms")
print(f"[Timer] Postprocess:   {(t_post_end - t_post_start) * 1000:7.2f} ms")
print(f"[Timer] Total:         {t_total * 1000:7.2f} ms")
print(f"{'=' * 50}")
