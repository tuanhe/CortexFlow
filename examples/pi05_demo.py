import os

import torch

from cortexflow.policies.factory import make_pre_post_processors

from cortexflow import AutoPolicy

# load a policy
# model_id = "/home/x/Documents/models/lerobot/pi05_base/"  # <- swap checkpoint
model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"  # <- swap checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = AutoPolicy.from_pretrained(model_id).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# load a sample frame
frame = torch.load(os.path.join(os.path.dirname(__file__), "sample_frame.pt"), weights_only=False)

# ── inference ───────────────────────────────────────────────────────
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)


print(f"frame length: {len(frame)}")

batch = preprocess(frame)
with torch.inference_mode():
    # print(f"frame : {frame}")
    pred_action = policy.select_action(batch)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    print(f"pred_action : {pred_action}")
    pred_action = postprocess(pred_action)
    print(f"pred_action : {pred_action}")
    