from openpi.training import config as _config
print(f"trace log ")
from openpi.policies import policy_config
print(f"trace log ")
from openpi.shared import download

print(f"trace log ")


config = _config.get_config("pi05_libero")
checkpoint_dir = "/home/x/Documents/models/pi05_libero_pytorch/"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

print(f"policy : {policy}")


# Run inference (same API as JAX)
# action_chunk = policy.infer(example)["actions"]