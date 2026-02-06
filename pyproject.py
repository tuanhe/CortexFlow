# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project.urls]
homepage = "https://github.com/tuanhe/cortexflow"

[project]
name = "cortexflow"
version = "0.0.1"
description = "Cortexflow support inference framework"
dynamic = ["readme"]
license = { text = "Apache-2.0" }
requires-python = ">=3.10"
authors = [
    { name = "hubin", email = "moozmoon@126.com" },
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
# keywords = ["cortexflow", "huggingface", "robotics",  "machine learning", "artificial intelligence"]

dependencies = [

    # Hugging Face dependencies
    "datasets>=4.0.0,<4.2.0",
    "diffusers>=0.27.2,<0.36.0",
    "huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0",
    "accelerate>=1.10.0,<2.0.0",

    # Core dependencies
    "setuptools>=71.0.0,<81.0.0",
    "cmake>=3.29.0.1,<4.2.0",
    "einops>=0.8.0,<0.9.0",
    "opencv-python-headless>=4.9.0,<4.13.0",
    "av>=15.0.0,<16.0.0",
    "jsonlines>=4.0.0,<5.0.0",
    "packaging>=24.2,<26.0",
    "pynput>=1.7.7,<1.9.0",
    "pyserial>=3.5,<4.0",
    "wandb>=0.24.0,<0.25.0",

    "torch>=2.2.1,<2.8.0", # TODO: Bumb dependency
    "torchcodec>=0.2.1,<0.6.0; sys_platform != 'win32' and (sys_platform != 'linux' or (platform_machine != 'aarch64' and platform_machine != 'arm64' and platform_machine != 'armv7l')) and (sys_platform != 'darwin' or platform_machine != 'x86_64')", # TODO: Bumb dependency
    "torchvision>=0.21.0,<0.23.0", # TODO: Bumb dependency

    "draccus==0.10.0", # TODO: Remove ==
    "gymnasium>=1.1.1,<2.0.0",
    "rerun-sdk>=0.24.0,<0.27.0",

    # Support dependencies
    "deepdiff>=7.0.1,<9.0.0",
    "imageio[ffmpeg]>=2.34.0,<3.0.0",
    "termcolor>=2.4.0,<4.0.0",
]

# Optional dependencies
[project.optional-dependencies]

# Common
pygame-dep = ["pygame>=2.5.1,<2.7.0"]
placo-dep = ["placo>=0.9.6,<0.10.0"]
transformers-dep = ["transformers>=4.57.1,<5.0.0"]
grpcio-dep = ["grpcio==1.73.1", "protobuf>=6.31.1,<6.32.0"]

# Motors
feetech = ["feetech-servo-sdk>=1.0.0,<2.0.0"]
dynamixel = ["dynamixel-sdk>=3.7.31,<3.9.0"]

# Robots
gamepad = ["cortexflow[pygame-dep]", "hidapi>=0.14.0,<0.15.0"]
hopejr = ["cortexflow[feetech]", "cortexflow[pygame-dep]"]
lekiwi = ["cortexflow[feetech]", "pyzmq>=26.2.1,<28.0.0"]
unitree_g1 = [
    "pyzmq>=26.2.1,<28.0.0",
    "onnxruntime>=1.16.0,<2.0.0"
]
reachy2 = ["reachy2_sdk>=1.0.15,<1.1.0"]
kinematics = ["cortexflow[placo-dep]"]
intelrealsense = [
    "pyrealsense2>=2.55.1.6486,<2.57.0 ; sys_platform != 'darwin'",
    "pyrealsense2-macosx>=2.54,<2.55.0 ; sys_platform == 'darwin'",
]
phone = ["hebi-py>=2.8.0,<2.12.0", "teleop>=0.1.0,<0.2.0", "fastapi<1.0"]

# Policies
wallx = [
    "transformers==4.49.0",
    "peft==0.17.1",
    "scipy==1.15.3",
    "torchdiffeq==0.2.5",
    "qwen_vl_utils==0.0.11"
]
pi = ["transformers @ git+https://github.com/huggingface/transformers.git@fix/cortexflow_openpi", "scipy>=1.10.1,<1.15"]
smolvla = ["cortexflow[transformers-dep]", "num2words>=0.5.14,<0.6.0", "accelerate>=1.7.0,<2.0.0", "safetensors>=0.4.3,<1.0.0"]
groot = [
    "cortexflow[transformers-dep]",
    "peft>=0.13.0,<1.0.0",
    "dm-tree>=0.1.8,<1.0.0",
    "timm>=1.0.0,<1.1.0",
    "safetensors>=0.4.3,<1.0.0",
    "Pillow>=10.0.0,<13.0.0",
    "decord>=0.6.0,<1.0.0; (platform_machine == 'AMD64' or platform_machine == 'x86_64')",
    "ninja>=1.11.1,<2.0.0",
    "flash-attn>=2.5.9,<3.0.0 ; sys_platform != 'darwin'"
]
sarm = ["cortexflow[transformers-dep]", "faker>=33.0.0,<35.0.0", "matplotlib>=3.10.3,<4.0.0", "qwen-vl-utils>=0.0.14,<0.1.0"]
xvla = ["cortexflow[transformers-dep]"]
hilserl = ["cortexflow[transformers-dep]", "gym-hil>=0.1.13,<0.2.0", "cortexflow[grpcio-dep]", "cortexflow[placo-dep]"]

# Features
async = ["cortexflow[grpcio-dep]", "matplotlib>=3.10.3,<4.0.0"]
peft = ["cortexflow[transformers-dep]", "peft>=0.18.0,<1.0.0"]

# Development
dev = ["pre-commit>=3.7.0,<5.0.0", "debugpy>=1.8.1,<1.9.0", "cortexflow[grpcio-dep]", "grpcio-tools==1.73.1", "mypy>=1.19.1"]
test = ["pytest>=8.1.0,<9.0.0", "pytest-timeout>=2.4.0,<3.0.0", "pytest-cov>=5.0.0,<8.0.0", "mock-serial>=0.0.1,<0.1.0 ; sys_platform != 'win32'"]
video_benchmark = ["scikit-image>=0.23.2,<0.26.0", "pandas>=2.2.2,<2.4.0"]

# Simulation
aloha = ["gym-aloha>=0.1.2,<0.2.0"]
pusht = ["gym-pusht>=0.1.5,<0.2.0", "pymunk>=6.6.0,<7.0.0"] # TODO: Fix pymunk version in gym-pusht instead
libero = ["cortexflow[transformers-dep]", "hf-libero>=0.1.3,<0.2.0"]
metaworld = ["metaworld==3.0.0"]

# All
all = [
    "cortexflow[dynamixel]",
    "cortexflow[gamepad]",
    "cortexflow[hopejr]",
    "cortexflow[lekiwi]",
    "cortexflow[reachy2]",
    "cortexflow[kinematics]",
    "cortexflow[intelrealsense]",
    # "cortexflow[wallx]",
    # "cortexflow[pi]", TODO(Pepijn): Update pi to transformers v5
    "cortexflow[smolvla]",
    # "cortexflow[groot]", TODO(Steven): Gr00t requires specific installation instructions for flash-attn
    "cortexflow[xvla]",
    "cortexflow[hilserl]",
    "cortexflow[async]",
    "cortexflow[dev]",
    "cortexflow[test]",
    "cortexflow[video_benchmark]",
    "cortexflow[aloha]",
    "cortexflow[pusht]",
    "cortexflow[phone]",
    "cortexflow[libero]",
    "cortexflow[metaworld]",
    "cortexflow[sarm]",
    "cortexflow[peft]",
]

[project.scripts]
cortexflow-calibrate="cortexflow.scripts.cortexflow_calibrate:main"
cortexflow-find-cameras="cortexflow.scripts.cortexflow_find_cameras:main"
cortexflow-find-port="cortexflow.scripts.cortexflow_find_port:main"
cortexflow-record="cortexflow.scripts.cortexflow_record:main"
cortexflow-replay="cortexflow.scripts.cortexflow_replay:main"
cortexflow-setup-motors="cortexflow.scripts.cortexflow_setup_motors:main"
cortexflow-teleoperate="cortexflow.scripts.cortexflow_teleoperate:main"
cortexflow-eval="cortexflow.scripts.cortexflow_eval:main"
cortexflow-train="cortexflow.scripts.cortexflow_train:main"
cortexflow-train-tokenizer="cortexflow.scripts.cortexflow_train_tokenizer:main"
cortexflow-dataset-viz="cortexflow.scripts.cortexflow_dataset_viz:main"
cortexflow-info="cortexflow.scripts.cortexflow_info:main"
cortexflow-find-joint-limits="cortexflow.scripts.cortexflow_find_joint_limits:main"
cortexflow-imgtransform-viz="cortexflow.scripts.cortexflow_imgtransform_viz:main"
cortexflow-edit-dataset="cortexflow.scripts.cortexflow_edit_dataset:main"

# ---------------- Tool Configurations ----------------
[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
target-version = "py310"
line-length = 110
exclude = ["tests/artifacts/**/*.safetensors", "*_pb2.py", "*_pb2_grpc.py"]

[tool.ruff.lint]
# E, W: pycodestyle errors and warnings
# F: PyFlakes
# I: isort
# UP: pyupgrade
# B: flake8-bugbear (good practices, potential bugs)
# C4: flake8-comprehensions (more concise comprehensions)
# A: flake8-builtins (shadowing builtins)
# SIM: flake8-simplify
# RUF: Ruff-specific rules
# D: pydocstyle (for docstring style/formatting)
# S: flake8-bandit (some security checks, complements Bandit)
# T20: flake8-print (discourage print statements in production code)
# N: pep8-naming
# TODO: Uncomment rules when ready to use
select = [
    "E", "W", "F", "I", "B", "C4", "T20", "N", "UP", "SIM" #, "A", "S", "D", "RUF"
]
ignore = [
    "E501", # Line too long
    "T201", # Print statement found
    "T203", # Pprint statement found
    "B008", # Perform function call in argument defaults
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401", "F403"]
"src/cortexflow/policies/wall_x/**" = ["N801", "N812", "SIM102", "SIM108", "SIM210", "SIM211", "B006", "B007", "SIM118"] # Supprese these as they are coming from original Qwen2_5_vl code TODO(pepijn): refactor original

[tool.ruff.lint.isort]
combine-as-imports = true
known-first-party = ["cortexflow"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
docstring-code-format = true

[tool.bandit]
exclude_dirs = [
    "tests",
    "benchmarks",
    "src/cortexflow/datasets/push_dataset_to_hub",
]
skips = ["B101", "B311", "B404", "B603", "B615"]

[tool.typos]
default.extend-ignore-re = [
    "(?Rm)^.*(#|//)\\s*spellchecker:disable-line$",                      # spellchecker:disable-line
    "(?s)(#|//)\\s*spellchecker:off.*?\\n\\s*(#|//)\\s*spellchecker:on", # spellchecker:<on|off>
]
default.extend-ignore-identifiers-re = [
    # Add individual words here to ignore them
    "2nd",
    "pn",
    "ser",
    "ein",
    "thw",
    "inpt",
    "ROBOTIS",
]

# TODO: Uncomment when ready to use
# [tool.interrogate]
# ignore-init-module = true
# ignore-init-method = true
# ignore-nested-functions = false
# ignore-magic = false
# ignore-semiprivate = false
# ignore-private = false
# ignore-property-decorators = false
# ignore-module = false
# ignore-setters = false
# fail-under = 80
# output-format = "term-missing"
# color = true
# paths = ["src/cortexflow"]

# TODO: Enable mypy gradually module by module across multiple PRs
# Uncomment [tool.mypy] first, then uncomment individual module overrides as they get proper type annotations

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
follow_imports = "skip"
# warn_return_any = true
# warn_unused_configs = true
# strict = true
# disallow_untyped_defs = true
# disallow_incomplete_defs = true
# check_untyped_defs = true

[[tool.mypy.overrides]]
module = "cortexflow.*"
ignore_errors = true

[[tool.mypy.overrides]]
module = "cortexflow.envs.*"
ignore_errors = false


# [[tool.mypy.overrides]]
# module = "cortexflow.utils.*"
# ignore_errors = false

[[tool.mypy.overrides]]
module = "cortexflow.configs.*"
ignore_errors = false

# extra strictness for configs
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true

[[tool.mypy.overrides]]
module = "cortexflow.optim.*"
ignore_errors = false

[[tool.mypy.overrides]]
module = "cortexflow.model.*"
ignore_errors = false

# [[tool.mypy.overrides]]
# module = "cortexflow.processor.*"
# ignore_errors = false

# [[tool.mypy.overrides]]
# module = "cortexflow.datasets.*"
# ignore_errors = false

[[tool.mypy.overrides]]
module = "cortexflow.cameras.*"
ignore_errors = false

# [[tool.mypy.overrides]]
# module = "cortexflow.motors.*"
# ignore_errors = false

# [[tool.mypy.overrides]]
# module = "cortexflow.robots.*"
# ignore_errors = false

# [[tool.mypy.overrides]]
# module = "cortexflow.teleoperators.*"
# ignore_errors = false

# [[tool.mypy.overrides]]
# module = "cortexflow.policies.*"
# ignore_errors = false

# [[tool.mypy.overrides]]
# module = "cortexflow.rl.*"
# ignore_errors = false


# [[tool.mypy.overrides]]
# module = "cortexflow.async_inference.*"
# ignore_errors = false

[[tool.mypy.overrides]]
module = "cortexflow.transport.*"
ignore_errors = false

# [[tool.mypy.overrides]]
# module = "cortexflow.scripts.*"
# ignore_errors = false

[tool.uv]
# wallx requires transformers==4.49.0 which conflicts with other extras that need >=4.53.0
conflicts = [
    [
        { extra = "wallx" },
        { extra = "transformers-dep" },
    ],
    [
        { extra = "wallx" },
        { extra = "pi" },
    ],
    [
        { extra = "wallx" },
        { extra = "smolvla" },
    ],
    [
        { extra = "wallx" },
        { extra = "groot" },
    ],
    [
        { extra = "wallx" },
        { extra = "xvla" },
    ],
    [
        { extra = "wallx" },
        { extra = "sarm" },
    ],
    [
        { extra = "wallx" },
        { extra = "hilserl" },
    ],
    [
        { extra = "wallx" },
        { extra = "libero" },
    ],
    [
        { extra = "wallx" },
        { extra = "peft" },
    ],
    [
        { extra = "wallx" },
        { extra = "all" },
    ],
    # pi uses custom branch which conflicts with transformers-dep
    [
        { extra = "pi" },
        { extra = "transformers-dep" },
    ],
    [
        { extra = "pi" },
        { extra = "smolvla" },
    ],
    [
        { extra = "pi" },
        { extra = "groot" },
    ],
    [
        { extra = "pi" },
        { extra = "xvla" },
    ],
    [
        { extra = "pi" },
        { extra = "sarm" },
    ],
    [
        { extra = "pi" },
        { extra = "hilserl" },
    ],
    [
        { extra = "pi" },
        { extra = "libero" },
    ],
    [
        { extra = "pi" },
        { extra = "peft" },
    ],
    [
        { extra = "pi" },
        { extra = "all" },
    ],
]
