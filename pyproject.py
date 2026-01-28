[build-system]
requires = ["setuptools>=61.0", "wheel"]  # 构建依赖
build-backend = "setuptools.build_meta"


[project]
name = "cortexflow"  # 包名（需与根目录下的cortexflow文件夹一致）
version = "0.1.0"  # 初始版本号
authors = [
  { name = "tuanhe", email = "moozmoon@126.com" },
]
description = "VLA Inference Framework for Autonomous Driving/Embodied AI"  # 项目描述
readme = "README.md"  # 关联README文件
license = { file = "LICENSE" }  # 若有LICENSE文件则添加（可选）
classifiers = [  # 项目分类（便于PyPI检索）
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "License :: OSI Approved :: MIT License",  # 需与实际协议一致
  "Operating System :: OS Independent",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Topic :: Software Development :: Libraries :: Python Modules",
]
requires-python = ">=3.10"  # Python版本要求（需匹配PyTorch/Triton支持的版本）
dependencies = [  # 核心依赖（根据项目实际使用的库调整）
  "torch>=2.1.0",  # 需支持bfloat16、CUDAGraph的版本
  "triton>=2.1.0",  # Triton算子依赖
  "numpy>=1.24.0",
  "einops>=0.6.1",  # 可选：若用einops处理张量形状
  "PyYAML>=6.0",  # 可选：若用YAML加载配置
]


[project.optional-dependencies]
dev = [  # 开发依赖（测试、格式化等）
  "pytest>=7.0",  # 单元测试
  "black>=23.0",  # 代码格式化
  "isort>=5.0",  # 导入排序
  "mypy>=1.0",  # 类型检查（可选）
]


[project.urls]
"Homepage" = "https://github.com/your-username/cortexflow"  # 项目仓库地址（可选）
"Bug Tracker" = "https://github.com/your-username/cortexflow/issues"  # 可选
