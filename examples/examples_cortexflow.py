"""
CortexFlow - 使用示例

演示如何使用模块化的 CortexFlow
"""

import torch
import numpy as np

# 示例 1: 基础使用 - Pi0.5
def example_basic_pi05():
    """最基本的 Pi0.5 推理示例"""
    from cortexflow import create_inference, BackendType
    
    print("="*60)
    print("示例 1: 基础 Pi0.5 推理")
    print("="*60)
    
    # 创建推理引擎
    engine = create_inference("pi05", checkpoint_path="checkpoints/pi05.pt",
        num_views=2,
        chunk_size=10,
        backend=BackendType.TRITON,
        tokenizer_path="tokenizers/pi05",
    )
    
    # 准备输入
    images = torch.randn(2, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
    task_prompt = "拿起红色方块"
    state_tokens = np.array([128, 64, 32, 200, 150])
    
    # 执行推理
    actions = engine.predict(
        observation_images=images,
        task_prompt=task_prompt,
        state_tokens=state_tokens,
    )
    
    print(f"✓ 输入图像: {images.shape}")
    print(f"✓ 任务: {task_prompt}")
    print(f"✓ 预测动作: {actions.shape}")
    print()


# 示例 2: 不同后端对比
def example_backend_comparison():
    """对比不同后端的性能"""
    from cortexflow import create_inference, BackendType
    from cortexflow.utils import benchmark_inference
    
    print("="*60)
    print("示例 2: 后端性能对比")
    print("="*60)
    
    backends = [
        (BackendType.TRITON, "Triton 优化内核"),
        (BackendType.PYTORCH, "纯 PyTorch 实现"),
    ]
    
    for backend_type, description in backends:
        print(f"\n[{backend_type.value}] {description}")
        print("-"*50)
        
        engine = create_inference("pi05", checkpoint_path="checkpoints/pi05.pt",
            num_views=2,
            chunk_size=10,
            backend=backend_type,
        )
        
        # 性能测试
        stats = benchmark_inference(engine, num_iterations=50)
        
        print(f"平均延迟: {stats['avg_latency_ms']:.2f} ms")
        print(f"吞吐量: {stats['throughput_fps']:.2f} FPS")
    
    print()


# 示例 3: 自定义配置
def example_custom_config():
    """使用完全自定义的配置"""
    from cortexflow import VLAInferenceEngine
    from cortexflow.configs import ModelConfig, InferenceConfig, VLAModelType, BackendType
    import torch
    
    print("="*60)
    print("示例 3: 自定义配置")
    print("="*60)
    
    # 自定义模型配置
    model_config = ModelConfig(
        model_type=VLAModelType.PI05,
        vision_encoder_dim=1152,
        encoder_dim=2048,
        decoder_dim=1024,
        num_encoder_layers=18,
        num_decoder_layers=18,
        action_dim=32,
        chunk_size=15,  # 自定义 chunk size
        num_diffusion_steps=10,
    )
    
    # 自定义推理配置
    inference_config = InferenceConfig(
        num_views=3,  # 3 个摄像头
        chunk_size=15,
        backend=BackendType.TRITON,
        device="cuda:0",
        dtype=torch.bfloat16,
        use_cuda_graphs=True,
        max_tokenize_len=256,
    )
    
    # 加载 checkpoint
    checkpoint = torch.load("checkpoints/pi05.pt", map_location='cpu')
    
    # 创建引擎
    engine = VLAInferenceEngine(checkpoint, model_config, inference_config)
    
    print(f"✓ 自定义配置:")
    print(f"  - 摄像头数: {inference_config.num_views}")
    print(f"  - 动作序列长度: {inference_config.chunk_size}")
    print(f"  - 后端: {inference_config.backend.value}")
    print()


# 示例 4: 实时控制循环
def example_realtime_control():
    """模拟实时机器人控制"""
    from cortexflow import create_inference, BackendType
    import time
    
    print("="*60)
    print("示例 4: 实时控制循环")
    print("="*60)
    
    engine = create_inference("pi05", checkpoint_path="checkpoints/pi05.pt",
        num_views=2,
        chunk_size=10,
        backend=BackendType.TRITON,
        use_cuda_graphs=True,  # 关键：降低延迟
    )
    
    task = "抓取物体"
    control_freq = 30  # Hz
    max_steps = 50
    
    print(f"任务: {task}")
    print(f"控制频率: {control_freq} Hz")
    print(f"最大步数: {max_steps}\n")
    
    current_state = np.array([128, 128, 128, 0, 0])
    
    for step in range(max_steps):
        step_start = time.perf_counter()
        
        # 获取观察（模拟）
        images = torch.randn(2, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
        
        # 预测动作
        actions = engine.predict(images, task, current_state)
        
        # 执行第一个动作（MPC 风格）
        action_to_execute = actions[0]
        
        # 更新状态（模拟）
        current_state += np.random.randn(5) * 5
        current_state = np.clip(current_state, 0, 255)
        
        step_time = time.perf_counter() - step_start
        
        if step % 10 == 0:
            print(f"步骤 {step:3d}: {step_time*1000:.2f}ms | "
                  f"状态: [{current_state[0]:.0f}, {current_state[1]:.0f}, ...]")
        
        # 保持控制频率
        sleep_time = max(0, 1/control_freq - step_time)
        time.sleep(sleep_time)
    
    print("\n✓ 控制循环完成")
    print()


# 示例 5: 批量处理
def example_batch_processing():
    """批量处理多个任务"""
    from cortexflow import create_inference, BackendType
    
    print("="*60)
    print("示例 5: 批量处理")
    print("="*60)
    
    engine = create_inference("pi05", checkpoint_path="checkpoints/pi05.pt",
        num_views=2,
        chunk_size=10,
        backend=BackendType.TRITON,
    )
    
    tasks = [
        "拿起红色方块",
        "打开抽屉",
        "关闭抽屉",
        "将蓝色立方体叠在红色立方体上",
        "将绿色圆柱推到左边",
    ]
    
    print(f"处理 {len(tasks)} 个任务...\n")
    
    all_actions = []
    for i, task in enumerate(tasks):
        print(f"  任务 {i+1}: '{task}'")
        
        images = torch.randn(2, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
        state_tokens = np.random.randint(0, 255, size=5)
        
        actions = engine.predict(images, task, state_tokens)
        all_actions.append(actions)
        
        print(f"    ✓ 预测了 {actions.shape[0]} 步动作")
    
    all_actions = np.stack(all_actions)
    print(f"\n✓ 总动作形状: {all_actions.shape}")
    print(f"  (任务数, 时间步, 动作维度)")
    print()


# 示例 6: OpenVLA 使用
def example_openvla():
    """使用 OpenVLA 模型"""
    from cortexflow import create_opencortexflow, BackendType
    
    print("="*60)
    print("示例 6: OpenVLA 推理")
    print("="*60)
    
    engine = create_opencortexflow(
        checkpoint_path="checkpoints/openvla.pt",
        num_views=1,
        backend=BackendType.PYTORCH,
    )
    
    # OpenVLA 通常预测单步动作
    images = torch.randn(1, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
    
    actions = engine.predict(
        observation_images=images,
        task_prompt="pick up the object",
    )
    
    print(f"✓ OpenVLA 预测: {actions.shape}")
    print(f"  注意：OpenVLA 通常输出单步动作")
    print()


# 示例 7: 性能分析
def example_profiling():
    """性能分析和调优"""
    from cortexflow import create_inference, BackendType
    from cortexflow.utils import benchmark_inference, profile_inference
    
    print("="*60)
    print("示例 7: 性能分析")
    print("="*60)
    
    engine = create_inference("openvla", checkpoint_path="checkpoints/pi05.pt",
        num_views=2,
        chunk_size=10,
        backend=BackendType.TRITON,
        use_cuda_graphs=True,
    )
    
    # 基准测试
    print("\n[基准测试]")
    stats = benchmark_inference(engine, num_iterations=100)
    print(f"平均延迟: {stats['avg_latency_ms']:.2f} ms")
    print(f"吞吐量: {stats['throughput_fps']:.2f} FPS")
    
    # 详细性能分析
    print("\n[详细性能分析]")
    profile_inference(engine, num_iterations=10)
    print()


# 示例 8: 动作平滑和后处理
def example_action_postprocessing():
    """动作平滑和后处理"""
    from cortexflow import create_inference, BackendType
    from cortexflow.utils import smooth_actions, compute_action_statistics
    
    print("="*60)
    print("示例 8: 动作后处理")
    print("="*60)
    
    engine = create_inference("pi05", checkpoint_path="checkpoints/pi05.pt",
        num_views=2,
        chunk_size=10,
        backend=BackendType.TRITON,
    )
    
    images = torch.randn(2, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
    state_tokens = np.array([128, 64, 32, 200, 150])
    
    # 预测原始动作
    raw_actions = engine.predict(images, "抓取物体", state_tokens)
    
    # 计算统计信息
    stats = compute_action_statistics(raw_actions)
    print(f"\n原始动作统计:")
    print(f"  均值: {stats['mean'][:3]}")
    print(f"  标准差: {stats['std'][:3]}")
    
    # 平滑动作
    smoothed_actions = smooth_actions(raw_actions, window_size=3)
    
    stats_smoothed = compute_action_statistics(smoothed_actions)
    print(f"\n平滑后动作统计:")
    print(f"  均值: {stats_smoothed['mean'][:3]}")
    print(f"  标准差: {stats_smoothed['std'][:3]}")
    print()


def main():
    """运行所有示例"""
    examples = [
        ("基础使用", example_basic_pi05),
        ("后端对比", example_backend_comparison),
        ("自定义配置", example_custom_config),
        ("实时控制", example_realtime_control),
        ("批量处理", example_batch_processing),
        ("OpenVLA", example_openvla),
        ("性能分析", example_profiling),
        ("动作后处理", example_action_postprocessing),
    ]
    
    print("\n" + "="*60)
    print("CortexFlow - 示例演示")
    print("="*60 + "\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"✗ {name} 失败: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("="*60)
    print("所有示例完成！")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # 可以运行单个示例
    if len(sys.argv) > 1:
        example_map = {
            "basic": example_basic_pi05,
            "backends": example_backend_comparison,
            "custom": example_custom_config,
            "realtime": example_realtime_control,
            "batch": example_batch_processing,
            "openvla": example_openvla,
            "profile": example_profiling,
            "postprocess": example_action_postprocessing,
        }
        
        example_name = sys.argv[1]
        if example_name in example_map:
            example_map[example_name]()
        else:
            print(f"未知示例: {example_name}")
            print(f"可用示例: {', '.join(example_map.keys())}")
    else:
        main()
