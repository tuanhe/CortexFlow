"""
PI0.5 inference benchmark — iterate over all frames in the dataset,
measure end-to-end latency (preprocess + model + postprocess).

Usage:
    python benchmarks/bench_pi05_infer.py              # run all frames
    python benchmarks/bench_pi05_infer.py -n 100       # run 100 frames
    python benchmarks/bench_pi05_infer.py --num_frames 50 --warmup 3
"""

import argparse
import os
import sys
import time

import torch

# allow importing lerobot_datasets from benchmarks/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lerobot_datasets.lerobot_dataset import LeRobotDataset

from cortexflow import AutoPolicy
from cortexflow.policies.factory import make_pre_post_processors


def parse_args():
    parser = argparse.ArgumentParser(description="PI0.5 inference benchmark")
    parser.add_argument("-n", "--num_frames", type=int, default=None,
                        help="Number of frames to benchmark (default: all)")
    parser.add_argument("--model_id", type=str,
                        default="/home/x/Documents/models/lerobot/pi05_base_migrated/")
    parser.add_argument("--dataset_id", type=str, default="lerobot/libero")
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def stats(times):
    n = len(times)
    times_ms = sorted(t * 1000 for t in times)
    return {
        "mean": sum(times_ms) / n,
        "min": times_ms[0],
        "max": times_ms[-1],
        "median": times_ms[n // 2],
        "p95": times_ms[int(n * 0.95)],
        "p99": times_ms[int(n * 0.99)],
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ────────────────────────────────────────────────────
    print(f"Loading model from {args.model_id} ...")
    policy = AutoPolicy.from_pretrained(args.model_id).to(device).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ── load dataset ──────────────────────────────────────────────────
    print(f"Loading dataset {args.dataset_id} ...")
    dataset = LeRobotDataset(args.dataset_id)
    total_frames = len(dataset)
    bench_frames = min(args.num_frames, total_frames) if args.num_frames else total_frames
    print(f"Total frames: {total_frames}, benchmarking: {bench_frames}")

    # ── warmup ────────────────────────────────────────────────────────
    print(f"Warming up ({args.warmup} iterations) ...")
    warmup_frame = dict(dataset[0])
    for _ in range(args.warmup):
        policy.reset()
        batch = preprocess(warmup_frame)
        with torch.inference_mode():
            policy.select_action(batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ── benchmark ─────────────────────────────────────────────────────
    print("Running benchmark ...")

    preprocess_times = []
    model_times = []
    postprocess_times = []
    total_times = []

    for i in range(bench_frames):
        policy.reset()
        t_start = time.perf_counter()

        # preprocess
        frame = dict(dataset[i])
        t0 = time.perf_counter()
        batch = preprocess(frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # model inference
        with torch.inference_mode():
            action = policy.select_action(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # postprocess
        _ = postprocess(action)
        t3 = time.perf_counter()

        preprocess_times.append(t1 - t0)
        model_times.append(t2 - t1)
        postprocess_times.append(t3 - t2)
        total_times.append(t3 - t_start)

        if (i + 1) % 100 == 0 or (i + 1) == bench_frames:
            avg_total = sum(total_times) / len(total_times) * 1000
            print(f"  [{i + 1}/{bench_frames}]  avg total: {avg_total:.2f} ms/frame")

    # ── results ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Benchmark Results  ({bench_frames} frames, device={device})")
    print("=" * 60)

    for name, times in [
        ("Preprocess ", preprocess_times),
        ("Model      ", model_times),
        ("Postprocess", postprocess_times),
        ("Total      ", total_times),
    ]:
        s = stats(times)
        print(
            f"  {name}:  mean={s['mean']:7.2f} ms  "
            f"median={s['median']:7.2f} ms  "
            f"p95={s['p95']:7.2f} ms  "
            f"p99={s['p99']:7.2f} ms  "
            f"min={s['min']:7.2f} ms  "
            f"max={s['max']:7.2f} ms"
        )

    throughput = bench_frames / sum(total_times)
    print(f"\n  Throughput: {throughput:.2f} frames/sec")
    print(f"  Total time: {sum(total_times):.2f} sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
