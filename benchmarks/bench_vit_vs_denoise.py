"""Benchmark: ViT / LLM Prefill / Denoise breakdown.

Patches model internals to measure:
  - ViT:           embed_image (SigLIP vision encoder)
  - LLM Prefill:   paligemma_with_expert.forward (KV-cache prefill, use_cache=True)
  - Denoise Loop:  10-step denoising (denoise_step × N)

Usage:
  python benchmarks/bench_vit_vs_denoise.py -n 10 --warmup 3
"""

import argparse
import os
import time

import torch

from cortexflow import AutoPolicy
from cortexflow.policies.factory import make_pre_post_processors


def patch_model(model):
    """Monkey-patch model internals with timing hooks."""
    model._timing = {}
    pwe = model.paligemma_with_expert

    # 1. Patch embed_image → ViT timing
    _orig_embed_image = pwe.embed_image

    def timed_embed_image(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_embed_image(*args, **kwargs)
        torch.cuda.synchronize()
        model._timing.setdefault("vit", []).append((time.perf_counter() - t0) * 1000)
        return result

    pwe.embed_image = timed_embed_image

    # 2. Patch paligemma_with_expert.forward → distinguish prefill vs denoise
    _orig_pwe_forward = pwe.forward

    def timed_pwe_forward(*args, **kwargs):
        use_cache = kwargs.get("use_cache", False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_pwe_forward(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        if use_cache:
            model._timing["llm_prefill"] = elapsed
        else:
            model._timing.setdefault("llm_denoise", []).append(elapsed)
        return result

    pwe.forward = timed_pwe_forward

    # 3. Patch denoise_step → per-step timing
    _orig_denoise_step = model.denoise_step

    def timed_denoise_step(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_denoise_step(*args, **kwargs)
        torch.cuda.synchronize()
        model._timing.setdefault("denoise_steps", []).append((time.perf_counter() - t0) * 1000)
        return result

    model.denoise_step = timed_denoise_step


def run_benchmark(n_runs, warmup):
    model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = AutoPolicy.from_pretrained(model_id).to(device).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    frame = torch.load(
        os.path.join(os.path.dirname(__file__), "..", "examples", "sample_frame.pt"),
        weights_only=False,
    )

    patch_model(policy.model)

    # Warmup
    for _ in range(warmup):
        policy.reset()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        batch = preprocess(frame)
        with torch.inference_mode():
            policy.select_action(batch)
        policy.model._timing.clear()
    print(f"[Warmup] {warmup} runs done\n")

    # Header
    print(f"{'Run':>4s} | {'Total':>8s} | {'ViT':>8s} | {'LLM Pre':>8s} | "
          f"{'Denoise':>8s} | {'Den/step':>8s} | {'Other':>8s}")
    print("-" * 76)

    # Timed runs
    all_timings = []
    for i in range(n_runs):
        policy.reset()
        policy.model._timing = {}
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        batch = preprocess(frame)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.inference_mode():
            policy.select_action(batch)

        torch.cuda.synchronize()
        t_total = (time.perf_counter() - t_start) * 1000

        t = policy.model._timing
        vit_ms = sum(t.get("vit", []))
        llm_prefill_ms = t.get("llm_prefill", 0)
        denoise_steps = t.get("denoise_steps", [])
        denoise_total = sum(denoise_steps)
        denoise_avg = denoise_total / len(denoise_steps) if denoise_steps else 0
        other_ms = t_total - vit_ms - llm_prefill_ms - denoise_total

        all_timings.append({
            "total": t_total,
            "vit": vit_ms,
            "llm_prefill": llm_prefill_ms,
            "denoise_total": denoise_total,
            "denoise_avg_step": denoise_avg,
            "other": other_ms,
        })

        print(f"{i+1:4d} | {t_total:7.2f}ms | {vit_ms:7.2f}ms | {llm_prefill_ms:7.2f}ms | "
              f"{denoise_total:7.2f}ms | {denoise_avg:7.2f}ms | {other_ms:7.2f}ms")

    # Summary
    n = len(all_timings)
    mean_total = sum(t["total"] for t in all_timings) / n

    print(f"\n{'=' * 76}")
    print(f"  Summary over {n_runs} runs (compile=OFF)")
    print(f"{'=' * 76}")

    labels = {
        "total":           "Total",
        "vit":             "ViT (SigLIP)",
        "llm_prefill":     "LLM Prefill (KV-cache)",
        "denoise_total":   "Denoise loop (all steps)",
        "denoise_avg_step":"  per step",
        "other":           "Other overhead",
    }

    for key, label in labels.items():
        vals = [t[key] for t in all_timings]
        mean_v = sum(vals) / n
        min_v = min(vals)
        max_v = max(vals)
        pct = (mean_v / mean_total) * 100
        print(f"  {label:28s}  mean {mean_v:7.2f} ms  "
              f"min {min_v:7.2f}  max {max_v:7.2f}  ({pct:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    args = parser.parse_args()

    run_benchmark(args.n, args.warmup)
