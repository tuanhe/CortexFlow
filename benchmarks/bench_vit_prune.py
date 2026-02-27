"""Benchmark: Vision token pruning speedup.

Patches embed_prefix to prune N% of vision tokens after ViT,
then measures end-to-end inference time.

Usage:
  python benchmarks/bench_vit_prune.py -n 10 --warmup 3
"""

import argparse
import math
import os
import time
import types

import torch

from cortexflow import AutoPolicy
from cortexflow.policies.factory import make_pre_post_processors


def patch_embed_prefix(model, keep_ratio):
    """Patch embed_prefix to prune vision tokens by keep_ratio."""

    _orig_embed_image = model.paligemma_with_expert.embed_image

    def pruned_embed_prefix(self, images, img_masks, tokens, masks):
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = _orig_embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]

            # Prune: keep every 1/keep_ratio tokens (stride pruning)
            num_keep = int(num_img_embs * keep_ratio)
            indices = torch.linspace(0, num_img_embs - 1, num_keep, dtype=torch.long, device=img_emb.device)
            img_emb = img_emb[:, indices, :]
            num_img_embs = num_keep

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    model.embed_prefix = types.MethodType(pruned_embed_prefix, model)


def run_benchmark(n_runs, warmup, keep_ratio):
    model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"
    device = torch.device("cuda")

    policy = AutoPolicy.from_pretrained(model_id).to(device).eval()

    preprocess, _ = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    frame = torch.load(
        os.path.join(os.path.dirname(__file__), "..", "examples", "sample_frame.pt"),
        weights_only=False,
    )

    if keep_ratio < 1.0:
        patch_embed_prefix(policy.model, keep_ratio)

    # Warmup
    for _ in range(warmup):
        policy.reset()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        with torch.inference_mode():
            policy.select_action(preprocess(frame))
    print(f"[Warmup] {warmup} runs done (keep_ratio={keep_ratio})\n")

    # Timed
    times = []
    for i in range(n_runs):
        policy.reset()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        b = preprocess(frame)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            policy.select_action(b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    mean_t = sum(times) / len(times)
    return mean_t, min(times), max(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    ratios = [1.0, 0.75, 0.50, 0.25]
    results = {}

    for ratio in ratios:
        mean_t, min_t, max_t = run_benchmark(args.n, args.warmup, ratio)
        results[ratio] = mean_t
        pct = f"({(1 - mean_t / results[1.0]) * 100:+.1f}%)" if ratio < 1.0 else ""
        print(f"  keep={ratio:.0%}: mean {mean_t:7.2f} ms  min {min_t:7.2f}  max {max_t:7.2f}  {pct}\n")

    print("=" * 60)
    print(f"{'Keep Ratio':>12s} | {'Tokens':>8s} | {'Time':>8s} | {'Speedup':>8s}")
    print("-" * 60)
    base = results[1.0]
    for ratio in ratios:
        n_vis = int(256 * ratio)
        n_total = n_vis + 200
        t = results[ratio]
        speedup = f"{(1 - t / base) * 100:+.1f}%" if ratio < 1.0 else "baseline"
        print(f"  {ratio:>10.0%} | {n_total:>6d}  | {t:>7.2f}ms | {speedup:>8s}")
