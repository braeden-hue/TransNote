"""
analyze_checkpoints.py  -- OMR .pt 체크포인트 분석 스크립트

각 라운드별 가중치 파일에서:
  1. 학습 메타데이터 (epoch, val_loss, val_ter, accuracy 등)
  2. 레이어별 가중치 통계 (mean, std, dead neuron 비율)
  3. 인코더/디코더 레이어 수렴 여부
  4. 이상 징후 (gradient vanish, exploding, 비정상 분포)
를 분석합니다.
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Repo path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "omr" / "training"))

MODEL_PATHS = {
    "round1/segnet":  REPO / "models/round1/segnet_best.pt",
    "round1/seq2seq": REPO / "models/round1/seq2seq_best.pt",
    "round2/segnet":  REPO / "models/round2/segnet_best.pt",
    "round2/seq2seq": REPO / "models/round2/seq2seq_best.pt",
    "round3/seq2seq": REPO / "models/round3/seq2seq_best.pt",
}

SEP = "=" * 70


def weight_stats(tensor: torch.Tensor, name: str) -> dict:
    t = tensor.float()
    abs_t = t.abs()
    return {
        "name":     name,
        "shape":    list(t.shape),
        "mean":     t.mean().item(),
        "std":      t.std().item(),
        "abs_mean": abs_t.mean().item(),
        "min":      t.min().item(),
        "max":      t.max().item(),
        "dead":     (abs_t < 1e-6).float().mean().item(),  # fraction near-zero
        "large":    (abs_t > 10.0).float().mean().item(),  # fraction too large
    }


def analyze_state_dict(sd: dict, label: str):
    print(f"\n  --- Weight statistics: {label} ---")
    print(f"  {'Layer':<50} {'mean':>8} {'std':>8} {'dead%':>7} {'large%':>7} {'max':>9}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*9}")

    dead_layers = []
    large_layers = []
    collapsed_layers = []

    for key, param in sd.items():
        if not isinstance(param, torch.Tensor) or param.numel() < 4:
            continue
        if "running_mean" in key or "running_var" in key or "num_batches" in key:
            continue

        s = weight_stats(param, key)

        dead_pct  = s["dead"]  * 100
        large_pct = s["large"] * 100
        flag = ""
        if dead_pct > 30:
            flag += " [DEAD]"
            dead_layers.append(key)
        if large_pct > 1:
            flag += " [LARGE]"
            large_layers.append(key)
        if s["std"] < 1e-4:
            flag += " [COLLAPSED]"
            collapsed_layers.append(key)

        short = key[-50:] if len(key) > 50 else key
        print(f"  {short:<50} {s['mean']:>8.4f} {s['std']:>8.4f} "
              f"{dead_pct:>6.1f}% {large_pct:>6.1f}% {s['max']:>9.4f}{flag}")

    return dead_layers, large_layers, collapsed_layers


def analyze_encoder_seq(sd: dict):
    """Encoder CNN 레이어별 std 추이 (수렴 방향 확인)."""
    print("\n  --- Encoder backbone std trend ---")
    stds = []
    for key in sd:
        if "encoder.backbone" in key and ".block.0.weight" in key:
            s = sd[key].float().std().item()
            stds.append((key.replace("encoder.backbone.", ""), s))
    for name, s in stds:
        bar = "#" * int(s * 200) + "." * max(0, 20 - int(s * 200))
        print(f"  {name:<45} std={s:.5f}  [{bar}]")


def analyze_decoder_layers(sd: dict):
    """Decoder Transformer 레이어별 self-attention Q/K/V weight std."""
    print("\n  --- Decoder transformer layer std trend ---")
    for i in range(8):
        key_q = f"decoder.transformer.layers.{i}.self_attn.in_proj_weight"
        key_ff = f"decoder.transformer.layers.{i}.linear1.weight"
        if key_q in sd:
            sq = sd[key_q].float().std().item()
            sf = sd[key_ff].float().std().item() if key_ff in sd else 0.0
            bar = "#" * int(sq * 500)
            print(f"  Layer {i}  attn_std={sq:.5f}  ffn_std={sf:.5f}  [{bar}]")


def check_embedding(sd: dict):
    key = "decoder.token_emb.weight"
    if key not in sd:
        return
    emb = sd[key].float()
    norms = emb.norm(dim=1)  # [vocab]
    dead_toks = (norms < 0.01).sum().item()
    large_toks = (norms > 5.0).sum().item()
    print(f"\n  --- Token embedding ---")
    print(f"  shape      : {list(emb.shape)}")
    print(f"  norm mean  : {norms.mean().item():.4f}")
    print(f"  norm std   : {norms.std().item():.4f}")
    print(f"  dead tokens (norm<0.01): {dead_toks}")
    print(f"  large tokens (norm>5.0): {large_toks}")
    # Check if new tokens (after round1) are uninitialised or well-learned
    if emb.shape[0] > 1012:
        extra = emb[1012:]
        print(f"  extra tokens [{emb.shape[0]-1012}] norm mean: {extra.norm(dim=1).mean():.4f}")


def main():
    for label, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"\n[MISSING] {label}: {path}")
            continue

        print(f"\n{SEP}")
        print(f"  Checkpoint: {label}")
        print(f"  File      : {path}")
        print(f"  Size      : {path.stat().st_size / 1e6:.1f} MB")

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        sd   = ckpt.get("model", ckpt)

        # ── Metadata ──────────────────────────────────────────────────────────
        meta = {k: v for k, v in ckpt.items() if k != "model"}
        print(f"\n  --- Training metadata ---")
        if meta:
            for k, v in meta.items():
                if k == "val_ter":
                    print(f"  {k:<20}: {v:.4f}  (Acc = {(1-v)*100:.1f}%)")
                elif k == "val_loss":
                    print(f"  {k:<20}: {v:.4f}")
                else:
                    print(f"  {k:<20}: {v}")
        else:
            print("  (no metadata saved)")

        # ── Weight statistics ─────────────────────────────────────────────────
        dead, large, collapsed = analyze_state_dict(sd, label)

        # ── Model-specific analysis ───────────────────────────────────────────
        is_seq2seq = any("encoder" in k for k in sd)
        if is_seq2seq:
            analyze_encoder_seq(sd)
            analyze_decoder_layers(sd)
            check_embedding(sd)

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n  --- Summary for {label} ---")
        print(f"  Dead layers (>30% zeros) : {len(dead)}")
        if dead:
            for d in dead[:5]: print(f"    - {d}")
        print(f"  Exploding layers (>10)   : {len(large)}")
        if large:
            for l in large[:5]: print(f"    - {l}")
        print(f"  Collapsed layers (std<1e-4): {len(collapsed)}")
        if collapsed:
            for c in collapsed[:5]: print(f"    - {c}")

    print(f"\n{SEP}")
    print("  Analysis complete.")
    print(SEP)


if __name__ == "__main__":
    main()
