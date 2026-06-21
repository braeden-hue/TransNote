import torch
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

paths = {
    "R1_seq2seq": BASE / "models/round1/seq2seq_best.pt",
    "R2_seq2seq": BASE / "models/round2/seq2seq_best.pt",
    "R3_seq2seq": BASE / "models/round3/seq2seq_best.pt",
    "R1_segnet":  BASE / "models/round1/segnet_best.pt",
    "R2_segnet":  BASE / "models/round2/segnet_best.pt",
}

print("=== Accuracy from metadata ===")
for label, path in paths.items():
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = {k: v for k, v in ckpt.items() if k != "model"}
    if "val_ter" in meta:
        acc = (1 - meta["val_ter"]) * 100
        print(f"  {label:<15} epoch={meta.get('epoch','?'):>3}  TER={meta['val_ter']*100:.1f}%  Acc={acc:.1f}%")
    elif "val_loss" in meta:
        print(f"  {label:<15} epoch={meta.get('epoch','?'):>3}  val_loss={meta['val_loss']:.6f}")

print()
print("=== Encoder backbone weight std (deeper = later layers) ===")
print(f"  {'stage':<15}  {'R1':>10}  {'R2':>10}  {'R3':>10}  trend")

ckpts = {}
for label, path in paths.items():
    ckpts[label] = torch.load(str(path), map_location="cpu", weights_only=False)["model"]

for i in range(12):
    key = "encoder.backbone.{}.block.0.weight".format(i)
    r1 = ckpts["R1_seq2seq"][key].float().std().item() if key in ckpts["R1_seq2seq"] else 0
    r2 = ckpts["R2_seq2seq"][key].float().std().item() if key in ckpts["R2_seq2seq"] else 0
    r3 = ckpts["R3_seq2seq"][key].float().std().item() if key in ckpts["R3_seq2seq"] else 0
    trend = "UP" if r3 > r1 * 1.05 else ("DOWN" if r3 < r1 * 0.95 else "=")
    print("  backbone[{:2d}]     {:>10.6f}  {:>10.6f}  {:>10.6f}  {}".format(i, r1, r2, r3, trend))

print()
print("=== Decoder cross-attention std per layer ===")
print("  {:>8}  {:>12}  {:>12}  {:>12}".format("layer", "R1", "R2", "R3"))
for i in range(8):
    key = "decoder.transformer.layers.{}.multihead_attn.in_proj_weight".format(i)
    r1 = ckpts["R1_seq2seq"][key].float().std().item() if key in ckpts["R1_seq2seq"] else 0
    r2 = ckpts["R2_seq2seq"][key].float().std().item() if key in ckpts["R2_seq2seq"] else 0
    r3 = ckpts["R3_seq2seq"][key].float().std().item() if key in ckpts["R3_seq2seq"] else 0
    print("  layer {:>3}  {:>12.6f}  {:>12.6f}  {:>12.6f}".format(i, r1, r2, r3))

print()
print("=== Token embedding norm stats ===")
for label in ["R1_seq2seq", "R2_seq2seq", "R3_seq2seq"]:
    emb = ckpts[label]["decoder.token_emb.weight"].float()
    norms = emb.norm(dim=1)
    vocab = emb.shape[0]
    print("  {}  vocab={:4d}  norm_mean={:.4f}  norm_std={:.4f}  min_norm={:.4f}  max_norm={:.4f}".format(
        label, vocab, norms.mean().item(), norms.std().item(),
        norms.min().item(), norms.max().item()))

print()
print("=== Embedding norm for extra tokens (beyond R1 base 1011) ===")
base_vocab = 1011
for label in ["R1_seq2seq", "R2_seq2seq", "R3_seq2seq"]:
    emb = ckpts[label]["decoder.token_emb.weight"].float()
    if emb.shape[0] > base_vocab:
        extra_norms = emb[base_vocab:].norm(dim=1)
        print("  {} extra[{}+]  mean={:.4f}  std={:.4f}".format(
            label, base_vocab, extra_norms.mean().item(), extra_norms.std().item()))

print()
print("=== Decoder FFN weight std trend (layer 0 vs layer 7) ===")
for label in ["R1_seq2seq", "R2_seq2seq", "R3_seq2seq"]:
    stds = []
    for i in range(8):
        key = "decoder.transformer.layers.{}.linear1.weight".format(i)
        if key in ckpts[label]:
            stds.append(ckpts[label][key].float().std().item())
    if stds:
        print("  {}  L0={:.5f}  L7={:.5f}  ratio={:.3f}".format(
            label, stds[0], stds[-1], stds[-1]/stds[0] if stds[0] > 0 else 0))

print()
print("=== BatchNorm gamma/beta saturation (SegNet) ===")
for label in ["R1_segnet", "R2_segnet"]:
    sd = ckpts[label]
    gammas = [sd[k].float() for k in sd if "block.1.weight" in k and k != "head.weight"]
    betas  = [sd[k].float() for k in sd if "block.1.bias" in k]
    if gammas:
        all_g = torch.cat(gammas)
        all_b = torch.cat(betas)
        saturated_g = (all_g > 1.05).float().mean().item() * 100
        print("  {}  BN gamma>1.05: {:.1f}%  beta abs_mean={:.5f}".format(
            label, saturated_g, all_b.abs().mean().item()))

print()
print("=== Decoder head (output projection) weight stats ===")
for label in ["R1_seq2seq", "R2_seq2seq", "R3_seq2seq"]:
    # head is weight-tied to token_emb, so check token_emb directly
    emb = ckpts[label]["decoder.token_emb.weight"].float()
    print("  {}  head_weight std={:.5f}  mean={:.5f}".format(
        label, emb.std().item(), emb.mean().item()))
