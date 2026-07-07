"""
train.py  –  OMR model training with three-phase curriculum.

Phase 1  –  SegNet pre-training (pixel-level segmentation)
Phase 2  –  Encoder-Decoder training (SegNet frozen/bypassed)
Phase 3  –  End-to-end fine-tuning (all weights unlocked)

Each phase saves:
  <out_dir>/segnet_best.pt       – best SegNet (by val loss)
  <out_dir>/segnet_last.pt       – last SegNet checkpoint
  <out_dir>/seq2seq_best.pt      – best Seq2Seq (by val TER)
  <out_dir>/seq2seq_last.pt      – last Seq2Seq checkpoint
  <out_dir>/training_log.csv     – per-epoch metrics

Usage examples:
  # Phase 1: train SegNet
  python omr/training/train.py --phase 1 \\
      --data_dir data/train --val_ratio 0.1 \\
      --tokenizer data/tokenizer.json \\
      --out_dir models/ --epochs 50 --batch 16

  # Phase 2: train Encoder+Decoder
  python omr/training/train.py --phase 2 \\
      --data_dir data/train --val_ratio 0.1 \\
      --tokenizer data/tokenizer.json \\
      --out_dir models/ --epochs 100 --batch 8 \\
      --resume models/seq2seq_last.pt

  # Phase 3: end-to-end fine-tune
  python omr/training/train.py --phase 3 \\
      --data_dir data/train --val_ratio 0.1 \\
      --tokenizer data/tokenizer.json \\
      --out_dir models/ --epochs 50 --batch 4 \\
      --resume models/seq2seq_best.pt \\
      --segnet_ckpt models/segnet_best.pt

Hardware target: RTX 3080 (10 GB VRAM).
  Phase 1: --batch 16  (320×320 patches, ~4 GB)
  Phase 2: --batch 8   (256×1280 canvases, ~7 GB with AMP)
  Phase 3: --batch 4   (segnet + seq2seq joint, ~9 GB with AMP)
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

# Allow running from project root or omr/training/.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from dataset import (OMRDataset, SegnetDataset, load_tokenizer, omr_collate,
                     split_dataset)
from model import (FocalLoss, OmrSeq2Seq, SegNet, build_models,
                    NUM_CLASSES, VOCAB_SIZE, PAD_ID, EOS_ID, SOS_ID, MAX_SEQ)

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False


# ─────────────────────────────────────────────────────────────────────────────
#  Utility: Levenshtein distance (for TER computation during validation)
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein(a: List[int], b: List[int]) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            curr[j] = (prev[j-1] if a[i-1] == b[j-1]
                       else 1 + min(prev[j], curr[j-1], prev[j-1]))
        prev, curr = curr, prev
    return prev[n]


def token_error_rate(pred: List[int], gt: List[int]) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


# ─────────────────────────────────────────────────────────────────────────────
#  포스트프로세싱 + 마디 단위 TER (OMR-NED 방식, 수정 1~4)
# ─────────────────────────────────────────────────────────────────────────────

_BARLINE_TOKEN_STRS = frozenset({
    'barline', 'barline-final', 'barline-start-repeat', 'barline-end-repeat',
})

_SPAN_PAIRS = {
    'slur-start':           'slur-end',
    'hairpin-cresc-start':  'hairpin-cresc-end',
    'hairpin-dim-start':    'hairpin-dim-end',
    'ottava-8va-start':     'ottava-8va-end',
    'ottava-8vb-start':     'ottava-8vb-end',
    'tuplet-3-start':       'tuplet-3-end',
}
_SPAN_ENDS = frozenset(_SPAN_PAIRS.values())


def fix_chord_tokens(token_ids: List[int], id2tok: dict) -> List[int]:
    """고아 chord- 토큰 제거: note- 또는 chord- 바로 뒤에만 허용."""
    result = []
    for tid in token_ids:
        tok = id2tok.get(tid, '')
        if tok.startswith('chord-'):
            prev = id2tok.get(result[-1], '') if result else ''
            if prev.startswith('note-') or prev.startswith('chord-'):
                result.append(tid)
        else:
            result.append(tid)
    return result


def fix_span_tokens(token_ids: List[int], id2tok: dict) -> List[int]:
    """짝 없는 span start/end 토큰 제거 (stack 기반)."""
    remove = set()
    stacks: dict = {s: [] for s in _SPAN_PAIRS}
    for idx, tid in enumerate(token_ids):
        tok = id2tok.get(tid, '')
        if tok in _SPAN_PAIRS:
            stacks[tok].append(idx)
        elif tok in _SPAN_ENDS:
            start = next(s for s, e in _SPAN_PAIRS.items() if e == tok)
            if stacks[start]:
                stacks[start].pop()
            else:
                remove.add(idx)
    for indices in stacks.values():
        remove.update(indices)
    return [tid for i, tid in enumerate(token_ids) if i not in remove]


def _split_measures(token_ids: List[int], barline_ids: set) -> List[List[int]]:
    """barline ID로 마디 분리 (barline 포함)."""
    measures, cur = [], []
    for tid in token_ids:
        cur.append(tid)
        if tid in barline_ids:
            measures.append(cur); cur = []
    if cur:
        measures.append(cur)
    return measures


def measure_segmented_ter(pred: List[int], gt: List[int],
                           barline_ids: set) -> float:
    """마디 단위 TER (OMR-NED 방식). 마디 간 오류 전파 차단."""
    if not gt:
        return 0.0 if not pred else 1.0
    pred_m = _split_measures(pred, barline_ids)
    gt_m   = _split_measures(gt,   barline_ids)
    total_err = total_len = 0
    for i in range(max(len(pred_m), len(gt_m))):
        p = pred_m[i] if i < len(pred_m) else []
        g = gt_m[i]   if i < len(gt_m)   else []
        total_err += levenshtein(p, g)
        total_len += len(g)
    return total_err / max(total_len, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Greedy decode (inference-mode, for validation TER)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(seq2seq: OmrSeq2Seq,
                  canvas:  torch.Tensor,    # [1, 1, H, W]
                  max_len: int = MAX_SEQ) -> List[int]:
    """Greedy decoding for a single canvas tile (validation / inference)."""
    seq2seq.eval()
    device  = canvas.device
    memory  = seq2seq.encode(canvas)            # [1, S, D]
    past    = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    result  = []

    for _ in range(max_len):
        logits = seq2seq.decode_step(None, memory, past)  # [1, vocab]
        nxt    = int(logits.argmax(-1).item())
        if nxt == EOS_ID:
            break
        if nxt != PAD_ID:
            result.append(nxt)
        past = torch.cat([past,
                          torch.tensor([[nxt]], dtype=torch.long, device=device)],
                         dim=1)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Class-frequency weights for SegNet Focal Loss calibration
# ─────────────────────────────────────────────────────────────────────────────

def make_seg_class_weights(device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency class weights for sheet-music pixel classes.
    Background (class 0) is dominant; upweight rare classes (noteheads, stems).
    Adjust empirically after inspecting your dataset.
    """
    # Estimated pixel-frequency ratios (background ≈ 80 %, staff ≈ 5 %, rest < 2 %).
    freq = torch.tensor([0.80, 0.02, 0.04, 0.01, 0.05, 0.08], dtype=torch.float32)
    w    = 1.0 / (freq + 1e-6)
    w    = w / w.sum() * NUM_CLASSES          # normalise so sum = num_classes
    return w.to(device)


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: str, extra: Optional[dict] = None):
    state = {'model': model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(model: nn.Module, path: str,
                    strict: bool = True) -> dict:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=strict)
    print(f"  Loaded checkpoint: {path}")
    return ckpt


def load_checkpoint_with_vocab_expansion(model: nn.Module, path: str) -> dict:
    """
    Load a checkpoint whose vocabulary may be smaller than the current model.

    The two vocab-size-dependent layers in OmrSeq2Seq are:
      decoder.token_emb.weight  [old_vocab, embed_dim]
      decoder.head.weight        [old_vocab, embed_dim]  (weight-tied to token_emb)

    New token rows (indices old_vocab … new_vocab-1) are Xavier-initialised.
    All other layers are loaded with strict=True so shape mismatches elsewhere
    raise an error immediately.
    """
    ckpt       = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']

    model_state = model.state_dict()
    new_state   = {}

    VOCAB_KEYS = {'decoder.token_emb.weight', 'decoder.head.weight'}

    for key, old_tensor in state_dict.items():
        if key in VOCAB_KEYS:
            new_tensor = model_state[key]                  # [new_vocab, embed_dim]
            old_v, dim = old_tensor.shape
            new_v      = new_tensor.shape[0]

            if old_v == new_v:
                new_state[key] = old_tensor
            elif old_v < new_v:
                # Copy existing rows; Xavier-init the new rows
                expanded = new_tensor.clone()
                expanded[:old_v] = old_tensor
                nn.init.xavier_uniform_(expanded[old_v:])   # [n_new, dim] — 2D, in-place
                new_state[key] = expanded
                print(f"  Vocab expanded: {key}  {old_v} → {new_v}  "
                      f"(+{new_v - old_v} rows Xavier-init)")
            else:
                raise ValueError(
                    f"Checkpoint vocab ({old_v}) is larger than model vocab ({new_v}) "
                    f"for key '{key}'. Cannot shrink vocabulary.")
        else:
            new_state[key] = old_tensor

    model.load_state_dict(new_state, strict=True)
    print(f"  Loaded checkpoint with vocab expansion: {path}")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
#  CSV logger
# ─────────────────────────────────────────────────────────────────────────────

class CsvLogger:
    def __init__(self, path: str):
        self.path = path
        self._rows: List[dict] = []

    def log(self, row: dict):
        self._rows.append(row)
        mode = 'w' if len(self._rows) == 1 else 'a'
        with open(self.path, mode, newline='') as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if mode == 'w':
                w.writeheader()
            w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1: SegNet training
# ─────────────────────────────────────────────────────────────────────────────

def train_segnet(args: argparse.Namespace, device: torch.device):
    print("\n" + "=" * 60)
    print("  Phase 1 — SegNet pre-training")
    print("=" * 60)

    # ── Datasets ─────────────────────────────────────────────────────────────
    full_ds = SegnetDataset(args.data_dir, patches_per_image=8, augment=True)
    train_ds, val_ds = split_dataset(full_ds, val_ratio=args.val_ratio,
                                      seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                               shuffle=True,  num_workers=args.workers,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                               shuffle=False, num_workers=args.workers,
                               pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SegNet(num_classes=NUM_CLASSES).to(device)
    if args.segnet_ckpt and os.path.isfile(args.segnet_ckpt):
        load_checkpoint(model, args.segnet_ckpt)

    cls_w = make_seg_class_weights(device)
    criterion = FocalLoss(gamma=2.0, alpha=cls_w)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler   = OneCycleLR(optimizer, max_lr=args.lr,
                              total_steps=total_steps, pct_start=0.05)
    scaler = GradScaler(enabled=device.type == 'cuda')

    logger = CsvLogger(os.path.join(args.out_dir, 'segnet_log.csv'))
    tb     = SummaryWriter(os.path.join(args.out_dir, 'tb_segnet')) if _HAS_TB else None
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            with autocast(enabled=device.type == 'cuda'):
                logits = model(imgs)             # [B, 6, 320, 320]
                loss   = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        n_pix    = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                with autocast(enabled=device.type == 'cuda'):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                val_loss += loss.item()
                preds    = logits.argmax(dim=1)
                val_acc  += (preds == labels).sum().item()
                n_pix    += labels.numel()

        val_loss /= len(val_loader)
        val_acc   = val_acc / n_pix * 100.0
        elapsed   = time.time() - t0

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.1f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({elapsed:.0f}s)")

        row = {'epoch': epoch, 'train_loss': train_loss,
               'val_loss': val_loss, 'val_acc': val_acc,
               'lr': scheduler.get_last_lr()[0]}
        logger.log(row)

        if tb:
            tb.add_scalar('segnet/train_loss', train_loss, epoch)
            tb.add_scalar('segnet/val_loss',   val_loss,   epoch)
            tb.add_scalar('segnet/val_acc',    val_acc,    epoch)

        # ── Checkpoints ───────────────────────────────────────────────────────
        save_checkpoint(model, os.path.join(args.out_dir, 'segnet_last.pt'),
                        extra={'epoch': epoch})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, os.path.join(args.out_dir, 'segnet_best.pt'),
                            extra={'epoch': epoch, 'val_loss': val_loss})
            print(f"    * New best SegNet saved  (val_loss={val_loss:.4f})")

    if tb: tb.close()
    print(f"\nPhase 1 done.  Best val_loss = {best_val_loss:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2 / 3: Encoder-Decoder training
# ─────────────────────────────────────────────────────────────────────────────

def train_seq2seq(args: argparse.Namespace,
                  device: torch.device,
                  phase: int = 2):
    label = 'Phase 2 — Encoder+Decoder' if phase == 2 else 'Phase 3 — End-to-End'
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    vocab_size = len(tok2id)
    print(f"  Vocabulary: {vocab_size} tokens")

    # ── Datasets ─────────────────────────────────────────────────────────────
    full_ds  = OMRDataset(args.data_dir, tokenizer=tok2id,
                           augment=True, tiles_per_staff=1)
    train_ds, val_ds = split_dataset(full_ds, val_ratio=args.val_ratio,
                                      seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                               shuffle=True,  num_workers=args.workers,
                               pin_memory=True, collate_fn=omr_collate)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                               shuffle=False, num_workers=args.workers,
                               pin_memory=True, collate_fn=omr_collate)

    # ── Models ────────────────────────────────────────────────────────────────
    seq2seq = OmrSeq2Seq(vocab_size=vocab_size).to(device)

    if args.resume and os.path.isfile(args.resume):
        load_checkpoint_with_vocab_expansion(seq2seq, args.resume)

    # Phase 2: freeze encoder for the first warm-up portion, then unfreeze.
    # Phase 3: everything is trainable from the start.
    if phase == 2:
        # Start with encoder frozen; unfreeze after warm-up epochs.
        for p in seq2seq.encoder.parameters():
            p.requires_grad = False
        print("  Encoder frozen for warm-up (will unfreeze after "
              f"{max(1, args.epochs // 5)} epochs)")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable = [p for p in seq2seq.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.98))
    total_steps = args.epochs * len(train_loader)
    scheduler   = OneCycleLR(optimizer, max_lr=args.lr,
                              total_steps=total_steps, pct_start=0.05)
    scaler = GradScaler(enabled=device.type == 'cuda')

    log_path = os.path.join(args.out_dir, f'seq2seq_phase{phase}_log.csv')
    logger = CsvLogger(log_path)
    tb     = (SummaryWriter(os.path.join(args.out_dir, f'tb_phase{phase}'))
              if _HAS_TB else None)
    best_val_ter = float('inf')
    unfreeze_done = False

    for epoch in range(1, args.epochs + 1):
        # ── Unfreeze encoder after warm-up (Phase 2 only) ─────────────────────
        warmup_epochs = max(1, args.epochs // 5)
        if phase == 2 and not unfreeze_done and epoch > warmup_epochs:
            print(f"  Epoch {epoch}: unfreezing encoder")
            for p in seq2seq.encoder.parameters():
                p.requires_grad = True
            # Re-create optimizer and scheduler with all params.
            optimizer = AdamW(seq2seq.parameters(), lr=args.lr / 1.5,
                               weight_decay=1e-4, betas=(0.9, 0.98))
            remaining = (args.epochs - epoch + 1) * len(train_loader)
            scheduler = OneCycleLR(optimizer, max_lr=args.lr / 1.5,
                                    total_steps=max(1, remaining),
                                    pct_start=0.1)
            unfreeze_done = True

        # ── Train ─────────────────────────────────────────────────────────────
        seq2seq.train()
        t0 = time.time()
        train_loss = 0.0

        for canvases, tgt_in, tgt_out, tgt_mask in train_loader:
            canvases = canvases.to(device)
            tgt_in   = tgt_in.to(device)
            tgt_out  = tgt_out.to(device)
            tgt_mask = tgt_mask.to(device)

            with autocast(enabled=device.type == 'cuda'):
                logits = seq2seq(canvases, tgt_in, tgt_mask)   # [B, T, V]
                # Reshape for cross-entropy: [B*T, V] and [B*T].
                B, T, V = logits.shape
                loss    = criterion(logits.reshape(B * T, V),
                                    tgt_out.reshape(B * T))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(seq2seq.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation: TER on a subset of val samples ────────────────────────
        seq2seq.eval()
        val_ter_sum = 0.0
        n_val       = 0

        # Evaluate on up to 200 samples for reliable best-checkpoint selection.
        MAX_VAL = 200
        for canvases, tgt_in, tgt_out, _ in val_loader:
            if n_val >= MAX_VAL:
                break
            canvas = canvases[0:1].to(device)   # process one sample at a time

            # Ground-truth: strip PAD and EOS from tgt_out.
            gt_ids = tgt_out[0].tolist()
            gt_ids = [t for t in gt_ids if t not in (PAD_ID, EOS_ID)]

            pred_ids = greedy_decode(seq2seq, canvas, max_len=MAX_SEQ)
            pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
            val_ter_sum += measure_segmented_ter(pred_ids, gt_ids, barline_ids)
            n_val       += 1

        val_ter = val_ter_sum / max(n_val, 1)
        val_acc = max(0.0, 1.0 - val_ter) * 100.0
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_TER={val_ter*100:.1f}%  "
              f"val_Acc={val_acc:.1f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({elapsed:.0f}s)")

        row = {'epoch': epoch, 'phase': phase,
               'train_loss': train_loss, 'val_ter': val_ter,
               'val_acc': val_acc, 'lr': scheduler.get_last_lr()[0]}
        logger.log(row)

        if tb:
            tb.add_scalar(f'p{phase}/train_loss', train_loss, epoch)
            tb.add_scalar(f'p{phase}/val_ter',    val_ter,    epoch)
            tb.add_scalar(f'p{phase}/val_acc',    val_acc,    epoch)

        # ── Checkpoints ───────────────────────────────────────────────────────
        save_checkpoint(seq2seq, os.path.join(args.out_dir, 'seq2seq_last.pt'),
                        extra={'epoch': epoch, 'phase': phase})
        if val_ter < best_val_ter:
            best_val_ter = val_ter
            save_checkpoint(seq2seq, os.path.join(args.out_dir, 'seq2seq_best.pt'),
                            extra={'epoch': epoch, 'val_ter': val_ter})
            print(f"    * New best Seq2Seq saved  "
                  f"(TER={val_ter*100:.1f}%  Acc={val_acc:.1f}%)")

    if tb: tb.close()
    print(f"\nPhase {phase} done.  Best val_TER = {best_val_ter*100:.1f}%  "
          f"(Acc = {(1-best_val_ter)*100:.1f}%)")
    return seq2seq


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='OMR model trainer (3-phase curriculum)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    p.add_argument('--phase',      type=int, required=True, choices=[1, 2, 3],
                   help='Training phase: 1=SegNet, 2=Enc+Dec, 3=End-to-end')
    p.add_argument('--data_dir',   required=True,
                   help='Training data directory (contains .png + .json pairs)')
    p.add_argument('--tokenizer',  default='data/tokenizer.json',
                   help='Path to tokenizer.json')
    p.add_argument('--out_dir',    default='models/',
                   help='Output directory for checkpoints and logs')

    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch',      type=int,   default=8,
                   help='Batch size (recommended: 16 for phase1, 8 for phase2, 4 for phase3)')
    p.add_argument('--lr',         type=float, default=1e-4,
                   help='Peak learning rate (OneCycleLR)')
    p.add_argument('--val_ratio',  type=float, default=0.1,
                   help='Fraction of data reserved for validation')
    p.add_argument('--workers',    type=int,   default=4,
                   help='DataLoader worker processes')
    p.add_argument('--seed',       type=int,   default=42)

    p.add_argument('--resume',     default=None,
                   help='Resume Seq2Seq from this checkpoint (.pt)')
    p.add_argument('--segnet_ckpt', default=None,
                   help='Load SegNet weights from this checkpoint')
    p.add_argument('--device',     default='auto',
                   help='Device: auto | cuda | cpu')

    return p.parse_args()


def main():
    args = parse_args()

    # ── Device selection ─────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Device  : {device}")
    if device.type == 'cuda':
        print(f"GPU     : {torch.cuda.get_device_name()}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # ── Output directory ─────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output  : {args.out_dir}")
    print(f"Data    : {args.data_dir}")

    # ── Run requested phase ──────────────────────────────────────────────────
    if args.phase == 1:
        train_segnet(args, device)

    elif args.phase == 2:
        train_seq2seq(args, device, phase=2)

    elif args.phase == 3:
        train_seq2seq(args, device, phase=3)

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
