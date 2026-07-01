"""
round3train/train.py  –  Round 3 누적 학습 (Round 2 가중치에서 시작).

Round 3 변경:
- 대보표(grand staff) 데이터 지원
- staff-bass 토큰을 vocab에 포함한 채 학습
- Round 2 seq2seq_best.pt에서 시작 (누적)

사용법:
  # Phase 1: SegNet (선택 — Round 2 SegNet 그대로 써도 됨)
  python round3train/train.py --phase 1 \\
      --data_dir "데이터 학습/Round3" \\
      --out_dir round3train/models \\
      --segnet_ckpt round2train/models/segnet_best.pt

  # Phase 2: Encoder+Decoder (Round 2 → Round 3 누적)
  python round3train/train.py --phase 2 \\
      --data_dir "데이터 학습/Round3" \\
      --out_dir round3train/models \\
      --resume round2train/models/seq2seq_best.pt

  # Phase 3: End-to-end
  python round3train/train.py --phase 3 \\
      --data_dir "데이터 학습/Round3" \\
      --out_dir round3train/models \\
      --resume round3train/models/seq2seq_best.pt

data_dir: Round1 + Round2 + Round3 데이터를 합쳐서 누적 학습 권장.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from model import (FocalLoss, OmrSeq2Seq, SegNet, NUM_CLASSES,
                   PAD_ID, EOS_ID, SOS_ID, MAX_SEQ)
from dataset import (OMRDataset, SegnetDataset, omr_collate,
                     load_tokenizer, split_dataset)

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False


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


def token_error_rate(pred, gt):
    if not gt: return 0.0 if not pred else 1.0
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


def fix_chord_tokens(token_ids, id2tok):
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


def fix_span_tokens(token_ids, id2tok):
    """짝 없는 span start/end 토큰 제거 (stack 기반)."""
    remove = set()
    stacks = {s: [] for s in _SPAN_PAIRS}
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


def _split_measures(token_ids, barline_ids):
    """barline ID로 마디 분리 (barline 포함)."""
    measures, cur = [], []
    for tid in token_ids:
        cur.append(tid)
        if tid in barline_ids:
            measures.append(cur); cur = []
    if cur:
        measures.append(cur)
    return measures


def measure_segmented_ter(pred, gt, barline_ids):
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


@torch.no_grad()
def greedy_decode(seq2seq, canvas, device, max_len=MAX_SEQ):
    seq2seq.eval()
    memory = seq2seq.encode(canvas)
    past   = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    result = []
    for _ in range(max_len):
        logits = seq2seq.decode_step(None, memory, past)
        nxt    = int(logits.argmax(-1).item())
        if nxt == EOS_ID: break
        if nxt != PAD_ID: result.append(nxt)
        past = torch.cat([past, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return result


def save_ckpt(model, path, extra=None):
    state = {'model': model.state_dict()}
    if extra: state.update(extra)
    torch.save(state, path)


def load_ckpt_vocab_expand(model, path):
    ckpt        = torch.load(path, map_location='cpu', weights_only=False)
    state_dict  = ckpt['model']
    model_state = model.state_dict()
    VOCAB_KEYS  = {'decoder.token_emb.weight', 'decoder.head.weight'}
    new_state   = {}
    for key, old_t in state_dict.items():
        if key in VOCAB_KEYS:
            new_t = model_state[key]
            old_v = old_t.shape[0]
            new_v = new_t.shape[0]
            if old_v == new_v:
                new_state[key] = old_t
            elif old_v < new_v:
                expanded       = new_t.clone()
                expanded[:old_v] = old_t
                nn.init.xavier_uniform_(expanded[old_v:])
                new_state[key] = expanded
                print(f"  Vocab expand: {key}  {old_v}→{new_v}")
            else:
                raise ValueError(f"Vocab shrink: {key}")
        else:
            new_state[key] = old_t
    model.load_state_dict(new_state, strict=True)
    print(f"  Loaded (vocab expand): {path}")
    return ckpt


class CsvLogger:
    def __init__(self, path):
        self.path = path; self._rows = []
    def log(self, row):
        self._rows.append(row)
        mode = 'w' if len(self._rows) == 1 else 'a'
        with open(self.path, mode, newline='') as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if mode == 'w': w.writeheader()
            w.writerow(row)


def train_segnet(args, device):
    print("\n" + "=" * 60)
    print("  Phase 1 — SegNet (Round 3)")
    print("=" * 60)
    full_ds = SegnetDataset(args.data_dir, patches_per_image=8, augment=True)
    train_ds, val_ds = split_dataset(full_ds, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
    model    = SegNet(num_classes=NUM_CLASSES).to(device)
    if args.segnet_ckpt and os.path.isfile(args.segnet_ckpt):
        ckpt = torch.load(args.segnet_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"  Loaded SegNet: {args.segnet_ckpt}")
    freq     = torch.tensor([0.80, 0.02, 0.04, 0.01, 0.05, 0.08], dtype=torch.float32)
    w        = (1.0 / (freq + 1e-6)).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=w / w.sum() * NUM_CLASSES)
    optimizer  = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = OneCycleLR(optimizer, max_lr=args.lr,
                             total_steps=args.epochs * len(train_loader), pct_start=0.05)
    scaler  = GradScaler(enabled=device.type == 'cuda')
    logger  = CsvLogger(os.path.join(args.out_dir, 'segnet_log.csv'))
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time(); train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(enabled=device.type == 'cuda'):
                loss = criterion(model(imgs), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = val_acc = n_pix = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(enabled=device.type == 'cuda'):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                val_loss += loss.item()
                val_acc  += (logits.argmax(1) == labels).sum().item()
                n_pix    += labels.numel()
        val_loss /= len(val_loader)
        val_acc   = val_acc / n_pix * 100.0
        print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  acc={val_acc:.1f}%  ({time.time()-t0:.0f}s)")
        logger.log({'epoch': epoch, 'train_loss': train_loss,
                    'val_loss': val_loss, 'val_acc': val_acc})
        save_ckpt(model, os.path.join(args.out_dir, 'segnet_last.pt'), {'epoch': epoch})
        if val_loss < best_loss:
            best_loss = val_loss
            save_ckpt(model, os.path.join(args.out_dir, 'segnet_best.pt'), {'epoch': epoch})
            print(f"    * Best SegNet (val_loss={val_loss:.4f})")
    print(f"\nPhase 1 done. best_loss={best_loss:.4f}")


def train_seq2seq(args, device, phase: int = 2):
    label = 'Phase 2 — Enc+Dec' if phase == 2 else 'Phase 3 — End-to-End'
    print("\n" + "=" * 60)
    print(f"  {label} (Round 3 Grand Staff)")
    print("=" * 60)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    print(f"  Vocab: {len(tok2id)} tokens  "
          f"(staff-bass={'staff-bass' in tok2id})")

    full_ds = OMRDataset(args.data_dir, tokenizer=tok2id, augment=True)
    train_ds, val_ds = split_dataset(full_ds, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.workers, pin_memory=True, collate_fn=omr_collate)
    val_loader   = DataLoader(val_ds,   batch_size=1,           shuffle=False,
                               num_workers=args.workers, pin_memory=True, collate_fn=omr_collate)

    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    if args.resume and os.path.isfile(args.resume):
        load_ckpt_vocab_expand(seq2seq, args.resume)

    if phase == 2:
        for p in seq2seq.encoder.parameters():
            p.requires_grad = False
        print(f"  Encoder frozen ({max(1, args.epochs//5)} warm-up epochs)")

    criterion   = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    trainable   = [p for p in seq2seq.parameters() if p.requires_grad]
    optimizer   = AdamW(trainable, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.98))
    scheduler   = OneCycleLR(optimizer, max_lr=args.lr,
                              total_steps=args.epochs * len(train_loader), pct_start=0.05)
    scaler      = GradScaler(enabled=device.type == 'cuda')
    logger      = CsvLogger(os.path.join(args.out_dir, f'seq2seq_phase{phase}_log.csv'))
    best_ter    = float('inf')
    unfreeze_done = False

    # Scheduled Sampling: epoch별 teacher forcing 비율 계산
    # tf_ratio: 1.0(완전 TF) → min_tf_ratio(최소 TF), ss_epochs에 걸쳐 선형 감소
    tf_ratio_start = getattr(args, 'tf_ratio',     1.0)
    tf_ratio_min   = getattr(args, 'min_tf_ratio', 0.1)
    ss_epochs      = getattr(args, 'ss_epochs',    0)
    unk_id         = tok2id.get('<UNK>', 3)

    for epoch in range(1, args.epochs + 1):
        warmup = max(1, args.epochs // 5)
        if phase == 2 and not unfreeze_done and epoch > warmup:
            print(f"  Epoch {epoch}: unfreezing encoder")
            for p in seq2seq.encoder.parameters():
                p.requires_grad = True
            optimizer = AdamW(seq2seq.parameters(), lr=args.lr / 3.0,
                               weight_decay=1e-4, betas=(0.9, 0.98))
            scheduler = OneCycleLR(optimizer, max_lr=args.lr / 3.0,
                                    total_steps=(args.epochs - epoch + 1) * len(train_loader),
                                    pct_start=0.0)
            unfreeze_done = True

        # 현재 epoch의 teacher forcing 비율 (ss_epochs=0이면 항상 1.0)
        if ss_epochs > 0:
            decay   = (tf_ratio_start - tf_ratio_min) / ss_epochs
            cur_tf  = max(tf_ratio_min, tf_ratio_start - decay * (epoch - 1))
        else:
            cur_tf  = tf_ratio_start

        seq2seq.train()
        t0 = time.time(); train_loss = 0.0
        for canvases, tgt_in, tgt_out, tgt_mask in train_loader:
            canvases = canvases.to(device)
            tgt_in   = tgt_in.to(device)
            tgt_out  = tgt_out.to(device)
            tgt_mask = tgt_mask.to(device)

            # Scheduled Sampling: SOS 제외한 입력 토큰을 cur_tf 비율로 유지,
            # 나머지는 <UNK>로 교체 → 모델이 잘못된 이전 토큰에서도 복구하도록 학습
            if cur_tf < 1.0:
                import random as _random
                drop_mask = torch.zeros_like(tgt_in, dtype=torch.bool)
                for b in range(tgt_in.size(0)):
                    for t in range(1, tgt_in.size(1)):           # SOS(pos 0)는 보존
                        if tgt_mask[b, t]: break                 # PAD 이후 무시
                        if _random.random() > cur_tf:
                            drop_mask[b, t] = True
                tgt_in = tgt_in.masked_fill(drop_mask, unk_id)

            with autocast(enabled=device.type == 'cuda'):
                logits   = seq2seq(canvases, tgt_in, tgt_mask)
                B, T, V  = logits.shape
                loss     = criterion(logits.reshape(B*T, V), tgt_out.reshape(B*T))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(seq2seq.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        seq2seq.eval()
        ter_sum = n_val = 0
        for canvases, tgt_in, tgt_out, _ in val_loader:
            if n_val >= 50: break
            canvas = canvases[0:1].to(device)
            gt     = [t for t in tgt_out[0].tolist() if t not in (PAD_ID, EOS_ID)]
            pred   = greedy_decode(seq2seq, canvas, device)
            pred   = fix_span_tokens(fix_chord_tokens(pred, id2tok), id2tok)
            ter_sum += measure_segmented_ter(pred, gt, barline_ids)
            n_val   += 1

        val_ter = ter_sum / max(n_val, 1)
        val_acc = max(0.0, 1.0 - val_ter) * 100.0
        print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  "
              f"TER={val_ter*100:.1f}%  Acc={val_acc:.1f}%  tf={cur_tf:.2f}  ({time.time()-t0:.0f}s)")
        logger.log({'epoch': epoch, 'phase': phase, 'train_loss': train_loss,
                    'val_ter': val_ter, 'val_acc': val_acc,
                    'lr': scheduler.get_last_lr()[0]})
        save_ckpt(seq2seq, os.path.join(args.out_dir, 'seq2seq_last.pt'), {'epoch': epoch})
        if val_ter < best_ter:
            best_ter = val_ter
            save_ckpt(seq2seq, os.path.join(args.out_dir, 'seq2seq_best.pt'), {'epoch': epoch})
            print(f"    * Best Seq2Seq (TER={val_ter*100:.1f}%  Acc={val_acc:.1f}%)")

    print(f"\nPhase {phase} done. Best TER={best_ter*100:.1f}%")


def parse_args():
    p = argparse.ArgumentParser(description='Round 3 OMR 누적 학습 (Grand Staff)')
    p.add_argument('--phase',       type=int, required=True, choices=[1, 2, 3])
    p.add_argument('--data_dir',    required=True)
    p.add_argument('--tokenizer',
                   default=str(_HERE / 'tokenizer.json'))
    p.add_argument('--out_dir',     default=str(_HERE / 'models'))
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch',       type=int,   default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--val_ratio',   type=float, default=0.1)
    p.add_argument('--workers',     type=int,   default=4)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--resume',        default=None,
                   help='Round 2 seq2seq_best.pt 경로')
    p.add_argument('--segnet_ckpt',   default=None)
    p.add_argument('--device',        default='auto')
    # Scheduled Sampling
    p.add_argument('--tf_ratio',      type=float, default=1.0,
                   help='초기 teacher forcing 비율 (기본 1.0 = 완전 TF)')
    p.add_argument('--min_tf_ratio',  type=float, default=0.1,
                   help='최소 teacher forcing 비율 (기본 0.1)')
    p.add_argument('--ss_epochs',     type=int,   default=0,
                   help='TF 비율을 감소시킬 에폭 수 (기본 0 = SS 비활성)')
    return p.parse_args()


def main():
    args   = parse_args()
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output : {args.out_dir}")
    print(f"Data   : {args.data_dir}")

    if args.phase == 1:
        train_segnet(args, device)
    elif args.phase in (2, 3):
        train_seq2seq(args, device, phase=args.phase)
    print("\n학습 완료.")


if __name__ == '__main__':
    main()
