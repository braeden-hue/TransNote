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
    """고아 chord- 토큰 제거: note-/dur-/chord- 바로 뒤에만 허용.
    (note-{pitch} 뒤 dur-{dur}이 오고 그 다음에 chord-가 이어지는 구조)"""
    result = []
    for tid in token_ids:
        tok = id2tok.get(tid, '')
        if tok.startswith('chord-'):
            prev = id2tok.get(result[-1], '') if result else ''
            if prev.startswith('note-') or prev.startswith('dur-') or prev.startswith('chord-'):
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
    memory   = seq2seq.encode(canvas)
    kv_cache = seq2seq.precompute_memory_kv(memory)  # cross-attention K,V 1회 계산 (Step1 가속)
    past   = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    result = []
    for _ in range(max_len):
        logits = seq2seq.decode_step_cached(kv_cache, past)
        nxt    = int(logits.argmax(-1).item())
        if nxt == EOS_ID: break
        if nxt != PAD_ID: result.append(nxt)
        past = torch.cat([past, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return result


class _Beam:
    __slots__ = ('seq', 'tokens', 'score', 'finished')

    def __init__(self, seq, tokens, score, finished):
        self.seq = seq            # [1, t] decoder input so far (incl. SOS)
        self.tokens = tokens       # generated token ids, excl. SOS/EOS/PAD
        self.score = score         # cumulative log-probability
        self.finished = finished   # hit EOS already


@torch.no_grad()
def beam_decode(seq2seq, canvas, device, beam_width=4, length_penalty=0.7, max_len=MAX_SEQ):
    """Phase 3 검증용 beam search. inference.py의 beam_decode와 동일한 알고리즘
    (KV-cache 없이 매 스텝 시퀀스 전체 재계산) — 다만 canvas가 val_loader에서 이미
    정규화된 텐서로 나오므로 여기선 재정규화하지 않고 바로 encode한다.
    cross-attention K,V는 precompute_memory_kv로 1회 계산해서 재사용한다(Step1 가속)."""
    if beam_width <= 1:
        return greedy_decode(seq2seq, canvas, device, max_len)

    seq2seq.eval()
    memory   = seq2seq.encode(canvas)
    kv_cache = seq2seq.precompute_memory_kv(memory)
    sos    = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    beams  = [_Beam(sos, [], 0.0, False)]

    for _ in range(max_len):
        if all(b.finished for b in beams):
            break

        candidates = []  # (score, parent_idx, token_id_or_None, advances)
        for i, b in enumerate(beams):
            if b.finished:
                candidates.append((b.score, i, None, False))
                continue
            logits = seq2seq.decode_step_cached(kv_cache, b.seq)  # [1, vocab]
            logp   = torch.log_softmax(logits[0], dim=-1)
            logp[PAD_ID] = float('-inf')
            topk_logp, topk_idx = logp.topk(beam_width)
            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                candidates.append((b.score + lp, i, idx, True))

        candidates.sort(key=lambda c: c[0], reverse=True)
        candidates = candidates[:beam_width]

        new_beams = []
        for score, parent, tok, advances in candidates:
            parent_beam = beams[parent]
            if not advances:
                new_beams.append(parent_beam)
                continue
            if tok == EOS_ID:
                new_beams.append(_Beam(parent_beam.seq, parent_beam.tokens, score, True))
            else:
                new_seq = torch.cat(
                    [parent_beam.seq, torch.tensor([[tok]], dtype=torch.long, device=device)],
                    dim=1)
                new_tokens = parent_beam.tokens + ([tok] if tok != PAD_ID else [])
                new_beams.append(_Beam(new_seq, new_tokens, score, False))
        beams = new_beams

    best = max(beams, key=lambda b: b.score / max(len(b.tokens), 1) ** length_penalty)
    return best.tokens


def save_ckpt(model, path, extra=None):
    state = {'model': model.state_dict()}
    if extra: state.update(extra)
    torch.save(state, path)


def load_ckpt_partial_vocab(model, path, old_tok2id, new_tok2id):
    """
    Vocab이 재구성된 체크포인트(예: note-{pitch}-{dur} 단일 토큰 →
    note-{pitch} + dur-{dur} 분해)를 토큰 "문자열" 기준으로 이어받는다.

    - 인코더/디코더 트랜스포머 레이어 등 vocab과 무관한 파라미터는 그대로 로드(워밍스타트).
    - token_emb / head(weight-tied)는 old/new 양쪽에 동일한 토큰 문자열이 있는 행만 복사하고,
      새로 생기거나 사라진 토큰의 행은 모델의 랜덤 초기값을 그대로 둔다.
    - raw index 대응을 가정하던 기존 vocab-expand 로직(append-only)을 완전히 대체한다.
    """
    ckpt        = torch.load(path, map_location='cpu', weights_only=False)
    state_dict  = ckpt['model']
    model_state = model.state_dict()
    VOCAB_KEYS  = {'decoder.token_emb.weight', 'decoder.head.weight'}
    old_id2tok  = {v: k for k, v in old_tok2id.items()}

    new_state = {}
    for key, old_t in state_dict.items():
        if key not in model_state:
            print(f"  Skip (no longer in model): {key}")
            continue
        if key in VOCAB_KEYS:
            new_t   = model_state[key].clone()
            matched = 0
            for old_id in range(old_t.shape[0]):
                tok = old_id2tok.get(old_id)
                new_id = new_tok2id.get(tok) if tok is not None else None
                if new_id is None:
                    continue   # 이 토큰은 새 vocab에 없음 (예: 옛 note-C4-1/4)
                new_t[new_id] = old_t[old_id]
                matched += 1
            new_state[key] = new_t
            print(f"  Vocab remap: {key}  {matched}/{new_t.shape[0]} rows carried over "
                  f"(old vocab={old_t.shape[0]}, new vocab={new_t.shape[0]})")
        elif old_t.shape == model_state[key].shape:
            new_state[key] = old_t
        else:
            print(f"  Skip (shape mismatch): {key}  {tuple(old_t.shape)} -> {tuple(model_state[key].shape)}")

    missing = model.load_state_dict(new_state, strict=False)
    print(f"  Loaded (partial vocab remap): {path}")
    if missing.missing_keys:
        print(f"  Missing (kept randomly-initialised): {missing.missing_keys}")
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
    full_ds = SegnetDataset(args.data_dir, patches_per_image=8, augment=True,
                            noise_level=args.noise_level)
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

    full_ds = OMRDataset(args.data_dir, tokenizer=tok2id, augment=not args.no_augment,
                          replay_dir=args.replay_dir, replay_count=args.replay_count,
                          noise_level=args.noise_level, noise_level_max=args.noise_level_max,
                          p_level_max=args.p_level_max, page_level_noise=args.page_level_noise,
                          in_ch=args.in_ch)
    train_ds, val_ds = split_dataset(full_ds, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.workers, pin_memory=True, collate_fn=omr_collate)
    val_loader   = DataLoader(val_ds,   batch_size=1,           shuffle=False,
                               num_workers=args.workers, pin_memory=True, collate_fn=omr_collate)

    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), in_ch=args.in_ch,
                         extra_height_stages=args.extra_height_stages,
                         pool_h=args.pool_h).to(device)
    if args.resume and os.path.isfile(args.resume):
        resume_tok2id = tok2id
        if args.resume_tokenizer:
            resume_tok2id, _ = load_tokenizer(args.resume_tokenizer)
        ckpt = load_ckpt_partial_vocab(seq2seq, args.resume, resume_tok2id, tok2id)
        # CoordConv 실험(2026-07-31): in_ch=2로 새로 만든 모델에 in_ch=1 체크포인트를
        # 이어받으면 첫 conv 레이어만 shape가 안 맞아 load_ckpt_partial_vocab이 스킵하고
        # 랜덤 초기화된 채로 남는다 -- 기존 학습된 그레이스케일 채널 가중치를 그대로
        # 복사하고 새 좌표 채널만 0으로 채워서, 재개 시작 시점에 기존 체크포인트와
        # 수학적으로 동일하게(좌표 채널이 아직 아무 영향 없음) 만든다.
        if args.in_ch == 2:
            old_w = ckpt['model'].get('encoder.backbone.0.block.0.weight')
            if old_w is not None and old_w.shape[1] == 1:
                with torch.no_grad():
                    new_w = seq2seq.encoder.backbone[0].block[0].weight
                    new_w.zero_()
                    new_w[:, :1] = old_w
                print("  CoordConv: encoder.backbone.0.block.0.weight 1ch->2ch "
                      "(기존 채널 복사, 좌표 채널 0으로 초기화)")

    freeze_epochs = args.freeze_epochs if args.freeze_epochs is not None else max(1, args.epochs // 5)
    proj_warmup_epochs = getattr(args, 'proj_warmup_epochs', 0) or 0
    proj_warmup_done = (proj_warmup_epochs == 0)
    proj_warmup_extra_bb = getattr(args, 'proj_warmup_extra_backbone_stages', 0) or 0
    warmup_bb_idxs = set()
    if phase == 2 and proj_warmup_epochs > 0:
        backbone_len = len(seq2seq.encoder.backbone)
        if proj_warmup_extra_bb > 0:
            warmup_bb_idxs = set(range(max(0, backbone_len - proj_warmup_extra_bb), backbone_len))

        def _warmup_trainable(name):
            if 'encoder.proj' in name:
                return True
            return any(f'encoder.backbone.{i}.' in name for i in warmup_bb_idxs)

        for name, p in seq2seq.named_parameters():
            p.requires_grad = _warmup_trainable(name)
        seq2seq.eval()  # 디코더/backbone 전부 동결(BatchNorm/dropout 흔들림 방지)
        for i in warmup_bb_idxs:
            seq2seq.encoder.backbone[i].train()  # 워밍업 대상 backbone 단계는 BN도 갱신되게 train 모드
        freeze_epochs = 0  # 반대 방향 옵션(디코더만 먼저 학습)과 동시 적용 방지 -- proj 워밍업 우선
        print(f"  Proj 워밍업: encoder.proj"
              + (f" + backbone{sorted(warmup_bb_idxs)}" if warmup_bb_idxs else "")
              + f" 학습 ({proj_warmup_epochs}에폭), 나머지 전체 동결")
    elif phase == 2:
        for p in seq2seq.encoder.parameters():
            p.requires_grad = False
        seq2seq.encoder.eval()  # BatchNorm도 freeze (running stats로 고정, 배치별 흔들림 방지)
        print(f"  Encoder frozen ({freeze_epochs} warm-up epochs)")

    # 희소 span 토큰(옥타브/헤어핀 시작-종료) 가중치 -- 시퀀스당 딱 2개뿐이라 얼마나 많은
    # 악보에 등장하든 토큰 단위 grdient는 여전히 희소함(2026-07-21, --ottava-prob/--hairpin-prob
    # 를 2~3.5배 올려도 recall이 거의 안 오른 것으로 확인). --span_weight(기본 1.0=미적용)로
    # 이 토큰들의 loss 가중치를 높여 신호를 강제로 키운다.
    span_weight = getattr(args, 'span_weight', 1.0)
    class_weight = None
    if span_weight != 1.0:
        SPAN_TOKENS = ['ottava-8va-start', 'ottava-8va-end', 'ottava-8vb-start', 'ottava-8vb-end',
                       'hairpin-cresc-start', 'hairpin-cresc-end',
                       'hairpin-dim-start', 'hairpin-dim-end']
        class_weight = torch.ones(len(tok2id), device=device)
        n_weighted = 0
        for t in SPAN_TOKENS:
            if t in tok2id:
                class_weight[tok2id[t]] = span_weight
                n_weighted += 1
        print(f"  Span 토큰 가중치 {span_weight}x 적용: {n_weighted}/{len(SPAN_TOKENS)}개 토큰")
    criterion   = nn.CrossEntropyLoss(weight=class_weight, ignore_index=PAD_ID, label_smoothing=0.1)
    trainable   = [p for p in seq2seq.parameters() if p.requires_grad]
    init_lr     = (args.proj_warmup_lr if (proj_warmup_epochs > 0 and args.proj_warmup_lr is not None)
                   else args.lr)
    optimizer   = AdamW(trainable, lr=init_lr, weight_decay=1e-4, betas=(0.9, 0.98))
    scheduler   = OneCycleLR(optimizer, max_lr=init_lr,
                              total_steps=(proj_warmup_epochs or args.epochs) * len(train_loader),
                              pct_start=0.05)
    scaler      = GradScaler(enabled=device.type == 'cuda')
    logger      = CsvLogger(os.path.join(args.out_dir, f'seq2seq_phase{phase}_log.csv'))
    best_ter    = float('inf')
    best_acc    = 0.0
    epochs_since_improve = 0
    target_acc  = getattr(args, 'target_acc', None)
    target_check_after_unfreeze = getattr(args, 'target_check_after_unfreeze', None)
    if target_check_after_unfreeze is not None:
        target_check_epoch = freeze_epochs + target_check_after_unfreeze
    else:
        target_check_epoch = getattr(args, 'target_check_epoch', None) or max(1, args.epochs // 2)
    patience    = getattr(args, 'patience', None)
    plateau_band = getattr(args, 'plateau_band', None)
    plateau_lo = plateau_hi = None
    if plateau_band:
        plateau_lo, plateau_hi = (float(x) for x in plateau_band.split(','))
    plateau_band_epochs = getattr(args, 'plateau_band_epochs', 5)
    plateau_min_epoch = getattr(args, 'plateau_min_epoch', None)
    plateau_streak = 0
    unfreeze_done = False

    # Scheduled Sampling: epoch별 teacher forcing 비율 계산
    # tf_ratio: 1.0(완전 TF) → min_tf_ratio(최소 TF), ss_epochs에 걸쳐 선형 감소
    tf_ratio_start = getattr(args, 'tf_ratio',     1.0)
    tf_ratio_min   = getattr(args, 'min_tf_ratio', 0.1)
    ss_epochs      = getattr(args, 'ss_epochs',    0)

    for epoch in range(1, args.epochs + 1):
        warmup = freeze_epochs
        if phase == 2 and proj_warmup_epochs > 0 and not proj_warmup_done and epoch > proj_warmup_epochs:
            print(f"  Epoch {epoch}: proj 워밍업 종료, 전체 학습으로 전환")
            for p in seq2seq.parameters():
                p.requires_grad = True
            optimizer = AdamW(seq2seq.parameters(), lr=args.lr / 3.0,
                               weight_decay=1e-4, betas=(0.9, 0.98))
            scheduler = OneCycleLR(optimizer, max_lr=args.lr / 3.0,
                                    total_steps=(args.epochs - epoch + 1) * len(train_loader),
                                    pct_start=0.0)
            proj_warmup_done = True
            unfreeze_done = True  # 아래 일반 unfreeze 분기가 다시 트리거되지 않도록
        if phase == 2 and proj_warmup_epochs == 0 and not unfreeze_done and epoch > warmup:
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
        # freeze 구간(encoder 동결) 동안은 항상 tf=1.0 유지, freeze 종료 후부터 감소 시작
        # (freeze 연장과 SS 조기 시작이 충돌하지 않도록 warmup 기준으로 오프셋)
        if ss_epochs > 0 and epoch > warmup:
            decay   = (tf_ratio_start - tf_ratio_min) / ss_epochs
            cur_tf  = max(tf_ratio_min, tf_ratio_start - decay * (epoch - warmup - 1))
        else:
            cur_tf  = tf_ratio_start

        seq2seq.train()
        if phase == 2 and proj_warmup_epochs > 0 and not proj_warmup_done:
            seq2seq.eval()  # proj 워밍업 중엔 전체 동결(BatchNorm/dropout 흔들림 방지)
            for i in warmup_bb_idxs:
                seq2seq.encoder.backbone[i].train()  # 워밍업 대상 backbone 단계만 BN 갱신 재허용
        elif phase == 2 and not unfreeze_done:
            seq2seq.encoder.eval()  # train()이 재귀적으로 되돌려놓은 BatchNorm을 다시 freeze
        t0 = time.time(); train_loss = 0.0
        for canvases, tgt_in, tgt_out, tgt_mask in train_loader:
            canvases = canvases.to(device)
            tgt_in   = tgt_in.to(device)
            tgt_out  = tgt_out.to(device)
            tgt_mask = tgt_mask.to(device)

            # Scheduled Sampling: cur_tf 확률로 GT 토큰 유지, 나머지는 모델 자신의 예측으로
            # 교체 -> exposure bias 완화. 기존엔 <UNK>로 가리기만 했는데(컨텍스트 불확실성에는
            # 강해지지만 "내가 틀린 토큰에서 이어가는 법"은 안 배움), 이제 실제로 모델이
            # 그 위치에서 뭘 예측했는지(1차 forward, no-grad)를 얻어 그 값으로 대체한다 --
            # 진짜 self-generated 컨텍스트를 재현(2배 forward, 순차 디코딩보다는 훨씬 저렴).
            if cur_tf < 1.0:
                with torch.no_grad():
                    with autocast(enabled=device.type == 'cuda'):
                        logits_ss = seq2seq(canvases, tgt_in, tgt_mask)
                    pred_next = logits_ss.argmax(-1)  # [B,T]: tgt_in[:,i]까지 보고 예측한 다음 토큰
                    # tgt_in[:,i] == tgt_out[:,i-1] 정렬이므로, 그 위치에 대응하는 "모델의 자기예측"은
                    # pred_next를 한 칸 오른쪽으로 밀고 SOS(위치 0)는 그대로 둔 것
                    self_pred_in = torch.cat([tgt_in[:, :1], pred_next[:, :-1]], dim=1)
                drop_mask = (torch.rand_like(tgt_in, dtype=torch.float) > cur_tf)
                drop_mask[:, 0] = False   # SOS 보존
                drop_mask = drop_mask & ~tgt_mask   # PAD 위치 제외
                tgt_in = torch.where(drop_mask, self_pred_in, tgt_in)

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
            pred   = (beam_decode(seq2seq, canvas, device) if phase == 3
                      else greedy_decode(seq2seq, canvas, device))
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
            best_acc = val_acc
            epochs_since_improve = 0
            save_ckpt(seq2seq, os.path.join(args.out_dir, 'seq2seq_best.pt'), {'epoch': epoch})
            print(f"    * Best Seq2Seq (TER={val_ter*100:.1f}%  Acc={val_acc:.1f}%)")
        else:
            epochs_since_improve += 1

        # 목표 미달 조기 중단: target_check_epoch까지 target_acc를 못 넘기면 남은 epoch을
        # 낭비하지 않고 멈춤 -- 여기서 GT vs PRED 오류 분석(error_breakdown.py) 후 코드 수정,
        # seq2seq_best.pt에서 --resume으로 재학습하라는 신호(EARLY_STOP 로그를 pod 쪽
        # 모니터링 스크립트가 감지해서 다음 단계로 넘어갈 수 있게 함).
        if target_acc is not None and epoch == target_check_epoch and best_acc < target_acc:
            print(f"EARLY_STOP: epoch {epoch}까지 목표 Acc {target_acc:.1f}%를 못 넘김 "
                  f"(현재 best={best_acc:.1f}%) -- 오류 분석 후 재학습 필요")
            break
        if patience is not None and epochs_since_improve >= patience:
            print(f"EARLY_STOP: {patience} epoch 연속 개선 없음 (best Acc={best_acc:.1f}%, "
                  f"epoch {epoch}) -- 정체 판단, 중단")
            break

        # 좁은 구간(예: 94~95%) 안에서 미세하게만 오르내리는 정체 감지 -- patience는
        # best_ter가 조금이라도 갱신되면 매번 리셋되어 이런 미세 정체를 못 잡아냄.
        if plateau_lo is not None:
            if plateau_lo <= val_acc <= plateau_hi:
                plateau_streak += 1
            else:
                plateau_streak = 0
            if plateau_streak >= plateau_band_epochs and (plateau_min_epoch is None or epoch >= plateau_min_epoch):
                print(f"EARLY_STOP: Acc가 {plateau_lo:.1f}~{plateau_hi:.1f}% 구간에서 "
                      f"{plateau_band_epochs} epoch 연속 정체 (epoch {epoch}, 현재={val_acc:.1f}%) "
                      f"-- 중단")
                break

    print(f"\nPhase {phase} done. Best TER={best_ter*100:.1f}%  Best Acc={best_acc:.1f}%")


def parse_args():
    p = argparse.ArgumentParser(description='Round 3 OMR 누적 학습 (Grand Staff)')
    p.add_argument('--phase',       type=int, required=True, choices=[1, 2, 3])
    p.add_argument('--data_dir',    required=True)
    p.add_argument('--tokenizer',
                   default=str(_HERE / 'tokenizer258.json'))
    p.add_argument('--out_dir',     default=str(_HERE / 'models'))
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch',       type=int,   default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--val_ratio',   type=float, default=0.1)
    p.add_argument('--workers',     type=int,   default=4)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--in_ch',       type=int,   default=1,
                   help='1(기본)=기존 그레이스케일 단일 채널. 2=CoordConv 실험 -- 세로 '
                        '좌표 채널 추가(2026-07-31, 단3도/옥타브 오독 가설 검증용)')
    p.add_argument('--extra_height_stages', type=int, default=4,
                   help='4(기본, 기존과 동일)=세로를 총 8단계로 1까지 완전히 뭉갬. '
                        '낮추면(예: 2) backbone 출력에 세로가 남은 채로 pool_h로 넘어감 '
                        '-- "높이 부분 보존" 실험(2026-07-31)')
    p.add_argument('--pool_h',      type=int,   default=1,
                   help='1(기본, 기존과 동일). >1이면 세로를 이 개수만큼 밴드로 나눠 '
                        '채널에 이어붙여 보존(extra_height_stages도 같이 낮춰야 의미 있음)')
    p.add_argument('--resume',        default=None,
                   help='이전 라운드 seq2seq_best.pt 경로')
    p.add_argument('--resume_tokenizer', default=None,
                   help='--resume 체크포인트가 학습될 당시 사용한 tokenizer.json 경로 '
                        '(--tokenizer와 vocab이 달라진 경우 지정, 예: note 토큰 분해 이전 버전)')
    p.add_argument('--segnet_ckpt',   default=None)
    p.add_argument('--device',        default='auto')
    p.add_argument('--freeze_epochs', type=int,   default=None,
                   help='Phase2 encoder 동결 warm-up epoch 수 (기본 None = epochs//5 fallback)')
    p.add_argument('--proj_warmup_epochs', type=int, default=0,
                   help='0(기본, 끔). >0이면 학습 시작 시 encoder.proj를 제외한 전체(디코더+'
                        '기존 backbone)를 얼리고 encoder.proj만 이 에폭 수만큼 먼저 학습한 뒤 '
                        '전체를 이어서 학습 -- "높이 부분 보존" 실험에서 encoder.proj가 완전히 '
                        '랜덤 초기화된 채 전체를 곧장 파인튜닝 학습률로 같이 학습시키면 이미 잘 '
                        '학습된 디코더/backbone이 초기 노이즈 신호에 끌려다닐 수 있다는 가설 검증용 '
                        '(2026-08-01). --freeze_epochs와 방향이 반대(그쪽은 encoder 전체를 얼리고 '
                        '디코더만 먼저 학습)라 함께 켜지 말 것 -- 켜지면 이 옵션이 우선 적용됨.')
    p.add_argument('--proj_warmup_lr', type=float, default=None,
                   help='proj 워밍업 구간 학습률 (기본 None = --lr 그대로 사용)')
    p.add_argument('--proj_warmup_extra_backbone_stages', type=int, default=0,
                   help='0(기본). >0이면 proj 워밍업 구간에 backbone 마지막 N단계도 함께 '
                        '학습 가능하게 열어둠(train() 모드로 BatchNorm도 갱신) -- proj만으로 '
                        '부족할 때, "뒤에 conv가 더 있다"는 전제로 학습됐던 마지막 이식 '
                        '단계(예: extra_height_stages=2일 때 backbone.8/9)가 pooling 직전 '
                        '역할에 맞게 같이 재적응하도록 (2026-08-01)')
    # Scheduled Sampling
    p.add_argument('--tf_ratio',      type=float, default=1.0,
                   help='초기 teacher forcing 비율 (기본 1.0 = 완전 TF)')
    p.add_argument('--min_tf_ratio',  type=float, default=0.1,
                   help='최소 teacher forcing 비율 (기본 0.1)')
    p.add_argument('--replay_dir',    default=None,
                   help='이전 단계 학습 디렉토리 경로 -- 지정 시 복사 없이 여기서 직접 '
                        'replay_count개를 무작위로 섞어 읽음 (원본 캐시 재사용)')
    p.add_argument('--replay_count',  type=int,   default=0,
                   help='--replay_dir에서 가져올 샘플 수 (기본 0 = replay 비활성)')
    p.add_argument('--ss_epochs',     type=int,   default=0,
                   help='TF 비율을 감소시킬 에폭 수 (기본 0 = SS 비활성)')
    p.add_argument('--span_weight',   type=float, default=1.0,
                   help='옥타브/헤어핀 시작-종료 토큰 loss 가중치 (기본 1.0 = 미적용)')
    p.add_argument('--noise_level',   type=int,   default=2, choices=[1, 2, 3, 4],
                   help='촬영 노이즈(기울기/블러/조명/압축) 강도 단계 (dataset.NOISE_LEVELS, 기본 2=잔여수준)')
    p.add_argument('--noise_level_max', type=int, default=None, choices=[1, 2, 3, 4],
                   help='지정 시 [noise_level, noise_level_max] 범위에서 샘플마다 무작위로 레벨을 뽑음')
    p.add_argument('--p_level_max', type=float, default=0.5,
                   help='noise_level_max를 뽑을 확률(기본 0.5=균등). 재검출 성공률이 낮은 '
                        '상위 레벨의 실효 노출량을 보정하려면 0.5보다 높게(예: 0.65)')
    p.add_argument('--page_level_noise', action='store_true',
                   help='페이지 레벨 노이즈+correct_perspective+오선 재검출을 거쳐 실제 추론 '
                        '경로를 재현(5n5 전용, dataset.page_noise_and_redetect). 재검출 실패 시 '
                        '기존 캔버스 레벨 경로로 자동 폴백')
    p.add_argument('--no_augment', action='store_true',
                   help='노이즈/기하 augmentation을 전부 끄고 깨끗한 렌더링 이미지로만 학습 '
                        '(OMRDataset augment=False -- noise_level/page_level_noise 등 다른 '
                        'noise 관련 인자는 전부 무시됨). 노이즈가 정확도 저하 원인인지, 아니면 '
                        '학습 분포 자체(음 선택/화성)가 실제 곡과 안 맞아서인지 분리 진단하고 '
                        '싶을 때 사용')
    p.add_argument('--target_acc', type=float, default=None,
                   help='목표 val Acc(%%). --target_check_epoch까지 이 값을 못 넘기면 '
                        '남은 epoch을 낭비하지 않고 조기 중단(EARLY_STOP 로그 출력 후 종료) '
                        '-- 원인 분석/코드 수정 후 --resume으로 재학습하라는 신호')
    p.add_argument('--target_check_epoch', type=int, default=None,
                   help='--target_acc 달성 여부를 확인할 epoch (기본 None=--epochs의 절반). '
                        '--target_check_after_unfreeze가 지정되면 이 값 대신 그쪽이 우선')
    p.add_argument('--target_check_after_unfreeze', type=int, default=None,
                   help='encoder unfreeze 시점(freeze_epochs) 기준 상대 epoch으로 --target_acc '
                        '체크 시점을 지정 (예: 20이면 unfreeze 후 20 epoch째). freeze_epochs가 '
                        '바뀌어도 "unfreeze 후 N epoch"이 항상 정확히 맞도록 --target_check_epoch '
                        '(절대 epoch) 대신 사용 권장 -- freeze 구간은 디코더만 적응해서 본격적인 '
                        '학습은 unfreeze 이후부터임')
    p.add_argument('--patience', type=int, default=None,
                   help='val TER(=1-Acc)이 이 epoch 수만큼 연속으로 개선(seq2seq_best.pt '
                        '갱신)되지 않으면 조기 중단 (기본 None=비활성, 정체 구간에서 GPU 시간 '
                        '낭비 방지)')
    p.add_argument('--plateau_band', type=str, default=None,
                   help='"저,고" 형식(예: "94,95")으로 지정 시, val Acc가 이 구간 안에서만 '
                        '--plateau_band_epochs만큼 연속으로 머물면 조기 중단 -- --patience와 '
                        '달리 미세 개선(94.0->94.3->94.6)이 있어도 구간을 못 벗어나면 정체로 '
                        '판단(patience는 "전혀 개선 없음"만 잡아서 이런 미세 정체는 못 잡음)')
    p.add_argument('--plateau_band_epochs', type=int, default=5,
                   help='--plateau_band 구간 안에 머무는 것을 정체로 판단할 연속 epoch 수 (기본 5)')
    p.add_argument('--plateau_min_epoch', type=int, default=None,
                   help='--plateau_band 조기중단을 이 epoch 이전에는 절대 발동시키지 않음 '
                        '(기본 None=제한 없음) -- 초반에 우연히 목표 구간에 들어가도 최소 이 '
                        'epoch까지는 계속 학습하도록 강제')
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
