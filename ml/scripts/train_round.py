#!/usr/bin/env python3
"""
train_round.py  --  라운드별 누적 학습 실행기 (SegNet → Encoder → Decoder)

각 라운드의 데이터 + 이전 라운드 가중치로 3-Phase 학습을 자동 실행합니다.
  Phase 1: SegNet (픽셀 세그멘테이션)
  Phase 2: Encoder + Decoder (SegNet 고정, warm-up 후 해제)
  Phase 3: End-to-End 파인튜닝

Round N은 Round N-1의 가중치를 시작점으로 사용하며, vocab이 확장된 경우
Xavier 초기화로 새 토큰 행을 자동 패딩합니다.

Usage:
    # Round 1 처음 학습 (이전 가중치 없음)
    python ml/scripts/train_round.py --round 1

    # Round 2: Round 1 가중치를 이어받아 학습
    python ml/scripts/train_round.py --round 2 \\
        --prev-segnet  ml/models/round1/segnet_best.pt \\
        --prev-seq2seq ml/models/round1/seq2seq_best.pt

    # Round 3 (Phase 1 건너뛰고 Phase 2/3만)
    python ml/scripts/train_round.py --round 3 \\
        --prev-segnet  ml/models/round2/segnet_best.pt \\
        --prev-seq2seq ml/models/round2/seq2seq_best.pt \\
        --skip-phase1

    # GPU/배치 설정 조정
    python ml/scripts/train_round.py --round 2 \\
        --prev-segnet  ml/models/round1/segnet_best.pt \\
        --prev-seq2seq ml/models/round1/seq2seq_best.pt \\
        --epochs-p1 30 --epochs-p2 80 --epochs-p3 30 \\
        --batch-p1 16 --batch-p2 8 --batch-p3 4

Round별 기본 경로 (--data-dir, --tokenizer, --out-dir 미지정 시 자동 설정):
    ml/data/Round{N}/          -- 학습 이미지+JSON 쌍
    ml/data/round_tokens/round{N}_tokens.json  -- 해당 라운드 토큰 정의
    ml/models/round{N}/        -- 체크포인트 출력
    ml/data/tokenizer.json     -- 공통 tokenizer (모든 라운드 공유)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Round defaults
# ─────────────────────────────────────────────────────────────────────────────

ROUND_DEFAULTS = {
    1: {"epochs_p1": 50,  "epochs_p2": 100, "epochs_p3": 50,
        "batch_p1": 16,   "batch_p2": 8,    "batch_p3": 4,
        "lr_p1": 1e-4,    "lr_p2": 1e-4,   "lr_p3": 3e-5},
    2: {"epochs_p1": 30,  "epochs_p2": 80,  "epochs_p3": 30,
        "batch_p1": 16,   "batch_p2": 8,    "batch_p3": 4,
        "lr_p1": 5e-5,    "lr_p2": 8e-5,   "lr_p3": 2e-5},
    3: {"epochs_p1": 20,  "epochs_p2": 80,  "epochs_p3": 30,
        "batch_p1": 16,   "batch_p2": 8,    "batch_p3": 4,
        "lr_p1": 5e-5,    "lr_p2": 2e-4,   "lr_p3": 5e-5},
    4: {"epochs_p1": 20,  "epochs_p2": 60,  "epochs_p3": 30,
        "batch_p1": 16,   "batch_p2": 8,    "batch_p3": 4,
        "lr_p1": 3e-5,    "lr_p2": 5e-5,   "lr_p3": 1e-5},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_phase(phase: int, data_dir: str, tokenizer: str, out_dir: str,
               epochs: int, batch: int, lr: float,
               resume: str | None, segnet_ckpt: str | None,
               workers: int, seed: int, device: str) -> int:
    """Invoke train.py for one phase; returns process returncode."""
    train_script = str(_repo_root() / "omr" / "training" / "train.py")

    cmd = [
        sys.executable, train_script,
        "--phase",    str(phase),
        "--data_dir", data_dir,
        "--tokenizer", tokenizer,
        "--out_dir",  out_dir,
        "--epochs",   str(epochs),
        "--batch",    str(batch),
        "--lr",       str(lr),
        "--workers",  str(workers),
        "--seed",     str(seed),
        "--device",   device,
    ]
    if resume:
        cmd += ["--resume", resume]
    if segnet_ckpt:
        cmd += ["--segnet_ckpt", segnet_ckpt]

    print(f"\n{'='*60}")
    print(f"  Phase {phase}  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, check=False)
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OMR 라운드별 누적 학습 실행기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--round", type=int, required=True, choices=[1, 2, 3, 4],
                   help="학습할 라운드 번호 (1–4)")

    # Path overrides (auto-set from round number if omitted)
    p.add_argument("--data-dir",   default=None,
                   help="학습 데이터 디렉토리 (기본: ml/data/Round{N})")
    p.add_argument("--tokenizer",  default=None,
                   help="tokenizer.json 경로 (기본: ml/data/tokenizer.json)")
    p.add_argument("--out-dir",    default=None,
                   help="체크포인트 출력 경로 (기본: ml/models/round{N})")

    # Previous round weights (for rounds 2-4)
    p.add_argument("--prev-segnet",  default=None,
                   help="이전 라운드 SegNet 가중치 (.pt)")
    p.add_argument("--prev-seq2seq", default=None,
                   help="이전 라운드 Seq2Seq 가중치 (.pt, vocab expansion 자동 처리)")

    # Phase control
    p.add_argument("--skip-phase1", action="store_true",
                   help="Phase 1 (SegNet 학습) 건너뛰기 — 이전 segnet 가중치 사용")
    p.add_argument("--skip-phase3", action="store_true",
                   help="Phase 3 (End-to-End) 건너뛰기")
    p.add_argument("--only-phase",  type=int, default=None, choices=[1, 2, 3],
                   help="해당 Phase만 실행 (--skip-phase 옵션 무시)")

    # Epoch / batch overrides
    p.add_argument("--epochs-p1",   type=int, default=None)
    p.add_argument("--epochs-p2",   type=int, default=None)
    p.add_argument("--epochs-p3",   type=int, default=None)
    p.add_argument("--batch-p1",    type=int, default=None)
    p.add_argument("--batch-p2",    type=int, default=None)
    p.add_argument("--batch-p3",    type=int, default=None)

    # General
    p.add_argument("--device",   default="auto")
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--seed",     type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    root = _repo_root()
    N    = args.round
    D    = ROUND_DEFAULTS[N]

    # ── Resolve paths ─────────────────────────────────────────────────────────
    data_dir  = args.data_dir  or str(root / f"data/Round{N}")
    tokenizer = args.tokenizer or str(root / "data/tokenizer.json")
    out_dir   = args.out_dir   or str(root / f"models/round{N}")

    os.makedirs(out_dir, exist_ok=True)

    # ── Epoch / batch with round defaults ────────────────────────────────────
    ep1 = args.epochs_p1 or D["epochs_p1"]
    ep2 = args.epochs_p2 or D["epochs_p2"]
    ep3 = args.epochs_p3 or D["epochs_p3"]
    bt1 = args.batch_p1  or D["batch_p1"]
    bt2 = args.batch_p2  or D["batch_p2"]
    bt3 = args.batch_p3  or D["batch_p3"]

    # ── Validate paths ────────────────────────────────────────────────────────
    if not os.path.isdir(data_dir):
        sys.exit(f"ERROR: data directory not found: {data_dir}")
    if not os.path.isfile(tokenizer):
        sys.exit(f"ERROR: tokenizer not found: {tokenizer}")

    print(f"\n{'#'*60}")
    print(f"  OMR Round-{N} 학습 시작")
    print(f"  Data    : {data_dir}")
    print(f"  Tokenizer: {tokenizer}")
    print(f"  Output  : {out_dir}")
    if args.prev_segnet:
        print(f"  Prev SegNet : {args.prev_segnet}")
    if args.prev_seq2seq:
        print(f"  Prev Seq2Seq: {args.prev_seq2seq}")
    print(f"{'#'*60}")

    phases_to_run = (
        [args.only_phase] if args.only_phase
        else [p for p in [1, 2, 3]
              if not (p == 1 and args.skip_phase1)
              and not (p == 3 and args.skip_phase3)]
    )

    # ── Phase 1: SegNet ───────────────────────────────────────────────────────
    if 1 in phases_to_run:
        rc = _run_phase(
            phase=1, data_dir=data_dir, tokenizer=tokenizer, out_dir=out_dir,
            epochs=ep1, batch=bt1, lr=D["lr_p1"],
            resume=None,
            segnet_ckpt=args.prev_segnet,
            workers=args.workers, seed=args.seed, device=args.device,
        )
        if rc != 0:
            sys.exit(f"Phase 1 failed (returncode={rc})")

    # Determine SegNet checkpoint for Phase 3
    segnet_for_p3 = os.path.join(out_dir, "segnet_best.pt")
    if not os.path.isfile(segnet_for_p3):
        segnet_for_p3 = args.prev_segnet  # fallback to prev round

    # ── Phase 2: Encoder + Decoder ───────────────────────────────────────────
    if 2 in phases_to_run:
        rc = _run_phase(
            phase=2, data_dir=data_dir, tokenizer=tokenizer, out_dir=out_dir,
            epochs=ep2, batch=bt2, lr=D["lr_p2"],
            resume=args.prev_seq2seq,           # vocab expansion handled inside train.py
            segnet_ckpt=None,
            workers=args.workers, seed=args.seed, device=args.device,
        )
        if rc != 0:
            sys.exit(f"Phase 2 failed (returncode={rc})")

    # ── Phase 3: End-to-End ───────────────────────────────────────────────────
    if 3 in phases_to_run:
        seq2seq_for_p3 = os.path.join(out_dir, "seq2seq_best.pt")
        if not os.path.isfile(seq2seq_for_p3):
            seq2seq_for_p3 = args.prev_seq2seq

        rc = _run_phase(
            phase=3, data_dir=data_dir, tokenizer=tokenizer, out_dir=out_dir,
            epochs=ep3, batch=bt3, lr=D["lr_p3"],
            resume=seq2seq_for_p3,
            segnet_ckpt=segnet_for_p3,
            workers=args.workers, seed=args.seed, device=args.device,
        )
        if rc != 0:
            sys.exit(f"Phase 3 failed (returncode={rc})")

    print(f"\n{'#'*60}")
    print(f"  Round-{N} 학습 완료!")
    print(f"  체크포인트 저장 위치: {out_dir}")
    print(f"  다음 라운드 실행 예시:")
    print(f"    python ml/scripts/train_round.py --round {N+1} \\")
    print(f"        --prev-segnet  {out_dir}/segnet_best.pt \\")
    print(f"        --prev-seq2seq {out_dir}/seq2seq_best.pt")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
