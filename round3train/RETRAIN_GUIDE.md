# Round3 Phase2 재학습 절차 (round3train/ 단독 실행)

> 이 폴더(`round3train/`) 하나만 있으면 실행 가능. 다른 코드 불필요.

## 전제 (round3train/ 안에 있어야 할 것)

- 코드: `train.py`, `dataset.py`, `model.py`, `inference.py`, `generate_scores.py`, `relabel_notes.py` (git에서 받음)
- `tokenizer258.json` — 새 vocab (note/dur 분리, 258개). `--tokenizer`에 계속 사용.
- `tokenizer1013.json` — 구 vocab (1013개). Phase2 최초 1회 `--resume_tokenizer`에만 사용, 이후 불필요.
- `seq2seq_best.pt` — 이어받을 체크포인트 (vocab 1013 시절, epoch 25 / TER 26.1%). git에 없으므로 별도 전달분.
- Round3 학습 데이터 (별도 준비).

## 1) 라벨 재변환 (1회성, 이미지 재렌더링 불필요)

```bash
python round3train/relabel_notes.py --data_dir "<Round3 데이터 경로>" --in_place
```

## 2) Phase 2 학습 — 80 epoch 한 번에 (메인, 계속 실행)

```bash
python round3train/train.py --phase 2 \
    --data_dir "<Round3 데이터 경로>" \
    --tokenizer round3train/tokenizer258.json \
    --resume round3train/seq2seq_best.pt \
    --resume_tokenizer round3train/tokenizer1013.json \
    --out_dir round3train/models/round3_p2_new_vocab \
    --epochs 80
```

## 3) (별도 터미널, 병렬) 10 epoch마다 체크포인트 스냅샷

`train.py`는 매 epoch `seq2seq_last.pt`만 덮어쓰므로, 로그를 폴링해 10의 배수 시점 파일을 복사해둔다. 메인 학습은 건드리지 않음.

```bash
OUT="round3train/models/round3_p2_new_vocab"
SNAP="round3train/checkpoints_by_epoch"
LOG="$OUT/seq2seq_phase2_log.csv"
mkdir -p "$SNAP"
last_saved=0
while true; do
  if [ -f "$LOG" ]; then
    epoch=$(tail -n 1 "$LOG" | cut -d',' -f1)
    if [[ "$epoch" =~ ^[0-9]+$ ]] && [ $((epoch % 10)) -eq 0 ] && [ "$epoch" -ne "$last_saved" ]; then
      cp "$OUT/seq2seq_last.pt" "$SNAP/seq2seq_epoch${epoch}.pt"
      echo "[snapshot] epoch $epoch"
      last_saved=$epoch
    fi
  fi
  sleep 60
done
```

## 4) (별도 터미널, 병렬) 스냅샷마다 test 20~30장 정확도 리포트 + push

```bash
N=10   # 스냅샷 생길 때마다 10,20,30...로 반복
mkdir -p round3train/reports
python round3train/inference.py \
    --seq2seq round3train/checkpoints_by_epoch/seq2seq_epoch${N}.pt \
    --tokenizer round3train/tokenizer258.json \
    --analyze "<test 데이터 경로, 20~30장>" \
    --n_analyze 30 \
    --device cpu \
    > round3train/reports/epoch${N}_report.txt

git add round3train/reports/epoch${N}_report.txt
git commit -m "report: round3 phase2 epoch${N} 중간 점검"
git push
```

**주의**: `git add round3train/`나 `-A`는 금지 (체크포인트 대용량 `.pt` 포함될 수 있음). 리포트 `.txt`/`.csv`만 콕 집어서 add.

## 5) Phase 3 (Phase 2 완료 후)

Phase 2 결과물은 이미 vocab 258로 저장돼 있으므로 `--resume_tokenizer` 불필요.

```bash
python round3train/train.py --phase 3 \
    --data_dir "<Round3 데이터 경로>" \
    --tokenizer round3train/tokenizer258.json \
    --resume round3train/models/round3_p2_new_vocab/seq2seq_best.pt \
    --out_dir round3train/models/round3_p3_new_vocab \
    --epochs 30
```
