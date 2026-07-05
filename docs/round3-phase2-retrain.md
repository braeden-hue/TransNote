# round3-phase2-retrain.md

> 관련 코드: `round3train/`, `ml/omr/engine/`
> 작성 시점: note vocab 분해 + beam search 도입 직후, Round3 Phase2 재학습 착수 전

## 배경

Round3 음표(note) 인식률이 다른 카테고리(clef/barline/staff-bass 등)보다 현저히 낮다는 문제에서 출발했다.
분석 결과 두 가지 원인이 확인됐다:

1. `note-{pitch}-{dur}` 토큰이 pitch+duration을 한 토큰으로 묶어 850개 클래스로 쪼개져 있어, rest(10개)·구조 토큰(수개)보다 훨씬 데이터가 희소했다.
2. Round3는 grand staff(대보표) 지원을 위한 Phase 2(인코더 고정 웜업)를 한 번도 제대로 실행하지 않았고, 이후 Phase 3 재시도들은 전부 loss가 증가하며 발산했다(`evidence_training_logs` 기준).

이번 소단계에서는 (1)을 코드 레벨에서 해결하고, (2)를 실제로 고치기 위한 재학습 준비를 마쳤다.

---

## 변경 사항

### 1. note 토큰 분해 (vocab 1013 → 258)

`note-{pitch}-{dur}` 단일 토큰(850개)을 `note-{pitch}`(85개, pitch 전용) + `dur-{dur}`(10개, duration 전용)로 분해했다.
`rest-*`/`chord-*`는 이미 pitch 또는 duration 단독 토큰이라 그대로 뒀다.

| 파일 | 변경 |
|---|---|
| `round3train/tokenizer.json` | note-* 850개 제거, note-{pitch}(85)+dur-{dur}(10) 추가. clef/key/time/chord/rest/barline/... 나머지는 내용·순서 보존한 채 ID만 재배열. **vocab 1013 → 258** |
| `round3train/generate_scores.py` | note/tuplet 토큰 생성부 3곳을 `note-{pitch}` + `dur-{dur}` 2토큰 방출로 수정 |
| `round3train/model.py` | `VOCAB_SIZE` 상수 258로 갱신 |
| `round3train/train.py` | `load_ckpt_vocab_expand`(raw index 기반, vocab 축소 시 예외)를 `load_ckpt_partial_vocab`(토큰 **문자열** 매칭 기반)로 교체. `--resume_tokenizer` 인자 추가 — 구버전 체크포인트를 새 vocab으로 워밍스타트할 때 구버전 tokenizer.json 경로를 지정한다. `fix_chord_tokens`의 orphan 판정 조건에 `dur-` 추가(그렇지 않으면 화음 토큰이 전부 삭제되는 버그가 있었음 — 검증 스크립트로 100% 재현 확인 후 수정) |
| `round3train/inference.py` | 동일하게 `fix_chord_tokens` 수정. `_NOTE_PREFIXES`에 `dur-` 추가(카테고리 집계/오류율 계산 반영) |
| `round3train/relabel_notes.py` (신규) | 이미지 재렌더링 없이 기존 라벨 JSON의 `note-{pitch}-{dur}`만 `note-{pitch}`+`dur-{dur}`로 변환하는 일회성 스크립트. `--in_place` 또는 `--out_dir`로 원본 보존 |

**검증**: `load_ckpt_partial_vocab`을 구/신 tokenizer로 시뮬레이션한 결과, 공유 토큰 163/258행(문자열이 같은 clef/rest/barline/dynamic/staff-bass 등)이 구 체크포인트 값 그대로 이어졌고, 신규 `note-{pitch}` 행은 정상적으로 랜덤 초기화됨을 확인. 실사 이미지 20장(`ml/data/test_eval/round3`)으로 OLD(1013) vs NEW(258) vocab을 동일 워밍스타트/lr/epoch로 12에폭 비교한 결과, **전체 토큰 정확도·loss는 NEW가 일관되게 우세**(val tok_acc: OLD ~14% vs NEW ~21%, train_loss 12epoch 후: OLD 6.93→3.19 vs NEW 5.89→2.54)했다. 다만 pitch 단독 정확도는 val 위치 수(126개)가 너무 적어 통계적으로 확정하지 못했다 — 실제 검증은 전체 데이터로 재학습해야 한다.

### 2. StaffCanvas 중복 정의 수정 (`ml/omr/engine`)

`types.hpp`가 미사용 `struct StaffCanvas { uint8_t pixels[...][...]; }`를, `staff_canvas.hpp`가 실제 사용되는 `class StaffCanvas`(`build_tiles()` 등)를 각각 정의하고 있어 **엔진 전체가 컴파일 자체가 안 되는 상태**였다. `.pixels` 멤버를 참조하는 코드가 전체 엔진에 없음을 확인 후 `types.hpp` 쪽을 제거(`CANVAS_H`/`CANVAS_W` 상수는 다른 파일에서 실사용되므로 유지).

### 3. Beam search 구현

기존엔 greedy(argmax) decoding만 있었다(`ml/scripts/make_ppt.py`에 아이디어만 언급, 실제 코드는 없었음). **참고**: 2026-07-02 커밋 `082e02a`에서 beam search를 한 번 시도했다가 "모든 빔이 key-C로 수렴한다"는 이유로 되돌린 이력이 있다 — 이번에 재현 검증한 결과 **greedy도 동일하게 key-C로 100% 수렴**함을 확인했다(아래 "알려진 한계" 참고). 즉 원인은 beam search가 아니라 모델 자체의 미학습이었다.

| 파일 | 변경 |
|---|---|
| `ml/omr/engine/include/decoder_runner.hpp`, `src/decoder_runner.cpp` | `decode()`(greedy)는 유지하고 `decode_beam(encoder_out, beam_width=4, length_penalty=0.7)` 추가. TFLite KV-cache가 빔마다 독립적이어야 해서, `step()`을 전체 로짓을 반환하는 `step_logits()`로 리팩터링하고, 빔 분기 시 `clone_kv_cache()`로 캐시를 복제한다. 완료(EOS)된 빔은 점수 고정한 채 다음 라운드 후보로 이월시켜 마지막에 길이 정규화 점수로 최종 선택. `beam_width<=1`이면 `decode()`로 위임(동치) |
| `ml/omr/engine/src/omr_engine.cpp` | 파이프라인 호출을 `decoder.decode(enc_out)` → `decoder.decode_beam(enc_out, kBeamWidth=4)`로 교체 |
| `round3train/inference.py` | C++와 동일 알고리즘의 파이썬/PyTorch 버전 `beam_decode()` 추가(이 모델은 KV-cache가 없어 캐시 복제 없이 빔마다 누적 시퀀스만 들고 있으면 됨) |

**검증**:
- C++ 쪽은 Android/TFLite 툴체인이 없어 실제 빌드는 못 했다. MinGW g++로 OpenCV/TFLite 최소 스텁을 만들어 `decoder_runner.cpp`/`omr_engine.cpp` 문법·타입 체크(`-Wall -Wextra`)만 통과시켰다 — TFLite 텐서 shape/인덱스 정합성은 미검증.
- 파이썬 쪽은 실제 체크포인트(`ml/models/round3/seq2seq_best.pt`, `load_ckpt_partial_vocab`로 워밍스타트)와 실사 이미지 20장으로 실행:
  - `beam_width=1`이 `greedy_decode`와 **토큰 시퀀스 100% 일치** (알고리즘 정합성 확인).
  - `beam_width=4`: 평균 TER 103.4%→**101.0%**, note_err 96.8%→97.2%(거의 동일), 소요시간은 약 3배(3.3s/장→9.7s/장). 장별로는 편차가 커서(num15: 147%→94%로 크게 개선 / num8: 95%→112%로 악화) 그리디의 국소적 실수를 종종 피해주는 정도이지, 모델 자체의 근본 문제(아래)를 해결해주진 않는다.

### 4. Prior correction 시도 — 실패 (기록만, 코드는 되돌림)

"beam search가 특정 key로만 수렴하는 문제"를 풀기 위해 `generate_scores.py`의 `KS_WEIGHTS`/`TS_WEIGHTS`(실제 생성 확률)로 디코드 시점 logit 보정(`logit - tau*log(prior)`, Menon et al. 2021)을 시도했으나 **효과가 없었다**:

| 설정 | key 정확도 (20장) |
|---|---|
| greedy (기준) | 30% (항상 key-C) |
| +prior 보정, tau=1.0 (버그: 미생성 key에도 epsilon prior로 보정) | 0% (key-Eb로 수렴 지점만 이동) |
| +prior 보정, tau=1.0 (수정: 실제 생성되는 5개 key만 보정) | 30% (greedy와 동일) |
| +prior 보정, tau=5.0 | 5% (key-Bb로 수렴 지점만 이동) |

보정 강도를 올려도 이미지별로 실제 key를 맞히기 시작하는 게 아니라 수렴 지점만 다른 고정값으로 옮겨간다 — 즉 "prior가 약한 신호를 가리는" 상황이 아니라 **모델이 이미지에서 key를 읽는 신호 자체를 학습 못 한** 상황이다. 이 경우 디코딩 시점 트릭으로는 해결 불가능하며, `round3train/inference.py`에 추가했던 `compute_prior_logbias`/`prior_bias`/`tau` 관련 코드는 제거했다(작동 안 하는 옵션을 남겨두지 않기 위함). **같은 시도를 반복하지 않도록 이 표만 기록해둔다.**

---

## 알려진 한계 / 근본 원인

- 위 실험들이 공통으로 가리키는 근본 원인은 **Round3가 Phase 2(인코더 고정 웜업)를 제대로 실행한 적이 없다**는 것이다. Phase 3(전체 언프리즈) 재시도들은 로그상 전부 loss가 증가하며 발산했다.
- vocab 분해·beam search 둘 다 "회귀는 아니지만 근본 해결도 아님" — 현재 체크포인트(`ml/models/round3/seq2seq_best.pt`)는 key/pitch 등 이미지 의존 신호를 사실상 못 읽는 상태이고, 이건 재학습으로만 풀린다.
- 이번 세션에서 돌린 실측 실험은 전부 **실사 이미지 20장짜리 toy 세트**(`ml/data/test_eval/round3`) 기준이다 — 정식 검증은 전체 Round3 데이터셋으로 해야 한다.

---

## 다음 단계: Round3 Phase2 재학습 실행 방법

Round1/Round2가 쌓은 인코더·디코더 구조 학습(vocab 포맷과 무관)은 재사용하고, Round3 지점에서만 새 vocab으로 전환한다 — Round1/2를 새 vocab으로 다시 돌 필요는 없다(비용 대비 이득 근거는 이 문서 "배경" 및 대화 로그 참고).

```bash
# 기존 라벨 재라벨링 (이미지 재렌더링 불필요)
python round3train/relabel_notes.py --data_dir "<Round3 데이터 경로>" --in_place

# Phase 2 — 반드시 실행 (지금까지 누락됐던 단계)
python round3train/train.py --phase 2 \
    --data_dir "<Round3 데이터 경로>" \
    --tokenizer round3train/tokenizer.json \
    --resume ml/models/round3/seq2seq_best.pt \
    --resume_tokenizer "<구버전 tokenizer.json 경로, vocab=1013>" \
    --out_dir ml/models/round3_p2_new_vocab \
    --epochs 80

# Phase 3
python round3train/train.py --phase 3 \
    --data_dir "<Round3 데이터 경로>" \
    --tokenizer round3train/tokenizer.json \
    --resume ml/models/round3_p2_new_vocab/seq2seq_best.pt \
    --out_dir ml/models/round3_p3_new_vocab \
    --epochs 30

# 검증 (note/rest/dur 카테고리별 recall 확인)
python round3train/inference.py \
    --seq2seq ml/models/round3_p3_new_vocab/seq2seq_best.pt \
    --tokenizer round3train/tokenizer.json \
    --analyze "<검증 데이터 경로>"
```

GPU(RTX 3080) 환경에서 실행 필요 — 이 세션은 CPU라 대규모 학습은 수행하지 못했다.

## 남은 항목 (별도 논의 필요)

- `ml/omr/training/` + `ml/data/tokenizer.json`(1012, 공유 tokenizer) 경로는 이번 vocab 변경에 포함되지 않았다. `ml/scripts/train_round.py`는 이 경로를 기본값으로 쓰므로, round3train과의 이원화를 어떻게 정리할지 별도 결정 필요.
- Android `externalNativeBuild` → `ml/omr/engine/CMakeLists.txt` 연결이 실제 빌드 환경에서 검증되지 않음(OpenCV prefab vs 수동 경로 변수 불일치 가능성, `CLAUDE.md` "Known Gaps" 참고).
