# 모바일 온디바이스 양자화 — 진행 상황

> 공식 TransNote 프로젝트(웹, `trans-note.vercel.app`) 제출과는 별개로 진행하는 개인 후속
> 작업. 목표: r15 체크포인트(`seq2seq_best.pt`)를 양자화해 모바일에서 빠른 속도 + 어느 정도
> 이상의 인식률로 온디바이스 추론. 포트폴리오 작성 염두 — 과정을 여기 계속 기록한다.

## 목표

- 속도: 모바일 기기에서 실사용 가능한 수준 (구체적 숫자 목표 아직 미정 — TODO)
- 정확도: FP32 기준(약 91%/캡처 96%) 대비 손실 최소화 (허용 손실폭 아직 미정 — TODO)

## 2026-08-24

### 발견 — 기존 시도가 이미 존재함

`train/export_tflite.py`가 Flutter/C++ TFLite 엔진 시절(2026-08-09 전면 폐기 전)에 만들어진
ONNX→TFLite export 스크립트로 저장소에 남아있었다. 처음부터 새로 설계하지 않고 여기서부터
이어가기로 함.

이 스크립트가 이미 내려둔 결론:
- **인코더는 INT8, 디코더는 기본 FP32** — "자기회귀 Transformer decoder의 INT8 양자화는
  토큰 오류율을 크게 악화시키는 경우가 많음"(스크립트 주석). 실무에서 이미 한 번 확인된
  리스크.
- **`--quantize_decoder` 미구현** — 이유: decoder는 past_ids+memory 두 입력을 함께 보정해야
  해서 인코더처럼 단일 입력 캘리브레이션 방식으로 안 됨.
- **디코더는 KV캐시 없이 export** — 매 스텝 past_ids 전체로 O(T) 재계산(608토큰 이하 감당
  가능 판단). 지금 서비스가 쓰는 `inference.py`의 `decode_step_cached()`(캐시로 가속)와는
  다른, 더 느린 경로. **속도 목표에 직결되는 트레이드오프 — 아직 미결정.**

### 버그 수정 — 하드코딩된 아키텍처 가정

`export_tflite.py`의 `load_seq2seq()`가 `OmrSeq2Seq(vocab_size=vocab_size)`를 생성자
기본값(`in_ch=1`)으로 호출하고 있었음. r15는 CoordConv(`in_ch=2`)라 그대로 실행하면
`load_state_dict()`에서 shape mismatch로 죽는 상태였음(실제로 미실행 상태였던 걸로 추정).

`handler.py`와 동일한 패턴으로 `infer_arch_from_state_dict()`를 이용해 체크포인트에서 실제
아키텍처(`in_ch`/`extra_height_stages`/`pool_h`)를 역산하도록 수정. 연쇄적으로
`_export_onnx_encoder()`/`_build_encoder_calib()`의 채널 수 하드코딩도 같이 수정
(`make_model_input()` 재사용).

**검증 완료**: r15 체크포인트로 직접 로드 테스트 →
`{'in_ch': 2, 'extra_height_stages': 4, 'pool_h': 1}` 정상 감지, 인코더 ONNX export 성공
(57MB). 커밋 `a3e961c`.

### 확인 — SEQ_LEN=320 하드코딩 값

디코더 export(`_export_onnx_decoder`)가 memory 더미 텐서를 `[1, SEQ_LEN=320, 512]`로 고정
사용. r15 인코더를 실제로 돌려 확인한 결과, 단일 오선(`CANVAS_H=320`)/대보표
(`SYSTEM_CANVAS_H=480`) 두 캔버스 모두 인코더 출력 길이가 320으로 동일하게 나옴 — 유효한
값, 수정 불필요. (`CANVAS_W=1280`에서 유도된 폭 기준 다운샘플링 값이라 캔버스 높이와
무관하게 고정되는 구조.)

### A단계 마무리 — 같은 버그가 `inference.py`의 CLI 진입점(`main()`)에도 있었음

CLAUDE.md가 공식 안내하는 명령(`python train/inference.py --seq2seq ... --analyze ...`)의
`main()`도 `export_tflite.py`와 똑같이 `OmrSeq2Seq(vocab_size=len(tok2id))`를 기본값으로
생성하고 있었음 — r15로 이 명령을 실행하면 그대로 shape mismatch. 동일 패턴으로
`infer_arch_from_state_dict()` 적용해 수정. 즉 이 버그가 **세 곳**(handler.py는 원래부터
정상, export_tflite.py, inference.py main())에서 반복됐던 것 — r15로의 전환(CoordConv 도입)
당시 handler.py만 고쳐지고 나머지 두 스크립트는 안 고쳐진 채로 방치됐던 것으로 보임.

### 저장소 결정 확정

TransNote 저장소가 이번에 "웹앱 제거 + RunPod/모바일 중심"으로 재편되면서 자연히 해결됨 —
`export_tflite.py`/양자화 작업은 TransNote에 그대로 두고 계속 진행. 표기법 규칙 문서만
Model_TransNote로 이전(학습 코드가 거기 있으므로).

**A단계 완료.**

### B단계 착수 — FP32 베이스라인 실측

`inference.py` 수정 직후 바로 실행 가능해져서 로컬에 남아있던 `test/data`(20개 합성 샘플,
.png+.json 쌍)로 첫 베이스라인을 재봤다:

```bash
python train/inference.py --seq2seq train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt \
    --tokenizer train/tokenizer258.json --analyze test/data --n_analyze 20
```

결과 (2026-08-24, CPU, greedy decode):
- 전체 TER 기준 Acc **67.9%**, 음표 단위 Acc **81.7%** (Treble 85.9% / Bass 88.0%)
- header 90.3%, note/rest 87.8%, barline 92.5%, dynamic 89.3%

**⚠️ 주의 — 포스터의 91%/96%와 직접 비교 불가**: 이 20개는 `test/data`에 남아있던 합성
데이터이고, 포스터의 91%(전체)/96%(캡처 이미지) 수치가 어떤 held-out 셋을 썼는지는 여기
기록에 없다(Model_TransNote 쪽 `TRAINING_REPORT.md`에 근거가 있을 것 — 아직 확인 안 함).
지금 숫자는 **"양자화 전/후 비교용 내부 기준점"**으로만 쓴다 — 같은 20개 샘플, 같은 명령으로
재실행하면 항상 비교 가능하다는 게 핵심이지, 포스터 수치를 재현한 게 아니다.

한 가지 구체적 오류 패턴도 확인됨: 3/4 vs 6/8 박자표 혼동(`correct_time_signature`가
휴리스틱으로 처리하는 바로 그 케이스, 길이 합이 같아서 구분이 원래 어려움) — 포트폴리오용
"실패 사례" 후보로 기록해둠.

**참고**: `realImage/scoped_test_kakao/compare_report.txt`에 실사(카카오톡) 사진 비교
리포트가 이미 존재하지만, 어느 체크포인트로 만들어졌는지 기록이 없어 지금은 신뢰 안 함 —
필요하면 나중에 같은 방식으로 재현.

**더 큰 held-out 후보 발견**: `train/data/local_pools/exactpicture_test_full/`에 135개
실제 곡 기반(Chopin/Czerny/뉴에이지 등, PPT 27p의 장르 목록과 일치) ground-truth JSON이
있음 — 다만 **매칭되는 렌더링 이미지(.png)가 없어서** 지금 바로는 못 씀. MuseScore 렌더링
파이프라인(`render_custom_notation.py` 등)으로 이미지부터 만들어야 함 — 다음 세션 후보
작업.

### B단계 — CPU 속도 1차 실측

같은 체크포인트로 타이밍 측정(개발 PC CPU, 실제 모바일 기기 아님 — GPU 없는 하한 기준점
용도):
- 20개 배치 전체: 104.0초
- 단일 이미지(모델 로드 포함): 11.0초
- → 역산: **모델 로드 ~6.1초(1회) + 이미지당 ~4.9초**(대보표, greedy decode, CPU)

4.9초/장은 실사용 목표로는 느린 수준 — GPU 없이 FP32로 도는 하한선이라는 의미가 크고,
양자화·모바일 하드웨어 가속(NPU 등) 없이는 목표 속도 달성이 어렵다는 근거가 됨.

### 중간에 발생한 우회 — 두 손 박자 타이밍 어긋남 버그 발견·수정 (본 프로젝트 범위 밖이지만 선결 과제)

myweb 실사용 중 "곡 후반부로 갈수록 치/베이스 타이밍이 어긋난다"는 리포트가 들어와 원인을
찾아보니 `server/token_to_notes.py`가 barline마다 치/베이스 누적 박자 위치를 **무조건
리셋**하고 있었음(모델이 한 마디에서 duration을 잘못 예측해도 검증 없이 그냥 다음 마디로
넘어감 -- 한 마디의 오차가 다음 마디로 계속 전파됨). 모델 재디코딩 없이, 확정된 박자표
기준으로 마디 안 모든 음표 duration을 비례 스케일링해서 정확히 맞추는 `snap_measure_to_
time_sig()`를 추가(커밋 `5d65beb`). 실사 10곡(newage21~30) 검증 후 비례 축소 방식으로
확정.

양자화 자체와는 별개 트랙이지만, **자기회귀 디코더 오차 누적**이라는 같은 클래스의 문제라
먼저 처리. 오차가 이미 누적되는 상태에서 양자화하면 노이즈가 더해져 악화될 위험이 있었음.

### RunPod git 자동배포 부작용 발견·수정

webpage 제거 커밋을 push한 후 Vercel이 여전히 GitHub 연동으로 자동 재배포를 시도하다
"No FastAPI entrypoint found"로 실패하는 걸 확인(빌드 실패 메일) -- `vercel.json`에
`deploymentEnabled:false`만 남겨서 방지(커밋 `a82b19a`). 이 저장소에 계속 push할 예정이라
없었으면 매번 반복됐을 문제.

**RunPod Container Image 캐싱 관련 중요 교훈**: RunPod은 `:latest` 태그를 캐싱하는
경우가 있어(Dockerfile 주석에 기록된 전례), 코드를 고쳐서 push해도 **RunPod 대시보드에서
Container Image를 새 커밋의 구체적 `:<sha>` 태그로 직접 바꿔줘야 실제로 반영된다.**
실제로 이번 세션 초반 "5-6초/2-3초" 속도 문제가 12일 전 이미지(cuDNN 워밍업 픽스 이전)가
계속 걸려있었던 게 원인이었음(Docker Hub API로 직접 확인) -- 이후 최신 sha로 교체하니
4초대/2초대로 개선 확인됨. **앞으로 이 저장소에 관련 코드를 push할 때마다 이 단계를
잊지 말 것.**

### 목표 스택/디코더 방식 결정

- **타겟 스택**: Android(TFLite)만 먼저 진행. iOS(Core ML)는 필요 시 나중에 별도 트랙으로.
- **디코더 KV캐시**: 우선 O(T²) 재계산(캐시 없음, `export_tflite.py`가 이미 이 방식으로
  구현돼 있음)으로 가서 실기 속도 측정 후 KV캐시 구현 필요 여부 재결정. 지금 바로 상태유지형
  캐시를 만들지 않음(엔지니어링 난이도 대비 아직 근거 부족).

## 다음 단계 (TODO)

- [x] ~~export_tflite.py 아키텍처 버그 수정~~ (2026-08-24, `a3e961c`)
- [x] ~~inference.py main() 같은 버그 수정~~ (2026-08-24)
- [x] ~~저장소 결정~~ — TransNote에서 계속
- [x] ~~FP32 베이스라인 1차 측정~~ — test/data 20개, Acc 81.7%(음표 기준). 더 크고 대표성
      있는 held-out 셋으로 재측정 필요(아래)
- [x] ~~CPU 속도 1차 실측~~ — 개발 PC CPU 기준 이미지당 ~4.9초(대보표) + 모델 로드 ~6.1초.
      실제 모바일 기기 아님, 하한 기준점일 뿐 — 실기 측정은 여전히 TODO
- [ ] export_tflite.py 전체 파이프라인(인코더+디코더) 끝까지 실행 테스트 — 지금은 인코더
      export만 개별 검증함
- [ ] 목표 스택 결정: Android(TFLite)만 vs iOS(Core ML)까지
- [ ] **디코더 KV캐시 결정**: 상태유지형 온디바이스 구현 vs O(T²) 재계산 감수
- [ ] 더 크고 대표성 있는 held-out 셋 확보 + 정확도 재측정 — `train/data/local_pools/
      exactpicture_test_full/`(135개, 실제 곡 기반) 후보 발견했으나 렌더링 이미지가 없어
      MuseScore 파이프라인으로 먼저 만들어야 함
- [ ] 실제 모바일 기기 기준 속도 베이스라인 측정 (지금 있는 건 개발 PC CPU 수치뿐)
- [ ] INT8 캘리브레이션용 대표 이미지셋 확보(장르 다양, 대보표 검출 가능한 것)
- [ ] 정량 목표치(정확도 허용폭, 속도 목표) 숫자로 확정
- [ ] 실제 타깃 기기 확보 + 프로파일링 도구 준비
- [ ] `--quantize_decoder` 이중 입력 캘리브레이션 문제 — 시도할지 여부 결정

## 실험 기록 (진행되면 추가)

| 날짜 | 정밀도 방식 | 크기 | held-out 정확도 (test/data 20개, 음표 기준) | 속도 | 메모 |
|---|---|---|---|---|---|
| 2026-08-24 | FP32 (r15, 베이스라인) | 185MB | 81.7% (Treble 85.9%/Bass 88.0%) | 개발 PC CPU ~4.9초/장(+로드 ~6.1초, 1회) | 표본 20개뿐·CPU 수치. 실기/실사 재측정 필요 |

## 포트폴리오용 캡처 포인트 (계획)

- [ ] 크기 비교 (FP32 vs 양자화 후)
- [ ] 정확도 표/그래프
- [ ] 실기 데모 영상/캡처
- [ ] 실패 사례(디코딩 붕괴 등) — 문제 발견→원인 분석→해결 서사용
