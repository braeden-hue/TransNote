# 모바일 온디바이스 양자화 — 진행 상황

> 공식 TransNote 프로젝트(웹, `trans-note.vercel.app`) 제출과는 별개로 진행하는 개인 후속
> 작업. 포트폴리오 작성 염두 — 과정을 여기 계속 기록한다.
>
> **범위 확정(2026-08-24)**: 모바일 앱(UI)은 만들지 않는다. 목표는 "TFLite 모델이 실제
> 모바일/Edge 런타임에서 얼마나 완벽하게 동작하는지"를 증명하는 벤치마크 코드+리포트
> 완성 — 아래 "범위 확정" 섹션의 ①②③ 참고.

## 목표

- 속도: PyTorch(서버) 대비 TFLite(CPU) 추론 레이턴시를 정량 비교 (ms 단위)
- 정확도: FP32 기준(newage 장르, held-out 10곡 평균 94.2% — 아래 "베이스라인 정정" 참고)
  대비 TFLite FP16의 손실을 정량 측정
- 크기/메모리: FP32(185MB) vs FP16(~93MB), Peak Memory 측정

## 베이스라인 정정 (중요)

`realImage/exactPicture/` 131개로 잰 82.1%는 **held-out 정확도로 쓰면 안 됨** — newage21~30
10곡을 제외한 나머지는 전부 학습에 이미 쓰인 데이터였음(사용자 확인). 진짜 held-out은
newage21~30(10곡)뿐 — **평균 94.2%, 중앙값 96.7%**. 앞으로 양자화 전/후 비교는 이 10곡
기준으로 한다. `test/data`(합성 20개)도 학습에 쓰였을 가능성 있어 불확실 — 필요하면
Model_TransNote의 TRAINING_REPORT.md에서 실제 검증셋을 확인할 것.

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

### B단계 — 136개 held-out 정확도 측정 (⚠️ 아래 82.1%는 정정됨, 파일 상단 "베이스라인 정정" 참고)

`train/eval_exactpicture.py`로 `realImage/exactPicture/` 전체(131개 유효 샘플, 장르
다양 — Chopin/Czerny/뉴에이지/사계 등, 5개는 파일 누락으로 건너뜀: newage16~20) 실측:

- **음표 레벨 평균 Acc 82.1%**, 전체 TER 평균 Acc 80.3% (617초, 4.7초/장 — CPU 속도
  기준점과도 일치)
- test/data 20개 합성 표본(81.7%)과 거의 같은 수치로 나와 서로 교차검증됨 — 표본이 훨씬
  크고(131개) 대표성 있는(실사, 장르 다양) 이 숫자를 앞으로 공식 FP32 베이스라인으로 쓴다.

**중요한 발견 — 평균만 보면 놓치는 분포**: 중앙값은 95.8%로 평균(82.1%)보다 훨씬 높음.
분포가 이분화돼 있음 — 45곡이 정확히 100%, 80~100% 구간까지 합치면 95곡(73%)이 잘 되는데,
**16곡(12%)은 50% 미만, 일부는 0% 이하(overshoot로 오류율이 100% 넘음)**로 완전히
실패함. 즉 "대체로 조금씩 틀리는" 게 아니라 "거의 맞거나 완전히 무너지거나" 패턴 —
최악 사례(`sonatineHa_8_13`, note_err 110%)는 디코딩 과잉생성(`LONG_DECODE_THRESHOLD`
안전장치가 있어도 가끔 뚫림) 계열로 보임. 포트폴리오 "실패 사례" 후보로 좋음 — 재현
가능(`python train/eval_exactpicture.py`로 언제든 같은 샘플 재확인 가능).

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
- [x] ~~더 크고 대표성 있는 held-out 셋 확보 + 정확도 재측정~~ — `realImage/exactPicture/`
      131개로 측정했으나 대부분 학습에 쓰인 데이터였음이 나중에 밝혀짐(위 "베이스라인
      정정" 참고) — 진짜 공식 베이스라인은 newage21~30(94.2%)
- [x] ~~목표 스택 결정~~ — Android(TFLite) 우선
- [x] ~~디코더 KV캐시 결정~~ — **결론이 바뀜**: 처음엔 "우선 O(T²)"로 정했으나, export한
      디코더가 실제로는 2번째 토큰부터 크래시하는 걸 발견(아래) → self-attention도 진짜로
      캐싱하는 방식을 구현·검증 완료(O(T³)→O(T²)). production 적용 여부만 남음(아래
      "미결정" 참고)
- [x] ~~export_tflite.py 전체 파이프라인(인코더+디코더) 끝까지 실행 테스트~~ — export 자체는
      성공(인코더 54.5MB+디코더 131MB FP32)했지만, **실제로 돌려보니 디코더가 2번째
      디코딩 스텝부터 reshape 에러로 깨짐**(`past_ids`를 매 스텝 growing 텐서로 리사이즈하는
      방식이 TFLite에서 안 됨) + 인코더 1회 추론에 18.7초(리사이즈마다 XNNPACK 델리게이트
      재구성 비용으로 추정) — "export 성공"과 "실제로 동작함"은 별개였음. `_INT8` 파일명은
      지금 FP32라 오해 소지 있음, 나중에 실제 INT8 붙일 때 네이밍 정리 필요
- [x] ~~self-attention KV캐시 구현·검증~~ — `model.py`에
      `precompute_self_attn_cache()`/`decode_step_kv_cached()` 추가(고정 크기 캐시 버퍼,
      매 스텝 새 토큰 1개만 처리 — O(T³)→O(T²), 고정 shape이라 TFLite에도 적합). **발견**:
      RunPod GPU 서빙(`forward_cached`)도 self-attention은 캐싱 안 하고 있어서 이 개선은
      모바일뿐 아니라 현재 서빙 속도에도 적용 가능. `inference.py`에 `greedy_decode_kv()`로
      기존 `greedy_decode`와 별도 추가, newage21~30 10곡 검증: **토큰 시퀀스 10/10 완전
      일치**(정확성 확인), **속도 평균 2.0배 개선**(CPU, 1.4~3.8배). 커밋 `0707781`
- [x] ~~production(run_image/beam_decode) 적용~~ — 처음엔 `time_correct` 없이 그대로
      적용했다가 production 검증(analyze_sample)에서 정확도가 94.2%→90.9%로 떨어지는
      걸 실측(newage25, 6/8박자, 97.8%→65.8% 급락 — InlineTimeCorrector가 원래 다루던
      3/4·6/8 혼동 케이스). **하이브리드로 재설계**: 첫 마디는 캐시없는 방식(`decode_
      step_cached`)으로 돌려 InlineTimeCorrector를 그대로 받고, 첫 마디가 끝나면
      `forward_bulk_capture()`로 그 구간 전체를 self-attention 캐시에 한 번에 채운 뒤,
      나머지는 `decode_step_kv_cached`로 빠르게 이어간다. **재검증(production 경로
      그대로): 정확도 94.2%(완전 복원) + 속도 2.95s/장(기존 4.7~5.5s/장 대비 약
      1.6~1.8배)** — 정확도 손실 없이 속도만 얻음. 커밋 `61d5586`. `train/*.py`라 다음
      Docker 빌드부터 실서비스에 반영됨(container image 태그 교체는 사용자가 직접 진행)

### O(T²) 이하로 더 줄일 수 있는지 조사 — 결론: 재학습 없이는 안 됨

O(T³)→O(T²)까지는 됐는데, 이것보다 더(O(T log T)급) 줄일 수 있는지 조사·실험함.

**이론적 검토**: Reformer(LSH attention), Linear/Performer attention 같은 O(T log T)·O(T)급
방법들은 전부 **attention을 계산하는 수식 자체를 바꾸는 방법**이라, 지금 모델의 Q/K/V
projection 가중치가 표준 softmax dot-product attention에 맞춰 학습돼 있어서 추론 시점에
수식만 바꿔치기하면 정확도가 깨진다 — 전부 재학습(또는 최소 파인튜닝)이 전제. FlashAttention은
복잡도 자체를 안 바꿈(여전히 O(T²), GPU 메모리 접근 패턴만 최적화하는 상수배 개선 — 이미
`F.scaled_dot_product_attention`이 백엔드에서 이런 최적화를 알아서 고름).

**재학습 없이 시도 가능했던 유일한 후보 — 슬라이딩 윈도우 attention**: attention을 전체 과거가
아니라 최근 W개 토큰으로만 제한하면(자기회귀 특성상 O(T·W), W가 상수면 사실상 O(T)) 재학습
없이도 될 수 있다는 가설로 실험(newage21~30, W=32/64/무제한):

| window | 음표 Acc |
|---|---|
| 32 | 63.0% |
| 64 | 78.8% |
| 무제한 | 85.5%(이 실험 자체의 베이스라인 — InlineTimeCorrector 없이 잰 수치라 production 공식
  94.2%와 직접 비교는 안 되지만, 아래 추세는 이 오프셋과 무관하게 유효) |

**결과: 가설 기각.** 윈도우를 줄일수록 정확도가 뚜렷하게 떨어짐 — W=64(전체 시퀀스 길이의
절반 정도)만 돼도 이미 7%p 가까이 손해. "음악 표기는 지역적 구조라 근거리만 봐도 될 것"이라는
가설이 틀렸음 — 모델이 실제로 꽤 먼 과거(첫 마디의 박자표/조성 등)를 계속 참조하며 디코딩하는
것으로 보임. 폐기.

**결론**: 재학습 없이 도달 가능한 사실상의 한계는 지금의 O(T²)(하이브리드 KV캐시)다. 더
줄이려면 Model_TransNote 쪽에서 sparse/linear attention 등을 적용한 재학습이 필요 — 별도
트랙, 지금 범위 밖.

- [x] ~~O(T²)보다 더 줄일 수 있는지 조사~~ — 이론 검토(재학습 없인 불가) + 슬라이딩 윈도우
      실험(가설 기각) 완료. 지금 O(T²)가 재학습 없이 도달 가능한 한계로 결론

## 범위 확정 (2026-08-24) — 앱은 안 만든다, 벤치마크로 마무리

포트폴리오 방향 정리: **모바일 앱(UI)은 만들지 않는다.** 대신 "TFLite 모델이 실제
모바일/Edge 런타임에서 얼마나 완벽하게 동작하는지"를 증명하는 벤치마크 코드+리포트로
마무리한다. 아래 세 가지만 완성하면 끝.

### ① TFLite 디코더 런타임 구동 성공 (동적 shape 문제 해결) — ✅ 완료 (2026-08-24)

- [x] ~~`export_tflite.py`의 디코더 export를 고정 크기 self-attention KV캐시 기반으로
      재설계~~ — `_DecoderStepWrapperKV` 추가: `token_id/pos/memory/k_cache/v_cache` 전부
      고정 shape(캐시는 `[num_layers,1,H,cache_len,Dh]` 고정 버퍼). 슬라이싱 대신 고정
      버퍼 전체+마스킹, 캐시 쓰기도 슬라이스 대입 대신 `torch.where` 마스킹 — 둘 다
      shape이 안 바뀌는 순수 elementwise 연산이라 TFLite 변환이 안전함. cross-attention은
      캐싱 없이 매 스텝 memory에서 재계산(단순화, 상수 비용). onnx2tf가 5차원 캐시 텐서
      축 순서를 자동으로 바꾸는 문제도 발견해 `keep_layout_input_names`로 방지. 커밋 `c2372b5`
- [x] ~~Python 인터프리터로 실제 이미지 → 토큰 출력까지 되는 추론 스크립트 작성~~ —
      `train/tflite_infer.py`(영구 보관, ① 최종 산출물). PyTorch 모델 로드 전혀 없이
      `encoder_INT8.tflite`+`decoder_INT8.tflite` 두 파일만으로 동작
- [x] ~~reshape 크래시 재현 안 되는지 확인~~ — 처음엔 newage24 1곡만 확인(111토큰 끝까지
      크래시 없이 생성, PyTorch 원본 출력과 토큰 완전 일치)했다가, **newage21~30 10곡
      전체로 재확인** — **10/10 크래시 없음.** 리사이즈는 최초 1회만 필요(이전 growing
      방식은 매 스텝 필요해서 18.7초/스텝이었음).
- [x] ~~production과 공정한 정확도 비교~~ — 처음엔 `tflite_infer.py`에 EOS_BOOST/후처리가
      빠져 있어 85.5%로 낮게 나옴 → 추가 후 재검증 **89.5%**. 남은 차이(PyTorch 94.2%
      대비)는 `InlineTimeCorrector`(마디 중간 박자표 재추정) 미지원 때문으로 특정 —
      newage25/26(6/8박자, 정확히 이 기능이 다루는 케이스)에 집중돼 있음. **InlineTimeCorrector
      없는 PyTorch 버전(90.9%)과 비교하면 1.4%p 차이**로 훨씬 가까움 → TFLite 변환 자체의
      정확도 손실은 작고, 아직 안 옮긴 기능 하나가 차이의 대부분. 커밋 `0d08ec2`
      (InlineTimeCorrector를 TFLite 쪽에도 이식할지는 별도 판단 — 고정 캐시 설계와
      구조적으로 안 맞아서 PyTorch만큼 간단하지 않음, ②/③ 이후 필요시 재검토)

### InlineTimeCorrector를 TFLite에도 이식 — "일괄 캐시 채우기" 그래프로 성공 (2026-08-24)

"동적 캐시로 하면 어떨까"라는 질문에서 출발 — 검토 결과 두 가지 확인:
1. **동적(growing) 캐시로 되돌리면 안 됨**: 정확히 처음 겪었던 크래시(2번째 스텝 reshape
   에러)와 스텝당 리사이즈 오버헤드(18.7초)가 재현되는 방향.
2. **"캐시 한 슬롯만 패치"는 애초에 수학적으로 불가능**: attention이 매 레이어 모든 앞선
   위치를 섞으므로, 한 위치(예: 박자표 토큰)의 토큰을 바꾸면 그 뒤 모든 위치의 hidden
   state가 이론적으로 다 달라져야 함 — 부분 패치 불가, "교정 후 그 뒤 전체를 다시 계산"만
   유효한 접근(PyTorch 하이브리드가 이미 이렇게 했던 이유).

**해법**: `decoder_bulk_INT8.tflite` 추가 — 고정 길이 청크(chunk_len=40)를 causal
masking으로 한 번에 처리해서 self-attention 캐시를 일괄 재구성하는 그래프
(`_BulkCaptureWrapperKV`, PyTorch `forward_bulk_capture()`와 동일 목적). 여기도 모든
shape이 고정이라 크래시 위험 없음. `tflite_infer.py`에 `decode_hybrid()`로 PyTorch와
같은 3단계(첫 마디 순차 디코딩+교정 → 일괄 재구성 → 빠른 경로) 구현.

**결과(newage21~30 10곡)**: 10/10 크래시 없음, **평균 Acc 93.0%**(이전 89.5% → 개선,
PyTorch 하이브리드 94.2%와 1.2%p 차이로 좁혀짐). **newage25(6/8박자, InlineTimeCorrector가
가장 크게 도움되던 곡)가 63.0%→97.8%로 급등** — PyTorch(98.7%)와 거의 일치, 이식 성공
확인. 커밋 `d9c3e6d`.

### ② 정량적 성능 프로파일링 (가장 중요 — README에 벤치마크 리포트로 작성)

- [ ] **Inference Latency**: PyTorch(서버 GPU / 개발 PC CPU) vs TFLite(단일 스레드 /
      멀티 스레드) 비교표. "Xms → Yms" 형태로.
- [ ] **Model Size**: FP32(185MB, 이미 앎) vs FP16 양자화(약 93MB — export 시 이미
      부산물로 생성됨, `_convert_tflite()`가 float32 대신 float16 변형을 고르도록만
      바꾸면 바로 나옴) 비교.
- [ ] **Memory Footprint**: 추론 중 Peak Memory 측정. 고정 크기 캐시로 바꾸면서 메모리
      사용량이 안정화된 정도도 수치화(growing 텐서 방식 대비).
- [ ] **정확도 유지율**: PyTorch 원본(94.2%, newage21~30 held-out) 대비 TFLite FP16의
      정확도 손실 — 거의 없음 또는 "-X%p" 형태로 명시.
- [ ] 위 네 가지를 표로 정리해서 README.md에 "벤치마크 리포트" 섹션으로 추가

### ③ 통제된 테스트 환경 구축

- [ ] 로컬 PC CPU 환경에서 위 벤치마크 재현 가능하게 스크립트화(이미 이 세션에서 CPU
      실측은 여러 번 했음 — 정식 스크립트로 정리)
- [ ] 가능하면 라즈베리파이 또는 안드로이드+Termux(순수 CLI)에서 같은 스크립트로 재현
      — "실제 Edge 환경 제약 속에서 테스트했다"는 근거. 갤럭시탭 S9 FE는 연결 보류 중이라,
      Termux로 CLI만 쓰면 지난번 논의했던 "USB 디버깅+adb" 없이도 가능(앱 설치만 필요)

### 이번 범위에서 제외(보류)

- INT8 양자화(`--quantize_decoder`) — FP16까지만 하고 INT8은 이번 범위 밖(디코더 이중
  입력 캘리브레이션 문제가 남아있고, ①②③ 완성이 우선)
- catastrophic 실패 곡 원인 분석 — 별도 트랙, 우선순위 낮음

## 실험 기록 (진행되면 추가)

| 날짜 | 정밀도 방식 | 크기 | held-out 정확도 (음표 기준) | 속도 | 메모 |
|---|---|---|---|---|---|
| 2026-08-24 | FP32 (r15, 20개 합성) | 185MB | 81.7% (Treble 85.9%/Bass 88.0%) | 개발 PC CPU ~4.9초/장(+로드 ~6.1초, 1회) | 표본 작음, 참고용. 학습 포함 여부 불확실 |
| 2026-08-24 | FP32 (r15, exactPicture 131개) | 185MB | 82.1%(TER기준 80.3%) | CPU 4.7초/장 | ⚠️ held-out 아님(대부분 학습에 쓰인 데이터), 베이스라인으로 쓰지 말 것 |
| 2026-08-24 | FP32 (r15, **공식 held-out 베이스라인**) | 185MB | **94.2%**(newage21~30 10곡, 중앙값 96.7%) | CPU, `greedy_decode`(캐시 없는 self-attn) 기준 | 진짜 학습에 안 쓰인 유일한 셋. n=10로 작음 |
| 2026-08-24 | FP32 (r15, self-attn KV캐시, time_correct 없이) | 185MB | 90.9% — ⚠️ 하락 확인 | CPU | InlineTimeCorrector 없이 순수 캐시만 적용한 시행착오. newage25가 65.8%로 급락(6/8박자) — production 반영 안 함 |
| 2026-08-24 | FP32 (r15, **하이브리드: KV캐시+InlineTimeCorrector**, production) | 185MB | **94.2%(원본과 동일)** | CPU 2.95s/장 (기존 4.7~5.5s/장 대비 1.6~1.8배) | 정확도 손실 없이 속도만 개선 — **현재 production 코드**, 커밋 `61d5586` |

## 포트폴리오용 캡처 포인트 (계획)

- [ ] 크기 비교 (FP32 vs 양자화 후)
- [ ] 정확도 표/그래프
- [ ] 실기 데모 영상/캡처
- [ ] 실패 사례(디코딩 붕괴 등) — 문제 발견→원인 분석→해결 서사용
