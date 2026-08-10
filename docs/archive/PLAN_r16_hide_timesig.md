# Round 16 계획: 박자표 미노출(mid-piece) 학습 데이터 보강

**상태**: 2026-08-05에 실제로 학습·검증까지 진행됨 — **결과는 기각(r15 대비 하락, 채택 안 함)**.
경과/수치는 [`HANDOFF_STATUS.md`](HANDOFF_STATUS.md)와
[`TRAINING_REPORT.md`](../../train/docs/TRAINING_REPORT.md)의 "r15 이후 시도" 절 참고. 아래는 계획 수립 시점
(2026-08-04)의 원인 진단·구현 기록으로, 문제 진단 자체는 여전히 유효하나 이 계획대로 학습해도
실사 검증 정확도는 개선되지 않았다.

**원 상태 메모(2026-08-04 작성 당시)**: 원인 진단 + 코드 구현 완료, 학습(pod)은 아직 미실행

## 문제

`diag_new6_analysis.py`(신규 6곡, r15_cropfix_coordconv 기준) 극초반 이탈 캐스케이드 분석에서
첫 이탈 토큰 종류 1위가 **time(36.8%)** — 박자표 관련 오류가 이후 전체 시퀀스 붕괴의 최대
유발원이었음. `InlineTimeCorrector`(마디 완료 후 beat-sum 다수결로 time 토큰 사후 교정)로도
완전히 해소되지 않음.

## 원인

실사 촬영 사진은 곡 중간부터 촬영되는 경우가 흔해 **박자표 기호가 이미지에 아예 안 보이는
경우가 많음**(라벨의 time-* 토큰 자체는 항상 정답으로 존재 — 원곡 전체를 알고 라벨링했으므로).
반면 **지금까지의 모든 합성 학습 데이터는 100% 첫 마디에 박자표를 렌더링**해왔음
(`generate_scores.py`가 항상 `meter.TimeSignature(...)`를 0번째 마디에 삽입, `dataset.py`의
`_split_grand_staff_interleaved`도 매 시스템 샘플마다 헤더(clef/key/time)를 통째로 복사하지만
애초에 시스템 줄바꿈 자체가 `wide_page` 스타일로 항상 억제되도록 설계돼 있어 이 분기가 실제로
쓰인 적이 없음). 합성(replay 포함, 실사 대비 ~5배 물량)이 "박자표는 항상 보인다"만 압도적으로
가르쳐온 셈.

## 검증 (완료, 2026-08-04)

같은 시드로 두 배치를 생성(음악 내용 완전 동일, `--hide-timesig-prob 0` vs `1.0`만 다름),
L4 노이즈 강제 적용 후 로컬 CPU로 r15_cropfix_coordconv 추론 비교:

| | 전체 정확도 | time- 토큰(헤더 3번째) 정확도 |
|---|---|---|
| 대조군(박자표 보임) | 65.6% | **20/20 (100%)** |
| 실험군(박자표 숨김) | 68.8% | **12/20 (60%)** |

동일 콘텐츠·동일 노이즈에서 오직 기호 가시성만 다른데 40pp 급락 — 원인이 확인됨.

## 구현 완료 (코드)

`round3train/generate_scores.py`:
- `HIDE_TIMESIG_PROB` 전역 추가(기본 0.0, 기존 커리큘럼 영향 없음).
- `_build_part()` / `_build_accompaniment_part()`에 `hide_timesig` 파라미터 추가 --
  `True`면 `TimeSignature` 객체에 `.style.hideObjectOnPrint = True`를 설정해 MuseScore
  렌더링에서만 숨기고(clef/key는 그대로 노출), 토큰 라벨은 정상적으로 정답 time-*를 포함
  (music21 스트림 상 논리적으로는 존재하므로 라벨 생성 로직에 영향 없음).
- `build_score_r3()`(대보표): 마디/성부 결정 시 `hide_ts = random.random() < HIDE_TIMESIG_PROB`
  한 번 결정 후 treble/bass 양쪽 빌더 호출에 동일하게 전달(같은 시스템이므로 한쪽만 숨기면 안 됨).
- `build_score_single_staff()`: 동일 패턴 적용(atom_only는 제외 -- 항상 4/4 고정이라 의미 없음).
- CLI: `--hide-timesig-prob <float>` 추가.
- 검증용 진단 스크립트: `diag_timesig_synth_l4.py`(로컬 CPU, `diag_chord_synth_l4.py`의
  decode/정렬 인프라 재사용).

## 다음 단계 (pod 재개 시 실행)

`curriculum_r16_hide_timesig.sh` 형태로, r15와 동일 패턴(r12_all120_realphotos 실사 메인
데이터 유지, `r12_replay_merged` replay 재사용) + **신규 합성 데이터 생성 시
`--hide-timesig-prob 0.3~0.4` 추가**해서 replay/보강 풀에 섞어 넣는 방식을 검토.

- RESUME_CKPT: `/workspace/models/r15_cropfix_coordconv/seq2seq_best.pt`
- "한 번에 축 하나만" 원칙 유지 -- 이번엔 박자표 가시성 축만 격리.
- 비율(0.3~0.4)은 임의값 -- 실사 데이터에서 실제 "박자표 안 보이는 사진" 비율을 먼저
  샘플링해 추정하면 더 근거 있는 값을 잡을 수 있음(아직 안 함).
- 학습 후 재검증: 신규 6곡 전체 정확도(r15 84.1% 기준)뿐 아니라, `diag_timesig_synth_l4.py`류
  진단으로 "박자표 숨김 시 time- 토큰 정확도"가 60%에서 얼마나 올라가는지 직접 재측정할 것
  (이게 이번 변경의 직접 타깃 지표).
