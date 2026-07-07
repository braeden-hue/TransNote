# RunPod 학습 인계 상태 (2026-07-07 기준)

## 파드 접속 정보
- Host: `213.173.103.141`
- Port: `43347`
- User: `root`
- SSH 키: `~/.ssh/runpod_auto` (로컬 노트북에만 있음 — 클라우드 세션엔 별도로 전달 필요, 아래 "SSH 키 전달" 참고)
- 접속 예시: `ssh -i ~/.ssh/runpod_auto -p 43347 root@213.173.103.141`

## 현재 학습 상태
- **Phase1 (SegNet)**: epoch 21/60 진행 중, val_acc 97.9%
- 조기 전환 조건: `epoch >= 30` (val_acc 조건은 제거함 — 97.5%대 정체로 무의미 판단)
- Phase1 로그: `/workspace/models/round1/segnet_log.csv`
- 전체 마스터 로그: `/workspace/round1_master.log`

## 실행 중인 백그라운드 스크립트 (파드 위)
- `/workspace/phase1_early_transition_v3.sh` — epoch≥30 되면 Phase1 종료 → Phase2(60ep) → Phase3(30ep) 자동 실행
- `/workspace/phase2_snapshot.sh` — Phase2 시작되면 10에폭마다 `seq2seq_epoch{N}.pt` 스냅샷 저장 (`/workspace/models/round1/checkpoints_by_epoch/`)

## 로컬(이 세션)에서 도는 것
- 스냅샷 자동 다운로드 모니터 → `musicscore_flutter/round3train/checkpoints_by_epoch/`로 받아옴 (노트북이 꺼지면 이건 안 돌아감)

## 오늘 수정한 코드 (pod + `musicscore_flutter/round3train/` 양쪽 동기화 완료)
1. **`dataset.py`**: alpha 채널 합성 버그 수정(생성된 PNG가 RGB=0 고정+alpha에만 실제 내용 — 안 고치면 완전히 검은 이미지로 학습됨), `preprocess()`/weak-label 결과 디스크 캐싱(uint8, 이미지당 1회 계산)
2. **`model.py`**: cross-attention K,V 캐싱 (`precompute_memory_kv`, `decode_step_cached`) — greedy/beam decode 2.94배 가속, 원본과 bit-identical 검증 완료
3. **`train.py`**: beam search(`beam_decode`) 추가, Phase3 검증에 연결
4. **`inference.py`**: 동일 캐싱 적용, `_extract_treble_bass` docstring 오류 수정

## 다음 세션에서 이어받을 때 할 일
1. 위 SSH 접속 정보로 파드 상태 확인 (`tail -n 20 /workspace/round1_master.log`)
2. Phase2/Phase3 전환 여부 확인 후 계속 모니터링
3. 필요시 `/workspace/phase2_snapshot.sh`가 만든 체크포인트로 로컬 테스트
