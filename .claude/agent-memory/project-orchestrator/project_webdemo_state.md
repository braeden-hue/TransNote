---
name: project-webdemo-state
description: online_webpage/ 현재 구현 상태 및 기능 갭 분석 (2026-05-13 기준)
metadata:
  type: project
---

online_webpage/는 바닐라 JS + ES Module 구조의 인터랙티브 웹 앱으로, 핵심 기능이 실질적으로 완성된 상태다. project.md의 "웹 ❌ 미구현" 표시는 stale하다.

**Why:** 2026-05-13 파일 검토 결과 확인. landing.html(마케팅) + index.html(앱) 분리 구조.

**How to apply:** 웹 데모 관련 작업 시 구현된 기능을 재구현하지 말고 기존 모듈 확장으로 접근할 것.

### 완료된 기능
- SVG 커스텀 악보 렌더러 (notation.js): 3존(옥타브) + 박자색 + 예상음 애니메이션
- 88건반 인터랙티브 피아노 (piano.js): 포인터 + 키보드 + 옥타브 스크롤 + 골든 건반 하이라이트
- Web Audio API 합성 엔진 (audio.js): ADSR + 3배음, 시퀀스 재생, cancel 지원
- 연주 가이드: 정답/오답 시각 피드백, 카운트다운 바, 자동재생/스텝/BPM 슬라이더
- 그랜드 스태프(2단) 렌더링 완료
- 샘플 4종 (반짝반짝, 학교종, 나비야, 기쁨의 송가 + 2단 편곡)
- localStorage 저장/삭제 (storage.js)
- 파일 업로드 UI (드래그앤드롭) — OMR 추론 없이 랜덤 샘플 반환으로 시연

### 미구현 기능 (우선순위 순)
1. 화음 연주 체험 — samples.js chord 필드 확장 + notation.js 렌더링 + piano.js 다중 하이라이트
2. URL 기반 악보 공유 링크 (서버 불필요, base64 직렬화)
3. MIDI 키보드 입력 (Web MIDI API)
4. 연주 점수 표시
5. Firebase/Supabase 악보 공유 커뮤니티 (Phase 9)
6. FastAPI OMR 추론 연동 (모델 학습 완료 후)

### 노트 데이터 형식
현재: `{ pitch: 'C4', duration: 1, beat: 1 }`
화음 확장 방향: `pitch: string | string[]` — 하위 호환 유지 가능

[[project_goals]]
