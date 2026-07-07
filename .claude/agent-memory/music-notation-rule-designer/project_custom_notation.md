---
name: project-custom-notation-system
description: 현재 커스텀 악보 시스템의 구조, vocab 매핑 현황, 구현 상태, 그리고 확장 제안 방향
metadata:
  type: project
---

## 현재 커스텀 악보 시스템 (online_webpage 기준, 2026-05-30)

### 기본 4규칙 (실제 구현 기준)
| 시각 요소 | 인코딩 내용 |
|---|---|
| 셀 세로 위치 (3존) | 음높이 옥타브: Treble=6옥+(0)/5옥(1)/4옥(2), Bass=3옥+(0)/2옥(1)/1옥↓(2) |
| 셀 가로 너비 | 음 길이 (duration): UNIT_W=80px 기준, 4분=80, 2분=160, 8분=40 |
| 셀 테두리 색(화음) / 텍스트 색(단음) | 박자 위치: 1박=#0076CE / 2박=#5BB8F5 / 3박=#A8D5F5 / 4박=#D0EBFA |
| 셀 내부 텍스트 | 음이름: 흰건반=C~B, 검은건반=1~5 (C#=1, D#=2, F#=3, G#=4, A#=5) |

### 핵심 데이터 구조
```js
{ pitch: 'C4', duration: 1, beat: 1, chordNotes: ['E4', 'G4'] }  // chordNotes는 화음 시
```
- pitch: 음명+옥타브 (예: C4, F#3)
- duration: 4분음표=1, 2분=2, 8분=0.5
- beat: 1~4 (박자 위치)
- chordNotes: 같은 박자의 추가 음 (화음)

### 렌더링 파일
- `online_webpage/js/notation.js` — renderNotation(), renderGrandStaff(), renderMiniNotation()
- `online_webpage/js/samples.js` — BEAT_COLORS, pitchToZone(), formatNoteName()
- 레이아웃: UNIT_W=80, CELL_H=46, ZONE_H=56, MARGIN_L=68, MARGIN_Y=8, INDICATOR_H=18

### 화음 vs 단음 렌더링 차이
- 화음(chordNotes 존재): 셀 테두리 박스 + 음이름 세로 스택
- 단음: 박스 없음, 텍스트만 (hit-box는 투명 rect로 별도 처리)

### vocab 매핑 구현 상태 (2026-05-30 기준, vocab 총 1,012개)
| 처리 방식 | 범주 |
|---|---|
| 완전 구현 | 음표(음이름+존+폭), 화음(chordNotes), 마디 기준점(●) |
| 흡수(표기 생략) | 조표 13개(key-*), 8va/8vb 4개 — 절대 pitch로 대체 |
| 미구현(규칙 정의됨) | 쉼표, 세로줄 종류, 셈여림, 아티큘레이션, 꾸밈음, 이음줄, 셋잇단음표, 페르마타 |
| ML 내부 전용 | PAD/SOS/EOS/UNK |

### Grand Staff 구조적 제약 (2026-05-30 확인)
- renderGrandStaff(): 두 보표를 flex-column으로 각각 독립 SVG + 독립 스크롤로 렌더링
- 보표 간 연결선(동시 연주 표시) 추가 시 DOM 구조 문제 발생
- 전제 조건: 두 보표 스크롤 동기화 먼저 구현 필요
- 권장 구현 방식: 오버레이 SVG (position:absolute로 두 보표 위에 덮어쓰기)

## 확장 제안 방향 (경험 있는 음악가를 위한 새로운 차원)

**Why:** 현재 시스템은 음높이/길이/박자 위치만 인코딩. 표준 악보도 못 하는 정보 레이어를 추가하면 숙련 음악가에게 차별화된 novelty 제공.

**How to apply:** 아래 3가지 제안을 note 객체 확장 필드로 구현 가능. 기존 규칙과 시각 충돌 없음.

### 제안 1: CUSTOM_TENSION_FILL (긴장-이완 그라디언트)
- 셀 배경의 radialGradient 밝기로 조성 내 음의 당김력(tonal tension) 표시
- Level 0 안정(C/E/G): fill #16162e, Level 1 중간(D/A): amber glow 25%, Level 2 고긴장(B/F): amber glow 45%
- note 객체 확장 필드: `tension_level: 0|1|2`
- SVG 구현: `<radialGradient>` per-note

### 제안 2: CUSTOM_PHRASE_ARC (프레이징 아치)
- 셀 상단 MARGIN_Y(10px) 공간에 SVG Q-curve arc로 프레이즈 경계 표시
- 아치 색 rgba(255,255,255,0.18), stroke 1.5px fill none
- 아치 정점 X위치 = 프레이즈 내 에너지 클라이맥스 음 위치
- note 객체 확장 필드: `phrase_id: number` (같은 ID끼리 하나의 프레이즈)
- SVG 구현: `<path>` M...Q...

### 제안 3: CUSTOM_HARMONIC_BASELINE (화성 기능 베이스라인)
- 셀 하단 가장자리에 2.5px 컬러 line으로 화성 기능 표시
- 토닉=#5BC0EB, 도미넌트=#FF6B35, 서브도미넌트=#7BC67E
- note 객체 확장 필드: `harmonic_function: 'tonic'|'dominant'|'subdominant'`
- SVG 구현: `<line x1=x y1=y+cellH-1 x2=x+w y2=y+cellH-1>`
