# Flutter UI 진행 현황 (2026-07-21)

`appMake.md` 스펙 기반으로 앱 UI 전체를 Glory Music / QuickScan UI 킷 스타일로 재구성한 작업 기록.

## 커밋 상태

- **푸시 완료**: `0e6c0e3` — 스플래시/홈메뉴/튜토리얼 1차/악보 병합/스캔 화면 전체 구조
- **로컬 미커밋**: 튜토리얼 2차 개편 (규칙 0·화음·테스트 페이지 추가, 존 색상·셀 테두리 버그 수정, 미니 피아노 다중 옥타브 지원) — `lib/screens/tutorial_screen.dart`, `lib/widgets/mini_piano_widget.dart`, `lib/widgets/notation_widget.dart`

## 앱 흐름

```
SplashScreen (로고 탭 → 원형 리빌 전환)
  └─ HomeMenuScreen (3버튼 허브)
       ├─ TutorialScreen   (8페이지)
       ├─ ScoreScreen      (감상하기 / 연주하기 토글)
       └─ CollectionScreen (다크 테마, 촬영/스캔)
```

기존 4탭(`AppShell`) 구조는 제거됨.

## 화면별 현황

### SplashScreen (`lib/screens/splash_screen.dart`)
- Glory Music "Now Playing" 레이아웃 재사용 — 중앙 사진 자리에 로고(원형 + 음표 아이콘), 곡명 자리에 "악보의 대중화 프로젝트", 아티스트 자리에 "유규태, 조준성"
- 하단 재생 컨트롤·네비게이션 바는 제거
- 로고 탭 → `PageRouteBuilder` + `ClipPath` 기반 원형 리빌(circular reveal) 전환으로 `HomeMenuScreen` 진입

### HomeMenuScreen (`lib/screens/home_menu_screen.dart`)
- 튜토리얼 / 예시 악보 체험 / 악보 모음집 3개 카드 버튼

### TutorialScreen (`lib/screens/tutorial_screen.dart`) — 8페이지, 스크롤 없이 한 화면에 맞도록 컴팩트하게 구성
1. **인트로**: 전통 오선보 vs 커스텀 악보를 **같은 한 마디**로 나란히 비교 (`_StaffPreview`가 실제 음높이·길이·임시표를 오선에 정확히 그림 — 이전엔 장식용 임의 위치라 커스텀 악보와 안 맞았던 버그 수정)
2. **규칙 0 (신설)**: 흰 건반 7개+검은 건반 5개 = 12음 전체를 음이름+계이름으로 라벨링한 전용 키보드 화면
3. **규칙 1**: 세로 위치=음높이. 예시가 6옥+/5옥/4옥 이하 3구역을 모두 지나가도록 수정(기존 예시는 사실상 2구역만 보여주던 버그), 범례 색과 `NotationWidget`의 존 배경색을 실제로 일치시킴(파랑끼/중립/브라운끼)
4. **규칙 2+3 (통합)**: 셀 너비=음길이 + 테두리 색=박자 위치. `NotationWidget`이 단일 음표에 테두리 박스를 그리지 않던 버그 수정(텍스트만 떠 있었음), 예시를 서로 다른 음 3개로 교체
5. **규칙 4 (신설)**: 화음(두 음 동시 연주) — 피아노에서 두 건반 동시 하이라이트로 표시
6-8. **테스트 3장면 (신설)**: 힌트 없이 커스텀 악보를 보고 건반을 맞히는 간단한 퀴즈, 정답/오답 피드백

- 각 규칙 페이지마다 `_NotePracticeDock`을 내장해, **그 페이지의 실제 예시 음**을 순서대로 실시간 연주 연습 (이전엔 예시와 무관한 고정 도레미파솔라시도 시퀀스였음)

### ScoreScreen (`lib/screens/score_screen.dart`)
- 기존 악보(감상) + 연주(점수 게임) 두 탭을 **감상하기/연주하기 토글**로 통합, `practice_screen.dart` 삭제

### CollectionScreen (`lib/screens/collection_screen.dart`)
- QuickScan UI 킷 참고한 다크 테마(이 화면만 다크, 나머지 앱은 라이트 유지)
- 상단 스캔 목록 + 하단 카메라(실제 촬영)/갤러리 가져오기 버튼, 스캔 라인 애니메이션
- 버튼 모양을 킷 원본과 맞춤: 가져오기 버튼은 원형이 아닌 스퀴클 배지+라벨, 목록 항목 화살표는 테두리 원형 버튼, 상단 뒤로가기는 프로스티드 원형 배지

### 공용 위젯
- `lib/theme/glory_theme.dart`: 앱 전역 라이트 팔레트 (배경/잉크/포인트 컬러/버튼 스타일)
- `lib/widgets/mini_piano_widget.dart`: 화면 폭에 맞춰 스크롤 없이 표시되는 컴팩트 피아노. **임의 음역대(여러 옥타브) 지원**, **다중 건반 동시 하이라이트 지원**(화음용), 학습용 상세 라벨 모드(`detailed`) 지원
- `lib/widgets/notation_widget.dart`: 커스텀 악보 렌더러 — 존 배경색 실색상화, 단일 음표 테두리 박스 렌더링 추가
- `lib/glory_music/`: Glory Music UI 킷을 그대로 구현한 참고용 데모(Now Playing / Library) — 현재 앱 내비게이션에는 연결되어 있지 않고 코드만 보존

## 알려진 제한 / 다음 단계
- 로고·앱 아이콘은 임시 플레이스홀더(원형 + 음표 아이콘) — 실제 로고 확정 시 교체 필요
- `CollectionScreen`의 스캔 결과는 실제 OMR 모델 미완성으로 샘플 악보로 대체 표시됨(모델 학습 완료 후 실제 연동 필요)
- 규칙 1 페이지의 피아노는 예시가 2옥타브(C4~C6) 범위라 건반이 촘촘함 — 실기기 확인 권장
- 브라우저 자동화로 일부만 시각 검증함(툴 환경의 좌표 스케일 이슈로 전체 클릭 흐름 검증에 한계) — 실기기/`flutter run -d windows`로 최종 확인 권장
- 위 "로컬 미커밋" 변경사항은 아직 GitHub에 푸시되지 않음
