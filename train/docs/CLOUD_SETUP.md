# Claude Code on the web (claude.ai/code) 설정 — RunPod 연동

## 1) 노트북에서 SSH 키를 클립보드로 복사 (아직 인터넷 되는 지금)

PowerShell에서:
```powershell
Get-Content ~/.ssh/runpod_auto | Set-Clipboard
```

또는 Git Bash에서:
```bash
cat ~/.ssh/runpod_auto | clip
```

(터미널에 출력하지 말고 바로 클립보드로만 복사하세요 — 화면 캡처나 로그에 남지 않도록.)

## 2) claude.ai/code에서 클라우드 환경 생성

1. https://claude.ai/code 접속 → GitHub 계정 연결 → `musicscore_flutter` 레포 선택해서 클라우드 환경 생성
2. 환경 설정(Environment settings)에서 **환경변수(Environment Variables)** 추가:
   - `RUNPOD_SSH_KEY` = (1번에서 복사한 키 전체 내용 붙여넣기)
   - `RUNPOD_HOST` = `213.173.103.141`
   - `RUNPOD_PORT` = `43347`

## 3) 환경의 setup script에 아래 추가

(키를 파일로 복원해서 SSH가 쓸 수 있게 권한 설정)

```bash
mkdir -p ~/.ssh
echo "$RUNPOD_SSH_KEY" > ~/.ssh/runpod_auto
chmod 600 ~/.ssh/runpod_auto
```

## 4) 이후 클라우드 세션에서 접속 확인

```bash
ssh -i ~/.ssh/runpod_auto -o StrictHostKeyChecking=no -p $RUNPOD_PORT root@$RUNPOD_HOST "echo connected"
```

## 5) 이어서 할 일

`train/docs/TRAINING_REPORT.md`(최신 상태 요약)를 클라우드 세션에게 보여주면 지금까지 진행 상황을 바로 파악하고 이어서 모니터링 가능합니다. (구 `HANDOFF_STATUS.md`는 `docs/archive/`로 이동, 내용은 TRAINING_REPORT.md에 흡수됨)

## 참고: 보안 주의사항
- claude.ai/code 환경변수는 **그 환경을 편집할 수 있는 사람에게는 보임** (전용 시크릿 저장소는 아직 없음)
- 개인 계정 단독 사용이면 실질 위험은 낮지만, 더 안전하게 하려면 이 용도 전용의 새 SSH 키를 따로 만들어 파드에 추가하고, 필요 없어지면 파드 쪽 authorized_keys에서 그 키만 제거하는 방법도 있음
