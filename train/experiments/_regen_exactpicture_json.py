"""exactPicture/의 클래식(비-newage) 폴더 전체를 mscz 기준으로 json 재생성.
json이 없으면 새로 만들고, 있으면 mscz와 비교해서 달라진 것만 보고한다.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'data' / 'local_pools' / 'exactPicture'
MUSESCORE = "/c/Program Files/MuseScore 4/bin/MuseScore4.exe"


def main():
    folders = sorted(p for p in SRC.iterdir() if p.is_dir() and not p.name.startswith('newage'))
    created, changed, unchanged, failed = [], [], [], []

    for folder in folders:
        json_path = folder / f"{folder.name}.json"
        old_content = None
        if json_path.exists():
            old_content = json_path.read_text(encoding='utf-8')

        result = subprocess.run(
            [sys.executable, str(HERE / 'mscz_to_tokens.py'), str(folder),
             '--musescore', MUSESCORE, '--force'],
            capture_output=True, text=True, encoding='utf-8', errors='replace')

        if 'ERROR' in result.stdout or result.returncode != 0:
            failed.append((folder.name, result.stdout.strip() + result.stderr.strip()))
            continue

        new_content = json_path.read_text(encoding='utf-8') if json_path.exists() else None
        if old_content is None:
            created.append(folder.name)
        elif new_content != old_content:
            old_toks = json.loads(old_content)['tokens'] if old_content else []
            new_toks = json.loads(new_content)['tokens'] if new_content else []
            changed.append((folder.name, len(old_toks), len(new_toks)))
        else:
            unchanged.append(folder.name)

    print(f"\n=== 결과: 총 {len(folders)}곡 ===")
    print(f"신규 생성: {len(created)}곡 -> {created}")
    print(f"변경됨(mscz와 불일치했던 것): {len(changed)}곡")
    for name, old_n, new_n in changed:
        print(f"  {name}: 토큰수 {old_n} -> {new_n}")
    print(f"동일(변경 없음): {len(unchanged)}곡")
    if failed:
        print(f"\n실패: {len(failed)}곡")
        for name, msg in failed:
            print(f"  {name}: {msg}")


if __name__ == '__main__':
    main()
