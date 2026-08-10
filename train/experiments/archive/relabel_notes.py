"""
round3train/relabel_notes.py  –  기존 라벨(JSON)을 note 토큰 분해 스킴으로 변환.

배경:
  note-{pitch}-{dur} 단일 토큰을 note-{pitch} + dur-{dur} 두 토큰으로 분해하도록
  tokenizer.json / generate_scores.py를 변경했다. 이미 렌더링된 PNG는 그대로 쓰고,
  각 이미지와 짝을 이루는 *.json 라벨의 "tokens" 배열만 이 스킴으로 다시 쓰면 된다
  (MuseScore 재렌더링 불필요).

사용법:
    python round3train/relabel_notes.py --data_dir "데이터 학습/Round3" --in_place
    python round3train/relabel_notes.py --data_dir "데이터 학습/Round1+2+3" --out_dir "데이터 학습/Round1+2+3_relabeled"

기본은 --out_dir로 원본 폴더를 건드리지 않고 변환 결과를 복사한다.
--in_place를 주면 원본 *.json을 직접 덮어쓴다(백업 없음, 주의).
"""
import argparse
import json
import re
import shutil
from pathlib import Path

_NOTE_DUR_RE = re.compile(r"^note-(.+)-(\d+/\d+)$")


def split_note_token(tok: str):
    """note-{pitch}-{dur} -> [note-{pitch}, dur-{dur}]. 그 외 토큰은 그대로 1개 리스트로."""
    m = _NOTE_DUR_RE.match(tok)
    if not m:
        return [tok]
    pitch, dur = m.group(1), m.group(2)
    return [f"note-{pitch}", f"dur-{dur}"]


def relabel_tokens(tokens: list) -> tuple[list, bool]:
    """tokens 배열을 변환. (새 배열, 변경 여부) 반환."""
    new_tokens = []
    changed = False
    for tok in tokens:
        split = split_note_token(tok)
        if len(split) > 1:
            changed = True
        new_tokens.extend(split)
    return new_tokens, changed


def process_dir(data_dir: Path, out_dir: Path, in_place: bool):
    json_paths = sorted(data_dir.glob("*.json"))
    n_changed = n_skipped = n_untouched = 0

    for jp in json_paths:
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        tokens = data.get("tokens")
        if tokens is None:
            n_skipped += 1
            continue

        new_tokens, changed = relabel_tokens(tokens)

        if not in_place:
            out_path = out_dir / jp.name
        else:
            out_path = jp

        if changed:
            data["tokens"] = new_tokens
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            n_changed += 1
        else:
            n_untouched += 1
            if not in_place:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(jp, out_path)

        if not in_place:
            # 같은 stem의 이미지/musicxml도 out_dir로 복사 (그대로, 내용 변경 없음)
            stem = jp.stem
            for ext in (".png", ".jpg", ".jpeg", ".musicxml"):
                src = data_dir / f"{stem}{ext}"
                if src.is_file():
                    dst = out_dir / f"{stem}{ext}"
                    if not dst.exists():
                        shutil.copy2(src, dst)

    print(f"{data_dir}: 변환 {n_changed}개, 이미 새 형식(무변경) {n_untouched}개, "
          f"tokens 없어 스킵 {n_skipped}개")


def main():
    p = argparse.ArgumentParser(description="기존 라벨을 note-{pitch}+dur-{dur} 분해 스킴으로 변환")
    p.add_argument("--data_dir", required=True, help="*.png + *.json 라벨이 있는 디렉토리")
    p.add_argument("--out_dir", default=None,
                   help="변환 결과를 저장할 새 디렉토리 (기본: <data_dir>_relabeled)")
    p.add_argument("--in_place", action="store_true",
                   help="원본 *.json을 직접 덮어쓴다 (백업 없음, 주의해서 사용)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir이 존재하지 않음: {data_dir}")

    if args.in_place:
        out_dir = data_dir
    else:
        out_dir = Path(args.out_dir) if args.out_dir else data_dir.parent / f"{data_dir.name}_relabeled"

    process_dir(data_dir, out_dir, args.in_place)


if __name__ == "__main__":
    main()
