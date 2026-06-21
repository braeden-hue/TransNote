#!/usr/bin/env python3
"""
imslp_pdf_to_images.py — IMSLP Public Domain PDF를 촬영 시뮬레이션 이미지로 변환.

IMSLP에서 다운로드한 공개 도메인 악보 PDF를 고해상도 PNG로 변환한 뒤,
실제 카메라로 촬영한 것처럼 보이는 augmentation을 적용합니다.
Round 5 도메인 적응 학습 데이터 생성에 사용합니다.

━━━ Public Domain 확인 절차 (필수) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMSLP의 악보 페이지 상단에 표시된 라이선스 태그를 확인:
  ✅ 사용 가능: "Public Domain"  /  "Creative Commons Zero"  /  "CC0"
  ❌ 사용 불가: "Creative Commons Attribution"  /  "Non-commercial"  /
               "Personal Use Only"  /  "IMSLP Performance Restricted"

확인 후 pd_manifest.txt 에 파일명을 등록해야 스크립트가 처리합니다.
이 절차는 자동화할 수 없으며 다운로드 담당자가 직접 확인해야 합니다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ Round 5 범위 제한 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Round 1~4 는 합성 데이터(1~2 오선)로 학습된 모델입니다.
IMSLP에는 관현악 총보(10오선 이상), 합창보(SATB), 가사+반주 등
현재 토큰 어휘에 없는 기호와 구조가 포함될 수 있습니다.

이 스크립트는 --max-staves (기본값 2) 를 초과하는 페이지를 자동 제외하여
모델이 학습한 범위(솔로 피아노 = 최대 Grand Staff) 로 도메인 적응을 제한합니다.
토큰 어휘에 없는 기호(가사, 코드 다이어그램, 피규어드 베이스 등)는
mscz_to_label.py 단계에서 자동으로 제거됩니다(기존 동작 유지).

권장 IMSLP 카테고리:
  https://imslp.org/wiki/Category:For_piano  (솔로 피아노, 1~2 오선)
  → Chopin, Mozart, Beethoven, Schubert 피아노 소나타/야상곡 등
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

의존성 설치:
    pip install pdf2image pillow numpy opencv-python

Poppler 설치:
    Windows: https://github.com/oschwartz10612/poppler-windows/releases
             압축 풀고 bin 폴더 경로를 --poppler-path 로 지정
    Linux:   sudo apt install poppler-utils
    macOS:   brew install poppler

pd_manifest.txt 형식 (input-dir 안에 위치):
    # 한 줄에 파일명 하나 (주석은 # 으로 시작)
    chopin_nocturne_op9_no2.pdf
    mozart_sonata_k331.pdf

Usage:
    # PDF 폴더 일괄 처리 (pd_manifest.txt 필수, augmentation 5배)
    python ml/scripts/imslp_pdf_to_images.py \\
        --input-dir ml/data/imslp_pdf \\
        --output-dir ml/data/Round5_real \\
        --augment 5

    # 단일 PDF (--pd-confirmed 플래그로 PD 직접 선언)
    python ml/scripts/imslp_pdf_to_images.py \\
        --input ml/data/imslp_pdf/chopin_waltz.pdf \\
        --output-dir ml/data/Round5_real \\
        --augment 0 --pd-confirmed

    # 오선 수 제한 조정 (Grand Staff = 2, 단선보 = 1)
    python ml/scripts/imslp_pdf_to_images.py \\
        --input-dir ml/data/imslp_pdf \\
        --output-dir ml/data/Round5_real \\
        --max-staves 2

    # Windows Poppler 경로 지정
    python ml/scripts/imslp_pdf_to_images.py \\
        --input-dir ml/data/imslp_pdf \\
        --output-dir ml/data/Round5_real \\
        --poppler-path "C:/poppler/bin"
"""

import argparse
import random
import sys
from pathlib import Path

try:
    from pdf2image import convert_from_path
except ImportError:
    sys.exit("ERROR: pdf2image 미설치.\n  pip install pdf2image")

try:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
except ImportError:
    sys.exit("ERROR: 의존성 미설치.\n  pip install pillow numpy opencv-python")


# ─────────────────────────────────────────────────────────────────────────────
#  Public Domain manifest 파싱
# ─────────────────────────────────────────────────────────────────────────────

def load_pd_manifest(manifest_path: Path) -> set[str]:
    """
    pd_manifest.txt 를 읽어 PD 확인된 파일명 집합 반환.
    파일이 없으면 빈 집합 반환.
    """
    if not manifest_path.exists():
        return set()
    confirmed = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            confirmed.add(line)
    return confirmed


def check_pd(pdf_path: Path, pd_set: set[str], pd_confirmed_flag: bool) -> bool:
    """
    단일 PDF 파일의 PD 상태 확인.
    pd_confirmed_flag(--pd-confirmed CLI 옵션) 또는 manifest 등록 여부로 판단.
    """
    if pd_confirmed_flag:
        return True
    if pdf_path.name in pd_set:
        return True
    print(f"  SKIP: '{pdf_path.name}' 이 pd_manifest.txt 에 없음 — IMSLP에서 Public Domain 확인 후 등록하세요.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  오선 수 감지 (Round 5 범위 제한)
# ─────────────────────────────────────────────────────────────────────────────

def count_staves(page_img: Image.Image) -> int:
    """
    페이지 이미지에서 오선계(staff system) 수를 추정.

    방법: 수평선 집중도 분석으로 오선 묶음 수를 카운트.
    반환값은 시스템(악단) 수 — 피아노 Grand Staff = 1시스템(오선 2개).
    실제 오선 줄(line) 수 / 5 로 근사.
    """
    gray = np.array(page_img.convert("L"))
    # 이진화 (악보 잉크 = 어두운 픽셀)
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # 수평 방향 픽셀 밀도 (행별 합산)
    row_density = bw.sum(axis=1).astype(np.float32)
    row_density /= (bw.shape[1] * 255 + 1e-6)  # 0~1 정규화

    # 밀도 임계값 이상인 행을 "오선 후보 행"으로 표시
    threshold = 0.25
    candidate = (row_density > threshold).astype(np.uint8)

    # 연속된 candidate 행 묶음을 하나의 오선 줄로 카운트
    line_count = 0
    in_line = False
    for v in candidate:
        if v and not in_line:
            line_count += 1
            in_line = True
        elif not v:
            in_line = False

    # 오선은 5줄 한 묶음, 시스템 수 = ceil(오선 줄 수 / 5)
    staff_systems = max(1, round(line_count / 5))
    return staff_systems


def page_within_stave_limit(page_img: Image.Image, max_staves: int) -> tuple[bool, int]:
    """
    페이지의 오선 시스템 수가 max_staves 이하인지 확인.
    Returns (ok, detected_count).
    """
    detected = count_staves(page_img)
    return detected <= max_staves, detected


# ─────────────────────────────────────────────────────────────────────────────
#  PDF → PNG 변환
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_pages(pdf_path: Path, dpi: int = 300, poppler_path: str = None) -> list[Image.Image]:
    """PDF 한 파일의 모든 페이지를 PIL Image 리스트로 반환."""
    kwargs = {"dpi": dpi, "fmt": "png"}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path
    try:
        pages = convert_from_path(str(pdf_path), **kwargs)
        return pages
    except Exception as exc:
        print(f"  ERROR: PDF 변환 실패 ({exc})")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  카메라 촬영 시뮬레이션 Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _random_perspective(img: np.ndarray, max_shift_ratio: float = 0.04) -> np.ndarray:
    """원근 왜곡: 카메라 각도 오차 시뮬레이션."""
    h, w = img.shape[:2]
    shift = lambda: random.uniform(-max_shift_ratio, max_shift_ratio)

    src = np.float32([
        [0,      0],
        [w - 1,  0],
        [w - 1,  h - 1],
        [0,      h - 1],
    ])
    dst = np.float32([
        [w * shift(),       h * shift()],
        [w - 1 + w * shift(), h * shift()],
        [w - 1 + w * shift(), h - 1 + h * shift()],
        [w * shift(),       h - 1 + h * shift()],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(245, 240, 228))


def _random_rotation(img: np.ndarray, max_deg: float = 2.0) -> np.ndarray:
    """미세 회전: 카메라 수평 오차 시뮬레이션."""
    h, w = img.shape[:2]
    angle = random.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(245, 240, 228))


def _paper_tint(img: Image.Image) -> Image.Image:
    """종이 색감: 약간 노란빛 (오래된 악보 느낌)."""
    r, g, b = img.split()
    r = r.point(lambda i: min(255, i + random.randint(0, 12)))
    g = g.point(lambda i: min(255, i + random.randint(0, 8)))
    b = b.point(lambda i: max(0,   i - random.randint(0, 15)))
    return Image.merge("RGB", (r, g, b))


def _random_brightness_contrast(img: Image.Image) -> Image.Image:
    """밝기/대비 변동: 촬영 환경 조명 차이."""
    brightness = random.uniform(0.80, 1.20)
    contrast   = random.uniform(0.85, 1.15)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def _shadow_gradient(img: np.ndarray) -> np.ndarray:
    """그림자 그라데이션: 페이지 한쪽이 어두운 카메라 효과."""
    h, w = img.shape[:2]
    shadow = np.ones((h, w), dtype=np.float32)

    direction = random.choice(["left", "right", "top", "bottom"])
    strength  = random.uniform(0.10, 0.25)

    if direction == "left":
        for x in range(w):
            shadow[:, x] = 1.0 - strength * (1 - x / w)
    elif direction == "right":
        for x in range(w):
            shadow[:, x] = 1.0 - strength * (x / w)
    elif direction == "top":
        for y in range(h):
            shadow[y, :] = 1.0 - strength * (1 - y / h)
    else:
        for y in range(h):
            shadow[y, :] = 1.0 - strength * (y / h)

    shadow = np.stack([shadow] * 3, axis=-1)
    return np.clip(img.astype(np.float32) * shadow, 0, 255).astype(np.uint8)


def _gaussian_blur(img: Image.Image) -> Image.Image:
    """카메라 초점 흐림."""
    radius = random.uniform(0.3, 1.2)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _jpeg_compression(img: Image.Image, quality: int = None) -> Image.Image:
    """JPEG 압축 아티팩트."""
    import io
    if quality is None:
        quality = random.randint(70, 92)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _add_noise(img: np.ndarray) -> np.ndarray:
    """카메라 센서 노이즈."""
    sigma = random.uniform(1.0, 4.0)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_page(page_img: Image.Image, severity: str = "medium") -> Image.Image:
    """
    인쇄된 악보를 카메라로 찍은 것처럼 augmentation 적용.

    severity:
        "light"  — 미세한 변형 (스캔에 가까운 품질)
        "medium" — 일반 스마트폰 촬영 수준
        "heavy"  — 불량 촬영 (흔들림, 강한 그림자)
    """
    prob = {"light": 0.3, "medium": 0.6, "heavy": 0.9}.get(severity, 0.6)

    # PIL → numpy
    img_np = np.array(page_img.convert("RGB"))

    # 원근 왜곡
    if random.random() < prob:
        max_shift = {"light": 0.02, "medium": 0.04, "heavy": 0.07}.get(severity, 0.04)
        img_np = _random_perspective(img_np, max_shift)

    # 미세 회전
    if random.random() < prob:
        max_deg = {"light": 0.5, "medium": 2.0, "heavy": 4.0}.get(severity, 2.0)
        img_np = _random_rotation(img_np, max_deg)

    # 그림자
    if random.random() < prob * 0.7:
        img_np = _shadow_gradient(img_np)

    # numpy → PIL
    pil_img = Image.fromarray(img_np)

    # 종이 색감
    if random.random() < 0.5:
        pil_img = _paper_tint(pil_img)

    # 밝기/대비
    pil_img = _random_brightness_contrast(pil_img)

    # 초점 흐림
    if random.random() < prob * 0.8:
        pil_img = _gaussian_blur(pil_img)

    # 센서 노이즈
    if random.random() < prob * 0.5:
        pil_img = Image.fromarray(_add_noise(np.array(pil_img)))

    # JPEG 압축
    if random.random() < prob:
        pil_img = _jpeg_compression(pil_img)

    return pil_img


# ─────────────────────────────────────────────────────────────────────────────
#  단일 PDF 처리
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    augment_count: int,
    dpi: int,
    severity: str,
    poppler_path: str | None,
    max_staves: int = 2,
) -> tuple[int, int, int]:
    """
    한 PDF 파일을 처리해서 이미지를 저장.
    오선 수가 max_staves 초과인 페이지는 자동 제외.
    Returns (저장된 원본 페이지 수, 저장된 augmented 이미지 수, 제외된 페이지 수)
    """
    stem = pdf_path.stem
    print(f"\n[{pdf_path.name}]")

    pages = pdf_to_pages(pdf_path, dpi=dpi, poppler_path=poppler_path)
    if not pages:
        return 0, 0, 0

    print(f"  페이지 수: {len(pages)}")

    saved_orig = 0
    saved_aug  = 0
    skipped    = 0

    for page_idx, page in enumerate(pages):
        # 오선 수 초과 페이지 제외 (관현악 총보, 합창보 등)
        ok, detected = page_within_stave_limit(page, max_staves)
        if not ok:
            print(f"  p{page_idx + 1:02d}: SKIP (오선 시스템 {detected}개 감지, 한계 {max_staves}개)")
            skipped += 1
            continue

        page_stem = f"{stem}_p{page_idx + 1:02d}"

        # 원본 (클린) 이미지 저장
        orig_path = output_dir / f"{page_stem}_orig.png"
        page.save(str(orig_path), format="PNG")
        saved_orig += 1

        # augmented 이미지 저장
        for aug_idx in range(augment_count):
            aug_img  = augment_page(page, severity=severity)
            aug_path = output_dir / f"{page_stem}_aug{aug_idx + 1:02d}.png"
            aug_img.save(str(aug_path), format="PNG")
            saved_aug += 1

        print(f"  p{page_idx + 1:02d}: 오선 {detected}개 — 원본 1장 + augmented {augment_count}장 저장")

    return saved_orig, saved_aug, skipped


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IMSLP Public Domain PDF를 촬영 시뮬레이션 이미지로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",         default=None, help="단일 PDF 경로")
    parser.add_argument("--input-dir",     default=None, help="PDF 폴더 경로")
    parser.add_argument("--output-dir",    required=True, help="이미지 출력 폴더")
    parser.add_argument("--augment",       type=int, default=5,
                        help="페이지당 augmented 이미지 수 (0이면 원본만 저장)")
    parser.add_argument("--dpi",           type=int, default=300,
                        help="PDF 렌더링 DPI (기본: 300)")
    parser.add_argument("--severity",      default="medium",
                        choices=["light", "medium", "heavy"],
                        help="카메라 시뮬레이션 강도 (기본: medium)")
    parser.add_argument("--max-staves",    type=int, default=20,
                        help="페이지당 허용 최대 오선 시스템 수 (기본: 20, 사실상 무제한)")
    parser.add_argument("--pd-confirmed",  action="store_true",
                        help="단일 --input 파일에 대해 PD 여부를 직접 선언")
    parser.add_argument("--pd-manifest",   default=None,
                        help="PD 확인 파일 목록 경로 (기본: input-dir/pd_manifest.txt)")
    parser.add_argument("--poppler-path",  default=None,
                        help="Windows Poppler bin 폴더 경로")
    parser.add_argument("--seed",          type=int, default=None,
                        help="난수 시드 (재현성)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PD manifest 로드
    pd_set: set[str] = set()
    if args.input_dir:
        manifest_path = Path(args.pd_manifest) if args.pd_manifest \
                        else Path(args.input_dir) / "pd_manifest.txt"
        pd_set = load_pd_manifest(manifest_path)
        if not pd_set and not args.pd_confirmed:
            print(f"WARNING: pd_manifest.txt 를 찾을 수 없거나 비어 있습니다.")
            print(f"  경로: {manifest_path}")
            print(f"  IMSLP에서 Public Domain 확인 후 파일명을 한 줄씩 등록하세요.")

    pdfs: list[Path] = []
    if args.input:
        pdfs = [Path(args.input)]
    elif args.input_dir:
        pdfs = sorted(Path(args.input_dir).glob("*.pdf"))
        if not pdfs:
            sys.exit(f"ERROR: {args.input_dir} 에서 PDF를 찾을 수 없습니다.")
    else:
        parser.print_help()
        sys.exit(1)

    print(f"PDF 수      : {len(pdfs)}")
    print(f"출력 폴더   : {out_dir.resolve()}")
    print(f"DPI         : {args.dpi}")
    print(f"Augment     : {args.augment}배/페이지")
    print(f"Severity    : {args.severity}")
    print(f"Max staves  : {args.max_staves}")

    total_orig = total_aug = total_skipped = total_pd_skip = 0
    for pdf_path in pdfs:
        # PD 확인되지 않은 파일 제외
        if not check_pd(pdf_path, pd_set, args.pd_confirmed):
            total_pd_skip += 1
            continue

        o, a, s = process_pdf(
            pdf_path, out_dir,
            augment_count=args.augment,
            dpi=args.dpi,
            severity=args.severity,
            poppler_path=args.poppler_path,
            max_staves=args.max_staves,
        )
        total_orig    += o
        total_aug     += a
        total_skipped += s

    print(f"\n{'─' * 50}")
    if total_pd_skip:
        print(f"PD 미확인   : {total_pd_skip}개 파일 제외 (pd_manifest.txt 에 등록 필요)")
    print(f"원본 페이지 : {total_orig}장")
    print(f"오선 초과 제외: {total_skipped}페이지")
    print(f"Augmented   : {total_aug}장")
    print(f"총 이미지   : {total_orig + total_aug}장")
    print(f"출력 폴더   : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
