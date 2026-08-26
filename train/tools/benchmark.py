"""benchmark.py — PyTorch(서버/개발 PC CPU) vs TFLite Hybrid(CPU) 정량 성능 비교.

train/QUANTIZATION_MOBILE.md ② 항목의 벤치마크 리포트를 만드는 스크립트. newage21~30
(held-out) 실사 이미지로 인코더+디코더 둘 다 "이미 로드된 상태"에서 스텝별 시간을 재서,
모델 로드/1회성 초기화 비용과 스테디스테이트 추론 비용을 분리한다 — 이 구분 없이 재면
"이미지당 몇 초"가 로드 오버헤드에 묻혀서 왜곡된다(이 프로젝트에서 실제로 겪은 문제).

Hybrid(인코더 FP32 + 디코더 계열 dynamic-range, 버킷팅 적용 -- EXPORT_NOTES.md §14)가
채택된 구성이라(QUANTIZATION_MOBILE.md "최종 결정") 순정 FP32 TFLite와의 비교는 더 안
한다 — 필요하면 `python train/tools/export_tflite.py --precision fp32`로 언제든 재현 가능.
FP16은 여전히 CPU 실행이 안 되므로(1장짜리 빠른 실패 확인만) 계속 확인한다.

정확도는 이 스크립트에서 다시 재지 않는다 — 이미 별도로 반복 검증된 숫자를 그대로 인용한다
(QUANTIZATION_MOBILE.md 참고). 대신 여기서는 속도/크기/메모리만 정량 측정한다.

사용법:
    python train/tools/benchmark.py
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # train/ (런타임 모듈)

import numpy as np
import psutil
import torch

from dataset import load_tokenizer, load_preprocessed, best_effort_staff_detection, extract_system_canvas
from inference import run_image
from model import OmrSeq2Seq, infer_arch_from_state_dict
from tflite_infer import TFLiteOmrModel

ROOT = Path(__file__).resolve().parent.parent.parent   # 저장소 루트(train/tools/ 기준)
CKPT = ROOT / "train" / "checkpoints" / "r15_cropfix_coordconv" / "seq2seq_best.pt"
TOKENIZER = ROOT / "train" / "tokenizer258.json"
TFLITE_FP16_DIR = ROOT / "train" / "tflite_export_fp16"
TFLITE_DR_DIR = ROOT / "train" / "tflite_export_dr"
DATA_DIR = ROOT / "realImage" / "exactPicture"
IMAGES = [f"newage{i}" for i in range(21, 31)]
TFLITE_FILES = ('encoder_INT8.tflite', 'decoder_memkv_INT8.tflite',
                'decoder_INT8.tflite', 'decoder_bulk_INT8.tflite',
                'decoder_large_INT8.tflite', 'decoder_rebucket_INT8.tflite')

PROC = psutil.Process()


def _peak_mem_mb():
    """Windows: peak_wset(피크 워킹셋). 다른 OS는 rss로 폴백."""
    mi = PROC.memory_info()
    peak = getattr(mi, 'peak_wset', None) or mi.rss
    return peak / (1024 * 1024)


def _collect_images():
    """cv2.imread가 Windows에서 비-ASCII 경로를 못 여는 경우가 있어(실사27_1.jpg 등),
    KakaoTalk_*.jpg(ASCII) 우선 -- 없으면 임시 ASCII 경로로 복사해서 사용."""
    paths = []
    for name in IMAGES:
        folder = DATA_DIR / name
        jpgs = sorted(folder.glob("KakaoTalk_*.jpg")) or sorted(folder.glob("*.jpg"))
        if not jpgs:
            continue
        img_path = jpgs[0]
        if not img_path.name.isascii():
            tmp = Path(tempfile.gettempdir()) / f"{name}_bench.jpg"
            shutil.copy(img_path, tmp)
            img_path = tmp
        paths.append((name, img_path))
    return paths


def bench_pytorch(images):
    tok2id, id2tok = load_tokenizer(str(TOKENIZER))
    ckpt = torch.load(str(CKPT), map_location='cpu', weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    model = OmrSeq2Seq(vocab_size=len(tok2id), **arch)
    model.load_state_dict(ckpt['model'])
    model.eval()
    device = torch.device('cpu')

    mem_before = _peak_mem_mb()
    times = []
    for name, img_path in images:
        gray0 = load_preprocessed(str(img_path))
        staffs, gray = best_effort_staff_detection(gray0, use_full_warp=True)
        if not staffs or len(staffs) < 2:
            continue
        t0 = time.time()
        run_image(str(img_path), model, tok2id, id2tok, device)
        times.append(time.time() - t0)
    mem_after = _peak_mem_mb()

    return {
        'backend': 'PyTorch (개발 PC CPU)',
        'avg_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'peak_mem_mb': mem_after,
        'mem_delta_mb': mem_after - mem_before,
        'model_size_mb': CKPT.stat().st_size / (1024 * 1024),
    }


def bench_tflite(images, tflite_dir: Path, label: str):
    tok2id, id2tok = load_tokenizer(str(TOKENIZER))
    mem_before = _peak_mem_mb()
    model = TFLiteOmrModel(str(tflite_dir))  # 인터프리터 로드는 1회만(스테디스테이트 측정)

    times = []
    errors = 0
    for name, img_path in images:
        gray0 = load_preprocessed(str(img_path))
        staffs, gray = best_effort_staff_detection(gray0, use_full_warp=True)
        if not staffs or len(staffs) < 2:
            continue
        canvas = extract_system_canvas(gray, staffs[:2])
        from dataset import IMG_MEAN, IMG_STD
        norm = (canvas.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
        try:
            t0 = time.time()
            memory = model.encode(norm)
            model.decode_bucketed(memory, tok2id, id2tok, stop_token_id=tok2id.get('barline-final'))
            times.append(time.time() - t0)
        except Exception:
            errors += 1
    mem_after = _peak_mem_mb()

    size_mb = sum((tflite_dir / f).stat().st_size for f in TFLITE_FILES
                  if (tflite_dir / f).exists()) / (1024 * 1024)

    result = {
        'backend': label,
        'peak_mem_mb': mem_after,
        'mem_delta_mb': mem_after - mem_before,
        'model_size_mb': size_mb,
        'errors': errors,
    }
    if times:
        result['avg_ms'] = sum(times) / len(times) * 1000
        result['min_ms'] = min(times) * 1000
        result['max_ms'] = max(times) * 1000
    return result


def main():
    images = _collect_images()
    print(f"이미지 {len(images)}개로 벤치마크 (newage21~30 held-out)\n")

    results = []
    print("[1/2] PyTorch(개발 PC CPU) 측정 중...")
    results.append(bench_pytorch(images))

    print("[2/2] TFLite Hybrid(FP32 인코더+Dynamic-Range 디코더/일괄캐시, CPU) 측정 중...")
    results.append(bench_tflite(images, TFLITE_DR_DIR, "TFLite Hybrid (CPU)"))

    # 순수 FP16(그래프 전체 fp16, 인코더까지)은 여전히 실패 -- 1장짜리 빠른 실패 확인만.
    print("\nTFLite FP16(그래프 전체) 시도(런타임 실패 여부 확인)...")
    fp16_size = sum((TFLITE_FP16_DIR / f).stat().st_size for f in TFLITE_FILES
                    if (TFLITE_FP16_DIR / f).exists()) / (1024 * 1024)
    try:
        r = bench_tflite(images[:1], TFLITE_FP16_DIR, "TFLite FP16 (CPU)")
        fp16_status = f"성공 ({r.get('avg_ms', 0):.0f}ms)"
    except Exception as e:
        fp16_status = f"런타임 실패 -- {str(e)[:80]}"

    print(f"\n{'='*70}")
    print("  벤치마크 리포트")
    print(f"{'='*70}")
    for r in results:
        print(f"\n{r['backend']}")
        print(f"  모델 크기      : {r['model_size_mb']:.1f} MB")
        if 'avg_ms' in r:
            print(f"  추론 레이턴시  : 평균 {r['avg_ms']:.0f}ms  (최소 {r['min_ms']:.0f}ms / 최대 {r['max_ms']:.0f}ms)")
        print(f"  Peak Memory    : {r['peak_mem_mb']:.0f} MB  (측정 구간 증가분 {r['mem_delta_mb']:+.0f} MB)")
        if 'errors' in r and r['errors']:
            print(f"  에러(크래시)   : {r['errors']}/{len(images)}")
    print(f"\nTFLite FP16(그래프 전체, CPU)")
    print(f"  모델 크기      : {fp16_size:.1f} MB")
    print(f"  런타임         : {fp16_status}")
    print(f"\n정확도(별도 검증된 값 인용, held-out newage21~30 10곡 평균):")
    print(f"  PyTorch(하이브리드)      : 94.2%")
    print(f"  TFLite Hybrid            : 94.3%")
    print(f"  TFLite FP16(그래프 전체) : 측정 불가(런타임 실패)")


if __name__ == '__main__':
    main()
