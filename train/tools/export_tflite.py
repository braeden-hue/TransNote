"""export_tflite.py — 학습된 대보표 OMR 모델(PyTorch)을 TFLite로 export.

파이프라인: PyTorch .pt -> ONNX -> (onnx2tf) -> TFLite
ONNX 래퍼 클래스 3종은 onnx_wrappers.py, 설계 배경은 EXPORT_NOTES.md 참고.

산출물(4개 그래프 + 토크나이저):
    encoder_INT8.tflite         인코더 (항상 FP32 -- EXPORT_NOTES.md §10)
    decoder_memkv_INT8.tflite   cross-attention K,V 사전계산 (이미지당 1회)
    decoder_INT8.tflite         디코더 단일 스텝 (self-attn KV캐시)
    decoder_bulk_INT8.tflite    캐시 일괄 재구성 (InlineTimeCorrector 호환용)
    tokenizer.json
  각각 <이름>_<version>.tflite 로도 같이 저장된다.

사용법:
    # Hybrid (권장 -- 인코더 FP32 + 나머지 dynamic-range INT8)
    python train/tools/export_tflite.py --seq2seq <ckpt.pt> \\
        --out_dir train/tflite_export_dr --version v1 --no_quantize --dynamic_range

    # 순정 FP32
    python train/tools/export_tflite.py --seq2seq <ckpt.pt> \\
        --out_dir train/tflite_export --version v1 --no_quantize
"""

import argparse
import glob
import os
import shutil
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import torch
import torch.nn as nn

_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # train/ (런타임 모듈)
sys.path.insert(0, _TRAIN)
from model import SegNet, OmrSeq2Seq, CANVAS_W, SEQ_LEN, NUM_CLASSES, SOS_ID, PAD_ID, infer_arch_from_state_dict
from dataset import (load_preprocessed, detect_staffs, extract_system_canvas,
                     IMG_MEAN, IMG_STD, SYSTEM_CANVAS_H, load_tokenizer, make_model_input)
from onnx_wrappers import _MemoryKVWrapper, _DecoderStepWrapperKV, _BulkCaptureWrapperKV


# ─────────────────────────────────────────────────────────────────────────────
#  체크포인트 로더
# ─────────────────────────────────────────────────────────────────────────────

def _load_state(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt.get('model', ckpt)


def load_segnet(ckpt_path: str, device: torch.device) -> SegNet:
    model = SegNet(num_classes=NUM_CLASSES)
    model.load_state_dict(_load_state(ckpt_path, device))
    return model.eval().to(device)


def load_seq2seq(ckpt_path: str, vocab_size: int, device: torch.device) -> OmrSeq2Seq:
    """아키텍처는 생성자 기본값이 아니라 state_dict에서 역산해야 함 -- EXPORT_NOTES.md §1."""
    state = _load_state(ckpt_path, device)
    arch = infer_arch_from_state_dict(state)
    print(f"    Detected arch: {arch}")
    model = OmrSeq2Seq(vocab_size=vocab_size, **arch)
    model.load_state_dict(state)
    return model.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
#  INT8 PTQ 캘리브레이션 데이터 -- EXPORT_NOTES.md §2
# ─────────────────────────────────────────────────────────────────────────────

def _collect_pngs(data_dir: str, n: int) -> list:
    paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))[:n]
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {data_dir}")
    return paths


def _build_segnet_calib(paths: list, out_npy: str, n: int = 100) -> str:
    """랜덤 320x320 패치 -> [N,1,320,320] float32, [-1,1] 정규화(SegNet 학습과 동일)."""
    arrays = []
    for p in paths[:n]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        y = np.random.randint(0, max(1, h - 320))
        x = np.random.randint(0, max(1, w - 320))
        patch = img[y:y + 320, x:x + 320]
        patch = cv2.resize(patch, (320, 320)).astype(np.float32) / 127.5 - 1.0
        arrays.append(patch[np.newaxis, np.newaxis])
    data = np.concatenate(arrays, axis=0)
    np.save(out_npy, data)
    print(f"    Calibration data: {out_npy}  shape={data.shape}")
    return out_npy


def _build_encoder_calib(paths: list, out_npy: str, n: int = 50, in_ch: int = 1) -> str:
    """실제 학습 전처리를 그대로 통과시켜 [N,in_ch,H,W] 캘리브레이션 셋 생성 -- §2.
    in_ch를 안 맞추면 양자화 스케일이 잘못 잡힌다."""
    arrays = []
    for p in paths:
        if len(arrays) >= n:
            break
        gray = load_preprocessed(p)
        staffs = detect_staffs(gray)
        if len(staffs) < 2:
            continue
        tile = extract_system_canvas(gray, staffs[:2])
        canvas = (tile.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
        arrays.append(make_model_input(canvas, in_ch).numpy()[np.newaxis])
    if not arrays:
        raise RuntimeError("대보표(2개 오선) 인식 가능한 캘리브레이션 이미지가 없음")
    data = np.concatenate(arrays, axis=0)
    np.save(out_npy, data)
    print(f"    Calibration data: {out_npy}  shape={data.shape}")
    return out_npy


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX export -- 래퍼 클래스는 onnx_wrappers.py, 셋 다 shape 고정이라 dynamic_axes 불필요
# ─────────────────────────────────────────────────────────────────────────────

def _export_onnx_segnet(model: SegNet, out: str):
    torch.onnx.export(
        model, torch.zeros(1, 1, 320, 320), out,
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17, dynamo=False,
    )
    print(f"    ONNX -> {out}")


def _export_onnx_encoder(encoder: nn.Module, out: str, in_ch: int = 1):
    torch.onnx.export(
        encoder, torch.zeros(1, in_ch, SYSTEM_CANVAS_H, CANVAS_W), out,
        input_names=['canvas'], output_names=['memory'],
        dynamic_axes={'canvas': {0: 'batch'}, 'memory': {0: 'batch'}},
        opset_version=17, dynamo=False,
    )
    print(f"    ONNX -> {out}")


def _decoder_dims(seq2seq: OmrSeq2Seq):
    """(num_layers, num_heads, head_dim) — 더미 입력 shape 계산용."""
    decoder = seq2seq.decoder
    D = decoder.embed_dim
    H = decoder.transformer.layers[0].self_attn.num_heads
    return len(decoder.transformer.layers), H, D // H


def _export_onnx_memory_kv(seq2seq: OmrSeq2Seq, out: str):
    wrapper = _MemoryKVWrapper(seq2seq.decoder).eval()
    torch.onnx.export(
        wrapper, (torch.zeros(1, SEQ_LEN, 512),), out,
        input_names=['memory'], output_names=['k_mem', 'v_mem'],
        opset_version=17, dynamo=False,
    )
    print(f"    ONNX(memory KV 사전계산) -> {out}")


def _export_onnx_decoder_kv(seq2seq: OmrSeq2Seq, out: str, cache_len: int = 300):
    L, H, Dh = _decoder_dims(seq2seq)
    wrapper = _DecoderStepWrapperKV(seq2seq.decoder, cache_len).eval()
    dummies = (
        torch.tensor([[SOS_ID]], dtype=torch.long),   # token_id
        torch.tensor([0], dtype=torch.long),          # pos
        torch.zeros(L, 1, H, SEQ_LEN, Dh),            # k_mem_in
        torch.zeros(L, 1, H, SEQ_LEN, Dh),            # v_mem_in
        torch.zeros(L, 1, H, cache_len, Dh),          # k_cache_in
        torch.zeros(L, 1, H, cache_len, Dh),          # v_cache_in
    )
    torch.onnx.export(
        wrapper, dummies, out,
        input_names=['token_id', 'pos', 'k_mem_in', 'v_mem_in', 'k_cache_in', 'v_cache_in'],
        output_names=['next_logits', 'k_cache_out', 'v_cache_out'],
        opset_version=17, dynamo=False,
    )
    print(f"    ONNX(KV캐시, cache_len={cache_len}) -> {out}")


def _export_onnx_bulk_capture(seq2seq: OmrSeq2Seq, out: str, chunk_len: int = 40,
                              cache_len: int = 300):
    L, H, Dh = _decoder_dims(seq2seq)
    wrapper = _BulkCaptureWrapperKV(seq2seq.decoder, chunk_len, cache_len).eval()
    dummy_tokens = torch.full((1, chunk_len), PAD_ID, dtype=torch.long)
    dummy_tokens[0, 0] = SOS_ID
    torch.onnx.export(
        wrapper, (dummy_tokens, torch.zeros(L, 1, H, SEQ_LEN, Dh), torch.zeros(L, 1, H, SEQ_LEN, Dh)), out,
        input_names=['tokens', 'k_mem_in', 'v_mem_in'],
        output_names=['k_cache_out', 'v_cache_out'],
        opset_version=17, dynamo=False,
    )
    print(f"    ONNX(일괄 캐시 채우기, chunk_len={chunk_len}) -> {out}")


def _simplify(onnx_path: str):
    try:
        import onnx
        import onnxsim
        m = onnx.load(onnx_path)
        m_sim, ok = onnxsim.simplify(m)
        if ok:
            onnx.save(m_sim, onnx_path)
            print("    ONNX simplified")
        else:
            print("    ONNX simplification had no effect")
    except ImportError:
        print("    onnxsim not installed -- skipping (pip install onnxsim)")


# ─────────────────────────────────────────────────────────────────────────────
#  TFLite 변환
# ─────────────────────────────────────────────────────────────────────────────

def _convert_tflite(onnx_path: str, tflite_path: str,
                    calib_npy: str | None, quantize: bool,
                    input_op_name: str = 'input',
                    keep_layout_input_names: list | None = None,
                    fp16: bool = False,
                    dynamic_range: bool = False) -> bool:
    """keep_layout_input_names: onnx2tf가 NCHW->NHWC로 자동 전치하면 안 되는 입력 이름 --
    새 입력을 추가하면 여기에도 넣어야 함(EXPORT_NOTES.md §8).
    calib_npy는 이미 정규화가 끝난 데이터라 onnx2tf엔 mean=0/std=1을 넘긴다(§2)."""
    try:
        import onnx2tf
    except ImportError:
        print("    ERROR: onnx2tf not installed. Run: pip install onnx2tf tensorflow")
        return False

    tmp_tf = tflite_path + "_saved_model"
    os.makedirs(tmp_tf, exist_ok=True)

    kwargs: dict = dict(
        input_onnx_file_path=onnx_path,
        output_folder_path=tmp_tf,
        non_verbose=True,
    )
    if keep_layout_input_names:
        kwargs['keep_ncw_or_nchw_or_ncdhw_input_names'] = keep_layout_input_names

    if quantize and calib_npy and os.path.exists(calib_npy):
        data = np.load(calib_npy)
        kwargs['output_integer_quantized_tflite'] = True
        kwargs['custom_input_op_name_np_data_path'] = [[
            input_op_name, calib_npy, np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ]]
        print(f"    Quantizing INT8 with: {calib_npy}  shape={data.shape}")
        want_substrings = ('full_integer_quant', 'integer_quant')
    elif fp16:
        # 이 그래프에서는 CPU 런타임 실행 불가 -- EXPORT_NOTES.md §9 (GPU 델리게이트 실험용)
        print("    Exporting as FP16")
        want_substrings = ('float16',)
    elif dynamic_range:
        # 가중치만 INT8, 활성값은 FP32 -- 캘리브레이션 불필요. §5의 2D 평탄화가 전제조건
        print("    Exporting as dynamic-range quantized (INT8 weight-only, FP32 activations)")
        kwargs['output_dynamic_range_quantized_tflite'] = True
        want_substrings = ('dynamic_range_quant',)
    else:
        print("    Exporting as FP32")
        want_substrings = ('_float32', 'float32')

    onnx2tf.convert(**kwargs)

    candidates = [f for f in os.listdir(tmp_tf) if f.endswith('.tflite')]
    if not candidates:
        print(f"    ERROR: no .tflite found in {tmp_tf}")
        shutil.rmtree(tmp_tf, ignore_errors=True)
        return False

    chosen = None
    for want in want_substrings:
        matches = [f for f in candidates if want in f]
        if matches:
            chosen = sorted(matches, key=len)[0]  # 가장 짧은(가장 기본형) 이름 우선
            break
    if chosen is None:
        print(f"    WARNING: 원하는 variant({want_substrings})를 못 찾아 첫 파일로 대체: {candidates}")
        chosen = candidates[0]
    print(f"    Selected variant: {chosen}  (candidates: {candidates})")

    shutil.copy(os.path.join(tmp_tf, chosen), tflite_path)
    shutil.rmtree(tmp_tf, ignore_errors=True)
    print(f"    TFLite -> {tflite_path}")
    return True


def _save_versioned(src: str, out_dir: str, stem: str, version: str):
    versioned = os.path.join(out_dir, f"{stem}_{version}.tflite")
    latest    = os.path.join(out_dir, f"{stem}.tflite")
    shutil.copy(src, versioned)
    shutil.copy(src, latest)
    print(f"    Saved -> {versioned}  ({os.path.getsize(versioned)/1024:.0f} KB)  + {stem}.tflite")


def _save_tokenizer(src: str, out_dir: str, version: str):
    shutil.copy(src, os.path.join(out_dir, f"tokenizer_{version}.json"))
    shutil.copy(src, os.path.join(out_dir, "tokenizer.json"))
    print(f"    Saved -> tokenizer_{version}.json + tokenizer.json")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="대보표 OMR 모델을 TFLite로 export.")
    p.add_argument('--segnet',           default=None,
                   help='segnet_best.pt (선택). 현재 추론 경로는 SegNet을 안 쓰므로 '
                        '안 주면 건너뜀 -- EXPORT_NOTES.md §13')
    p.add_argument('--seq2seq',          required=True, help='seq2seq_best.pt (encoder+decoder)')
    p.add_argument('--tokenizer',        default=os.path.join(_TRAIN, 'tokenizer258.json'))
    p.add_argument('--data_dir',         default=None,
                   help='대보표 PNG 디렉토리 (INT8 캘리브레이션용, 없으면 FP32로 fallback)')
    p.add_argument('--out_dir',          default=os.path.join(_TRAIN, 'tflite_export'))
    p.add_argument('--version',          default='v1')
    p.add_argument('--calib_n',          type=int, default=100)
    p.add_argument('--quantize_decoder', action='store_true')
    p.add_argument('--no_quantize',      action='store_true')
    p.add_argument('--device',           default='cpu')
    p.add_argument('--cache_len',        type=int, default=300,
                   help='디코더 KV캐시 고정 크기(최대 디코딩 스텝 수, INFER_MAX_LEN과 맞춤)')
    p.add_argument('--chunk_len',        type=int, default=40,
                   help='일괄 캐시 채우기 청크 길이(첫 마디 최대 토큰 수 상한)')
    p.add_argument('--fp16',             action='store_true',
                   help='FP16 변형 선택 -- 이 그래프는 CPU에서 실행 불가, GPU 델리게이트 '
                        '실험용으로만 남겨둠 (EXPORT_NOTES.md §9)')
    p.add_argument('--dynamic_range',    action='store_true',
                   help='디코더 3종을 dynamic-range 양자화(가중치 INT8/활성값 FP32)로 export. '
                        '인코더는 발산 문제로 항상 FP32 유지 (EXPORT_NOTES.md §10)')
    args = p.parse_args()
    if args.fp16 and args.dynamic_range:
        p.error('--fp16과 --dynamic_range는 동시에 줄 수 없음')

    os.makedirs(args.out_dir, exist_ok=True)
    tmp = os.path.join(args.out_dir, '_export_tmp')
    os.makedirs(tmp, exist_ok=True)

    device   = torch.device(args.device)
    quantize = not args.no_quantize
    tok2id, _ = load_tokenizer(args.tokenizer)
    vocab_size = len(tok2id)

    mode = 'FP16' if args.fp16 else ('Hybrid(dynamic-range)' if args.dynamic_range else 'FP32')
    print(f"\n{'='*60}")
    print(f"  대보표 OMR TFLite Export  —  version: {args.version}  mode: {mode}")
    print(f"{'='*60}")
    print(f"  seq2seq    : {args.seq2seq}")
    print(f"  vocab_size : {vocab_size}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"{'='*60}\n")

    image_paths: list = []
    if quantize:
        if not args.data_dir:
            print("WARNING: --data_dir 없음 -- INT8 캘리브레이션 불가, FP32로 전환.\n")
            quantize = False
        else:
            try:
                image_paths = _collect_pngs(args.data_dir, max(args.calib_n, 100))
                print(f"Calibration images: {len(image_paths)} from {args.data_dir}\n")
            except FileNotFoundError as e:
                print(f"WARNING: {e}\nFalling back to FP32 export.\n")
                quantize = False

    # ── 1. SegNet (선택) ──────────────────────────────────
    print("--- [1/5] SegNet ---------------------------------------")
    if not args.segnet:
        print("    --segnet 안 줌 -- 건너뜀 (EXPORT_NOTES.md §13)\n")
    else:
        onnx_seg = os.path.join(tmp, 'segnet.onnx')
        _export_onnx_segnet(load_segnet(args.segnet, device), onnx_seg)
        _simplify(onnx_seg)
        calib_seg = None
        if quantize:
            calib_seg = os.path.join(tmp, 'calib_segnet.npy')
            _build_segnet_calib(image_paths, calib_seg, n=args.calib_n)
        tflite_seg = os.path.join(tmp, 'segnet_INT8.tflite')
        if _convert_tflite(onnx_seg, tflite_seg, calib_seg, quantize, input_op_name='input'):
            _save_versioned(tflite_seg, args.out_dir, 'segnet_INT8', args.version)
        print()

    # ── 2. Encoder (dynamic_range와 무관하게 항상 FP32 -- §10) ──
    print("--- [2/5] Encoder --------------------------------------")
    seq2seq  = load_seq2seq(args.seq2seq, vocab_size, device)
    in_ch    = seq2seq.encoder.backbone[0].block[0].weight.shape[1]   # 1 또는 CoordConv 2
    onnx_enc = os.path.join(tmp, 'encoder.onnx')
    _export_onnx_encoder(seq2seq.encoder, onnx_enc, in_ch=in_ch)
    _simplify(onnx_enc)

    calib_enc = None
    if quantize:
        calib_enc = os.path.join(tmp, 'calib_encoder.npy')
        _build_encoder_calib(image_paths, calib_enc, n=min(50, len(image_paths)), in_ch=in_ch)
    if args.dynamic_range:
        print("    (--dynamic_range여도 인코더는 FP32 고정 -- EXPORT_NOTES.md §10)")

    tflite_enc = os.path.join(tmp, 'encoder_INT8.tflite')
    if _convert_tflite(onnx_enc, tflite_enc, calib_enc, quantize, input_op_name='canvas',
                       fp16=args.fp16, dynamic_range=False):
        _save_versioned(tflite_enc, args.out_dir, 'encoder_INT8', args.version)
    print()

    # ── 3. cross-attention K,V 사전계산 -- §6 ──────────────
    print("--- [3/5] Decoder memory-KV precompute -----------------")
    print("    memory[1,S,512] -> (k_mem, v_mem)[L,1,H,S,Dh], 이미지당 1회만 실행")
    onnx_memkv = os.path.join(tmp, 'decoder_memkv.onnx')
    _export_onnx_memory_kv(seq2seq, onnx_memkv)
    _simplify(onnx_memkv)
    tflite_memkv = os.path.join(tmp, 'decoder_memkv_INT8.tflite')
    if _convert_tflite(onnx_memkv, tflite_memkv, None, False, input_op_name='memory',
                       keep_layout_input_names=['memory'],
                       fp16=args.fp16, dynamic_range=args.dynamic_range):
        _save_versioned(tflite_memkv, args.out_dir, 'decoder_memkv_INT8', args.version)
    print()

    # ── 4. Decoder 단일 스텝 (고정 shape KV캐시 -- §3) ─────
    print("--- [4/5] Decoder step ---------------------------------")
    print(f"    (token_id, pos, k_mem_in, v_mem_in, k_cache_in, v_cache_in)")
    print(f"    -> (next_logits[1,{vocab_size}], k_cache_out, v_cache_out), cache_len={args.cache_len}")
    onnx_dec = os.path.join(tmp, 'decoder.onnx')
    _export_onnx_decoder_kv(seq2seq, onnx_dec, cache_len=args.cache_len)
    _simplify(onnx_dec)

    if args.quantize_decoder:
        print("    WARNING: --quantize_decoder 미구현(디코더는 여러 입력을 함께 보정해야 함) -- 무시")

    tflite_dec = os.path.join(tmp, 'decoder_INT8.tflite')
    if _convert_tflite(onnx_dec, tflite_dec, None, False, input_op_name='token_id',
                       keep_layout_input_names=['token_id', 'pos', 'k_mem_in', 'v_mem_in',
                                                 'k_cache_in', 'v_cache_in'],
                       fp16=args.fp16, dynamic_range=args.dynamic_range):
        _save_versioned(tflite_dec, args.out_dir, 'decoder_INT8', args.version)
    print()

    # ── 5. 캐시 일괄 재구성 (InlineTimeCorrector 호환 -- §7) ──
    print("--- [5/5] Decoder bulk-capture -------------------------")
    print(f"    (tokens[1,{args.chunk_len}], k_mem_in, v_mem_in) -> (k_cache_out, v_cache_out)")
    onnx_bulk = os.path.join(tmp, 'decoder_bulk.onnx')
    _export_onnx_bulk_capture(seq2seq, onnx_bulk, chunk_len=args.chunk_len,
                              cache_len=args.cache_len)
    _simplify(onnx_bulk)
    tflite_bulk = os.path.join(tmp, 'decoder_bulk_INT8.tflite')
    if _convert_tflite(onnx_bulk, tflite_bulk, None, False, input_op_name='tokens',
                       keep_layout_input_names=['tokens', 'k_mem_in', 'v_mem_in'],
                       fp16=args.fp16, dynamic_range=args.dynamic_range):
        _save_versioned(tflite_bulk, args.out_dir, 'decoder_bulk_INT8', args.version)
    print()

    print("--- Tokenizer ------------------------------------------")
    if os.path.exists(args.tokenizer):
        _save_tokenizer(args.tokenizer, args.out_dir, args.version)
    else:
        print(f"    WARNING: tokenizer not found at {args.tokenizer}")
    print()

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{'='*60}")
    print(f"  Done.  Output: {os.path.abspath(args.out_dir)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
