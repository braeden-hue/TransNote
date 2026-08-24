"""
export_tflite.py  —  Export trained Round3 (grand-staff) OMR PyTorch models to TFLite.

round3train 전용 export 스크립트. `ml/omr/training/export_tflite.py`는 구버전 파이프라인의
model.py(vocab=1012, 단일 오선용 256px 캔버스 가정)를 참조해서 이 폴더의 체크포인트(vocab=258,
대보표 SYSTEM_CANVAS_H=480 캔버스)를 그대로 쓸 수 없다 — state_dict shape mismatch.
이 스크립트는 round3train/model.py + round3train/dataset.py의 실제 전처리 파이프라인
(extract_system_canvas, IMG_MEAN/IMG_STD)을 그대로 사용해 정확히 학습 때와 같은 입력으로 export한다.

Conversion pipeline for each model:
    PyTorch .pt  →  ONNX  →  (onnx2tf)  →  TFLite

Decoder quantization note
──────────────────────────
SegNet과 Encoder는 INT8로 양자화(실사용상 정확도 저하 거의 없음). Decoder는 기본 FP32로
export(자기회귀 Transformer decoder의 INT8 양자화는 토큰 오류율을 크게 악화시키는 경우가
많음) — --quantize_decoder로 실험적으로 켤 수 있음.

Usage
─────
    # FP32만 (캘리브레이션 이미지 불필요, 파이프라인 검증용)
    python round3train/export_tflite.py \\
      --segnet   round3train/models/segnet_best.pt \\
      --seq2seq  round3train/models/round1_curriculum_p2s4g/seq2seq_best.pt \\
      --tokenizer round3train/tokenizer258.json \\
      --out_dir  round3train/tflite_export \\
      --version  v1 --no_quantize

    # INT8 (실제 대보표 PNG 필요, data_dir에 *.png + *.json 쌍 존재해야 함)
    python round3train/export_tflite.py \\
      --segnet   round3train/models/segnet_best.pt \\
      --seq2seq  round3train/models/round1_curriculum_p2s4g/seq2seq_best.pt \\
      --tokenizer round3train/tokenizer258.json \\
      --data_dir round3train/calib_data \\
      --out_dir  round3train/tflite_export \\
      --version  v1

Output:
    tflite_export/segnet_INT8_v1.tflite
    tflite_export/encoder_INT8_v1.tflite
    tflite_export/decoder_INT8_v1.tflite   (FP32 by default)
    tflite_export/tokenizer_v1.json
    tflite_export/{segnet,encoder,decoder}_INT8.tflite  (latest, overwritten each run)
    tflite_export/tokenizer.json
"""

import argparse
import glob
import json
import os
import shutil
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from model import SegNet, OmrSeq2Seq, CANVAS_W, SEQ_LEN, NUM_CLASSES, SOS_ID, infer_arch_from_state_dict
from dataset import (load_preprocessed, detect_staffs, extract_system_canvas,
                     IMG_MEAN, IMG_STD, SYSTEM_CANVAS_H, load_tokenizer, make_model_input)


# ─────────────────────────────────────────────────────────────────────────────
#  Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_state(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt.get('model', ckpt)


def load_segnet(ckpt_path: str, device: torch.device) -> SegNet:
    model = SegNet(num_classes=NUM_CLASSES)
    model.load_state_dict(_load_state(ckpt_path, device))
    return model.eval().to(device)


def load_seq2seq(ckpt_path: str, vocab_size: int, device: torch.device) -> OmrSeq2Seq:
    # 체크포인트마다 아키텍처(in_ch/extra_height_stages/pool_h)가 다를 수 있어(예: CoordConv
    # 실험용 in_ch=2) 생성자 기본값으로 만들면 r15처럼 논디폴트 아키텍처인 체크포인트에서
    # shape mismatch가 난다 — handler.py와 동일하게 state_dict에서 실제 아키텍처를 역산해서 쓴다.
    state = _load_state(ckpt_path, device)
    arch = infer_arch_from_state_dict(state)
    print(f"    Detected arch: {arch}")
    model = OmrSeq2Seq(vocab_size=vocab_size, **arch)
    model.load_state_dict(state)
    return model.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration data builders (for INT8 PTQ) — 실제 학습 전처리 파이프라인 재사용
# ─────────────────────────────────────────────────────────────────────────────

def _collect_pngs(data_dir: str, n: int) -> list:
    paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))[:n]
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {data_dir}")
    return paths


def _build_segnet_calib(paths: list, out_npy: str, n: int = 100) -> str:
    """Random 320×320 patches → [N, 1, 320, 320] float32, [-1, 1] 정규화 (SegNet 학습과 동일)."""
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
    """실제 grand-staff 학습 전처리(load_preprocessed → detect_staffs → extract_system_canvas)를
    그대로 통과시켜 [N, in_ch, SYSTEM_CANVAS_H, CANVAS_W] 캘리브레이션 셋을 만든다.
    detect_staffs가 2개 오선(treble+bass)을 못 찾는 이미지는 건너뛴다. in_ch=2(CoordConv)면
    make_model_input()이 학습/추론과 동일하게 좌표 채널을 붙여준다 — 여기서 채널 수를 안 맞추면
    캘리브레이션 입력 shape이 실제 모델 입력과 달라져 양자화 스케일이 잘못 잡힌다."""
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
        inp = make_model_input(canvas, in_ch).numpy()  # [in_ch, H, W]
        arrays.append(inp[np.newaxis])  # [1, in_ch, H, W]
    if not arrays:
        raise RuntimeError("대보표(2개 오선) 인식 가능한 캘리브레이션 이미지가 없음")
    data = np.concatenate(arrays, axis=0)
    np.save(out_npy, data)
    print(f"    Calibration data: {out_npy}  shape={data.shape}")
    return out_npy


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def _export_onnx_segnet(model: SegNet, out: str):
    dummy = torch.zeros(1, 1, 320, 320)
    torch.onnx.export(
        model, dummy, out,
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17,
        dynamo=False,
    )
    print(f"    ONNX -> {out}")


def _export_onnx_encoder(encoder: nn.Module, out: str, in_ch: int = 1):
    dummy = torch.zeros(1, in_ch, SYSTEM_CANVAS_H, CANVAS_W)
    torch.onnx.export(
        encoder, dummy, out,
        input_names=['canvas'], output_names=['memory'],
        dynamic_axes={'canvas': {0: 'batch'}, 'memory': {0: 'batch'}},
        opset_version=17,
        dynamo=False,
    )
    print(f"    ONNX -> {out}")


class _DecoderStepWrapper(nn.Module):
    """Single-step decoder wrapper for export.

    Inputs:
      past_ids : [1, T]      int64   — 지금까지 생성된 토큰 id (SOS 포함)
      memory   : [1, S, 512] float32 — 인코더 출력

    Output:
      next_logits : [1, VOCAB_SIZE] float32 — 다음 토큰 logits

    C++ decoder_runner는 스텝마다 이걸 한 번씩 호출하며 past_ids를 1씩 늘려간다
    (명시적 KV-cache 텐서 없이 매 스텝 O(T) 재계산 — 608 토큰 이하에서는 감당 가능).
    """
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, past_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(past_ids, memory)
        return logits[:, -1, :]


def _export_onnx_decoder(seq2seq: OmrSeq2Seq, out: str):
    wrapper = _DecoderStepWrapper(seq2seq.decoder).eval()
    dummy_ids = torch.tensor([[SOS_ID]], dtype=torch.long)
    dummy_mem = torch.zeros(1, SEQ_LEN, 512)
    torch.onnx.export(
        wrapper, (dummy_ids, dummy_mem), out,
        input_names=['past_ids', 'memory'], output_names=['next_logits'],
        dynamic_axes={
            'past_ids':    {0: 'batch', 1: 'seq_len'},
            'memory':      {0: 'batch', 1: 'enc_seq'},
            'next_logits': {0: 'batch'},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"    ONNX -> {out}")


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


def _convert_tflite(onnx_path: str, tflite_path: str,
                    calib_npy: str | None, quantize: bool,
                    input_op_name: str = 'input',
                    keep_layout_input_names: list | None = None) -> bool:
    """calib_npy는 이미 (px/255 - mean)/std로 정규화가 끝난 데이터라고 가정한다 --
    onnx2tf에는 mean=0/std=1을 넘겨서 추가 정규화를 건너뛰게 한다.

    keep_layout_input_names: onnx2tf가 NCHW->NHWC로 자동 전치하면 안 되는 입력 이름 목록.
    이미지가 아닌 3차원 이상 텐서(예: Transformer의 [B, seq, dim] memory)를 이미지로 오인해서
    축 순서를 뒤섞는 경우가 있어(예: [1,320,512] -> [1,512,1]로 깨짐) 명시적으로 막아야 한다."""
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
    size_kb = os.path.getsize(versioned) / 1024
    print(f"    Versioned -> {versioned}  ({size_kb:.0f} KB)")
    print(f"    Latest    -> {latest}")


def _save_tokenizer(src: str, out_dir: str, version: str):
    versioned = os.path.join(out_dir, f"tokenizer_{version}.json")
    latest    = os.path.join(out_dir, "tokenizer.json")
    shutil.copy(src, versioned)
    shutil.copy(src, latest)
    print(f"    Versioned -> {versioned}")
    print(f"    Latest    -> {latest}")


def main():
    p = argparse.ArgumentParser(
        description="Export round3train (grand-staff) OMR models to TFLite."
    )
    p.add_argument('--segnet',           default=None,
                   help='segnet_best.pt (선택 -- 현재 추론 경로는 오선 검출에 SegNet을 안 쓰고 '
                        '순수 OpenCV(detect_staffs())만 쓰므로, 안 주면 SegNet export는 건너뜀)')
    p.add_argument('--seq2seq',          required=True, help='seq2seq_best.pt (encoder+decoder)')
    p.add_argument('--tokenizer',        default=os.path.join(os.path.dirname(__file__), 'tokenizer258.json'))
    p.add_argument('--data_dir',         default=None,
                   help='실제 대보표 PNG 디렉토리 (INT8 캘리브레이션용, 없으면 자동 FP32 fallback)')
    p.add_argument('--out_dir',          default=os.path.join(os.path.dirname(__file__), 'tflite_export'))
    p.add_argument('--version',          default='v1')
    p.add_argument('--calib_n',          type=int, default=100)
    p.add_argument('--quantize_decoder', action='store_true')
    p.add_argument('--no_quantize',      action='store_true')
    p.add_argument('--device',           default='cpu')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp = os.path.join(args.out_dir, '_export_tmp')
    os.makedirs(tmp, exist_ok=True)

    device   = torch.device(args.device)
    quantize = not args.no_quantize

    tok2id, _ = load_tokenizer(args.tokenizer)
    vocab_size = len(tok2id)

    print(f"\n{'='*60}")
    print(f"  Round3 Grand-Staff OMR TFLite Export  —  version: {args.version}")
    print(f"{'='*60}")
    print(f"  segnet     : {args.segnet}")
    print(f"  seq2seq    : {args.seq2seq}")
    print(f"  vocab_size : {vocab_size}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"  INT8       : {quantize}")
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

    # ── 1. SegNet ─────────────────────────────────────────
    print("--- [1/3] SegNet ---------------------------------------")
    if not args.segnet:
        print("    --segnet 안 줌 -- 건너뜀 (현재 오선 검출은 OpenCV detect_staffs()만 씀)\n")
    else:
        segnet   = load_segnet(args.segnet, device)
        onnx_seg = os.path.join(tmp, 'segnet.onnx')
        _export_onnx_segnet(segnet, onnx_seg)
        _simplify(onnx_seg)

        calib_seg = None
        if quantize:
            calib_seg = os.path.join(tmp, 'calib_segnet.npy')
            _build_segnet_calib(image_paths, calib_seg, n=args.calib_n)

        tflite_seg = os.path.join(tmp, 'segnet_INT8.tflite')
        if _convert_tflite(onnx_seg, tflite_seg, calib_seg, quantize, input_op_name='input'):
            _save_versioned(tflite_seg, args.out_dir, 'segnet_INT8', args.version)
        print()

    # ── 2. Encoder ────────────────────────────────────────
    print("--- [2/3] Encoder --------------------------------------")
    seq2seq  = load_seq2seq(args.seq2seq, vocab_size, device)
    in_ch    = seq2seq.encoder.backbone[0].block[0].weight.shape[1]  # 체크포인트 실제 입력 채널(1 또는 CoordConv 2)
    onnx_enc = os.path.join(tmp, 'encoder.onnx')
    _export_onnx_encoder(seq2seq.encoder, onnx_enc, in_ch=in_ch)
    _simplify(onnx_enc)

    calib_enc = None
    if quantize:
        calib_enc = os.path.join(tmp, 'calib_encoder.npy')
        _build_encoder_calib(image_paths, calib_enc, n=min(50, len(image_paths)), in_ch=in_ch)

    tflite_enc = os.path.join(tmp, 'encoder_INT8.tflite')
    if _convert_tflite(onnx_enc, tflite_enc, calib_enc, quantize, input_op_name='canvas'):
        _save_versioned(tflite_enc, args.out_dir, 'encoder_INT8', args.version)
    print()

    # ── 3. Decoder ────────────────────────────────────────
    print("--- [3/3] Decoder ----------------------------------------")
    print(f"    Interface: (past_ids[1,T], memory[1,S,512]) -> next_logits[1,{vocab_size}]")
    print("    T grows by 1 each decoding step (no explicit KV-cache tensors)")
    onnx_dec = os.path.join(tmp, 'decoder.onnx')
    _export_onnx_decoder(seq2seq, onnx_dec)
    _simplify(onnx_dec)

    if args.quantize_decoder:
        print("    WARNING: --quantize_decoder는 아직 미구현 (decoder는 past_ids+memory 두 입력을 "
              "함께 보정해야 해서 segnet/encoder와 같은 단일 캘리브레이션 방식으로 안 됨) -- FP32로 진행")
    else:
        print("    Quantization: FP32 (기본값, 자기회귀 decoder는 INT8 시 오류율 악화 위험)")

    tflite_dec = os.path.join(tmp, 'decoder_INT8.tflite')
    if _convert_tflite(onnx_dec, tflite_dec, None, False,
                       keep_layout_input_names=['memory', 'past_ids']):
        _save_versioned(tflite_dec, args.out_dir, 'decoder_INT8', args.version)
    print()

    # ── 4. Tokenizer ──────────────────────────────────────
    print("--- Tokenizer --------------------------------------------")
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
