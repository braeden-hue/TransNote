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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import SegNet, OmrSeq2Seq, CANVAS_W, SEQ_LEN, NUM_CLASSES, SOS_ID, PAD_ID, infer_arch_from_state_dict
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


class _DecoderStepWrapperKV(nn.Module):
    """고정 크기 self-attention KV캐시 기반 단일 스텝 디코더 wrapper (2026-08-24).

    _DecoderStepWrapper(위)는 past_ids를 매 스텝 growing 텐서로 넣는 방식이라 TFLite에서
    실제로 돌려보니 두 번째 디코딩 스텝부터 reshape 에러로 깨졌다(QUANTIZATION_MOBILE.md
    참고). 이 wrapper는 모든 입출력 shape을 처음부터 끝까지 고정시켜서(pos가 바뀌어도
    텐서 shape은 안 바뀜) 문제를 근본적으로 피한다:
      - self-attention K,V 캐시를 슬라이싱(`cache[:, :, :pos+1, :]`)하는 대신, 고정 크기
        버퍼 전체에 대해 attention을 계산하고 pos보다 뒤쪽은 마스킹(-inf)으로 제외한다.
      - 캐시에 새 K,V를 써넣는 것도 in-place 슬라이스 대입 대신 torch.where 기반 마스킹으로
        한다(pos 위치만 새 값, 나머지는 기존 값 유지).
      두 방식 모두 텐서 shape이 안 바뀌는 순수 elementwise/브로드캐스트 연산이라 TFLite
      변환이 안전하다.
    cross-attention(memory 대상) K,V는 캐싱하지 않고 매 스텝 memory에서 다시 projection한다
    (self-attention의 O(T^2) 비용에 비해 상수 비용이라 단순화 -- production PyTorch 경로는
    이 부분도 캐싱하지만, export 그래프를 간단히 유지하려고 여기서는 뺐다).

    Inputs:
      token_id   : [1, 1]                          int64
      pos        : [1]                             int64 — 이 토큰의 0-index 위치(SOS=0)
      memory     : [1, SEQ_LEN, 512]                float32 — 인코더 출력
      k_cache_in : [num_layers, 1, H, cache_len, Dh] float32
      v_cache_in : [num_layers, 1, H, cache_len, Dh] float32

    Outputs:
      next_logits : [1, vocab_size]
      k_cache_out : [num_layers, 1, H, cache_len, Dh]
      v_cache_out : [num_layers, 1, H, cache_len, Dh]
    """
    def __init__(self, decoder: nn.Module, cache_len: int):
        super().__init__()
        self.decoder = decoder
        self.cache_len = cache_len

    def forward(self, token_id, pos, memory, k_cache_in, v_cache_in):
        decoder = self.decoder
        D = decoder.embed_dim
        C = self.cache_len
        pos_idx = pos[0]
        emb = decoder.token_emb(token_id) * decoder.emb_scale        # [1,1,D]
        x = emb + decoder.pos_enc[:, pos_idx:pos_idx + 1, :]

        idx = torch.arange(C, device=token_id.device)
        write_mask = (idx == pos_idx).view(1, 1, C, 1)   # 새 K,V를 써넣을 캐시 슬롯
        valid_mask = (idx <= pos_idx).view(1, 1, 1, C)   # attention에서 허용할 슬롯

        k_out_layers, v_out_layers = [], []
        for li, layer in enumerate(decoder.transformer.layers):
            k_cache, v_cache = k_cache_in[li], v_cache_in[li]   # [1,H,C,Dh]

            x1 = layer.norm1(x)
            sa = layer.self_attn
            H = sa.num_heads
            Dh = D // H
            B, T, _ = x1.shape   # T=1(단일 스텝)
            # (2026-08-26) 3D 입력([B,T,D])으로 F.linear/nn.Linear를 그대로 부르면 ONNX가
            # MatMul(→TFLite BATCH_MATMUL)로 export해서 dynamic-range 양자화/fp16 hybrid
            # 커널의 양자화 대상에서 빠진다(QUANTIZATION_MOBILE.md 참고, 팀원 제안). 모든
            # linear 적용 전에 2D([B*T,D])로 펴서 Gemm(→TFLite FULLY_CONNECTED)으로 유도하고
            # 끝나면 다시 3D로 되돌린다 — 수학적으로 동일 연산, export 표현만 바뀜.
            x1_2d = x1.reshape(B * T, D)
            in_b = sa.in_proj_bias
            q = F.linear(x1_2d, sa.in_proj_weight[:D],    in_b[:D]    if in_b is not None else None).view(B, T, D)
            k = F.linear(x1_2d, sa.in_proj_weight[D:2*D], in_b[D:2*D] if in_b is not None else None).view(B, T, D)
            v = F.linear(x1_2d, sa.in_proj_weight[2*D:],  in_b[2*D:]  if in_b is not None else None).view(B, T, D)
            q = q.view(B, 1, H, Dh).transpose(1, 2)              # [1,H,1,Dh]
            k = k.view(B, 1, H, Dh).transpose(1, 2)
            v = v.view(B, 1, H, Dh).transpose(1, 2)

            # (2026-08-26) torch.where(mask, a, b) 대신 산술 블렌드(mask*a + (1-mask)*b)를 쓴다 --
            # onnx2tf의 dynamic-range 양자화가 Where/Select 노드의 상수 피연산자를 가중치로
            # 오인해 스케일을 2.68e36 같은 값으로 잘못 계산해서 즉시 NaN이 되는 버그를 확인함
            # (QUANTIZATION_MOBILE.md 참고). Mul/Add만 쓰면 Where 노드 자체가 안 생겨서 회피된다.
            write_f = write_mask.to(k_cache.dtype)
            k_cache_new = k_cache * (1 - write_f) + k.expand(-1, -1, C, -1) * write_f
            v_cache_new = v_cache * (1 - write_f) + v.expand(-1, -1, C, -1) * write_f

            scores = torch.matmul(q, k_cache_new.transpose(-2, -1)) / (Dh ** 0.5)  # [1,H,1,C]
            scores = scores + (~valid_mask).to(scores.dtype) * -1e9
            attn = torch.matmul(torch.softmax(scores, dim=-1), v_cache_new)        # [1,H,1,Dh]
            attn = attn.transpose(1, 2).reshape(B * 1, D)
            x = x + layer.dropout1(sa.out_proj(attn).view(B, 1, D))

            x2 = layer.norm2(x)
            mha = layer.multihead_attn
            in_bc = mha.in_proj_bias
            S = memory.shape[1]
            x2_2d = x2.reshape(B * 1, D)
            mem_2d = memory.reshape(B * S, D)
            qc = F.linear(x2_2d, mha.in_proj_weight[:D],    in_bc[:D]    if in_bc is not None else None).view(B, 1, D)
            kc = F.linear(mem_2d, mha.in_proj_weight[D:2*D], in_bc[D:2*D] if in_bc is not None else None).view(B, S, D)
            vc = F.linear(mem_2d, mha.in_proj_weight[2*D:],  in_bc[2*D:]  if in_bc is not None else None).view(B, S, D)
            qc = qc.view(B, 1, H, Dh).transpose(1, 2)
            kc = kc.view(B, S, H, Dh).transpose(1, 2)
            vc = vc.view(B, S, H, Dh).transpose(1, 2)
            attn2 = F.scaled_dot_product_attention(qc, kc, vc)
            attn2 = attn2.transpose(1, 2).reshape(B * 1, D)
            x = x + layer.dropout2(mha.out_proj(attn2).view(B, 1, D))

            x3_2d = layer.norm3(x).reshape(B * 1, D)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x3_2d))))
            x = x + layer.dropout3(ff).view(B, 1, D)

            k_out_layers.append(k_cache_new)
            v_out_layers.append(v_cache_new)

        x = decoder.transformer.norm(x)
        logits = decoder.head(x)[:, -1, :]           # [1, vocab_size]
        k_cache_out = torch.stack(k_out_layers, dim=0)
        v_cache_out = torch.stack(v_out_layers, dim=0)
        return logits, k_cache_out, v_cache_out


def _export_onnx_decoder_kv(seq2seq: OmrSeq2Seq, out: str, cache_len: int = 300):
    """고정 크기 KV캐시 기반 디코더 스텝 export (2026-08-24). 모든 입출력 shape이 pos와
    무관하게 고정이라 dynamic_axes가 필요 없다 -- TFLite 변환·리사이즈 문제 근본 해결책."""
    decoder = seq2seq.decoder
    D = decoder.embed_dim
    H = decoder.transformer.layers[0].self_attn.num_heads
    Dh = D // H
    L = len(decoder.transformer.layers)

    wrapper = _DecoderStepWrapperKV(decoder, cache_len).eval()
    dummy_token = torch.tensor([[SOS_ID]], dtype=torch.long)
    dummy_pos = torch.tensor([0], dtype=torch.long)
    dummy_mem = torch.zeros(1, SEQ_LEN, 512)
    dummy_k = torch.zeros(L, 1, H, cache_len, Dh)
    dummy_v = torch.zeros(L, 1, H, cache_len, Dh)
    torch.onnx.export(
        wrapper, (dummy_token, dummy_pos, dummy_mem, dummy_k, dummy_v), out,
        input_names=['token_id', 'pos', 'memory', 'k_cache_in', 'v_cache_in'],
        output_names=['next_logits', 'k_cache_out', 'v_cache_out'],
        opset_version=17,
        dynamo=False,
    )
    print(f"    ONNX(KV캐시, cache_len={cache_len}) -> {out}")
    print(f"    ONNX -> {out}")


class _BulkCaptureWrapperKV(nn.Module):
    """고정 길이 청크(chunk_len개 토큰)를 한 번에 넣어서 self-attention K,V 캐시를
    일괄로 채우는 wrapper (2026-08-24). PyTorch의 forward_bulk_capture()와 같은 목적:
    InlineTimeCorrector로 교정된 "첫 마디" 구간을 _DecoderStepWrapperKV로 넘어가기 전에
    한 번에 캐시에 반영한다.

    한 위치(예: 박자표 토큰)만 나중에 고치는 게 안 되는 이유: attention이 매 레이어 모든
    앞선 위치를 섞기 때문에, 한 위치의 토큰을 바꾸면 그 뒤 모든 위치의 hidden state가
    이론적으로 다 달라져야 한다(부분 패치 불가) -- 그래서 "고친 뒤 전체를 한 번에
    다시 계산"하는 이 방식이 필요하다.

    chunk_len은 고정(예: 40 -- 첫 마디가 보통 이보다 짧음, 실제 길이보다 짧으면 뒤쪽은
    PAD_ID로 채움). causal masking이라 패딩 위치는 앞쪽 실제 위치의 계산에 전혀 영향을
    주지 않는다(뒤쪽만 보고, 패딩은 항상 뒤쪽에 있으므로).

    Inputs:
      tokens : [1, chunk_len]         int64 — SOS 포함, 실제 길이보다 뒤는 PAD_ID
      memory : [1, SEQ_LEN, 512]      float32

    Outputs:
      k_cache_out, v_cache_out : [num_layers, 1, H, cache_len, Dh] — 0..chunk_len-1 위치가
      채워짐(cache_len - chunk_len만큼은 0으로 패딩). 이후 _DecoderStepWrapperKV가
      pos=실제 길이-1부터 이어서 씀.
    """
    def __init__(self, decoder: nn.Module, chunk_len: int, cache_len: int):
        super().__init__()
        self.decoder = decoder
        self.chunk_len = chunk_len
        self.cache_len = cache_len

    def forward(self, tokens, memory):
        decoder = self.decoder
        D = decoder.embed_dim
        M = self.chunk_len
        B = tokens.shape[0]

        emb = decoder.token_emb(tokens) * decoder.emb_scale     # [1,M,D]
        x = emb + decoder.pos_enc[:, :M, :]

        k_out_layers, v_out_layers = [], []
        for layer in decoder.transformer.layers:
            x1 = layer.norm1(x)
            sa = layer.self_attn
            H = sa.num_heads
            Dh = D // H
            # (2026-08-26) _DecoderStepWrapperKV와 동일 이유로 2D 평탄화 -- Gemm/FULLY_CONNECTED
            # 유도(QUANTIZATION_MOBILE.md 참고)
            x1_2d = x1.reshape(B * M, D)
            in_b = sa.in_proj_bias
            q = F.linear(x1_2d, sa.in_proj_weight[:D],    in_b[:D]    if in_b is not None else None).view(B, M, D)
            k = F.linear(x1_2d, sa.in_proj_weight[D:2*D], in_b[D:2*D] if in_b is not None else None).view(B, M, D)
            v = F.linear(x1_2d, sa.in_proj_weight[2*D:],  in_b[2*D:]  if in_b is not None else None).view(B, M, D)
            q = q.view(B, M, H, Dh).transpose(1, 2)            # [1,H,M,Dh]
            k = k.view(B, M, H, Dh).transpose(1, 2)
            v = v.view(B, M, H, Dh).transpose(1, 2)
            # (2026-08-26) is_causal=True는 SDPA 내부에서 causal mask를 Where/Select로 적용하는데,
            # 이 상수가 onnx2tf dynamic-range 양자화에서 스케일이 깨지는 대상이 됨(디코더 step
            # wrapper의 torch.where와 동일 버그, QUANTIZATION_MOBILE.md 참고). 덧셈 방식의 명시적
            # causal bias로 대체해서 Where 노드 자체를 없앤다.
            causal_bias = torch.triu(torch.full((M, M), -1e9, device=q.device), diagonal=1)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_bias)
            attn = attn.transpose(1, 2).reshape(B * M, D)
            x = x + layer.dropout1(sa.out_proj(attn).view(B, M, D))

            x2 = layer.norm2(x)
            mha = layer.multihead_attn
            in_bc = mha.in_proj_bias
            S = memory.shape[1]
            x2_2d = x2.reshape(B * M, D)
            mem_2d = memory.reshape(B * S, D)
            qc = F.linear(x2_2d, mha.in_proj_weight[:D],    in_bc[:D]    if in_bc is not None else None).view(B, M, D)
            kc = F.linear(mem_2d, mha.in_proj_weight[D:2*D], in_bc[D:2*D] if in_bc is not None else None).view(B, S, D)
            vc = F.linear(mem_2d, mha.in_proj_weight[2*D:],  in_bc[2*D:]  if in_bc is not None else None).view(B, S, D)
            qc = qc.view(B, M, H, Dh).transpose(1, 2)
            kc = kc.view(B, S, H, Dh).transpose(1, 2)
            vc = vc.view(B, S, H, Dh).transpose(1, 2)
            attn2 = F.scaled_dot_product_attention(qc, kc, vc)
            attn2 = attn2.transpose(1, 2).reshape(B * M, D)
            x = x + layer.dropout2(mha.out_proj(attn2).view(B, M, D))

            x3_2d = layer.norm3(x).reshape(B * M, D)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x3_2d))))
            x = x + layer.dropout3(ff).view(B, M, D)

            pad_len = self.cache_len - M
            k_pad = F.pad(k, (0, 0, 0, pad_len))   # [1,H,cache_len,Dh]
            v_pad = F.pad(v, (0, 0, 0, pad_len))
            k_out_layers.append(k_pad)
            v_out_layers.append(v_pad)

        k_cache_out = torch.stack(k_out_layers, dim=0)
        v_cache_out = torch.stack(v_out_layers, dim=0)
        return k_cache_out, v_cache_out


def _export_onnx_bulk_capture(seq2seq: OmrSeq2Seq, out: str, chunk_len: int = 40,
                              cache_len: int = 300):
    """_BulkCaptureWrapperKV export. 모든 shape이 고정(chunk_len, cache_len)이라
    dynamic_axes 불필요 -- _export_onnx_decoder_kv와 동일한 이유로 TFLite 안전."""
    decoder = seq2seq.decoder
    wrapper = _BulkCaptureWrapperKV(decoder, chunk_len, cache_len).eval()
    dummy_tokens = torch.full((1, chunk_len), PAD_ID, dtype=torch.long)
    dummy_tokens[0, 0] = SOS_ID
    dummy_mem = torch.zeros(1, SEQ_LEN, 512)
    torch.onnx.export(
        wrapper, (dummy_tokens, dummy_mem), out,
        input_names=['tokens', 'memory'],
        output_names=['k_cache_out', 'v_cache_out'],
        opset_version=17,
        dynamo=False,
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


def _convert_tflite(onnx_path: str, tflite_path: str,
                    calib_npy: str | None, quantize: bool,
                    input_op_name: str = 'input',
                    keep_layout_input_names: list | None = None,
                    fp16: bool = False,
                    dynamic_range: bool = False) -> bool:
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
    elif fp16:
        # onnx2tf가 기본으로 float32와 함께 float16 변형도 항상 같이 만들어둠(부산물) --
        # 그중 float16만 골라 쓴다. 재학습·캘리브레이션 불필요, 단순 반정밀도 변환이라
        # 정확도 손실이 INT8보다 훨씬 적은 게 일반적.
        #
        # 주의(2026-08-25 확인): onnx2tf의 float16 변환은 "가중치만 fp16 + 활성값은 fp32"가
        # 아니라 그래프 전체(입력 텐서 포함)를 fp16으로 만든다 -- GPU 델리게이트 전용 설계라
        # CPU 인터프리터의 BATCH_MATMUL 커널이 fp16 입력을 거부해서 이 커스텀 attention
        # 그래프(디코더)에서는 런타임에 아예 실행이 안 된다(QUANTIZATION_MOBILE.md 참고).
        print("    Exporting as FP16")
        want_substrings = ('float16',)
    elif dynamic_range:
        # TFLite 표준 "dynamic range quantization" -- 가중치만 INT8로 저장하고 활성값은
        # 항상 FP32로 유지(캘리브레이션 데이터 불필요, 런타임에 가중치를 즉석 역양자화).
        # fp16과 달리 활성값이 그대로 FP32라 BATCH_MATMUL 등 CPU 커널이 못 받는 문제가 없다.
        #
        # 단, 이 최적화는 TFLite가 FULLY_CONNECTED/CONV_2D 등 특정 op으로 인식한 가중치만
        # 압축한다 -- 이 프로젝트의 디코더는 nn.Linear가 아니라 in_proj_weight를 수동
        # 슬라이싱한 F.linear로 attention을 구현해서 ONNX MatMul -> TFLite BATCH_MATMUL로
        # export되고, dynamic-range 양자화가 이 가중치를 건드리지 못한다(2026-08-25 확인,
        # 디코더 실측 130.7MB -> 130.5MB, 사실상 무압축). CNN인 인코더(Conv2D)는 정상
        # 압축된다(54.5MB -> 13.7MB). QUANTIZATION_MOBILE.md 참고.
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
    p.add_argument('--cache_len',        type=int, default=300,
                   help='디코더 self-attention KV캐시 고정 크기(최대 디코딩 스텝 수, INFER_MAX_LEN과 맞춤)')
    p.add_argument('--chunk_len',        type=int, default=40,
                   help='일괄 캐시 채우기 그래프의 고정 청크 길이(첫 마디 최대 토큰 수 상한)')
    p.add_argument('--fp16',             action='store_true',
                   help='FP32 대신 FP16 변형을 선택(재학습/캘리브레이션 불필요, 크기 약 절반) -- '
                        '이 커스텀 attention 그래프에서는 CPU 런타임에서 실행 자체가 안 됨(2026-08-26: '
                        '2D-flatten으로 Gemm/FULLY_CONNECTED 유도해도 동일 -- onnx2tf의 fp16 변환이 '
                        '가중치뿐 아니라 활성값까지 통째로 fp16으로 캐스팅해서 CPU 커널이 거부함,'
                        ' QUANTIZATION_MOBILE.md 참고), GPU 델리게이트 실험용으로만 남겨둠')
    p.add_argument('--dynamic_range',    action='store_true',
                   help='디코더+일괄캐시 그래프를 dynamic-range 양자화(가중치 INT8, 활성값 FP32,'
                        ' 캘리브레이션 불필요)로 export -- 인코더는 항상 FP32로 유지한다(CoordConv '
                        'CNN에서 dynamic-range 적용 시 출력이 수치적으로 발산하는 걸 확인, 2026-08-26,'
                        ' QUANTIZATION_MOBILE.md 참고). 디코더/일괄캐시는 2026-08-26에 attention 내부'
                        ' linear 연산을 3D→2D로 평탄화해 Gemm/FULLY_CONNECTED로 인식되게 고쳐서'
                        ' 정상 압축됨(130.7MB→35.3MB, 115.0MB→29.9MB)')
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

    if args.dynamic_range:
        print("    (--dynamic_range: 인코더는 CoordConv CNN에서 수치 발산이 확인돼 FP32로 고정 --"
              " QUANTIZATION_MOBILE.md 참고)")
    tflite_enc = os.path.join(tmp, 'encoder_INT8.tflite')
    if _convert_tflite(onnx_enc, tflite_enc, calib_enc, quantize, input_op_name='canvas',
                       fp16=args.fp16, dynamic_range=False):
        _save_versioned(tflite_enc, args.out_dir, 'encoder_INT8', args.version)
    print()

    # ── 3. Decoder (self-attention KV캐시, 고정 shape) ─────
    print("--- [3/3] Decoder ----------------------------------------")
    print(f"    Interface: (token_id[1,1], pos[1], memory[1,S,512], k_cache_in, v_cache_in)")
    print(f"               -> (next_logits[1,{vocab_size}], k_cache_out, v_cache_out)")
    print(f"    고정 크기 캐시(cache_len={args.cache_len}) -- 매 스텝 shape 동일, 리사이즈 불필요")
    print("    (2026-08-24: 이전 growing past_ids 방식은 실제 실행 시 2번째 스텝부터 TFLite")
    print("     reshape 에러로 깨지는 걸 확인해서 이 방식으로 교체함)")
    onnx_dec = os.path.join(tmp, 'decoder.onnx')
    _export_onnx_decoder_kv(seq2seq, onnx_dec, cache_len=args.cache_len)
    _simplify(onnx_dec)

    if args.quantize_decoder:
        print("    WARNING: --quantize_decoder는 아직 미구현 (decoder는 여러 입력을 함께 보정해야 "
              "해서 segnet/encoder와 같은 단일 캘리브레이션 방식으로 안 됨) -- FP32로 진행")
    else:
        print("    Quantization: FP32 (기본값, 자기회귀 decoder는 INT8 시 오류율 악화 위험)")

    tflite_dec = os.path.join(tmp, 'decoder_INT8.tflite')
    if _convert_tflite(onnx_dec, tflite_dec, None, False, input_op_name='token_id',
                       keep_layout_input_names=['token_id', 'pos', 'memory',
                                                 'k_cache_in', 'v_cache_in'],
                       fp16=args.fp16, dynamic_range=args.dynamic_range):
        _save_versioned(tflite_dec, args.out_dir, 'decoder_INT8', args.version)
    print()

    # ── 3b. Decoder 일괄 캐시 채우기(InlineTimeCorrector 호환용) ──
    print("--- [3b] Decoder bulk-capture ------------------------------")
    print(f"    Interface: (tokens[1,{args.chunk_len}], memory[1,S,512]) -> (k_cache_out, v_cache_out)")
    print("    첫 마디(InlineTimeCorrector 교정 구간)를 한 번에 캐시에 반영하는 용도")
    onnx_bulk = os.path.join(tmp, 'decoder_bulk.onnx')
    _export_onnx_bulk_capture(seq2seq, onnx_bulk, chunk_len=args.chunk_len,
                              cache_len=args.cache_len)
    _simplify(onnx_bulk)
    tflite_bulk = os.path.join(tmp, 'decoder_bulk_INT8.tflite')
    if _convert_tflite(onnx_bulk, tflite_bulk, None, False, input_op_name='tokens',
                       keep_layout_input_names=['tokens', 'memory'],
                       fp16=args.fp16, dynamic_range=args.dynamic_range):
        _save_versioned(tflite_bulk, args.out_dir, 'decoder_bulk_INT8', args.version)
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
