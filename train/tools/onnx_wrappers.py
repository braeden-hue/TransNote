"""onnx_wrappers.py — TFLite export용 nn.Module 래퍼 3종 (클래스 정의 전용).

export_tflite.py가 이 래퍼들을 torch.onnx.export()에 넣는다. 셋 다 입출력 shape이
완전히 고정이라 dynamic_axes가 필요 없다.

  _MemoryKVWrapper      cross-attention K,V 사전계산 (이미지당 1회)
  _DecoderStepWrapperKV self-attention KV캐시 기반 단일 스텝 디코더
  _BulkCaptureWrapperKV 고정 길이 청크를 한 번에 처리해 캐시를 일괄 재구성

**수정 전 EXPORT_NOTES.md를 반드시 읽을 것** — 아래 규칙들은 지키지 않으면 실제로 깨졌던
것들이고, 크래시 없이 조용히 NaN이 되는 경우도 있다:
  §3  고정 shape 유지(동적 슬라이싱 금지)
  §4  torch.where / is_causal=True 사용 금지 (양자화 스케일이 깨져 NaN)
  §5  linear는 2D로 펴서 호출 (Gemm/FULLY_CONNECTED 유도, 압축률 4배 차이)
  §6  cross-attention K,V는 캐시로 받기
  §7  왜 "한 위치 패치"가 아니라 "일괄 재계산"인지
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MemoryKVWrapper(nn.Module):
    """cross-attention 대상(memory)의 K,V를 레이어별로 한 번만 계산 — EXPORT_NOTES.md §6.

    memory[1, SEQ_LEN, 512] -> k_mem, v_mem [num_layers, 1, H, SEQ_LEN, Dh]
    """
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, memory):
        decoder = self.decoder
        D = decoder.embed_dim
        B, S, _ = memory.shape
        mem_2d = memory.reshape(B * S, D)   # 2D 평탄화 -- §5

        k_layers, v_layers = [], []
        for layer in decoder.transformer.layers:
            mha = layer.multihead_attn
            H = mha.num_heads
            Dh = D // H
            in_bc = mha.in_proj_bias
            kc = F.linear(mem_2d, mha.in_proj_weight[D:2*D], in_bc[D:2*D] if in_bc is not None else None).view(B, S, D)
            vc = F.linear(mem_2d, mha.in_proj_weight[2*D:], in_bc[2*D:] if in_bc is not None else None).view(B, S, D)
            k_layers.append(kc.view(B, S, H, Dh).transpose(1, 2))   # [B,H,S,Dh]
            v_layers.append(vc.view(B, S, H, Dh).transpose(1, 2))
        return torch.stack(k_layers, dim=0), torch.stack(v_layers, dim=0)


class _DecoderStepWrapperKV(nn.Module):
    """고정 크기 self-attention KV캐시 기반 단일 스텝 디코더 — EXPORT_NOTES.md §3.

    Inputs:
      token_id   : [1, 1]                            int64
      pos        : [1]                               int64  이 토큰의 0-index 위치(SOS=0)
      k_mem_in   : [num_layers, 1, H, SEQ_LEN, Dh]   float32  _MemoryKVWrapper 결과
      v_mem_in   : [num_layers, 1, H, SEQ_LEN, Dh]   float32
      k_cache_in : [num_layers, 1, H, cache_len, Dh] float32
      v_cache_in : [num_layers, 1, H, cache_len, Dh] float32

    Outputs:
      next_logits [1, vocab_size], k_cache_out, v_cache_out (입력 캐시와 동일 shape)
    """
    def __init__(self, decoder: nn.Module, cache_len: int):
        super().__init__()
        self.decoder = decoder
        self.cache_len = cache_len

    def forward(self, token_id, pos, k_mem_in, v_mem_in, k_cache_in, v_cache_in):
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
            x1_2d = x1.reshape(B * T, D)   # 2D 평탄화 -- §5
            in_b = sa.in_proj_bias
            q = F.linear(x1_2d, sa.in_proj_weight[:D],    in_b[:D]    if in_b is not None else None).view(B, T, D)
            k = F.linear(x1_2d, sa.in_proj_weight[D:2*D], in_b[D:2*D] if in_b is not None else None).view(B, T, D)
            v = F.linear(x1_2d, sa.in_proj_weight[2*D:],  in_b[2*D:]  if in_b is not None else None).view(B, T, D)
            q = q.view(B, 1, H, Dh).transpose(1, 2)              # [1,H,1,Dh]
            k = k.view(B, 1, H, Dh).transpose(1, 2)
            v = v.view(B, 1, H, Dh).transpose(1, 2)

            # 캐시 쓰기·마스킹 모두 torch.where 대신 산술 연산 -- §4 (Where 노드가 생기면 NaN)
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
            x2_2d = x2.reshape(B * 1, D)
            kc, vc = k_mem_in[li], v_mem_in[li]   # 사전계산된 cross-attn K,V -- §6
            qc = F.linear(x2_2d, mha.in_proj_weight[:D], in_bc[:D] if in_bc is not None else None).view(B, 1, D)
            qc = qc.view(B, 1, H, Dh).transpose(1, 2)
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
        return logits, torch.stack(k_out_layers, dim=0), torch.stack(v_out_layers, dim=0)


class _BulkCaptureWrapperKV(nn.Module):
    """고정 길이 청크를 한 번에 처리해 self-attention 캐시를 일괄 재구성 — EXPORT_NOTES.md §7.

    InlineTimeCorrector로 교정된 첫 마디를 _DecoderStepWrapperKV로 넘어가기 전에 반영한다
    ("한 위치만 패치"는 원리적으로 불가능 — §7). chunk_len보다 짧으면 뒤는 PAD_ID로 채운다.

    Inputs:
      tokens   : [1, chunk_len]                     int64  SOS 포함, 뒤쪽은 PAD_ID
      k_mem_in : [num_layers, 1, H, SEQ_LEN, Dh]    float32
      v_mem_in : [num_layers, 1, H, SEQ_LEN, Dh]    float32

    Outputs:
      k_cache_out, v_cache_out : [num_layers, 1, H, cache_len, Dh]
        0..chunk_len-1 위치가 채워지고 나머지는 0 패딩. 이후 스텝 그래프가 이어서 씀.
    """
    def __init__(self, decoder: nn.Module, chunk_len: int, cache_len: int):
        super().__init__()
        self.decoder = decoder
        self.chunk_len = chunk_len
        self.cache_len = cache_len

    def forward(self, tokens, k_mem_in, v_mem_in):
        decoder = self.decoder
        D = decoder.embed_dim
        M = self.chunk_len
        B = tokens.shape[0]

        emb = decoder.token_emb(tokens) * decoder.emb_scale     # [1,M,D]
        x = emb + decoder.pos_enc[:, :M, :]

        k_out_layers, v_out_layers = [], []
        for li, layer in enumerate(decoder.transformer.layers):
            x1 = layer.norm1(x)
            sa = layer.self_attn
            H = sa.num_heads
            Dh = D // H
            x1_2d = x1.reshape(B * M, D)   # 2D 평탄화 -- §5
            in_b = sa.in_proj_bias
            q = F.linear(x1_2d, sa.in_proj_weight[:D],    in_b[:D]    if in_b is not None else None).view(B, M, D)
            k = F.linear(x1_2d, sa.in_proj_weight[D:2*D], in_b[D:2*D] if in_b is not None else None).view(B, M, D)
            v = F.linear(x1_2d, sa.in_proj_weight[2*D:],  in_b[2*D:]  if in_b is not None else None).view(B, M, D)
            q = q.view(B, M, H, Dh).transpose(1, 2)            # [1,H,M,Dh]
            k = k.view(B, M, H, Dh).transpose(1, 2)
            v = v.view(B, M, H, Dh).transpose(1, 2)
            # is_causal=True 대신 명시적 덧셈 마스크 -- §4 (내부 Where 노드가 생기면 NaN)
            causal_bias = torch.triu(torch.full((M, M), -1e9, device=q.device), diagonal=1)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_bias)
            attn = attn.transpose(1, 2).reshape(B * M, D)
            x = x + layer.dropout1(sa.out_proj(attn).view(B, M, D))

            x2 = layer.norm2(x)
            mha = layer.multihead_attn
            in_bc = mha.in_proj_bias
            x2_2d = x2.reshape(B * M, D)
            kc, vc = k_mem_in[li], v_mem_in[li]   # 사전계산된 cross-attn K,V -- §6
            qc = F.linear(x2_2d, mha.in_proj_weight[:D], in_bc[:D] if in_bc is not None else None).view(B, M, D)
            qc = qc.view(B, M, H, Dh).transpose(1, 2)
            attn2 = F.scaled_dot_product_attention(qc, kc, vc)
            attn2 = attn2.transpose(1, 2).reshape(B * M, D)
            x = x + layer.dropout2(mha.out_proj(attn2).view(B, M, D))

            x3_2d = layer.norm3(x).reshape(B * M, D)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x3_2d))))
            x = x + layer.dropout3(ff).view(B, M, D)

            pad_len = self.cache_len - M
            k_out_layers.append(F.pad(k, (0, 0, 0, pad_len)))   # [1,H,cache_len,Dh]
            v_out_layers.append(F.pad(v, (0, 0, 0, pad_len)))

        return torch.stack(k_out_layers, dim=0), torch.stack(v_out_layers, dim=0)
