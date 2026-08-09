"""
model.py  –  OMR neural network architectures.

Three models (all built from PyTorch primitives, BSD-3-Clause):

1. SegNet
   U-Net style 6-class pixel segmentation network.
   Input:  [B, 1, 320, 320]  (grayscale patch)
   Output: [B, 6, 320, 320]  (class logits)

2. Encoder
   Convolutional backbone that maps a staff canvas to a latent sequence.
   Input:  [B, 1, CANVAS_H, 1280]   (CANVAS_H × CANVAS_W, single channel; height is
                                     adaptively pooled away, so CANVAS_H is not fixed here)
   Output: [B, 320, 512]        (seq_len=320, embed_dim=512)
   Normalisation: (pixel/255 − 0.7931) / 0.1738  (must match training)

3. Decoder
   Autoregressive Transformer decoder.
   Input:  memory [B, enc_seq, 512]  +  tgt_ids [B, tgt_seq]
   Output: logits  [B, tgt_seq, vocab_size]  (teacher-forcing, training)
   Architecture: 8 layers, 8 heads, d_model=512, ff_dim=2048, pre-LN

4. OmrSeq2Seq
   Thin wrapper combining Encoder + Decoder for end-to-end training.

Architecture references (all public-domain or BSD/MIT):
  U-Net:       Ronneberger et al. (2015) MICCAI. (MIT)
  Transformer: Vaswani et al. (2017) NeurIPS. (public domain)
  Pre-LN:      Wang et al. (2019) "Learning Deep Transformer Models for
               Machine Translation." ACL. (open)
  Sin. PE:     Vaswani et al. (2017) §3.5.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
#  Constants (must match C++ engine and dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

CANVAS_H    = 320
CANVAS_W    = 1280
SEQ_LEN     = 320       # CANVAS_W // 4 = encoder output width
EMBED_DIM   = 512       # d_model = NUM_HEADS × HEAD_DIM
NUM_HEADS   = 8
HEAD_DIM    = 64        # EMBED_DIM / NUM_HEADS
NUM_LAYERS  = 8         # Transformer decoder depth
FF_DIM      = 2048      # FFN hidden dim (4 × EMBED_DIM)
NUM_CLASSES = 6         # SegNet output classes
PATCH_SIZE  = 320       # SegNet input patch
VOCAB_SIZE  = 258       # Round 3 vocabulary (note token split into pitch-only + dur-only, was 1013)
MAX_SEQ     = 608       # Max decoder steps
PAD_ID      = 0
SOS_ID      = 1
EOS_ID      = 2


# ─────────────────────────────────────────────────────────────────────────────
#  Shared building block
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNGELU(nn.Module):
    """Conv2d → BatchNorm2d → GELU (standard feed-forward block)."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride=1, padding: int = 1,
                 groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
#  1. SegNet  (U-Net, 4-level encoder-decoder)
# ─────────────────────────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c = nn.Sequential(
            ConvBNGELU(in_ch,  out_ch),
            ConvBNGELU(out_ch, out_ch),
        )
    def forward(self, x): return self.c(x)


class SegNet(nn.Module):
    """
    U-Net style 6-class semantic segmentation for music symbols.

    Input  : [B, 1, 320, 320]   single-channel grayscale patch
    Output : [B, 6, 320, 320]   per-pixel class logits

    Channel progression: 1 → 32 → 64 → 128 → 256 → 512 (bottleneck)
    Each encoder level halves spatial resolution via max-pool.
    Each decoder level doubles via transposed convolution + skip concat.

    Ref: Ronneberger O. et al. (2015) "U-Net." MICCAI. MIT licence.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, base_ch: int = 32):
        super().__init__()
        b = base_ch

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = _DoubleConv(1,     b)      # [B,  b, 320, 320]
        self.enc2 = _DoubleConv(b,  2*b)       # [B, 2b, 160, 160]
        self.enc3 = _DoubleConv(2*b, 4*b)      # [B, 4b,  80,  80]
        self.enc4 = _DoubleConv(4*b, 8*b)      # [B, 8b,  40,  40]
        self.bot  = _DoubleConv(8*b, 16*b)     # [B,16b,  20,  20]
        self.pool = nn.MaxPool2d(2, 2)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up4  = nn.ConvTranspose2d(16*b, 8*b, 2, stride=2)
        self.dec4 = _DoubleConv(16*b, 8*b)     # 8b + 8b skip

        self.up3  = nn.ConvTranspose2d(8*b, 4*b, 2, stride=2)
        self.dec3 = _DoubleConv(8*b,  4*b)

        self.up2  = nn.ConvTranspose2d(4*b, 2*b, 2, stride=2)
        self.dec2 = _DoubleConv(4*b,  2*b)

        self.up1  = nn.ConvTranspose2d(2*b, b, 2, stride=2)
        self.dec1 = _DoubleConv(2*b,  b)

        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)                        # [B,  b, 320, 320]
        s2 = self.enc2(self.pool(s1))            # [B, 2b, 160, 160]
        s3 = self.enc3(self.pool(s2))            # [B, 4b,  80,  80]
        s4 = self.enc4(self.pool(s3))            # [B, 8b,  40,  40]
        b  = self.bot(self.pool(s4))             # [B,16b,  20,  20]

        x  = self.dec4(torch.cat([self.up4(b),  s4], 1))
        x  = self.dec3(torch.cat([self.up3(x),  s3], 1))
        x  = self.dec2(torch.cat([self.up2(x),  s2], 1))
        x  = self.dec1(torch.cat([self.up1(x),  s1], 1))

        return self.head(x)                      # [B, num_classes, 320, 320]


# ─────────────────────────────────────────────────────────────────────────────
#  2. Encoder  (strided CNN → positional sequence)
# ─────────────────────────────────────────────────────────────────────────────

def _make_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding.
    Ref: Vaswani et al. (2017) §3.5.
    """
    pos   = torch.arange(seq_len).unsqueeze(1).float()          # [L, 1]
    dim   = torch.arange(0, d_model, 2).float()                 # [D/2]
    denom = torch.pow(10000.0, dim / d_model)                   # [D/2]
    pe    = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos / denom)
    pe[:, 1::2] = torch.cos(pos / denom)
    return pe                                                     # [L, D]


class Encoder(nn.Module):
    """
    CNN encoder: [B, 1, CANVAS_H, CANVAS_W] → [B, SEQ_LEN, EMBED_DIM]

    Architecture:
      - 8 ConvBNGELU stages with mixed strides:
          Stage 1   : stride (2,2) → H/2, W/2    [B, 64, 128, 640]
          Stages 2-8: stride (2,1) → H/2, W same  (height collapses to 1)
      - AdaptiveAvgPool2d((1, SEQ_LEN)) → exact width normalisation
      - Squeeze height → [B, 512, SEQ_LEN]
      - Permute → [B, SEQ_LEN, 512]
      - Fixed sinusoidal positional encoding added

    Output seq_len = 320 = CANVAS_W / 4.
    """

    def __init__(self,
                 in_ch:     int = 1,
                 embed_dim: int = EMBED_DIM,
                 seq_len:   int = SEQ_LEN,
                 extra_height_stages: int = 4,
                 pool_h: int = 1):
        """
        extra_height_stages / pool_h (2026-07-31, "높이 부분 보존" 실험): 기존엔 세로를
        8단계(초기 성장단계 4번 + extra_height_stages=4번) 만에 완전히 1로 뭉개서
        AdaptiveAvgPool2d((1, seq_len))로 넘겼는데, 이러면 CoordConv처럼 좌표 신호를
        따로 넣어도 그 신호가 8개 conv를 통과하며 "학습으로 살아남아야" 해서 간접적이고,
        실측(89곡 3도 오독 273건, CoordConv 단독 235건)으로도 완전히 안 풀렸음.
        extra_height_stages를 줄이면(기본 4->2) backbone 출력 단계에서 세로가 아직
        여러 줄 남은 채로 pool로 넘어가고, pool_h>1로 그 중 일부를 실제로 보존한다
        (좌표를 "학습해서 기억"하는 게 아니라 구조적으로 유지) -- proj 레이어 입력 차원만
        바뀌므로(512 -> 512*pool_h) 그 레이어만 재초기화되고 나머지(backbone 앞부분,
        디코더 전체)는 체크포인트 그대로 이어받을 수 있다.
        기본값(4, 1)은 기존과 완전히 동일한 동작.
        """
        super().__init__()
        self.seq_len   = seq_len
        self.embed_dim = embed_dim
        self.pool_h    = pool_h

        # ── CNN backbone ─────────────────────────────────────────────────────
        stages = [
            # Stage 1: reduce both H and W by 2.
            ConvBNGELU(in_ch, 64,  3, stride=(2, 2), padding=1),
            ConvBNGELU(64,    64,  3, stride=1,       padding=1),

            # Stages 2-4: reduce H only.
            ConvBNGELU(64,  128, 3, stride=(2, 1), padding=1),
            ConvBNGELU(128, 128, 3, stride=1,      padding=1),

            ConvBNGELU(128, 256, 3, stride=(2, 1), padding=1),
            ConvBNGELU(256, 256, 3, stride=1,      padding=1),

            ConvBNGELU(256, 512, 3, stride=(2, 1), padding=1),
            ConvBNGELU(512, 512, 3, stride=1,      padding=1),
        ]
        # 나머지 세로 축소 단계(기존엔 항상 4개 고정 -> 총 8단계로 H를 1까지 완전히 뭉갬).
        for _ in range(extra_height_stages):
            stages.append(ConvBNGELU(512, 512, 3, stride=(2, 1), padding=1))
        self.backbone = nn.Sequential(*stages)
        # extra_height_stages=4(기본): H=1, W≈640. extra_height_stages=2: H가 아직
        # 여러 줄 남은 채로 pool에 전달됨(예: CANVAS_H=320 -> 약 5줄).

        # 세로를 pool_h(기본 1=기존과 동일)로, 가로는 정확히 seq_len으로.
        self.pool = nn.AdaptiveAvgPool2d((pool_h, seq_len))

        # 입력 차원 = 512 * pool_h. pool_h=1(기본)이면 기존과 동일(512==embed_dim이면 no-op).
        proj_in = 512 * pool_h
        self.proj = (nn.Linear(proj_in, embed_dim, bias=False)
                     if proj_in != embed_dim else nn.Identity())

        # Sinusoidal PE (fixed, not learned).
        pe = _make_sinusoidal_pe(seq_len, embed_dim)
        self.register_buffer('pos_enc', pe.unsqueeze(0))  # [1, seq_len, D]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, CANVAS_H, CANVAS_W] → [B, seq_len, embed_dim]"""
        feat = self.backbone(x)          # [B, 512, ~pool_h*k, ~640]
        feat = self.pool(feat)           # [B, 512, pool_h, seq_len]
        if self.pool_h == 1:
            feat = feat.squeeze(2)              # [B, 512, seq_len]
            feat = feat.permute(0, 2, 1)         # [B, seq_len, 512]
        else:
            # 세로 pool_h개 밴드를 채널에 이어붙여 가로 위치별 벡터에 세로 위치 정보를
            # 구조적으로(학습 없이) 보존한다 -- [B, 512, pool_h, seq_len] -> [B, seq_len, 512*pool_h].
            feat = feat.permute(0, 3, 1, 2)      # [B, seq_len, 512, pool_h]
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)  # [B, seq_len, 512*pool_h]
        feat = self.proj(feat)           # [B, seq_len, embed_dim]
        feat = feat + self.pos_enc       # add sinusoidal PE
        return feat


# ─────────────────────────────────────────────────────────────────────────────
#  3. Decoder  (Transformer, teacher-forcing training mode)
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Autoregressive Transformer decoder.

    Training (teacher forcing):
      tgt    : [B, T]   int64 token IDs = [SOS, t1, ..., t_{n-1}]
      memory : [B, S, D] encoder output
      returns: [B, T, vocab_size] logits

    Architecture:
      - Learned token embedding (vocab_size × embed_dim)
      - Sinusoidal positional encoding
      - num_layers × TransformerDecoderLayer (pre-LN, batch_first)
      - LayerNorm output
      - Linear projection → vocab_size logits
      - Weight tying: embedding weights shared with output projection

    Ref: Vaswani et al. (2017) NeurIPS.  Pre-LN: Wang et al. (2019) ACL.
    nn.TransformerDecoder: PyTorch BSD-3-Clause.
    """

    def __init__(self,
                 vocab_size: int   = VOCAB_SIZE,
                 embed_dim:  int   = EMBED_DIM,
                 num_heads:  int   = NUM_HEADS,
                 num_layers: int   = NUM_LAYERS,
                 ff_dim:     int   = FF_DIM,
                 max_seq:    int   = MAX_SEQ,
                 dropout:    float = 0.1):
        super().__init__()
        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size

        # Token embedding.
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.emb_scale = math.sqrt(embed_dim)

        # Sinusoidal PE.
        pe = _make_sinusoidal_pe(max_seq + 2, embed_dim)
        self.register_buffer('pos_enc', pe.unsqueeze(0))  # [1, max_seq+2, D]

        # Transformer decoder (pre-LN for better training stability).
        layer = nn.TransformerDecoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,    # pre-LN
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer = layer,
            num_layers    = num_layers,
            norm          = nn.LayerNorm(embed_dim, eps=1e-6),
        )

        # Output projection (weight-tied to embedding).
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight     # weight tying

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        # head.weight is tied, no separate init needed.
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.head:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                tgt:    torch.Tensor,           # [B, T] int64
                memory: torch.Tensor,           # [B, S, D] float32
                tgt_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] bool
                ) -> torch.Tensor:
        """Teacher-forcing forward. Returns logits [B, T, vocab_size]."""
        T = tgt.size(1)

        # Embedding + positional encoding.
        emb = self.token_emb(tgt) * self.emb_scale         # [B, T, D]
        emb = self.dropout(emb + self.pos_enc[:, :T, :])

        # Causal mask: position i cannot attend to j > i.
        causal = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt.device, dtype=emb.dtype)          # [T, T]

        out = self.transformer(
            tgt                  = emb,
            memory               = memory,
            tgt_mask             = causal,
            tgt_key_padding_mask = tgt_key_padding_mask,
        )                                                    # [B, T, D]

        return self.head(out)                                # [B, T, vocab_size]

    def precompute_memory_kv(self, memory: torch.Tensor):
        """Cross-attention 대상(memory)의 K,V를 레이어별로 한 번만 계산해서 캐싱한다.
        memory(인코더 출력)는 디코딩 내내 고정이므로, decode_step마다 매번
        재계산되던 K,V projection을 건너뛸 수 있다. 추론 전용 최적화 — forward()의
        학습 경로는 그대로 두고 self-attention/FFN도 원본과 동일하게 재계산한다."""
        B, S, D = memory.shape
        H  = self.transformer.layers[0].multihead_attn.num_heads
        Dh = D // H
        cache = []
        for layer in self.transformer.layers:
            mha  = layer.multihead_attn
            in_b = mha.in_proj_bias
            k = F.linear(memory, mha.in_proj_weight[D:2*D],
                         in_b[D:2*D] if in_b is not None else None)
            v = F.linear(memory, mha.in_proj_weight[2*D:],
                         in_b[2*D:] if in_b is not None else None)
            k = k.view(B, S, H, Dh).transpose(1, 2)   # [B, H, S, Dh]
            v = v.view(B, S, H, Dh).transpose(1, 2)
            cache.append((k, v))
        return cache

    def forward_cached(self, tgt: torch.Tensor, memory_kv_cache) -> torch.Tensor:
        """forward()와 동일한 계산 (bit-identical 결과), 다만 cross-attention의
        K,V는 precompute_memory_kv()로 미리 계산된 캐시를 재사용한다."""
        T = tgt.size(1)
        D = self.embed_dim
        emb = self.token_emb(tgt) * self.emb_scale
        x   = self.dropout(emb + self.pos_enc[:, :T, :])
        causal = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt.device, dtype=x.dtype)

        for layer, (k_mem, v_mem) in zip(self.transformer.layers, memory_kv_cache):
            x = x + layer._sa_block(layer.norm1(x), causal, None)

            x2   = layer.norm2(x)
            mha  = layer.multihead_attn
            H    = mha.num_heads
            Dh   = D // H
            B, Tt, _ = x2.shape
            in_b = mha.in_proj_bias
            q = F.linear(x2, mha.in_proj_weight[:D], in_b[:D] if in_b is not None else None)
            q = q.view(B, Tt, H, Dh).transpose(1, 2)          # [B, H, Tt, Dh]
            attn = F.scaled_dot_product_attention(q, k_mem, v_mem)
            attn = attn.transpose(1, 2).reshape(B, Tt, D)
            x = x + layer.dropout2(mha.out_proj(attn))

            x = x + layer._ff_block(layer.norm3(x))

        x = self.transformer.norm(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
#  4. Combined Seq2Seq model
# ─────────────────────────────────────────────────────────────────────────────

class OmrSeq2Seq(nn.Module):
    """
    Encoder + Decoder combined model for end-to-end training.
    SegNet is trained separately (Phase 1) and is NOT included here.

    Phase 2: train this module with SegNet frozen (or bypassed).
    Phase 3: joint fine-tuning (attach SegNet externally if desired).
    """

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim:  int = EMBED_DIM,
                 in_ch:      int = 1,
                 extra_height_stages: int = 4,
                 pool_h: int = 1):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim, in_ch=in_ch,
                                extra_height_stages=extra_height_stages, pool_h=pool_h)
        self.decoder = Decoder(vocab_size=vocab_size, embed_dim=embed_dim)

    def forward(self,
                canvas:              torch.Tensor,           # [B, 1, H, W]
                tgt_ids:             torch.Tensor,           # [B, T] int64
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Returns logits [B, T, vocab_size]."""
        memory = self.encoder(canvas)
        return self.decoder(tgt_ids, memory, tgt_key_padding_mask)

    def encode(self, canvas: torch.Tensor) -> torch.Tensor:
        """Encode only (for inference)."""
        return self.encoder(canvas)

    def decode_step(self,
                    token_id: torch.Tensor,   # [B, 1] int64
                    memory:   torch.Tensor,   # [B, S, D]
                    past_ids: torch.Tensor,   # [B, t] all tokens so far (incl. SOS)
                    ) -> torch.Tensor:
        """
        Single greedy-decoding step (inference, no KV-cache).
        Returns logits [B, vocab_size] for the next token.
        """
        tgt    = past_ids                                    # [B, t]
        logits = self.decoder(tgt, memory)                  # [B, t, vocab_size]
        return logits[:, -1, :]                              # [B, vocab_size]

    def precompute_memory_kv(self, memory: torch.Tensor):
        """decode_step_cached에서 쓸 cross-attention K,V 캐시를 준비한다 (encode() 직후 1회 호출)."""
        return self.decoder.precompute_memory_kv(memory)

    def decode_step_cached(self,
                           memory_kv_cache,
                           past_ids: torch.Tensor,   # [B, t] all tokens so far (incl. SOS)
                           ) -> torch.Tensor:
        """decode_step과 bit-identical한 결과를 내는 가속 버전. cross-attention의
        K,V를 매 스텝 재계산하지 않고 precompute_memory_kv() 캐시를 재사용한다
        (self-attention/FFN은 원본과 동일하게 매 스텝 전체 재계산 — Step1 범위)."""
        logits = self.decoder.forward_cached(past_ids, memory_kv_cache)
        return logits[:, -1, :]


# ─────────────────────────────────────────────────────────────────────────────
#  Focal Loss for SegNet (handles heavy class imbalance)
#
#  Ref: Lin T.Y. et al. (2017) "Focal Loss for Dense Object Detection."
#       ICCV. (BSD-style research licence; formula is public domain)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss = -(1-p_t)^gamma × log(p_t)
    Downweights easy (well-classified) examples; focuses on hard ones.

    gamma=0 recovers standard cross-entropy.
    alpha: per-class weight tensor [num_classes] or scalar.
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 ignore_index: int = -100,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.ignore_index    = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, C, H, W] or [N, C]  unnormalised
        targets : [B, H, W]    or [N]     int64 class indices
        (F.cross_entropy accepts either shape, so this doubles as a
        per-token sequence loss when given [N, C]/[N].)
        """
        # Standard cross-entropy gives log p_t per pixel/token.
        ce = F.cross_entropy(logits, targets,
                              weight=self.alpha,
                              ignore_index=self.ignore_index,
                              label_smoothing=self.label_smoothing,
                              reduction='none')    # [B, H, W] or [N]

        # Probability of the correct class.
        with torch.no_grad():
            p_t = torch.exp(-ce)

        # Focal weight.
        loss = ((1.0 - p_t) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Model factory / parameter count helper
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"total={total/1e6:.1f}M  trainable={train/1e6:.1f}M"


def infer_arch_from_state_dict(state_dict: dict) -> dict:
    """체크포인트 state_dict만 보고 in_ch/extra_height_stages/pool_h를 역산한다
    (2026-07-31, "높이 부분 보존" 실험 도입 후 -- 여러 평가 스크립트가 각자
    in_ch만 감지하던 걸 pool_h까지 포함해 한 곳에서 처리하도록 통합).
    - in_ch: 첫 conv 레이어 입력 채널 수(1=기존, 2=CoordConv).
    - backbone 길이: encoder.backbone.{i}...가 몇 개까지 있는지 세서
      extra_height_stages = (길이 - 8)로 역산(기본 backbone은 8+4=12개).
    - pool_h: encoder.proj.weight가 있으면 그 입력 차원을 512로 나눔.
      proj가 Identity(가중치 없음)면 1(기본, 512==embed_dim일 때).
    """
    in_ch = state_dict['encoder.backbone.0.block.0.weight'].shape[1]

    backbone_len = 0
    while f'encoder.backbone.{backbone_len}.block.0.weight' in state_dict:
        backbone_len += 1
    extra_height_stages = max(0, backbone_len - 8)

    if 'encoder.proj.weight' in state_dict:
        pool_h = state_dict['encoder.proj.weight'].shape[1] // 512
    else:
        pool_h = 1

    return {'in_ch': in_ch, 'extra_height_stages': extra_height_stages, 'pool_h': pool_h}


def build_models(vocab_size: int = VOCAB_SIZE,
                 embed_dim:  int = EMBED_DIM
                 ) -> tuple[SegNet, OmrSeq2Seq]:
    """Build and return (segnet, seq2seq) with default hyperparameters."""
    segnet  = SegNet(num_classes=NUM_CLASSES)
    seq2seq = OmrSeq2Seq(vocab_size=vocab_size, embed_dim=embed_dim)
    print(f"SegNet   params: {count_params(segnet)}")
    print(f"Encoder  params: {count_params(seq2seq.encoder)}")
    print(f"Decoder  params: {count_params(seq2seq.decoder)}")
    return segnet, seq2seq
