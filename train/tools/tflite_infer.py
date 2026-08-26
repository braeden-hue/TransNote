"""tflite_infer.py — TFLite로 export한 인코더+디코더만으로(PyTorch 모델 로드 없이) 실제
악보 이미지를 인식하는 CLI 스크립트.

모바일 앱 없이도 "TFLite 모델이 실제 인터프리터에서 완전하게 동작하는지"를 증명하는 게
목적이다(train/QUANTIZATION_MOBILE.md ① 항목). export_tflite.py가 만든 6개 파일을 그대로
쓴다 -- Hybrid 구성(2026-08-26)은 인코더/memkv는 FP32, decoder 계열은 dynamic-range(INT8
가중치)다. 버킷팅(2026-08-26, EXPORT_NOTES.md §14)으로 decoder_INT8/decoder_bulk_INT8은
작은 KV캐시(기본 큼), decoder_large_INT8/decoder_rebucket_INT8은 넘칠 때만 쓰는 큰 캐시로
나뉜다 -- decode_bucketed()가 이 전환을 처리한다.

디코더는 고정 크기 self-attention KV캐시 인터페이스(_export_onnx_decoder_kv 참고)라
매 스텝 텐서 shape이 동일 — resize_tensor_input이 처음 한 번만 필요하고 이후 스텝마다는
필요 없다(예전 growing past_ids 방식은 매 스텝 리사이즈가 필요해서 2번째 스텝부터
크래시했었다).

사용법:
    python train/tools/tflite_infer.py --tflite_dir train/tflite_export_dr --image <악보사진.jpg>
"""
import argparse
import os
import sys
import time
from pathlib import Path

_TRAIN = Path(__file__).resolve().parent.parent   # train/ (런타임 모듈·체크포인트가 있는 곳)
sys.path.insert(0, str(_TRAIN))

import numpy as np
import tensorflow as tf

from dataset import (load_preprocessed, best_effort_staff_detection, extract_system_canvas,
                     IMG_MEAN, IMG_STD, make_model_input, load_tokenizer)
from inference import (EOS_BOOST, LONG_DECODE_THRESHOLD, LONG_DECODE_RAMP,
                       fix_chord_tokens, fix_span_tokens, correct_time_signature,
                       correct_accidentals_by_key, InlineTimeCorrector, _BARLINE_TOKEN_STRS)

PAD_ID, SOS_ID, EOS_ID = 0, 1, 2


class TFLiteOmrModel:
    """export_tflite.py가 만든 6개 파일(encoder/memkv/decoder 4종)로 동작하는 추론 래퍼 --
    버킷팅(§14) 파일이 없는 옛 export 디렉토리도 자동 폴백으로 지원."""

    def __init__(self, tflite_dir: str, chunk_len: int = 40,
                 enc_threads: int | None = None, dec_threads: int = 1):
        """enc_threads/dec_threads (2026-08-26 실측): 인코더(이미지당 1회, 큰 CNN)는
        스레드가 많을수록 빨라지지만(이 개발 PC 18코어 기준 16스레드까지 계속 개선), 디코더는
        정반대 — 스텝마다 토큰 1개짜리 아주 작은 연산이라 스레드 동기화 오버헤드가 병렬화
        이득보다 커서 스레드를 늘릴수록 오히려 느려진다(4곡 평균 실측: dec_threads=1이 2,3,4
        보다 항상 빠름). enc_threads 기본값은 머신의 전체 코어 수(os.cpu_count()), dec_threads
        기본값은 1 — 이 비대칭성은 "작은 연산은 스레드 오버헤드가 이득을 압도한다"는 일반적
        원리라 코어 수가 다른 기기에서도 유효할 것으로 예상(다만 실기 재검증 필요, ③ 참고)."""
        if enc_threads is None:
            enc_threads = os.cpu_count() or 4
        d = Path(tflite_dir)
        self.enc = tf.lite.Interpreter(model_path=str(d / "encoder_INT8.tflite"), num_threads=enc_threads)
        # memkv(cross-attention K,V 사전계산, 2026-08-26)는 인코더처럼 이미지당 1회만 도는
        # 큰 연산(S=SEQ_LEN 전체)이라 enc_threads를 같이 씀.
        self.memkv = tf.lite.Interpreter(model_path=str(d / "decoder_memkv_INT8.tflite"), num_threads=enc_threads)
        # 버킷팅(2026-08-26, EXPORT_NOTES.md §14): decoder_INT8/decoder_bulk_INT8은 작은
        # 버킷(기본으로 씀), decoder_large_INT8/decoder_rebucket_INT8은 넘칠 때만 쓰는
        # 안전판. 옛 단일-버킷 파일 레이아웃(decoder_large_INT8.tflite 없음)과도 호환되게
        # 없으면 작은 버킷 파일을 그대로 재사용한다(리버킷은 이 경우 의미가 없어짐).
        self.dec = tf.lite.Interpreter(model_path=str(d / "decoder_INT8.tflite"), num_threads=dec_threads)
        self.bulk = tf.lite.Interpreter(model_path=str(d / "decoder_bulk_INT8.tflite"), num_threads=dec_threads)
        large_path = d / "decoder_large_INT8.tflite"
        rebucket_path = d / "decoder_rebucket_INT8.tflite"
        self._bucketed = large_path.exists() and rebucket_path.exists()
        if self._bucketed:
            self.dec_large = tf.lite.Interpreter(model_path=str(large_path), num_threads=dec_threads)
            self.rebucket = tf.lite.Interpreter(model_path=str(rebucket_path), num_threads=dec_threads)
        else:
            self.dec_large = self.dec   # 버킷팅 이전 export 디렉토리 호환용 폴백
            self.rebucket = None
        self.enc.allocate_tensors()
        self.memkv.allocate_tensors()
        self.dec.allocate_tensors()
        self.bulk.allocate_tensors()
        if self._bucketed:
            self.dec_large.allocate_tensors()
            self.rebucket.allocate_tensors()
        self.enc_in = self.enc.get_input_details()[0]
        self.enc_out_idx = self.enc.get_output_details()[0]['index']
        self.memkv_ins = {t['name']: t['index'] for t in self.memkv.get_input_details()}
        self.memkv_outs = {t['name']: t['index'] for t in self.memkv.get_output_details()}
        self.dec_ins = {t['name']: t['index'] for t in self.dec.get_input_details()}
        self.dec_outs = {t['name']: t['index'] for t in self.dec.get_output_details()}
        self.bulk_ins = {t['name']: t['index'] for t in self.bulk.get_input_details()}
        self.bulk_outs = {t['name']: t['index'] for t in self.bulk.get_output_details()}
        self.dec_large_ins = {t['name']: t['index'] for t in self.dec_large.get_input_details()}
        self.dec_large_outs = {t['name']: t['index'] for t in self.dec_large.get_output_details()}
        if self.rebucket is not None:
            self.rebucket_ins = {t['name']: t['index'] for t in self.rebucket.get_input_details()}
            self.rebucket_outs = {t['name']: t['index'] for t in self.rebucket.get_output_details()}
        self.chunk_len = chunk_len
        self.in_ch = self.enc_in['shape'][-1]  # NHWC라 채널이 마지막 차원

        k_shape = self.dec.get_tensor_details()[self.dec_ins['k_cache_in']]['shape']
        self.num_layers, _, self.num_heads, _, self.head_dim = k_shape
        self.small_cache_len = k_shape[3]
        kl_shape = self.dec_large.get_tensor_details()[self.dec_large_ins['k_cache_in']]['shape']
        self.large_cache_len = kl_shape[3]
        self.cache_len = self.small_cache_len   # 하위 호환(decode()/decode_hybrid()가 참조)

    def precompute_memory_kv(self, memory: np.ndarray):
        """cross-attention K,V를 이미지당 1회 계산(2026-08-26) -- production PyTorch의
        precompute_memory_kv()와 동일 목적. decode_hybrid()가 디코딩 루프 시작 전 1번만
        호출하고, 이후 모든 _dec_step()/bulk 호출에 그 결과(k_mem,v_mem)를 넘긴다."""
        self.memkv.set_tensor(self.memkv_ins['memory'], memory.astype(np.float32))
        self.memkv.invoke()
        k_mem = self.memkv.get_tensor(self.memkv_outs['k_mem'])
        v_mem = self.memkv.get_tensor(self.memkv_outs['v_mem'])
        return k_mem, v_mem

    def encode(self, canvas_norm: np.ndarray) -> np.ndarray:
        """canvas_norm: [H,W] float32, 이미 (px/255-mean)/std 정규화 완료.

        입력 shape은 (SYSTEM_CANVAS_H, CANVAS_W, in_ch)로 항상 고정이다(extract_system_canvas가
        매번 같은 캔버스 크기를 반환) -- __init__의 allocate_tensors()가 이미 이 shape으로
        인터프리터를 할당해뒀으므로 매 호출마다 resize_tensor_input()을 다시 부를 필요가
        없다. FP32에서는 이 재할당이 저비용이라 안 드러났지만, dynamic-range(INT8 가중치)
        인코더에서 XNNPACK 델리게이트가 매번 가중치를 재포장하며 300초 이상 걸리는 걸
        확인해서(2026-08-26) 근본 원인을 찾아 제거함 -- shape이 실제로 달라질 때만 재할당."""
        inp = make_model_input(canvas_norm, self.in_ch).numpy()      # [in_ch,H,W]
        inp_nhwc = np.transpose(inp, (1, 2, 0))[None].astype(np.float32)
        if tuple(self.enc_in['shape']) != inp_nhwc.shape:
            self.enc.resize_tensor_input(self.enc_in['index'], inp_nhwc.shape)
            self.enc.allocate_tensors()
            self.enc_in['shape'] = inp_nhwc.shape
        self.enc.set_tensor(self.enc_in['index'], inp_nhwc)
        self.enc.invoke()
        return self.enc.get_tensor(self.enc_out_idx)

    def decode(self, memory: np.ndarray, max_steps: int = 300, stop_token_id: int = None):
        """반환: 생성된 토큰 id 리스트(SOS/EOS/PAD 제외). 매 스텝 텐서 shape이 고정이라
        resize_tensor_input을 루프 안에서 다시 호출할 필요가 없다.

        inference.py의 greedy_decode/greedy_decode_kv와 동일하게 EOS_BOOST +
        LONG_DECODE_RAMP(과잉생성 방지)를 적용한다 — 이게 없으면 production과 공정한
        비교가 안 됨(2026-08-24, 처음 이걸 빼고 10곡 검증했다가 정확도가 실제보다 낮게
        나와서 나중에 추가함)."""
        k_mem, v_mem = self.precompute_memory_kv(memory)
        L, H, C, Dh = self.num_layers, self.num_heads, self.cache_len, self.head_dim
        k_cache = np.zeros((L, 1, H, C, Dh), dtype=np.float32)
        v_cache = np.zeros((L, 1, H, C, Dh), dtype=np.float32)
        cur_id, tokens = SOS_ID, []
        for pos in range(max_steps):
            self.dec.set_tensor(self.dec_ins['token_id'], np.array([[cur_id]], dtype=np.int64))
            self.dec.set_tensor(self.dec_ins['pos'], np.array([pos], dtype=np.int64))
            self.dec.set_tensor(self.dec_ins['k_mem_in'], k_mem)
            self.dec.set_tensor(self.dec_ins['v_mem_in'], v_mem)
            self.dec.set_tensor(self.dec_ins['k_cache_in'], k_cache)
            self.dec.set_tensor(self.dec_ins['v_cache_in'], v_cache)
            self.dec.invoke()
            logits = self.dec.get_tensor(self.dec_outs['next_logits'])[0]
            k_cache = self.dec.get_tensor(self.dec_outs['k_cache_out'])
            v_cache = self.dec.get_tensor(self.dec_outs['v_cache_out'])

            logits[EOS_ID] *= EOS_BOOST
            if pos > LONG_DECODE_THRESHOLD:
                long_boost = 1.0 + (pos - LONG_DECODE_THRESHOLD) * LONG_DECODE_RAMP
                logits[EOS_ID] *= long_boost
                if stop_token_id is not None:
                    logits[stop_token_id] *= long_boost

            nxt = int(np.argmax(logits))
            if nxt == EOS_ID:
                break
            if nxt != PAD_ID:
                tokens.append(nxt)
                if stop_token_id is not None and nxt == stop_token_id:
                    break
            cur_id = nxt
        return tokens

    def _dec_step(self, cur_id, pos, k_mem, v_mem, k_cache, v_cache, stop_token_id, step_for_ramp,
                  interp=None, ins=None, outs=None):
        """decode()/decode_hybrid()/decode_bucketed()가 공유하는 디코더 1스텝 호출
        (EOS_BOOST/과잉생성 방지 포함). k_mem/v_mem은 precompute_memory_kv()로 이미지당
        1회 계산해둔 cross-attention K,V(2026-08-26, 이전엔 매 스텝 memory에서
        재계산했음). interp/ins/outs를 안 주면 기본(작은 버킷, self.dec)을 쓴다 --
        decode_bucketed()가 큰 버킷으로 전환할 때 self.dec_large/dec_large_ins/outs를
        넘긴다. 반환: (다음 토큰 id, 갱신된 캐시)."""
        interp = interp if interp is not None else self.dec
        ins = ins if ins is not None else self.dec_ins
        outs = outs if outs is not None else self.dec_outs
        interp.set_tensor(ins['token_id'], np.array([[cur_id]], dtype=np.int64))
        interp.set_tensor(ins['pos'], np.array([pos], dtype=np.int64))
        interp.set_tensor(ins['k_mem_in'], k_mem)
        interp.set_tensor(ins['v_mem_in'], v_mem)
        interp.set_tensor(ins['k_cache_in'], k_cache)
        interp.set_tensor(ins['v_cache_in'], v_cache)
        interp.invoke()
        logits = interp.get_tensor(outs['next_logits'])[0]
        k_cache = interp.get_tensor(outs['k_cache_out'])
        v_cache = interp.get_tensor(outs['v_cache_out'])

        logits[EOS_ID] *= EOS_BOOST
        if step_for_ramp > LONG_DECODE_THRESHOLD:
            lb = 1.0 + (step_for_ramp - LONG_DECODE_THRESHOLD) * LONG_DECODE_RAMP
            logits[EOS_ID] *= lb
            if stop_token_id is not None:
                logits[stop_token_id] *= lb
        nxt = int(np.argmax(logits))
        return nxt, k_cache, v_cache

    def decode_hybrid(self, memory: np.ndarray, tok2id: dict, id2tok: dict,
                      max_steps: int = 300, stop_token_id: int = None):
        """InlineTimeCorrector(마디 중간 박자표 재추정)를 지원하는 3단계 하이브리드 디코딩
        — PyTorch의 greedy_decode_kv와 동일한 설계(2026-08-24):
          1) 첫 마디는 step 그래프로 순서대로 디코딩하면서 InlineTimeCorrector 교정을 받는다
             (이 구간은 매 스텝 캐시가 갱신되지만, 교정이 일어나면 뒤에서 어차피 통째로
             다시 채우므로 중간 캐시 상태는 최종 결과에 영향을 안 준다).
          2) 첫 마디가 끝나면(barline류 토큰) (교정된) 토큰들을 decoder_bulk_INT8.tflite에
             한 번에 넣어서 캐시를 처음부터 다시 채운다.
          3) 나머지는 step 그래프로 빠르게 이어간다.
        첫 마디가 chunk_len보다 길면(드묾) 일괄 채우기를 건너뛰고 1단계의 캐시를 그대로
        이어쓴다 — 정확도가 약간 저하될 수 있는 알려진 예외 케이스.

        ⚠ 버킷팅(2026-08-26, §14) 적용 후: self.dec/self.bulk는 작은 버킷 그래프라, 이
        메서드는 **버킷 전환을 안 한다** — 시퀀스가 small_cache_len을 넘으면 캐시 쓰기가
        조용히 무효화되며 정확도가 깨진다. 실제 추론에는 decode_bucketed()를 쓸 것 —
        이 메서드는 비교/참고용으로만 남겨둠(decode()가 InlineTimeCorrector 없이
        참고용으로 남아있는 것과 같은 이유)."""
        time_correct = InlineTimeCorrector(tok2id, id2tok, is_grand=True)
        barline_ids = {tid for tid, s in id2tok.items() if s in _BARLINE_TOKEN_STRS}

        k_mem, v_mem = self.precompute_memory_kv(memory)   # 이미지당 1회(2026-08-26)
        L, H, C, Dh = self.num_layers, self.num_heads, self.cache_len, self.head_dim
        k_cache = np.zeros((L, 1, H, C, Dh), dtype=np.float32)
        v_cache = np.zeros((L, 1, H, C, Dh), dtype=np.float32)

        seq = [SOS_ID]     # SOS 포함 전체 시퀀스(호스트 리스트) -- InlineTimeCorrector가 패치
        result = []
        first_measure_done = False
        step = 0
        for step in range(max_steps):
            nxt, k_cache, v_cache = self._dec_step(seq[-1], step, k_mem, v_mem, k_cache, v_cache,
                                                    stop_token_id, step)
            if nxt == EOS_ID:
                return result
            if nxt != PAD_ID:
                result.append(nxt)
                tok_str = id2tok.get(nxt, '')
                new_time_id = time_correct.observe(tok_str, len(result) - 1)
                if new_time_id is not None and result[time_correct.time_idx] != new_time_id:
                    result[time_correct.time_idx] = new_time_id
                    seq[time_correct.time_idx + 1] = new_time_id
                if stop_token_id is not None and nxt == stop_token_id:
                    seq.append(nxt)
                    first_measure_done = True
                    break
            seq.append(nxt)
            if nxt != PAD_ID and nxt in barline_ids:
                first_measure_done = True
                break

        if not first_measure_done or len(seq) <= 1:
            return result

        # ── 2단계: 일괄 캐시 채우기 (첫 마디가 chunk_len 이내일 때만) ──
        if len(seq) <= self.chunk_len:
            chunk = seq + [PAD_ID] * (self.chunk_len - len(seq))
            self.bulk.set_tensor(self.bulk_ins['tokens'], np.array([chunk], dtype=np.int64))
            self.bulk.set_tensor(self.bulk_ins['k_mem_in'], k_mem)
            self.bulk.set_tensor(self.bulk_ins['v_mem_in'], v_mem)
            self.bulk.invoke()
            k_cache = self.bulk.get_tensor(self.bulk_outs['k_cache_out'])
            v_cache = self.bulk.get_tensor(self.bulk_outs['v_cache_out'])
        # else: 첫 마디가 너무 길어 일괄 채우기 스킵 -- 1단계 캐시를 그대로 이어씀

        pos, cur_id = len(seq) - 1, seq[-1]

        # ── 3단계: 빠른 경로 ──
        for step2 in range(step + 1, max_steps):
            nxt, k_cache, v_cache = self._dec_step(cur_id, pos, k_mem, v_mem, k_cache, v_cache,
                                                    stop_token_id, step2)
            if nxt == EOS_ID:
                break
            if nxt != PAD_ID:
                result.append(nxt)
                if stop_token_id is not None and nxt == stop_token_id:
                    break
            pos += 1
            cur_id = nxt
        return result

    def decode_bucketed(self, memory: np.ndarray, tok2id: dict, id2tok: dict,
                        max_steps: int = 300, stop_token_id: int = None):
        """decode_hybrid()에 버킷 전환을 추가한 버전 -- 실제 추론에 쓰는 기본 메서드
        (2026-08-26, EXPORT_NOTES.md §14). 5단계:
          1) 첫 마디: 작은 버킷 step 그래프로 순차 디코딩(InlineTimeCorrector 교정 받음)
          2) 첫 마디 끝나면 작은 버킷 bulk-capture(chunk_len,small_cache_len)로 캐시 일괄 채움
          3) 작은 버킷 step 그래프로 계속 -- position이 small_cache_len-1에 도달할 때까지
          4) [넘칠 때만] 지금까지 전체(길이==small_cache_len)를 리버킷 그래프
             (chunk_len=small_cache_len, cache_len=large_cache_len)에 넣어서 큰 버퍼를
             한 번에 채운다
          5) [넘칠 때만] 큰 버킷 step 그래프로 계속
        newage01~30 실측(2026-08-26)으로는 4)/5)가 한 번도 발동하지 않았다(전부
        small_cache_len 안에서 끝남) -- 그래도 리버킷 텐서를 직접 비교해서 정확성은
        검증해뒀다(§14, 오차가 기존 chunk_len=40 bulk-capture와 동일 수준)."""
        time_correct = InlineTimeCorrector(tok2id, id2tok, is_grand=True)
        barline_ids = {tid for tid, s in id2tok.items() if s in _BARLINE_TOKEN_STRS}

        k_mem, v_mem = self.precompute_memory_kv(memory)
        L, H, Dh = self.num_layers, self.num_heads, self.head_dim
        SC = self.small_cache_len
        k_cache = np.zeros((L, 1, H, SC, Dh), dtype=np.float32)
        v_cache = np.zeros((L, 1, H, SC, Dh), dtype=np.float32)

        seq = [SOS_ID]
        result = []
        first_measure_done = False
        step = 0
        for step in range(max_steps):
            nxt, k_cache, v_cache = self._dec_step(seq[-1], step, k_mem, v_mem, k_cache, v_cache,
                                                    stop_token_id, step)
            if nxt == EOS_ID:
                return result
            if nxt != PAD_ID:
                result.append(nxt)
                tok_str = id2tok.get(nxt, '')
                new_time_id = time_correct.observe(tok_str, len(result) - 1)
                if new_time_id is not None and result[time_correct.time_idx] != new_time_id:
                    result[time_correct.time_idx] = new_time_id
                    seq[time_correct.time_idx + 1] = new_time_id
                if stop_token_id is not None and nxt == stop_token_id:
                    seq.append(nxt)
                    first_measure_done = True
                    break
            seq.append(nxt)
            if nxt != PAD_ID and nxt in barline_ids:
                first_measure_done = True
                break

        if not first_measure_done or len(seq) <= 1:
            return result

        # ── 2단계: 작은 버킷 bulk-capture ──
        if len(seq) <= self.chunk_len:
            chunk = seq + [PAD_ID] * (self.chunk_len - len(seq))
            self.bulk.set_tensor(self.bulk_ins['tokens'], np.array([chunk], dtype=np.int64))
            self.bulk.set_tensor(self.bulk_ins['k_mem_in'], k_mem)
            self.bulk.set_tensor(self.bulk_ins['v_mem_in'], v_mem)
            self.bulk.invoke()
            k_cache = self.bulk.get_tensor(self.bulk_outs['k_cache_out'])
            v_cache = self.bulk.get_tensor(self.bulk_outs['v_cache_out'])
        # else: 첫 마디가 너무 길어 일괄 채우기 스킵 -- 1단계 캐시를 그대로 이어씀

        pos, cur_id = len(seq) - 1, seq[-1]

        # ── 3단계(+4·5단계): 작은 버킷으로 계속, 넘치면 큰 버킷으로 전환 ──
        cur_interp, cur_ins, cur_outs = self.dec, self.dec_ins, self.dec_outs
        rebucketed = not self._bucketed   # 큰 버킷 파일이 없으면 애초에 전환 불가
        for step2 in range(step + 1, max_steps):
            if not rebucketed and pos >= SC - 1:
                rebucketed = True
                chunk = seq[:SC]
                self.rebucket.set_tensor(self.rebucket_ins['tokens'], np.array([chunk], dtype=np.int64))
                self.rebucket.set_tensor(self.rebucket_ins['k_mem_in'], k_mem)
                self.rebucket.set_tensor(self.rebucket_ins['v_mem_in'], v_mem)
                self.rebucket.invoke()
                k_cache = self.rebucket.get_tensor(self.rebucket_outs['k_cache_out'])
                v_cache = self.rebucket.get_tensor(self.rebucket_outs['v_cache_out'])
                cur_interp, cur_ins, cur_outs = self.dec_large, self.dec_large_ins, self.dec_large_outs

            nxt, k_cache, v_cache = self._dec_step(cur_id, pos, k_mem, v_mem, k_cache, v_cache,
                                                    stop_token_id, step2,
                                                    interp=cur_interp, ins=cur_ins, outs=cur_outs)
            if nxt == EOS_ID:
                break
            if nxt != PAD_ID:
                result.append(nxt)
                seq.append(nxt)
                if stop_token_id is not None and nxt == stop_token_id:
                    break
            pos += 1
            cur_id = nxt
        return result


def run_image_tflite(image_path: str, tflite_dir: str, tok2id: dict, id2tok: dict,
                     chunk_len: int = 40, max_steps: int = 300):
    """production run_image()와 공정하게 비교되도록, EOS_BOOST/과잉생성 방지 + InlineTimeCorrector
    + 버킷 전환(decode_bucketed() 내부, §14) + 디코딩 후 후처리(fix_chord_tokens/
    fix_span_tokens/correct_time_signature/correct_accidentals_by_key)까지 동일하게
    적용한다 — 후처리는 순수 토큰 리스트 연산이라 모델 없이 그대로 재사용 가능.

    캐시 크기는 이제 인자로 안 받는다 -- export된 파일(decoder_INT8.tflite의 작은 버킷,
    decoder_large_INT8.tflite의 큰 버킷)에서 TFLiteOmrModel이 직접 읽는다."""
    model = TFLiteOmrModel(tflite_dir, chunk_len=chunk_len)
    is_real_photo = image_path.lower().endswith(('.jpg', '.jpeg'))
    gray0 = load_preprocessed(image_path)
    staffs, gray = best_effort_staff_detection(gray0, use_full_warp=is_real_photo)
    if not staffs or len(staffs) < 2:
        raise RuntimeError("오선 검출 실패(대보표 2개 필요) — 단일 오선 이미지는 아직 미지원")
    canvas = extract_system_canvas(gray, staffs[:2])
    norm = (canvas.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    stop_id = tok2id.get('barline-final')

    t0 = time.time()
    memory = model.encode(norm)
    t_enc = time.time() - t0

    t1 = time.time()
    ids = model.decode_bucketed(memory, tok2id, id2tok, max_steps=max_steps, stop_token_id=stop_id)
    t_dec = time.time() - t1

    ids = fix_span_tokens(fix_chord_tokens(ids, id2tok), id2tok)
    ids = correct_time_signature(ids, id2tok, tok2id, is_grand=True)
    ids = correct_accidentals_by_key(ids, id2tok, tok2id)
    tokens = [id2tok.get(i, '<UNK>') for i in ids]
    return tokens, t_enc, t_dec


def main():
    ap = argparse.ArgumentParser(description="TFLite(인코더+디코더)만으로 악보 이미지 인식")
    ap.add_argument('--tflite_dir', default=str(_TRAIN / 'tflite_export_dr'))
    ap.add_argument('--tokenizer', default=str(_TRAIN / 'tokenizer258.json'))
    ap.add_argument('--image', required=True)
    args = ap.parse_args()

    tok2id, id2tok = load_tokenizer(args.tokenizer)
    tokens, t_enc, t_dec = run_image_tflite(args.image, args.tflite_dir, tok2id, id2tok)
    print(f"인코더: {t_enc:.2f}s  디코더: {t_dec:.2f}s ({len(tokens)}토큰, "
          f"{t_dec/max(1,len(tokens))*1000:.0f}ms/토큰)")
    print(' '.join(tokens))


if __name__ == '__main__':
    main()
