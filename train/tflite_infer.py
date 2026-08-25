"""tflite_infer.py — TFLite로 export한 인코더+디코더만으로(PyTorch 모델 로드 없이) 실제
악보 이미지를 인식하는 CLI 스크립트.

모바일 앱 없이도 "TFLite 모델이 실제 인터프리터에서 완전하게 동작하는지"를 증명하는 게
목적이다(train/QUANTIZATION_MOBILE.md ① 항목). export_tflite.py가 만든
encoder_INT8.tflite/decoder_INT8.tflite(둘 다 지금은 실제로는 FP32)를 그대로 쓴다.

디코더는 고정 크기 self-attention KV캐시 인터페이스(_export_onnx_decoder_kv 참고)라
매 스텝 텐서 shape이 동일 — resize_tensor_input이 처음 한 번만 필요하고 이후 스텝마다는
필요 없다(예전 growing past_ids 방식은 매 스텝 리사이즈가 필요해서 2번째 스텝부터
크래시했었다).

사용법:
    python train/tflite_infer.py --tflite_dir train/tflite_export --image <악보사진.jpg>
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf

from dataset import (load_preprocessed, best_effort_staff_detection, extract_system_canvas,
                     IMG_MEAN, IMG_STD, make_model_input, load_tokenizer)
from inference import (EOS_BOOST, LONG_DECODE_THRESHOLD, LONG_DECODE_RAMP,
                       fix_chord_tokens, fix_span_tokens, correct_time_signature,
                       correct_accidentals_by_key, InlineTimeCorrector, _BARLINE_TOKEN_STRS)

PAD_ID, SOS_ID, EOS_ID = 0, 1, 2


class TFLiteOmrModel:
    """encoder_INT8.tflite/decoder_INT8.tflite(+decoder_bulk_INT8.tflite) 세 파일만으로
    동작하는 추론 래퍼."""

    def __init__(self, tflite_dir: str, cache_len: int = 300, chunk_len: int = 40):
        d = Path(tflite_dir)
        self.enc = tf.lite.Interpreter(model_path=str(d / "encoder_INT8.tflite"))
        self.dec = tf.lite.Interpreter(model_path=str(d / "decoder_INT8.tflite"))
        self.bulk = tf.lite.Interpreter(model_path=str(d / "decoder_bulk_INT8.tflite"))
        self.enc.allocate_tensors()
        self.dec.allocate_tensors()
        self.bulk.allocate_tensors()
        self.enc_in = self.enc.get_input_details()[0]
        self.enc_out_idx = self.enc.get_output_details()[0]['index']
        self.dec_ins = {t['name']: t['index'] for t in self.dec.get_input_details()}
        self.dec_outs = {t['name']: t['index'] for t in self.dec.get_output_details()}
        self.bulk_ins = {t['name']: t['index'] for t in self.bulk.get_input_details()}
        self.bulk_outs = {t['name']: t['index'] for t in self.bulk.get_output_details()}
        self.cache_len = cache_len
        self.chunk_len = chunk_len
        self.in_ch = self.enc_in['shape'][-1]  # NHWC라 채널이 마지막 차원

        k_shape = self.dec.get_tensor_details()[self.dec_ins['k_cache_in']]['shape']
        self.num_layers, _, self.num_heads, _, self.head_dim = k_shape

    def encode(self, canvas_norm: np.ndarray) -> np.ndarray:
        """canvas_norm: [H,W] float32, 이미 (px/255-mean)/std 정규화 완료."""
        inp = make_model_input(canvas_norm, self.in_ch).numpy()      # [in_ch,H,W]
        inp_nhwc = np.transpose(inp, (1, 2, 0))[None].astype(np.float32)
        self.enc.resize_tensor_input(self.enc_in['index'], inp_nhwc.shape)
        self.enc.allocate_tensors()
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
        L, H, C, Dh = self.num_layers, self.num_heads, self.cache_len, self.head_dim
        k_cache = np.zeros((L, 1, H, C, Dh), dtype=np.float32)
        v_cache = np.zeros((L, 1, H, C, Dh), dtype=np.float32)
        cur_id, tokens = SOS_ID, []
        for pos in range(max_steps):
            self.dec.set_tensor(self.dec_ins['token_id'], np.array([[cur_id]], dtype=np.int64))
            self.dec.set_tensor(self.dec_ins['pos'], np.array([pos], dtype=np.int64))
            self.dec.set_tensor(self.dec_ins['memory'], memory.astype(np.float32))
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

    def _dec_step(self, cur_id, pos, memory, k_cache, v_cache, stop_token_id, step_for_ramp):
        """decode()/decode_hybrid()가 공유하는 디코더 1스텝 호출 (EOS_BOOST/과잉생성
        방지 포함). 반환: (다음 토큰 id, 다음 토큰 logits argmax 전 로짓, 갱신된 캐시)."""
        self.dec.set_tensor(self.dec_ins['token_id'], np.array([[cur_id]], dtype=np.int64))
        self.dec.set_tensor(self.dec_ins['pos'], np.array([pos], dtype=np.int64))
        self.dec.set_tensor(self.dec_ins['memory'], memory.astype(np.float32))
        self.dec.set_tensor(self.dec_ins['k_cache_in'], k_cache)
        self.dec.set_tensor(self.dec_ins['v_cache_in'], v_cache)
        self.dec.invoke()
        logits = self.dec.get_tensor(self.dec_outs['next_logits'])[0]
        k_cache = self.dec.get_tensor(self.dec_outs['k_cache_out'])
        v_cache = self.dec.get_tensor(self.dec_outs['v_cache_out'])

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
        이어쓴다 — 정확도가 약간 저하될 수 있는 알려진 예외 케이스."""
        time_correct = InlineTimeCorrector(tok2id, id2tok, is_grand=True)
        barline_ids = {tid for tid, s in id2tok.items() if s in _BARLINE_TOKEN_STRS}

        L, H, C, Dh = self.num_layers, self.num_heads, self.cache_len, self.head_dim
        k_cache = np.zeros((L, 1, H, C, Dh), dtype=np.float32)
        v_cache = np.zeros((L, 1, H, C, Dh), dtype=np.float32)

        seq = [SOS_ID]     # SOS 포함 전체 시퀀스(호스트 리스트) -- InlineTimeCorrector가 패치
        result = []
        first_measure_done = False
        step = 0
        for step in range(max_steps):
            nxt, k_cache, v_cache = self._dec_step(seq[-1], step, memory, k_cache, v_cache,
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
            self.bulk.set_tensor(self.bulk_ins['memory'], memory.astype(np.float32))
            self.bulk.invoke()
            k_cache = self.bulk.get_tensor(self.bulk_outs['k_cache_out'])
            v_cache = self.bulk.get_tensor(self.bulk_outs['v_cache_out'])
        # else: 첫 마디가 너무 길어 일괄 채우기 스킵 -- 1단계 캐시를 그대로 이어씀

        pos, cur_id = len(seq) - 1, seq[-1]

        # ── 3단계: 빠른 경로 ──
        for step2 in range(step + 1, max_steps):
            nxt, k_cache, v_cache = self._dec_step(cur_id, pos, memory, k_cache, v_cache,
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


def run_image_tflite(image_path: str, tflite_dir: str, tok2id: dict, id2tok: dict,
                     cache_len: int = 300, chunk_len: int = 40, max_steps: int = 300):
    """production run_image()와 공정하게 비교되도록, EOS_BOOST/과잉생성 방지 + InlineTimeCorrector
    (decode_hybrid() 내부) + 디코딩 후 후처리(fix_chord_tokens/fix_span_tokens/
    correct_time_signature/correct_accidentals_by_key)까지 동일하게 적용한다 — 후처리는
    순수 토큰 리스트 연산이라 모델 없이 그대로 재사용 가능."""
    model = TFLiteOmrModel(tflite_dir, cache_len=cache_len, chunk_len=chunk_len)
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
    ids = model.decode_hybrid(memory, tok2id, id2tok, max_steps=max_steps, stop_token_id=stop_id)
    t_dec = time.time() - t1

    ids = fix_span_tokens(fix_chord_tokens(ids, id2tok), id2tok)
    ids = correct_time_signature(ids, id2tok, tok2id, is_grand=True)
    ids = correct_accidentals_by_key(ids, id2tok, tok2id)
    tokens = [id2tok.get(i, '<UNK>') for i in ids]
    return tokens, t_enc, t_dec


def main():
    ap = argparse.ArgumentParser(description="TFLite(인코더+디코더)만으로 악보 이미지 인식")
    ap.add_argument('--tflite_dir', default=str(Path(__file__).parent / 'tflite_export'))
    ap.add_argument('--tokenizer', default=str(Path(__file__).parent / 'tokenizer258.json'))
    ap.add_argument('--image', required=True)
    ap.add_argument('--cache_len', type=int, default=300)
    args = ap.parse_args()

    tok2id, id2tok = load_tokenizer(args.tokenizer)
    tokens, t_enc, t_dec = run_image_tflite(args.image, args.tflite_dir, tok2id, id2tok,
                                            cache_len=args.cache_len)
    print(f"인코더: {t_enc:.2f}s  디코더: {t_dec:.2f}s ({len(tokens)}토큰, "
          f"{t_dec/max(1,len(tokens))*1000:.0f}ms/토큰)")
    print(' '.join(tokens))


if __name__ == '__main__':
    main()
