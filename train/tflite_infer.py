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

PAD_ID, SOS_ID, EOS_ID = 0, 1, 2


class TFLiteOmrModel:
    """encoder_INT8.tflite/decoder_INT8.tflite 두 파일만으로 동작하는 추론 래퍼."""

    def __init__(self, tflite_dir: str, cache_len: int = 300):
        d = Path(tflite_dir)
        self.enc = tf.lite.Interpreter(model_path=str(d / "encoder_INT8.tflite"))
        self.dec = tf.lite.Interpreter(model_path=str(d / "decoder_INT8.tflite"))
        self.enc.allocate_tensors()
        self.dec.allocate_tensors()
        self.enc_in = self.enc.get_input_details()[0]
        self.enc_out_idx = self.enc.get_output_details()[0]['index']
        self.dec_ins = {t['name']: t['index'] for t in self.dec.get_input_details()}
        self.dec_outs = {t['name']: t['index'] for t in self.dec.get_output_details()}
        self.cache_len = cache_len
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
        resize_tensor_input을 루프 안에서 다시 호출할 필요가 없다."""
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
            nxt = int(np.argmax(logits))
            if nxt == EOS_ID:
                break
            if nxt != PAD_ID:
                tokens.append(nxt)
                if stop_token_id is not None and nxt == stop_token_id:
                    break
            cur_id = nxt
        return tokens


def run_image_tflite(image_path: str, tflite_dir: str, id2tok: dict, cache_len: int = 300,
                     max_steps: int = 300):
    model = TFLiteOmrModel(tflite_dir, cache_len=cache_len)
    is_real_photo = image_path.lower().endswith(('.jpg', '.jpeg'))
    gray0 = load_preprocessed(image_path)
    staffs, gray = best_effort_staff_detection(gray0, use_full_warp=is_real_photo)
    if not staffs or len(staffs) < 2:
        raise RuntimeError("오선 검출 실패(대보표 2개 필요) — 단일 오선 이미지는 아직 미지원")
    canvas = extract_system_canvas(gray, staffs[:2])
    norm = (canvas.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD

    t0 = time.time()
    memory = model.encode(norm)
    t_enc = time.time() - t0

    t1 = time.time()
    ids = model.decode(memory, max_steps=max_steps)
    t_dec = time.time() - t1

    tokens = [id2tok.get(i, '<UNK>') for i in ids]
    return tokens, t_enc, t_dec


def main():
    ap = argparse.ArgumentParser(description="TFLite(인코더+디코더)만으로 악보 이미지 인식")
    ap.add_argument('--tflite_dir', default=str(Path(__file__).parent / 'tflite_export'))
    ap.add_argument('--tokenizer', default=str(Path(__file__).parent / 'tokenizer258.json'))
    ap.add_argument('--image', required=True)
    ap.add_argument('--cache_len', type=int, default=300)
    args = ap.parse_args()

    _, id2tok = load_tokenizer(args.tokenizer)
    tokens, t_enc, t_dec = run_image_tflite(args.image, args.tflite_dir, id2tok,
                                            cache_len=args.cache_len)
    print(f"인코더: {t_enc:.2f}s  디코더: {t_dec:.2f}s ({len(tokens)}토큰, "
          f"{t_dec/max(1,len(tokens))*1000:.0f}ms/토큰)")
    print(' '.join(tokens))


if __name__ == '__main__':
    main()
