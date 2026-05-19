#!/usr/bin/env python3
"""
inspect_tflite.py — TFLite 모델 입출력 형식 확인
사용법: python inspect_tflite.py <model.tflite>
"""
import sys
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

import numpy as np

def dtype_name(d):
    return {1:'float32',2:'int32',3:'uint8',4:'int64',7:'string',9:'int8'}.get(d, f'?{d}')

def inspect(path):
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()

    print(f"\n{'='*55}")
    print(f"  모델: {path}")
    print(f"{'='*55}")
    for tag, details in [("INPUT", interp.get_input_details()),
                         ("OUTPUT", interp.get_output_details())]:
        for d in details:
            print(f"\n  [{tag}] index={d['index']}  name={d['name']}")
            print(f"    shape : {d['shape']}")
            print(f"    dtype : {dtype_name(d['dtype'])}")
            if d.get('quantization_parameters'):
                qp = d['quantization_parameters']
                scales = qp.get('scales', [])
                zps    = qp.get('zero_points', [])
                if len(scales):
                    print(f"    quant : scale={scales[0]:.6f}  zero_point={zps[0]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        assets = "/mnt/c/Users/kyutae/AndroidStudioProjects/MusicScore/app/src/main/assets"
        for name in ["segnet_308_int8.tflite", "encoder_331_int8.tflite"]:
            inspect(f"{assets}/{name}")
    else:
        inspect(sys.argv[1])
