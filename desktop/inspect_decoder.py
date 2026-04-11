"""Inspect decoder ONNX model input/output names and test a single step."""
import os, sys
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("pip install onnxruntime")
    sys.exit(1)

assets_dir = sys.argv[1] if len(sys.argv) > 1 else "assets"
model_path = os.path.join(assets_dir, "decoder_331_int8.onnx")

print(f"Loading {model_path} ...")
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print("\n=== INPUTS ===")
for inp in sess.get_inputs():
    print(f"  {inp.name!r:40s}  shape={inp.shape}  type={inp.type}")

print("\n=== OUTPUTS ===")
for out in sess.get_outputs():
    print(f"  {out.name!r:40s}  shape={out.shape}  type={out.type}")

# Quick smoke test: step 0 with a tiny fake encoder output (seq_len=4, dim=512)
print("\n=== SMOKE TEST (step 0, seq_len=4) ===")
SEQ = 4
HEADS = 8
HEAD_DIM = 64
NUM_CACHE = 32

inputs_dict = {
    # adjust names based on actual names above if needed
    "rhythms":       np.array([[1]], dtype=np.int64),    # BOS
    "pitchs":        np.array([[1]], dtype=np.int64),
    "lifts":         np.array([[1]], dtype=np.int64),
    "articulations": np.array([[1]], dtype=np.int64),
    "context":       np.zeros((1, SEQ, 512), dtype=np.float32),
    "cache_len":     np.array([0], dtype=np.int64),
}
for i in range(NUM_CACHE):
    inputs_dict[f"cache_in{i}"] = np.zeros((1, HEADS, 0, HEAD_DIM), dtype=np.float32)

# Filter to only pass inputs the model actually has
model_input_names = {inp.name for inp in sess.get_inputs()}
feed = {k: v for k, v in inputs_dict.items() if k in model_input_names}
missing = model_input_names - set(inputs_dict.keys())
extra = set(inputs_dict.keys()) - model_input_names
if missing: print(f"  [!] Missing inputs: {missing}")
if extra:   print(f"  [!] Extra inputs not used: {extra}")

try:
    outputs = sess.run(None, feed)
    out_names = [o.name for o in sess.get_outputs()]
    print(f"  Run OK, {len(outputs)} outputs")
    for name, arr in zip(out_names[:5], outputs[:5]):
        tok = int(np.argmax(arr))
        print(f"    {name}: argmax={tok}  shape={arr.shape}")
except Exception as e:
    print(f"  Run FAILED: {e}")
