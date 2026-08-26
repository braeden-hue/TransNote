# TFLite export 설계 노트

`export_tflite.py` / `onnx_wrappers.py` / `tflite_infer.py` 코드에서 뺀 배경 설명 모음.
코드에는 한 줄 요약 + `EXPORT_NOTES.md §N` 포인터만 남겼다 — **수정 전에 해당 항목을 읽을 것.**
여기 적힌 것들은 대부분 "이렇게 안 하면 실제로 깨졌던" 사례라서, 모르고 되돌리면 같은 버그가 재발한다.

파이프라인: `PyTorch .pt → ONNX → (onnx2tf) → TFLite`.
실측 성능·정확도 수치와 시행착오 전체 서사는 [`../QUANTIZATION_MOBILE.md`](../QUANTIZATION_MOBILE.md) 참고.

---

## §1. 체크포인트 아키텍처는 반드시 역산해서 쓸 것

`OmrSeq2Seq(vocab_size=...)`를 생성자 기본값으로만 부르면 r15처럼 논디폴트 아키텍처
(CoordConv `in_ch=2`)인 체크포인트에서 state_dict shape mismatch가 난다.
반드시 `infer_arch_from_state_dict()`로 역산한 값을 넘긴다(`handler.py`와 동일 패턴).

2026-08-24에 `export_tflite.py`와 `inference.py` CLI 두 곳에서 실제로 겪은 버그다.

## §2. 캘리브레이션 입력은 학습 전처리를 그대로 통과시킬 것

INT8 PTQ용 캘리브레이션 셋은 `load_preprocessed → detect_staffs → extract_system_canvas`
실제 파이프라인을 그대로 태워서 만든다. `in_ch=2`(CoordConv)면 `make_model_input()`이
좌표 채널을 붙여준다 — 여기서 채널 수를 안 맞추면 캘리브레이션 입력 shape이 실제 모델
입력과 달라져 **양자화 스케일이 잘못 잡힌다**.

`_convert_tflite()`에 넘기는 `calib_npy`는 이미 `(px/255 - mean)/std` 정규화가 끝난
데이터다 — 그래서 onnx2tf에는 mean=0/std=1을 줘서 추가 정규화를 건너뛰게 한다.

## §3. 고정 shape 디코더 — 동적 슬라이싱으로 되돌리지 말 것

디코더 step 그래프(`_DecoderStepWrapperKV`)는 입출력 shape이 `pos`와 무관하게 전부 고정이다.
PyTorch 쪽(`model.py: decode_step_kv_cached`)처럼 `k_cache[:, :, :pos+1, :]`로 실제로
잘라 쓰는 게 아니라, **고정 크기 버퍼(`cache_len`) 전체에 attention을 계산하고 `pos` 뒤쪽을
마스킹으로 제외**한다. 캐시 쓰기도 in-place 슬라이스 대입이 아니라 마스크 기반이다.

왜 이렇게까지 하냐면 — 예전 방식(`past_ids`를 매 스텝 growing 텐서로 넣기)은 TFLite에서
**2번째 디코딩 스텝부터 reshape 에러로 깨졌고**, 매 스텝 `resize_tensor_input`이 호출되면서
XNNPACK 델리게이트 재구성으로 스텝당 18.7초가 걸렸다. 고정 shape이면 두 문제가 원천 소멸한다.

대가: self-attention이 `pos`와 무관하게 항상 `O(cache_len)`이다(PyTorch는 `O(pos)`).
재학습 없이 이걸 더 줄이는 후보는 "cache_len 버킷팅"인데 우선순위 낮음으로 보류
(`QUANTIZATION_MOBILE.md` "이번 범위에서 제외" 참고).

## §4. `torch.where`·`is_causal`을 쓰면 양자화가 NaN으로 깨진다 ★

**이 문서에서 제일 중요한 항목.** dynamic-range 양자화 시:

- `torch.where(mask, a, b)` (self-attention 캐시 쓰기)
- `F.scaled_dot_product_attention(..., is_causal=True)` (causal masking, 내부적으로 같은 패턴)

둘 다 ONNX `Where`/`Select` 노드를 만드는데, **onnx2tf가 이 노드의 상수 피연산자를
"가중치"로 잘못 인식해서 양자화 스케일을 `2.68e36` 같은 값으로 계산**한다 → 디코딩
1스텝만에 전부 NaN. 크래시가 안 나고 조용히 NaN이 전파돼서 발견이 늦었다(10곡 정확도 0%).

그래서 코드에서는:
- `torch.where(mask, a, b)` → 산술 블렌드 `a*mask + b*(1-mask)` (Mul/Add만 사용)
- `is_causal=True` → 명시적 덧셈 마스크 `attn_mask=torch.triu(-1e9, diagonal=1)`

`Where` 노드 자체를 안 만들어서 회피한다. `quant_type='per-tensor'`로 바꿔도 동일 버그가
재현되고, onnx2tf에는 텐서 단위 양자화 제외 옵션이 없다(`convert()` 시그니처 확인함).

**`Where`/`is_causal`을 다시 쓰고 싶어지면 반드시 dynamic-range로 export해서 단일 스텝
logits이 NaN이 아닌지부터 확인할 것.**

## §5. linear는 2D로 펴서 넣을 것 (Gemm/FULLY_CONNECTED 유도)

3D 입력(`[B,T,D]`)으로 `F.linear`/`nn.Linear`를 부르면 ONNX가 `MatMul`
(→ TFLite `BATCH_MATMUL`)로 export한다. TFLite의 dynamic-range 양자화는
`FULLY_CONNECTED`/`CONV_2D`로 인식된 가중치만 압축하므로 **이러면 디코더가 사실상
압축이 안 된다**(실측 130.7MB → 130.5MB).

그래서 모든 linear 적용 전에 `reshape(B*T, D)`로 2D로 펴고 끝나면 3D로 되돌린다.
수학적으로 동일 연산이고 export 표현만 바뀌는데, 이것만으로 **디코더 130.7MB → 35.3MB,
일괄캐시 115.0MB → 29.9MB**로 압축된다(둘 다 ~73% 절감).

ONNX op 분포로 확인 가능: `Gemm`이 80개면 정상. 남아있는 `MatMul` 33개는 실제 attention
score 계산(가중치 없는 연산)이라 정상이다.

## §6. cross-attention K,V는 이미지당 1회만 계산

`memory`(인코더 출력)는 디코딩 내내 고정이므로 cross-attention K,V도 한 번만 계산하면 된다.
`_MemoryKVWrapper`를 별도 그래프(`decoder_memkv_INT8.tflite`)로 분리해서 이미지당 1회
호출하고, 그 결과를 디코더 step/일괄캐시 그래프에 입력으로 넘긴다
(production PyTorch의 `precompute_memory_kv()`와 동일한 설계).

원래는 그래프를 간단히 유지하려고 스텝마다 `memory`에서 다시 projection했는데, 레이어 8개
× 스텝 수(~100)만큼 반복되는 낭비였다. 고치니 **TFLite FP32 10.6초 → 4.2초,
Hybrid 8.0초 → 3.4초**로 PyTorch 대비 배율이 2.5~3.5배에서 1.3배로 줄었다.

## §7. 한 위치만 고쳐 넣는 건 수학적으로 불가능 — 일괄 재계산이 필요한 이유

`InlineTimeCorrector`(마디 중간 박자표 재추정)는 이미 지나간 위치의 토큰을 되돌려 고친다.
그런데 attention은 매 레이어에서 앞선 모든 위치를 섞기 때문에, 한 위치의 토큰을 바꾸면
그 뒤 모든 위치의 hidden state가 이론적으로 전부 달라져야 한다 — **캐시 한 슬롯만 패치하는
건 원리적으로 불가능**하다.

그래서 `_BulkCaptureWrapperKV`(고정 길이 `chunk_len` 청크를 causal mask로 한 번에 처리)로
"교정된 첫 마디 구간 전체를 한 번에 캐시에 다시 채우는" 방식을 쓴다. PyTorch의
`forward_bulk_capture()`와 같은 목적이다.

`chunk_len`보다 실제 길이가 짧으면 뒤쪽은 `PAD_ID`로 채운다 — causal masking이라 패딩 위치는
앞쪽 실제 위치의 계산에 영향을 주지 않는다(뒤쪽만 보고, 패딩은 항상 뒤쪽에 있으므로).

이 3단계 하이브리드 디코딩(첫 마디 순차+교정 → 일괄 재계산 → 빠른 경로)을 이식했더니
6/8박자 곡 정확도가 63.0% → 97.8%로 회복됐다.

## §8. onnx2tf가 축 순서를 뒤섞는 것 막기

onnx2tf는 이미지가 아닌 3차원 이상 텐서를 이미지로 오인해서 NCHW→NHWC 자동 전치를 해버린다
(예: Transformer의 `[1,320,512]` memory가 `[1,512,1]`로 깨짐, 5D 캐시 텐서가
`[8,1,8,300,64]` 대신 `[8,8,300,64,1]`로 나옴).

`_convert_tflite(keep_layout_input_names=[...])`로 전치하면 안 되는 입력 이름을 명시한다.
**새 입력을 추가할 때마다 이 목록에도 넣어야 한다.**

## §9. FP16은 이 그래프에서 CPU 실행 불가 (미해결)

onnx2tf의 float16 변환은 "가중치만 fp16 + 활성값 fp32"가 아니라 **그래프 전체(입력 텐서
포함)를 fp16**으로 만든다 — GPU 델리게이트 전용 설계다. CPU 인터프리터에는 fp16 활성값을
받는 커널이 없어서 런타임에 아예 실행이 안 된다.

§5의 Gemm 유도를 적용해도 실패한다(에러 지점만 `batch_matmul.cc` → `fully_connected.cc`로
이동할 뿐, 활성값까지 캐스팅하는 근본 구조는 안 바뀜). `--fp16` 플래그는 나중에 실기
GPU 델리게이트 테스트용으로만 남겨뒀다.

XNNPACK의 `xnnpack_fp16_compute` 델리게이트 플래그는 해법이 아니다 — 우리가 만난 건
델리게이트 선택 문제가 아니라 **코어 커널의 dtype 하드 체크**라서.

## §10. 인코더는 dynamic-range 양자화하면 발산한다

인코더(CoordConv CNN)를 dynamic-range로 양자화하면 압축은 되지만(54.5MB → 13.7MB)
**출력이 수치적으로 발산**한다(정상 `mean=0.5477, std=0.8324` → `mean≈3.35e28, std=inf`).
원인 불명이고, 인코더는 전체 용량의 절반 미만이라 이득 대비 조사 비용이 커서 **FP32로 고정**했다.

그래서 `--dynamic_range`는 디코더/일괄캐시/memkv에만 적용되고 인코더는 이 플래그와
무관하게 항상 FP32다("Hybrid" 구성).

## §11. 인코더 입력 shape은 고정 — 매번 재할당하지 말 것

`extract_system_canvas()`는 항상 같은 캔버스 크기를 반환하므로 인코더 입력 shape은 고정이다.
`tflite_infer.py: encode()`가 호출마다 `resize_tensor_input()` + `allocate_tensors()`를
다시 부르고 있었는데, FP32에서는 저비용이라 안 드러나다가 dynamic-range 인코더에서
XNNPACK이 매번 가중치를 재포장하며 **1장에 300초 이상** 걸리는 걸로 터졌다.

shape이 실제로 달라질 때만 재할당하도록 고쳤고, 이 수정만으로 FP32 경로도 15.6초 → 10.6초로
빨라졌다.

## §12. 인터프리터 스레드 수는 인코더/디코더가 정반대

- **인코더**(이미지당 1회, 큰 CNN): 스레드가 많을수록 빠름 (18코어 기준 16까지 계속 개선)
- **디코더**(스텝마다, 토큰 1개짜리 작은 연산): 스레드가 적을수록 빠름 — 동기화 오버헤드가
  병렬화 이득을 압도

그래서 기본값이 `enc_threads=os.cpu_count()`, `dec_threads=1`이다.

다만 이 개발 PC의 정식 벤치마크에서는 런투런 변동이 커서(같은 코드 연속 2회에 34% 차이)
개선폭을 숫자로 확정하지 못했다 — 원리만 채택하고 수치는 실기 측정에서 재확인할 것.

## §13. SegNet export는 현재 추론 경로와 무관

현재 오선 검출은 학습 모델이 아니라 순수 OpenCV(`detect_staffs()`)로 한다. SegNet은 폐기된
옛 C++ 모바일 엔진 전용이었고, `--segnet`을 안 주면 export를 건너뛴다(기본 `None`).
모바일 파이프라인에서 SegNet을 다시 쓸지는 미결정 상태라 코드만 남겨뒀다.

## §14. 버킷팅 — self-attn 고정 비용을 줄이는 두 번째 캐시 크기 (2026-08-26)

§3에서 self-attention을 고정 크기 버퍼로 만든 대가는 "매 스텝 `pos`와 무관하게 항상
`O(cache_len)`을 계산한다"는 것이다(PyTorch는 `O(pos)`). `cache_len=300`은
`INFER_MAX_LEN`과 맞춰 잡은 안전판인데, **실제로 그 크기가 필요한 곡은 드물다** —
newage01~30(30곡, 소나타류 제외 — 학습 데이터가 더 복잡해서 테스트셋에선 제외) 실측 결과
29/30곡이 133 이하였다(하나만 237, 근데 그 곡도 모델이 실제로 예측하는 길이는 125라 결국
안 넘음). 그래서 그래프를 **두 크기로 나눠서 export**한다:

- **작은 버킷**(`cache_len_small`, 기본 160) — `decoder_INT8.tflite`/`decoder_bulk_INT8.tflite`.
  대부분의 곡이 여기서 안 넘치고 끝난다(300 대비 self-attn 비용 47%).
- **큰 버킷**(`cache_len`, 기본 300, 기존과 동일) — `decoder_large_INT8.tflite`. 작은 버킷이
  꽉 찼을 때만 쓰는 안전판.
- **리버킷 그래프**(`decoder_rebucket_INT8.tflite`) — `_BulkCaptureWrapperKV`를
  `chunk_len=cache_len_small`로 재사용한 것. 작은 버킷이 꽉 찼을 때 지금까지 생성된 전체
  시퀀스(정확히 `cache_len_small`개)를 한 번에 큰 버킷 캐시로 옮겨 담는다. §7의 일괄
  재계산과 원리가 완전히 같다(한 위치만 옮기는 게 아니라 전체를 다시 계산) — 다만 여긴
  교정이 아니라 "그릇을 옮겨 담는" 목적이라는 차이만 있다.

**정확도 검증**: 리버킷은 실측 30곡에서 한 번도 실제로 발동하지 않았다(다 160 안에서
끝남) — 그래서 강제로 트리거시켜 직접 텐서 비교로 검증했다. 순차 계산 대비 오차가 있었지만
(최대 0.12, 레이어당 완만히 누적), **이미 프로덕션에 있던 `chunk_len=40` 일괄캐시
그래프도 순차 계산 대비 정확히 같은 크기의 오차를 보인다** — dynamic-range 양자화가 그래프
마다 독립적으로 스케일을 잡아서 생기는, 이미 받아들여진 수준의 노이즈이지 새 버그가
아니다.

**재검증이 필요한 시점**: `cache_len_small=160`은 *현재* 체크포인트의 실제 디코딩 길이
분포로 정한 값이다. 체크포인트가 바뀌면(예: QAT 재학습) 예측 길이 분포도 바뀔 수 있으니
newage01~30로 다시 측정해서 160이 여전히 안전한지 확인할 것 — 스크립트는
`train/tools/eval_exactpicture.py` 패턴을 참고해 재사용 가능.
