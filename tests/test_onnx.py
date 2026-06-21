"""완료조건 2 — ONNX Runtime 동작 증명(유닛테스트).

학습된 체크포인트 없이도 *결정론적으로* 검증한다: 작은 랜덤 모델을 ONNX 로 export →
onnxruntime 으로 로드·추론 → PyTorch 출력과 수치 일치(parity). 이로써 "라리엔 클라가
onnxruntime 으로 본 모델을 돌릴 수 있다"를 증명한다. int8 양자화 모델도 로드·추론된다.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcm.export_onnx import PAD_LEN, ExportWrapper  # noqa: E402
from lcm.model import LcmEncoder  # noqa: E402
from lcm.schema import LabelSpace, load_ssot  # noqa: E402

LS = LabelSpace(load_ssot())


def _tiny_model() -> LcmEncoder:
    torch.manual_seed(0)
    m = LcmEncoder(vocab_size=256, ls=LS, pad_id=0, d_model=32, n_layers=1,
                   n_heads=2, d_ff=64, max_len=64)
    m.eval()
    return m


def _export(model, path: Path, dynamo: bool = True) -> list[str]:
    wrap = ExportWrapper(model)
    head_names = [n for n, _, _ in model.head_specs]
    ids = torch.full((1, PAD_LEN), 0, dtype=torch.long)
    ids[0, :4] = torch.tensor([5, 9, 12, 3])
    attn = torch.zeros((1, PAD_LEN), dtype=torch.long)
    attn[0, :4] = 1
    # dynamo exporter 는 attention 의 mask/shape 분기를 symbolic 하게 처리해 어떤 입력
    # 길이에도 일반화된다(legacy TorchScript exporter 는 상수 폴딩되어 parity 깨짐).
    torch.onnx.export(
        wrap, (ids, attn), str(path),
        input_names=["input_ids", "attention_mask"], output_names=head_names,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=dynamo)
    return head_names


def test_onnxruntime_parity(tmp_path):
    """export → onnxruntime 추론이 PyTorch 와 수치 일치(ONNX Runtime 동작 증명)."""
    import onnxruntime as ort

    model = _tiny_model()
    onnx_path = tmp_path / "lcm_tiny.onnx"
    head_names = _export(model, onnx_path)

    ids = torch.full((1, PAD_LEN), 0, dtype=torch.long)
    ids[0, :5] = torch.tensor([7, 4, 19, 2, 11])
    attn = torch.zeros((1, PAD_LEN), dtype=torch.long)
    attn[0, :5] = 1

    with torch.no_grad():
        torch_out = ExportWrapper(model)(ids, attn)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": attn.numpy()})

    assert len(ort_out) == len(head_names)
    for i, name in enumerate(head_names):
        np.testing.assert_allclose(
            torch_out[i].numpy(), ort_out[i], atol=1e-3, rtol=1e-3,
            err_msg=f"head '{name}' onnxruntime != torch")


def test_int8_accuracy_preserved(tmp_path):
    """학습된 int8 ONNX 가 fp32(PyTorch) 대비 action 결정을 보존(양자화 배포 안전성).

    체크포인트가 있을 때만(없으면 skip). val 전체에서 int8 onnxruntime argmax 가 PyTorch
    argmax 와 거의 일치(≥0.97)해야 양자화 모델을 안심하고 배포한다.
    """
    import pytest
    from lcm.export_onnx import CKPT, PAD_LEN as PL, _load_model
    if not CKPT.exists():
        pytest.skip("학습된 체크포인트 없음")
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from lcm.dataset import read_jsonl

    model, ls, tk, c = _load_model()
    wrap = ExportWrapper(model)
    onnx_path = tmp_path / "m.onnx"
    head_names = _export_fixed(wrap, c, onnx_path)
    int8_path = tmp_path / "m.int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    ai = head_names.index("action")

    val = read_jsonl(Path(__file__).resolve().parents[1] / "data" / "generated" / "val.jsonl")
    agree = 0
    for r in val:
        raw = tk.encode(r["text"]).ids[:PL]
        ids = np.full((1, PL), c["pad_id"], dtype=np.int64); ids[0, :len(raw)] = raw
        attn = np.zeros((1, PL), dtype=np.int64); attn[0, :len(raw)] = 1
        o = sess.run(None, {"input_ids": ids, "attention_mask": attn})
        with torch.no_grad():
            t = wrap(torch.tensor(ids), torch.tensor(attn))
        if int(o[ai].argmax()) == int(t[ai].argmax()):
            agree += 1
    rate = agree / len(val)
    assert rate >= 0.97, f"int8 vs fp32 action 일치 {rate:.3f} < 0.97 — 양자화 손실 과다"


def _export_fixed(wrap, c, path: Path) -> list[str]:
    """고정 길이 int8 호환 export(legacy)."""
    head_names = wrap.head_names
    pids = torch.full((1, PAD_LEN), c["pad_id"], dtype=torch.long)
    pattn = torch.ones((1, PAD_LEN), dtype=torch.long)
    torch.onnx.export(
        wrap, (pids, pattn), str(path),
        input_names=["input_ids", "attention_mask"], output_names=head_names,
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
        opset_version=17, dynamo=False)
    return head_names


def test_onnxruntime_int8_runs(tmp_path):
    """int8 동적 양자화 모델이 onnxruntime 에서 로드·추론된다(모바일 탑재 검증)."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = _tiny_model()
    onnx_path = tmp_path / "lcm_tiny.onnx"
    head_names = _export(model, onnx_path)
    int8_path = tmp_path / "lcm_tiny.int8.onnx"
    # int8 양자화는 legacy(고정 shape) 그래프에서 안정적이라 재-export.
    head_names = _export(model, onnx_path, dynamo=False)
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    assert int8_path.stat().st_size < onnx_path.stat().st_size  # 더 작아야

    ids = np.zeros((1, PAD_LEN), dtype=np.int64)
    ids[0, :3] = [5, 8, 2]
    attn = np.zeros((1, PAD_LEN), dtype=np.int64)
    attn[0, :3] = 1
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_ids": ids, "attention_mask": attn})
    # action 헤드가 라벨 수만큼 logits 를 낸다(동작·shape 검증).
    action_idx = head_names.index("action")
    assert out[action_idx].shape[-1] == len(LS.actions)
