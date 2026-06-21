"""ONNX export + int8 동적 양자화 — 라리엔 클라(onnxruntime) 탑재용.

라리엔 클라는 sherpa_onnx 가 이미 onnxruntime 을 들고 있어 *추가 런타임 0* 으로 본
모델을 추론한다. 입력은 토큰 id(int64)·attention mask, 출력은 헤드별 logits.

사용:  python -m lcm.export_onnx        # checkpoints/lcm.pt → artifacts/lcm.onnx (+ .int8.onnx)
"""
from __future__ import annotations

from pathlib import Path

import torch

from .model import LcmEncoder
from .schema import LabelSpace, load_ssot
from .tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "lcm.pt"
OUT = ROOT / "artifacts" / "lcm.onnx"
OUT_INT8 = ROOT / "artifacts" / "lcm.int8.onnx"
PAD_LEN = 32  # 고정 입력 길이(라리엔 클라와 일치 — 짧은 명령 발화에 충분)


class ExportWrapper(torch.nn.Module):
    """forward(dict) → tuple(고정 순서 logits). ONNX 는 dict 출력이 까다로워 튜플로."""

    def __init__(self, model: LcmEncoder):
        super().__init__()
        self.model = model
        self.head_names = [n for n, _, _ in model.head_specs]

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask.bool())
        return tuple(out[n] for n in self.head_names)


def main() -> int:
    ls = LabelSpace(load_ssot())
    tk = load_tokenizer()
    blob = torch.load(CKPT, map_location="cpu")
    c = blob["config"]
    model = LcmEncoder(c["vocab_size"], ls, pad_id=c["pad_id"],
                       d_model=c["d_model"], n_layers=c["layers"], max_len=c["max_len"])
    model.load_state_dict(blob["model"])
    model.eval()
    wrap = ExportWrapper(model)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 고정 길이(PAD_LEN) 입력 — 명령 발화는 짧아 항상 32 패딩한다(라리엔 클라도 동일).
    # 고정 shape 라 legacy exporter 로 양자화·추론이 안정적이고 모바일에서 더 빠르다.
    pad = c["pad_id"]
    dummy_ids = torch.full((1, PAD_LEN), pad, dtype=torch.long)
    dummy_attn = torch.ones((1, PAD_LEN), dtype=torch.long)
    head_names = [n for n, _, _ in model.head_specs]
    torch.onnx.export(
        wrap, (dummy_ids, dummy_attn), str(OUT),
        input_names=["input_ids", "attention_mask"],
        output_names=head_names,
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                      **{n: {0: "batch"} for n in head_names}},
        opset_version=17,
        dynamo=False,  # 고정 shape + 양자화 호환(TorchScript exporter)
    )
    print(f"✅ ONNX → {OUT.relative_to(ROOT)} (고정 길이 {PAD_LEN})")

    # int8 동적 양자화(가중치 int8 — 임베딩·linear 위주라 효과 큼).
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(OUT), str(OUT_INT8), weight_type=QuantType.QInt8)
        size = OUT_INT8.stat().st_size / 1024
        print(f"✅ int8 → {OUT_INT8.relative_to(ROOT)} ({size:.0f} KB)")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 양자화 건너뜀: {e}")

    # onnxruntime 추론 검증.
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(OUT_INT8), providers=["CPUExecutionProvider"])
        raw = tk.encode("강남에서 사냥").ids[:PAD_LEN]
        ids = np.full((1, PAD_LEN), c["pad_id"], dtype=np.int64)
        ids[0, : len(raw)] = raw
        attn = np.zeros((1, PAD_LEN), dtype=np.int64)
        attn[0, : len(raw)] = 1
        res = sess.run(None, {"input_ids": ids, "attention_mask": attn})
        action_logits = res[head_names.index("action")]
        print(f"✅ onnxruntime 검증 OK — action argmax={int(action_logits.argmax())} "
              f"({ls.actions[int(action_logits.argmax())]})")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ onnxruntime 검증 건너뜀: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
