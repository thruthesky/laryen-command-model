"""ONNX export — 라리엔 클라(onnxruntime) 탑재용.

라리엔 클라는 sherpa_onnx 가 이미 onnxruntime 을 들고 있어 *추가 런타임 0* 으로 본
모델을 추론한다. 입력은 토큰 id(int64)·attention mask, 출력은 헤드별 logits.

두 산출물:
  - lcm.onnx       : dynamo exporter, **dynamic seq**(정확 — 어떤 입력 길이에도 일반화).
                     주 배포 산출물. attention 의 mask/shape 분기를 symbolic 처리한다.
  - lcm.int8.onnx  : legacy exporter + 고정 길이(PAD_LEN) + int8 동적 양자화(가장 작음).
                     **입력을 항상 PAD_LEN 으로 패딩**해 써야 한다(legacy 는 trace 길이에
                     그래프가 고정되므로). 크기 우선일 때 채택.

사용:  python -m lcm.export_onnx
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import LcmEncoder
from .schema import LabelSpace, load_ssot
from .tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "lcm.pt"
OUT = ROOT / "artifacts" / "lcm.onnx"
OUT_INT8 = ROOT / "artifacts" / "lcm.int8.onnx"
PAD_LEN = 32  # int8(고정 길이) 입력 길이 — 라리엔 클라와 일치(짧은 명령 발화에 충분)


class ExportWrapper(torch.nn.Module):
    """forward(ids, mask) → tuple(고정 순서 logits). ONNX 는 dict 출력이 까다로워 튜플로."""

    def __init__(self, model: LcmEncoder):
        super().__init__()
        self.model = model
        self.head_names = [n for n, _, _ in model.head_specs]

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask.bool())
        return tuple(out[n] for n in self.head_names)


def _load_model() -> tuple[LcmEncoder, LabelSpace, object, dict]:
    ls = LabelSpace(load_ssot())
    tk = load_tokenizer()
    blob = torch.load(CKPT, map_location="cpu")
    c = blob["config"]
    model = LcmEncoder(c["vocab_size"], ls, pad_id=c["pad_id"],
                       d_model=c["d_model"], n_layers=c["layers"], max_len=c["max_len"])
    model.load_state_dict(blob["model"])
    model.eval()
    return model, ls, tk, c


def main() -> int:
    model, ls, tk, c = _load_model()
    wrap = ExportWrapper(model)
    head_names = wrap.head_names
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── (1) dynamic seq fp32 — 주 배포(정확) ─────────────────────────────────
    ids = torch.full((1, PAD_LEN), c["pad_id"], dtype=torch.long)
    ids[0, :4] = torch.tensor([5, 9, 12, 3])
    attn = torch.zeros((1, PAD_LEN), dtype=torch.long)
    attn[0, :4] = 1
    torch.onnx.export(
        wrap, (ids, attn), str(OUT),
        input_names=["input_ids", "attention_mask"], output_names=head_names,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      **{n: {0: "batch"} for n in head_names}},
        opset_version=17, dynamo=True)
    print(f"✅ ONNX(dynamic) → {OUT.relative_to(ROOT)}")

    # ── (2) 고정 길이 int8 — 크기 우선(best-effort) ──────────────────────────
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        fixed = OUT.with_suffix(".fixed.onnx")
        pids = torch.full((1, PAD_LEN), c["pad_id"], dtype=torch.long)
        pattn = torch.ones((1, PAD_LEN), dtype=torch.long)
        torch.onnx.export(
            wrap, (pids, pattn), str(fixed),
            input_names=["input_ids", "attention_mask"], output_names=head_names,
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
            opset_version=17, dynamo=False)
        quantize_dynamic(str(fixed), str(OUT_INT8), weight_type=QuantType.QInt8)
        fixed.unlink(missing_ok=True)
        print(f"✅ int8(고정 {PAD_LEN}) → {OUT_INT8.relative_to(ROOT)} "
              f"({OUT_INT8.stat().st_size/1024:.0f} KB)")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ int8 양자화 건너뜀: {e}")

    # 배포용 라벨 사전(OTA — onnx 와 함께 배포). 라리엔 클라가 헤드 인덱스→이름 디코드에 사용.
    import json
    labels_out = {
        "labels": {n: lab for n, _, lab in ls.heads()},
        "head_specs": [{"name": n, "kind": k} for n, k, _ in ls.heads()],
        "threshold": 0.8, "pad_len": PAD_LEN, "pad_id": c["pad_id"],
    }
    (ROOT / "artifacts" / "lcm-labels.json").write_text(
        json.dumps(labels_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 라벨 사전 → artifacts/lcm-labels.json ({len(labels_out['labels'])} 헤드)")

    # ── (3) onnxruntime 검증 — 두 산출물 모두 PyTorch 와 일치(고정 PAD_LEN 패딩) ──
    def encode_fixed(text: str):
        raw = tk.encode(text).ids[:PAD_LEN]
        a_ids = np.full((1, PAD_LEN), c["pad_id"], dtype=np.int64)
        a_ids[0, : len(raw)] = raw
        a_attn = np.zeros((1, PAD_LEN), dtype=np.int64)
        a_attn[0, : len(raw)] = 1
        return a_ids, a_attn

    ai = head_names.index("action")
    # 검증용 PyTorch 기준은 export 가 건드리지 않은 *새* 모델로(dynamo export 가 원본
    # wrap 의 추론 상태에 영향을 줄 수 있어 fresh load).
    vwrap = ExportWrapper(_load_model()[0])
    for label, path, atol in (("dynamic fp32", OUT, 1e-3), ("int8", OUT_INT8, 0.6)):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            okc = 0
            for text in ("강남에서 사냥", "왼쪽으로 가", "멈춰", "강철 세트 착용", "인벤토리 열어"):
                a_ids, a_attn = encode_fixed(text)
                res = sess.run(None, {"input_ids": a_ids, "attention_mask": a_attn})
                with torch.no_grad():
                    t_out = vwrap(torch.tensor(a_ids), torch.tensor(a_attn))
                if int(res[ai].argmax()) == int(t_out[ai].argmax()):
                    okc += 1
                np.testing.assert_allclose(res[ai], t_out[ai].numpy(), atol=atol, rtol=0.2)
            print(f"✅ {label} 검증 — action argmax 일치 {okc}/5, logit parity(atol={atol})")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ {label} 검증 실패: {str(e).strip().splitlines()[0][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
