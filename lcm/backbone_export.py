"""Backbone LCM → ONNX int8 (온디바이스 OTA). 멀티헤드 dict 출력을 tuple 로 펴서 export.

산출: artifacts/lcm_backbone.onnx(fp32) · lcm_backbone.int8.onnx(동적 양자화) · 헤드 순서 메타.
크기를 출력해 200M/온디바이스 적합성을 확인한다(int8 ≈ params MB).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .backbone_train import BACKBONE, CKPT, BackboneLcm
from .schema import LabelSpace, load_ssot

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FP32 = ART / "lcm_backbone.onnx"
INT8 = ART / "lcm_backbone.int8.onnx"
META = ART / "lcm_backbone.meta.json"


class ExportWrapper(torch.nn.Module):
    """dict 출력 → 고정 순서 tuple (ONNX output_names 와 1:1)."""

    def __init__(self, model: BackboneLcm, head_names: list[str]):
        super().__init__()
        self.model = model
        self.head_names = head_names

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask)
        return tuple(out[n] for n in self.head_names)


def main() -> int:
    ls = LabelSpace(load_ssot())
    model = BackboneLcm(ls)
    blob = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(blob["model"])
    model.eval()
    head_names = [n for n, _, _ in ls.heads()]
    wrapper = ExportWrapper(model, head_names)

    ART.mkdir(parents=True, exist_ok=True)
    dummy_ids = torch.ones(1, 16, dtype=torch.long)
    dummy_mask = torch.ones(1, 16, dtype=torch.long)
    torch.onnx.export(
        wrapper, (dummy_ids, dummy_mask), str(FP32),
        input_names=["input_ids", "attention_mask"], output_names=head_names,
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}},
        opset_version=17)
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8)

    META.write_text(json.dumps({
        "backbone": BACKBONE, "head_order": head_names,
        "head_specs": [(n, k, lbls) for n, k, lbls in ls.heads()],
        "max_len": 48,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    fp32_mb = os.path.getsize(FP32) / 1e6
    int8_mb = os.path.getsize(INT8) / 1e6
    print(f"✅ ONNX export — fp32 {fp32_mb:.0f}MB → int8 {int8_mb:.0f}MB  (헤드 {len(head_names)})")
    if int8_mb > 200:
        print(f"🛑 경고(사람 개발자): int8 {int8_mb:.0f}MB > 200MB — 온디바이스 부적합, 더 작은 backbone 필요.")
    else:
        print(f"✅ 온디바이스 용량 가드: int8 {int8_mb:.0f}MB ≤ 200MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
