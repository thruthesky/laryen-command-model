"""Backbone → 온디바이스 자산: e5 transformer int8(optimum) + 헤드 weight JSON.

**왜 분리인가**: 멀티헤드 통합 ONNX 는 ORT quantizer 의 shape inference 가 깨진다(멀티 output
384 vs 10). e5 transformer 만 optimum 으로 int8(검증 경로, 470MB→118MB) 하고, 17개 헤드는
작은 Linear 라 weight JSON(0.7MB)으로 분리한다. 클라 추론 파이프라인:
  SentencePiece(XLM-R) → e5 int8 ONNX → mean pooling([B,seq,384]→[B,384])
    → 17개 헤드 행렬곱(JSON weight) → decode_intent → 5-route.

산출(artifacts/):
  - e5_int8/model_quantized.onnx  : e5 transformer int8 (~118MB, OTA)
  - e5_ft/                        : SentencePiece 토크나이저(클라 동봉/OTA)
  - lcm_backbone_heads.json       : 17 헤드 weight/bias + 라벨 + 메타 (~0.7MB, OTA)

사용: HF_HOME=.ai_models/huggingface .venv/bin/python -m lcm.backbone_export
"""
from __future__ import annotations

import json
import os

import torch

from .backbone_train import BACKBONE, CKPT, BackboneLcm
from .schema import LabelSpace, load_ssot

MAX_ONDEVICE_MB = 200  # 🛑 온디바이스 용량 상한(2026-06-24 사용자 지시).


def main() -> int:
    ls = LabelSpace(load_ssot())
    model = BackboneLcm(ls)
    blob = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(blob["model"])
    model.eval()

    # 1) fine-tuned e5 backbone 을 HF 형식으로 저장 → optimum 표준 int8 경로.
    from transformers import AutoTokenizer
    model.backbone.save_pretrained("artifacts/e5_ft")
    AutoTokenizer.from_pretrained(BACKBONE).save_pretrained("artifacts/e5_ft")

    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    m = ORTModelForFeatureExtraction.from_pretrained("artifacts/e5_ft", export=True)
    m.save_pretrained("artifacts/e5_onnx")
    q = ORTQuantizer.from_pretrained("artifacts/e5_onnx")
    qc = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    q.quantize(save_dir="artifacts/e5_int8", quantization_config=qc)

    # 2) 17개 헤드 weight/bias → JSON (클라 dart 행렬곱).
    heads = {}
    for name, kind, labels in ls.heads():
        lin = model.heads[name]
        heads[name] = {
            "kind": kind, "labels": labels,
            "weight": [[round(x, 5) for x in row] for row in lin.weight.detach().tolist()],
            "bias": [round(x, 5) for x in lin.bias.detach().tolist()],
        }
    meta = {
        "backbone": BACKBONE, "hidden": model.backbone.config.hidden_size,
        "pooling": "mean", "max_len": 48, "heads": heads,
        "head_order": [n for n, _, _ in ls.heads()],
        "semantic_types": ls.semantic_types, "answer_intents": ls.answer_intents,
    }
    hp = "artifacts/lcm_backbone_heads.json"
    json.dump(meta, open(hp, "w"), ensure_ascii=False)

    # 3) 용량 가드.
    int8 = "artifacts/e5_int8/model_quantized.onnx"
    int8_mb = os.path.getsize(int8) / 1e6
    head_mb = os.path.getsize(hp) / 1e6
    total = int8_mb + head_mb
    print(f"✅ e5 int8 {int8_mb:.0f}MB + 헤드 {head_mb:.1f}MB = 총 {total:.0f}MB")
    if total > MAX_ONDEVICE_MB:
        print(f"🛑 경고(사람 개발자): 온디바이스 총량 {total:.0f}MB > {MAX_ONDEVICE_MB}MB — 더 작은 backbone 필요.")
    else:
        print(f"✅ 온디바이스 용량 가드 통과: {total:.0f}MB ≤ {MAX_ONDEVICE_MB}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
