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

# golden 회귀 fixture 텍스트 — 4언어 대표(question/command/ambiguous). backbone_heads_test 가
# 이 pooled+기대라벨로 Dart heads.forward 정합을 검증한다. export 가 onnx/heads 와 *같은
# 체크포인트*에서 golden 도 재생성하므로 셋이 항상 동기 → golden stale(가짜 경보) 원천 차단.
GOLDEN_TEXTS = [
    "물약 효과 뭐야", "강남으로 가", "거기로 가서 그거 잡아",
    "药水有什么效果", "ポーションの効果は", "사냥해",
]


def _export_golden(model: "BackboneLcm", tk) -> None:
    """onnx/heads 와 같은 체크포인트에서 golden(pooled + 기대 라벨)을 생성한다.

    e5 가 fine-tuned 라 golden(구 e5 pooled)이 낡으면 heads_test 가 *가짜로* 실패한다
    (2026-06-25 "26"/"타타타" 오진 회고). 그래서 export 마다 항상 함께 재생성한다.
    """
    golden = []
    for t in GOLDEN_TEXTS:
        enc = tk(t, return_tensors="pt", truncation=True, max_length=48)
        with torch.no_grad():
            out = model.backbone(input_ids=enc["input_ids"],
                                 attention_mask=enc["attention_mask"]).last_hidden_state
            logits = model(enc["input_ids"], enc["attention_mask"])
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = ((out * m).sum(1) / m.sum(1).clamp_min(1.0))[0].tolist()
        golden.append({
            "text": t, "pooled": pooled,
            "action": int(logits["action"].argmax()),
            "semantic_type": int(logits["semantic_type"].argmax()),
            "answer_intent": int(logits["answer_intent"].argmax()),
        })
    json.dump(golden, open("artifacts/backbone_heads_golden.json", "w"),
              ensure_ascii=False)
    print(f"✅ golden {len(golden)}건 재생성(onnx/heads 와 같은 체크포인트 — stale 불가)")


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

    # 2.5) golden 회귀 fixture 자동 재생성 — onnx/heads 와 *같은 체크포인트*에서.
    #   이 한 줄이 "backbone 재학습했는데 golden 재생성 깜빡" → stale 가짜 경보를 원천 차단한다.
    _export_golden(model, AutoTokenizer.from_pretrained(BACKBONE))

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
