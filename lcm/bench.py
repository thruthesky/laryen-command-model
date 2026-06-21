"""추론 지연 벤치마크 + confidence calibration(ECE) — 모바일 실용성·신뢰도 정량.

- 지연: int8 ONNX 를 onnxruntime(CPU)으로 N회 추론 → p50/p95 ms. 라리엔 클라가 발화당
  들이는 비용의 상한 근사(모바일 CPU 는 데스크톱보다 느리나 모델이 작아 충분).
- ECE(Expected Calibration Error): confidence 가 실제 정확도와 얼마나 맞는지. 3계층
  threshold 판정의 신뢰 근거(label smoothing 으로 낮췄어야 — iter3).

사용:  python -m lcm.bench
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import LcmDataset
from .model import LcmEncoder, action_confidence
from .schema import LabelSpace, load_ssot
from .tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
CKPT = ROOT / "checkpoints" / "lcm.pt"
INT8 = ROOT / "artifacts" / "lcm.int8.onnx"
PAD_LEN = 32


def bench_latency(tk, c, n: int = 200) -> None:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(INT8), providers=["CPUExecutionProvider"])
    raw = tk.encode("강동 꽃밭에서 Bone 사냥하고 체력 30% 피신").ids[:PAD_LEN]
    ids = np.full((1, PAD_LEN), c["pad_id"], dtype=np.int64)
    ids[0, : len(raw)] = raw
    attn = np.zeros((1, PAD_LEN), dtype=np.int64)
    attn[0, : len(raw)] = 1
    feed = {"input_ids": ids, "attention_mask": attn}
    for _ in range(10):  # warmup
        sess.run(None, feed)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, feed)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    print(f"[추론 지연] int8 onnxruntime CPU, n={n}")
    print(f"  p50 {statistics.median(ts):.2f}ms  p95 {ts[int(n*0.95)]:.2f}ms  "
          f"mean {statistics.mean(ts):.2f}ms")


def compute_ece(model, ls, tk, n_bins: int = 10) -> tuple[float, float, float]:
    """val set ECE + 평균 confidence + 평균 정확도(테스트/리포트 공용)."""
    ds = LcmDataset(DATA / "val.jsonl", tk, ls, max_len=32)
    confs, accs = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            ids = torch.tensor([item["input_ids"]], dtype=torch.long)
            attn = torch.ones_like(ids, dtype=torch.bool)
            logits = model(ids, attn)
            confs.append(action_confidence(logits))
            accs.append(int(int(logits["action"].argmax(-1)) == item["labels"]["action"]))
    confs, accs = np.array(confs), np.array(accs)
    ece = 0.0
    for b in range(n_bins):
        m = (confs > b / n_bins) & (confs <= (b + 1) / n_bins)
        if m.sum():
            ece += m.mean() * abs(accs[m].mean() - confs[m].mean())
    return float(ece), float(confs.mean()), float(accs.mean())


def bench_ece(model, ls, tk, n_bins: int = 10) -> None:
    ece, mc, ma = compute_ece(model, ls, tk, n_bins)
    print(f"[Calibration] ECE={ece:.3f}  (낮을수록 confidence 가 실제 정확도와 일치)")
    print(f"  평균 confidence {mc:.3f}  평균 정확도 {ma:.3f}")


def main() -> int:
    ls = LabelSpace(load_ssot())
    tk = load_tokenizer()
    blob = torch.load(CKPT, map_location="cpu")
    c = blob["config"]
    model = LcmEncoder(c["vocab_size"], ls, pad_id=c["pad_id"], d_model=c["d_model"],
                       n_layers=c["layers"], max_len=c["max_len"])
    model.load_state_dict(blob["model"])
    model.eval()
    if INT8.exists():
        bench_latency(tk, c)
    else:
        print("⚠️ int8 ONNX 없음 — python -m lcm.export_onnx 먼저")
    bench_ece(model, ls, tk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
