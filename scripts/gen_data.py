#!/usr/bin/env python3
"""합성 데이터셋 생성 → data/generated/{train,val}.jsonl.

사용:
  python scripts/gen_data.py                 # 기본 seed, val 10%
  python scripts/gen_data.py --val-ratio 0.15 --seed 7
각 줄: {"text": "<발화>", "intent": {<라리엔 action JSON 1개>}}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcm.schema import load_ssot  # noqa: E402
from lcm.synth import generate  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "generated"
# 사람 개발자가 *직접 타이핑* 하는 추가 학습 샘플(코드 수정 없이 jsonl 한 줄씩).
CUSTOM = Path(__file__).resolve().parents[1] / "data" / "custom_samples.jsonl"


def load_custom() -> list[dict]:
    """data/custom_samples.jsonl 의 사용자 정의 샘플을 읽는다(없으면 빈 리스트).

    한 줄에 하나의 JSON. 빈 줄·'#'/'//' 주석은 무시. 세 형식 지원:
      간편 질문 : {"text": "나는 누구야", "answer_intent": "query_player_name"}
      간편 명령 : {"text": "강남에서 사냥", "action": "hunt"}
      전체     : {"text": "...", "intent": {"action": "...", "semantic_type": "...", "answer_intent": "..."}}
    answer_intent/action 만 주면 나머지(semantic_type 등)는 자동으로 채운다.
    """
    if not CUSTOM.exists():
        return []
    out: list[dict] = []
    for ln in CUSTOM.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        d = json.loads(s)
        t = d["text"]
        if "intent" in d:
            out.append({"text": t, "intent": d["intent"]})
        elif "answer_intent" in d:  # 질문(answer_local) 간편 형식
            out.append({"text": t, "intent": {
                "action": "unknown", "semantic_type": "question",
                "answer_intent": d["answer_intent"]}})
        elif "action" in d:  # 명령(execute) 간편 형식
            out.append({"text": t, "intent": {
                "action": d["action"], "semantic_type": "command",
                "answer_intent": "<none>"}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = generate(load_ssot(), seed=args.seed)
    custom = load_custom()  # 사람이 data/custom_samples.jsonl 에 타이핑한 추가 샘플
    rows += custom
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    val, train = rows[:n_val], rows[n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # action 분포 요약(편향 점검).
    from collections import Counter
    dist = Counter(r["intent"]["action"] for r in rows)
    print(f"✅ 생성 {len(rows)}건(합성 {len(rows) - len(custom)} + 사용자 custom {len(custom)})"
          f" → train {len(train)} / val {len(val)}")
    print("   action 분포:", dict(sorted(dist.items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
