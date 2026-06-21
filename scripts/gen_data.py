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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = generate(load_ssot(), seed=args.seed)
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
    print(f"✅ 생성 {len(rows)}건 → train {len(train)} / val {len(val)}")
    print("   action 분포:", dict(sorted(dist.items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
