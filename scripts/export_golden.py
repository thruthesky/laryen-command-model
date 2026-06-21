#!/usr/bin/env python3
"""dart 통합 검증용 golden 산출 — 토큰화(text→ids) + 라벨 공간을 JSON 으로.

라리엔 클라(dart)가 ① ByteLevelBPE 포팅의 정확성(text→ids 일치) ② decode 라벨 매핑을
파이썬과 1:1 검증하는 데 쓴다. 산출물은 artifacts/(gitignore) — dart 통합 시점에 생성해
라리엔 assets 로 가져간다(토크나이저는 재학습 시 vocab 이 바뀌므로 커밋하지 않는다).

사용:  python scripts/export_golden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcm.bpe_ref import BpeRef  # noqa: E402
from lcm.schema import LabelSpace, load_ssot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "golden_tokenize.json"

SAMPLES = [
    "강남에서 사냥", "왼쪽으로 가", "멈춰", "강철 세트 착용", "인벤토리 열어",
    "체력 물약 먹어", "안녕 너 누구야", "plate 세트 입고 사냥", "5시 방향으로 걸어가",
    "강동 꽃밭에서 Bone 사냥하고 체력 30% 아래면 안전지대로 피신", "자동사냥 켜줘",
    "hello", "auto hunt on", "what is a caster", "도착하면 자동공격",
]


def main() -> int:
    ref = BpeRef.load()
    ls = LabelSpace(load_ssot())
    golden = {
        "_note": "dart ByteLevelBPE 포팅 검증용 — bpe_ref.encode_with_special 와 일치해야 함",
        "tokenize": [{"text": t, "ids": ref.encode_with_special(t)} for t in SAMPLES],
        "labels": {name: labels for name, _, labels in ls.heads()},
        "pad_len": 32,
        "threshold": 0.7,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)} — 토큰화 {len(SAMPLES)}건 + 라벨 {len(golden['labels'])}헤드")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
