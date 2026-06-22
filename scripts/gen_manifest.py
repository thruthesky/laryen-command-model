#!/usr/bin/env python3
"""OTA 배포 manifest 생성 — laryen.com/models/lcm/manifest.json.

3자산(model·tokenizer·labels)의 sha256/bytes + 버전 메타를 채운다(OTA-DEPLOY.md SSOT).
빌드 산출물(artifacts/)에서 자동 생성하므로 export 후 실행한다.

버전 정책(다른 팀 S2 결정):
  - version        : config/model_version.txt(사람이 재훈련 시 올림 — 패키지 버전과 분리)
  - schema_version : ls.heads() 시그니처 해시(action/slot 구조 변경 자동 감지)
  - ssot_hash      : config/ssot.json sha256(클라 SSOT 동기 표기)
  - min_app_version: LCM decoder 가 들어간 앱 버전(통합 시 갱신)

사용:  python scripts/gen_manifest.py   (export_onnx 이후)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcm.schema import LabelSpace, load_ssot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
BASE_URL = "https://laryen.com/models/lcm/"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ls = LabelSpace(load_ssot())
    heads_sig = json.dumps([[n, k, len(lab)] for n, k, lab in ls.heads()], ensure_ascii=False)
    schema_version = hashlib.sha256(heads_sig.encode()).hexdigest()[:12]
    ssot_hash = _sha(ROOT / "config" / "ssot.json")
    version = (ROOT / "config" / "model_version.txt").read_text(encoding="utf-8").strip()

    # 배포 자산(모두 models/lcm/ 평면 배치) — 원자 세트.
    assets = {"model": ART / "lcm.int8.onnx",
              "vocab": ART / "tokenizer" / "vocab.json",
              "merges": ART / "tokenizer" / "merges.txt",
              "labels": ART / "lcm-labels.json"}
    missing = [k for k, p in assets.items() if not p.exists()]
    if missing:
        sys.exit(f"❌ 자산 없음: {missing} — export_onnx 먼저 실행")

    files = {k: {"name": p.name, "sha256": _sha(p), "bytes": p.stat().st_size}
             for k, p in assets.items()}
    manifest = {
        "version": version,
        "schema_version": schema_version,
        "channel": "stable",
        "min_app_version": "1.3.9",      # LCM decoder 통합 앱 버전(통합 시 갱신)
        "min_lcm_version": "0.0.0",      # 강제 sunset 시에만 사람이 올림
        "ssot_hash": ssot_hash,
        "base_url": BASE_URL,
        "files": files,
        "threshold": 0.8,
        "pad_len": 32,
    }
    out = ART / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ manifest → {out.relative_to(ROOT)}  "
          f"(v{version}·schema {schema_version}·{len(files)}자산)")
    for k, f in files.items():
        print(f"   {k:7s} {f['name']:18s} {f['bytes']:>9,}B  {f['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
