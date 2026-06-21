#!/usr/bin/env bash
# LCM 전체 파이프라인 재현 — SSOT 추출 → 데이터 → 학습 → ONNX → 벤치 → 평가 → 테스트.
# 사용: bash scripts/run_all.sh [epochs]
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${1:-200}"

echo "▶ 1/7 의존성"
uv sync --extra dev >/dev/null

echo "▶ 2/7 라리엔 SSOT 추출 (dart → config/ssot.json)"
uv run python scripts/sync_ssot.py

echo "▶ 3/7 합성 데이터 생성"
uv run python scripts/gen_data.py

echo "▶ 4/7 학습 (M5 MPS, ${EPOCHS}ep)"
rm -rf artifacts/tokenizer checkpoints/lcm.pt
uv run python -m lcm.train --epochs "${EPOCHS}" --lr 5e-4 --d-model 160 --layers 3

echo "▶ 5/7 ONNX export (fp32 dynamic + int8) + dart golden(토크나이저 갱신 반영)"
uv run python -m lcm.export_onnx 2>&1 | grep -E "✅|⚠️" || true
uv run python scripts/export_golden.py || true   # 재학습 시 토크나이저 변경 → golden 갱신 필수

echo "▶ 6/7 벤치(지연·ECE) + 평가(헤드별·홀드아웃)"
uv run python -m lcm.bench 2>&1 | grep -E "지연|Calibration|ms|ECE" || true
uv run python -m lcm.eval 2>&1 | grep -E "exact_acc|fallback recall" || true

echo "▶ 7/7 테스트"
uv run python -m pytest tests/ -q

echo "✅ LCM 전체 파이프라인 완료"
