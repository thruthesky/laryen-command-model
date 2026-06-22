#!/usr/bin/env bash
# LCM 모델 OTA 배포 — artifacts → VPS /opt/laryen-downloads/files/models/lcm/
# → https://laryen.com/models/lcm/ (sherpa 모델·fast_path 와 같은 정적 호스팅).
#
# 사용: bash scripts/deploy_model.sh   (export_onnx + gen_manifest 이후)
set -euo pipefail

VPS="${VPS_HOST:-root@209.97.169.136}"
DST="${DST:-/opt/laryen-downloads/files/models/lcm}"
cd "$(dirname "$0")/.."

for f in artifacts/lcm.int8.onnx artifacts/lcm-labels.json artifacts/manifest.json \
         artifacts/tokenizer/vocab.json artifacts/tokenizer/merges.txt; do
  [ -f "$f" ] || { echo "❌ 없음: $f (export_onnx + gen_manifest 먼저)"; exit 1; }
done

echo "==> 1) 원격 디렉토리 준비 $DST"
ssh "$VPS" "mkdir -p $DST"

echo "==> 2) 5자산 업로드(rsync)"
rsync -az artifacts/lcm.int8.onnx artifacts/lcm-labels.json artifacts/manifest.json \
  artifacts/tokenizer/vocab.json artifacts/tokenizer/merges.txt "$VPS:$DST/"

echo "==> 3) 원격 파일 확인"
ssh "$VPS" "ls -lh $DST"

echo "==> 4) https://laryen.com/models/lcm/manifest.json 검증"
sleep 1
curl -sS --max-time 15 "https://laryen.com/models/lcm/manifest.json" | head -8 || echo "(manifest fetch 실패 — Traefik 라우팅 확인)"
echo ""
echo "✅ LCM 모델 배포 완료"
