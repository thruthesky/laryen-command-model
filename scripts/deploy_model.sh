#!/usr/bin/env bash
# LCM 모델 OTA 배포 — staging/production 분리(zone 서버 v5/sg 와 *동일* 철학).
#   staging    : /models/lcm-staging/  → AI 자율(실사용자 무영향). ENV=staging 빌드가 받는다.
#   production : /models/lcm/           → 🛑 사람 전용 차단지점(실사용자 OTA). 비대화형/AI 차단.
#
# 사용: bash scripts/deploy_model.sh [staging|production]   (기본 staging)
#   (export_onnx + gen_manifest 이후 — artifacts/ 5자산 필요)
set -euo pipefail

TARGET="${1:-staging}"
case "$TARGET" in
  staging)          SUBDIR="lcm-staging" ;;
  production | prod) SUBDIR="lcm" ;;
  *) echo "사용: deploy_model.sh [staging|production]"; exit 1 ;;
esac

VPS="${VPS_HOST:-root@209.97.169.136}"
DST="${DST:-/opt/laryen-downloads/files/models/$SUBDIR}"
URL="https://laryen.com/models/$SUBDIR"
cd "$(dirname "$0")/.."

# 🛑 production 은 사람 전용 차단지점(실사용자 OTA) — game-server production-sync.sh 와 동일.
#    AI/자동(비대화형 tty)은 차단하고, 사람이 터미널에서 'yes' 를 입력해야 진행한다.
if [ "$SUBDIR" = "lcm" ]; then
  if [ ! -t 0 ]; then
    echo "❌ LCM production 배포는 사람이 터미널에서 직접 실행해야 합니다(비대화형/AI 차단)."
    echo "   AI 는 'deploy_model.sh staging' 만 자율 실행하세요."
    exit 1
  fi
  echo "⚠️  LCM *production* 배포 — 실사용자가 OTA 로 새 모델을 받습니다($URL)."
  read -r -p "    계속하려면 'yes' 입력: " ans
  [ "$ans" = "yes" ] || { echo "취소됨."; exit 1; }
fi

for f in artifacts/lcm.int8.onnx artifacts/lcm-labels.json artifacts/manifest.json \
         artifacts/tokenizer/vocab.json artifacts/tokenizer/merges.txt; do
  [ -f "$f" ] || { echo "❌ 없음: $f (export_onnx + gen_manifest 먼저)"; exit 1; }
done

echo "==> [$TARGET] 1) 원격 디렉토리 준비 $DST"
ssh "$VPS" "mkdir -p $DST"

echo "==> 2) 5자산 업로드(rsync)"
rsync -az artifacts/lcm.int8.onnx artifacts/lcm-labels.json artifacts/manifest.json \
  artifacts/tokenizer/vocab.json artifacts/tokenizer/merges.txt "$VPS:$DST/"

echo "==> 3) 원격 파일 확인"
ssh "$VPS" "ls -lh $DST"

echo "==> 4) $URL/manifest.json 검증"
sleep 1
curl -sS --max-time 15 "$URL/manifest.json" | head -8 \
  || echo "(manifest fetch 실패 — nginx/Traefik 라우팅 확인)"
echo ""
echo "✅ LCM 모델 [$TARGET] 배포 완료 → $URL/"
