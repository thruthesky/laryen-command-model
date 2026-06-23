#!/usr/bin/env python3
"""라리엔 dart SSOT → config/ssot.json 추출기.

**왜 추출인가**: action/위치/archetype/장비/물약/메뉴/fast-path 별칭은 라리엔 클라
(lib/) 가 *단일 진실(SSOT)* 이다. 이를 파이썬에 손으로 미러하면 신규 사냥터·몬스터가
추가될 때마다 drift 한다(라리엔의 ai_generate/voice_intent 가 과거 12종 손-미러로
drift 했던 것과 같은 문제). 따라서 모델 학습의 라벨 공간은 *코드를 복사하지 않고*
dart 파일을 파싱해 ssot.json 으로 뽑아 쓴다. dart 가 바뀌면 본 스크립트를 다시 돌린다.

파싱 대상(라리엔 repo = 이 submodule 의 부모):
  - lib/features/game/control_set/landmark_catalog.dart  → 위치(id·종류·별칭·좌표)
  - lib/features/game/render/data/archetype.dart         → archetype wire 이름
  - lib/features/game/control_set/voice_intent.dart      → 물약·장비 세트/단품
  - lib/services/voice/voice_fast_path.dart              → fast-path 별칭(stop/potion/menu)

사용:
  python scripts/sync_ssot.py            # ../lib 에서 추출 → config/ssot.json
  python scripts/sync_ssot.py --check    # 추출만 하고 diff 없는지 확인(CI 가드)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 이 submodule = laryen-command-model/, 그 부모 = 라리엔 repo 루트.
REPO = Path(__file__).resolve().parents[1]
LARYEN = REPO.parent
LIB = LARYEN / "lib"
OUT = REPO / "config" / "ssot.json"

# ── dart valid-set 미러(추출이 비효율적인 소수 고정값 — voice_intent.dart 의 switch
#    validation 과 1:1. 거의 안 변하지만, 바뀌면 여기와 dart 를 함께 고친다) ──────────
ACTIONS = [
    "move", "hunt", "stop", "potion", "equip", "unequip",
    "auto_combat", "auto_potion", "open_menu", "unknown",
]
POTIONS = ["hp", "run", "atkspeed", "crit"]          # _potionAlias 키
UNEQUIP_SLOTS = ["weapon", "armor", "accessory"]      # unequip 'slot'
COMBAT_MODES = ["off", "magnetic", "auto_hunt"]       # auto_combat 'mode'
MENU_TARGETS = [                                       # open_menu 'target' valid set
    "menu", "chat", "groupchat", "inventory", "potion", "sound", "autocombat",
    "debug",                                           # 디버그 패널 토글(음성 "디버그 꺼")
]
# 방향 이동(move 'direction') — 화면 시계각도 16버킷(0,22.5,...). 분류 헤드용.
DIRECTION_BUCKETS = [round(i * 22.5, 1) for i in range(16)]


def _read(rel: str) -> str:
    p = LIB / rel
    if not p.exists():
        sys.exit(f"❌ dart 파일 없음: {p}\n   (이 스크립트는 라리엔 repo 안의 submodule 에서 실행해야 합니다)")
    return p.read_text(encoding="utf-8")


def parse_landmarks(text: str) -> list[dict]:
    """landmark_catalog.dart 의 Landmark(...) 블록들을 파싱."""
    out: list[dict] = []
    # 'Landmark(' 로 split → 각 조각은 다음 Landmark 전까지(별칭 [] 안엔 ')' 없음).
    for chunk in re.split(r"\bLandmark\(", text)[1:]:
        # 이 블록의 끝('),')까지만 본다 — 마지막 블록이 파일 뒤쪽 코드
        # (landmarkIntentBlock 의 'LandmarkKind.safe' 문자열)까지 삼켜 kind 가
        # 오염되는 것을 막는다(중첩 괄호 없음 → 첫 '),' 가 블록 끝).
        end = chunk.find("),")
        if end != -1:
            chunk = chunk[:end]
        id_m = re.search(r"id:\s*'([^']+)'", chunk)
        if not id_m:
            continue
        ko_m = re.search(r"displayKo:\s*'([^']*)'", chunk)
        x_m = re.search(r"xCm:\s*(-?[\d.]+)", chunk)
        y_m = re.search(r"yCm:\s*(-?[\d.]+)", chunk)
        al_m = re.search(r"aliases:\s*\[(.*?)\]", chunk, re.S)
        aliases = re.findall(r"'([^']*)'", al_m.group(1)) if al_m else []
        hint_m = re.search(r"monsterHint:\s*'([^']*)'", chunk)
        if re.search(r"LandmarkKind\.waypoint", chunk):
            kind = "waypoint"   # 이동 지점(역 출구 등) — 사냥 발화 생성 제외(_gen_hunt)
        elif re.search(r"LandmarkKind\.safe", chunk):
            kind = "safe"
        else:
            kind = "hunt"
        out.append({
            "id": id_m.group(1),
            "ko": ko_m.group(1) if ko_m else id_m.group(1),
            "kind": kind,
            "x": float(x_m.group(1)) if x_m else 0.0,
            "y": float(y_m.group(1)) if y_m else 0.0,
            "aliases": aliases,
            "monsterHint": hint_m.group(1) if hint_m else "",
        })
    if not out:
        sys.exit("❌ landmark 파싱 0건 — landmark_catalog.dart 형식이 바뀌었을 수 있습니다.")
    return out


def parse_archetypes(text: str) -> list[str]:
    """archetype.dart 의 wireName switch → ['Brute','Caster',...] (서버 enum 순서)."""
    m = re.search(r"extension ArchetypeWireName.*?\{(.*?)\n\}", text, re.S)
    body = m.group(1) if m else text
    names = re.findall(r"ArchetypeKind\.\w+\s*=>\s*'([A-Za-z]+)'", body)
    if not names:
        sys.exit("❌ archetype 파싱 0건 — archetype.dart 의 wireName 형식이 바뀌었습니다.")
    return names


def _map_keys(text: str, var: str) -> list[str]:
    """const Map ... <var> = { '키': ... } 의 최상위 문자열 키들을 추출."""
    m = re.search(re.escape(var) + r"\s*=\s*\{(.*?)\n\};", text, re.S)
    if not m:
        return []
    # 들여쓰기 2칸(최상위 엔트리)으로 시작하는 "'key':" 만.
    return re.findall(r"^\s{2,4}'([^']+)':", m.group(1), re.M)


def parse_fast_path(text: str) -> dict:
    """voice_fast_path.dart 의 FastPathRules.defaults 추출."""
    def set_of(name: str) -> list[str]:
        m = re.search(name + r":\s*\{(.*?)\},?\n", text, re.S)
        return re.findall(r"'([^']*)'", m.group(1)) if m else []

    menu: dict[str, list[str]] = {}
    menu_m = re.search(r"menu:\s*\{(.*?)\n\s{4}\},", text, re.S)
    if menu_m:
        for key, vals in re.findall(r"'(\w+)':\s*\{([^}]*)\}", menu_m.group(1)):
            menu[key] = re.findall(r"'([^']*)'", vals)
    return {"version": 1, "stop": set_of("stop"), "potionHp": set_of("potionHp"), "menu": menu}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="diff 없는지만 확인(쓰지 않음)")
    args = ap.parse_args()

    landmarks = parse_landmarks(_read("features/game/control_set/landmark_catalog.dart"))
    archetypes = parse_archetypes(_read("features/game/render/data/archetype.dart"))
    vi = _read("features/game/control_set/voice_intent.dart")
    gear_singles = _map_keys(vi, "kGearKo")
    gear_sets = _map_keys(vi, "kGearSets")
    fast_path = parse_fast_path(_read("services/voice/voice_fast_path.dart"))

    ssot = {
        "_generated_by": "scripts/sync_ssot.py (라리엔 lib/ dart SSOT 추출 — 손수정 금지)",
        "actions": ACTIONS,
        "landmarks": landmarks,
        "landmark_ids": [lm["id"] for lm in landmarks],
        "archetypes": archetypes,
        "potions": POTIONS,
        "gear_sets": gear_sets,
        "gear_singles": gear_singles,
        "unequip_slots": UNEQUIP_SLOTS,
        "combat_modes": COMBAT_MODES,
        "menu_targets": MENU_TARGETS,
        "direction_buckets": DIRECTION_BUCKETS,
        "fast_path": fast_path,
    }
    text = json.dumps(ssot, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old != text:
            print("❌ ssot.json 이 dart SSOT 와 다릅니다 — `python scripts/sync_ssot.py` 를 다시 돌리세요.")
            return 1
        print("✅ ssot.json 최신")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"✅ {OUT.relative_to(REPO)} 생성 — "
          f"위치 {len(landmarks)}개 · archetype {len(archetypes)}종 · "
          f"세트 {len(gear_sets)}종 · 단품 {len(gear_singles)}종")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
