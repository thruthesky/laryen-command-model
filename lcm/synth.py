"""합성 학습 데이터 생성 — SSOT 값 + 한/영 템플릿 → (발화, intent) 페어.

**왜 합성인가**: 라리엔은 위치/몬스터/장비/물약/메뉴를 코드 SSOT(ssot.json)로 갖고
있어, 템플릿에 그 값들을 끼워 (발화 → intent JSON) 페어를 대량 생성할 수 있다. 이것이
SML 학습의 1차 데이터다(2차로 실사용 L3 로그를 Gemini 로 라벨링해 분포를 보정 —
README 의 distillation). 한/영만 우선 지원한다(코드스위칭 "plate 세트 입고" 포함).

생성물 1건 = {"text": <발화>, "intent": <라리엔 action JSON 1개>}.
intent 는 schema.encode_intent 가 헤드 라벨로 바꾼다.
"""
from __future__ import annotations

import random

# ── 방향 단어 → 화면 시계각도(voice_intent.dart [방향 이동] 규칙과 동일) ──────────
_DIR_WORDS = {
    0: ["위", "위쪽", "북", "북쪽", "up", "north"],
    45: ["오른쪽 위", "북동", "북동쪽", "northeast"],
    90: ["오른쪽", "우", "동", "동쪽", "right", "east"],
    135: ["오른쪽 아래", "남동", "동남", "동남쪽", "southeast"],
    180: ["아래", "아래쪽", "남", "남쪽", "down", "south"],
    225: ["왼쪽 아래", "남서", "southwest"],
    270: ["왼쪽", "좌", "서", "서쪽", "left", "west"],
    315: ["왼쪽 위", "북서", "northwest"],
}
_POTION_WORDS = {
    "hp": ["체력", "회복", "hp", "HP", "힐"],
    "run": ["이동속도", "이동 속도", "런", "run", "스피드"],
    "atkspeed": ["공격속도", "공격 속도", "공속", "atkspeed"],
    "crit": ["크리", "크리티컬", "치명타", "crit"],
}
_SLOT_WORDS = {"weapon": ["무기", "weapon"], "armor": ["갑옷", "armor"], "accessory": ["장신구", "accessory"]}
_MODE_PHRASES = {
    "auto_hunt": ["자동사냥 켜", "자동 사냥 켜줘", "오토 켜", "자동사냥 시작", "turn on auto hunt"],
    "off": ["자동사냥 꺼", "자동 사냥 꺼줘", "오토 꺼", "자동사냥 중지", "turn off auto hunt"],
    "magnetic": ["도착하면 자동공격", "도착 후 자동 공격", "근처 자동 공격", "magnetic 모드"],
}
# unknown(잡담·모호) — 게임 조작이 아니라서 SML 이 unknown 을 내면 클라가 CF 로 폴백.
_UNKNOWN = [
    "안녕", "안녕하세요", "넌 누구야", "이름이 뭐야", "고마워", "오늘 날씨 어때",
    "이 게임 재밌다", "심심해", "뭐하고 놀까", "라리엔이 뭐야", "도움말",
    "hello", "who are you", "thanks", "what can you do", "어디서 사냥하면 좋아",
    "이 몬스터 뭐야", "강해지려면 어떻게 해", "레벨 어떻게 올려",
]


def _ko_obj(word: str) -> str:
    """받침에 따라 '로/으로' 선택(대략적 — 합성 다양성용)."""
    if not word:
        return word + "로"
    last = word[-1]
    if "가" <= last <= "힣":
        code = (ord(last) - 0xAC00) % 28
        return word + ("로" if code == 0 or code == 8 else "으로")
    return word + "로"


def _gen_move(ssot, rng) -> list[tuple[str, dict]]:
    out = []
    for lm in ssot["landmarks"]:
        al = rng.choice(lm["aliases"]) if lm["aliases"] else lm["ko"]
        intent = {"action": "move", "location": lm["id"]}
        for t in (f"{_ko_obj(al)} 가", f"{_ko_obj(al)} 가줘", f"{_ko_obj(al)} 이동",
                  f"{_ko_obj(al)} 이동해줘", f"{al}로 이동", f"move to {al}", f"go to {al}"):
            out.append((t, intent))
    # 안전지대 대기(move location=safe).
    for t in ("세이프존에서 대기해", "안전지대로 가서 쉬어", "쉼터로 가", "마을로 이동"):
        out.append((t, {"action": "move", "location": "safe"}))
    # 순수 방향.
    for deg, words in _DIR_WORDS.items():
        for w in words:
            intent = {"action": "move", "direction": deg}
            out.append((f"{_ko_obj(w)} 가", intent))
            out.append((f"{w}으로 걸어가", intent))
            out.append((f"go {w}", intent))
    for n in range(1, 13):  # "N시 방향"
        out.append((f"{n}시 방향으로 가", {"action": "move", "direction": (n * 30) % 360}))
    return out


def _gen_hunt(ssot, rng) -> list[tuple[str, dict]]:
    out = []
    hunts = [lm for lm in ssot["landmarks"] if lm["kind"] == "hunt"]
    archs = ssot["archetypes"]
    for lm in hunts:
        al = rng.choice(lm["aliases"]) if lm["aliases"] else lm["ko"]
        out.append((f"{al}에서 사냥", {"action": "hunt", "location": lm["id"]}))
        out.append((f"{al}에서 사냥해줘", {"action": "hunt", "location": lm["id"]}))
        out.append((f"hunt at {al}", {"action": "hunt", "location": lm["id"]}))
        mon = rng.choice(archs)
        out.append((f"{al}에서 {mon} 잡아", {"action": "hunt", "location": lm["id"], "monsters": [mon]}))
        hp = rng.choice([20, 30, 40, 50])
        out.append((f"{al}에서 {mon} 사냥하고 체력 {hp}% 아래면 안전지대로 피신",
                    {"action": "hunt", "location": lm["id"], "monsters": [mon],
                     "retreatToSafeZone": True, "retreatHpPct": hp}))
    # 위치 없는 사냥(레벨 추천 — location 비움).
    for t in ("사냥하자", "사냥 시작", "자동으로 사냥해", "사냥하러 가자", "let's hunt"):
        out.append((t, {"action": "hunt"}))
    return out


def _gen_simple(ssot) -> list[tuple[str, dict]]:
    out = []
    # stop / potion / open_menu 는 fast-path 별칭을 그대로 학습(SML 도 동일 표현 커버).
    fp = ssot["fast_path"]
    for w in fp["stop"]:
        out.append((w, {"action": "stop"}))
    for extra in ("그만 멈춰", "이제 그만해", "정지해줘", "동작 멈춰"):
        out.append((extra, {"action": "stop"}))
    for w in fp["potionHp"]:
        out.append((w, {"action": "potion", "potion": "hp"}))
    for target, aliases in fp["menu"].items():
        for w in aliases:
            out.append((w, {"action": "open_menu", "target": target}))
    # potion(4종).
    for pid, words in _POTION_WORDS.items():
        for w in words:
            out.append((f"{w} 물약", {"action": "potion", "potion": pid}))
            out.append((f"{w} 물약 먹어", {"action": "potion", "potion": pid}))
    # equip(세트 + 단품).
    set_ko = {"victor": "빅터", "immortal": "불멸", "plate": ["강철", "판금", "plate"]}
    for sid in ssot["gear_sets"]:
        names = set_ko.get(sid, [sid])
        names = names if isinstance(names, list) else [names]
        for nm in names:
            for t in (f"{nm} 세트 착용", f"{nm} 세트 입어", f"{nm}의 세트 아이템 착용", f"equip {sid} set"):
                out.append((t, {"action": "equip", "set": sid}))
    gear_ko = {
        "victor_weapon": "빅터의 검", "victor_armor": "빅터의 갑옷", "victor_accessory": "빅터의 장신구",
        "immortal_weapon": "불멸의 검", "immortal_armor": "불멸의 갑옷", "immortal_accessory": "불멸의 장신구",
        "plate_weapon": "강철의 검", "plate_armor": "강철의 갑옷", "plate_accessory": "강철의 장신구",
    }
    for gid in ssot["gear_singles"]:
        nm = gear_ko.get(gid, gid)
        out.append((f"{nm}만 착용", {"action": "equip", "gear": gid}))
    # unequip.
    for slot, words in _SLOT_WORDS.items():
        for w in words:
            out.append((f"{w} 벗어", {"action": "unequip", "slot": slot}))
            out.append((f"{w} 해제", {"action": "unequip", "slot": slot}))
    # auto_combat.
    for mode, phrases in _MODE_PHRASES.items():
        for p in phrases:
            out.append((p, {"action": "auto_combat", "mode": mode}))
    # auto_potion.
    for pid, words in _POTION_WORDS.items():
        w = words[0]
        out.append((f"{w} 물약 자동 사용", {"action": "auto_potion", "potions": [pid], "enable": True}))
        out.append((f"{w} 물약 자동으로 켜", {"action": "auto_potion", "potions": [pid], "enable": True}))
    out.append(("모든 물약 자동 사용", {"action": "auto_potion", "potions": ["all"], "enable": True}))
    out.append(("물약 자동 꺼", {"action": "auto_potion", "potions": ["all"], "enable": False}))
    # unknown(잡담·게임 질문 — CF 폴백 대상).
    for w in _UNKNOWN:
        out.append((w, {"action": "unknown"}))
    return out


def generate(ssot: dict, seed: int = 7) -> list[dict]:
    """모든 템플릿을 펼쳐 (text, intent) 페어 리스트를 만든다(재현 가능)."""
    rng = random.Random(seed)
    pairs: list[tuple[str, dict]] = []
    pairs += _gen_move(ssot, rng)
    pairs += _gen_hunt(ssot, rng)
    pairs += _gen_simple(ssot)
    # 중복 제거(같은 발화는 한 번만 — 마지막 라벨 우선).
    dedup: dict[str, dict] = {}
    for text, intent in pairs:
        dedup[text.strip()] = intent
    return [{"text": t, "intent": i} for t, i in dedup.items()]
