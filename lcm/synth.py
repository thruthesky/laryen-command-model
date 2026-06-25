"""합성 학습 데이터 생성 — SSOT 값 + 한/영 템플릿 → (발화, intent) 페어.

**왜 합성인가**: 라리엔은 위치/몬스터/장비/물약/메뉴를 코드 SSOT(ssot.json)로 갖고
있어, 템플릿에 그 값들을 끼워 (발화 → intent JSON) 페어를 대량 생성할 수 있다. 이것이
SML 학습의 1차 데이터다(2차로 실사용 L3 로그를 Gemini 로 라벨링해 분포를 보정 —
README 의 distillation). 한/영만 우선 지원한다(코드스위칭 "plate 세트 입고" 포함).

**다양성 & 균형(v2)**: 어순·어미·공손 변형을 곱해 표현을 늘리고, action 간 표본 수를
균형 있게 맞춘다(과거 move 330 vs auto_potion 10 편향 → 헤드 학습 실패의 원인이었다).

생성물 1건 = {"text": <발화>, "intent": <라리엔 action JSON 1개>}.
intent 는 schema.encode_intent 가 헤드 라벨로 바꾼다.
"""
from __future__ import annotations

import random

from .phonetics import phonetic_noise
from .synth_i18n import gen_i18n

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
    "hp": ["체력", "회복", "hp", "HP", "힐", "피"],
    "run": ["이동속도", "이동 속도", "런", "run", "스피드", "speed"],
    "atkspeed": ["공격속도", "공격 속도", "공속", "atkspeed", "attack speed"],
    "crit": ["크리", "크리티컬", "치명타", "crit", "critical"],
}
# archetype wire name → 한국어 발음/의미 alias(STT 한국어 발화 대응 — 다른 팀 #3).
# 입력은 한국어 alias 도 받되, intent monsters 는 항상 wire name 으로 정규화한다.
# 추후 라리엔 SSOT(fast_path·CF prompt 공유)로 승격 예정.
_MONSTER_KO = {
    "Brute": ["브루트"], "Caster": ["캐스터", "캐스타", "케스터", "캐스 터"], "Skirmisher": ["스커미셔"],
    "Coward": ["코워드", "겁쟁이"], "PackHunter": ["팩헌터"], "Guardian": ["가디언"],
    "Ambusher": ["앰부셔"], "Trickster": ["트릭스터"], "Bone": ["본", "뼈"],
    "Demonic": ["데모닉", "악마"], "Paladin": ["팔라딘", "성기사"],
    "DemonicKing": ["데모닉킹", "악마왕"], "Morak": ["모락"], "Zorak": ["조락", "조라크"],
    "Fred": ["프레드"], "Swat": ["스왓", "스왓군인"], "Xbot": ["엑스봇"], "Mira": ["미라"],
    "Hellion": ["헬리온"], "BiBot": ["바이봇"], "Skeleton": ["스켈레톤", "해골"],
    "GreenRobot": ["그린로봇", "초록로봇"], "Gozila": ["고질라"], "Mecha": ["메카"],
    "GuitarBot": ["기타봇"], "Ganfaul": ["간폴"], "Mannequin": ["마네킹"],
    "Pirate": ["파이렛", "해적"], "PumkinHulk": ["펌킨헐크", "호박헐크"],
    "SkeletonZombie": ["스켈레톤좀비", "해골좀비"], "Vampire": ["뱀파이어", "흡혈귀"],
    "Yaku": ["야쿠"],
}


def _mon_name(arch: str, rng) -> str:
    """archetype 의 입력 표기 — 한국어 alias 또는 wire name 무작위(둘 다 학습)."""
    return rng.choice(_MONSTER_KO.get(arch, []) + [arch])


_SLOT_WORDS = {
    "weapon": ["무기", "weapon"], "armor": ["갑옷", "방어구", "armor"],
    "accessory": ["장신구", "악세사리", "accessory"],
}
_MODE_PHRASES = {
    "auto_hunt": ["자동사냥 켜", "자동 사냥 켜줘", "오토 켜", "자동사냥 시작", "자동사냥 돌려",
                  "자동으로 사냥해 줘", "오토헌트 켜", "오토 모드 켜", "자동전투 켜", "자동전투 시작",
                  "자동사냥 켜줘", "오토 사냥 켜", "자동 전투 켜줘", "자동사냥 활성화", "오토 시작",
                  "자동사냥 돌리자", "오토 돌리자", "자동사냥 하자", "오토 켜자", "자동전투 돌려",
                  "turn on auto hunt", "auto hunt on", "enable auto hunt", "auto combat on"],
    "off": ["자동사냥 꺼", "자동 사냥 꺼줘", "오토 꺼", "자동사냥 중지", "자동사냥 멈춰",
            "오토 끄기", "자동전투 꺼", "자동사냥 꺼줘", "오토 모드 꺼", "자동전투 중지",
            "자동사냥 비활성화", "오토 정지", "자동 전투 꺼줘", "오토 사냥 꺼", "자동사냥 끄기",
            "자동사냥 그만", "오토 그만", "자동사냥 빼", "오토 빼줘", "자동사냥 해제", "오토 해제",
            "turn off auto hunt", "auto hunt off", "disable auto hunt", "auto combat off",
            "turn it off", "turn auto hunt off", "stop auto hunt", "stop auto hunting",
            "disable auto combat", "auto off", "turn off auto combat", "switch off auto hunt",
            "turn the auto hunt off", "stop the auto hunt"],
    "magnetic": ["도착하면 자동공격", "도착 후 자동 공격", "근처 자동 공격", "자석 모드",
                 "마그네틱 모드", "도착하면 주변 공격", "도착 후 주변 공격", "도착하면 근처 공격",
                 "자석모드 켜", "도착하면 알아서 공격", "도착 후 자동 전투", "magnetic mode", "magnetic on"],
}
# 어미·공손 변형(한국어). 동사형 발화에 곱해 표현을 늘린다.
_KO_TAILS = ["", " 해", " 해줘", " 해주세요", " 좀", " 줘", "줘"]
# 한글 숫자(HP %) — STT 가 "삼십 퍼센트" 로 전사할 수 있어 학습.
_KO_NUM = {10: "십", 20: "이십", 30: "삼십", 40: "사십", 50: "오십",
           60: "육십", 70: "칠십", 80: "팔십", 90: "구십"}
# unknown(잡담·게임 질문/설명) — 게임 *조작* 이 아니라서 SML 이 unknown 을 내면 클라가
# CF(explain/chat route)로 폴백. "라리엔 이해"의 경계: 명령 vs 질문을 가르는 학습.
# **fallback 안전장치의 핵심** — 잡담/질문을 명령으로 오인하면 오작동하므로 다양하게 학습.
_UNKNOWN = [
    # 인사·잡담·정체성·감사.
    "안녕", "안녕하세요", "반가워",  # 정체성 질문은 query_assistant_identity 로 이전(중복 제거)
    "고마워", "수고해", "잘 자", "또 보자", "사랑해", "재밌다", "심심해", "지루해",
    "오늘 날씨 어때", "밥 먹었어", "뭐하고 놀까", "노래 불러줘", "농담 해줘",
    "hello", "hi", "hey", "thanks", "thank you",
    "good job", "see you", "i'm bored", "tell me a joke",
    # 게임 질문(explain) — "무엇/어디/어떻게/왜" 패턴.
    "라리엔이 뭐야", "이 게임 어떻게 해", "도움말", "튜토리얼 보여줘", "조작법 알려줘",
    "어디서 사냥하면 좋아", "지금 뭐 하면 좋아", "다음에 뭐 해야 돼", "추천 사냥터 어디야",
    "이 몬스터 뭐야", "캐스터가 뭐야", "브루트는 어때", "보스는 어디 있어", "제일 센 몬스터 뭐야",
    "강해지려면 어떻게 해", "레벨 어떻게 올려", "경험치 어떻게 벌어", "공격력 어떻게 올려",
    "강철 세트 효과가 뭐야", "불멸 세트 좋아", "어떤 무기가 좋아", "장비 어디서 구해",
    "파티 어떻게 만들어", "거래 어떻게 해", "친구 추가 어떻게 해", "채팅 어떻게 해",
    "물약 어디서 사", "내 레벨 몇이야", "내 체력 얼마야", "왜 자꾸 죽어", "왜 안 움직여",
    "안전지대가 뭐야", "자동사냥이 뭐야", "크리티컬이 뭐야",
    "what should i do now", "how do i level up", "which monster is strong",
    "where should i hunt", "what is a caster", "how do i make a party",
    "how to trade", "what is the strongest weapon", "where is the boss",
    "how do i get stronger", "what does plate set do", "is immortal set good",
]


def _has_batchim(word: str) -> bool:
    if not word:
        return False
    last = word[-1]
    return "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28 != 0


def _ko_obj(word: str) -> str:
    """받침에 따라 '로/으로'(예: 강남으로 / 강서로). 받침 ㄹ 도 '로'."""
    if not word:
        return word + "로"
    last = word[-1]
    if "가" <= last <= "힣":
        code = (ord(last) - 0xAC00) % 28
        return word + ("로" if code in (0, 8) else "으로")
    return word + "로"


def _tails(stem: str, rng, k: int = 3) -> list[str]:
    """동사 어간(예: '벗어') 에 어미 변형을 곱해 k개 표현을 만든다."""
    cands = [stem + t for t in _KO_TAILS]
    uniq = list(dict.fromkeys(cands))
    return uniq[:k] if k < len(uniq) else uniq


def _gen_move(ssot, rng) -> list[tuple[str, dict]]:
    out = []
    move_verbs = ["가", "가줘", "이동", "이동해", "이동해줘", "이동시켜줘", "가자",
                  "데려가", "데려다줘", "가자고", "가보자", "가야겠어"]
    for lm in ssot["landmarks"]:
        intent = {"action": "move", "location": lm["id"]}
        # 한국어 별칭 + 영문 별칭 모두 활용.
        names = list(dict.fromkeys([lm["ko"], *lm["aliases"]]))
        for al in names:
            if any("가" <= c <= "힣" for c in al):  # 한글 별칭
                for v in rng.sample(move_verbs, k=4):
                    out.append((f"{_ko_obj(al)} {v}".strip(), intent))
                # 조사 없는 짧은 표현("강남 가"·"강북 이동") + 조사 단독("강남으로").
                out.append((f"{al} 가", intent))
                out.append((f"{al} 이동", intent))
                out.append((f"{al} 가자", intent))
                out.append((f"{_ko_obj(al)}", intent))   # "강남으로"·"연습장으로" 단독
                # 지시어 prefix("저기/거기 X로 가").
                for pre in rng.sample(["저기", "거기", "그", "일단"], k=2):
                    out.append((f"{pre} {al}로 가", intent))
                    out.append((f"{pre} {_ko_obj(al)} 이동", intent))
            else:  # 영문 별칭 — 영어 단독 + 한글 조사 혼용("safe zone으로 가").
                out.append((f"move to {al}", intent))
                out.append((f"go to {al}", intent))
                out.append((f"{al}으로 가", intent))
                out.append((f"{al}로 이동", intent))
                out.append((f"{al}으로 이동해", intent))
    # 안전지대 대기(move location=safe) — '대기/쉬어'도 move(safe).
    safe_phr = ["세이프존에서 대기해", "안전지대로 가서 쉬어", "쉼터로 가", "마을로 이동",
                "안전지대로 피신", "세이프존으로 가줘", "안전한 곳으로 가", "대기 장소로 가",
                "도망쳐", "후퇴해", "피신해", "대피해", "안전지대로 도망", "안전한 곳으로 피해",
                "세이프존으로 후퇴", "도망가자", "피하자",
                "세이프존", "안전지대", "쉼터", "세이프존 가", "안전지대 가"]  # 단독·짧은
    for t in safe_phr:
        out.append((t, {"action": "move", "location": "safe"}))
    # 순수 방향.
    for deg, words in _DIR_WORDS.items():
        intent = {"action": "move", "direction": deg}
        for w in words:
            if any("가" <= c <= "힣" for c in w):
                out.append((f"{_ko_obj(w)} 가", intent))
                out.append((f"{w}으로 걸어가", intent))
                out.append((f"{w}쪽으로 이동", intent))
                out.append((f"{_ko_obj(w)} 좀 가볼까", intent))  # 구어체
                out.append((f"{_ko_obj(w)} 가보자", intent))
                out.append((w, intent))                         # 단독 방향("왼쪽")
                out.append((f"{_ko_obj(w)}", intent))            # "왼쪽으로"
            else:
                out.append((f"go {w}", intent))
                out.append((f"move {w}", intent))
    for n in range(1, 13):  # "N시 방향"
        d = (n * 30) % 360
        out.append((f"{n}시 방향으로 가", {"action": "move", "direction": d}))
        out.append((f"{n}시 방향으로 걸어", {"action": "move", "direction": d}))
    return out


def _gen_hunt(ssot, rng) -> list[tuple[str, dict]]:
    out = []
    hunts = [lm for lm in ssot["landmarks"] if lm["kind"] == "hunt"]
    archs = ssot["archetypes"]
    hunt_verbs = ["사냥", "사냥해", "사냥해줘", "사냥하자", "에서 사냥", "에서 잡아"]
    for lm in hunts:
        loc = {"action": "hunt", "location": lm["id"]}
        names = list(dict.fromkeys([lm["ko"], *lm["aliases"]]))
        for al in names[:6]:
            if any("가" <= c <= "힣" for c in al):
                out.append((f"{al}에서 사냥", loc))
                out.append((f"{al}에서 사냥해줘", loc))
                out.append((f"{al}에서 자동 사냥", loc))
                # 구어체/도치(실사용 발화).
                out.append((f"사냥하자 {al}에서", loc))
                out.append((f"{al} 가서 사냥", loc))
                out.append((f"{al} 가서 사냥해줘", loc))
                out.append((f"{al} 가서 사냥 좀 해줘", loc))
                out.append((f"{al}가서 사냥 좀 해줘", loc))
                out.append((f"{al} 가서 사냥하자", loc))
                out.append((f"{al} 가서 몹 잡아", loc))
                out.append((f"사냥 좀 하러 가자 {al}", loc))
                out.append((f"{al} 가서 몬스터 잡자", loc))
                out.append((f"{al}으로 사냥 가자", loc))
                out.append((f"{al} 가서 사냥하면 될까", loc))
                out.append((f"{al}으로 사냥 가줄래", loc))
            else:
                out.append((f"hunt at {al}", loc))
        al = rng.choice([a for a in names if any("가" <= c <= "힣" for c in a)] or [lm["ko"]])
        # monster 포함 hunt — *한국어 alias/wire 무작위* 입력, intent 는 wire name.
        # k=6(↑4): monster-only 보강이 location+monster TP 를 희석한 회귀(2026-06-22
        #   test_monsters_true_positive 3/6) 대응 — location+monster 표본을 늘려 균형.
        for mon in rng.sample(archs, k=6):
            nm = _mon_name(mon, rng)
            for v in ("잡아", "사냥해", "사냥해줘", "처치해"):
                out.append((f"{al}에서 {nm} {v}",
                            {"action": "hunt", "location": lm["id"], "monsters": [mon]}))
        # 2종 동시 사냥(멀티라벨 학습).
        m2 = rng.sample(archs, k=2)
        out.append((f"{al}에서 {_mon_name(m2[0], rng)}랑 {_mon_name(m2[1], rng)} 사냥",
                    {"action": "hunt", "location": lm["id"], "monsters": m2}))
        # monster + retreat (소수 — mon 을 hp 마다 다르게 골라 location+hp→mon 상관 분산).
        for hp in rng.sample([10, 30, 50, 70, 90], k=2):
            mon = rng.choice(archs)
            out.append((f"{al}에서 {_mon_name(mon, rng)} 사냥하고 체력 {hp}% 아래면 안전지대로 피신",
                        {"action": "hunt", "location": lm["id"], "monsters": [mon],
                         "retreatToSafeZone": True, "retreatHpPct": hp}))
        # monster 없는 retreat — *전체 hp*(10~90) 충분히 학습해 "hp≠monster" 를 각인,
        # "강남 60% 사냥"→특정 monster false positive(spurious)를 근본 차단.
        for hp in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            base = {"action": "hunt", "location": lm["id"],
                    "retreatToSafeZone": True, "retreatHpPct": hp}
            out.append((f"{al}에서 사냥하고 체력 {hp}%면 안전지대로", base))
            out.append((f"{al}에서 사냥하고 체력 {hp}% 아래면 피신", base))
            ko = _KO_NUM[hp]
            out.append((f"{al}에서 사냥하고 체력 {ko} 퍼센트면 안전지대로", base))
            out.append((f"{al}에서 사냥하고 체력 {ko}프로면 피신", base))
            if hp == 50:
                out.append((f"{al}에서 사냥하고 체력 절반이면 안전지대로", base))
    # 모든 archetype 균등 학습 — monsters(multi-label) TP 가 학습 비결정성에 흔들리지 않게
    # 32종 각각 충분한 hunt 예시 보장(과거 일부 archetype false negative 반복).
    for arch in archs:
        for _ in range(9):  # archetype 당 충분(↑5→9: monster-only 보강 후 location+monster
            #   TP 안정 — 약/혼동 방지. test_monsters_true_positive 회귀 대응 2026-06-22)
            lm = rng.choice(hunts)
            names = list(dict.fromkeys([lm["ko"], *lm["aliases"]]))
            al = rng.choice([a for a in names
                             if any("가" <= c <= "힣" for c in a)] or [lm["ko"]])
            nm = _mon_name(arch, rng)  # 한국어 alias/wire 무작위
            for v in ("잡아", "사냥해", "처치해", "사냥해줘", "잡자"):
                out.append((f"{al}에서 {nm} {v}",
                            {"action": "hunt", "location": lm["id"], "monsters": [arch]}))
    # 위치 없는 사냥(레벨 추천 — location 비움).
    for t in ("사냥하자", "사냥 시작", "자동으로 사냥해", "사냥하러 가자", "let's hunt",
              "사냥터 가서 사냥", "몬스터 잡으러 가자", "사냥 좀 하자",
              "사냥해", "사냥", "사냥해줘", "잡아줘", "몹 잡아", "사냥 좀", "몬스터 잡아"):
        out.append((t, {"action": "hunt"}))
    # ── monster-only(위치 없이 *몬스터만*) — 실측 발굴 이슈(2026-06-22 probe_lang):
    #    "캐스터 사냥"(monsters 누락)·"hunt casters"(fallback) 미지원이었다. location 없이
    #    monster 지정 발화를 한/영으로 학습한다(클라가 레벨추천 location 으로 보충). ──────
    for arch in archs:
        ko = _mon_name(arch, rng)               # 한국어 alias/wire 무작위
        low = arch.lower()                      # 영어 wire 소문자(caster)
        for v in ("사냥", "사냥해", "사냥해줘", "잡아", "처치해", "잡자"):
            out.append((f"{ko} {v}", {"action": "hunt", "monsters": [arch]}))
        # 영어 monster(소문자 단수/복수) — "hunt caster"·"kill casters"·"go hunt brutes".
        for en in (low, low + "s"):
            for tmpl in (f"hunt {en}", f"kill {en}", f"go hunt {en}", f"hunt the {en}",
                         f"attack {en}", f"fight {en}"):
                out.append((tmpl, {"action": "hunt", "monsters": [arch]}))
    return out


def _gen_typo_robustness(ssot, rng) -> list[tuple[str, dict]]:
    """의미는 통하지만 STT/철자가 흔들린 발화 — LCM 이 로컬에서 잡아야 하는 안전 표본.

    완전한 자유 오타 교정기는 아니지만, 실측에서 자주 나온 한국어 음소/STT 오류와 영어
    철자 누락을 닫힌 게임 명령으로 직접 노출한다. 위치·몬스터 슬롯이 있는 케이스도
    *의미가 분명한* 표본만 둔다.
    """
    out: list[tuple[str, dict]] = []
    # 한국어 STT/구어 오타 — 의미가 분명한 명령.
    out += [
        ("멈처", {"action": "stop"}),
        ("멈처줘", {"action": "stop"}),
        ("멈추어 줘", {"action": "stop"}),
        ("정지헤", {"action": "stop"}),
        ("피 채어줘", {"action": "potion", "potion": "hp"}),
        ("피채워", {"action": "potion", "potion": "hp"}),
        ("피 체워줘", {"action": "potion", "potion": "hp"}),
        ("물약 머거", {"action": "potion", "potion": "hp"}),
        ("문약 먹어", {"action": "potion", "potion": "hp"}),
        ("포션 머거", {"action": "potion", "potion": "hp"}),
        ("힐 해줘", {"action": "potion", "potion": "hp"}),
        ("자동 산양 켜줘", {"action": "auto_combat", "mode": "auto_hunt"}),
        ("자동사양 켜줘", {"action": "auto_combat", "mode": "auto_hunt"}),
        ("자동사냥 시작해조", {"action": "auto_combat", "mode": "auto_hunt"}),
        ("오토 헌트 꺼조", {"action": "auto_combat", "mode": "off"}),
        ("새이프존으로 가", {"action": "move", "location": "safe"}),
        ("세이프 존으로 가", {"action": "move", "location": "safe"}),
        ("가로수길로 이동", {"action": "move", "location": "gangnam_garosoo"}),
        ("강남 가로수길로 이동", {"action": "move", "location": "gangnam_garosoo"}),
        ("강남가로수길로 이동", {"action": "move", "location": "gangnam_garosoo"}),
        ("강남 가서 사냥 좀 해줘", {"action": "hunt", "location": "gangnam"}),
        ("강남가서 사냥 좀 해줘", {"action": "hunt", "location": "gangnam"}),
        ("케스터 사냥", {"action": "hunt", "monsters": ["Caster"]}),
        ("캐스 터 사냥", {"action": "hunt", "monsters": ["Caster"]}),
        ("해골 잡아조", {"action": "hunt", "monsters": ["Skeleton"]}),
    ]
    # 영어 철자/문법 흔들림 — target 슬롯 쏠림 방지용으로 menu 계열을 두껍게.
    out += [
        ("opn inventory", {"action": "open_menu", "target": "inventory"}),
        ("open inventry", {"action": "open_menu", "target": "inventory"}),
        ("open invantory", {"action": "open_menu", "target": "inventory"}),
        ("open invntory", {"action": "open_menu", "target": "inventory"}),
        ("show inventry", {"action": "open_menu", "target": "inventory"}),
        ("open cht", {"action": "open_menu", "target": "groupchat"}),
        ("open chatt", {"action": "open_menu", "target": "groupchat"}),
        ("open group chat", {"action": "open_menu", "target": "groupchat"}),
        ("show chat", {"action": "open_menu", "target": "groupchat"}),
        ("turn of auto hunt", {"action": "auto_combat", "mode": "off"}),
        ("turn off auto hant", {"action": "auto_combat", "mode": "off"}),
        ("auto hunt of", {"action": "auto_combat", "mode": "off"}),
        ("disable auto hant", {"action": "auto_combat", "mode": "off"}),
        ("start auto hant", {"action": "auto_combat", "mode": "auto_hunt"}),
        ("turn on auto hant", {"action": "auto_combat", "mode": "auto_hunt"}),
        ("hunt castars", {"action": "hunt", "monsters": ["Caster"]}),
        ("hunt castrs", {"action": "hunt", "monsters": ["Caster"]}),
        ("kill bruts", {"action": "hunt", "monsters": ["Brute"]}),
        ("go to saf zone", {"action": "move", "location": "safe"}),
        ("go to safezon", {"action": "move", "location": "safe"}),
        ("go gangnam", {"action": "move", "location": "gangnam"}),
        ("go to gangnum", {"action": "move", "location": "gangnam"}),
    ]
    return out


def _gen_simple(ssot, rng) -> list[tuple[str, dict]]:
    out = []
    fp = ssot["fast_path"]
    # stop — fast-path 별칭 + 다양한 표현(균형 위해 증강).
    for w in fp["stop"]:
        out.append((w, {"action": "stop"}))
    for w in ("그만 멈춰", "이제 그만해", "정지해줘", "동작 멈춰", "당장 멈춰", "지금 멈춰",
              "다 멈춰", "이동 멈춰", "그만둬", "멈추세요", "전부 멈춰", "모두 정지", "행동 멈춰",
              "그만하라고", "멈추라고", "동작 정지", "이동 중지", "정지시켜줘", "그만 가",
              "움직이지 마", "가만 있어", "거기서 멈춰", "스톱해", "그만 움직여",
              "이제 그만하자", "그만하자", "여기까지 하자", "그만 좀 해", "멈추자", "이제 멈추자",
              "stop now", "stop moving", "stop it", "freeze", "halt", "hold on", "pause",
              "wait", "don't move", "stop right now", "cease", "hold", "stay there"):
        out.append((w, {"action": "stop"}))
    # potion(4종) — 풍부한 표현.
    pot_verbs = ["물약", "물약 먹어", "물약 마셔", "물약 써", "물약 사용", "물약 줘", "포션",
                 "물약 좀 먹자", "물약 좀 줘", "물약 마시자",
                 "물약 빨아", "물약 빨아줘", "빨아", "들이켜", "물약 들이켜"]  # 구어체·은어
    for pid, words in _POTION_WORDS.items():
        for w in words:
            for v in rng.sample(pot_verbs, k=3):
                out.append((f"{w} {v}".strip(), {"action": "potion", "potion": pid}))
    for w in fp["potionHp"]:  # "물약"의 기본 = hp(fast-path 와 동일 학습).
        out.append((w, {"action": "potion", "potion": "hp"}))
    # HP 회복 구어 + 단독("물약 먹어"·"회복 좀 하자").
    for w in ["회복 좀 하자", "체력 좀 채워", "체력 채워줘", "회복하자", "힐 좀 줘",
              "힐포션 좀", "피 좀 채워", "체력 회복", "회복 좀 시켜줘",
              "물약 먹어", "물약 먹어줘", "물약 좀 먹어", "물약 마셔", "물약 마셔줘",
              "hp 물약 먹어", "체력 물약 먹어", "회복 물약 먹어", "물약 좀"]:
        out.append((w, {"action": "potion", "potion": "hp"}))
    # open_menu — fast-path 별칭 + "{메뉴} 열어/보여줘/띄워".
    menu_ko = {"menu": ["메뉴", "메인 메뉴"], "chat": ["챗봇", "도우미", "라리아"],
               "groupchat": ["채팅", "전체 채팅", "그룹 채팅"], "inventory": ["인벤토리", "장비창", "가방"],
               "potion": ["물약창", "포션창"], "sound": ["소리 설정", "음량 설정", "볼륨 설정"],
               "autocombat": ["자동전투 설정", "전투 설정", "자동사냥 설정"]}
    for target, aliases in fp["menu"].items():
        for w in aliases:
            out.append((w, {"action": "open_menu", "target": target}))
    for target, names in menu_ko.items():
        for nm in names:
            for v in ("열어", "열어줘", "보여줘", "띄워줘", "켜줘"):
                out.append((f"{nm} {v}", {"action": "open_menu", "target": target}))
    # equip(세트 + 단품).
    set_ko = {"victor": ["빅터", "victor"], "immortal": ["불멸", "immortal"],
              "plate": ["강철", "판금", "plate"]}
    for sid in ssot["gear_sets"]:
        for nm in set_ko.get(sid, [sid]):
            for t in (f"{nm} 세트 착용", f"{nm} 세트 입어", f"{nm}의 세트 아이템 착용",
                      f"{nm} 세트 장착", f"{nm} 풀세트 착용", f"equip {sid} set", f"{nm} 세트로 갈아입어",
                      f"{nm}로 갈아입자", f"{nm} 장비", f"{nm} 입어", f"{nm}로 갈아입어",
                      f"{nm} 세트로", f"{nm} 장비 착용"):
                out.append((t, {"action": "equip", "set": sid}))
    gear_ko = {
        "victor_weapon": "빅터의 검", "victor_armor": "빅터의 갑옷", "victor_accessory": "빅터의 장신구",
        "immortal_weapon": "불멸의 검", "immortal_armor": "불멸의 갑옷", "immortal_accessory": "불멸의 장신구",
        "plate_weapon": "강철의 검", "plate_armor": "강철의 갑옷", "plate_accessory": "강철의 장신구",
    }
    for gid in ssot["gear_singles"]:
        nm = gear_ko.get(gid, gid)
        for t in (f"{nm}만 착용", f"{nm} 장착", f"{nm} 착용해줘", f"{nm}만 끼워",
                  f"{nm} 껴", f"{nm}만", f"{nm} 차", f"{nm}만 장착", f"{nm} 끼워"):
            out.append((t, {"action": "equip", "gear": gid}))
    # unequip — 3슬롯 × 표현.
    for slot, words in _SLOT_WORDS.items():
        for w in words:
            for v in ("벗어", "해제", "풀어", "벗겨줘", "빼", "벗어줘"):
                out.append((f"{w} {v}", {"action": "unequip", "slot": slot}))
            out.append((f"unequip {w}", {"action": "unequip", "slot": slot}))
    # auto_combat — 3모드 × 표현.
    for mode, phrases in _MODE_PHRASES.items():
        for p in phrases:
            out.append((p, {"action": "auto_combat", "mode": mode}))
    # auto_potion — 4물약 + all × enable/disable × 표현.
    for pid, words in _POTION_WORDS.items():
        for w in words[:2]:
            for v in ("물약 자동 사용", "물약 자동으로 켜", "물약 자동", "물약 자동 사용해줘",
                      "물약 오토", "물약 자동으로 먹어", "물약 자동 켜줘", "물약 자동 마셔"):
                out.append((f"{w} {v}", {"action": "auto_potion", "potions": [pid], "enable": True}))
            out.append((f"{w} 물약 자동 꺼", {"action": "auto_potion", "potions": [pid], "enable": False}))
            out.append((f"{w} 물약 자동 끄기", {"action": "auto_potion", "potions": [pid], "enable": False}))
    for t in ("모든 물약 자동 사용", "전체 물약 자동", "물약 전부 자동 사용", "all 물약 자동",
              "모든 물약 자동으로", "물약 다 자동 사용", "전부 자동 물약"):
        out.append((t, {"action": "auto_potion", "potions": ["all"], "enable": True}))
    for t in ("물약 자동 꺼", "물약 자동 사용 꺼", "자동 물약 끄기", "물약 오토 꺼",
              "모든 물약 자동 꺼", "자동 물약 중지"):
        out.append((t, {"action": "auto_potion", "potions": ["all"], "enable": False}))
    # 영어 명령(한/영 우선 — 영어 발화 다양성).
    for t in ["drink hp potion", "use potion", "take potion", "drink potion", "use hp potion",
              "heal", "heal me", "drink heal potion"]:
        out.append((t, {"action": "potion", "potion": "hp"}))
    for t in ["halt", "stop now", "stop it", "freeze", "hold on", "stop moving", "stop please"]:
        out.append((t, {"action": "stop"}))
    for t, tgt in [("open menu", "menu"), ("open inventory", "inventory"),
                   ("open chat", "groupchat"), ("open settings", "sound"),
                   ("show inventory", "inventory"), ("open the menu", "menu"),
                   ("open potion menu", "potion")]:
        out.append((t, {"action": "open_menu", "target": tgt}))
    for t in ["auto hunt on", "turn on auto hunt", "enable auto hunt", "start auto hunt"]:
        out.append((t, {"action": "auto_combat", "mode": "auto_hunt"}))
    for t in ["auto hunt off", "turn off auto hunt", "stop auto hunt"]:
        out.append((t, {"action": "auto_combat", "mode": "off"}))
    # 디버그 패널 토글(open_menu/debug) — 한/영 on/off/toggle. 사용자 핵심 발화
    # "turn off the debug panel"(영어 전체 문장) 포함. 게임은 토글이라 on/off/show/hide 를
    # 모두 같은 target=debug 로 학습(실행 시 현재 상태를 반전한다).
    debug_phrases = [
        "디버그 패널", "디버그패널", "디버그 켜", "디버그 꺼", "디버그 패널 켜",
        "디버그 패널 꺼", "디버그 패널 열어", "디버그 패널 닫아", "디버그 모드",
        "디버그 패널 보여줘", "디버그 패널 숨겨", "디버그 켜줘", "디버그 꺼줘",
        "디버그 패널 켜줘", "디버그 패널 꺼줘", "디버그 정보 보여줘", "디버그 정보 켜",
        "fps 표시", "fps 보여줘", "fps 켜", "fps 꺼", "성능 표시 켜", "프레임 표시",
        "debug panel", "debug", "the debug panel", "turn on the debug panel",
        "turn off the debug panel", "turn on debug panel", "turn off debug panel",
        "show the debug panel", "hide the debug panel", "show debug panel",
        "hide debug panel", "toggle debug panel", "toggle the debug panel",
        "enable debug panel", "disable debug panel", "open debug panel",
        "close debug panel", "show fps", "hide fps", "debug mode on",
        "debug mode off", "show debug info", "turn the debug panel off",
        "turn the debug panel on",
        # "panel" 생략형(실측 발굴: "toggle debug"→move 오분류) + fps 토글.
        "toggle debug", "toggle fps", "toggle the debug", "debug toggle",
        "switch debug", "switch on debug", "switch off debug", "fps toggle",
        "디버그 토글", "fps 토글", "디버그 전환",
    ]
    for t in debug_phrases:
        out.append((t, {"action": "open_menu", "target": "debug"}))
    # unknown(잡담·게임 질문 — CF explain/chat 폴백 대상).
    for w in _UNKNOWN:
        out.append((w, {"action": "unknown"}))
    return out


def _gen_questions(ssot, rng) -> list[tuple[str, dict]]:
    """게임 명사 × 질문 어미 → unknown(explain route). 명령과 *명사를 공유하되 어미로
    구분* 되도록 대량 생성한다("강남으로 가"=move vs "강남 어디야"=질문). fallback
    안전장치의 일반화를 위해 무한한 질문 표현을 명사 조합으로 근사한다."""
    out = []
    nouns = ([lm["ko"] for lm in ssot["landmarks"]] + list(ssot["archetypes"])
             + ["안전지대", "자동사냥", "자동전투", "파티", "거래", "크리티컬", "경험치",
                "물약", "장비", "세트 아이템", "사냥터", "보스", "레벨", "공격력", "방어력"])
    q_ko = ["{} 뭐야", "{}가 뭐야", "{} 어디 있어", "{} 어디야", "{}에 뭐 나와",
            "{} 어때", "{} 설명해줘", "{} 알려줘", "{} 좋아", "{} 추천해줘", "{}는 어떻게 가"]
    archset = set(ssot["archetypes"])
    for n in nouns:
        it = {"action": "unknown", "semantic_type": "question"}
        if n in archset:
            it["answer_intent"] = "query_monster_info"
        for q in rng.sample(q_ko, k=3):
            out.append((q.format(n), dict(it)))
    en_nouns = list(ssot["archetypes"])[:12] + ["safe zone", "auto hunt", "party", "boss", "the best weapon"]
    for n in en_nouns:
        it = {"action": "unknown", "semantic_type": "question"}
        if n in archset:
            it["answer_intent"] = "query_monster_info"
        for q in ("what is {}", "where is {}", "how about {}", "tell me about {}"):
            out.append((q.format(n), dict(it)))
    return out


def _gen_complex_location(ssot, rng) -> list[tuple[str, dict]]:
    """복합/상대/위치의존 위치 표현 → unknown(=CF 폴백).

    **왜 unknown 인가**: "강남역 동쪽 세이프존"·"가까운 사냥터"·"강북 말고 강남" 은 *공간
    추론*(기준점 대비 방위·거리·정정)이라 단일 landmark 분류기(LCM)로는 풀 수 없다. LCM 이
    *자신있게 틀린 landmark* 를 내는 대신(과거 "강남역 동쪽 세이프존"→gangnam_station 버그),
    이런 발화를 unknown 으로 학습해 **CF(LLM, 좌표 프롬프트 보유)로 폴백**하게 한다.
    단일 landmark(+'근처/쪽' 근사)는 그대로 sml(여기 포함 안 함)."""
    out = []
    lms = [lm for lm in ssot["landmarks"]]
    ko_names = [lm["ko"] for lm in lms]
    rel_dirs = ["동쪽", "서쪽", "남쪽", "북쪽", "동", "서", "남", "북", "왼쪽", "오른쪽", "위", "아래"]
    kinds = ["사냥터", "안전지대", "세이프존", "지역", "곳", "쪽 지역"]

    # (1) landmark + 상대 방위 (+선택적 다른 종류) — "강남역 동쪽으로", "강남역 동쪽 세이프존".
    #     unknown 과다(→ monsters/slot 헤드 희석)를 피해 landmark 당 표본을 절제한다.
    for lm in lms:
        nm = lm["ko"]
        for d in rng.sample(rel_dirs, k=3):
            out.append((f"{nm} {d}으로 가", {"action": "unknown"}))
            out.append((f"{nm} {d}로 가", {"action": "unknown"}))
            out.append((f"{nm} {d} 세이프존으로 이동해", {"action": "unknown"}))
    # (2) 위치 의존(가까운/근처 + 절대 종류) — 현재 위치를 모르므로 LCM 불가.
    for w in ["가까운", "제일 가까운", "가장 가까운", "근처", "주변", "여기서 가까운", "근처에 있는"]:
        for k in kinds:
            out.append((f"{w} {k}로 가", {"action": "unknown"}))
            out.append((f"{w} {k}으로 이동", {"action": "unknown"}))
    # (3) 모호 방위 지역(어느 사냥터인지 불명) — "북쪽 사냥터로", "남쪽 지역으로".
    for d in ["동쪽", "서쪽", "남쪽", "북쪽", "위", "아래"]:
        for k in ["사냥터", "지역", "쪽"]:
            out.append((f"{d} {k}으로 가", {"action": "unknown"}))
            out.append((f"{d} {k}로 가", {"action": "unknown"}))
            out.append((f"{d} {k}로 이동", {"action": "unknown"}))
    # (4) 정정/배제 — "강북 말고 강남", "강남 아니고 관악".
    for a in rng.sample(ko_names, k=min(10, len(ko_names))):
        b = rng.choice(ko_names)
        if a != b:
            out.append((f"{a} 말고 {b}로 가", {"action": "unknown"}))
            out.append((f"{a} 아니고 {b}으로", {"action": "unknown"}))
    return out


def _gen_negation_compound(ssot, rng) -> list[tuple[str, dict]]:
    """부정("사냥하지마")·다중동작("강철 입고 사냥") → unknown(=CF 폴백).

    **부정**: "X하지마" 는 의미 반전(중단/안함)이라 단일 긍정 분류기가 *반대로 실행*("사냥
    하지마"→hunt)하면 치명적. CF(LLM)가 의미 해석하도록 fallback. **다중동작**: 라리엔은
    actions 배열(복합)을 지원하나 단일 action 분류기는 한 동작만 — "물약 먹고 사냥"을 hunt
    하나로 *자신있게 누락* 하므로 CF(다중 action 생성)로 폴백한다."""
    out = []
    # 부정 — 명시 발화(어미별 자연스러운 형태). "X하지마" + "어간+지마"("멈추지마").
    neg_phrases = [
        "사냥하지마", "사냥하지 마", "사냥하지 말고", "사냥 안 할래", "사냥 안 해",
        "이동하지마", "이동하지 마", "공격하지마", "공격하지 말고",
        "멈추지마", "멈추지 마", "멈추지 말고", "멈추지마라",
        "가지마", "가지 마", "가지 말고", "움직이지마", "움직이지 마", "움직이지 말고",
        "물약 먹지마", "물약 먹지 마", "착용하지마", "장착하지마",
        "자동사냥 하지마", "자동사냥 끄지마", "도망가지마", "도망치지마",
        "거기 가지마", "공격하지마라", "사냥하지마라", "하지마", "그러지마",
    ]
    for p in neg_phrases:
        out.append((p, {"action": "unknown"}))
    # 다중동작 — 착용+사냥 / 물약+사냥 / 이동+행동.
    hunts = [lm["ko"] for lm in ssot["landmarks"] if lm["kind"] == "hunt"]
    sets = ["강철", "불멸", "빅터", "판금"]
    for s in sets:
        for l in rng.sample(hunts, k=3):
            out.append((f"{s} 세트 입고 {l}에서 사냥", {"action": "unknown"}))
            out.append((f"{s} 입고 {l} 사냥", {"action": "unknown"}))
            out.append((f"{s} 착용하고 {l} 사냥", {"action": "unknown"}))
    for l in rng.sample(hunts, k=8):
        out.append((f"물약 먹고 {l}에서 사냥", {"action": "unknown"}))
        out.append((f"{l} 가서 물약 먹어", {"action": "unknown"}))
        out.append((f"{l}에서 사냥하고 물약 먹어", {"action": "unknown"}))
    return out


def _gen_context_dependent(rng) -> list[tuple[str, dict]]:
    """맥락 지시어("거기서"·"그것도"·"아까") 발화 → unknown(=CF 멀티턴 폴백).

    단일 발화 분류기는 이전 발화 맥락이 없어 "거기"(어느 곳)·"그것"(무엇)을 풀 수 없다.
    자신있게 틀리는 대신("그것도 착용"→auto_combat 오류) CF(멀티턴 세션)로 폴백한다."""
    ctx = [
        "거기서 사냥", "거기로 가", "거기서 잡아", "거기 가자", "거기로 이동",
        "그것도 착용", "그거 써", "그거 먹어", "그것 장착", "그거 입어",
        "아까 거기로", "아까 그곳으로", "방금 그거", "방금 거기로", "방금 그 위치",
        "그 다음 사냥", "그 다음 물약", "그 다음에 이동", "그러고 나서 사냥",
        "계속 사냥", "계속 이동", "또 해줘", "한 번 더 해", "그거 취소", "방금 명령 취소",
        "거기 말고 딴 데", "그쪽 말고", "아까 그 몬스터", "거기 있는 거",
    ]
    return [(t, {"action": "unknown"}) for t in ctx]


def _gen_polite(ssot, rng) -> list[tuple[str, dict]]:
    """존댓말 어미("~해 주세요/주실래요/주시겠어요") — 실사용 흔한 공손 표현 sml 화."""
    out = []
    hon = ["주세요", "주실래요", "주시겠어요", "주시겠어요?", "줄래요"]
    # stop
    for v in ["멈춰", "정지해", "그만해", "멈춰"]:
        for h in rng.sample(hon, k=3):
            out.append((f"{v} {h}", {"action": "stop"}))
    # potion(hp 기본)
    for v in ["물약", "체력 물약", "회복 물약"]:
        for h in rng.sample(hon, k=2):
            out.append((f"{v} {h}", {"action": "potion", "potion": "hp"}))
    # open_menu
    menu_ko = {"menu": "메뉴", "inventory": "인벤토리", "groupchat": "채팅", "potion": "물약창"}
    for tgt, nm in menu_ko.items():
        for h in rng.sample(hon, k=2):
            out.append((f"{nm} 열어 {h}", {"action": "open_menu", "target": tgt}))
    # move / hunt (landmark)
    for lm in ssot["landmarks"]:
        al = lm["ko"]
        for h in rng.sample(hon, k=2):
            out.append((f"{al}으로 가 {h}", {"action": "move", "location": lm["id"]}))
        if lm["kind"] == "hunt":
            for h in rng.sample(hon, k=2):
                out.append((f"{al}에서 사냥해 {h}", {"action": "hunt", "location": lm["id"]}))
    return out


def _gen_smalltalk(rng) -> list[tuple[str, dict]]:
    """게임과 *무관한* 일상 문장을 unknown 으로(outlier exposure). softmax 분류기는
    학습 분포 밖(OOD) 입력을 가까운 명령으로 *과신* 하는데("날씨 좋다"→hunt conf 1.0),
    게임 무관 일상 문장을 unknown 경계로 노출해 OOD 일반화 fallback 을 끌어올린다."""
    topics = ["날씨", "점심", "저녁", "커피", "영화", "음악", "주말", "여행", "운동", "잠",
              "책", "드라마", "게임", "친구", "가족", "회사", "학교", "숙제", "시험", "월급",
              "고양이", "강아지", "라면", "치킨", "피자", "비", "눈", "바람", "기분", "꿈"]
    tmpl = ["{} 좋다", "{} 어때", "오늘 {} 생각나", "{} 하고 싶다", "{} 별로야",
            "{} 너무 좋아", "{} 했어", "{} 싫어", "{} 최고야", "어제 {} 봤어"]
    out = []
    for t in topics:
        for tm in rng.sample(tmpl, k=3):
            out.append((tm.format(t), {"action": "unknown"}))
    fixed = [
        "배고프다", "졸려", "피곤해", "행복해", "슬퍼", "화가 나", "재밌다", "지루하다",
        "사랑해", "보고 싶어", "고생했어", "축하해", "미안해", "괜찮아", "잘했어",
        "지금 몇 시야", "내일 비 와", "주말 잘 보내", "맛있겠다", "예쁘다", "멋지다",
        "그게 무슨 말이야", "이해가 안 돼", "다시 말해줘", "잠깐만", "글쎄", "아마도",
        # "뭐 할까/언제/할 수 있어" 패턴(명령 아님 — 의향·일정·가능성 질문).
        "오늘 뭐 할까", "내일 뭐 하지", "이제 뭐 하지", "다음 뭐 할까", "언제 끝나",
        "언제 시작해", "언제 와", "이거 할 수 있어", "그거 가능해", "될까", "되나요",
        "얼마나 걸려", "몇 시간 남았어", "지금 해도 돼", "해도 될까", "뭐가 좋을까",
        "i'm hungry", "i'm tired", "good night", "see you tomorrow", "that's funny",
        "i love it", "nice weather", "what's up", "long time no see", "take care",
        "어디 가", "뭐 해", "왜 그래", "어떻게 생각해", "진짜야", "대박", "헐", "응 알겠어",
    ]
    for f in fixed:
        out.append((f, {"action": "unknown"}))
    return out


def _jamo_variant(text: str, rng) -> str:
    """한글 음소 혼동 변형(받침/초성/중성 혼동 + 탈락 — phonetics 공유 매트릭스). 기존
    '받침 탈락+모음 ±1' 보다 실제 STT 오류(간남·먼춰·문약)를 잘 모사한다."""
    return phonetic_noise(text, rng)


# ── LCM v2 (R2 의미 게이트 + R4a QnA) 라벨 데이터 ─────────────────────────────
# 잡담(chat) — 게임 무관·인사. 클라는 cloud(또는 고정 응답).
_V2_CHAT = [
    "안녕", "안녕하세요", "반가워", "고마워", "수고해",  # 정체성 질문 → query_assistant_identity
    "잘 자", "또 보자", "사랑해", "재밌다", "심심해", "지루해", "오늘 날씨 어때", "밥 먹었어",
    "뭐하고 놀까", "노래 불러줘", "농담 해줘", "주식 사도 돼", "내일 비 와", "점심 뭐 먹지",
    "hello", "hi", "hey", "thanks", "thank you",
    "good job", "see you", "i'm bored", "tell me a joke",
]
# 게임 질문(question). (발화, answer_intent) — None 이면 로컬 답변 토픽 미상(→ cloud 후보).
_V2_QUESTION = [
    ("내 레벨 몇이야", "query_player_level"), ("나 몇 레벨이야", "query_player_level"),
    ("지금 몇 레벨", "query_player_level"), ("내 레벨 알려줘", "query_player_level"),
    ("나 몇렙이야", "query_player_level"), ("내 렙 뭐야", "query_player_level"),
    ("레벨 알려줘", "query_player_level"), ("what level am i", "query_player_level"),
    ("my level", "query_player_level"), ("나 레벨 몇", "query_player_level"),
    ("내 레벨이 몇이야", "query_player_level"), ("현재 레벨 얼마야", "query_player_level"),
    ("레벨 보여줘", "query_player_level"), ("나 몇 렙", "query_player_level"),
    ("렙 몇이야", "query_player_level"), ("내 레벨 확인", "query_player_level"),
    ("지금 내 레벨", "query_player_level"), ("나 레벨 알려줘", "query_player_level"),
    ("몇 레벨이지", "query_player_level"), ("내가 몇 레벨", "query_player_level"),
    # 플레이어 *자기* 캐릭터 이름(query_player_name) — AI 정체성(query_assistant_identity "너는
    # 누구야")과 반드시 *구분*. "나/내/제/내가"=플레이어, "너/넌/라리아/당신"=AI. (2026-06-25
    # "나는 누구야"→"저는 라리아입니다" 오분류 회귀 — 명시 샘플로 "나/너" 경계 학습.)
    ("나는 누구야", "query_player_name"), ("내 캐릭터 이름이 뭐야", "query_player_name"),
    ("내 이름이 뭐야", "query_player_name"), ("나 누구야", "query_player_name"),
    ("내 캐릭터 이름 알려줘", "query_player_name"), ("제 캐릭터 이름이 뭐예요", "query_player_name"),
    ("나의 이름은 뭐야", "query_player_name"), ("내가 누구야", "query_player_name"),
    ("내 캐릭터 이름 말해줘", "query_player_name"), ("나 이름 뭐야", "query_player_name"),
    ("내 닉네임이 뭐야", "query_player_name"), ("내 캐릭 이름", "query_player_name"),
    ("내 캐릭터 누구야", "query_player_name"), ("나 이름이 뭐지", "query_player_name"),
    ("내 캐릭터 이름은", "query_player_name"), ("나 이름 알려줘", "query_player_name"),
    ("what is my name", "query_player_name"), ("my character name", "query_player_name"),
    ("who am i", "query_player_name"), ("what's my character name", "query_player_name"),
    ("tell me my name", "query_player_name"), ("my name", "query_player_name"),
    # AI 정체성(query_assistant_identity) — "너/라리아" 가 핵심. "나"(query_player_name)와 대조.
    ("너는 누구야", "query_assistant_identity"), ("넌 누구니", "query_assistant_identity"),
    ("너 이름 뭐야", "query_assistant_identity"), ("라리아 누구야", "query_assistant_identity"),
    ("당신은 누구세요", "query_assistant_identity"), ("너 이름이 뭐니", "query_assistant_identity"),
    ("who are you", "query_assistant_identity"), ("what's your name", "query_assistant_identity"),
    ("내 레벨에 맞는 사냥터 어디야", "query_recommended_hunt_zone"),
    ("어디서 사냥하면 좋아", "query_recommended_hunt_zone"),
    ("추천 사냥터 어디야", "query_recommended_hunt_zone"),
    ("지금 어디서 사냥해", "query_recommended_hunt_zone"),
    ("내 레벨로 어디서 사냥해", "query_recommended_hunt_zone"),
    ("어디 가서 사냥할까", "query_recommended_hunt_zone"),
    ("where should i hunt", "query_recommended_hunt_zone"),
    ("사냥터 추천해줘", "query_recommended_hunt_zone"),
    ("갈 만한 사냥터 어디", "query_recommended_hunt_zone"),
    ("내 레벨 사냥터", "query_recommended_hunt_zone"),
    ("어디 사냥 가면 돼", "query_recommended_hunt_zone"),
    ("사냥 어디서 해", "query_recommended_hunt_zone"),
    ("맞는 사냥터 추천", "query_recommended_hunt_zone"),
    ("레벨에 맞는 곳 어디", "query_recommended_hunt_zone"),
    ("이 몬스터 뭐야", "query_monster_info"), ("제일 센 몬스터 뭐야", "query_monster_info"),
    # ── Smart LCM 고정지식 7종(한/영) — 답변은 게임지식 OTA answers 에서 조회 ──
    ("물약 효과 뭐야", "query_potion_effect"), ("물약 종류 뭐 있어", "query_potion_effect"),
    ("포션 효과 알려줘", "query_potion_effect"), ("어떤 물약 있어", "query_potion_effect"),
    ("hp 물약 효과", "query_potion_effect"), ("공격속도 물약 뭐야", "query_potion_effect"),
    ("물약 뭐 있어", "query_potion_effect"), ("달리기 물약 효과", "query_potion_effect"),
    ("what do potions do", "query_potion_effect"), ("potion effects", "query_potion_effect"),
    ("what potions are there", "query_potion_effect"),
    ("빅터 세트 효과", "query_gear_set_effect"), ("불멸 세트 뭐야", "query_gear_set_effect"),
    ("강철 세트 효과", "query_gear_set_effect"), ("세트 효과 알려줘", "query_gear_set_effect"),
    ("세트 아이템 뭐야", "query_gear_set_effect"), ("장비 세트 효과", "query_gear_set_effect"),
    ("어떤 세트가 강해", "query_gear_set_effect"), ("세트 종류 뭐 있어", "query_gear_set_effect"),
    ("victor set effect", "query_gear_set_effect"), ("immortal set bonus", "query_gear_set_effect"),
    ("what is gear set", "query_gear_set_effect"), ("set item effect", "query_gear_set_effect"),
    ("레벨업 어떻게 해", "query_level_progression"), ("레벨 어떻게 올려", "query_level_progression"),
    ("경험치 어떻게 벌어", "query_level_progression"), ("강해지려면 어떻게 해", "query_level_progression"),
    ("레벨업 방법", "query_level_progression"), ("최대 레벨 몇이야", "query_level_progression"),
    ("어떻게 강해져", "query_level_progression"), ("렙업 어떻게", "query_level_progression"),
    ("how do i level up", "query_level_progression"), ("how do i get stronger", "query_level_progression"),
    ("max level", "query_level_progression"), ("how to gain exp", "query_level_progression"),
    ("파티 어떻게 만들어", "query_party_info"), ("파티 뭐야", "query_party_info"),
    ("파티 어떻게 해", "query_party_info"), ("같이 사냥 어떻게 해", "query_party_info"),
    ("파티 인원 몇명", "query_party_info"), ("파티 경험치 어떻게", "query_party_info"),
    ("파티 맺는 법", "query_party_info"),
    ("how does party work", "query_party_info"), ("how do i make a party", "query_party_info"),
    ("교환 어떻게 해", "query_trade_info"), ("거래 어떻게 해", "query_trade_info"),
    ("교환 뭐야", "query_trade_info"), ("아이템 교환 방법", "query_trade_info"),
    ("물약 교환 어떻게", "query_trade_info"), ("거래 방법 알려줘", "query_trade_info"),
    ("how to trade", "query_trade_info"), ("how does trade work", "query_trade_info"),
    ("세계관이 뭐야", "query_world_lore"), ("이 게임 스토리 뭐야", "query_world_lore"),
    ("옵시디언이 뭐야", "query_world_lore"), ("넥서스가 뭐야", "query_world_lore"),
    ("이 게임 배경 뭐야", "query_world_lore"), ("왜 ai가 나와", "query_world_lore"),
    ("게임 세계관 알려줘", "query_world_lore"), ("라리엔이 뭐야", "query_world_lore"),
    ("what is this world", "query_world_lore"), ("game story", "query_world_lore"),
    ("who is obsidian", "query_world_lore"), ("what is the lore", "query_world_lore"),
    ("어떻게 조작해", "query_help_controls"), ("어떻게 플레이해", "query_help_controls"),
    ("조작법 알려줘", "query_help_controls"), ("어떻게 움직여", "query_help_controls"),
    ("이 게임 어떻게 해", "query_help_controls"), ("도움말", "query_help_controls"),
    ("튜토리얼 보여줘", "query_help_controls"), ("자동사냥이 뭐야", "query_help_controls"),
    ("how do i play", "query_help_controls"), ("how to control", "query_help_controls"),
    ("how do i move", "query_help_controls"),
    # 일반 질문(answer_intent 없음 → cloud, AI 챗봇이 답) — 고정지식으로 못 답하는 것만.
    ("물약 어디서 사", None), ("왜 자꾸 죽어", None), ("크리티컬이 뭐야", None),
]

# Smart LCM 고정지식 의도 — 주어 × 어미 동적 증강 SSOT(각 ~50개, 한/영). _V2_QUESTION 의
# 명시 발화에 더해 표현 다양성을 키워 표면 패턴 과적합을 완화한다. 측정: monster_info 가
# "X 뭐야" 패턴을 독점(76%)해 "세계관이 뭐야"가 monster 로 끌려갔다 → 주어 토큰(세계관·물약·
# 파티…)에 의도가 묶이도록 주어를 다양화하고 의도별 데이터를 균형 맞춘다.
_FIXED_INTENT_AUG = {
    "query_potion_effect": (
        ["물약", "포션", "물약은", "물약들", "회복 물약", "버프 물약", "potion", "potions"],
        ["효과 뭐야", "효과 알려줘", "뭐가 있어", "어떤 게 있어", "설명해줘", "종류 뭐야",
         "효과가 궁금해", "what do they do", "effects", "list them", "explain"]),
    "query_gear_set_effect": (
        ["세트", "세트 아이템", "장비 세트", "빅터 세트", "불멸 세트", "강철 세트", "gear set", "set"],
        ["효과 뭐야", "효과 알려줘", "뭐가 좋아", "설명해줘", "능력치 뭐야", "보너스 뭐야",
         "effect", "bonus", "what does it give", "explain"]),
    "query_level_progression": (
        ["레벨", "레벨업", "경험치", "렙업", "캐릭터"],
        ["어떻게 올려", "어떻게 해", "방법 알려줘", "빨리 올리는 법", "어떻게 벌어", "어떻게 강해져",
         "how to level up", "how does it work", "how to get stronger"]),
    "query_party_info": (
        ["파티", "파티는", "같이 사냥", "협동 사냥", "party"],
        ["어떻게 만들어", "어떻게 해", "뭐야", "방법 알려줘", "인원 몇명", "경험치 어떻게",
         "how does it work", "how to make", "explain"]),
    "query_trade_info": (
        ["교환", "거래", "아이템 교환", "물약 교환", "trade", "trading"],
        ["어떻게 해", "방법 알려줘", "뭐야", "어떻게 하는거야", "how to do it",
         "how does it work", "explain"]),
    "query_world_lore": (
        ["세계관", "스토리", "이 게임 배경", "옵시디언", "넥서스", "게임 이야기", "lore", "the world"],
        ["뭐야", "알려줘", "설명해줘", "어떤 거야", "궁금해",
         "what is it", "tell me", "explain"]),
    "query_help_controls": (
        ["조작", "조작법", "이 게임", "게임", "플레이", "controls"],
        ["어떻게 해", "어떻게 하는거야", "방법 알려줘", "어떻게 움직여", "어떻게 시작해",
         "how to play", "how does it work", "how do i start"]),
}

# 괴상/무의미(nonsense) — STT 붕괴. 클라는 reject(되묻기).
# 명령 단어(캐스터·강남·멈춰·물약 등)를 *넣지 않는다* — 넣으면 모델이 그 단어를 nonsense
# 신호로 학습해 진짜 명령까지 침범한다. 순수 단음절/감탄사/영문난타만.
_V2_NONSENSE_SEED = [
    "므 어 저기 음 그", "어 음 그 저 막", "끄아 으악 뿡", "아아아 어 머", "사 어 음 캬",
    "asdk qwe asdf", "sdf qwe bcd", "어버 그저 막에", "음 그 캬 뿡", "으 어 흠 막 그",
]


def _gen_v2_semantic(ssot, rng) -> list[tuple[str, dict]]:
    """R2/R4a 라벨 데이터 — semantic_type(question/chat/nonsense) + answer_intent."""
    out: list[tuple[str, dict]] = []
    for t in _V2_CHAT:
        out.append((t, {"action": "unknown", "semantic_type": "chat"}))
    for t, ai in _V2_QUESTION:
        it = {"action": "unknown", "semantic_type": "question"}
        if ai:
            it["answer_intent"] = ai
        out.append((t, it))
    # 몬스터 정보 질문 — archetype × 표현. "뭐야" 표면 편향을 막으려 표현을 다양화하고
    # 전체에서 다운샘플(과대표 방지: ~600→160). monster 가 'X 뭐야' 패턴을 독점하면
    # 다른 의도("세계관이 뭐야")가 monster 로 끌려가는 표면 과적합이 측정됐다.
    mob_cases: list[tuple[str, dict]] = []
    for arch in ssot["archetypes"]:
        for nm in _MONSTER_KO.get(arch, [arch]):
            for q in ("뭐야", "어때", "세냐", "약점이 뭐야", "몇 레벨이야", "스탯 알려줘",
                      "강해", "정보 알려줘"):
                mob_cases.append((f"{nm} {q}", {"action": "unknown",
                                  "semantic_type": "question",
                                  "answer_intent": "query_monster_info"}))
    rng.shuffle(mob_cases)
    out.extend(mob_cases[:160])
    # Smart LCM 고정지식 의도 — 주어 × 어미 동적 증강(각 ~50-80개, 한/영). 의도별 균형 +
    # 주어 토큰(세계관·물약·파티…)에 의도를 묶어 표면 패턴 과적합을 완화한다.
    for intent, (subjects, suffixes) in _FIXED_INTENT_AUG.items():
        for s in subjects:
            for suf in suffixes:
                out.append((f"{s} {suf}".strip(), {"action": "unknown",
                            "semantic_type": "question", "answer_intent": intent}))
    # 무의미 — 시드 + 명령을 *심하게* 깨뜨리고 단음절 filler 삽입(2중 음소 노이즈).
    for t in _V2_NONSENSE_SEED:
        out.append((t, {"action": "unknown", "semantic_type": "nonsense"}))
    # 무의미 — *명령 단어를 쓰지 않고* 순수 단음절/감탄사 나열로 생성. 명령을 깨뜨리면("캐스터
    # 사냥"→"캐스 터…") 모델이 명령 단어를 nonsense 신호로 학습해 진짜 명령("캐스터 사냥"·
    # "간남으로 가")까지 nonsense 로 침범한다. 음소 오타 명령(R1, 1중)과 명확히 분리한다.
    syl = ["어", "음", "그", "저", "막", "에", "으", "뭐", "캬", "뿡", "쿵", "끄", "악",
           "아", "우", "흠", "엥", "쩜", "냐", "꺄", "뀨", "삐"]
    for _ in range(280):
        n = rng.randint(3, 6)
        out.append((" ".join(rng.choice(syl) for _ in range(n)),
                    {"action": "unknown", "semantic_type": "nonsense"}))
    return out


def generate(ssot: dict, seed: int = 7) -> list[dict]:
    """모든 템플릿을 펼쳐 (text, intent) 페어 리스트를 만든다(재현 가능)."""
    rng = random.Random(seed)
    pairs: list[tuple[str, dict]] = []
    pairs += _gen_move(ssot, rng)
    pairs += _gen_hunt(ssot, rng)
    pairs += _gen_simple(ssot, rng)
    pairs += _gen_typo_robustness(ssot, rng)
    pairs += _gen_questions(ssot, rng)
    pairs += _gen_complex_location(ssot, rng)
    pairs += _gen_negation_compound(ssot, rng)
    pairs += _gen_context_dependent(rng)
    pairs += _gen_polite(ssot, rng)
    pairs += _gen_v2_semantic(ssot, rng)  # R2/R4a — question/chat/nonsense + answer_intent.
    pairs += gen_i18n()  # 한·영·중·일 4개 언어 동등(backbone 전제, 2026-06-24 사용자).
    # 중복 제거(같은 발화는 한 번만 — 마지막 라벨 우선).
    dedup: dict[str, dict] = {}
    for text, intent in pairs:
        dedup[text.strip()] = intent
    rows = [{"text": t, "intent": i} for t, i in dedup.items()]
    # 결정적 음소 혼동 변형(명령 발화만) — phonetic 강건성 안정화. *전체* 명령에 발화당 1개
    # 변형(받침/초성/중성 혼동). 2개는 노이즈 과다로 *깨끗한 정상 발화* confidence 가 하락하는
    # 회귀가 측정됐다(R1 튜닝: "여기서 사냥하자" sml→fb). dataset aug_p 와 합쳐 균형을 잡는다.
    rng2 = random.Random(seed + 13)
    # 명령 + 게임 질문(question)에 음소 변형 — "내 랩 몇이야" 같은 STT 오타도 robust 하게.
    # nonsense/chat 은 변형하지 않는다(그 자체가 라벨).
    cmd = [r for r in rows if r["intent"]["action"] != "unknown"
           or r["intent"].get("semantic_type") == "question"]
    for r in cmd:
        v = _jamo_variant(r["text"], rng2)
        if v.strip() and v not in dedup:
            dedup[v] = r["intent"]
            rows.append({"text": v, "intent": r["intent"]})
    return rows
