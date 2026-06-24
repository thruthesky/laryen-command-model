"""라벨 공간(LabelSpace) + intent ↔ 라벨 인코딩/디코딩.

**설계 핵심 — 생성이 아니라 분류**: 라리엔 VoiceIntent 출력은 자유 텍스트가 아니라
고정 스키마(action enum + 슬롯)다. 따라서 모델은 디코더로 JSON 을 *생성* 하지 않고,
여러 분류 헤드(action, location, monsters …)로 각 슬롯을 *예측* 한다. 이 파일은 그
헤드들의 라벨 공간을 ssot.json(= 라리엔 dart SSOT 추출물)에서 동적으로 만들고,
intent dict ↔ 정수/멀티핫 라벨을 왕복 변환한다.

**1차 범위 — single-action**: 모델은 *한 발화당 1개 action* 을 예측한다. 라리엔
스키마는 actions 배열(복합 명령 "강철 입고 사냥")을 지원하지만, 복합/모호 발화는
confidence 가 낮으므로 3차 CF Gemini 로 폴백한다(README 의 3계층). decode 결과는
항상 actions 배열에 1개를 담아 라리엔 parseVoiceCommand 와 같은 형태로 돌려준다.
"""
from __future__ import annotations

import json
from pathlib import Path

SSOT_PATH = Path(__file__).resolve().parents[1] / "config" / "ssot.json"

# 부가 슬롯(HP %) 버킷 — huntHpPotionPct / retreatHpPct (0~100 을 10단위로).
HP_BUCKETS = ["<none>", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
# auto_potion 의 potions — 물약 4종 + 'all'(전부). archetype 과 무관한 별도 멀티라벨.
AUTO_POTION_LABELS_SUFFIX = ["all"]
NONE = "<none>"

# ── LCM v2 (R2/R4a) ──────────────────────────────────────────────────────────
# semantic_type — 1단계 의미 게이트(plan §2.2). route 결정의 핵심 신호:
#   command(로컬 실행) / question(게임 QnA→answer_local) / chat(잡담→cloud) /
#   nonsense(STT 붕괴→reject). confidence 로는 안 갈리므로(측정) *학습된 헤드* 로 판단.
SEMANTIC_TYPES = ["command", "question", "chat", "nonsense"]
# answer_intent — 게임 QnA 토픽(plan §2.4, R4a 1차). 실제 답변은 클라가 게임 상태/SSOT 에서
# 조립(상태값은 학습 안 함). <none> = 질문 아님/토픽 미상.
ANSWER_INTENTS = [
    NONE,
    "query_player_level",          # "내 레벨 몇이야"
    "query_recommended_hunt_zone", # "내 레벨에 맞는 사냥터 어디야"
    "query_monster_info",          # "캐스터 뭐야"
]


def load_ssot(path: Path | str = SSOT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class LabelSpace:
    """ssot.json → 각 분류 헤드의 라벨 목록 + 인덱스. 모델 헤드 차원의 SSOT."""

    def __init__(self, ssot: dict):
        self.ssot = ssot
        # single-label 헤드(softmax). 첫 원소는 <none>(해당 action 이 아닐 때).
        self.actions = list(ssot["actions"])
        self.locations = [NONE] + list(ssot["landmark_ids"])
        self.directions = [NONE] + [str(b) for b in ssot["direction_buckets"]]
        self.potions = [NONE] + list(ssot["potions"])
        self.gear_sets = [NONE] + list(ssot["gear_sets"])
        self.gear_singles = [NONE] + list(ssot["gear_singles"])
        self.slots = [NONE] + list(ssot["unequip_slots"])
        self.modes = [NONE] + list(ssot["combat_modes"])
        self.targets = [NONE] + list(ssot["menu_targets"])
        self.hunt_hp = list(HP_BUCKETS)
        self.retreat_hp = list(HP_BUCKETS)
        # multi-label 헤드(sigmoid). monsters=archetype, auto_potions=물약+all.
        self.archetypes = list(ssot["archetypes"])
        self.auto_potions = list(ssot["potions"]) + AUTO_POTION_LABELS_SUFFIX
        # LCM v2 — 의미 게이트(R2) + QnA(R4a) 헤드.
        self.semantic_types = list(SEMANTIC_TYPES)
        self.answer_intents = list(ANSWER_INTENTS)

    # ── 헤드 정의: (이름, 종류, 라벨목록). 종류 = 'single' | 'multi' | 'binary' ──
    def heads(self) -> list[tuple[str, str, list[str]]]:
        return [
            ("action", "single", self.actions),
            ("location", "single", self.locations),
            ("direction", "single", self.directions),
            ("potion", "single", self.potions),
            ("gear_set", "single", self.gear_sets),
            ("gear_single", "single", self.gear_singles),
            ("slot", "single", self.slots),
            ("mode", "single", self.modes),
            ("target", "single", self.targets),
            ("hunt_hp", "single", self.hunt_hp),
            ("retreat_hp", "single", self.retreat_hp),
            ("monsters", "multi", self.archetypes),
            ("auto_potions", "multi", self.auto_potions),
            ("retreat_to_safe", "binary", ["0", "1"]),
            ("auto_potion_enable", "binary", ["0", "1"]),
            # LCM v2 — 의미 게이트(R2) + QnA 토픽(R4a).
            ("semantic_type", "single", self.semantic_types),
            ("answer_intent", "single", self.answer_intents),
        ]

    def index(self, head: str, label: str) -> int:
        labels = dict((n, ls) for n, _, ls in self.heads())[head]
        return labels.index(label) if label in labels else 0

    @staticmethod
    def from_ssot_file(path: Path | str = SSOT_PATH) -> "LabelSpace":
        return LabelSpace(load_ssot(path))


def _hp_bucket(pct) -> str:
    if pct is None:
        return NONE
    try:
        v = max(10, min(90, round(int(pct) / 10) * 10))
    except (TypeError, ValueError):
        return NONE
    return str(v)


def _dir_bucket(deg, buckets: list[str]) -> int:
    """0~359 → 가장 가까운 22.5° 버킷 인덱스(+1, 0 은 <none>)."""
    if deg is None:
        return 0
    i = round((float(deg) % 360) / 22.5) % 16
    return i + 1  # directions[0] = <none>


def encode_intent(intent: dict, ls: LabelSpace) -> dict:
    """라리엔 action JSON 1개(dict) → 헤드별 라벨(정수/멀티핫/이진)."""
    a = intent.get("action", "unknown")
    out: dict = {"action": ls.actions.index(a) if a in ls.actions else ls.actions.index("unknown")}
    # 기본값(<none>/0).
    out.update({
        "location": 0, "direction": 0, "potion": 0, "gear_set": 0, "gear_single": 0,
        "slot": 0, "mode": 0, "target": 0, "hunt_hp": 0, "retreat_hp": 0,
        "monsters": [0] * len(ls.archetypes),
        "auto_potions": [0] * len(ls.auto_potions),
        "retreat_to_safe": 0, "auto_potion_enable": 1,
    })
    if a in ("move", "hunt"):
        loc = intent.get("location")
        if loc in ls.locations:
            out["location"] = ls.locations.index(loc)
    if a == "move":
        out["direction"] = _dir_bucket(intent.get("direction"), ls.directions)
    if a == "hunt":
        for m in intent.get("monsters", []) or []:
            if m in ls.archetypes:
                out["monsters"][ls.archetypes.index(m)] = 1
        out["hunt_hp"] = ls.hunt_hp.index(_hp_bucket(intent.get("huntHpPotionPct")))
        out["retreat_hp"] = ls.retreat_hp.index(_hp_bucket(intent.get("retreatHpPct")))
        out["retreat_to_safe"] = 1 if intent.get("retreatToSafeZone") else 0
    if a == "potion":
        p = intent.get("potion")
        if p in ls.potions:
            out["potion"] = ls.potions.index(p)
    if a == "equip":
        s = intent.get("set")
        g = intent.get("gear")
        if s in ls.gear_sets:
            out["gear_set"] = ls.gear_sets.index(s)
        if g in ls.gear_singles:
            out["gear_single"] = ls.gear_singles.index(g)
    if a == "unequip" and intent.get("slot") in ls.slots:
        out["slot"] = ls.slots.index(intent["slot"])
    if a == "auto_combat" and intent.get("mode") in ls.modes:
        out["mode"] = ls.modes.index(intent["mode"])
    if a == "open_menu" and intent.get("target") in ls.targets:
        out["target"] = ls.targets.index(intent["target"])
    if a == "auto_potion":
        for p in intent.get("potions", []) or []:
            if p in ls.auto_potions:
                out["auto_potions"][ls.auto_potions.index(p)] = 1
        out["auto_potion_enable"] = 0 if intent.get("enable") is False else 1
    # LCM v2 — 의미 게이트(R2): 미지정 시 명령은 command, action=unknown 은 chat 으로 본다.
    st = intent.get("semantic_type") or ("command" if a != "unknown" else "chat")
    out["semantic_type"] = (
        ls.semantic_types.index(st) if st in ls.semantic_types
        else ls.semantic_types.index("chat"))
    ai = intent.get("answer_intent", NONE)
    out["answer_intent"] = ls.answer_intents.index(ai) if ai in ls.answer_intents else 0
    return out


def decode_intent(heads: dict, ls: LabelSpace) -> dict:
    """헤드별 *예측 인덱스* (argmax/threshold 적용 후) → 라리엔 action JSON 1개.

    `heads` 값은 정수(single/binary) 또는 0/1 리스트(multi). say 는 비워 둔다 —
    라리엔 _parseOneIntent 가 say 가 비면 기본 요약을 채운다.
    """
    a = ls.actions[heads["action"]]
    out: dict = {"action": a}
    if a == "move":
        loc = ls.locations[heads["location"]]
        if loc != NONE:
            out["location"] = loc
        else:
            d = ls.directions[heads["direction"]]
            if d != NONE:
                out["direction"] = float(d)
    elif a == "hunt":
        loc = ls.locations[heads["location"]]
        if loc != NONE:
            out["location"] = loc
        mons = [ls.archetypes[i] for i, v in enumerate(heads["monsters"]) if v]
        if mons:
            out["monsters"] = mons
        if ls.hunt_hp[heads["hunt_hp"]] != NONE:
            out["huntHpPotionPct"] = int(ls.hunt_hp[heads["hunt_hp"]])
        if heads["retreat_to_safe"]:
            out["retreatToSafeZone"] = True
        if ls.retreat_hp[heads["retreat_hp"]] != NONE:
            out["retreatHpPct"] = int(ls.retreat_hp[heads["retreat_hp"]])
    elif a == "potion":
        p = ls.potions[heads["potion"]]
        if p != NONE:
            out["potion"] = p
    elif a == "equip":
        s = ls.gear_sets[heads["gear_set"]]
        g = ls.gear_singles[heads["gear_single"]]
        if s != NONE:
            out["set"] = s
        elif g != NONE:
            out["gear"] = g
    elif a == "unequip":
        s = ls.slots[heads["slot"]]
        if s != NONE:
            out["slot"] = s
    elif a == "auto_combat":
        m = ls.modes[heads["mode"]]
        if m != NONE:
            out["mode"] = m
    elif a == "open_menu":
        t = ls.targets[heads["target"]]
        if t != NONE:
            out["target"] = t
    elif a == "auto_potion":
        pots = [ls.auto_potions[i] for i, v in enumerate(heads["auto_potions"]) if v]
        if pots:
            out["potions"] = pots
        out["enable"] = bool(heads["auto_potion_enable"])
    # LCM v2 — 의미 게이트 + QnA 토픽(클라가 route 결정·답변 조립에 사용).
    if "semantic_type" in heads:
        out["semantic_type"] = ls.semantic_types[heads["semantic_type"]]
        ai = ls.answer_intents[heads["answer_intent"]]
        if ai != NONE:
            out["answer_intent"] = ai
    return out


def to_voice_command_json(intent: dict, say: str = "") -> dict:
    """라리엔 parseVoiceCommand 가 읽는 최상위 형태 {"actions":[...], "say":...}."""
    return {"actions": [intent], "say": say}
