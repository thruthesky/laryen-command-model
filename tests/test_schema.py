"""encode_intent → decode_intent 왕복 검증(torch 불필요 — 순수 파이썬)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcm.schema import LabelSpace, decode_intent, encode_intent, load_ssot  # noqa: E402

LS = LabelSpace(load_ssot())

CASES = [
    {"action": "move", "location": "safe"},
    {"action": "move", "direction": 270.0},
    {"action": "hunt", "location": "gangnam_station", "monsters": ["Caster"],
     "retreatToSafeZone": True, "retreatHpPct": 30},
    {"action": "hunt"},
    {"action": "potion", "potion": "hp"},
    {"action": "equip", "set": "plate"},
    {"action": "equip", "gear": "victor_weapon"},
    {"action": "unequip", "slot": "weapon"},
    {"action": "auto_combat", "mode": "auto_hunt"},
    {"action": "open_menu", "target": "inventory"},
    {"action": "auto_potion", "potions": ["hp"], "enable": True},
    {"action": "stop"},
    {"action": "unknown"},
]


def test_roundtrip():
    for intent in CASES:
        heads = encode_intent(intent, LS)
        back = decode_intent(heads, LS)
        assert back == intent, f"{intent} → {back}"


def test_direction_bucketing():
    # 269 → 270 버킷으로 수렴.
    heads = encode_intent({"action": "move", "direction": 269}, LS)
    back = decode_intent(heads, LS)
    assert back["direction"] == 270.0


def test_all_synth_labels_valid():
    from lcm.synth import generate
    for row in generate(load_ssot()):
        intent = row["intent"]
        assert intent["action"] in LS.actions
        # encode 가 예외 없이 되고, action 이 보존돼야 한다.
        heads = encode_intent(intent, LS)
        assert LS.actions[heads["action"]] == intent["action"]
