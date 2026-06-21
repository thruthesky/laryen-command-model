"""완료조건 1·4 — golden 발화셋에 대한 LCM 정확도 회귀 테스트.

학습된 체크포인트(checkpoints/lcm.pt)가 있을 때만 실행한다(없으면 skip — CI 가드).
action 정확도와 3계층 폴백(unknown/저신뢰 → fallback)을 검증한다. 임계값 미달이면
학습이 회귀한 것이므로 실패시킨다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcm.infer import CKPT  # noqa: E402

pytestmark = pytest.mark.skipif(not CKPT.exists(),
                                reason="학습된 체크포인트 없음(python -m lcm.train 먼저)")

# (발화, 기대 action). action 이 가장 중요한 라우팅 결정 — 슬롯은 별도 검증.
GOLDEN = [
    ("왼쪽으로 가", "move"),
    ("세이프존으로 가", "move"),
    ("강남에서 사냥", "hunt"),
    ("연습 사냥터에서 사냥해줘", "hunt"),
    ("멈춰", "stop"),
    ("그만 멈춰", "stop"),
    ("체력 물약 먹어", "potion"),
    ("공격속도 물약 마셔", "potion"),
    ("강철 세트 착용", "equip"),
    ("불멸 세트 입어", "equip"),
    ("무기 벗어", "unequip"),
    ("갑옷 해제", "unequip"),
    ("자동사냥 켜줘", "auto_combat"),
    ("자동사냥 꺼", "auto_combat"),
    ("HP 물약 자동 사용", "auto_potion"),
    ("인벤토리 열어", "open_menu"),
    ("메뉴 열어줘", "open_menu"),
    ("채팅 열어", "open_menu"),
]
# fallback(unknown 또는 저신뢰 → CF 폴백) 이어야 하는 비명령 발화.
FALLBACK = ["안녕", "넌 누구야", "레벨 어떻게 올려", "오늘 날씨 어때", "고마워"]


@pytest.fixture(scope="module")
def rt():
    from lcm.infer import LcmRuntime
    return LcmRuntime()


def test_action_accuracy(rt):
    ok = 0
    wrong = []
    for text, want in GOLDEN:
        intent, _ = rt.predict(text)
        if intent["action"] == want:
            ok += 1
        else:
            wrong.append((text, want, intent["action"]))
    acc = ok / len(GOLDEN)
    assert acc >= 0.8, f"action 정확도 {acc:.2f} < 0.8 — 오답: {wrong}"


def test_fallback_routing(rt):
    """비명령 발화는 fallback 으로 가야 한다(SML 이 억지로 명령 만들지 않음)."""
    routed = [(t, rt.classify(t)["layer"]) for t in FALLBACK]
    bad = [t for t, layer in routed if layer != "fallback"]
    # 최소 60% 는 fallback 으로(완벽하지 않아도 명령 오인은 적어야).
    assert len(bad) <= len(FALLBACK) * 0.4, f"명령으로 오인된 비명령 발화: {bad}"


def test_high_confidence_commands(rt):
    """명확한 단순 명령은 높은 confidence 로 sml 채택되어야 한다."""
    for text in ("왼쪽으로 가", "멈춰", "체력 물약 먹어", "인벤토리 열어"):
        res = rt.classify(text)
        assert res["layer"] == "sml", f"'{text}' → {res['layer']} (conf {res.get('confidence'):.2f})"


def test_holdout_fallback_recall(rt):
    """홀드아웃 비명령(학습셋에 없는 잡담/질문)이 fallback 되어야 한다(OOD 과신 회귀 가드).

    좁은 분류기는 모르는 입력을 가까운 명령으로 *확신* 하는 경향이 있다("날씨 좋다"→hunt
    conf 1.0). 이는 잡담을 명령 실행하는 심각한 오작동이므로 임계 이상 fallback 을 강제한다.
    """
    from lcm.eval import HOLDOUT_NONCMD
    fb = sum(1 for t in HOLDOUT_NONCMD if rt.classify(t)["layer"] == "fallback")
    recall = fb / len(HOLDOUT_NONCMD)
    misfired = [t for t in HOLDOUT_NONCMD if rt.classify(t)["layer"] != "fallback"]
    assert recall >= 0.7, f"홀드아웃 fallback recall {recall:.2f} < 0.7 — OOD 과신: {misfired[:8]}"


def test_whitespace_robustness(rt):
    """공백 변형(STT 전사·사용자 입력의 띄어쓰기 불규칙)에 강건해야 한다."""
    cases = [
        ("강남에서사냥", "hunt"), ("왼쪽으로가", "move"), ("체력물약먹어", "potion"),
        ("강철세트착용", "equip"), ("인벤토리열어", "open_menu"), ("자동사냥켜", "auto_combat"),
        ("연습장에서사냥", "hunt"), ("물약 줘", "potion"),
    ]
    ok = sum(1 for t, w in cases if rt.predict(t)[0]["action"] == w)
    rate = ok / len(cases)
    wrong = [(t, rt.predict(t)[0]["action"]) for t, w in cases if rt.predict(t)[0]["action"] != w]
    assert rate >= 0.85, f"공백 강건성 {rate:.2f} < 0.85 — 오답: {wrong}"


def test_calibration_ece(rt):
    """confidence 가 실제 정확도와 일치(ECE)해야 threshold 판정이 신뢰된다(label smoothing)."""
    from lcm.bench import compute_ece
    ece, mean_conf, mean_acc = compute_ece(rt.model, rt.ls, rt.tk)
    assert ece < 0.15, f"ECE {ece:.3f} ≥ 0.15 — confidence 미보정(conf {mean_conf:.2f} vs acc {mean_acc:.2f})"
