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
    cmds = ("왼쪽으로 가", "멈춰", "체력 물약 먹어", "인벤토리 열어")
    sml = sum(1 for t in cmds if rt.classify(t)["layer"] == "sml")
    assert sml >= len(cmds) - 1, f"명확 명령 sml {sml}/{len(cmds)}(저신뢰 fallback 과다)"


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


def test_phonetic_robustness(rt):
    """자모 변형(STT 음소 오류 — 받침 탈락/모음 혼동)에 어느 정도 강건해야 한다."""
    cases = [
        ("멈처", "stop"), ("사양하자", "hunt"), ("강남에서 사양", "hunt"),
        ("체력 물략 줘", "potion"), ("인벤토리 여러", "open_menu"),
        ("강철 세트 차용", "equip"), ("연습장 사냥", "hunt"), ("왼쪽으로 가줘", "move"),
    ]
    # 자모 변형은 STT 음소 오류 — 정답 sml 이 최선이나, *fallback 도 안전*(CF 가 처리).
    # 위험한 건 *틀린 action 을 sml* 로 내는 것이라, "정답 또는 fallback" 비율을 본다.
    safe = 0
    for t, w in cases:
        r = rt.classify(t)
        a = r["command"]["actions"][0]["action"] if r["layer"] == "sml" else "fallback"
        if a == w or a == "fallback":
            safe += 1
    rate = safe / len(cases)
    assert rate >= 0.75, f"유사발음 안전(정답+fallback) {rate:.2f} < 0.75(틀린 sml 과다)"


def test_edge_cases_safe(rt):
    """이상 입력(빈·구두점·이모지·초장문)에 crash 없고, 의미 문자 없으면 fallback."""
    for t in ["", " ", "   ", "!!!", "...", "?????", "😀😀", "@#$%^", "\n\t"]:
        res = rt.classify(t)
        assert res["layer"] == "fallback", f"'{t}' 의미문자 없는데 {res['layer']}"
    for t in ["a" * 500, "강" * 200, "1 2 3 4 5"]:
        res = rt.classify(t)  # crash 없으면 OK
        assert res["layer"] in ("sml", "fallback")


def test_location_routing(rt):
    """단일 명확 위치 → sml(정확한 location), 복합/상대/위치의존 → fallback(CF 공간추론).

    과거 버그: "강남역 동쪽 세이프존"→sml move gangnam_station(자신있게 틀림). 복합/상대는
    분류기로 불가하므로 fallback 해야 한다(CF 가 좌표 프롬프트로 공간추론).
    """
    # 단일 명확 → sml + location 일치.
    simple = [("강남역으로 가", "gangnam_station"), ("세이프존으로 이동", "safe"),
              ("연습 사냥터로 가", "practice"), ("강북으로 이동", "gangbuk")]
    for t, loc in simple:
        r = rt.classify(t)
        assert r["layer"] == "sml", f"'{t}' → {r['layer']}(단일은 sml 이어야)"
        assert r["command"]["actions"][0].get("location") == loc, \
            f"'{t}' location={r['command']['actions'][0].get('location')} != {loc}"
    # 복합/상대/위치의존 → fallback(자신있게 틀린 landmark 금지).
    complex_ = [
        "강남역 동쪽 세이프 존으로 이동해", "강남역 동쪽으로 가", "강남역 위로 가",
        "가까운 사냥터로 가", "제일 가까운 안전지대로", "북쪽 사냥터로 가",
        "강북 말고 강남으로 가",
    ]
    # 사용자 예시는 반드시 fallback(자신있게 틀림 절대 금지).
    assert rt.classify("강남역 동쪽 세이프 존으로 이동해")["layer"] == "fallback"
    # 나머지 복합/상대는 무한해 일부 sml 허용(≤2) — 핵심은 자신있는 오류 최소화.
    bad = [t for t in complex_ if rt.classify(t)["layer"] != "fallback"]
    assert len(bad) <= 2, f"복합/상대 위치인데 sml(자신있게 틀림): {bad}"


def test_english_commands(rt):
    """영어 명령(한/영 우선)의 action 정확도(≥0.85)."""
    cases = [("drink hp potion", "potion"), ("use potion", "potion"), ("halt", "stop"),
             ("open inventory", "open_menu"), ("open menu", "open_menu"),
             ("hunt at gangnam", "hunt"), ("go to safe zone", "move"),
             ("auto hunt on", "auto_combat"), ("equip plate set", "equip")]
    ok = sum(1 for t, w in cases if (r := rt.classify(t))["layer"] == "sml"
             and r["command"]["actions"][0]["action"] == w)
    assert ok >= len(cases) * 0.85, f"영어 명령 {ok}/{len(cases)}"


def test_holdout_command_generalization(rt):
    """train 에 없는 다양한 실사용 명령 표현의 action 정확도(일반화 ≥0.8)."""
    H = [("강남 데려가", "move"), ("저기 강북으로", "move"), ("관악산 가자고", "move"),
         ("거기 강남역으로 이동", "move"), ("사냥 좀 하러 가자 연습장", "hunt"),
         ("강동 꽃밭 가서 몹 잡아", "hunt"), ("회복 좀 하자", "potion"),
         ("체력 좀 채워", "potion"), ("장비창 띄워", "open_menu"), ("강철로 갈아입어", "equip")]
    ok = sum(1 for t, w in H if (r := rt.classify(t))["layer"] == "sml"
             and r["command"]["actions"][0]["action"] == w)
    assert ok >= len(H) * 0.8, f"holdout 명령 일반화 {ok}/{len(H)}"


def test_polite_and_english_alias(rt):
    """존댓말("가 주실래요")·영어별칭+조사("safe zone으로 가")가 sml 로 정확 처리돼야 한다."""
    cases = [
        ("강남으로 가 주실래요", "move"), ("멈춰 주세요", "stop"),
        ("물약 주세요", "potion"), ("인벤토리 열어 주세요", "open_menu"),
        ("safe zone으로 가", "move"),
    ]
    ok = sum(1 for t, w in cases if (r := rt.classify(t))["layer"] == "sml"
             and r["command"]["actions"][0]["action"] == w)
    assert ok >= len(cases) - 1, f"존댓말/영어별칭 {ok}/{len(cases)}"
    # "safe zone으로 가" 는 안전지대로(direction 아님).
    r = rt.classify("safe zone으로 가")
    if r["layer"] == "sml":
        assert r["command"]["actions"][0].get("location") == "safe", \
            f"safe zone → {r['command']['actions'][0]}"


def test_monsters_no_false_positive(rt):
    """monster 미언급 hunt 는 monsters 가 비어야 한다(가중 3x 의 false positive 차단)."""
    cases = ["강남에서 사냥", "연습장에서 사냥해", "강남에서 사냥하고 체력 60%면 피신",
             "관악산에서 사냥해줘", "강동 꽃밭에서 사냥", "강북에서 자동 사냥"]
    fp = []
    for t in cases:
        r = rt.classify(t)
        if r["layer"] == "sml" and r["command"]["actions"][0].get("monsters"):
            fp.append((t, r["command"]["actions"][0]["monsters"]))
    assert not fp, f"monster 미언급인데 삽입(false positive): {fp}"


def test_monsters_true_positive(rt):
    """monster 명시 hunt 는 그 monster 가 잡혀야 한다(현실 조합 ≥80% — 작은 모델 비결정 허용)."""
    cases = [
        ("연습장에서 Skeleton 잡아", "Skeleton"), ("강동 꽃밭에서 Bone 사냥", "Bone"),
        ("강남에서 Hellion 사냥", "Hellion"), ("강북 진달래 동산에서 Caster 잡아", "Caster"),
        ("강서 산책로에서 Brute 사냥", "Brute"), ("강남역에서 Xbot 잡아", "Xbot"),
    ]
    ok = 0
    miss = []
    for t, m in cases:
        r = rt.classify(t)
        mons = r["command"]["actions"][0].get("monsters", []) if r["layer"] == "sml" else []
        if m in mons:
            ok += 1
        else:
            miss.append((t, mons))
    assert ok >= len(cases) * 0.8, f"monster TP {ok}/{len(cases)} — 누락: {miss}"


def test_compound_multiaction(rt):
    """다중동작("강철 입고 사냥")을 actions 배열로 직접 처리하고, 오분할/단일을 구분한다."""
    r = rt.classify("강철 세트 입고 강남에서 사냥")
    assert r["layer"] == "sml", "다중동작이 fallback"
    acts = [a["action"] for a in r["command"]["actions"]]
    assert acts == ["equip", "hunt"], f"다중동작 → {acts}"
    # hunt 옵션 연결("사냥하고 체력 30%")은 단일 hunt 로 유지(오분할 금지).
    r2 = rt.classify("강남에서 사냥하고 체력 30%면 피신")
    assert len(r2["command"]["actions"]) == 1, "hunt 옵션 오분할"
    # 단일은 1개.
    assert len(rt.classify("강남에서 사냥")["command"]["actions"]) == 1
    # 3-action 재귀 분할("물약 먹고 강철 입고 사냥").
    r3 = rt.classify("물약 먹고 강철 입고 강남에서 사냥")
    if r3["layer"] == "sml":
        a3 = [a["action"] for a in r3["command"]["actions"]]
        assert a3 == ["potion", "equip", "hunt"], f"3-action → {a3}"


def test_negation_fallback(rt):
    """부정("사냥하지마")은 fallback 해야(자신있게 반대 실행 금지). 다중동작은 iter30 이후
    분할 sml(test_compound_multiaction)이므로 여기선 부정만 검증한다."""
    cases = ["사냥하지마", "멈추지마", "이동하지 마", "공격하지 말고", "가지마", "물약 먹지마"]
    bad = [t for t in cases if rt.classify(t)["layer"] != "fallback"]
    assert len(bad) <= 1, f"부정인데 sml(자신있게 반대 실행): {bad}"


def test_colloquial_robustness(rt):
    """도치·구어체 발화(실사용)에 강건해야 한다."""
    cases = [
        ("사냥하자 강남에서", "hunt"), ("강남 가서 사냥 좀 해줘", "hunt"),
        ("물약 좀 먹자", "potion"), ("왼쪽으로 좀 가볼까", "move"),
        ("이제 그만하자", "stop"), ("자동사냥 돌리자", "auto_combat"),
        ("연습장 가서 사냥하면 될까", "hunt"), ("강남으로 사냥 가줄래", "hunt"),
    ]
    ok = sum(1 for t, w in cases if rt.predict(t)[0]["action"] == w)
    rate = ok / len(cases)
    wrong = [(t, rt.predict(t)[0]["action"]) for t, w in cases if rt.predict(t)[0]["action"] != w]
    assert rate >= 0.8, f"구어체 강건성 {rate:.2f} < 0.8 — 오답: {wrong}"


def test_calibration_ece(rt):
    """confidence 가 실제 정확도와 일치(ECE)해야 threshold 판정이 신뢰된다(label smoothing)."""
    from lcm.bench import compute_ece
    ece, mean_conf, mean_acc = compute_ece(rt.model, rt.ls, rt.tk)
    assert ece < 0.15, f"ECE {ece:.3f} ≥ 0.15 — confidence 미보정(conf {mean_conf:.2f} vs acc {mean_acc:.2f})"
