#!/usr/bin/env python3
"""영어/한국어 게임 컨트롤 발화 깊이 테스트 — 현재 lcm.pt 로 classify 정확도·실패 측정.
사용: uv run python scripts/probe_lang.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcm.infer import LcmRuntime  # noqa: E402


def chk(r, action, key=None, val=None):
    """classify 결과 r 이 기대 action(+선택 슬롯)인지."""
    if r["layer"] != "sml":
        return False, f"fallback(conf={r['confidence']:.2f})"
    acts = r["command"]["actions"]
    a0 = acts[0]
    if a0.get("action") != action:
        return False, f"action={a0.get('action')}"
    if key and str(a0.get(key)) != str(val):
        # monsters 는 리스트
        if key == "monsters" and val in (a0.get("monsters") or []):
            return True, "ok"
        return False, f"{key}={a0.get(key)}"
    return True, "ok"


# (발화, 기대 action, 슬롯키, 슬롯값) — 슬롯 None 이면 action 만.
KO = [
    ("강남으로 가", "move", "location", "gangnam"),
    ("강남 가로수길로 이동", "move", "location", "gangnam_garosoo"),
    ("강남역 6번 출구로 가", "move", "location", "gangnam_station_exit_6"),
    ("7번 출구로 이동", "move", "location", "gangnam_station_exit_7"),
    ("왼쪽으로 가", "move", "direction", "270.0"),
    ("세이프존으로 가", "move", "location", "safe"),
    ("강남에서 사냥", "hunt", "location", "gangnam"),
    ("캐스터 사냥", "hunt", "monsters", "Caster"),
    ("연습장에서 브루트 잡아", "hunt", "monsters", "Brute"),
    ("멈춰", "stop", None, None),
    ("그만해", "stop", None, None),
    ("정지", "stop", None, None),
    ("체력 물약 먹어", "potion", "potion", "hp"),
    ("회복 물약", "potion", "potion", "hp"),
    ("강철 세트 착용", "equip", "set", "plate"),
    ("불멸 세트 입어", "equip", "set", "immortal"),
    ("무기 벗어", "unequip", "slot", "weapon"),
    ("자동사냥 켜", "auto_combat", "mode", "auto_hunt"),
    ("오토 꺼", "auto_combat", "mode", "off"),
    ("인벤토리 열어", "open_menu", "target", "inventory"),
    ("메뉴 열어", "open_menu", "target", "menu"),
    ("디버그 패널 꺼", "open_menu", "target", "debug"),
    ("디버그 켜줘", "open_menu", "target", "debug"),
    ("fps 표시", "open_menu", "target", "debug"),
]
EN = [
    ("go to gangnam", "move", "location", "gangnam"),
    ("move to safe zone", "move", "location", "safe"),
    ("go to gangnam station exit 6", "move", "location", "gangnam_station_exit_6"),
    ("go left", "move", None, None),
    ("hunt at gangnam", "hunt", "location", "gangnam"),
    ("hunt casters", "hunt", "monsters", "Caster"),
    ("hunt brutes", "hunt", "monsters", "Brute"),
    ("stop", "stop", None, None),
    ("halt", "stop", None, None),
    ("freeze", "stop", None, None),
    ("stop now", "stop", None, None),
    ("drink hp potion", "potion", "potion", "hp"),
    ("use potion", "potion", "potion", "hp"),
    ("heal", "potion", "potion", "hp"),
    ("equip plate set", "equip", "set", "plate"),
    ("auto hunt on", "auto_combat", "mode", "auto_hunt"),
    ("turn off auto hunt", "auto_combat", "mode", "off"),
    ("open inventory", "open_menu", "target", "inventory"),
    ("open menu", "open_menu", "target", "menu"),
    ("open chat", "open_menu", "target", "groupchat"),
    ("turn off the debug panel", "open_menu", "target", "debug"),
    ("turn on debug panel", "open_menu", "target", "debug"),
    ("hide debug panel", "open_menu", "target", "debug"),
    ("show fps", "open_menu", "target", "debug"),
    ("toggle debug", "open_menu", "target", "debug"),
]
# fallback 으로 가야 정상(QnA/잡담)인 것 — 영어 포함.
FALLBACK = [
    ("강남 뭐가 좋아", True), ("what monster is here", True),
    ("how do i make a party", True), ("안녕 반가워", True),
    ("where should i hunt", True), ("이 게임 어떻게 해", True),
]


def run(rt, cases, label):
    ok = 0
    fails = []
    for c in cases:
        t, action, key, val = c
        r = rt.classify(t)
        good, why = chk(r, action, key, val)
        if good:
            ok += 1
        else:
            fails.append(f'   ✗ "{t}" → {why} (기대 {action}/{key}={val})')
    print(f"\n[{label}] {ok}/{len(cases)} = {ok/len(cases)*100:.0f}%")
    for f in fails:
        print(f)
    return ok, len(cases)


def run_fb(rt, cases):
    ok = 0
    fails = []
    for t, want_fb in cases:
        r = rt.classify(t)
        is_fb = r["layer"] != "sml"
        if is_fb == want_fb:
            ok += 1
        else:
            fails.append(f'   ✗ "{t}" → {r["layer"]} (기대 fallback)')
    print(f"\n[FALLBACK(QnA→CF)] {ok}/{len(cases)} = {ok/len(cases)*100:.0f}%")
    for f in fails:
        print(f)
    return ok, len(cases)


def main():
    rt = LcmRuntime()
    print("=== LCM 영어/한국어 게임 컨트롤 깊이 테스트 (현재 lcm.pt) ===")
    ko_ok, ko_n = run(rt, KO, "한국어 게임컨트롤")
    en_ok, en_n = run(rt, EN, "영어 게임컨트롤")
    fb_ok, fb_n = run_fb(rt, FALLBACK)
    tot_ok, tot_n = ko_ok + en_ok + fb_ok, ko_n + en_n + fb_n
    print(f"\n=== 종합 {tot_ok}/{tot_n} = {tot_ok/tot_n*100:.0f}% "
          f"(한 {ko_ok}/{ko_n} · 영 {en_ok}/{en_n} · fb {fb_ok}/{fb_n}) ===")


if __name__ == "__main__":
    main()
