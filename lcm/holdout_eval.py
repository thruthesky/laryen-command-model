"""Holdout 평가 — 실사용 안정성 측정(양팀 합의). synth val(0.986)은 학습과 같은 템플릿
분포라 실사용을 증명 못 한다. tests/holdout.json 의 4 family 로 from-scratch vs backbone 을
*같은 기준* 으로 비교한다.

  - template_family : 학습에 없는 자연 표현 → answer_intent 일치율(의미 일반화).
  - stt_error       : 망가진 전사 → action(+location) 일치율(명령 robust).
  - context_deixis  : 거기/그거/아까 → route!=execute 비율(🛑 임의 실행 금지 = 안전).
  - adversarial     : 잡담/조건문/무의미 → route!=execute 비율(안전).

predict_one(text)->dict{route, answer_intent, intent_all} 만 주입하면 어떤 모델이든 평가.

사용: .venv/bin/python -m lcm.holdout_eval            (from-scratch LcmRuntime)
"""
from __future__ import annotations

import json
from pathlib import Path

HOLDOUT = Path(__file__).resolve().parents[1] / "tests" / "holdout.json"


def run(predict_one, title: str = "") -> dict:
    h = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    print(f"\n========== holdout {title} ==========")
    res = {}

    tf = h["template_family"]["cases"]
    ok = 0
    for c in tf:
        r = predict_one(c["text"])
        if r.get("answer_intent") == c["answer_intent"]:
            ok += 1
        else:
            print(f"  ✗[tmpl] {c['text'][:30]!r} exp={c['answer_intent']} "
                  f"got={r.get('answer_intent')} route={r.get('route')}")
    res["template_family"] = (ok, len(tf))
    print(f"[template_family] 의미 일반화: {ok}/{len(tf)} = {100*ok/len(tf):.0f}%")

    ml = h["multilingual"]["cases"]
    ok = 0
    for c in ml:
        r = predict_one(c["text"])
        if r.get("answer_intent") == c["answer_intent"]:
            ok += 1
        else:
            print(f"  ✗[多言語] {c['text'][:20]!r} exp={c['answer_intent']} "
                  f"got={r.get('answer_intent')} route={r.get('route')}")
    res["multilingual"] = (ok, len(ml))
    print(f"[multilingual] 한·영·중·일 동등: {ok}/{len(ml)} = {100*ok/len(ml):.0f}%")

    se = h["stt_error"]["cases"]
    ok = 0
    for c in se:
        r = predict_one(c["text"])
        it = r.get("intent_all") or {}
        act_ok = r.get("route") == "execute" and it.get("action") == c["action"]
        loc_ok = "location" not in c or it.get("location") == c["location"]
        if act_ok and loc_ok:
            ok += 1
        else:
            print(f"  ✗[stt] {c['text'][:24]!r} exp={c['action']} "
                  f"got={it.get('action')} loc={it.get('location')} route={r.get('route')}")
    res["stt_error"] = (ok, len(se))
    print(f"[stt_error] 명령 robust: {ok}/{len(se)} = {100*ok/len(se):.0f}%")

    for fam in ("context_deixis", "adversarial"):
        cs = h[fam]["cases"]
        safe = 0
        for c in cs:
            r = predict_one(c["text"])
            if r.get("route") != "execute":
                safe += 1
            else:
                print(f"  🛑[{fam}] 위험 실행: {c['text']!r} → {r.get('intent_all')}")
        res[fam] = (safe, len(cs))
        print(f"[{fam}] 안전(실행 안 함): {safe}/{len(cs)} = {100*safe/len(cs):.0f}%")

    tot_ok = sum(a for a, _ in res.values())
    tot_n = sum(b for _, b in res.values())
    print(f"--- 종합: {tot_ok}/{tot_n} = {100*tot_ok/tot_n:.0f}% ---")
    return res


if __name__ == "__main__":
    from .infer import LcmRuntime
    rt = LcmRuntime()
    run(rt._classify_one, title="from-scratch(train8)")
