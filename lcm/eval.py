"""헤드별·action별 정확도 진단 — 어느 슬롯이 약한지 짚어 데이터/학습을 겨냥한다.

사용:  python -m lcm.eval            # val.jsonl 평가 리포트
       python -m lcm.eval --split train --errors 20
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .dataset import LcmDataset, make_collate
from .model import LcmEncoder, action_confidence, predict_heads
from .schema import LabelSpace, decode_intent, load_ssot
from .tokenizer import load_tokenizer, pad_id

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
CKPT = ROOT / "checkpoints" / "lcm.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--errors", type=int, default=12, help="출력할 오답 샘플 수")
    args = ap.parse_args()

    ls = LabelSpace(load_ssot())
    tk = load_tokenizer()
    blob = torch.load(CKPT, map_location="cpu")
    c = blob["config"]
    model = LcmEncoder(c["vocab_size"], ls, pad_id=c["pad_id"], d_model=c["d_model"],
                       n_layers=c["layers"], max_len=c["max_len"])
    model.load_state_dict(blob["model"])
    model.eval()

    ds = LcmDataset(DATA / f"{args.split}.jsonl", tk, ls, max_len=32)

    # 헤드별 정확도(single/binary), multi 는 jaccard. action 일 때만 의미있는 슬롯은
    # 해당 action 표본에서만 집계(gold 슬롯이 <none> 이 아니거나 action 이 그 슬롯 소유).
    head_correct: dict = defaultdict(int)
    head_total: dict = defaultdict(int)
    action_correct = defaultdict(int)
    action_total = defaultdict(int)
    exact_ok = 0
    errors = []

    # 어느 action 이 어느 슬롯을 쓰는지(집계 범위 한정).
    owner = {
        "location": {"move", "hunt"}, "direction": {"move"},
        "monsters": {"hunt"}, "hunt_hp": {"hunt"}, "retreat_hp": {"hunt"},
        "retreat_to_safe": {"hunt"}, "potion": {"potion"},
        "gear_set": {"equip"}, "gear_single": {"equip"}, "slot": {"unequip"},
        "mode": {"auto_combat"}, "target": {"open_menu"},
        "auto_potions": {"auto_potion"}, "auto_potion_enable": {"auto_potion"},
    }

    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            ids = torch.tensor([item["input_ids"]], dtype=torch.long)
            attn = torch.ones_like(ids, dtype=torch.bool)
            logits = model(ids, attn)
            pred = predict_heads(logits, model.head_specs)
            gold = item["labels"]
            gact = ls.actions[gold["action"]]

            action_total[gact] += 1
            if pred["action"] == gold["action"]:
                action_correct[gact] += 1

            for name, kind, _ in model.head_specs:
                if name in owner and gact not in owner[name]:
                    continue
                head_total[name] += 1
                if kind == "multi":
                    p, g = set(j for j, v in enumerate(pred[name]) if v), \
                           set(j for j, v in enumerate(gold[name]) if v)
                    if p == g:
                        head_correct[name] += 1
                else:
                    if pred[name] == gold[name]:
                        head_correct[name] += 1

            gi = decode_intent(gold, ls)
            pi = decode_intent(pred, ls)
            if gi == pi:
                exact_ok += 1
            elif len(errors) < args.errors:
                errors.append((ds.rows[i]["text"], gi, pi))

    n = len(ds)
    print(f"=== {args.split} ({n}건) ===")
    print(f"exact_acc: {exact_ok/n:.3f}")
    print("\n[헤드별 정확도] (해당 슬롯 소유 action 표본만)")
    for name, _, _ in model.head_specs:
        t = head_total[name]
        if t:
            print(f"  {name:20s} {head_correct[name]/t:.3f}  (n={t})")
    print("\n[action별 정확도]")
    for a in sorted(action_total, key=lambda x: -action_total[x]):
        print(f"  {a:14s} {action_correct[a]/action_total[a]:.3f}  (n={action_total[a]})")
    print(f"\n[오답 샘플 {len(errors)}]")
    for text, gi, pi in errors:
        print(f"  '{text}'\n    gold={gi}\n    pred={pi}")

    # ── 3계층 fallback 안전장치 분석 — confidence threshold sweep ─────────────
    # 핵심 위험: 비명령(unknown)을 *명령 실행*(오작동). fallback = action==unknown 또는
    # confidence < th. 명령은 sml 채택(과도 fallback=불필요 CF 호출)되길 원한다.
    cmd_conf, noncmd_conf = [], []  # (confidence, is_unknown_pred)
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            ids = torch.tensor([item["input_ids"]], dtype=torch.long)
            attn = torch.ones_like(ids, dtype=torch.bool)
            logits = model(ids, attn)
            conf = action_confidence(logits)
            pred_a = ls.actions[int(logits["action"].argmax(-1).item())]
            gold_a = ls.actions[item["labels"]["action"]]
            rec = (conf, pred_a == "unknown")
            (noncmd_conf if gold_a == "unknown" else cmd_conf).append(rec)

    print("\n[3계층 fallback 분석] (fallback = pred==unknown 또는 conf<th)")
    print(f"  비명령(unknown) {len(noncmd_conf)}건 / 명령 {len(cmd_conf)}건")
    for th in (0.5, 0.6, 0.7, 0.8):
        # 비명령 fallback recall(↑좋음 — 오작동 방지), 명령 sml 채택(↑좋음 — CF 절약).
        nc_fb = sum(1 for c, u in noncmd_conf if u or c < th) / max(len(noncmd_conf), 1)
        cmd_sml = sum(1 for c, u in cmd_conf if not u and c >= th) / max(len(cmd_conf), 1)
        print(f"  th={th}: 비명령 fallback {nc_fb:.2f} | 명령 sml 채택 {cmd_sml:.2f}")

    # ── 홀드아웃 비명령(train 에 *없는* 잡담/질문) — 진짜 일반화 fallback recall ──
    print("\n[홀드아웃 비명령 일반화] (학습셋에 없는 표현 — 진짜 오작동 위험 측정)")
    misfired = []
    fb = 0
    with torch.no_grad():
        for text in HOLDOUT_NONCMD:
            ids = torch.tensor([tk.encode(text).ids[:32]], dtype=torch.long)
            attn = torch.ones_like(ids, dtype=torch.bool)
            logits = model(ids, attn)
            conf = action_confidence(logits)
            a = ls.actions[int(logits["action"].argmax(-1).item())]
            if a == "unknown" or conf < 0.6:
                fb += 1
            else:
                misfired.append((text, a, round(conf, 2)))
    print(f"  fallback recall(th=0.6): {fb}/{len(HOLDOUT_NONCMD)} = {fb/len(HOLDOUT_NONCMD):.2f}")
    if misfired:
        print(f"  ⚠️ 명령 오인(오작동 위험) {len(misfired)}건:")
        for text, a, c in misfired[:12]:
            print(f"      '{text}' → {a} (conf {c})")
    return 0


# train/synth 에 *없는* 비명령 발화(잡담·게임 질문) — fallback 일반화 측정용.
HOLDOUT_NONCMD = [
    "넌 어디서 왔어", "기분이 어때", "오늘 뭐 먹지", "주말에 뭐 할까", "날씨 좋다",
    "이 게임 만든 사람 누구야", "업데이트 언제 해", "버그 신고 어디서 해", "환불 돼",
    "내 친구 어디 있어", "길드 어떻게 가입해", "초보자 팁 알려줘", "공략 좀",
    "어떤 직업이 좋아", "과금 해야 돼", "이벤트 언제 끝나", "랭킹 어떻게 봐",
    "스킬 트리 어떻게 찍어", "리셋 어떻게 해", "계정 어떻게 지워",
    "what time is it", "are you a robot", "tell me a story", "i like this game",
    "how much hp do i have", "is this game free", "what's the max level",
    "can i play with friends", "where do i buy potions", "good morning",
]


if __name__ == "__main__":
    raise SystemExit(main())
