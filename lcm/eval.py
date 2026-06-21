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
from .model import LcmEncoder, predict_heads
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
