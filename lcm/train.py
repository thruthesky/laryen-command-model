"""학습 루프 — M5 Apple Silicon(MPS) 단일 머신. 분산학습 없음.

사용:
  python -m lcm.train                       # 기본값
  python -m lcm.train --epochs 30 --batch 64
산출물: checkpoints/lcm.pt (+ 토크나이저 artifacts/tokenizer/).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import LcmDataset, make_collate, read_jsonl
from .model import LcmEncoder, action_confidence, multihead_loss, predict_heads
from .schema import LabelSpace, decode_intent, load_ssot
from .tokenizer import DEFAULT_DIR, load_tokenizer, pad_id, train_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
CKPT = ROOT / "checkpoints" / "lcm.pt"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, ls, device) -> tuple[float, float]:
    model.eval()
    action_ok = exact_ok = total = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            logits = model(ids, attn)
            for b in range(ids.size(0)):
                one = {k: v[b : b + 1] for k, v in logits.items()}
                pred = predict_heads(one, model.head_specs)
                gold = {k: (v[b].tolist() if v[b].ndim else int(v[b]))
                        for k, v in batch["labels"].items()}
                total += 1
                if pred["action"] == gold["action"]:
                    action_ok += 1
                if decode_intent(pred, ls) == decode_intent(gold, ls):
                    exact_ok += 1
    return action_ok / max(total, 1), exact_ok / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=32)
    args = ap.parse_args()

    ssot = load_ssot()
    ls = LabelSpace(ssot)
    train_rows = read_jsonl(DATA / "train.jsonl")

    # 토크나이저 — 없으면 학습 코퍼스로 새로 학습.
    if not (DEFAULT_DIR / "vocab.json").exists():
        train_tokenizer([r["text"] for r in train_rows], vocab_size=args.vocab)
    tk = load_tokenizer()

    collate = make_collate(pad_id(tk), ls, args.max_len)
    train_ds = LcmDataset(DATA / "train.jsonl", tk, ls, args.max_len, augment=True, aug_p=0.5)
    val_ds = LcmDataset(DATA / "val.jsonl", tk, ls, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = pick_device()
    model = LcmEncoder(tk.get_vocab_size(), ls, pad_id=pad_id(tk),
                       d_model=args.d_model, n_layers=args.layers, max_len=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device.type}  params={n_params/1e3:.0f}K  vocab={tk.get_vocab_size()}  "
          f"train={len(train_ds)} val={len(val_ds)}")

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            logits = model(ids, attn)
            loss = multihead_loss(logits, labels, model.head_specs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        acc_a, acc_e = evaluate(model, val_dl, ls, device)
        print(f"ep {ep:3d}  loss {running/len(train_dl):.3f}  "
              f"val action_acc {acc_a:.3f}  exact_acc {acc_e:.3f}")
        if acc_e >= best:
            best = acc_e
            CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "config": {"d_model": args.d_model, "layers": args.layers,
                           "vocab_size": tk.get_vocab_size(), "max_len": 64,
                           "pad_id": pad_id(tk)},
            }, CKPT)
    print(f"✅ best exact_acc {best:.3f} → {CKPT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
