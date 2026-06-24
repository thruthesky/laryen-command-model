"""Backbone LCM — 사전학습 다국어 인코더(multilingual-e5-small) + 기존 멀티헤드 fine-tune.

**왜**(2026-06-24 양팀 합의): from-scratch 1.2M 인코더는 합성 템플릿의 *표면 토큰 패턴* 을
암기할 뿐 의미를 일반화하지 못한다(측정: "뭐야"의 76%가 monster → "세계관이 뭐야" 오분류,
템플릿 밖 자연 발화/STT 오류/애매함에 취약). 사전학습된 다국어 인코더를 backbone 으로 쓰면
의미 임베딩으로 *표면이 달라도 의미가 가까우면 같은 의도* 로 인식 → STT 오타·문법오류·애매함
강건. 헤드/라벨/loss/decode 는 기존 schema 를 그대로 재사용(인코더만 교체).

생성형이 아니라 *분류 + abstain* 이 게임에 안전하다는 판단(생성 불필요).

사용: .venv/bin/python -m lcm.backbone_train --epochs 8 --lr-head 1e-3 --lr-bb 2e-5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .dataset import make_collate, read_jsonl
from .model import multihead_loss, predict_heads
from .schema import LabelSpace, encode_intent, load_ssot

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
CKPT = ROOT / "checkpoints" / "lcm_backbone.pt"
BACKBONE = "intfloat/multilingual-e5-small"

# 🛑 LCM-SML 온디바이스 용량 상한(2026-06-24 사용자 지시). 모바일(iOS/Android) 메모리·
# OTA 다운로드·추론 속도 제약상 이 한도를 넘으면 온디바이스에 부적합하다. 초과 시 사람
# 개발자에게 경고하고 학습/배포를 중단한다(더 작은 backbone 으로 교체 결정 필요).
MAX_PARAMS = 200_000_000


def assert_within_capacity(n_params: int, label: str = "model") -> None:
    """모델 파라미터 수가 200M 한도 이내인지 가드. 초과 시 경고 + 중단."""
    if n_params > MAX_PARAMS:
        raise SystemExit(
            f"🛑 경고(사람 개발자): {label} = {n_params/1e6:.0f}M params > 200M 제한.\n"
            f"   온디바이스(모바일) 용량/속도 부적합 — 더 작은 다국어 backbone 으로 교체하세요.\n"
            f"   (예: multilingual-e5-small 118M, Multilingual-MiniLM-L12 117M 등 ≤200M).")
    print(f"✅ 용량 가드 통과: {label} {n_params/1e6:.0f}M params ≤ 200M")


class BackboneLcm(nn.Module):
    """e5 인코더(mean pooling) + 기존 멀티헤드. logits dict 형식은 LcmEncoder 와 동일."""

    def __init__(self, ls: LabelSpace, backbone: str = BACKBONE):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        h = self.backbone.config.hidden_size
        self.head_specs = ls.heads()
        self.heads = nn.ModuleDict({
            name: nn.Linear(h, len(labels) if kind != "binary" else 1)
            for name, kind, labels in self.head_specs
        })

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp_min(1.0)  # e5 표준 mean pooling
        # binary 헤드는 [B,1]→[B] squeeze (LcmEncoder.forward 와 동일 — multihead_loss 호환).
        res = {}
        for name, kind, _ in self.head_specs:
            logit = self.heads[name](pooled)
            res[name] = logit.squeeze(-1) if kind == "binary" else logit
        return res


class BBDataset(Dataset):
    def __init__(self, path, tk, ls: LabelSpace, max_len: int):
        self.rows = read_jsonl(path)
        self.tk = tk
        self.ls = ls
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # input_ids 만 반환 — dataset.make_collate 가 attention_mask/패딩/라벨을 처리한다
        # (기존 collate 재사용 → binary/multi 라벨 형식이 from-scratch 와 100% 일치).
        row = self.rows[i]
        enc = self.tk(row["text"], truncation=True, max_length=self.max_len)
        return {"input_ids": enc["input_ids"], "labels": encode_intent(row["intent"], self.ls)}


@torch.no_grad()
def evaluate(model, dl, head_specs, device) -> float:
    """배치 단위 exact-match — predict_heads(단일 샘플용) 대신 배치 argmax/threshold."""
    model.eval()
    exact = total = 0
    for batch in dl:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        logits = model(ids, attn)
        preds = {}
        for name, kind, _ in head_specs:
            lg = logits[name].cpu()
            if kind == "single":
                preds[name] = lg.argmax(-1)          # [B]
            elif kind == "binary":
                preds[name] = (lg > 0).long()        # [B]
            else:  # multi
                preds[name] = (lg > 0).long()        # [B, C]
        bs = ids.size(0)
        for i in range(bs):
            ok = True
            for name, kind, _ in head_specs:
                gold = batch["labels"][name]
                if kind in ("single", "binary"):
                    if int(preds[name][i]) != int(gold[i]):
                        ok = False
                        break
                else:
                    if [int(x) for x in preds[name][i]] != [int(x) for x in gold[i]]:
                        ok = False
                        break
            exact += int(ok)
            total += 1
    return exact / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-bb", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=48)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ls = LabelSpace(load_ssot())
    tk = AutoTokenizer.from_pretrained(BACKBONE)
    model = BackboneLcm(ls).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"backbone={BACKBONE} params={n/1e6:.0f}M device={device.type} heads={len(ls.heads())}")
    assert_within_capacity(n, label=f"BackboneLcm({BACKBONE})")  # 🛑 200M 가드

    collate = make_collate(tk.pad_token_id, ls, args.max_len)  # 기존 collate 재사용
    train_dl = DataLoader(BBDataset(DATA / "train.jsonl", tk, ls, args.max_len),
                          batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(BBDataset(DATA / "val.jsonl", tk, ls, args.max_len),
                        batch_size=args.batch, shuffle=False, collate_fn=collate)

    # 차등 학습률 — backbone 은 낮게(사전학습 보존), 헤드는 높게(새 분류 학습).
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_bb},
        {"params": model.heads.parameters(), "lr": args.lr_head},
    ])
    head_specs = ls.heads()
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            logits = model(ids, attn)
            loss = multihead_loss(logits, labels, head_specs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        acc = evaluate(model, val_dl, head_specs, device)
        print(f"ep {ep:2d}  loss {running/len(train_dl):.3f}  val exact_acc {acc:.3f}", flush=True)
        if acc >= best:
            best = acc
            CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "backbone": BACKBONE,
                        "hidden": model.backbone.config.hidden_size}, CKPT)
    print(f"✅ best val exact_acc {best:.3f} → {CKPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
